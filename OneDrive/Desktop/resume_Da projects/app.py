"""
Causal Inference — LaLonde Job Training Study
Streamlit App  |  PSM + DiD
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Causal Inference | LaLonde",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #2a2d3e;
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label {
    color: #a0a8c0 !important;
    font-size: 0.78rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Metric cards */
.metric-card {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-left: 3px solid;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.metric-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #7a84a0;
    margin-bottom: 4px;
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 600;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #7a84a0;
    margin-top: 2px;
}

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #5a6380;
    border-bottom: 1px solid #2a2d3e;
    padding-bottom: 6px;
    margin: 28px 0 14px 0;
}

/* Tag pills */
.tag {
    display: inline-block;
    background: #1a2540;
    color: #7ab4f5;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 3px;
    margin-right: 6px;
    border: 1px solid #2a3d60;
}

/* Plot containers */
.plot-wrap {
    background: #111320;
    border: 1px solid #2a2d3e;
    border-radius: 8px;
    padding: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Colour palette ────────────────────────────────────────────────────────────
TREAT_COLOR   = "#4f9cf9"
CONTROL_COLOR = "#f97b4f"
GOOD_COLOR    = "#4ade80"
WARN_COLOR    = "#f97316"
BIAS_COLOR    = "#e53935"

plt.rcParams.update({
    "figure.facecolor": "#111320",
    "axes.facecolor":   "#111320",
    "axes.edgecolor":   "#2a2d3e",
    "axes.labelcolor":  "#c0c8e0",
    "xtick.color":      "#7a84a0",
    "ytick.color":      "#7a84a0",
    "text.color":       "#c0c8e0",
    "grid.color":       "#1e2235",
    "grid.alpha":       1.0,
    "axes.grid":        True,
    "font.family":      "monospace",
})

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df["black"]  = (df["race"] == "black").astype(int)
    df["hispan"] = (df["race"] == "hispan").astype(int)
    df = df[["treat","age","educ","black","hispan","married","nodegree","re74","re75","re78"]]
    return df

def smd(df_in, col, treat_col="treat"):
    t = df_in[df_in[treat_col] == 1][col]
    c = df_in[df_in[treat_col] == 0][col]
    pooled_std = np.sqrt((t.var() + c.var()) / 2)
    return (t.mean() - c.mean()) / pooled_std if pooled_std > 0 else 0

def run_psm(df, covariates, caliper):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[covariates])
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, df["treat"])
    df = df.copy()
    df["propensity_score"] = lr.predict_proba(X)[:, 1]

    treated = df[df["treat"] == 1].copy()
    control = df[df["treat"] == 0].copy()

    matched_control_idx = []
    used = set()
    for _, trow in treated.iterrows():
        diffs = np.abs(control["propensity_score"] - trow["propensity_score"])
        diffs = diffs[~diffs.index.isin(used)]
        if len(diffs) == 0:
            continue
        best_idx = diffs.idxmin()
        if diffs[best_idx] <= caliper:
            matched_control_idx.append(best_idx)
            used.add(best_idx)
        else:
            matched_control_idx.append(None)

    valid = [(i, j) for i, j in zip(treated.index, matched_control_idx) if j is not None]
    matched_treated_idx  = [i for i, _ in valid]
    matched_control_idx2 = [j for _, j in valid]

    matched_df = pd.concat([
        df.loc[matched_treated_idx],
        df.loc[matched_control_idx2],
    ])
    return df, matched_df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ Config")
    st.markdown("---")

    DATA_URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MatchIt/lalonde.csv"

    st.markdown("**Covariates for PSM**")
    all_covs = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]
    selected_covs = st.multiselect(
        "Select covariates",
        options=all_covs,
        default=all_covs,
        label_visibility="collapsed",
    )

    caliper = st.slider("PSM caliper", 0.01, 0.20, 0.05, 0.01,
                        help="Max propensity score distance for a match")

    show_raw = st.checkbox("Show raw data table", value=False)

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.72rem; color:#5a6380; font-family:monospace; line-height:1.7'>
Methods<br>
<span style='color:#7ab4f5'>PSM</span> — Propensity Score Matching<br>
<span style='color:#7ab4f5'>OLS</span> — Regression ATE<br>
<span style='color:#7ab4f5'>DiD</span> — Difference-in-Differences<br><br>
Dataset: LaLonde (1986)<br>
n = 614 obs
</div>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:1.6rem; margin-bottom:2px; color:#e8eaf6'>
Causal Inference <span style='color:#4f9cf9'>—</span> LaLonde Job Training Study
</h1>
<p style='color:#5a6380; font-family:IBM Plex Mono,monospace; font-size:0.8rem; margin-top:0'>
Estimating the TRUE causal effect of job training on earnings
</p>
""", unsafe_allow_html=True)

st.markdown(
    '<span class="tag">PSM</span>'
    '<span class="tag">DiD</span>'
    '<span class="tag">OLS</span>'
    '<span class="tag">LaLonde 1986</span>'
    '<span class="tag">Selection Bias</span>',
    unsafe_allow_html=True,
)



# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()
if show_raw:
    st.dataframe(df.head(20), use_container_width=True)

if not selected_covs:
    st.warning("Select at least one covariate in the sidebar.")
    st.stop()

# ── Run analysis ──────────────────────────────────────────────────────────────
df_ps, matched_df = run_psm(df, selected_covs, caliper)

treated_n = df["treat"].sum()
control_n = (df["treat"] == 0).sum()
matched_n = (matched_df["treat"] == 1).sum()

naive_est = df[df["treat"] == 1]["re78"].mean() - df[df["treat"] == 0]["re78"].mean()

# OLS ATE
formula = "re78 ~ treat + " + " + ".join(selected_covs)
ols_model = smf.ols(formula, data=df).fit()
ate = ols_model.params["treat"]

# ATT PSM
att_psm = matched_df[matched_df["treat"] == 1]["re78"].mean() - \
          matched_df[matched_df["treat"] == 0]["re78"].mean()

# DiD
matched_df2 = matched_df.copy()
matched_df2["pre_earnings"] = (matched_df2["re74"] + matched_df2["re75"]) / 2
did_formula = "re78 ~ treat + pre_earnings + " + " + ".join(
    [c for c in selected_covs if c not in ["re74", "re75"]]
)
did_model = smf.ols(did_formula, data=matched_df2).fit()
did_est = did_model.params["treat"]

bias = naive_est - att_psm

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Results</p>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

def kpi(col, label, value, sub, border_color, is_dollar=True):
    v = f"${value:+,.0f}" if is_dollar else f"{value:.3f}"
    color = GOOD_COLOR if value > 0 else BIAS_COLOR
    if label == "NAIVE (BIASED)":
        color = BIAS_COLOR
    col.markdown(f"""
<div class="metric-card" style="border-left-color:{border_color}">
  <div class="label">{label}</div>
  <div class="value" style="color:{color}">{v}</div>
  <div class="sub">{sub}</div>
</div>""", unsafe_allow_html=True)

kpi(c1, "NAIVE (BIASED)",    naive_est, "Raw earnings gap", BIAS_COLOR)
kpi(c2, "ATE — OLS",         ate,       "Regression adjusted", "#7ab4f5")
kpi(c3, "ATT — PSM",         att_psm,   f"{matched_n} matched pairs", GOOD_COLOR)
kpi(c4, "DiD — MATCHED",     did_est,   "Pre/post difference", "#a78bfa")

st.markdown(f"""
<div style='background:#1a0f0f; border:1px solid #5a1a1a; border-radius:6px;
            padding:12px 18px; font-family:IBM Plex Mono,monospace; font-size:0.82rem;
            color:#f87171; margin-top:6px'>
⚠ Confounding bias removed: <strong>${abs(bias):,.0f}</strong> — 
naive estimate {'overstated' if bias > 0 else 'understated'} the true effect.
True causal lift = <strong>${att_psm:+,.0f}</strong> in annual earnings.
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Pre-Match Balance",
    "🎯 Propensity Scores",
    "⚖ Post-Match Balance",
    "📈 Final Comparison",
])

# ── Tab 1: Pre-match group means ──────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Group Means — Before Matching</p>',
                unsafe_allow_html=True)

    cols_to_show = selected_covs + ["re78"]
    gs = df.groupby("treat")[cols_to_show].mean().T
    gs.columns = ["Control", "Treated"]
    gs["Difference"] = gs["Treated"] - gs["Control"]
    gs["SMD (before)"] = [smd(df, c) for c in gs.index]

    def highlight_smd(val):
        if abs(val) > 0.5:
            return "color: #f87171"
        elif abs(val) > 0.1:
            return "color: #fb923c"
        return "color: #4ade80"

    styled = gs.style\
        .format({"Control": "{:,.2f}", "Treated": "{:,.2f}",
                 "Difference": "{:+,.2f}", "SMD (before)": "{:+.3f}"})\
        .map(highlight_smd, subset=["SMD (before)"])

    st.dataframe(styled, use_container_width=True)

    st.caption("SMD > 0.1 means imbalanced. Red = large imbalance, orange = moderate.")

    # Plot earnings distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Pre-training vs Post-training Earnings by Group",
                 fontsize=11, fontweight="bold", color="#e8eaf6")

    for ax, col, title in zip(axes, ["re74", "re75", "re78"],
                               ["1974 Earnings", "1975 Earnings", "1978 Earnings (Outcome)"]):
        for treat_val, color, label in [
            (0, CONTROL_COLOR, "Control"), (1, TREAT_COLOR, "Treated")
        ]:
            data_plot = df[df["treat"] == treat_val][col]
            ax.hist(data_plot.clip(0, 30000), bins=30, alpha=0.6,
                    color=color, label=label, density=True)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Earnings ($)", fontsize=8)
        ax.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Tab 2: Propensity scores ──────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Propensity Score Distribution</p>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Before Matching**")
        fig, ax = plt.subplots(figsize=(6, 4))
        for treat_val, color, label in [
            (0, CONTROL_COLOR, "Control"), (1, TREAT_COLOR, "Treated")
        ]:
            grp = df_ps[df_ps["treat"] == treat_val]
            ax.hist(grp["propensity_score"], bins=25, alpha=0.65,
                    color=color, label=label, density=True)
        ax.set_xlabel("Propensity Score")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"All data  (n={len(df_ps)})", fontsize=9)
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**After Matching**")
        fig, ax = plt.subplots(figsize=(6, 4))
        for treat_val, color, label in [
            (0, CONTROL_COLOR, "Matched Control"), (1, TREAT_COLOR, "Treated")
        ]:
            grp = matched_df[matched_df["treat"] == treat_val]
            ax.hist(grp["propensity_score"], bins=25, alpha=0.65,
                    color=color, label=label, density=True)
        ax.set_xlabel("Propensity Score")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"Matched data  (n={len(matched_df)})", fontsize=9)
        st.pyplot(fig)
        plt.close()

    st.info(f"**{matched_n}** treated units matched out of **{int(treated_n)}** "
            f"(caliper = {caliper:.2f}). Overlap should look much more similar after matching.")

# ── Tab 3: Post-match balance (Love plot) ─────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">Covariate Balance — Love Plot</p>',
                unsafe_allow_html=True)

    smd_data = []
    for col in selected_covs:
        before = smd(df, col)
        after  = smd(matched_df, col)
        smd_data.append({"Variable": col, "Before": before, "After": after,
                         "Balanced": "✅" if abs(after) < 0.1 else "⚠️"})
    smd_df = pd.DataFrame(smd_data)

    fig, ax = plt.subplots(figsize=(9, max(4, len(selected_covs) * 0.55)))
    y = np.arange(len(smd_df))
    ax.scatter(smd_df["Before"], y, color=BIAS_COLOR, s=80, zorder=3, label="Before matching")
    ax.scatter(smd_df["After"],  y, color=GOOD_COLOR, s=80, zorder=3, label="After matching")
    for i, row in smd_df.iterrows():
        ax.plot([row["Before"], row["After"]], [i, i],
                color="#2a2d3e", lw=1.5, zorder=2)
    ax.axvline(0, color="#e8eaf6", lw=0.8, ls="--")
    ax.axvline( 0.1, color="#f97316", lw=0.7, ls=":", alpha=0.7)
    ax.axvline(-0.1, color="#f97316", lw=0.7, ls=":", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(smd_df["Variable"], fontsize=9)
    ax.set_xlabel("Standardised Mean Difference (SMD)")
    ax.set_title("Balance before vs after PSM  (|SMD| < 0.1 = balanced)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    st.pyplot(fig)
    plt.close()

    st.dataframe(
        smd_df.style.format({"Before": "{:+.3f}", "After": "{:+.3f}"}),
        use_container_width=True,
    )

    mean_before = smd_df["Before"].abs().mean()
    mean_after  = smd_df["After"].abs().mean()
    reduction   = (1 - mean_after / mean_before) * 100 if mean_before > 0 else 0
    st.success(f"Mean |SMD| reduced from **{mean_before:.3f}** → **{mean_after:.3f}**  "
               f"({reduction:.0f}% imbalance reduction)")

# ── Tab 4: Final comparison ───────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-header">All Estimates — Side by Side</p>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Naive vs Causal Estimates of Training Effect on Earnings",
                 fontsize=12, fontweight="bold", color="#e8eaf6")

    # Left: bar chart
    ax = axes[0]
    labels_bar = ["Naive\n(biased)", "ATE\n(OLS)", "ATT\n(PSM)", "DiD\n(matched)"]
    values_bar = [naive_est, ate, att_psm, did_est]
    colors_bar = [BIAS_COLOR, "#7ab4f5", GOOD_COLOR, "#a78bfa"]
    bars = ax.bar(labels_bar, values_bar, color=colors_bar,
                  alpha=0.85, width=0.5, edgecolor="#111320", linewidth=1.5)
    ax.axhline(0, color="#e8eaf6", lw=0.8)
    for bar, val in zip(bars, values_bar):
        ypos = val + 80 if val >= 0 else val - 300
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"${val:,.0f}", ha="center", fontsize=9, fontweight="bold",
                color="#e8eaf6")
    ax.set_ylabel("Estimated effect on 1978 earnings ($)")
    ax.set_title("All estimates", fontsize=10, fontweight="bold")

    # Right: bias decomposition
    ax = axes[1]
    cats = ["Naive estimate\n(observed)", "True effect\n(ATT after PSM)", "Confounding bias\n(naive − ATT)"]
    vals = [naive_est, att_psm, bias]
    cols = [BIAS_COLOR, GOOD_COLOR, WARN_COLOR]
    bars2 = ax.barh(cats, vals, color=cols, alpha=0.85, edgecolor="#111320")
    ax.axvline(0, color="#e8eaf6", lw=0.8)
    for bar, val in zip(bars2, vals):
        ax.text(val + 30, bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}", va="center", fontsize=10, fontweight="bold",
                color="#e8eaf6")
    ax.set_xlabel("Dollars ($)")
    ax.set_title("Bias decomposition", fontsize=10, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Summary box
    st.markdown(f"""
<div style='background:#0d1a0d; border:1px solid #1a4a1a; border-radius:8px;
            padding:20px 24px; font-family:IBM Plex Mono,monospace;
            font-size:0.82rem; color:#c0e8c0; line-height:2;'>
<strong style='color:#4ade80; font-size:0.9rem'>FINAL SUMMARY</strong><br><br>
Dataset      : LaLonde (1986) — {len(df)} observations<br>
Treatment    : Job training program (1=trained, 0=control)<br>
Outcome      : Annual earnings in 1978 (re78)<br>
Confounders  : {', '.join(selected_covs)}<br><br>
Naive estimate (biased)    : <strong style='color:#f87171'>${naive_est:>8,.0f}</strong><br>
ATE  — OLS regression      : <strong style='color:#7ab4f5'>${ate:>8,.0f}</strong><br>
ATT  — PSM matching        : <strong style='color:#4ade80'>${att_psm:>8,.0f}</strong><br>
DiD  — matched regression  : <strong style='color:#a78bfa'>${did_est:>8,.0f}</strong><br><br>
Confounding bias removed   : <strong style='color:#fb923c'>${abs(bias):>8,.0f}</strong><br><br>
Conclusion: Ignoring confounders {'overstated' if bias > 0 else 'understated'} the program's<br>
effect by ${abs(bias):,.0f}. Job training caused a TRUE<br>
increase of <strong style='color:#4ade80'>${att_psm:,.0f}</strong> in annual earnings.
</div>
""", unsafe_allow_html=True)


