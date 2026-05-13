"""
Survival Analysis Page -- Kaplan-Meier and Hazard Ratios
Requires: pip install lifelines scipy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

st.set_page_config(page_title="Survival Analysis", page_icon=None, layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
.section-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #58a6ff;
    text-transform: uppercase; letter-spacing: 0.15em;
    border-bottom: 1px solid #21262d; padding-bottom: 8px; margin-bottom: 16px;
}
.stat-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 16px 20px; text-align: center;
}
.stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #58a6ff; }
.stat-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

TMB_COUNTED_VARIANTS = [
    "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
    "In_Frame_Del","In_Frame_Ins","Splice_Site","Translation_Start_Site","Nonstop_Mutation",
]

ACTIONABLE_GENES = [
    "EGFR","KRAS","BRAF","TP53","STK11","KEAP1","RB1",
    "BRCA1","BRCA2","PIK3CA","PTEN","MET","RET","ALK","ERBB2",
]

EXOME_SIZE_MB = 38.0

PALETTE = [
    "#58a6ff","#f85149","#3fb950","#e3b341",
    "#d2a8ff","#f0883e","#bc8cff","#ff7b72",
]

# ── GDC clinical data ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def download_clinical_gdc(cancer_type: str = "TCGA-LUAD") -> pd.DataFrame:
    GDC = "https://api.gdc.cancer.gov"
    fields = [
        "submitter_id",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.age_at_diagnosis",
        "diagnoses.tumor_stage",
    ]
    filters = {
        "op": "in",
        "content": {"field": "project.project_id", "value": [cancer_type]}
    }
    params = {
        "filters":  json.dumps(filters),
        "fields":   ",".join(fields),
        "format":   "JSON",
        "size":     2000,
        "expand":   "demographic,diagnoses",
    }
    resp = requests.get(f"{GDC}/cases", params=params, timeout=30)
    hits = resp.json()["data"]["hits"]
    rows = []
    for h in hits:
        demo  = h.get("demographic", {})
        diags = h.get("diagnoses", [{}])
        diag  = diags[0] if diags else {}
        vital = demo.get("vital_status", None)
        dtd   = demo.get("days_to_death", None)
        dtlf  = diag.get("days_to_last_follow_up", None)
        dur   = dtd if (vital == "Dead" and dtd) else dtlf
        rows.append({
            "sample_id":     h["submitter_id"],
            "vital_status":  vital,
            "duration_days": dur,
            "event":         1 if vital == "Dead" else 0,
            "age_at_dx":     diag.get("age_at_diagnosis", None),
            "tumor_stage":   diag.get("tumor_stage", None),
        })
    df = pd.DataFrame(rows)
    df["duration_days"] = pd.to_numeric(df["duration_days"], errors="coerce")
    df = df.dropna(subset=["duration_days"])
    df = df[df["duration_days"] > 0]
    df["duration_months"] = (df["duration_days"] / 30.44).round(1)
    return df

# ── Local data ────────────────────────────────────────────────────────────────

@st.cache_data
def load_tmb_data():
    p = Path("data/tmb_scores.csv")
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_mutation_data():
    csvs = list(Path("data").glob("*_merged.csv"))
    return pd.read_csv(csvs[0]) if csvs else None

def get_cols(df):
    gene_col    = next((c for c in ["Hugo_Symbol","gene_hugoGeneSymbol"] if c in df.columns), None)
    sample_col  = next((c for c in ["Tumor_Sample_Barcode","sampleId"]   if c in df.columns), None)
    variant_col = next((c for c in ["Variant_Classification","mutationType"] if c in df.columns), None)
    return gene_col, sample_col, variant_col

def match_sample_ids(clinical_df, tmb_df):
    tmb_df      = tmb_df.copy()
    clinical_df = clinical_df.copy()
    tmb_df["patient_id"]      = tmb_df["sample_id"].str[:12]
    clinical_df["patient_id"] = clinical_df["sample_id"].str[:12]
    return clinical_df.merge(tmb_df, on="patient_id", how="inner", suffixes=("_clin","_tmb"))

def build_gene_flags(mut_df, gene_col, sample_col, variant_col, genes):
    df_coding = mut_df[mut_df[variant_col].isin(TMB_COUNTED_VARIANTS)]
    flags = {}
    for gene in genes:
        flags[gene] = df_coding[df_coding[gene_col] == gene][sample_col].unique()
    return flags

# ── Plot helpers ──────────────────────────────────────────────────────────────

def km_trace(kmf, label, color, show_ci=True):
    t  = kmf.timeline
    sf = kmf.survival_function_.iloc[:, 0].values
    traces = [go.Scatter(
        x=t, y=sf, mode="lines", name=label,
        line=dict(color=color, width=2.5),
        hovertemplate=f"<b>{label}</b><br>Time: %{{x:.0f}} mo<br>Survival: %{{y:.3f}}<extra></extra>",
    )]
    if show_ci:
        ci_u = kmf.confidence_interval_.iloc[:, 1].values
        ci_l = kmf.confidence_interval_.iloc[:, 0].values
        r    = int(color[1:3], 16)
        g    = int(color[3:5], 16)
        b    = int(color[5:7], 16)
        traces.append(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([ci_u, ci_l[::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
    return traces

def base_km_layout(title="", height=420):
    return dict(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#e6edf3", font_family="IBM Plex Mono",
        title=dict(text=title, font=dict(size=13, color="#8b949e")),
        xaxis=dict(title="Time (months)", color="#8b949e", gridcolor="#21262d", zeroline=False),
        yaxis=dict(title="Survival Probability", color="#8b949e", gridcolor="#21262d",
                   range=[0, 1.05], tickformat=".0%"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1, font=dict(size=11)),
        margin=dict(l=0, r=0, t=40, b=0),
        height=height,
    )

def pval_badge(p):
    sig   = p < 0.05
    color = "#3fb950" if sig else "#8b949e"
    label = "significant" if sig else "not significant"
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    return f"<span style='color:{color};font-family:IBM Plex Mono'>{stars} p = {p:.4f} ({label})</span>"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Survival Analysis")
    st.markdown("---")
    cancer_type = st.selectbox(
        "Cancer Type (GDC)",
        ["TCGA-LUAD","TCGA-BRCA","TCGA-COAD","TCGA-SKCM","TCGA-GBM","TCGA-PRAD"],
        index=0
    )
    show_ci   = st.toggle("Show confidence intervals", value=True)
    time_unit = st.radio("Time axis", ["Months","Days"], horizontal=True)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#8b949e'>"
        "Clinical data is fetched live from GDC.<br>"
        "Mutation data loaded from your local data/ folder.</div>",
        unsafe_allow_html=True
    )

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# Survival Analysis")
st.markdown(
    "<div style='color:#8b949e;font-size:0.9rem;margin-bottom:24px'>"
    "Kaplan-Meier curves | TMB stratification | Gene-specific survival | Hazard ratios"
    "</div>",
    unsafe_allow_html=True
)

# ── Load data ─────────────────────────────────────────────────────────────────

col_load1, col_load2 = st.columns(2)

with col_load1:
    with st.spinner(f"Fetching clinical data from GDC ({cancer_type})..."):
        try:
            clinical_df = download_clinical_gdc(cancer_type)
            st.success(f"{len(clinical_df)} patients with survival data")
        except Exception as e:
            st.error(f"GDC fetch failed: {e}")
            st.stop()

with col_load2:
    tmb_df = load_tmb_data()
    mut_df = load_mutation_data()
    if tmb_df is not None:
        st.success(f"{len(tmb_df)} samples with TMB scores")
    else:
        st.warning("No TMB data -- run python3 tmb_data_download.py first")

merged_df = match_sample_ids(clinical_df, tmb_df) if tmb_df is not None else pd.DataFrame()

time_col   = "duration_months" if time_unit == "Months" else "duration_days"
time_label = "months" if time_unit == "Months" else "days"

# ── Metrics ───────────────────────────────────────────────────────────────────

n_patients = len(clinical_df)
n_events   = int(clinical_df["event"].sum())
median_fu  = clinical_df[time_col].median()
pct_dead   = round(n_events / n_patients * 100, 1) if n_patients else 0

c1, c2, c3, c4 = st.columns(4)
for col_w, val, label in zip(
    [c1, c2, c3, c4],
    [n_patients, n_events, f"{pct_dead}%", f"{median_fu:.0f}"],
    ["Total Patients", "Deaths (Events)", "Mortality Rate", f"Median Follow-up ({time_label})"]
):
    col_w.markdown(
        f"<div class='stat-card'>"
        f"<div class='stat-value'>{val}</div>"
        f"<div class='stat-label'>{label}</div></div>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 1: Overall survival ───────────────────────────────────────────────

st.markdown('<div class="section-header">Overall Survival -- All Patients</div>', unsafe_allow_html=True)

kmf_all = KaplanMeierFitter()
kmf_all.fit(
    clinical_df[time_col].clip(lower=0),
    event_observed=clinical_df["event"],
    label="All Patients"
)

fig_all = go.Figure()
fig_all.add_traces(km_trace(kmf_all, "All Patients", PALETTE[0], show_ci))
fig_all.update_layout(**base_km_layout(f"Overall Survival -- {cancer_type}"))
st.plotly_chart(fig_all, use_container_width=True)

median_os = kmf_all.median_survival_time_
st.markdown(
    f"<div style='font-size:0.78rem;color:#8b949e;margin-top:-8px'>"
    f"Median overall survival: <b style='color:#58a6ff'>{median_os:.1f} {time_label}</b> &nbsp;|&nbsp; "
    f"n = {n_patients} patients &nbsp;|&nbsp; {n_events} events</div>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 2: TMB high vs low ────────────────────────────────────────────────

st.markdown('<div class="section-header">Survival by TMB Status (High vs Low)</div>', unsafe_allow_html=True)

if merged_df.empty:
    st.info("No matched samples between TMB data and clinical data. Make sure you have downloaded mutation data for the same cancer type.")
else:
    tmb_threshold = st.slider("TMB threshold (mut/Mb)", 1.0, 30.0, 10.0, step=0.5)

    merged_df["TMB_group"] = merged_df["TMB"].apply(
        lambda x: "TMB-High" if x >= tmb_threshold else "TMB-Low"
    )
    grp_high = merged_df[merged_df["TMB_group"] == "TMB-High"]
    grp_low  = merged_df[merged_df["TMB_group"] == "TMB-Low"]

    if len(grp_high) < 3 or len(grp_low) < 3:
        st.warning(f"Not enough samples in each group (High: {len(grp_high)}, Low: {len(grp_low)}). Try adjusting the threshold.")
    else:
        kmf_h = KaplanMeierFitter()
        kmf_l = KaplanMeierFitter()
        kmf_h.fit(grp_high[time_col].clip(lower=0), grp_high["event"], label=f"TMB-High (>={tmb_threshold})")
        kmf_l.fit(grp_low[time_col].clip(lower=0),  grp_low["event"],  label=f"TMB-Low (<{tmb_threshold})")

        lr    = logrank_test(
            grp_high[time_col], grp_low[time_col],
            event_observed_A=grp_high["event"],
            event_observed_B=grp_low["event"]
        )
        p_val = lr.p_value

        fig_tmb = go.Figure()
        fig_tmb.add_traces(km_trace(kmf_h, f"TMB-High (n={len(grp_high)})", PALETTE[0], show_ci))
        fig_tmb.add_traces(km_trace(kmf_l, f"TMB-Low (n={len(grp_low)})",   PALETTE[1], show_ci))
        fig_tmb.update_layout(**base_km_layout(f"TMB-High vs TMB-Low -- {cancer_type}"))
        st.plotly_chart(fig_tmb, use_container_width=True)

        st.markdown(
            f"<div style='font-size:0.82rem;margin-top:-8px'>"
            f"Log-rank test: {pval_badge(p_val)} &nbsp;|&nbsp; "
            f"Median High: <b style='color:{PALETTE[0]}'>{kmf_h.median_survival_time_:.1f} {time_label}</b> &nbsp;|&nbsp; "
            f"Median Low: <b style='color:{PALETTE[1]}'>{kmf_l.median_survival_time_:.1f} {time_label}</b>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

# ── Section 3: TMB tertile ────────────────────────────────────────────────────

if not merged_df.empty and len(merged_df) >= 9:
    st.markdown('<div class="section-header">Survival by TMB Tertile (Low / Medium / High)</div>', unsafe_allow_html=True)

    t33, t66 = merged_df["TMB"].quantile([0.33, 0.66])
    merged_df["TMB_tertile"] = pd.cut(
        merged_df["TMB"],
        bins=[-np.inf, t33, t66, np.inf],
        labels=["Low","Medium","High"]
    )

    fig_tert = go.Figure()
    for grp, color, label in zip(
        ["Low","Medium","High"],
        [PALETTE[2], PALETTE[3], PALETTE[0]],
        [f"Low (<{t33:.1f})", f"Medium ({t33:.1f}-{t66:.1f})", f"High (>={t66:.1f})"]
    ):
        sub = merged_df[merged_df["TMB_tertile"] == grp]
        if len(sub) < 3:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub[time_col].clip(lower=0), sub["event"], label=f"{label} (n={len(sub)})")
        fig_tert.add_traces(km_trace(kmf, f"{label} (n={len(sub)})", color, show_ci))

    mlr = multivariate_logrank_test(
        merged_df[time_col].clip(lower=0),
        merged_df["TMB_tertile"].astype(str),
        event_observed=merged_df["event"]
    )
    fig_tert.update_layout(**base_km_layout(f"TMB Tertile Survival -- {cancer_type}"))
    st.plotly_chart(fig_tert, use_container_width=True)
    st.markdown(
        f"<div style='font-size:0.82rem;margin-top:-8px'>"
        f"Multivariate log-rank: {pval_badge(mlr.p_value)}</div>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

# ── Section 4: Gene-specific survival ─────────────────────────────────────────

st.markdown('<div class="section-header">Survival by Gene Mutation Status</div>', unsafe_allow_html=True)

if mut_df is None or merged_df.empty:
    st.info("Mutation data and matched clinical data required for this section.")
else:
    gene_col, sample_col, variant_col = get_cols(mut_df)
    gene_flags = build_gene_flags(mut_df, gene_col, sample_col, variant_col, ACTIONABLE_GENES)

    available_genes = [g for g in ACTIONABLE_GENES if len(gene_flags.get(g, [])) >= 3]
    if not available_genes:
        st.info("Not enough mutated samples per gene for survival analysis (need at least 3).")
    else:
        selected_gene = st.selectbox("Select gene", available_genes, index=0)

        mutated_samples = set(gene_flags[selected_gene])
        merged_df["patient_id_short"] = merged_df["patient_id"].str[:12]

        grp_mut = merged_df[merged_df["patient_id_short"].isin([s[:12] for s in mutated_samples])]
        grp_wt  = merged_df[~merged_df["patient_id_short"].isin([s[:12] for s in mutated_samples])]

        if len(grp_mut) < 3 or len(grp_wt) < 3:
            st.warning(f"Not enough matched samples: Mutated={len(grp_mut)}, WT={len(grp_wt)}")
        else:
            kmf_mut = KaplanMeierFitter()
            kmf_wt  = KaplanMeierFitter()
            kmf_mut.fit(grp_mut[time_col].clip(lower=0), grp_mut["event"],
                        label=f"{selected_gene} Mutated (n={len(grp_mut)})")
            kmf_wt.fit(grp_wt[time_col].clip(lower=0),  grp_wt["event"],
                       label=f"{selected_gene} Wild-type (n={len(grp_wt)})")

            lr_gene = logrank_test(
                grp_mut[time_col], grp_wt[time_col],
                event_observed_A=grp_mut["event"],
                event_observed_B=grp_wt["event"]
            )

            fig_gene = go.Figure()
            fig_gene.add_traces(km_trace(kmf_mut, f"{selected_gene} Mutated (n={len(grp_mut)})", PALETTE[1], show_ci))
            fig_gene.add_traces(km_trace(kmf_wt,  f"{selected_gene} WT (n={len(grp_wt)})",      PALETTE[0], show_ci))
            fig_gene.update_layout(**base_km_layout(f"{selected_gene} Mutation -- Survival Impact"))
            st.plotly_chart(fig_gene, use_container_width=True)

            st.markdown(
                f"<div style='font-size:0.82rem;margin-top:-8px'>"
                f"Log-rank test: {pval_badge(lr_gene.p_value)} &nbsp;|&nbsp; "
                f"Median Mutated: <b style='color:{PALETTE[1]}'>{kmf_mut.median_survival_time_:.1f} {time_label}</b> &nbsp;|&nbsp; "
                f"Median WT: <b style='color:{PALETTE[0]}'>{kmf_wt.median_survival_time_:.1f} {time_label}</b>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

# ── Section 5: Hazard ratios forest plot ──────────────────────────────────────

st.markdown('<div class="section-header">Hazard Ratios -- Gene Mutation Forest Plot</div>', unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:0.78rem;color:#8b949e;margin-bottom:12px'>"
    "Cox proportional hazards model. HR > 1 = worse survival when mutated.</div>",
    unsafe_allow_html=True
)

if mut_df is None or merged_df.empty:
    st.info("Mutation data and clinical data required.")
else:
    hr_rows = []
    for gene in ACTIONABLE_GENES:
        mutated = set(gene_flags.get(gene, []))
        if len(mutated) < 3:
            continue
        merged_df[f"mut_{gene}"] = merged_df["patient_id_short"].isin(
            [s[:12] for s in mutated]
        ).astype(int)
        sub = merged_df[[time_col, "event", f"mut_{gene}"]].dropna()
        if sub[f"mut_{gene}"].sum() < 3:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(sub, duration_col=time_col, event_col="event", formula=f"mut_{gene}")
            summary = cph.summary
            hr      = float(summary.loc[f"mut_{gene}", "exp(coef)"])
            ci_low  = float(summary.loc[f"mut_{gene}", "exp(coef) lower 95%"])
            ci_high = float(summary.loc[f"mut_{gene}", "exp(coef) upper 95%"])
            p       = float(summary.loc[f"mut_{gene}", "p"])
            hr_rows.append({
                "Gene": gene, "HR": hr, "CI_low": ci_low, "CI_high": ci_high,
                "p-value": p, "n_mutated": int(sub[f"mut_{gene}"].sum())
            })
        except Exception:
            continue

    if not hr_rows:
        st.info("Could not compute hazard ratios -- not enough matched samples per gene.")
    else:
        hr_df  = pd.DataFrame(hr_rows).sort_values("HR", ascending=True)
        colors = ["#f85149" if hr > 1 else "#3fb950" for hr in hr_df["HR"]]
        sig    = hr_df["p-value"] < 0.05

        fig_forest = go.Figure()
        for _, row in hr_df.iterrows():
            color = "#f85149" if row["HR"] > 1 else "#3fb950"
            fig_forest.add_trace(go.Scatter(
                x=[row["CI_low"], row["CI_high"]],
                y=[row["Gene"], row["Gene"]],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False, hoverinfo="skip",
            ))

        fig_forest.add_trace(go.Scatter(
            x=hr_df["HR"], y=hr_df["Gene"],
            mode="markers",
            marker=dict(
                size=10, color=colors,
                symbol=["star" if s else "circle" for s in sig],
                line=dict(color="#0d1117", width=1),
            ),
            hovertemplate=(
                "<b>%{y}</b><br>HR: %{x:.2f}<br>"
                "n mutated: %{customdata[0]}<br>p-value: %{customdata[1]:.4f}<extra></extra>"
            ),
            customdata=hr_df[["n_mutated","p-value"]].values,
            showlegend=False,
        ))

        fig_forest.add_vline(x=1, line_dash="dash", line_color="#8b949e",
                             annotation_text="HR=1", annotation_font_color="#8b949e")
        fig_forest.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
            font_family="IBM Plex Mono",
            xaxis=dict(title="Hazard Ratio (95% CI)", color="#8b949e",
                       gridcolor="#21262d", type="log"),
            yaxis=dict(title="", color="#8b949e", tickfont=dict(size=11)),
            margin=dict(l=10, r=10, t=10, b=0),
            height=max(300, len(hr_df) * 38),
        )
        st.plotly_chart(fig_forest, use_container_width=True)
        st.markdown(
            "<div style='font-size:0.72rem;color:#8b949e'>"
            "<span style='color:#f85149'>-- HR > 1 (worse survival)</span> &nbsp;"
            "<span style='color:#3fb950'>-- HR < 1 (better survival)</span> &nbsp;"
            "star = p &lt; 0.05</div>",
            unsafe_allow_html=True
        )

        with st.expander("Full hazard ratio table"):
            display_hr = hr_df.copy()
            display_hr["HR (95% CI)"] = display_hr.apply(
                lambda r: f"{r['HR']:.2f} ({r['CI_low']:.2f}-{r['CI_high']:.2f})", axis=1
            )
            display_hr["Significant"] = display_hr["p-value"].apply(lambda p: "Yes" if p < 0.05 else "No")
            st.dataframe(
                display_hr[["Gene","HR (95% CI)","p-value","n_mutated","Significant"]].reset_index(drop=True),
                use_container_width=True
            )
            st.download_button("Download HR table", hr_df.to_csv(index=False).encode(), "hazard_ratios.csv", "text/csv")

# ── Section 6: Custom clinical upload ─────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Upload Custom Clinical Data</div>', unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:0.78rem;color:#8b949e;margin-bottom:12px'>"
    "Upload your own clinical CSV with columns: "
    "sample_id, days_to_death, days_to_last_followup, vital_status</div>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader("Upload clinical file (CSV or TSV)", type=["csv","tsv","txt"])

if uploaded:
    sep        = "\t" if uploaded.name.endswith((".tsv",".txt")) else ","
    custom_df  = pd.read_csv(uploaded, sep=sep, comment="#")
    st.dataframe(custom_df.head(), use_container_width=True)

    sid_col   = next((c for c in custom_df.columns if "sample" in c.lower() or "barcode" in c.lower()), None)
    dtd_col   = next((c for c in custom_df.columns if "days_to_death" in c.lower()), None)
    dtlf_col  = next((c for c in custom_df.columns if "last_follow" in c.lower()), None)
    vital_col = next((c for c in custom_df.columns if "vital" in c.lower()), None)

    if all([sid_col, dtd_col, dtlf_col, vital_col]):
        custom_df["event"]    = (custom_df[vital_col].str.lower() == "dead").astype(int)
        custom_df["duration"] = custom_df.apply(
            lambda r: r[dtd_col] if r["event"] == 1 and pd.notna(r[dtd_col]) else r[dtlf_col], axis=1
        )
        custom_df["duration"] = pd.to_numeric(custom_df["duration"], errors="coerce")
        custom_df = custom_df.dropna(subset=["duration"])
        custom_df = custom_df[custom_df["duration"] > 0]
        custom_df["duration_months"] = (custom_df["duration"] / 30.44).round(1)

        kmf_custom = KaplanMeierFitter()
        kmf_custom.fit(custom_df["duration_months"], custom_df["event"], label="Custom Cohort")

        fig_custom = go.Figure()
        fig_custom.add_traces(km_trace(kmf_custom, "Custom Cohort", PALETTE[4], show_ci))
        fig_custom.update_layout(**base_km_layout("Overall Survival -- Custom Upload"))
        st.plotly_chart(fig_custom, use_container_width=True)
        st.success(f"Parsed {len(custom_df)} patients | Median OS: {kmf_custom.median_survival_time_:.1f} months")
    else:
        st.warning(
            f"Could not auto-detect all required columns. "
            f"Found: sample={sid_col}, days_to_death={dtd_col}, "
            f"last_followup={dtlf_col}, vital_status={vital_col}"
        )