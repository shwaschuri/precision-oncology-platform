"""
MSI Prediction Page -- Microsatellite Instability Estimator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="MSI Prediction", page_icon=None, layout="wide")

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
.msi-h-card {
    background: linear-gradient(135deg, #1a2e1a, #162416);
    border: 1px solid #3fb950; border-radius: 10px; padding: 20px 24px; margin-bottom: 12px;
}
.msi-l-card {
    background: linear-gradient(135deg, #1a1a2e, #161624);
    border: 1px solid #58a6ff; border-radius: 10px; padding: 20px 24px; margin-bottom: 12px;
}
.msi-badge-H {
    display: inline-block; background: #3fb950; color: #0d1117;
    font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 1rem;
    padding: 4px 14px; border-radius: 20px; letter-spacing: 0.08em;
}
.msi-badge-L {
    display: inline-block; background: #58a6ff; color: #0d1117;
    font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 1rem;
    padding: 4px 14px; border-radius: 20px; letter-spacing: 0.08em;
}
.msi-badge-IND {
    display: inline-block; background: #e3b341; color: #0d1117;
    font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 1rem;
    padding: 4px 14px; border-radius: 20px; letter-spacing: 0.08em;
}
.rec-card {
    background: #161b22; border-left: 3px solid #3fb950;
    border-radius: 0 8px 8px 0; padding: 14px 18px; margin-bottom: 8px;
}
.rec-card-warn {
    background: #161b22; border-left: 3px solid #e3b341;
    border-radius: 0 8px 8px 0; padding: 14px 18px; margin-bottom: 8px;
}
div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

EXOME_SIZE_MB = 38.0

TMB_COUNTED_VARIANTS = [
    "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
    "In_Frame_Del","In_Frame_Ins","Splice_Site","Translation_Start_Site","Nonstop_Mutation",
]

MMR_GENES = {
    "MLH1":  {"role": "MMR",          "note": "Most common cause of Lynch syndrome MSI"},
    "MSH2":  {"role": "MMR",          "note": "Lynch syndrome; MSH2 loss causes MSI-H"},
    "MSH6":  {"role": "MMR",          "note": "Lynch syndrome variant; partial MMR loss"},
    "PMS2":  {"role": "MMR",          "note": "Lynch syndrome; elevated indel burden"},
    "POLE":  {"role": "Proofreading", "note": "Ultramutator phenotype; very high TMB"},
    "POLD1": {"role": "Proofreading", "note": "Proofreading defect; hypermutation"},
    "MLH3":  {"role": "MMR",          "note": "Minor MMR role; rarely causes MSI alone"},
    "MSH3":  {"role": "MMR",          "note": "Dinucleotide instability when lost"},
    "EPCAM": {"role": "MMR",          "note": "Silences MSH2 via promoter methylation"},
}

INDEL_TYPES = [
    "Frame_Shift_Del","Frame_Shift_Ins","In_Frame_Del","In_Frame_Ins",
]

CHECKPOINT_INHIBITORS = [
    {"drug": "Pembrolizumab (Keytruda)", "target": "PD-1",   "approval": "FDA-approved (MSI-H/dMMR, pan-tumour)"},
    {"drug": "Nivolumab (Opdivo)",       "target": "PD-1",   "approval": "FDA-approved (MSI-H CRC)"},
    {"drug": "Dostarlimab (Jemperli)",   "target": "PD-1",   "approval": "FDA-approved (dMMR endometrial)"},
    {"drug": "Ipilimumab (Yervoy)",      "target": "CTLA-4", "approval": "Used in combination with nivolumab"},
]

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_mutation_data():
    csvs = list(Path("data").glob("*_merged.csv"))
    return pd.read_csv(csvs[0]) if csvs else None

@st.cache_data
def load_tmb_data():
    p = Path("data/tmb_scores.csv")
    return pd.read_csv(p) if p.exists() else None

def get_cols(df):
    gene_col    = next((c for c in ["Hugo_Symbol","gene_hugoGeneSymbol"] if c in df.columns), None)
    sample_col  = next((c for c in ["Tumor_Sample_Barcode","sampleId"]   if c in df.columns), None)
    variant_col = next((c for c in ["Variant_Classification","mutationType"] if c in df.columns), None)
    return gene_col, sample_col, variant_col

# ── MSI scoring engine ────────────────────────────────────────────────────────

def compute_msi_scores(mut_df, tmb_df, gene_col, sample_col, variant_col) -> pd.DataFrame:
    df_coding   = mut_df[mut_df[variant_col].isin(TMB_COUNTED_VARIANTS)]
    all_samples = mut_df[sample_col].unique()
    rows = []

    for sample in all_samples:
        s_all    = mut_df[mut_df[sample_col] == sample]
        s_coding = df_coding[df_coding[sample_col] == sample]

        # Signal 1: TMB
        tmb_row = tmb_df[tmb_df["sample_id"] == sample] if tmb_df is not None else pd.DataFrame()
        if not tmb_row.empty:
            tmb_val = float(tmb_row["TMB"].values[0])
        else:
            tmb_val = len(s_coding) / EXOME_SIZE_MB

        if tmb_val >= 20:
            tmb_score = 60
        elif tmb_val >= 10:
            tmb_score = 40
        elif tmb_val >= 5:
            tmb_score = 20
        else:
            tmb_score = 0

        # Signal 2: Indel fraction
        n_total   = len(s_coding)
        n_indels  = len(s_coding[s_coding[variant_col].isin(INDEL_TYPES)])
        indel_frac = (n_indels / n_total) if n_total > 0 else 0.0

        if indel_frac >= 0.20:
            indel_score = 30
        elif indel_frac >= 0.10:
            indel_score = 15
        else:
            indel_score = 0

        # Signal 3: MMR/POLE gene mutations
        sample_genes = set(s_all[gene_col].unique()) if gene_col else set()
        mmr_hits     = [g for g in MMR_GENES if g in sample_genes]
        pole_hit     = "POLE" in mmr_hits or "POLD1" in mmr_hits
        mmr_score    = min(len(mmr_hits) * 10, 30)
        if pole_hit:
            mmr_score = min(mmr_score + 10, 40)

        raw_score = tmb_score + indel_score + mmr_score

        if raw_score >= 60:
            msi_status = "MSI-H"
        elif raw_score >= 30:
            msi_status = "Indeterminate"
        else:
            msi_status = "MSS"

        if msi_status == "MSI-H":
            immuno_rec = "Checkpoint inhibitor eligible"
        elif msi_status == "Indeterminate":
            immuno_rec = "Consider MMR IHC / PCR confirmation"
        else:
            immuno_rec = "Not recommended (standard chemotherapy)"

        rows.append({
            "sample_id":       sample,
            "TMB":             round(tmb_val, 2),
            "indel_count":     n_indels,
            "total_coding":    n_total,
            "indel_fraction":  round(indel_frac * 100, 1),
            "MMR_genes_hit":   ", ".join(mmr_hits) if mmr_hits else "None",
            "POLE_mutated":    pole_hit,
            "TMB_score":       tmb_score,
            "indel_score":     indel_score,
            "MMR_score":       mmr_score,
            "MSI_score":       min(raw_score, 100),
            "MSI_status":      msi_status,
            "immuno_rec":      immuno_rec,
        })

    return pd.DataFrame(rows).sort_values("MSI_score", ascending=False).reset_index(drop=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## MSI Prediction")
    st.markdown("---")
    st.markdown("### Score Thresholds")
    msi_h_thresh = st.slider("MSI-H threshold", 40, 80, 60, step=5)
    mss_thresh   = st.slider("MSS threshold",   10, 50, 30, step=5)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#8b949e'>"
        "MSI score is a composite of:<br>"
        "- TMB (weight: 60%)<br>"
        "- Indel fraction (weight: 30%)<br>"
        "- MMR/POLE mutations (weight: 10%)<br><br>"
        "This is a computational estimate. "
        "Clinical MSI status requires PCR or IHC confirmation.</div>",
        unsafe_allow_html=True
    )

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# MSI Prediction")
st.markdown(
    "<div style='color:#8b949e;font-size:0.9rem;margin-bottom:24px'>"
    "Microsatellite instability estimation | MMR deficiency | Immunotherapy eligibility"
    "</div>",
    unsafe_allow_html=True
)

# ── Load data ─────────────────────────────────────────────────────────────────

mut_df = load_mutation_data()
tmb_df = load_tmb_data()

if mut_df is None:
    st.warning("No mutation data found. Run python3 tmb_data_download.py first.")
    st.stop()

gene_col, sample_col, variant_col = get_cols(mut_df)
if not all([gene_col, sample_col, variant_col]):
    st.error(f"Missing required columns. Found: {list(mut_df.columns[:10])}")
    st.stop()

with st.spinner("Computing MSI scores for all samples..."):
    msi_df = compute_msi_scores(mut_df, tmb_df, gene_col, sample_col, variant_col)

msi_df["MSI_status"] = msi_df["MSI_score"].apply(
    lambda s: "MSI-H" if s >= msi_h_thresh else ("MSS" if s < mss_thresh else "Indeterminate")
)
msi_df["immuno_rec"] = msi_df["MSI_status"].map({
    "MSI-H":         "Checkpoint inhibitor eligible",
    "Indeterminate": "Consider MMR IHC / PCR confirmation",
    "MSS":           "Not recommended (standard chemotherapy)",
})

# ── Metrics ───────────────────────────────────────────────────────────────────

n_total   = len(msi_df)
n_msi_h   = int((msi_df["MSI_status"] == "MSI-H").sum())
n_mss     = int((msi_df["MSI_status"] == "MSS").sum())
n_ind     = int((msi_df["MSI_status"] == "Indeterminate").sum())
pct_msi_h = round(n_msi_h / n_total * 100, 1) if n_total else 0
mean_score = msi_df["MSI_score"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
for col_w, val, label in zip(
    [c1, c2, c3, c4, c5],
    [n_total, f"{n_msi_h} ({pct_msi_h}%)", n_mss, n_ind, f"{mean_score:.1f}"],
    ["Total Samples", "MSI-H", "MSS", "Indeterminate", "Mean MSI Score"]
):
    col_w.markdown(
        f"<div class='stat-card'>"
        f"<div class='stat-value'>{val}</div>"
        f"<div class='stat-label'>{label}</div></div>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 1: MSI score bar ──────────────────────────────────────────────────

st.markdown('<div class="section-header">MSI Score per Sample</div>', unsafe_allow_html=True)

color_map  = {"MSI-H": "#3fb950", "Indeterminate": "#e3b341", "MSS": "#58a6ff"}
bar_colors = msi_df["MSI_status"].map(color_map).tolist()

fig_bar = go.Figure(go.Bar(
    x=msi_df["sample_id"].str[:16],
    y=msi_df["MSI_score"],
    marker_color=bar_colors,
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"
        "MSI Score: %{y}<br>Status: %{customdata[1]}<br>"
        "TMB: %{customdata[2]} mut/Mb<br>"
        "Indel fraction: %{customdata[3]}%<br>"
        "MMR genes: %{customdata[4]}<extra></extra>"
    ),
    customdata=msi_df[["sample_id","MSI_status","TMB","indel_fraction","MMR_genes_hit"]].values,
))
fig_bar.add_hline(y=msi_h_thresh, line_dash="dash", line_color="#3fb950",
                  annotation_text=f"MSI-H threshold ({msi_h_thresh})",
                  annotation_font_color="#3fb950")
fig_bar.add_hline(y=mss_thresh, line_dash="dash", line_color="#e3b341",
                  annotation_text=f"MSS threshold ({mss_thresh})",
                  annotation_font_color="#e3b341")
fig_bar.update_layout(
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
    font_family="IBM Plex Mono",
    xaxis=dict(showticklabels=False, title="Samples", color="#8b949e"),
    yaxis=dict(title="MSI Score (0-100)", color="#8b949e", gridcolor="#21262d", range=[0,105]),
    margin=dict(l=0,r=0,t=10,b=0), height=320,
)
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown(
    "<div style='font-size:0.72rem;color:#8b949e'>"
    "<span style='color:#3fb950'>-- MSI-H</span> &nbsp;"
    "<span style='color:#e3b341'>-- Indeterminate</span> &nbsp;"
    "<span style='color:#58a6ff'>-- MSS</span></div>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ── Section 2: Donut + scatter ────────────────────────────────────────────────

col_pie, col_scatter = st.columns(2)

with col_pie:
    st.markdown('<div class="section-header">MSI Status Distribution</div>', unsafe_allow_html=True)
    pie_counts = msi_df["MSI_status"].value_counts().reset_index()
    pie_counts.columns = ["status","count"]
    fig_pie = go.Figure(go.Pie(
        labels=pie_counts["status"],
        values=pie_counts["count"],
        marker_colors=[color_map.get(s,"#30363d") for s in pie_counts["status"]],
        hole=0.55, textinfo="label+percent",
        hovertemplate="%{label}<br>%{value} samples (%{percent})<extra></extra>",
    ))
    fig_pie.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        font_family="IBM Plex Mono", showlegend=False,
        margin=dict(l=0,r=0,t=10,b=0), height=300,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_scatter:
    st.markdown('<div class="section-header">TMB vs Indel Fraction</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.75rem;color:#8b949e;margin-bottom:8px'>"
        "MSI-H samples cluster toward high TMB and high indel fraction.</div>",
        unsafe_allow_html=True
    )
    fig_sc = go.Figure()
    for status, color in color_map.items():
        sub = msi_df[msi_df["MSI_status"] == status]
        if sub.empty:
            continue
        fig_sc.add_trace(go.Scatter(
            x=sub["TMB"], y=sub["indel_fraction"],
            mode="markers", name=status,
            marker=dict(size=9, color=color, line=dict(color="#0d1117",width=1)),
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "TMB: %{x:.1f} mut/Mb<br>"
                "Indel fraction: %{y:.1f}%<extra></extra>"
            ),
            customdata=sub["sample_id"],
        ))
    fig_sc.add_vline(x=10, line_dash="dot", line_color="#8b949e",
                     annotation_text="TMB=10", annotation_font_color="#8b949e", annotation_font_size=10)
    fig_sc.add_hline(y=20, line_dash="dot", line_color="#8b949e",
                     annotation_text="Indel 20%", annotation_font_color="#8b949e", annotation_font_size=10)
    fig_sc.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        font_family="IBM Plex Mono",
        xaxis=dict(title="TMB (mut/Mb)", color="#8b949e", gridcolor="#21262d"),
        yaxis=dict(title="Indel Fraction (%)", color="#8b949e", gridcolor="#21262d"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        margin=dict(l=0,r=0,t=10,b=0), height=300,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 3: Score component stacked bar ────────────────────────────────────

st.markdown('<div class="section-header">MSI Score Component Breakdown</div>', unsafe_allow_html=True)
fig_stack = go.Figure()
for col_name, color, label in [
    ("TMB_score",   "#58a6ff", "TMB signal"),
    ("indel_score", "#e3b341", "Indel signal"),
    ("MMR_score",   "#3fb950", "MMR/POLE signal"),
]:
    fig_stack.add_trace(go.Bar(
        name=label,
        x=msi_df["sample_id"].str[:16],
        y=msi_df[col_name],
        marker_color=color,
        hovertemplate=f"<b>{label}</b><br>Score: %{{y}}<extra></extra>",
    ))
fig_stack.update_layout(
    barmode="stack",
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
    font_family="IBM Plex Mono",
    xaxis=dict(showticklabels=False, title="Samples", color="#8b949e"),
    yaxis=dict(title="Score", color="#8b949e", gridcolor="#21262d"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0,r=0,t=40,b=0), height=280,
)
st.plotly_chart(fig_stack, use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Section 4: MMR gene summary ───────────────────────────────────────────────

st.markdown('<div class="section-header">MMR / POLE Gene Mutations Detected</div>', unsafe_allow_html=True)

mmr_rows = []
for gene in MMR_GENES:
    df_coding = mut_df[mut_df[variant_col].isin(TMB_COUNTED_VARIANTS)]
    n_hit = df_coding[df_coding[gene_col] == gene][sample_col].nunique()
    if n_hit > 0:
        mmr_rows.append({
            "Gene":            gene,
            "Role":            MMR_GENES[gene]["role"],
            "Samples Mutated": n_hit,
            "% of Cohort":     round(n_hit / len(mut_df[sample_col].unique()) * 100, 1),
            "Clinical Note":   MMR_GENES[gene]["note"],
        })

if mmr_rows:
    mmr_summary = pd.DataFrame(mmr_rows).sort_values("Samples Mutated", ascending=False)
    fig_mmr = go.Figure(go.Bar(
        x=mmr_summary["Gene"],
        y=mmr_summary["Samples Mutated"],
        marker_color=["#f0883e" if r == "Proofreading" else "#3fb950" for r in mmr_summary["Role"]],
        hovertemplate="<b>%{x}</b><br>%{y} samples mutated<extra></extra>",
    ))
    fig_mmr.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        font_family="IBM Plex Mono",
        xaxis=dict(title="Gene", color="#8b949e"),
        yaxis=dict(title="Samples Mutated", color="#8b949e", gridcolor="#21262d"),
        margin=dict(l=0,r=0,t=10,b=0), height=250,
    )
    st.plotly_chart(fig_mmr, use_container_width=True)
    st.markdown(
        "<div style='font-size:0.72rem;color:#8b949e'>"
        "<span style='color:#3fb950'>-- MMR gene</span> &nbsp;"
        "<span style='color:#f0883e'>-- Proofreading gene (POLE/POLD1)</span></div>",
        unsafe_allow_html=True
    )
    st.dataframe(mmr_summary.reset_index(drop=True), use_container_width=True)
else:
    st.info("No MMR or POLE gene mutations detected in this dataset.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 5: Per-sample report card ─────────────────────────────────────────

st.markdown('<div class="section-header">Per-Sample MSI Report</div>', unsafe_allow_html=True)

selected_sample = st.selectbox(
    "Select a sample to inspect",
    msi_df["sample_id"].tolist(),
    format_func=lambda s: (
        f"{s}  [{msi_df[msi_df['sample_id']==s]['MSI_status'].values[0]}"
        f"  --  score {msi_df[msi_df['sample_id']==s]['MSI_score'].values[0]}]"
    )
)

row        = msi_df[msi_df["sample_id"] == selected_sample].iloc[0]
status     = row["MSI_status"]
card_class = {"MSI-H":"msi-h-card","MSS":"msi-l-card","Indeterminate":"msi-l-card"}.get(status,"msi-l-card")
badge_class= {"MSI-H":"msi-badge-H","MSS":"msi-badge-L","Indeterminate":"msi-badge-IND"}.get(status,"msi-badge-L")
pole_text  = "Yes" if row["POLE_mutated"] else "No"

st.markdown(f"""
<div class="{card_class}">
    <div style='display:flex;align-items:center;gap:16px;margin-bottom:14px'>
        <span class="{badge_class}">{status}</span>
        <span style='font-family:IBM Plex Mono;font-size:1.1rem;color:#e6edf3'>{selected_sample}</span>
    </div>
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px'>
        <div>
            <div style='font-size:0.68rem;color:#8b949e;text-transform:uppercase'>MSI Score</div>
            <div style='font-family:IBM Plex Mono;font-size:1.4rem;color:#e6edf3'>{row["MSI_score"]}<span style='font-size:0.8rem;color:#8b949e'>/100</span></div>
        </div>
        <div>
            <div style='font-size:0.68rem;color:#8b949e;text-transform:uppercase'>TMB</div>
            <div style='font-family:IBM Plex Mono;font-size:1.4rem;color:#e6edf3'>{row["TMB"]}<span style='font-size:0.8rem;color:#8b949e'> mut/Mb</span></div>
        </div>
        <div>
            <div style='font-size:0.68rem;color:#8b949e;text-transform:uppercase'>Indel Fraction</div>
            <div style='font-family:IBM Plex Mono;font-size:1.4rem;color:#e6edf3'>{row["indel_fraction"]}<span style='font-size:0.8rem;color:#8b949e'>%</span></div>
        </div>
        <div>
            <div style='font-size:0.68rem;color:#8b949e;text-transform:uppercase'>Coding Mutations</div>
            <div style='font-family:IBM Plex Mono;font-size:1.4rem;color:#e6edf3'>{row["total_coding"]}</div>
        </div>
        <div>
            <div style='font-size:0.68rem;color:#8b949e;text-transform:uppercase'>MMR Genes Hit</div>
            <div style='font-family:IBM Plex Mono;font-size:1rem;color:#e6edf3'>{row["MMR_genes_hit"]}</div>
        </div>
        <div>
            <div style='font-size:0.68rem;color:#8b949e;text-transform:uppercase'>POLE Mutated</div>
            <div style='font-family:IBM Plex Mono;font-size:1rem;color:#e6edf3'>{pole_text}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=float(row["MSI_score"]),
    domain={"x":[0,1],"y":[0,1]},
    gauge=dict(
        axis=dict(range=[0,100], tickcolor="#8b949e", tickfont=dict(color="#8b949e")),
        bar=dict(color=color_map.get(status,"#58a6ff"), thickness=0.25),
        bgcolor="#161b22", bordercolor="#30363d",
        steps=[
            dict(range=[0, mss_thresh],              color="#0d1117"),
            dict(range=[mss_thresh, msi_h_thresh],   color="#1a1a0d"),
            dict(range=[msi_h_thresh, 100],          color="#0d1a0d"),
        ],
        threshold=dict(line=dict(color="#e6edf3",width=2), thickness=0.75, value=float(row["MSI_score"])),
    ),
    number=dict(font=dict(color="#e6edf3",family="IBM Plex Mono",size=36), suffix="/100"),
))
fig_gauge.update_layout(
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    font_color="#e6edf3", height=220,
    margin=dict(l=30,r=30,t=20,b=10),
)
st.plotly_chart(fig_gauge, use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Section 6: Immunotherapy recommendation ───────────────────────────────────

st.markdown('<div class="section-header">Immunotherapy Recommendation</div>', unsafe_allow_html=True)

if status == "MSI-H":
    st.markdown(
        "<div class='rec-card'>"
        "<b style='color:#3fb950'>MSI-H / dMMR Detected</b><br>"
        "<span style='color:#8b949e;font-size:0.85rem'>"
        "This tumour shows features consistent with high microsatellite instability (MSI-H) or "
        "mismatch repair deficiency (dMMR). MSI-H tumours have a high neoantigen burden and "
        "typically respond well to immune checkpoint inhibitors."
        "</span></div>",
        unsafe_allow_html=True
    )
    st.markdown("**Recommended checkpoint inhibitors:**")
    for drug in CHECKPOINT_INHIBITORS:
        st.markdown(
            f"<div class='rec-card'>"
            f"<b style='color:#e6edf3'>{drug['drug']}</b> &nbsp;"
            f"<span style='color:#58a6ff;font-size:0.8rem'>({drug['target']})</span><br>"
            f"<span style='color:#8b949e;font-size:0.82rem'>{drug['approval']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
elif status == "Indeterminate":
    st.markdown(
        "<div class='rec-card-warn'>"
        "<b style='color:#e3b341'>Indeterminate -- Further Testing Required</b><br>"
        "<span style='color:#8b949e;font-size:0.85rem'>"
        "MSI status cannot be confidently called from WES data alone. "
        "Recommend MMR immunohistochemistry (IHC) or PCR-based MSI testing."
        "</span></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='rec-card-warn'>"
        "<b style='color:#e3b341'>Suggested next steps:</b><br>"
        "<span style='color:#8b949e;font-size:0.85rem'>"
        "1. MMR IHC panel: MLH1, MSH2, MSH6, PMS2<br>"
        "2. PCR-based MSI testing (Bethesda panel)<br>"
        "3. Germline testing if Lynch syndrome suspected"
        "</span></div>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<div class='msi-l-card'>"
        "<b style='color:#58a6ff'>MSS -- Microsatellite Stable</b><br>"
        "<span style='color:#8b949e;font-size:0.85rem'>"
        "This tumour appears microsatellite stable. Single-agent checkpoint inhibitor therapy "
        "is unlikely to be effective. Standard chemotherapy or targeted therapy based on "
        "driver mutations is recommended."
        "</span></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='msi-l-card'>"
        "<b style='color:#58a6ff'>Alternative options to consider:</b><br>"
        "<span style='color:#8b949e;font-size:0.85rem'>"
        "- Targeted therapy based on driver mutations (EGFR, KRAS, BRAF, ALK)<br>"
        "- TMB-based pembrolizumab if TMB >= 10 mut/Mb (FDA pan-tumour approval)<br>"
        "- Clinical trial enrolment"
        "</span></div>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Full table ────────────────────────────────────────────────────────────────

with st.expander("Full MSI prediction table"):
    display_cols = ["sample_id","TMB","indel_fraction","MMR_genes_hit",
                    "POLE_mutated","MSI_score","MSI_status","immuno_rec"]
    st.dataframe(msi_df[display_cols].reset_index(drop=True), use_container_width=True)
    st.download_button("Download MSI predictions CSV", msi_df.to_csv(index=False).encode(), "msi_predictions.csv", "text/csv")

st.markdown(
    "<div style='margin-top:24px;padding:14px 18px;background:#161b22;"
    "border:1px solid #30363d;border-radius:8px;font-size:0.75rem;color:#8b949e'>"
    "<b>Clinical Disclaimer:</b> MSI predictions on this page are computational estimates "
    "derived from somatic mutation data (TMB, indel fraction, MMR gene status). "
    "They are not a substitute for clinical MSI/MMR testing. "
    "Gold-standard methods include PCR-based fragment analysis (Bethesda panel) and "
    "MMR immunohistochemistry (IHC). Always confirm with a certified clinical laboratory "
    "before making treatment decisions."
    "</div>",
    unsafe_allow_html=True
)