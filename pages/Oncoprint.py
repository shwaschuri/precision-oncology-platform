"""
Oncoprint Page -- Mutation Landscape Visualizer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from itertools import combinations
from scipy.stats import fisher_exact

st.set_page_config(page_title="Oncoprint", page_icon=None, layout="wide")

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
div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

TMB_COUNTED_VARIANTS = [
    "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
    "In_Frame_Del","In_Frame_Ins","Splice_Site","Translation_Start_Site","Nonstop_Mutation",
]

VARIANT_COLORS = {
    "Missense_Mutation":        "#3fb950",
    "Nonsense_Mutation":        "#f85149",
    "Frame_Shift_Del":          "#ff7b72",
    "Frame_Shift_Ins":          "#ffa657",
    "In_Frame_Del":             "#e3b341",
    "In_Frame_Ins":             "#d2a8ff",
    "Splice_Site":              "#58a6ff",
    "Translation_Start_Site":   "#bc8cff",
    "Nonstop_Mutation":         "#f0883e",
    "Silent":                   "#21262d",
    "Other":                    "#30363d",
}

@st.cache_data
def load_mutation_data():
    merged_csvs = list(Path("data").glob("*_merged.csv")) if Path("data").exists() else []
    if merged_csvs:
        return pd.read_csv(merged_csvs[0])
    return None

def get_cols(df):
    gene_col    = next((c for c in ["Hugo_Symbol","gene_hugoGeneSymbol"] if c in df.columns), None)
    sample_col  = next((c for c in ["Tumor_Sample_Barcode","sampleId"]   if c in df.columns), None)
    variant_col = next((c for c in ["Variant_Classification","mutationType"] if c in df.columns), None)
    return gene_col, sample_col, variant_col

def build_mutation_matrix(df, gene_col, sample_col, variant_col, top_n=30):
    df_coding = df[df[variant_col].isin(TMB_COUNTED_VARIANTS)] if variant_col else df
    top_genes = df_coding[gene_col].value_counts().head(top_n).index.tolist()
    df_top    = df_coding[df_coding[gene_col].isin(top_genes)]
    variant_matrix = (
        df_top.groupby([sample_col, gene_col])[variant_col]
        .first().unstack(fill_value="None")
        .reindex(columns=top_genes, fill_value="None")
    )
    binary_matrix = (variant_matrix != "None").astype(int)
    return binary_matrix, variant_matrix, top_genes

def mutation_frequency(df, gene_col, sample_col, variant_col, top_n=30):
    df_coding     = df[df[variant_col].isin(TMB_COUNTED_VARIANTS)] if variant_col else df
    total_samples = df[sample_col].nunique()
    freq = (
        df_coding.groupby(gene_col)[sample_col]
        .nunique().sort_values(ascending=False).head(top_n)
    )
    return (freq / total_samples * 100).round(1)

def co_occurrence_matrix(binary_matrix):
    genes  = binary_matrix.columns.tolist()
    n      = len(genes)
    co_mat = pd.DataFrame(np.zeros((n, n)), index=genes, columns=genes)
    for i, g1 in enumerate(genes):
        for j, g2 in enumerate(genes):
            if i != j:
                co_mat.loc[g1, g2] = (binary_matrix[g1] & binary_matrix[g2]).sum()
            else:
                co_mat.loc[g1, g2] = binary_matrix[g1].sum()
    return co_mat

def exclusivity_analysis(binary_matrix, min_freq=2):
    genes     = binary_matrix.columns.tolist()
    n_samples = len(binary_matrix)
    rows      = []
    for g1, g2 in combinations(genes, 2):
        both    = int((binary_matrix[g1] & binary_matrix[g2]).sum())
        only_g1 = int((binary_matrix[g1] & ~binary_matrix[g2].astype(bool)).sum())
        only_g2 = int((~binary_matrix[g1].astype(bool) & binary_matrix[g2]).sum())
        neither = n_samples - both - only_g1 - only_g2
        if binary_matrix[g1].sum() < min_freq or binary_matrix[g2].sum() < min_freq:
            continue
        _, p_val  = fisher_exact([[both, only_g1], [only_g2, neither]])
        expected  = (binary_matrix[g1].sum() * binary_matrix[g2].sum()) / n_samples
        rows.append({
            "Gene A":       g1,
            "Gene B":       g2,
            "Co-mutated":   both,
            "Only A":       only_g1,
            "Only B":       only_g2,
            "Neither":      neither,
            "p-value":      round(p_val, 4),
            "Relationship": "Co-occurrence" if both > expected else "Mutual Exclusivity"
        })
    return pd.DataFrame(rows).sort_values("p-value")

with st.sidebar:
    st.markdown("## Oncoprint")
    st.markdown("---")
    top_n    = st.slider("Top N genes to show", 5, 50, 20)
    min_freq = st.slider("Min mutation frequency (%)", 1, 30, 5)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#8b949e'>"
        "Each column = one sample.<br>"
        "Each row = one gene.<br>"
        "Sorted by mutation burden (left = most mutated).</div>",
        unsafe_allow_html=True
    )

st.markdown("# Oncoprint -- Mutation Landscape")
st.markdown(
    "<div style='color:#8b949e;font-size:0.9rem;margin-bottom:24px'>"
    "Gene-level mutation patterns | co-occurrence | mutual exclusivity"
    "</div>",
    unsafe_allow_html=True
)

mut_df = load_mutation_data()
if mut_df is None:
    st.warning("No mutation data found. Run python3 tmb_data_download.py first.")
    st.stop()

gene_col, sample_col, variant_col = get_cols(mut_df)
if not all([gene_col, sample_col, variant_col]):
    st.error(f"Missing required columns. Found: {list(mut_df.columns[:10])}")
    st.stop()

total_samples = mut_df[sample_col].nunique()
min_count     = max(1, int(min_freq / 100 * total_samples))
df_coding     = mut_df[mut_df[variant_col].isin(TMB_COUNTED_VARIANTS)]
freq_genes    = (
    df_coding.groupby(gene_col)[sample_col].nunique()
    .where(lambda x: x >= min_count).dropna()
    .sort_values(ascending=False).head(top_n).index.tolist()
)

if not freq_genes:
    st.warning("No genes meet the frequency threshold. Try lowering the filter.")
    st.stop()

binary_matrix, variant_matrix, top_genes = build_mutation_matrix(
    mut_df, gene_col, sample_col, variant_col, top_n=top_n
)
sample_order   = binary_matrix.sum(axis=1).sort_values(ascending=False).index
binary_matrix  = binary_matrix.loc[sample_order]
variant_matrix = variant_matrix.loc[sample_order]

# ── Section 1: Mutation frequency bar ─────────────────────────────────────────

st.markdown('<div class="section-header">Mutation Frequency per Gene</div>', unsafe_allow_html=True)
freq    = mutation_frequency(mut_df, gene_col, sample_col, variant_col, top_n=top_n)
freq_df = freq.reset_index()
freq_df.columns = ["gene", "frequency_pct"]

fig_freq = go.Figure(go.Bar(
    x=freq_df["gene"], y=freq_df["frequency_pct"],
    marker=dict(
        color=freq_df["frequency_pct"],
        colorscale=[[0,"#21262d"],[0.5,"#58a6ff"],[1,"#f85149"]],
        showscale=True,
        colorbar=dict(title="% Samples", tickfont=dict(color="#8b949e"), title_font=dict(color="#8b949e")),
    ),
    hovertemplate="<b>%{x}</b><br>Mutated in %{y:.1f}% of samples<extra></extra>",
))
fig_freq.update_layout(
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
    font_family="IBM Plex Mono",
    xaxis=dict(title="Gene", color="#8b949e", tickangle=-45),
    yaxis=dict(title="% Samples Mutated", color="#8b949e", gridcolor="#21262d"),
    margin=dict(l=0,r=0,t=10,b=80), height=320,
)
st.plotly_chart(fig_freq, use_container_width=True)

# ── Section 2: Oncoprint heatmap ──────────────────────────────────────────────

st.markdown('<div class="section-header">Oncoprint -- Mutation Heatmap</div>', unsafe_allow_html=True)

variant_type_order = list(VARIANT_COLORS.keys())

def encode_variant(v):
    if v == "None" or pd.isna(v):
        return 0
    try:
        return variant_type_order.index(v) + 1
    except ValueError:
        return len(variant_type_order)

z_matrix = variant_matrix.T.map(encode_variant)
n_types   = len(variant_type_order) + 2
colorscale = [[0,"#0d1117"],[1/n_types,"#0d1117"]]
for i, vtype in enumerate(variant_type_order):
    lo    = (i+1) / n_types
    hi    = (i+2) / n_types
    color = VARIANT_COLORS.get(vtype,"#30363d")
    colorscale += [[lo, color],[hi, color]]

sample_labels = [s[:16]+"..." if len(str(s)) > 16 else str(s) for s in z_matrix.columns]

fig_onco = go.Figure(go.Heatmap(
    z=z_matrix.values, x=sample_labels, y=z_matrix.index.tolist(),
    colorscale=colorscale, showscale=False,
    hovertemplate="Gene: %{y}<br>Sample: %{x}<extra></extra>",
    xgap=1, ygap=1,
))
fig_onco.update_layout(
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
    font_family="IBM Plex Mono",
    xaxis=dict(showticklabels=False, title="Samples (sorted by mutation burden)", color="#8b949e"),
    yaxis=dict(title="", color="#8b949e", tickfont=dict(size=11)),
    margin=dict(l=120,r=0,t=10,b=40),
    height=max(300, top_n * 22),
)
st.plotly_chart(fig_onco, use_container_width=True)

legend_html = " &nbsp; ".join(
    f"<span style='color:{c};font-size:0.72rem'>-- {v.replace('_',' ')}</span>"
    for v, c in VARIANT_COLORS.items() if v not in ("Silent","Other")
)
st.markdown(legend_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Section 3: Co-occurrence matrix ───────────────────────────────────────────

st.markdown('<div class="section-header">Mutation Co-occurrence Matrix</div>', unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:0.78rem;color:#8b949e;margin-bottom:12px'>"
    "Number of samples where both genes are mutated simultaneously. "
    "Diagonal = total samples with that gene mutated.</div>",
    unsafe_allow_html=True
)

bm_cols = [g for g in top_genes if g in binary_matrix.columns]
co_mat  = co_occurrence_matrix(binary_matrix[bm_cols])

fig_co = go.Figure(go.Heatmap(
    z=co_mat.values, x=co_mat.columns.tolist(), y=co_mat.index.tolist(),
    colorscale=[[0,"#0d1117"],[0.01,"#161b22"],[0.5,"#1f6feb"],[1,"#58a6ff"]],
    hovertemplate="<b>%{y} + %{x}</b><br>Co-mutated in %{z} samples<extra></extra>",
    text=co_mat.values.astype(int), texttemplate="%{text}",
    textfont=dict(size=9, color="#e6edf3"),
))
fig_co.update_layout(
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
    font_family="IBM Plex Mono",
    xaxis=dict(tickangle=-45, color="#8b949e", tickfont=dict(size=10)),
    yaxis=dict(color="#8b949e", tickfont=dict(size=10)),
    margin=dict(l=10,r=10,t=10,b=80), height=500,
)
st.plotly_chart(fig_co, use_container_width=True)

# ── Section 4: Exclusivity table ──────────────────────────────────────────────

st.markdown('<div class="section-header">Co-occurrence and Mutual Exclusivity (Fisher Exact Test)</div>', unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:0.78rem;color:#8b949e;margin-bottom:12px'>"
    "Statistically significant gene pairs (p &lt; 0.05).</div>",
    unsafe_allow_html=True
)

with st.spinner("Running Fisher exact test on all gene pairs..."):
    excl_df = exclusivity_analysis(binary_matrix[bm_cols], min_freq=2)

sig_df = excl_df[excl_df["p-value"] < 0.05].copy()

if sig_df.empty:
    st.info("No statistically significant pairs found (p < 0.05). Try increasing top N genes or lowering min frequency.")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Co-occurring Pairs**")
        co_pairs = sig_df[sig_df["Relationship"] == "Co-occurrence"].head(15)
        if co_pairs.empty:
            st.info("No significant co-occurring pairs.")
        else:
            st.dataframe(co_pairs[["Gene A","Gene B","Co-mutated","p-value"]].reset_index(drop=True), use_container_width=True, height=300)
    with col_b:
        st.markdown("**Mutually Exclusive Pairs**")
        excl_pairs = sig_df[sig_df["Relationship"] == "Mutual Exclusivity"].head(15)
        if excl_pairs.empty:
            st.info("No significant mutually exclusive pairs.")
        else:
            st.dataframe(excl_pairs[["Gene A","Gene B","Co-mutated","Only A","Only B","p-value"]].reset_index(drop=True), use_container_width=True, height=300)

    st.markdown('<div class="section-header" style="margin-top:24px">Significance vs Co-mutation Frequency</div>', unsafe_allow_html=True)
    plot_excl = excl_df.head(60).copy()
    plot_excl["-log10(p)"] = (-np.log10(plot_excl["p-value"] + 1e-10)).round(2)
    plot_excl["pair"]      = plot_excl["Gene A"] + " / " + plot_excl["Gene B"]
    plot_excl["color"]     = plot_excl["Relationship"].map({"Co-occurrence":"#3fb950","Mutual Exclusivity":"#f85149"})

    fig_scatter = go.Figure(go.Scatter(
        x=plot_excl["Co-mutated"], y=plot_excl["-log10(p)"],
        mode="markers+text", text=plot_excl["pair"],
        textposition="top center", textfont=dict(size=8, color="#8b949e"),
        marker=dict(size=10, color=plot_excl["color"], line=dict(color="#30363d",width=1)),
        hovertemplate="<b>%{text}</b><br>Co-mutated: %{x}<br>-log10(p): %{y:.2f}<extra></extra>",
    ))
    fig_scatter.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#e3b341",
                          annotation_text="p=0.05", annotation_font_color="#e3b341")
    fig_scatter.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        font_family="IBM Plex Mono",
        xaxis=dict(title="Co-mutated Samples", color="#8b949e", gridcolor="#21262d"),
        yaxis=dict(title="-log10(p-value)", color="#8b949e", gridcolor="#21262d"),
        margin=dict(l=0,r=0,t=10,b=0), height=380,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown(
        "<div style='font-size:0.72rem;color:#8b949e'>"
        "<span style='color:#3fb950'>-- Co-occurrence</span> &nbsp;"
        "<span style='color:#f85149'>-- Mutual Exclusivity</span></div>",
        unsafe_allow_html=True
    )

with st.expander("Full gene pair table"):
    st.dataframe(excl_df.reset_index(drop=True), use_container_width=True)
    st.download_button("Download gene pair CSV", excl_df.to_csv(index=False).encode(), "gene_pairs.csv", "text/csv")