"""
TMB Dashboard -- Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import gzip
from io import StringIO

st.set_page_config(
    page_title="TMB Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background-color: #0d1117; color: #e6edf3; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
.metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 20px 24px; text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace; font-size: 2rem;
    font-weight: 600; color: #58a6ff;
}
.metric-label {
    font-size: 0.78rem; color: #8b949e; text-transform: uppercase;
    letter-spacing: 0.08em; margin-top: 4px;
}
.tmb-high   { color: #f85149 !important; }
.tmb-medium { color: #e3b341 !important; }
.tmb-low    { color: #3fb950 !important; }
.section-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    color: #58a6ff; text-transform: uppercase; letter-spacing: 0.15em;
    border-bottom: 1px solid #21262d; padding-bottom: 8px; margin-bottom: 16px;
}
div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

EXOME_SIZE_MB = 38.0

TMB_COUNTED_VARIANTS = [
    "Missense_Mutation", "Nonsense_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins",
    "In_Frame_Del", "In_Frame_Ins",
    "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation",
]

ACTIONABLE_GENES = {
    "EGFR":   {"drug": "Erlotinib / Osimertinib", "therapy": "Targeted"},
    "KRAS":   {"drug": "Sotorasib (G12C)",         "therapy": "Targeted"},
    "BRAF":   {"drug": "Vemurafenib / Dabrafenib", "therapy": "Targeted"},
    "ALK":    {"drug": "Crizotinib / Alectinib",   "therapy": "Targeted"},
    "MET":    {"drug": "Capmatinib",               "therapy": "Targeted"},
    "RET":    {"drug": "Selpercatinib",            "therapy": "Targeted"},
    "ERBB2":  {"drug": "Trastuzumab",              "therapy": "Targeted"},
    "PIK3CA": {"drug": "Alpelisib",                "therapy": "Targeted"},
    "TP53":   {"drug": "N/A (tumour suppressor)",  "therapy": "Prognostic"},
    "STK11":  {"drug": "N/A",                      "therapy": "Prognostic"},
    "KEAP1":  {"drug": "N/A",                      "therapy": "Prognostic"},
    "BRCA1":  {"drug": "Olaparib (PARP inhibitor)","therapy": "Targeted"},
    "BRCA2":  {"drug": "Olaparib (PARP inhibitor)","therapy": "Targeted"},
    "PTEN":   {"drug": "N/A",                      "therapy": "Prognostic"},
    "RB1":    {"drug": "N/A",                      "therapy": "Prognostic"},
}

COLORS = {
    "High (>16)":    "#f85149",
    "Medium (6-16)": "#e3b341",
    "Low (<6)":      "#3fb950",
}

def parse_maf(file_content: str) -> pd.DataFrame:
    lines = [l for l in file_content.splitlines() if not l.startswith("#")]
    return pd.read_csv(StringIO("\n".join(lines)), sep="\t", low_memory=False)

def calculate_tmb(df: pd.DataFrame) -> pd.DataFrame:
    sample_col  = next((c for c in ["Tumor_Sample_Barcode", "sampleId"] if c in df.columns), None)
    variant_col = next((c for c in ["Variant_Classification", "mutationType"] if c in df.columns), None)
    if not sample_col:
        return pd.DataFrame()
    df_f = df[df[variant_col].isin(TMB_COUNTED_VARIANTS)] if variant_col else df
    tmb_df = df_f.groupby(sample_col).size().reset_index(name="mutation_count")
    tmb_df.rename(columns={sample_col: "sample_id"}, inplace=True)
    tmb_df["TMB"] = (tmb_df["mutation_count"] / EXOME_SIZE_MB).round(2)
    tmb_df["TMB_category"] = pd.cut(
        tmb_df["TMB"],
        bins=[0, 6, 16, float("inf")],
        labels=["Low (<6)", "Medium (6-16)", "High (>16)"]
    )
    return tmb_df.sort_values("TMB", ascending=False).reset_index(drop=True)

def get_top_genes(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    gene_col    = next((c for c in ["Hugo_Symbol", "gene_hugoGeneSymbol", "hugoGeneSymbol"] if c in df.columns), None)
    variant_col = next((c for c in ["Variant_Classification", "mutationType"] if c in df.columns), None)
    if not gene_col:
        return pd.DataFrame()
    df_f = df[df[variant_col].isin(TMB_COUNTED_VARIANTS)] if variant_col else df
    return df_f[gene_col].value_counts().head(n).reset_index().rename(
        columns={"index": "gene", gene_col: "gene", "count": "count", 0: "count"}
    )

def get_actionable_mutations(df: pd.DataFrame) -> pd.DataFrame:
    gene_col    = next((c for c in ["Hugo_Symbol", "gene_hugoGeneSymbol"] if c in df.columns), None)
    sample_col  = next((c for c in ["Tumor_Sample_Barcode", "sampleId"] if c in df.columns), None)
    variant_col = next((c for c in ["Variant_Classification", "mutationType"] if c in df.columns), None)
    hgvsp_col   = next((c for c in ["HGVSp_Short", "proteinChange"] if c in df.columns), None)
    if not gene_col:
        return pd.DataFrame()
    hits = df[df[gene_col].isin(ACTIONABLE_GENES.keys())].copy()
    if hits.empty:
        return pd.DataFrame()
    rows = []
    for _, row in hits.iterrows():
        gene = row[gene_col]
        info = ACTIONABLE_GENES.get(gene, {})
        rows.append({
            "Gene":          gene,
            "Sample":        row.get(sample_col, "--") if sample_col else "--",
            "Variant":       row.get(variant_col, "--") if variant_col else "--",
            "Protein Change":row.get(hgvsp_col,   "--") if hgvsp_col  else "--",
            "Drug":          info.get("drug",    "--"),
            "Therapy Type":  info.get("therapy", "--"),
        })
    return pd.DataFrame(rows).drop_duplicates()

@st.cache_data
def load_default_data():
    tmb_path     = Path("data/tmb_scores.csv")
    merged_csvs  = list(Path("data").glob("*_merged.csv")) if Path("data").exists() else []
    tmb_df, mut_df = None, None
    if tmb_path.exists():
        tmb_df = pd.read_csv(tmb_path)
    if merged_csvs:
        mut_df = pd.read_csv(merged_csvs[0])
    return tmb_df, mut_df

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## TMB Dashboard")
    st.markdown("---")
    st.markdown("### Data Source")
    data_source = st.radio(
        "Choose input",
        ["Use downloaded TCGA data", "Upload MAF file", "Upload TMB CSV"],
        label_visibility="collapsed"
    )
    uploaded_file = None
    if data_source == "Upload MAF file":
        uploaded_file = st.file_uploader("Upload MAF / MAF.GZ", type=["maf", "gz", "txt"])
    elif data_source == "Upload TMB CSV":
        uploaded_file = st.file_uploader("Upload TMB scores CSV", type=["csv"])
    st.markdown("---")
    st.markdown("### Filters")
    tmb_min = st.slider("Min TMB score", 0.0, 100.0, 0.0, step=0.5)
    show_categories = st.multiselect(
        "TMB Categories",
        ["Low (<6)", "Medium (6-16)", "High (>16)"],
        default=["Low (<6)", "Medium (6-16)", "High (>16)"]
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#8b949e'>TMB thresholds based on FDA guidance.<br>"
        "Exome size: 38 Mb (WES standard).</div>",
        unsafe_allow_html=True
    )

# ── Load data ─────────────────────────────────────────────────────────────────

mut_df = None
tmb_df = None

if data_source == "Use downloaded TCGA data":
    tmb_df, mut_df = load_default_data()
    if tmb_df is None:
        st.warning("No downloaded data found. Run python3 tmb_data_download.py first, or upload a file.")
        st.stop()
elif data_source == "Upload MAF file" and uploaded_file:
    content = (
        gzip.decompress(uploaded_file.read()).decode("utf-8")
        if uploaded_file.name.endswith(".gz")
        else uploaded_file.read().decode("utf-8")
    )
    mut_df = parse_maf(content)
    tmb_df = calculate_tmb(mut_df)
elif data_source == "Upload TMB CSV" and uploaded_file:
    tmb_df = pd.read_csv(uploaded_file)
else:
    st.info("Select a data source from the sidebar to get started.")
    st.stop()

if tmb_df is None and mut_df is not None:
    tmb_df = calculate_tmb(mut_df)

if tmb_df is not None and not tmb_df.empty:
    tmb_df = tmb_df[tmb_df["TMB"] >= tmb_min]
    if "TMB_category" in tmb_df.columns:
        tmb_df = tmb_df[tmb_df["TMB_category"].astype(str).isin(show_categories)]

if tmb_df is None or tmb_df.empty:
    st.warning("No data after applying filters.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# Tumor Mutational Burden Dashboard")
st.markdown(
    "<div style='color:#8b949e;font-size:0.9rem;margin-bottom:24px'>"
    "Somatic mutation analysis · TCGA-LUAD · Exome-based TMB (mut/Mb)"
    "</div>",
    unsafe_allow_html=True
)

# ── Metrics ───────────────────────────────────────────────────────────────────

n_samples  = len(tmb_df)
median_tmb = tmb_df["TMB"].median()
cat_counts = tmb_df["TMB_category"].value_counts() if "TMB_category" in tmb_df.columns else {}
n_high     = int(cat_counts.get("High (>16)", 0))
n_medium   = int(cat_counts.get("Medium (6-16)", 0))
n_low      = int(cat_counts.get("Low (<6)", 0))

c1, c2, c3, c4, c5 = st.columns(5)
for col_w, val, label in zip(
    [c1, c2, c3, c4, c5],
    [n_samples, f"{median_tmb:.1f}", n_high, n_medium, n_low],
    ["Total Samples", "Median TMB", "TMB-High (>16)", "TMB-Medium (6-16)", "TMB-Low (<6)"]
):
    col_w.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-value'>{val}</div>"
        f"<div class='metric-label'>{label}</div></div>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── TMB bar + pie ─────────────────────────────────────────────────────────────

col_left, col_right = st.columns([3, 1])

with col_left:
    st.markdown('<div class="section-header">TMB Score per Sample</div>', unsafe_allow_html=True)
    plot_df = tmb_df.copy()
    plot_df["color"] = plot_df["TMB_category"].astype(str).map(COLORS).fillna("#58a6ff")
    fig_bar = go.Figure(go.Bar(
        x=plot_df["sample_id"].astype(str).str[:20],
        y=plot_df["TMB"],
        marker_color=plot_df["color"],
        hovertemplate="<b>%{customdata}</b><br>TMB: %{y:.2f} mut/Mb<extra></extra>",
        customdata=plot_df["sample_id"],
    ))
    fig_bar.add_hline(y=16, line_dash="dash", line_color="#f85149",
                      annotation_text="High (16)", annotation_font_color="#f85149")
    fig_bar.add_hline(y=6,  line_dash="dash", line_color="#e3b341",
                      annotation_text="Medium (6)", annotation_font_color="#e3b341")
    fig_bar.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        font_family="IBM Plex Mono",
        xaxis=dict(showticklabels=False, title="Samples", color="#8b949e"),
        yaxis=dict(title="TMB (mut/Mb)", color="#8b949e", gridcolor="#21262d"),
        margin=dict(l=0, r=0, t=10, b=0), height=320,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.markdown('<div class="section-header">Category Breakdown</div>', unsafe_allow_html=True)
    if "TMB_category" in tmb_df.columns:
        pie_data = tmb_df["TMB_category"].value_counts().reset_index()
        pie_data.columns = ["category", "count"]
        fig_pie = go.Figure(go.Pie(
            labels=pie_data["category"],
            values=pie_data["count"],
            marker_colors=[COLORS.get(c, "#58a6ff") for c in pie_data["category"]],
            hole=0.55, textinfo="percent",
            hovertemplate="%{label}<br>%{value} samples<extra></extra>",
        ))
        fig_pie.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
            font_family="IBM Plex Mono",
            legend=dict(font_size=11),
            margin=dict(l=0, r=0, t=10, b=0), height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ── Top genes + distribution ──────────────────────────────────────────────────

col_genes, col_dist = st.columns(2)

with col_genes:
    st.markdown('<div class="section-header">Top Mutated Genes</div>', unsafe_allow_html=True)
    if mut_df is not None:
        gene_df = get_top_genes(mut_df, n=20)
        if not gene_df.empty:
            cols      = list(gene_df.columns)
            gene_col  = cols[0]
            count_col = cols[1]
            gene_df["is_actionable"] = gene_df[gene_col].isin(ACTIONABLE_GENES.keys())
            gene_df["color"] = gene_df["is_actionable"].map({True: "#f0883e", False: "#58a6ff"})
            fig_genes = go.Figure(go.Bar(
                x=gene_df[count_col], y=gene_df[gene_col],
                orientation="h", marker_color=gene_df["color"],
                hovertemplate="<b>%{y}</b><br>%{x} mutations<extra></extra>",
            ))
            fig_genes.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                font_family="IBM Plex Mono",
                xaxis=dict(title="Mutation Count", color="#8b949e", gridcolor="#21262d"),
                yaxis=dict(autorange="reversed", color="#8b949e"),
                margin=dict(l=0, r=0, t=10, b=0), height=400,
            )
            st.plotly_chart(fig_genes, use_container_width=True)
            st.markdown(
                "<div style='font-size:0.72rem;color:#8b949e'>"
                "Orange = actionable gene | Blue = other</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("Upload a MAF file to see gene-level data.")

with col_dist:
    st.markdown('<div class="section-header">TMB Distribution</div>', unsafe_allow_html=True)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=tmb_df["TMB"], nbinsx=30, marker_color="#58a6ff", opacity=0.8,
        hovertemplate="TMB: %{x:.1f}<br>Count: %{y}<extra></extra>",
    ))
    fig_hist.add_vline(x=6,  line_dash="dash", line_color="#e3b341", annotation_text="6")
    fig_hist.add_vline(x=16, line_dash="dash", line_color="#f85149", annotation_text="16")
    fig_hist.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        font_family="IBM Plex Mono",
        xaxis=dict(title="TMB (mut/Mb)", color="#8b949e", gridcolor="#21262d"),
        yaxis=dict(title="Number of Samples", color="#8b949e", gridcolor="#21262d"),
        margin=dict(l=0, r=0, t=10, b=0), height=400,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Variant classification ────────────────────────────────────────────────────

if mut_df is not None:
    variant_col = next((c for c in ["Variant_Classification", "mutationType"] if c in mut_df.columns), None)
    if variant_col:
        st.markdown('<div class="section-header">Variant Classification Breakdown</div>', unsafe_allow_html=True)
        var_counts = mut_df[variant_col].value_counts().reset_index()
        var_counts.columns = ["variant_type", "count"]
        fig_var = px.bar(
            var_counts, x="variant_type", y="count",
            color="variant_type", color_discrete_sequence=px.colors.qualitative.Dark24,
        )
        fig_var.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
            font_family="IBM Plex Mono",
            xaxis=dict(title="", color="#8b949e", tickangle=-30),
            yaxis=dict(title="Count", color="#8b949e", gridcolor="#21262d"),
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=60), height=300,
        )
        st.plotly_chart(fig_var, use_container_width=True)

# ── Actionable mutations ──────────────────────────────────────────────────────

if mut_df is not None:
    st.markdown('<div class="section-header">Actionable Mutations Detected</div>', unsafe_allow_html=True)
    actionable_df = get_actionable_mutations(mut_df)
    if not actionable_df.empty:
        def highlight_therapy(val):
            if val == "Targeted":
                return "color: #f0883e; font-weight: 600"
            return "color: #8b949e"
        styled = actionable_df.style.map(highlight_therapy, subset=["Therapy Type"])
        st.dataframe(styled, use_container_width=True, height=300)
        st.markdown(
            f"<div style='font-size:0.75rem;color:#8b949e'>"
            f"{len(actionable_df)} actionable mutation(s) found across "
            f"{actionable_df['Sample'].nunique()} sample(s)</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("No actionable mutations found in the loaded dataset.")

# ── Raw table ─────────────────────────────────────────────────────────────────

with st.expander("View full TMB scores table"):
    st.dataframe(tmb_df, use_container_width=True)
    csv = tmb_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download TMB scores CSV", csv, "tmb_scores.csv", "text/csv")