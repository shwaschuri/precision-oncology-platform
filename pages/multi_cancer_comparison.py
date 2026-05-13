"""
Multi-Cancer Comparison Page
Compare mutation landscapes across TCGA cancer types
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import gzip
from pathlib import Path
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Multi-Cancer Comparison", page_icon=None, layout="wide")

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
.cancer-chip {
    display: inline-block; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; font-weight: 700; padding: 3px 12px;
    border-radius: 12px; margin-right: 6px; margin-bottom: 4px;
}
.stat-card {
    background:#161b22; border:1px solid #30363d; border-radius:8px;
    padding:14px 18px; text-align:center;
}
.stat-value { font-family:'IBM Plex Mono',monospace; font-size:1.4rem; font-weight:600; color:#58a6ff; }
.stat-label { font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }
div[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

CANCER_TYPES = {
    "TCGA-LUAD": {"name": "Lung Adenocarcinoma",        "color": "#58a6ff", "short": "LUAD"},
    "TCGA-BRCA": {"name": "Breast Cancer",              "color": "#f0883e", "short": "BRCA"},
    "TCGA-SKCM": {"name": "Skin Melanoma",              "color": "#3fb950", "short": "SKCM"},
    "TCGA-BLCA": {"name": "Bladder Cancer",             "color": "#e3b341", "short": "BLCA"},
    "TCGA-COAD": {"name": "Colorectal Cancer",          "color": "#d2a8ff", "short": "COAD"},
    "TCGA-UCEC": {"name": "Uterine Corpus Endometrial", "color": "#f85149", "short": "UCEC"},
}

EXOME_SIZE_MB = 38.0

TMB_COUNTED = [
    "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
    "In_Frame_Del","In_Frame_Ins","Splice_Site","Translation_Start_Site","Nonstop_Mutation",
]

DRIVER_GENES = [
    "TP53","KRAS","PIK3CA","PTEN","APC","BRAF","EGFR","CDH1","RB1",
    "BRCA1","BRCA2","MYC","CDKN2A","STK11","KEAP1","NFE2L2","SMAD4",
    "MLH1","MSH2","MSH6","POLE","VHL","IDH1","FBXW7","CTNNB1",
]

VARIANT_COLORS = {
    "Missense_Mutation":      "#3fb950",
    "Nonsense_Mutation":      "#f85149",
    "Frame_Shift_Del":        "#ff7b72",
    "Frame_Shift_Ins":        "#ffa657",
    "In_Frame_Del":           "#e3b341",
    "In_Frame_Ins":           "#d2a8ff",
    "Splice_Site":            "#58a6ff",
    "Translation_Start_Site": "#bc8cff",
    "Nonstop_Mutation":       "#f0883e",
}

GDC_API = "https://api.gdc.cancer.gov"

# ── GDC fetchers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_gdc_maf(cancer_type: str, max_files: int = 20) -> pd.DataFrame:
    filters = {
        "op": "and", "content": [
            {"op":"in","content":{"field":"cases.project.project_id","value":[cancer_type]}},
            {"op":"in","content":{"field":"data_type",  "value":["Masked Somatic Mutation"]}},
            {"op":"in","content":{"field":"data_format","value":["MAF"]}},
        ]
    }
    try:
        resp  = requests.get(
            f"{GDC_API}/files",
            params={"filters":json.dumps(filters),"fields":"file_id,file_name",
                    "format":"JSON","size":max_files},
            timeout=10
        )
        files = resp.json()["data"]["hits"]
    except Exception:
        return pd.DataFrame()

    cache_dir = Path("data/multicancer")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dfs = []

    for finfo in files:
        fid   = finfo["file_id"]
        fname = finfo["file_name"]
        fpath = cache_dir / fname
        if not fpath.exists():
            try:
                r = requests.get(f"{GDC_API}/data/{fid}", stream=True, timeout=30)
                with open(fpath, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            except Exception:
                continue
        try:
            opener = gzip.open if fname.endswith(".gz") else open
            mode   = "rt"      if fname.endswith(".gz") else "r"
            with opener(fpath, mode) as f:
                lines = [l for l in f if not l.startswith("#")]
            df = pd.read_csv(
                StringIO("".join(lines)), sep="\t", low_memory=False,
                usecols=lambda c: c in [
                    "Hugo_Symbol","Variant_Classification","Variant_Type",
                    "Tumor_Sample_Barcode","HGVSp_Short","t_depth","t_alt_count",
                ]
            )
            df["cancer_type"] = cancer_type
            dfs.append(df)
        except Exception:
            continue

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_gdc_clinical(cancer_type: str) -> pd.DataFrame:
    filters = {"op":"in","content":{"field":"project.project_id","value":[cancer_type]}}
    fields  = ["submitter_id","demographic.vital_status",
               "demographic.days_to_death","diagnoses.days_to_last_follow_up"]
    try:
        resp = requests.get(
            f"{GDC_API}/cases",
            params={"filters":json.dumps(filters),"fields":",".join(fields),
                    "format":"JSON","size":500,"expand":"demographic,diagnoses"},
            timeout=10
        )
        hits = resp.json()["data"]["hits"]
    except Exception:
        return pd.DataFrame()

    rows = []
    for h in hits:
        demo  = h.get("demographic",{})
        diags = h.get("diagnoses",[{}])
        diag  = diags[0] if diags else {}
        vital = demo.get("vital_status","")
        dtd   = demo.get("days_to_death")
        dtlf  = diag.get("days_to_last_follow_up")
        dur   = dtd if vital == "Dead" and dtd else dtlf
        rows.append({
            "sample_id":       h["submitter_id"],
            "event":           1 if vital == "Dead" else 0,
            "duration_months": round(float(dur) / 30.44, 1) if dur else None,
            "cancer_type":     cancer_type,
        })
    df = pd.DataFrame(rows)
    return df.dropna(subset=["duration_months"]).query("duration_months > 0")

# ── Compute helpers ───────────────────────────────────────────────────────────

def compute_tmb(df: pd.DataFrame) -> pd.DataFrame:
    df_c = df[df["Variant_Classification"].isin(TMB_COUNTED)]
    tmb  = df_c.groupby(["cancer_type","Tumor_Sample_Barcode"]).size().reset_index(name="mut_count")
    tmb["TMB"] = (tmb["mut_count"] / EXOME_SIZE_MB).round(2)
    return tmb

def compute_gene_freq(df: pd.DataFrame, genes: list) -> pd.DataFrame:
    df_c     = df[df["Variant_Classification"].isin(TMB_COUNTED)]
    df_genes = df_c[df_c["Hugo_Symbol"].isin(genes)]
    rows = []
    for ct, grp in df_genes.groupby("cancer_type"):
        total = df[df["cancer_type"] == ct]["Tumor_Sample_Barcode"].nunique()
        for gene, ggrp in grp.groupby("Hugo_Symbol"):
            n = ggrp["Tumor_Sample_Barcode"].nunique()
            rows.append({
                "cancer_type": ct, "gene": gene,
                "n_mutated":   n,
                "freq_pct":    round(n / total * 100, 1) if total else 0,
            })
    return pd.DataFrame(rows)

def compute_variant_mix(df: pd.DataFrame) -> pd.DataFrame:
    df_c   = df[df["Variant_Classification"].isin(TMB_COUNTED)]
    counts = df_c.groupby(["cancer_type","Variant_Classification"]).size().reset_index(name="count")
    totals = counts.groupby("cancer_type")["count"].transform("sum")
    counts["pct"] = (counts["count"] / totals * 100).round(1)
    return counts

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Multi-Cancer Comparison")
    st.markdown("---")
    selected_cancers = st.multiselect(
        "Cancer types to compare",
        list(CANCER_TYPES.keys()),
        default=["TCGA-LUAD","TCGA-BRCA","TCGA-SKCM"],
        format_func=lambda k: f"{CANCER_TYPES[k]['short']} -- {CANCER_TYPES[k]['name']}"
    )
    files_per_cancer = st.slider("Samples per cancer type", 5, 50, 15)
    st.markdown("---")
    st.markdown("### Gene Filter")
    selected_genes = st.multiselect(
        "Genes to compare", DRIVER_GENES,
        default=["TP53","KRAS","PIK3CA","BRAF","EGFR","PTEN","APC","CDH1"]
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.68rem;color:#8b949e'>"
        "Data is cached after first download in data/multicancer/.<br>"
        "SKCM has the highest TMB -- great for contrast.</div>",
        unsafe_allow_html=True
    )

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# Multi-Cancer Comparison")
st.markdown(
    "<div style='color:#8b949e;font-size:0.9rem;margin-bottom:16px'>"
    "TMB | Gene frequencies | Variant landscape | Survival -- across TCGA cancer types"
    "</div>",
    unsafe_allow_html=True
)

def make_chip(c):
    color = CANCER_TYPES[c]["color"]
    short = CANCER_TYPES[c]["short"]
    return (
        f"<span class='cancer-chip' style='background:{color}20;"
        f"color:{color};border:1px solid {color}40'>{short}</span>"
    )

chips = " ".join(make_chip(c) for c in selected_cancers)
st.markdown(chips, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if len(selected_cancers) < 2:
    st.warning("Select at least 2 cancer types in the sidebar to compare.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────

all_muts     = []
all_clinical = []
load_status  = {}

def fetch_cancer(ct):
    df_mut  = fetch_gdc_maf(ct, max_files=files_per_cancer)
    df_clin = fetch_gdc_clinical(ct)
    return ct, df_mut, df_clin

prog      = st.progress(0, text="Fetching cancer data in parallel...")
completed = 0

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(fetch_cancer, ct): ct for ct in selected_cancers}
    for future in as_completed(futures):
        ct, df_mut, df_clin = future.result()
        completed += 1
        prog.progress(
            completed / len(selected_cancers),
            text=f"Loaded {CANCER_TYPES[ct]['short']} ({completed}/{len(selected_cancers)})"
        )
        if df_mut.empty:
            load_status[ct] = "No data"
        else:
            load_status[ct] = f"{df_mut['Tumor_Sample_Barcode'].nunique()} samples"
            all_muts.append(df_mut)
        if not df_clin.empty:
            all_clinical.append(df_clin)

prog.empty()

if not all_muts:
    st.error("Could not load any mutation data. Check your internet connection.")
    st.stop()

combined      = pd.concat(all_muts,     ignore_index=True)
combined_clin = pd.concat(all_clinical, ignore_index=True) if all_clinical else pd.DataFrame()

# Load status row
scols = st.columns(len(selected_cancers))
for col_w, ct in zip(scols, selected_cancers):
    info        = CANCER_TYPES[ct]
    border      = info["color"]
    short_label = info["short"]
    status_text = load_status.get(ct, "--")
    col_w.markdown(
        f"<div class='stat-card' style='border-color:{border}40'>"
        f"<div class='stat-value' style='color:{border};font-size:1rem'>{short_label}</div>"
        f"<div class='stat-label'>{status_text}</div></div>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Precompute
tmb_df    = compute_tmb(combined)
freq_df   = compute_gene_freq(combined, selected_genes) if selected_genes else pd.DataFrame()
var_df    = compute_variant_mix(combined)
color_map = {ct: CANCER_TYPES[ct]["color"] for ct in selected_cancers}

# ── Section 1: TMB distribution ───────────────────────────────────────────────

st.markdown('<div class="section-header">Tumor Mutational Burden Distribution</div>', unsafe_allow_html=True)

tab_violin, tab_box, tab_bar = st.tabs(["Violin","Box","Median Bar"])

with tab_violin:
    fig_v = go.Figure()
    for ct in selected_cancers:
        sub  = tmb_df[tmb_df["cancer_type"] == ct]["TMB"]
        info = CANCER_TYPES[ct]
        if sub.empty:
            continue
        r = int(info["color"][1:3], 16)
        g = int(info["color"][3:5], 16)
        b = int(info["color"][5:7], 16)
        fig_v.add_trace(go.Violin(
            y=sub, name=info["short"], box_visible=True, meanline_visible=True,
            fillcolor=f"rgba({r},{g},{b},0.19)",
            line_color=info["color"],
            hovertemplate=f"<b>{info['short']}</b><br>TMB: %{{y:.1f}}<extra></extra>",
        ))
    fig_v.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        font_family="IBM Plex Mono", violingap=0.3, violinmode="overlay",
        yaxis=dict(title="TMB (mut/Mb)", color="#8b949e", gridcolor="#21262d"),
        xaxis=dict(color="#8b949e"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        margin=dict(l=0,r=0,t=10,b=0), height=380,
    )
    st.plotly_chart(fig_v, use_container_width=True)

with tab_box:
    fig_b = go.Figure()
    for ct in selected_cancers:
        sub  = tmb_df[tmb_df["cancer_type"] == ct]["TMB"]
        info = CANCER_TYPES[ct]
        if sub.empty:
            continue
        fig_b.add_trace(go.Box(
            y=sub, name=info["short"], marker_color=info["color"], boxmean=True,
            hovertemplate=f"<b>{info['short']}</b><br>TMB: %{{y:.1f}}<extra></extra>",
        ))
    fig_b.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        font_family="IBM Plex Mono",
        yaxis=dict(title="TMB (mut/Mb)", color="#8b949e", gridcolor="#21262d"),
        xaxis=dict(color="#8b949e"),
        margin=dict(l=0,r=0,t=10,b=0), height=380,
    )
    st.plotly_chart(fig_b, use_container_width=True)

with tab_bar:
    medians = []
    for ct in selected_cancers:
        sub  = tmb_df[tmb_df["cancer_type"] == ct]["TMB"]
        info = CANCER_TYPES[ct]
        if sub.empty:
            continue
        medians.append({
            "cancer": info["short"], "median": round(sub.median(),2),
            "mean":   round(sub.mean(),2),  "p90": round(sub.quantile(0.9),2),
            "color":  info["color"],         "n":   len(sub),
        })
    med_df = pd.DataFrame(medians).sort_values("median", ascending=False)
    fig_med = go.Figure()
    fig_med.add_trace(go.Bar(
        name="Median TMB", x=med_df["cancer"], y=med_df["median"],
        marker_color=med_df["color"],
        error_y=dict(type="data", array=(med_df["p90"]-med_df["median"]).tolist(),
                     color="#8b949e", thickness=1.5, width=6),
        hovertemplate="<b>%{x}</b><br>Median: %{y:.2f}<br>n=%{customdata}<extra></extra>",
        customdata=med_df["n"],
    ))
    fig_med.add_trace(go.Scatter(
        name="Mean TMB", x=med_df["cancer"], y=med_df["mean"],
        mode="markers",
        marker=dict(symbol="diamond", size=10, color="#e6edf3", line=dict(color="#0d1117",width=1)),
    ))
    fig_med.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        font_family="IBM Plex Mono",
        yaxis=dict(title="TMB (mut/Mb)", color="#8b949e", gridcolor="#21262d"),
        xaxis=dict(title="Cancer Type", color="#8b949e"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        margin=dict(l=0,r=0,t=10,b=0), height=340,
    )
    st.plotly_chart(fig_med, use_container_width=True)
    st.markdown(
        "<div style='font-size:0.72rem;color:#8b949e'>"
        "Error bars = 90th percentile. Diamond = mean.</div>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 2: Gene frequency ─────────────────────────────────────────────────

st.markdown('<div class="section-header">Driver Gene Mutation Frequency by Cancer Type</div>', unsafe_allow_html=True)

if freq_df.empty:
    st.info("No gene frequency data -- select genes in the sidebar.")
else:
    tab_heatmap, tab_grouped, tab_bubble = st.tabs(["Heatmap","Grouped Bar","Bubble"])

    with tab_heatmap:
        pivot = freq_df.pivot_table(index="gene", columns="cancer_type", values="freq_pct", fill_value=0)
        pivot.columns = [CANCER_TYPES.get(c,{}).get("short",c) for c in pivot.columns]
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[[0,"#0d1117"],[0.1,"#161b22"],[0.5,"#1f6feb"],[1,"#f85149"]],
            text=pivot.values.round(1), texttemplate="%{text}%",
            textfont=dict(size=10, color="#e6edf3"),
            hovertemplate="<b>%{y}</b> in <b>%{x}</b><br>%{z:.1f}% of samples<extra></extra>",
            colorbar=dict(title="% Mutated", tickfont=dict(color="#8b949e"), title_font=dict(color="#8b949e")),
        ))
        fig_hm.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
            font_family="IBM Plex Mono",
            xaxis=dict(color="#8b949e", side="top"),
            yaxis=dict(color="#8b949e", tickfont=dict(size=11)),
            margin=dict(l=10,r=10,t=40,b=0),
            height=max(300, len(pivot) * 28),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab_grouped:
        fig_grp = go.Figure()
        for ct in selected_cancers:
            sub  = freq_df[freq_df["cancer_type"] == ct]
            info = CANCER_TYPES[ct]
            fig_grp.add_trace(go.Bar(
                name=info["short"], x=sub["gene"], y=sub["freq_pct"],
                marker_color=info["color"],
                hovertemplate=f"<b>%{{x}}</b> in {info['short']}<br>%{{y:.1f}}%<extra></extra>",
            ))
        fig_grp.update_layout(
            barmode="group",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
            font_family="IBM Plex Mono",
            xaxis=dict(title="Gene", color="#8b949e", tickangle=-40),
            yaxis=dict(title="% Samples Mutated", color="#8b949e", gridcolor="#21262d"),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
            margin=dict(l=0,r=0,t=10,b=80), height=380,
        )
        st.plotly_chart(fig_grp, use_container_width=True)

    with tab_bubble:
        fig_bub = go.Figure()
        for ct in selected_cancers:
            sub  = freq_df[freq_df["cancer_type"] == ct]
            info = CANCER_TYPES[ct]
            fig_bub.add_trace(go.Scatter(
                x=[info["short"]] * len(sub),
                y=sub["gene"],
                mode="markers",
                name=info["short"],
                marker=dict(
                    size=sub["freq_pct"].clip(lower=1) * 1.8,
                    color=info["color"], opacity=0.8,
                    line=dict(color="#0d1117",width=1),
                ),
                hovertemplate=f"<b>%{{y}}</b> in {info['short']}<br>%{{customdata:.1f}}% mutated<extra></extra>",
                customdata=sub["freq_pct"],
            ))
        fig_bub.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
            font_family="IBM Plex Mono",
            xaxis=dict(title="Cancer Type", color="#8b949e"),
            yaxis=dict(title="Gene", color="#8b949e", tickfont=dict(size=11)),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
            margin=dict(l=10,r=10,t=10,b=0),
            height=max(350, len(selected_genes) * 26),
        )
        st.plotly_chart(fig_bub, use_container_width=True)
        st.markdown(
            "<div style='font-size:0.72rem;color:#8b949e'>Bubble size = mutation frequency (%)</div>",
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 3: Variant mix ────────────────────────────────────────────────────

st.markdown('<div class="section-header">Variant Classification Mix by Cancer Type</div>', unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:0.75rem;color:#8b949e;margin-bottom:12px'>"
    "Proportion of coding mutation types. "
    "MSI-H cancers (UCEC, COAD) show elevated frameshift indels.</div>",
    unsafe_allow_html=True
)

var_pivot = var_df.pivot_table(index="cancer_type", columns="Variant_Classification", values="pct", fill_value=0)
var_pivot.index = [CANCER_TYPES.get(c,{}).get("short",c) for c in var_pivot.index]

fig_var = go.Figure()
for vtype in var_pivot.columns:
    if vtype not in VARIANT_COLORS:
        continue
    fig_var.add_trace(go.Bar(
        name=vtype.replace("_"," "),
        x=var_pivot.index,
        y=var_pivot[vtype],
        marker_color=VARIANT_COLORS[vtype],
        hovertemplate=f"<b>%{{x}}</b><br>{vtype.replace('_',' ')}: %{{y:.1f}}%<extra></extra>",
    ))
fig_var.update_layout(
    barmode="stack",
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
    font_family="IBM Plex Mono",
    xaxis=dict(title="Cancer Type", color="#8b949e"),
    yaxis=dict(title="% of Coding Mutations", color="#8b949e", gridcolor="#21262d"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1, font=dict(size=10)),
    margin=dict(l=0,r=0,t=10,b=0), height=380,
)
st.plotly_chart(fig_var, use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Section 4: Summary table ──────────────────────────────────────────────────

st.markdown('<div class="section-header">Mutation Burden Summary Table</div>', unsafe_allow_html=True)

summary_rows = []
for ct in selected_cancers:
    sub  = tmb_df[tmb_df["cancer_type"] == ct]["TMB"]
    info = CANCER_TYPES[ct]
    if sub.empty:
        continue
    summary_rows.append({
        "Cancer Type":      info["name"],
        "Short":            info["short"],
        "Samples":          len(sub),
        "Median TMB":       round(sub.median(), 2),
        "Mean TMB":         round(sub.mean(),   2),
        "Max TMB":          round(sub.max(),    2),
        "% TMB-High (>16)": round((sub > 16).sum() / len(sub) * 100, 1),
        "% TMB-Med (6-16)": round(((sub >= 6) & (sub <= 16)).sum() / len(sub) * 100, 1),
        "% TMB-Low (<6)":   round((sub < 6).sum() / len(sub) * 100, 1),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("Median TMB", ascending=False)

def colour_tmb(val):
    if val > 16:
        return "color:#f85149;font-weight:700"
    if val > 6:
        return "color:#e3b341;font-weight:600"
    return "color:#3fb950"

styled = summary_df.style.map(colour_tmb, subset=["Median TMB"])
st.dataframe(styled, use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Section 5: Survival comparison ───────────────────────────────────────────

st.markdown('<div class="section-header">Overall Survival Comparison</div>', unsafe_allow_html=True)

if combined_clin.empty:
    st.info("No clinical/survival data could be fetched.")
else:
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test

        fig_km = go.Figure()
        km_rows = []

        for ct in selected_cancers:
            sub  = combined_clin[combined_clin["cancer_type"] == ct].dropna(subset=["duration_months"])
            info = CANCER_TYPES[ct]
            if len(sub) < 5:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub["duration_months"], sub["event"], label=f"{info['short']} (n={len(sub)})")
            t    = kmf.timeline
            sf   = kmf.survival_function_.iloc[:, 0].values
            ci_u = kmf.confidence_interval_.iloc[:, 1].values
            ci_l = kmf.confidence_interval_.iloc[:, 0].values
            color = info["color"]
            r = int(color[1:3],16)
            g = int(color[3:5],16)
            b = int(color[5:7],16)
            fig_km.add_trace(go.Scatter(
                x=t, y=sf, mode="lines", name=f"{info['short']} (n={len(sub)})",
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{info['short']}</b><br>Time: %{{x:.0f}} mo<br>S(t): %{{y:.3f}}<extra></extra>",
            ))
            fig_km.add_trace(go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([ci_u, ci_l[::-1]]),
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))
            km_rows.append({
                "cancer":    info["short"],
                "n":         len(sub),
                "median_os": round(kmf.median_survival_time_, 1),
            })

        ct_map   = {ct: CANCER_TYPES[ct]["short"] for ct in selected_cancers}
        clin_sel = combined_clin[combined_clin["cancer_type"].isin(selected_cancers)].dropna(subset=["duration_months"]).copy()
        clin_sel["short"] = clin_sel["cancer_type"].map(ct_map)
        mlr   = multivariate_logrank_test(
            clin_sel["duration_months"], clin_sel["short"], event_observed=clin_sel["event"]
        )
        p_val = mlr.p_value
        sig   = "significant" if p_val < 0.05 else "not significant"
        p_color = "#3fb950" if p_val < 0.05 else "#8b949e"

        fig_km.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
            font_family="IBM Plex Mono",
            xaxis=dict(title="Time (months)", color="#8b949e", gridcolor="#21262d"),
            yaxis=dict(title="Survival Probability", color="#8b949e", gridcolor="#21262d",
                       range=[0,1.05], tickformat=".0%"),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
            margin=dict(l=0,r=0,t=10,b=0), height=420,
        )
        st.plotly_chart(fig_km, use_container_width=True)
        st.markdown(
            f"<div style='font-size:0.82rem;margin-top:-8px'>"
            f"Multivariate log-rank p = "
            f"<b style='color:{p_color}'>{p_val:.4f} ({sig})</b></div>",
            unsafe_allow_html=True
        )

        km_df = pd.DataFrame(km_rows).sort_values("median_os", ascending=False)
        km_df.columns = ["Cancer","Samples","Median OS (months)"]

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.dataframe(km_df.reset_index(drop=True), use_container_width=True)
        with col_b:
            short_to_ct = {v["short"]: k for k, v in CANCER_TYPES.items()}
            bar_colors  = [CANCER_TYPES.get(short_to_ct.get(c,""),{}).get("color","#58a6ff") for c in km_df["Cancer"]]
            fig_mos = go.Figure(go.Bar(
                x=km_df["Cancer"], y=km_df["Median OS (months)"],
                marker_color=bar_colors,
                hovertemplate="<b>%{x}</b><br>Median OS: %{y:.1f} months<extra></extra>",
            ))
            fig_mos.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                font_family="IBM Plex Mono",
                xaxis=dict(title="Cancer Type", color="#8b949e"),
                yaxis=dict(title="Median OS (months)", color="#8b949e", gridcolor="#21262d"),
                margin=dict(l=0,r=0,t=10,b=0), height=260,
            )
            st.plotly_chart(fig_mos, use_container_width=True)

    except ImportError:
        st.info("Install lifelines for survival comparisons: pip3 install lifelines")

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 6: Per-gene cross-cancer ─────────────────────────────────────────

st.markdown('<div class="section-header">Per-Gene Cross-Cancer Deep Dive</div>', unsafe_allow_html=True)

if not freq_df.empty:
    focus_gene = st.selectbox("Select gene", sorted(freq_df["gene"].unique()))
    gene_sub   = freq_df[freq_df["gene"] == focus_gene].copy()
    gene_sub["short"] = gene_sub["cancer_type"].map(lambda c: CANCER_TYPES.get(c,{}).get("short",c))
    gene_sub["color"] = gene_sub["cancer_type"].map(lambda c: CANCER_TYPES.get(c,{}).get("color","#58a6ff"))
    gene_sub = gene_sub.sort_values("freq_pct", ascending=True)

    fig_gene = go.Figure(go.Bar(
        x=gene_sub["freq_pct"], y=gene_sub["short"],
        orientation="h", marker_color=gene_sub["color"],
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% of samples mutated<extra></extra>",
        text=gene_sub["freq_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside", textfont=dict(color="#8b949e", size=11),
    ))
    fig_gene.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        font_family="IBM Plex Mono",
        title=dict(text=f"{focus_gene} mutation frequency across cancer types",
                   font=dict(size=13, color="#8b949e")),
        xaxis=dict(title="% Samples Mutated", color="#8b949e", gridcolor="#21262d"),
        yaxis=dict(color="#8b949e", tickfont=dict(size=12)),
        margin=dict(l=0,r=60,t=40,b=0),
        height=max(260, len(gene_sub) * 48),
    )
    st.plotly_chart(fig_gene, use_container_width=True)

# ── Downloads ─────────────────────────────────────────────────────────────────

with st.expander("Download comparison data"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download TMB data CSV",
            tmb_df.to_csv(index=False).encode(),
            "multicancer_tmb.csv","text/csv"
        )
    with c2:
        if not freq_df.empty:
            st.download_button(
                "Download gene frequency CSV",
                freq_df.to_csv(index=False).encode(),
                "multicancer_gene_freq.csv","text/csv"
            )
    with c3:
        st.download_button(
            "Download summary table CSV",
            summary_df.to_csv(index=False).encode(),
            "multicancer_summary.csv","text/csv"
        )