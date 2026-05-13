"""
Biomarker Explorer -- Live OncoKB + ClinVar + CIViC integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Biomarker Explorer", page_icon=None, layout="wide")

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
.ev-badge {
    display: inline-block; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; font-weight: 700; padding: 2px 10px;
    border-radius: 12px; letter-spacing: 0.06em;
}
.ev-1  { background:#3fb950; color:#0d1117; }
.ev-2  { background:#58a6ff; color:#0d1117; }
.ev-3  { background:#e3b341; color:#0d1117; }
.ev-4  { background:#f0883e; color:#0d1117; }
.ev-R  { background:#f85149; color:#ffffff; }
.ev-NA { background:#30363d; color:#8b949e; }
.biomarker-card {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:18px 22px; margin-bottom:12px;
}
.source-tag {
    display:inline-block; font-size:0.65rem; font-family:'IBM Plex Mono',monospace;
    padding:1px 8px; border-radius:8px; margin-right:4px; border:1px solid #30363d; color:#8b949e;
}
.oncokb-tag  { border-color:#f0883e!important; color:#f0883e!important; }
.clinvar-tag { border-color:#58a6ff!important; color:#58a6ff!important; }
.civic-tag   { border-color:#3fb950!important; color:#3fb950!important; }
div[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

ONCOKB_LEVEL_INFO = {
    "LEVEL_1":          {"label":"1",  "css":"ev-1", "meaning":"FDA-approved biomarker and therapy in this tumour type"},
    "LEVEL_2":          {"label":"2",  "css":"ev-2", "meaning":"Standard care biomarker; consensus guidelines"},
    "LEVEL_3A":         {"label":"3A", "css":"ev-3", "meaning":"Compelling clinical evidence (early trials)"},
    "LEVEL_3B":         {"label":"3B", "css":"ev-3", "meaning":"Clinical evidence in different tumour type"},
    "LEVEL_4":          {"label":"4",  "css":"ev-4", "meaning":"Biological evidence; preclinical"},
    "LEVEL_R1":         {"label":"R1", "css":"ev-R", "meaning":"Standard care resistance biomarker"},
    "LEVEL_R2":         {"label":"R2", "css":"ev-R", "meaning":"Resistance evidence (investigational)"},
    "ONCOGENIC":        {"label":"ON", "css":"ev-2", "meaning":"Oncogenic but no matched therapy"},
    "LIKELY_ONCOGENIC": {"label":"LO", "css":"ev-3", "meaning":"Likely oncogenic"},
}

CLINVAR_COLORS = {
    "Pathogenic":             "#f85149",
    "Likely pathogenic":      "#f0883e",
    "Uncertain significance": "#e3b341",
    "Likely benign":          "#58a6ff",
    "Benign":                 "#3fb950",
}

CIVIC_LEVEL_CSS = {"A":"ev-1","B":"ev-2","C":"ev-3","D":"ev-4","E":"ev-NA"}

TMB_COUNTED = [
    "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
    "In_Frame_Del","In_Frame_Ins","Splice_Site","Translation_Start_Site","Nonstop_Mutation",
]

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_mutation_data():
    csvs = list(Path("data").glob("*_merged.csv"))
    return pd.read_csv(csvs[0]) if csvs else None

def get_cols(df):
    gene_col    = next((c for c in ["Hugo_Symbol","gene_hugoGeneSymbol"] if c in df.columns), None)
    sample_col  = next((c for c in ["Tumor_Sample_Barcode","sampleId"]   if c in df.columns), None)
    variant_col = next((c for c in ["Variant_Classification","mutationType"] if c in df.columns), None)
    hgvsp_col   = next((c for c in ["HGVSp_Short","proteinChange"]        if c in df.columns), None)
    hgvsc_col   = next((c for c in ["HGVSc","cdnaChange"]                 if c in df.columns), None)
    return gene_col, sample_col, variant_col, hgvsp_col, hgvsc_col

# ── API clients ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def oncokb_annotate_gene(gene: str, oncokb_token: str) -> dict:
    headers = {"Authorization": f"Bearer {oncokb_token}", "Accept": "application/json"}
    try:
        r = requests.get(
            f"https://www.oncokb.org/api/v1/genes/{gene}",
            headers=headers, timeout=7
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

@st.cache_data(ttl=3600, show_spinner=False)
def oncokb_annotate_variant(gene: str, alteration: str, tumor_type: str, oncokb_token: str) -> dict:
    headers = {"Authorization": f"Bearer {oncokb_token}", "Accept": "application/json"}
    params  = {
        "hugoSymbol":      gene,
        "alteration":      alteration,
        "tumorType":       tumor_type,
        "referenceGenome": "GRCh38",
    }
    try:
        r = requests.get(
            "https://www.oncokb.org/api/v1/annotate/mutations/byProteinChange",
            headers=headers, params=params, timeout=7
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

@st.cache_data(ttl=3600, show_spinner=False)
def clinvar_search(gene: str, alteration: str = "") -> list:
    query = f"{gene}[gene] AND cancer[condition]"
    if alteration:
        query = f"{gene}[gene] AND {alteration}[variant]"
    try:
        search_r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db":"clinvar","term":query,"retmax":10,"retmode":"json"},
            timeout=8
        )
        ids = search_r.json().get("esearchresult",{}).get("idlist",[])
        if not ids:
            return []
        summary_r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db":"clinvar","id":",".join(ids),"retmode":"json"},
            timeout=8
        )
        results = summary_r.json().get("result",{})
        rows = []
        for uid in ids:
            item = results.get(uid, {})
            if not item:
                continue
            sig = item.get("clinical_significance",{})
            rows.append({
                "clinvar_id":    uid,
                "title":         item.get("title",""),
                "significance":  sig.get("description","Unknown") if isinstance(sig,dict) else str(sig),
                "review_status": item.get("review_status",""),
                "variation_set": item.get("variation_set",[{}])[0].get("variation_name","") if item.get("variation_set") else "",
            })
        return rows
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def civic_search(gene: str) -> list:
    base = "https://civicdb.org/api/v2"
    try:
        r = requests.get(f"{base}/genes", params={"name": gene, "count": 5}, timeout=8)
        if r.status_code != 200:
            return []
        records = r.json().get("records", [])
        if not records:
            return []
        gene_id = records[0]["id"]
        ev_r = requests.get(
            f"{base}/evidence_items",
            params={"gene_id": gene_id, "status": "accepted", "count": 50},
            timeout=8
        )
        if ev_r.status_code != 200:
            return []
        items = ev_r.json().get("records", [])
        rows = []
        for item in items:
            therapies = [t.get("name","") for t in (item.get("therapies") or [])]
            disease   = (item.get("disease") or {}).get("name","")
            variant   = (item.get("variant") or {}).get("name","")
            rows.append({
                "gene":           gene,
                "variant":        variant,
                "evidence_level": item.get("evidence_level",""),
                "evidence_type":  item.get("evidence_type",""),
                "significance":   item.get("significance",""),
                "disease":        disease,
                "therapies":      ", ".join(therapies) if therapies else "N/A",
                "description":    (item.get("description") or "")[:250],
                "citation":       (item.get("source") or {}).get("citation",""),
            })
        return rows
    except Exception:
        return []

# ── Rendering helpers ─────────────────────────────────────────────────────────

def evidence_badge(level_key: str) -> str:
    info = ONCOKB_LEVEL_INFO.get(level_key, {"label": level_key or "N/A", "css": "ev-NA"})
    return f"<span class='ev-badge {info['css']}'>{info['label']}</span>"

def clinvar_badge(sig: str) -> str:
    color      = CLINVAR_COLORS.get(sig, "#30363d")
    text_color = "#0d1117" if sig in ("Pathogenic","Benign","Likely benign") else "#e6edf3"
    return (
        f"<span style='background:{color};color:{text_color};"
        f"padding:2px 10px;border-radius:12px;"
        f"font-size:0.72rem;font-family:IBM Plex Mono,monospace;font-weight:700'>{sig}</span>"
    )

def civic_badge(level: str) -> str:
    css = CIVIC_LEVEL_CSS.get(level, "ev-NA")
    return f"<span class='ev-badge {css}'>CIViC-{level}</span>"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Biomarker Explorer")
    st.markdown("---")
    st.markdown("### OncoKB API Token")
    oncokb_token = st.text_input(
        "Token", type="password",
        placeholder="Get free token at oncokb.org",
        help="Register at oncokb.org/account/register for a free API token"
    )
    st.markdown(
        "<div style='font-size:0.68rem;color:#8b949e'>"
        "Free token at oncokb.org/account/register</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("### Tumour Type")
    tumor_type = st.selectbox("Tumour type for OncoKB", [
        "Lung Adenocarcinoma","Non-Small Cell Lung Cancer",
        "Breast Cancer","Colorectal Cancer","Melanoma",
        "Glioblastoma","Prostate Cancer","Ovarian Cancer",
        "Endometrial Cancer","Bladder Cancer","Pancreatic Cancer",
    ], index=0)
    st.markdown("---")
    st.markdown("### Filters")
    show_sources = st.multiselect(
        "Data sources",
        ["OncoKB","ClinVar","CIViC"],
        default=["OncoKB","ClinVar","CIViC"]
    )
    min_civic_level = st.selectbox("Min CIViC evidence level", ["A","B","C","D","E"], index=2)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.68rem;color:#8b949e'>"
        "ClinVar and CIViC are free with no token required. "
        "OncoKB requires a free academic token.</div>",
        unsafe_allow_html=True
    )

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# Biomarker Explorer")
st.markdown(
    "<div style='color:#8b949e;font-size:0.9rem;margin-bottom:24px'>"
    "Live clinical annotation | OncoKB | ClinVar | CIViC | FDA evidence levels | Therapy resistance"
    "</div>",
    unsafe_allow_html=True
)

# ── Load mutation data ────────────────────────────────────────────────────────

mut_df = load_mutation_data()
if mut_df is None:
    st.warning("No mutation data found. Run python3 tmb_data_download.py first.")
    st.stop()

gene_col, sample_col, variant_col, hgvsp_col, hgvsc_col = get_cols(mut_df)
df_coding       = mut_df[mut_df[variant_col].isin(TMB_COUNTED)] if variant_col else mut_df
top_genes_in_data = df_coding[gene_col].value_counts().head(40).index.tolist()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_cohort, tab_single = st.tabs(["Cohort View", "Single Gene Deep-Dive"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: COHORT VIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab_cohort:
    st.markdown('<div class="section-header">Top Mutated Genes -- Live Annotation</div>', unsafe_allow_html=True)

    n_genes          = st.slider("Number of top genes to annotate", 5, 30, 10)
    genes_to_annotate = top_genes_in_data[:n_genes]

    if st.button("Fetch annotations for all genes", type="primary"):

        civic_results   = []
        clinvar_results = []
        oncokb_results  = []

        def fetch_gene_annotations(gene):
            civic_items = civic_search(gene)   if "CIViC"   in show_sources else []
            cv_items    = clinvar_search(gene)  if "ClinVar" in show_sources else []
            okb_item    = {}
            if "OncoKB" in show_sources and oncokb_token:
                okb_item = oncokb_annotate_gene(gene, oncokb_token)
            return gene, civic_items, cv_items, okb_item

        prog      = st.progress(0, text="Fetching annotations in parallel...")
        completed = 0

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_gene_annotations, g): g for g in genes_to_annotate}
            for future in as_completed(futures):
                gene, civic_items, cv_items, okb_item = future.result()
                completed += 1
                prog.progress(
                    completed / len(genes_to_annotate),
                    text=f"Annotated {gene} ({completed}/{len(genes_to_annotate)})"
                )
                for item in civic_items:
                    item["gene"] = gene
                civic_results.extend(civic_items)
                for item in cv_items:
                    item["gene"] = gene
                clinvar_results.extend(cv_items)
                if okb_item:
                    oncokb_results.append({"gene": gene, **okb_item})

        prog.empty()

        # CIViC summary
        if civic_results:
            st.markdown('<div class="section-header" style="margin-top:16px">CIViC Evidence Summary</div>', unsafe_allow_html=True)
            civic_df   = pd.DataFrame(civic_results)
            level_order = {"A":0,"B":1,"C":2,"D":3,"E":4}
            min_ord    = level_order.get(min_civic_level, 4)
            civic_df   = civic_df[civic_df["evidence_level"].apply(lambda l: level_order.get(l,99) <= min_ord)]

            if not civic_df.empty:
                pivot = civic_df.groupby(["gene","evidence_type"]).size().unstack(fill_value=0)
                fig_heat = go.Figure(go.Heatmap(
                    z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                    colorscale=[[0,"#0d1117"],[0.2,"#161b22"],[1,"#3fb950"]],
                    hovertemplate="<b>%{y}</b><br>%{x}: %{z} evidence items<extra></extra>",
                    text=pivot.values, texttemplate="%{text}",
                    textfont=dict(size=10, color="#e6edf3"),
                ))
                fig_heat.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                    font_family="IBM Plex Mono",
                    xaxis=dict(color="#8b949e", tickangle=-30),
                    yaxis=dict(color="#8b949e"),
                    margin=dict(l=0,r=0,t=10,b=60), height=320,
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                level_counts = civic_df["evidence_level"].value_counts().reset_index()
                level_counts.columns = ["level","count"]
                fig_lev = go.Figure(go.Bar(
                    x=level_counts["level"], y=level_counts["count"],
                    marker_color=["#3fb950","#58a6ff","#e3b341","#f0883e","#8b949e"][:len(level_counts)],
                    hovertemplate="Level %{x}: %{y} items<extra></extra>",
                ))
                fig_lev.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                    font_family="IBM Plex Mono",
                    xaxis=dict(title="CIViC Evidence Level", color="#8b949e"),
                    yaxis=dict(title="Count", color="#8b949e", gridcolor="#21262d"),
                    margin=dict(l=0,r=0,t=10,b=0), height=220,
                )
                st.plotly_chart(fig_lev, use_container_width=True)

                with st.expander("Full CIViC table"):
                    st.dataframe(civic_df.reset_index(drop=True), use_container_width=True)
            else:
                st.info(f"No CIViC evidence at level {min_civic_level} or better.")

        # ClinVar summary
        if clinvar_results:
            st.markdown('<div class="section-header" style="margin-top:16px">ClinVar Clinical Significance</div>', unsafe_allow_html=True)
            cv_df      = pd.DataFrame(clinvar_results)
            sig_counts = cv_df["significance"].value_counts().reset_index()
            sig_counts.columns = ["significance","count"]
            fig_cv = go.Figure(go.Bar(
                x=sig_counts["significance"], y=sig_counts["count"],
                marker_color=[CLINVAR_COLORS.get(s,"#30363d") for s in sig_counts["significance"]],
                hovertemplate="<b>%{x}</b><br>%{y} variants<extra></extra>",
            ))
            fig_cv.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                font_family="IBM Plex Mono",
                xaxis=dict(title="", color="#8b949e", tickangle=-25),
                yaxis=dict(title="Variant Count", color="#8b949e", gridcolor="#21262d"),
                margin=dict(l=0,r=0,t=10,b=60), height=260,
            )
            st.plotly_chart(fig_cv, use_container_width=True)

        # OncoKB table
        if oncokb_results:
            st.markdown('<div class="section-header" style="margin-top:16px">OncoKB Annotated Genes</div>', unsafe_allow_html=True)
            ok_df       = pd.DataFrame(oncokb_results)
            display_cols = [c for c in ["gene","hugoSymbol","oncogene","tsg",
                                         "highestSensitiveLevel","highestResistanceLevel"] if c in ok_df.columns]
            if display_cols:
                st.dataframe(ok_df[display_cols].reset_index(drop=True), use_container_width=True)
        elif "OncoKB" in show_sources and not oncokb_token:
            st.info("Add your OncoKB token in the sidebar to enable OncoKB annotations.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: SINGLE GENE DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_single:
    col_gene, col_alt = st.columns([1, 1])

    with col_gene:
        selected_gene = st.selectbox("Gene", sorted(set(top_genes_in_data)), index=0)

    with col_alt:
        gene_muts    = df_coding[df_coding[gene_col] == selected_gene]
        alts_in_data = []
        if hgvsp_col and hgvsp_col in gene_muts.columns:
            alts_in_data = gene_muts[hgvsp_col].dropna().unique().tolist()
        selected_alt = st.selectbox(
            "Alteration (for OncoKB variant lookup)",
            ["(gene-level)"] + alts_in_data, index=0
        )

    # Mutation summary
    st.markdown('<div class="section-header">Mutation Profile in Cohort</div>', unsafe_allow_html=True)

    n_samples_total   = mut_df[sample_col].nunique()
    n_samples_mutated = gene_muts[sample_col].nunique()
    freq_pct          = round(n_samples_mutated / n_samples_total * 100, 1) if n_samples_total else 0
    var_counts        = gene_muts[variant_col].value_counts().reset_index()
    var_counts.columns = ["variant_type","count"]

    mc1, mc2, mc3 = st.columns(3)
    for col_w, val, label in zip(
        [mc1, mc2, mc3],
        [n_samples_mutated, f"{freq_pct}%", len(gene_muts)],
        ["Samples Mutated","Frequency","Total Mutations"]
    ):
        col_w.markdown(
            f"<div style='background:#161b22;border:1px solid #30363d;"
            f"border-radius:8px;padding:12px 16px;text-align:center'>"
            f"<div style='font-family:IBM Plex Mono;font-size:1.4rem;color:#58a6ff'>{val}</div>"
            f"<div style='font-size:0.7rem;color:#8b949e;text-transform:uppercase'>{label}</div></div>",
            unsafe_allow_html=True
        )

    if not var_counts.empty:
        fig_var = go.Figure(go.Bar(
            x=var_counts["variant_type"], y=var_counts["count"],
            marker_color="#58a6ff",
            hovertemplate="<b>%{x}</b><br>%{y} mutations<extra></extra>",
        ))
        fig_var.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
            font_family="IBM Plex Mono",
            xaxis=dict(title="", color="#8b949e", tickangle=-30),
            yaxis=dict(title="Count", color="#8b949e", gridcolor="#21262d"),
            margin=dict(l=0,r=0,t=10,b=60), height=220,
        )
        st.plotly_chart(fig_var, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # OncoKB section
    if "OncoKB" in show_sources:
        st.markdown('<div class="section-header">OncoKB Annotation</div>', unsafe_allow_html=True)
        if not oncokb_token:
            st.markdown(
                "<div style='background:#161b22;border:1px solid #f0883e;border-radius:8px;"
                "padding:14px 18px;font-size:0.85rem;color:#8b949e'>"
                "<b style='color:#f0883e'>OncoKB token required.</b> "
                "Register free at oncokb.org/account/register and paste your token in the sidebar."
                "</div>",
                unsafe_allow_html=True
            )
        else:
            with st.spinner(f"Querying OncoKB for {selected_gene}..."):
                gene_info    = oncokb_annotate_gene(selected_gene, oncokb_token)
                variant_info = {}
                if selected_alt and selected_alt != "(gene-level)":
                    alt_clean    = selected_alt.replace("p.","")
                    variant_info = oncokb_annotate_variant(selected_gene, alt_clean, tumor_type, oncokb_token)

            if gene_info:
                is_oncogene = gene_info.get("oncogene", False)
                is_tsg      = gene_info.get("tsg", False)
                gene_type   = "Oncogene" if is_oncogene else ("Tumour Suppressor" if is_tsg else "Unknown")
                hs_level    = gene_info.get("highestSensitiveLevel","")
                hr_level    = gene_info.get("highestResistanceLevel","")
                hs_meaning  = ONCOKB_LEVEL_INFO.get(hs_level,{}).get("meaning","") if hs_level else ""
                hr_meaning  = ONCOKB_LEVEL_INFO.get(hr_level,{}).get("meaning","") if hr_level else ""
                hs_badge    = evidence_badge(hs_level) if hs_level else "<span style='color:#8b949e'>N/A</span>"
                hr_badge    = evidence_badge(hr_level) if hr_level else "<span style='color:#8b949e'>N/A</span>"

                st.markdown(
                    f"<div class='biomarker-card'>"
                    f"<span class='source-tag oncokb-tag'>OncoKB</span> "
                    f"<b style='font-size:1.1rem;color:#e6edf3'>{selected_gene}</b> "
                    f"<span style='color:#8b949e;font-size:0.85rem;margin-left:8px'>{gene_type}</span>"
                    f"<div style='margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:8px'>"
                    f"<div><div style='font-size:0.68rem;color:#8b949e;text-transform:uppercase'>Highest Sensitivity Level</div>"
                    f"<div style='margin-top:4px'>{hs_badge}</div>"
                    f"<div style='font-size:0.72rem;color:#8b949e;margin-top:4px'>{hs_meaning}</div></div>"
                    f"<div><div style='font-size:0.68rem;color:#8b949e;text-transform:uppercase'>Highest Resistance Level</div>"
                    f"<div style='margin-top:4px'>{hr_badge}</div>"
                    f"<div style='font-size:0.72rem;color:#8b949e;margin-top:4px'>{hr_meaning}</div></div>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

            if variant_info:
                treatments   = variant_info.get("treatments",[]) or []
                oncogenicity = variant_info.get("oncogenic","")
                mut_effect   = (variant_info.get("mutationEffect") or {}).get("knownEffect","")

                st.markdown(
                    f"<div class='biomarker-card'>"
                    f"<span class='source-tag oncokb-tag'>OncoKB Variant</span> "
                    f"<b style='color:#e6edf3'>{selected_alt}</b>"
                    f"<div style='margin-top:10px;display:flex;gap:16px'>"
                    f"<div><div style='font-size:0.68rem;color:#8b949e;text-transform:uppercase'>Oncogenicity</div>"
                    f"<div style='font-family:IBM Plex Mono;color:#e3b341;font-size:0.9rem'>{oncogenicity or 'Unknown'}</div></div>"
                    f"<div><div style='font-size:0.68rem;color:#8b949e;text-transform:uppercase'>Mutation Effect</div>"
                    f"<div style='font-family:IBM Plex Mono;color:#f0883e;font-size:0.9rem'>{mut_effect or 'Unknown'}</div></div>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

                if treatments:
                    st.markdown("**Matched Treatments (OncoKB)**")
                    for tx in treatments[:8]:
                        level  = tx.get("level","")
                        drugs  = ", ".join([d.get("drugName","") for d in tx.get("drugs",[])])
                        cancer = tx.get("approvedIndications","") or tx.get("cancerTypes","")
                        if isinstance(cancer, list):
                            cancer = ", ".join([c.get("mainType","") for c in cancer])
                        st.markdown(
                            f"<div class='biomarker-card' style='padding:12px 16px'>"
                            f"{evidence_badge(level)} &nbsp;"
                            f"<b style='color:#e6edf3'>{drugs}</b> &nbsp;"
                            f"<span style='color:#8b949e;font-size:0.82rem'>{cancer}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # CIViC section
    if "CIViC" in show_sources:
        st.markdown('<div class="section-header">CIViC Evidence</div>', unsafe_allow_html=True)
        with st.spinner(f"Querying CIViC for {selected_gene}..."):
            civic_items = civic_search(selected_gene)

        if not civic_items:
            st.info(f"No CIViC evidence found for {selected_gene}.")
        else:
            civic_df    = pd.DataFrame(civic_items)
            level_order = {"A":0,"B":1,"C":2,"D":3,"E":4}
            min_ord     = level_order.get(min_civic_level, 4)
            civic_df    = civic_df[civic_df["evidence_level"].apply(lambda l: level_order.get(l,99) <= min_ord)]

            if civic_df.empty:
                st.info(f"No CIViC evidence at level {min_civic_level} or better.")
            else:
                for variant_name, grp in civic_df.groupby("variant"):
                    with st.expander(f"{selected_gene} -- {variant_name}  ({len(grp)} evidence items)"):
                        for _, ev_row in grp.iterrows():
                            therapies = ev_row.get("therapies","N/A")
                            disease   = ev_row.get("disease","")
                            ev_type   = ev_row.get("evidence_type","")
                            level     = ev_row.get("evidence_level","")
                            sig       = ev_row.get("significance","")
                            desc      = ev_row.get("description","")
                            cite      = ev_row.get("citation","")
                            sig_color = {
                                "SENSITIVITYRESPONSE": "#3fb950",
                                "RESISTANCE":          "#f85149",
                                "PREDICTIVE":          "#58a6ff",
                                "PROGNOSTIC":          "#e3b341",
                            }.get(sig,"#8b949e")
                            st.markdown(
                                f"<div class='biomarker-card' style='padding:12px 16px;margin-bottom:8px'>"
                                f"{civic_badge(level)} &nbsp;"
                                f"<span style='color:{sig_color};font-size:0.8rem;font-weight:600'>{sig}</span> &nbsp;"
                                f"<span style='color:#8b949e;font-size:0.78rem'>{ev_type}</span><br>"
                                f"<b style='color:#e6edf3;font-size:0.9rem'>{therapies}</b> "
                                f"<span style='color:#8b949e;font-size:0.8rem'>in {disease}</span><br>"
                                f"<span style='color:#8b949e;font-size:0.75rem'>{desc}</span><br>"
                                f"<span style='color:#30363d;font-size:0.7rem'>{cite}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

    # ClinVar section
    if "ClinVar" in show_sources:
        st.markdown('<div class="section-header">ClinVar Clinical Significance</div>', unsafe_allow_html=True)
        with st.spinner(f"Querying ClinVar for {selected_gene}..."):
            cv_items = clinvar_search(selected_gene)

        if not cv_items:
            st.info(f"No ClinVar entries found for {selected_gene}.")
        else:
            cv_df = pd.DataFrame(cv_items)
            for _, cv_row in cv_df.iterrows():
                sig = cv_row.get("significance","Unknown")
                st.markdown(
                    f"<div class='biomarker-card' style='padding:12px 16px;margin-bottom:8px'>"
                    f"<span class='source-tag clinvar-tag'>ClinVar</span> &nbsp;"
                    f"{clinvar_badge(sig)}<br>"
                    f"<span style='color:#e6edf3;font-size:0.85rem'>"
                    f"{cv_row.get('variation_set','') or cv_row.get('title','')}</span><br>"
                    f"<span style='color:#8b949e;font-size:0.75rem'>"
                    f"Review status: {cv_row.get('review_status','')} &nbsp;|&nbsp; "
                    f"ID: {cv_row.get('clinvar_id','')}</span></div>",
                    unsafe_allow_html=True
                )

    # Evidence legend
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Evidence Level Guide</div>', unsafe_allow_html=True)
    legend_rows = [
        ("OncoKB Level 1",  "ev-1", "FDA-approved biomarker and matched therapy in this tumour type"),
        ("OncoKB Level 2",  "ev-2", "Standard care; supported by consensus guidelines"),
        ("OncoKB Level 3A", "ev-3", "Compelling clinical evidence from early trials"),
        ("OncoKB Level R1", "ev-R", "Standard care resistance biomarker"),
        ("CIViC Level A",   "ev-1", "Validated association in published clinical guidelines"),
        ("CIViC Level B",   "ev-2", "Clinical evidence from trial or cohort study"),
        ("CIViC Level C",   "ev-3", "Case study or small series"),
        ("ClinVar P/LP",    "ev-R", "Pathogenic / Likely Pathogenic"),
    ]
    for label, css, meaning in legend_rows:
        st.markdown(
            f"<span class='ev-badge {css}'>{label.split()[-1]}</span> &nbsp;"
            f"<span style='color:#8b949e;font-size:0.82rem'>"
            f"<b style='color:#e6edf3'>{label}</b> -- {meaning}</span><br>",
            unsafe_allow_html=True
        )