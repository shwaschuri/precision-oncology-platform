"""
Data Upload Portal -- MAF, MAF.GZ, CSV, VCF support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gzip
import io
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Data Upload Portal", page_icon=None, layout="wide")

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
    background:#161b22; border:1px solid #30363d; border-radius:8px;
    padding:14px 18px; text-align:center;
}
.stat-value { font-family:'IBM Plex Mono',monospace; font-size:1.4rem; font-weight:600; color:#58a6ff; }
.stat-label { font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }
.upload-zone {
    background:#161b22; border:2px dashed #30363d; border-radius:10px;
    padding:24px; text-align:center; margin-bottom:16px;
}
.format-tag {
    display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
    padding:3px 10px; border-radius:8px; margin:3px;
    background:#21262d; color:#58a6ff; border:1px solid #30363d;
}
.status-ok   { color:#3fb950; font-family:'IBM Plex Mono',monospace; font-size:0.85rem; }
.status-warn { color:#e3b341; font-family:'IBM Plex Mono',monospace; font-size:0.85rem; }
.status-err  { color:#f85149; font-family:'IBM Plex Mono',monospace; font-size:0.85rem; }
div[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

EXOME_SIZE_MB = 38.0

TMB_COUNTED_VARIANTS = [
    "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
    "In_Frame_Del","In_Frame_Ins","Splice_Site","Translation_Start_Site","Nonstop_Mutation",
]

MAF_REQUIRED_COLS = [
    "Hugo_Symbol","Variant_Classification","Tumor_Sample_Barcode",
]

MAF_OPTIONAL_COLS = [
    "Chromosome","Start_Position","End_Position","Variant_Type",
    "Reference_Allele","Tumor_Seq_Allele1","Tumor_Seq_Allele2",
    "HGVSp_Short","HGVSc","t_depth","t_alt_count",
]

VCF_INFO_FIELDS = ["DP","AF","AD","MQ","QD","FS","SOR"]

# ── Parser functions ──────────────────────────────────────────────────────────

def parse_maf_content(content: str) -> tuple:
    """Parse MAF text content. Returns (dataframe, warnings)."""
    warnings = []
    lines = [l for l in content.splitlines() if not l.startswith("#")]
    if not lines:
        return None, ["File appears empty after removing comment lines."]

    try:
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", low_memory=False)
    except Exception as e:
        return None, [f"Could not parse as TSV: {e}"]

    missing_required = [c for c in MAF_REQUIRED_COLS if c not in df.columns]
    if missing_required:
        warnings.append(f"Missing expected MAF columns: {', '.join(missing_required)}")

    present_optional = [c for c in MAF_OPTIONAL_COLS if c in df.columns]
    if len(present_optional) < 3:
        warnings.append("Few optional MAF columns found. Some analyses may be limited.")

    return df, warnings

def parse_vcf_content(content: str) -> tuple:
    """
    Parse VCF into a MAF-compatible DataFrame.
    Extracts CHROM, POS, REF, ALT, FILTER, INFO fields.
    Assigns one sample per file (sample name from #CHROM header if present).
    """
    warnings = []
    meta_lines  = []
    header_line = None
    data_lines  = []

    for line in content.splitlines():
        if line.startswith("##"):
            meta_lines.append(line)
        elif line.startswith("#CHROM"):
            header_line = line
        elif line.strip():
            data_lines.append(line)

    if not data_lines:
        return None, ["No variant data found in VCF."]

    # Determine sample name
    sample_name = "UPLOADED_SAMPLE"
    if header_line:
        cols = header_line.lstrip("#").split("\t")
        if len(cols) > 9:
            sample_name = cols[9]

    rows = []
    for line in data_lines:
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        chrom  = parts[0]
        pos    = parts[1]
        ref    = parts[3]
        alt    = parts[4].split(",")[0]  # take first ALT
        filt   = parts[6] if len(parts) > 6 else "."
        info   = parts[7] if len(parts) > 7 else "."

        # Determine variant type
        if len(ref) == 1 and len(alt) == 1:
            vtype = "SNP"
        elif len(ref) > len(alt):
            vtype = "DEL"
        elif len(ref) < len(alt):
            vtype = "INS"
        else:
            vtype = "ONP"

        # Extract DP and AF from INFO
        dp, af = None, None
        for field in info.split(";"):
            if field.startswith("DP="):
                try:
                    dp = int(field.split("=")[1])
                except Exception:
                    pass
            if field.startswith("AF="):
                try:
                    af = float(field.split("=")[1].split(",")[0])
                except Exception:
                    pass

        rows.append({
            "Hugo_Symbol":          ".",
            "Chromosome":           chrom,
            "Start_Position":       pos,
            "Reference_Allele":     ref,
            "Tumor_Seq_Allele2":    alt,
            "Variant_Type":         vtype,
            "Variant_Classification": map_vcf_variant_class(ref, alt, vtype),
            "Tumor_Sample_Barcode": sample_name,
            "FILTER":               filt,
            "t_depth":              dp,
            "allele_freq":          af,
        })

    if not rows:
        return None, ["Could not parse any variants from VCF."]

    df = pd.DataFrame(rows)
    warnings.append(
        "VCF parsed successfully. Note: Hugo_Symbol (gene name) is not available in standard VCF "
        "format and has been set to '.'. Gene-level analyses will be limited. "
        "Consider annotating your VCF with a tool such as VEP or ANNOVAR first."
    )
    return df, warnings

def map_vcf_variant_class(ref: str, alt: str, vtype: str) -> str:
    """Approximate MAF Variant_Classification from VCF alleles."""
    if vtype == "SNP":
        return "Missense_Mutation"
    elif vtype == "DEL":
        diff = len(ref) - len(alt)
        return "Frame_Shift_Del" if diff % 3 != 0 else "In_Frame_Del"
    elif vtype == "INS":
        diff = len(alt) - len(ref)
        return "Frame_Shift_Ins" if diff % 3 != 0 else "In_Frame_Ins"
    return "Missense_Mutation"

def parse_csv_tmb(content: str) -> tuple:
    """Parse a TMB CSV (sample_id, TMB, TMB_category, ...)."""
    warnings = []
    try:
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return None, [f"Could not parse CSV: {e}"]

    required = ["sample_id","TMB"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        warnings.append(f"Missing expected columns: {', '.join(missing)}. Found: {list(df.columns)}")

    return df, warnings

def infer_file_type(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".vcf") or name.endswith(".vcf.gz"):
        return "VCF"
    if name.endswith(".maf.gz") or name.endswith(".maf"):
        return "MAF"
    if name.endswith(".gz"):
        return "MAF_GZ"
    if name.endswith(".csv"):
        return "CSV"
    if name.endswith(".tsv") or name.endswith(".txt"):
        return "MAF"
    return "UNKNOWN"

def compute_tmb_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TMB scores from a parsed MAF-like dataframe."""
    sample_col  = next((c for c in ["Tumor_Sample_Barcode","sampleId"] if c in df.columns), None)
    variant_col = next((c for c in ["Variant_Classification","mutationType"] if c in df.columns), None)
    if not sample_col or not variant_col:
        return pd.DataFrame()
    df_c = df[df[variant_col].isin(TMB_COUNTED_VARIANTS)]
    tmb  = df_c.groupby(sample_col).size().reset_index(name="mutation_count")
    tmb.rename(columns={sample_col:"sample_id"}, inplace=True)
    tmb["TMB"] = (tmb["mutation_count"] / EXOME_SIZE_MB).round(2)
    tmb["TMB_category"] = pd.cut(
        tmb["TMB"], bins=[0,6,16,float("inf")],
        labels=["Low (<6)","Medium (6-16)","High (>16)"]
    )
    return tmb.sort_values("TMB", ascending=False).reset_index(drop=True)

def validate_dataframe(df: pd.DataFrame, file_type: str) -> list:
    """Run validation checks and return list of (level, message) tuples."""
    checks = []
    n_rows = len(df)
    n_cols = len(df.columns)

    checks.append(("ok", f"Parsed {n_rows:,} rows and {n_cols} columns"))

    if n_rows == 0:
        checks.append(("error", "File contains no data rows"))
        return checks

    sample_col = next((c for c in ["Tumor_Sample_Barcode","sampleId","sample_id"] if c in df.columns), None)
    if sample_col:
        n_samples = df[sample_col].nunique()
        checks.append(("ok", f"{n_samples} unique sample(s) detected"))
    else:
        checks.append(("warn", "No sample ID column detected"))

    variant_col = next((c for c in ["Variant_Classification","mutationType"] if c in df.columns), None)
    if variant_col:
        unknown_types = set(df[variant_col].unique()) - set(TMB_COUNTED_VARIANTS + ["Silent","3'UTR","5'UTR","Intron","IGR","RNA","lincRNA"])
        if unknown_types:
            checks.append(("warn", f"Unrecognised variant classifications: {', '.join(list(unknown_types)[:5])}"))
        else:
            checks.append(("ok", "All variant classifications recognised"))
    else:
        checks.append(("warn", "No variant classification column detected"))

    # Check for duplicates
    if sample_col and variant_col:
        gene_col = next((c for c in ["Hugo_Symbol","gene_hugoGeneSymbol"] if c in df.columns), None)
        if gene_col:
            dup_cols = [sample_col, gene_col, "Chromosome","Start_Position"]
            dup_cols = [c for c in dup_cols if c in df.columns]
            n_dups   = df.duplicated(subset=dup_cols).sum()
            if n_dups > 0:
                checks.append(("warn", f"{n_dups} potential duplicate rows detected"))
            else:
                checks.append(("ok", "No duplicate rows detected"))

    # VCF-specific
    if file_type == "VCF":
        if "FILTER" in df.columns:
            n_pass   = (df["FILTER"] == "PASS").sum()
            n_filter = len(df) - n_pass
            checks.append(("ok" if n_pass > 0 else "warn",
                            f"{n_pass} PASS variants, {n_filter} filtered variants"))
        checks.append(("warn",
            "VCF lacks gene annotations. TMB calculation uses all variants as proxies for missense mutations."))

    return checks

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Data Upload Portal")
    st.markdown("---")
    st.markdown("### Supported Formats")
    for fmt in ["MAF","MAF.GZ","CSV (TMB scores)","VCF"]:
        st.markdown(f"<span class='format-tag'>{fmt}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Options")
    save_to_disk  = st.checkbox("Save processed data to data/ folder", value=True)
    show_raw_data = st.checkbox("Show raw data preview", value=False)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.68rem;color:#8b949e'>"
        "Uploaded files are processed in memory. "
        "Enable save to persist them for use in other pages.</div>",
        unsafe_allow_html=True
    )

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# Data Upload Portal")
st.markdown(
    "<div style='color:#8b949e;font-size:0.9rem;margin-bottom:24px'>"
    "Upload MAF, MAF.GZ, VCF, or CSV files | Automatic validation | TMB computation | Save for analysis"
    "</div>",
    unsafe_allow_html=True
)

# ── Format guide ──────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Supported File Formats</div>', unsafe_allow_html=True)

fg1, fg2, fg3, fg4 = st.columns(4)
for col_w, fmt, desc, ext in zip(
    [fg1, fg2, fg3, fg4],
    ["MAF", "MAF.GZ", "VCF", "CSV"],
    [
        "Mutation Annotation Format from TCGA/GDC. Standard format for somatic mutations.",
        "Gzip-compressed MAF file. Same format as MAF but compressed.",
        "Variant Call Format. Standard output from GATK, VarScan, etc.",
        "TMB scores CSV with sample_id and TMB columns. For pre-computed scores.",
    ],
    [".maf, .txt, .tsv", ".maf.gz", ".vcf, .vcf.gz", ".csv"],
):
    col_w.markdown(
        f"<div class='stat-card' style='text-align:left'>"
        f"<div style='font-family:IBM Plex Mono;color:#58a6ff;font-size:1rem;font-weight:700'>{fmt}</div>"
        f"<div style='font-size:0.72rem;color:#8b949e;margin-top:6px'>{desc}</div>"
        f"<div style='font-size:0.68rem;color:#30363d;margin-top:6px'>{ext}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── File uploader ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Upload File</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop or browse",
    type=["maf","gz","vcf","csv","tsv","txt"],
    help="Supported: MAF, MAF.GZ, VCF, VCF.GZ, CSV, TSV"
)

if uploaded_file is None:
    st.markdown(
        "<div class='upload-zone'>"
        "<div style='font-size:0.9rem;color:#8b949e'>No file uploaded yet.</div>"
        "<div style='font-size:0.78rem;color:#30363d;margin-top:8px'>"
        "Accepted formats: .maf | .maf.gz | .vcf | .vcf.gz | .csv | .tsv</div>"
        "</div>",
        unsafe_allow_html=True
    )
    st.stop()

# ── Parse file ────────────────────────────────────────────────────────────────

file_type  = infer_file_type(uploaded_file.name)
raw_bytes  = uploaded_file.read()
parse_warnings = []
parsed_df      = None
tmb_df         = None
file_type_label = file_type

st.markdown('<div class="section-header">File Information</div>', unsafe_allow_html=True)

fi1, fi2, fi3 = st.columns(3)
fi1.markdown(
    f"<div class='stat-card'><div class='stat-label'>File Name</div>"
    f"<div style='font-family:IBM Plex Mono;font-size:0.85rem;color:#e6edf3;word-break:break-all'>"
    f"{uploaded_file.name}</div></div>",
    unsafe_allow_html=True
)
fi2.markdown(
    f"<div class='stat-card'><div class='stat-label'>File Size</div>"
    f"<div class='stat-value'>{len(raw_bytes)/1024:.1f} KB</div></div>",
    unsafe_allow_html=True
)
fi3.markdown(
    f"<div class='stat-card'><div class='stat-label'>Detected Format</div>"
    f"<div class='stat-value' style='font-size:1rem'>{file_type_label}</div></div>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# Decompress if needed
if uploaded_file.name.lower().endswith(".gz"):
    try:
        raw_bytes = gzip.decompress(raw_bytes)
    except Exception as e:
        st.error(f"Failed to decompress .gz file: {e}")
        st.stop()

content = raw_bytes.decode("utf-8", errors="replace")

# Parse based on type
with st.spinner("Parsing file..."):
    if file_type in ("MAF","MAF_GZ"):
        parsed_df, parse_warnings = parse_maf_content(content)
    elif file_type == "VCF":
        parsed_df, parse_warnings = parse_vcf_content(content)
        file_type_label = "VCF"
    elif file_type == "CSV":
        parsed_df, parse_warnings = parse_csv_tmb(content)
        file_type_label = "CSV (TMB)"
    else:
        # Try MAF as fallback
        parsed_df, parse_warnings = parse_maf_content(content)
        if parsed_df is None:
            parsed_df, parse_warnings = parse_vcf_content(content)

if parsed_df is None:
    st.error("Could not parse the uploaded file.")
    for w in parse_warnings:
        st.warning(w)
    st.stop()

# ── Validation ────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Validation Report</div>', unsafe_allow_html=True)

checks = validate_dataframe(parsed_df, file_type)
for level, msg in checks:
    css = {"ok":"status-ok","warn":"status-warn","error":"status-err"}.get(level,"status-warn")
    icon = {"ok":"[OK]","warn":"[WARN]","error":"[ERROR]"}.get(level,"[INFO]")
    st.markdown(f"<div class='{css}'>{icon} {msg}</div>", unsafe_allow_html=True)

for w in parse_warnings:
    st.markdown(f"<div class='status-warn'>[WARN] {w}</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Compute TMB ───────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">TMB Computation</div>', unsafe_allow_html=True)

if file_type == "CSV":
    tmb_df = parsed_df
    st.info("CSV detected as pre-computed TMB scores. Displaying directly.")
else:
    with st.spinner("Computing TMB..."):
        tmb_df = compute_tmb_from_df(parsed_df)

if tmb_df is not None and not tmb_df.empty:
    n_samples = len(tmb_df)
    median_tmb = tmb_df["TMB"].median()
    n_high = int((tmb_df["TMB"] > 16).sum()) if "TMB" in tmb_df.columns else 0

    tc1, tc2, tc3 = st.columns(3)
    for col_w, val, label in zip(
        [tc1, tc2, tc3],
        [n_samples, f"{median_tmb:.2f}", n_high],
        ["Samples","Median TMB (mut/Mb)","TMB-High (>16)"]
    ):
        col_w.markdown(
            f"<div class='stat-card'><div class='stat-value'>{val}</div>"
            f"<div class='stat-label'>{label}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # TMB bar chart
    if "TMB_category" in tmb_df.columns:
        color_map  = {"Low (<6)":"#3fb950","Medium (6-16)":"#e3b341","High (>16)":"#f85149"}
        bar_colors = tmb_df["TMB_category"].astype(str).map(color_map).fillna("#58a6ff")
    else:
        bar_colors = "#58a6ff"

    fig = go.Figure(go.Bar(
        x=tmb_df["sample_id"].astype(str).str[:20],
        y=tmb_df["TMB"],
        marker_color=bar_colors,
        hovertemplate="<b>%{x}</b><br>TMB: %{y:.2f} mut/Mb<extra></extra>",
    ))
    fig.add_hline(y=16, line_dash="dash", line_color="#f85149",
                  annotation_text="High (16)", annotation_font_color="#f85149")
    fig.add_hline(y=6,  line_dash="dash", line_color="#e3b341",
                  annotation_text="Medium (6)", annotation_font_color="#e3b341")
    fig.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        font_family="IBM Plex Mono",
        xaxis=dict(showticklabels=False, title="Samples", color="#8b949e"),
        yaxis=dict(title="TMB (mut/Mb)", color="#8b949e", gridcolor="#21262d"),
        margin=dict(l=0,r=0,t=10,b=0), height=280,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(tmb_df.reset_index(drop=True), use_container_width=True)

else:
    st.warning("Could not compute TMB from this file. Check that Variant_Classification and Tumor_Sample_Barcode columns are present.")

# ── Raw data preview ──────────────────────────────────────────────────────────

if show_raw_data:
    st.markdown('<div class="section-header">Raw Data Preview (first 100 rows)</div>', unsafe_allow_html=True)
    st.dataframe(parsed_df.head(100), use_container_width=True)
    st.markdown(
        f"<div style='font-size:0.75rem;color:#8b949e'>"
        f"Showing 100 of {len(parsed_df):,} rows | {len(parsed_df.columns)} columns</div>",
        unsafe_allow_html=True
    )

# ── Column map ────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Column Summary</div>', unsafe_allow_html=True)

col_rows = []
for col in parsed_df.columns[:30]:
    n_null  = parsed_df[col].isna().sum()
    n_uniq  = parsed_df[col].nunique()
    sample  = str(parsed_df[col].dropna().iloc[0]) if not parsed_df[col].dropna().empty else ""
    col_rows.append({
        "Column":     col,
        "Non-null":   len(parsed_df) - n_null,
        "Null":       n_null,
        "Unique":     n_uniq,
        "Example":    sample[:60],
    })
col_summary = pd.DataFrame(col_rows)
st.dataframe(col_summary, use_container_width=True)

# ── Save to disk ──────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Save and Export</div>', unsafe_allow_html=True)

date_str  = datetime.now().strftime("%Y%m%d_%H%M")
safe_name = uploaded_file.name.replace(".gz","").replace(".maf","").replace(".vcf","").replace(".csv","")

if save_to_disk and tmb_df is not None and not tmb_df.empty:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Save TMB scores
    tmb_path = data_dir / "tmb_scores.csv"
    tmb_df.to_csv(tmb_path, index=False)

    # Save merged mutations if MAF/VCF
    if file_type in ("MAF","MAF_GZ","VCF"):
        mut_path = data_dir / f"{safe_name}_merged.csv"
        parsed_df.to_csv(mut_path, index=False)
        st.success(
            f"Saved: data/tmb_scores.csv and data/{safe_name}_merged.csv. "
            "These files will be available in all other dashboard pages."
        )
    else:
        st.success("Saved: data/tmb_scores.csv")

ec1, ec2, ec3 = st.columns(3)

with ec1:
    if tmb_df is not None and not tmb_df.empty:
        st.download_button(
            "Download TMB Scores CSV",
            tmb_df.to_csv(index=False).encode(),
            f"tmb_scores_{date_str}.csv","text/csv"
        )

with ec2:
    st.download_button(
        "Download Parsed Mutations CSV",
        parsed_df.to_csv(index=False).encode(),
        f"mutations_{safe_name}_{date_str}.csv","text/csv"
    )

with ec3:
    st.download_button(
        "Download Column Summary CSV",
        col_summary.to_csv(index=False).encode(),
        f"column_summary_{date_str}.csv","text/csv"
    )

# ── Batch upload note ─────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Batch Upload</div>', unsafe_allow_html=True)
st.markdown(
    "<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;"
    "padding:16px 20px;font-size:0.85rem;color:#8b949e'>"
    "<b style='color:#e6edf3'>Batch upload (multiple files):</b><br><br>"
    "For batch processing of many MAF files, use the data downloader script which handles "
    "parallel downloads directly from GDC:<br><br>"
    "<code style='color:#58a6ff'>python3 tmb_data_download.py</code><br><br>"
    "This downloads and merges multiple MAF files automatically and saves them to the data/ folder, "
    "making them available across all dashboard pages."
    "</div>",
    unsafe_allow_html=True
)