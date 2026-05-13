"""
Clinical Report Page -- Generate downloadable per-patient reports
Requires: pip install reportlab
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO, StringIO
from datetime import datetime

st.set_page_config(page_title="Clinical Report", page_icon=None, layout="wide")

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
.report-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 20px 24px; margin-bottom: 14px;
}
.report-label {
    font-size: 0.68rem; color: #8b949e; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 4px;
}
.report-value {
    font-family: 'IBM Plex Mono', monospace; font-size: 1rem; color: #e6edf3;
}
.tmb-high   { color: #f85149; font-weight: 700; }
.tmb-medium { color: #e3b341; font-weight: 700; }
.tmb-low    { color: #3fb950; font-weight: 700; }
div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

EXOME_SIZE_MB = 38.0

TMB_COUNTED_VARIANTS = [
    "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
    "In_Frame_Del","In_Frame_Ins","Splice_Site","Translation_Start_Site","Nonstop_Mutation",
]

ACTIONABLE_GENES = {
    "EGFR":   {"drug": "Erlotinib / Osimertinib",  "therapy": "Targeted",    "level": "1"},
    "KRAS":   {"drug": "Sotorasib (G12C only)",     "therapy": "Targeted",    "level": "2"},
    "BRAF":   {"drug": "Vemurafenib / Dabrafenib",  "therapy": "Targeted",    "level": "1"},
    "ALK":    {"drug": "Crizotinib / Alectinib",    "therapy": "Targeted",    "level": "1"},
    "MET":    {"drug": "Capmatinib",                "therapy": "Targeted",    "level": "2"},
    "RET":    {"drug": "Selpercatinib",             "therapy": "Targeted",    "level": "1"},
    "ERBB2":  {"drug": "Trastuzumab / Tucatinib",   "therapy": "Targeted",    "level": "2"},
    "PIK3CA": {"drug": "Alpelisib",                 "therapy": "Targeted",    "level": "2"},
    "BRCA1":  {"drug": "Olaparib (PARP inhibitor)", "therapy": "Targeted",    "level": "1"},
    "BRCA2":  {"drug": "Olaparib (PARP inhibitor)", "therapy": "Targeted",    "level": "1"},
    "TP53":   {"drug": "N/A",                       "therapy": "Prognostic",  "level": "N/A"},
    "STK11":  {"drug": "N/A",                       "therapy": "Prognostic",  "level": "N/A"},
    "KEAP1":  {"drug": "N/A",                       "therapy": "Prognostic",  "level": "N/A"},
    "PTEN":   {"drug": "N/A",                       "therapy": "Prognostic",  "level": "N/A"},
    "RB1":    {"drug": "N/A",                       "therapy": "Prognostic",  "level": "N/A"},
    "MLH1":   {"drug": "Pembrolizumab (MSI-H)",     "therapy": "Immunotherapy","level": "1"},
    "MSH2":   {"drug": "Pembrolizumab (MSI-H)",     "therapy": "Immunotherapy","level": "1"},
    "MSH6":   {"drug": "Pembrolizumab (MSI-H)",     "therapy": "Immunotherapy","level": "1"},
    "PMS2":   {"drug": "Pembrolizumab (MSI-H)",     "therapy": "Immunotherapy","level": "1"},
    "POLE":   {"drug": "Pembrolizumab (TMB-H)",     "therapy": "Immunotherapy","level": "2"},
}

INDEL_TYPES = ["Frame_Shift_Del","Frame_Shift_Ins","In_Frame_Del","In_Frame_Ins"]

# ── Data loaders ──────────────────────────────────────────────────────────────

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
    hgvsp_col   = next((c for c in ["HGVSp_Short","proteinChange"]        if c in df.columns), None)
    return gene_col, sample_col, variant_col, hgvsp_col

# ── Report builder ────────────────────────────────────────────────────────────

def build_report_data(sample_id, mut_df, tmb_df, gene_col, sample_col, variant_col, hgvsp_col):
    """Compile all data needed for one patient report."""

    # TMB
    tmb_row = tmb_df[tmb_df["sample_id"] == sample_id] if tmb_df is not None else pd.DataFrame()
    if not tmb_row.empty:
        tmb_val = float(tmb_row["TMB"].values[0])
        mut_count = int(tmb_row["mutation_count"].values[0]) if "mutation_count" in tmb_row.columns else None
    else:
        s_coding  = mut_df[
            (mut_df[sample_col] == sample_id) &
            (mut_df[variant_col].isin(TMB_COUNTED_VARIANTS))
        ]
        tmb_val   = round(len(s_coding) / EXOME_SIZE_MB, 2)
        mut_count = len(s_coding)

    if tmb_val >= 16:
        tmb_category = "High"
        tmb_class    = "tmb-high"
    elif tmb_val >= 6:
        tmb_category = "Medium"
        tmb_class    = "tmb-medium"
    else:
        tmb_category = "Low"
        tmb_class    = "tmb-low"

    # Mutations for this sample
    s_df      = mut_df[mut_df[sample_col] == sample_id]
    s_coding  = s_df[s_df[variant_col].isin(TMB_COUNTED_VARIANTS)]

    # Variant classification counts
    var_counts = s_coding[variant_col].value_counts().to_dict()

    # Indel fraction
    n_indels   = s_coding[s_coding[variant_col].isin(INDEL_TYPES)].shape[0]
    n_total    = len(s_coding)
    indel_frac = round(n_indels / n_total * 100, 1) if n_total > 0 else 0.0

    # Actionable biomarkers
    biomarkers = []
    for gene, info in ACTIONABLE_GENES.items():
        gene_muts = s_coding[s_coding[gene_col] == gene]
        if gene_muts.empty:
            continue
        for _, row in gene_muts.iterrows():
            protein = row.get(hgvsp_col, "") if hgvsp_col and hgvsp_col in row.index else ""
            biomarkers.append({
                "gene":         gene,
                "protein":      protein if pd.notna(protein) else "",
                "variant_type": row.get(variant_col, ""),
                "drug":         info["drug"],
                "therapy":      info["therapy"],
                "level":        info["level"],
            })

    # MSI estimate
    mmr_hits  = [b["gene"] for b in biomarkers if b["gene"] in ["MLH1","MSH2","MSH6","PMS2","POLE","POLD1"]]
    if tmb_val >= 20 or (indel_frac >= 20 and tmb_val >= 10) or len(mmr_hits) >= 2:
        msi_status = "MSI-H (suspected)"
    elif tmb_val >= 10 or indel_frac >= 15 or len(mmr_hits) >= 1:
        msi_status = "Indeterminate"
    else:
        msi_status = "MSS (suspected)"

    # Clinical interpretation
    interpretations = []
    if tmb_val >= 10:
        interpretations.append(
            "Patient has elevated TMB (>= 10 mut/Mb). "
            "May be eligible for pembrolizumab under FDA pan-tumour TMB-H approval."
        )
    if "MSI-H" in msi_status:
        interpretations.append(
            "Suspected MSI-H phenotype based on TMB, indel fraction, and MMR gene mutations. "
            "Recommend confirmatory MMR IHC or PCR testing. "
            "If confirmed, checkpoint inhibitor therapy (e.g. pembrolizumab) is indicated."
        )
    targeted = [b for b in biomarkers if b["therapy"] == "Targeted" and b["level"] in ("1","2")]
    for b in targeted:
        interpretations.append(
            f"{b['gene']} {b['protein']} mutation detected. "
            f"Level {b['level']} evidence supports {b['drug']}."
        )
    if not interpretations:
        interpretations.append(
            "No high-confidence actionable biomarkers identified. "
            "Standard-of-care chemotherapy may be appropriate. "
            "Consider clinical trial enrolment."
        )

    return {
        "sample_id":      sample_id,
        "tmb_val":        tmb_val,
        "tmb_category":   tmb_category,
        "tmb_class":      tmb_class,
        "mut_count":      mut_count or n_total,
        "indel_fraction": indel_frac,
        "msi_status":     msi_status,
        "var_counts":     var_counts,
        "biomarkers":     biomarkers,
        "interpretations":interpretations,
        "report_date":    datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

# ── Export functions ──────────────────────────────────────────────────────────

def export_csv(report: dict) -> bytes:
    rows = []
    rows.append({"Field": "Patient ID",       "Value": report["sample_id"]})
    rows.append({"Field": "Report Date",      "Value": report["report_date"]})
    rows.append({"Field": "TMB (mut/Mb)",     "Value": report["tmb_val"]})
    rows.append({"Field": "TMB Category",     "Value": report["tmb_category"]})
    rows.append({"Field": "Mutation Count",   "Value": report["mut_count"]})
    rows.append({"Field": "Indel Fraction %", "Value": report["indel_fraction"]})
    rows.append({"Field": "MSI Status",       "Value": report["msi_status"]})
    rows.append({"Field": "", "Value": ""})
    rows.append({"Field": "--- BIOMARKERS ---", "Value": ""})
    for b in report["biomarkers"]:
        label = f"{b['gene']} {b['protein']}".strip()
        rows.append({"Field": label, "Value": f"{b['drug']} (Level {b['level']})"})
    rows.append({"Field": "", "Value": ""})
    rows.append({"Field": "--- INTERPRETATIONS ---", "Value": ""})
    for i, interp in enumerate(report["interpretations"], 1):
        rows.append({"Field": f"Interpretation {i}", "Value": interp})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

def export_html(report: dict) -> bytes:
    bio_rows = ""
    for b in report["biomarkers"]:
        label    = f"{b['gene']} {b['protein']}".strip()
        lv_color = "#3fb950" if b["level"] in ("1","2") else "#8b949e"
        bio_rows += (
            f"<tr>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #21262d'><b>{label}</b></td>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #21262d'>{b['variant_type']}</td>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #21262d'>{b['therapy']}</td>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #21262d'>{b['drug']}</td>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #21262d;"
            f"color:{lv_color};font-weight:700'>Level {b['level']}</td>"
            f"</tr>"
        )

    if not bio_rows:
        bio_rows = "<tr><td colspan='5' style='padding:8px 12px;color:#8b949e'>No actionable biomarkers detected</td></tr>"

    interp_items = "".join(f"<li style='margin-bottom:8px'>{i}</li>" for i in report["interpretations"])

    tmb_color = {"High":"#f85149","Medium":"#e3b341","Low":"#3fb950"}.get(report["tmb_category"],"#58a6ff")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Clinical Report -- {report['sample_id']}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#0d1117; color:#e6edf3; margin:0; padding:32px; }}
  h1   {{ font-family: monospace; color:#58a6ff; border-bottom:2px solid #21262d; padding-bottom:12px; }}
  h2   {{ font-family: monospace; color:#58a6ff; font-size:0.85rem; text-transform:uppercase;
           letter-spacing:0.15em; margin-top:28px; border-bottom:1px solid #21262d; padding-bottom:6px; }}
  .grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:16px 0; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px 18px; }}
  .card-label {{ font-size:0.68rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.1em; }}
  .card-value {{ font-family:monospace; font-size:1.2rem; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
  th    {{ background:#161b22; padding:10px 12px; text-align:left; font-size:0.78rem;
           color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; }}
  td    {{ font-size:0.88rem; }}
  ul    {{ color:#c9d1d9; line-height:1.8; }}
  .footer {{ margin-top:40px; padding-top:16px; border-top:1px solid #21262d;
             font-size:0.72rem; color:#8b949e; }}
</style>
</head>
<body>
<h1>Clinical Genomic Report</h1>
<p style='color:#8b949e;margin-top:-8px'>Generated: {report['report_date']} &nbsp;|&nbsp; Patient: {report['sample_id']}</p>

<h2>Summary</h2>
<div class="grid">
  <div class="card">
    <div class="card-label">Tumor Mutational Burden</div>
    <div class="card-value" style="color:{tmb_color}">{report['tmb_val']} mut/Mb</div>
    <div style="font-size:0.8rem;color:{tmb_color};margin-top:2px">{report['tmb_category']}</div>
  </div>
  <div class="card">
    <div class="card-label">Total Coding Mutations</div>
    <div class="card-value">{report['mut_count']}</div>
  </div>
  <div class="card">
    <div class="card-label">Indel Fraction</div>
    <div class="card-value">{report['indel_fraction']}%</div>
  </div>
  <div class="card">
    <div class="card-label">MSI Status (estimated)</div>
    <div class="card-value" style="font-size:0.95rem">{report['msi_status']}</div>
  </div>
  <div class="card">
    <div class="card-label">Actionable Biomarkers</div>
    <div class="card-value">{len(report['biomarkers'])}</div>
  </div>
  <div class="card">
    <div class="card-label">Therapy Recommendations</div>
    <div class="card-value">{len([b for b in report['biomarkers'] if b['level'] in ('1','2')])}</div>
  </div>
</div>

<h2>Detected Biomarkers</h2>
<table>
  <tr>
    <th>Gene / Variant</th><th>Type</th><th>Therapy Category</th>
    <th>Suggested Drug</th><th>Evidence Level</th>
  </tr>
  {bio_rows}
</table>

<h2>Clinical Interpretation</h2>
<ul>{interp_items}</ul>

<h2>Variant Classification Breakdown</h2>
<table>
  <tr><th>Variant Type</th><th>Count</th></tr>
  {"".join(f"<tr><td style='padding:6px 12px;border-bottom:1px solid #21262d'>{k}</td><td style='padding:6px 12px;border-bottom:1px solid #21262d'>{v}</td></tr>" for k,v in report['var_counts'].items())}
</table>

<div class="footer">
  <b>Disclaimer:</b> This report is generated from computational analysis of somatic mutation data
  and is intended for research purposes only. It does not constitute a clinical diagnosis or
  treatment recommendation. All findings should be validated by a certified clinical laboratory
  and interpreted by a qualified healthcare professional before any clinical decisions are made.
  TMB and MSI estimates are computational approximations and require orthogonal confirmation.
</div>
</body>
</html>"""
    return html.encode("utf-8")

def export_pdf(report: dict) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table,
            TableStyle, HRFlowable
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER

        buf    = BytesIO()
        doc    = SimpleDocTemplate(buf, pagesize=A4,
                                   leftMargin=2*cm, rightMargin=2*cm,
                                   topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "title", parent=styles["Heading1"],
            fontSize=18, textColor=colors.HexColor("#58a6ff"),
            spaceAfter=4,
        )
        sub_style = ParagraphStyle(
            "sub", parent=styles["Normal"],
            fontSize=9, textColor=colors.HexColor("#8b949e"),
            spaceAfter=12,
        )
        h2_style = ParagraphStyle(
            "h2", parent=styles["Heading2"],
            fontSize=10, textColor=colors.HexColor("#58a6ff"),
            spaceBefore=14, spaceAfter=6,
            fontName="Helvetica-Bold",
        )
        body_style = ParagraphStyle(
            "body", parent=styles["Normal"],
            fontSize=9, textColor=colors.HexColor("#c9d1d9"),
            spaceAfter=4, leading=14,
        )
        disclaimer_style = ParagraphStyle(
            "disc", parent=styles["Normal"],
            fontSize=7, textColor=colors.HexColor("#8b949e"),
            spaceAfter=0, leading=11,
        )

        tmb_color = {"High":"#f85149","Medium":"#e3b341","Low":"#3fb950"}.get(report["tmb_category"],"#58a6ff")

        story = []

        # Header
        story.append(Paragraph("Clinical Genomic Report", title_style))
        story.append(Paragraph(
            f"Generated: {report['report_date']}  |  Patient: {report['sample_id']}",
            sub_style
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#21262d"), spaceAfter=12))

        # Summary table
        story.append(Paragraph("Summary", h2_style))
        summary_data = [
            ["Field", "Value"],
            ["Patient ID",              report["sample_id"]],
            ["TMB (mut/Mb)",            f"{report['tmb_val']}  ({report['tmb_category']})"],
            ["Total Coding Mutations",   str(report["mut_count"])],
            ["Indel Fraction",           f"{report['indel_fraction']}%"],
            ["MSI Status (estimated)",   report["msi_status"]],
            ["Actionable Biomarkers",    str(len(report["biomarkers"]))],
        ]
        tbl = Table(summary_data, colWidths=[6*cm, 11*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(1,0),  colors.HexColor("#161b22")),
            ("TEXTCOLOR",    (0,0),(1,0),  colors.HexColor("#58a6ff")),
            ("FONTNAME",     (0,0),(1,0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0,0),(-1,-1), 9),
            ("BACKGROUND",   (0,1),(1,-1), colors.HexColor("#0d1117")),
            ("TEXTCOLOR",    (0,1),(0,-1), colors.HexColor("#8b949e")),
            ("TEXTCOLOR",    (1,1),(1,-1), colors.HexColor("#e6edf3")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#0d1117"),colors.HexColor("#161b22")]),
            ("GRID",         (0,0),(-1,-1), 0.5, colors.HexColor("#30363d")),
            ("LEFTPADDING",  (0,0),(-1,-1), 8),
            ("RIGHTPADDING", (0,0),(-1,-1), 8),
            ("TOPPADDING",   (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 12))

        # Biomarkers table
        story.append(Paragraph("Detected Biomarkers", h2_style))
        if report["biomarkers"]:
            bio_data = [["Gene / Variant", "Type", "Therapy", "Drug", "Level"]]
            for b in report["biomarkers"]:
                label = f"{b['gene']} {b['protein']}".strip()
                bio_data.append([label, b["variant_type"], b["therapy"], b["drug"], f"Level {b['level']}"])
            bio_tbl = Table(bio_data, colWidths=[3.5*cm, 3.5*cm, 3*cm, 5*cm, 2*cm])
            bio_tbl.setStyle(TableStyle([
                ("BACKGROUND",  (0,0),(-1,0),  colors.HexColor("#161b22")),
                ("TEXTCOLOR",   (0,0),(-1,0),  colors.HexColor("#58a6ff")),
                ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
                ("FONTSIZE",    (0,0),(-1,-1),  8),
                ("TEXTCOLOR",   (0,1),(-1,-1),  colors.HexColor("#c9d1d9")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#0d1117"),colors.HexColor("#161b22")]),
                ("GRID",        (0,0),(-1,-1),  0.5, colors.HexColor("#30363d")),
                ("LEFTPADDING", (0,0),(-1,-1),  6),
                ("RIGHTPADDING",(0,0),(-1,-1),  6),
                ("TOPPADDING",  (0,0),(-1,-1),  4),
                ("BOTTOMPADDING",(0,0),(-1,-1), 4),
                ("WORDWRAP",    (3,1),(3,-1),   True),
            ]))
            story.append(bio_tbl)
        else:
            story.append(Paragraph("No actionable biomarkers detected.", body_style))

        story.append(Spacer(1, 12))

        # Clinical interpretation
        story.append(Paragraph("Clinical Interpretation", h2_style))
        for i, interp in enumerate(report["interpretations"], 1):
            story.append(Paragraph(f"{i}. {interp}", body_style))

        story.append(Spacer(1, 12))

        # Variant breakdown
        story.append(Paragraph("Variant Classification Breakdown", h2_style))
        vc_data = [["Variant Type", "Count"]]
        for k, v in report["var_counts"].items():
            vc_data.append([k, str(v)])
        vc_tbl = Table(vc_data, colWidths=[10*cm, 3*cm])
        vc_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0),(-1,0),  colors.HexColor("#161b22")),
            ("TEXTCOLOR",   (0,0),(-1,0),  colors.HexColor("#58a6ff")),
            ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0,0),(-1,-1),  9),
            ("TEXTCOLOR",   (0,1),(-1,-1),  colors.HexColor("#c9d1d9")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#0d1117"),colors.HexColor("#161b22")]),
            ("GRID",        (0,0),(-1,-1),  0.5, colors.HexColor("#30363d")),
            ("LEFTPADDING", (0,0),(-1,-1),  8),
            ("TOPPADDING",  (0,0),(-1,-1),  4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ]))
        story.append(vc_tbl)
        story.append(Spacer(1, 20))

        # Disclaimer
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#21262d"), spaceAfter=8))
        story.append(Paragraph(
            "Disclaimer: This report is generated from computational analysis of somatic mutation data "
            "and is intended for research purposes only. It does not constitute a clinical diagnosis or "
            "treatment recommendation. All findings should be validated by a certified clinical laboratory "
            "and interpreted by a qualified healthcare professional. TMB and MSI estimates require "
            "orthogonal confirmation.",
            disclaimer_style
        ))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Clinical Report")
    st.markdown("---")
    st.markdown("### Export Formats")
    export_pdf_enabled  = st.checkbox("PDF (requires reportlab)", value=True)
    export_html_enabled = st.checkbox("HTML", value=True)
    export_csv_enabled  = st.checkbox("CSV", value=True)
    st.markdown("---")
    st.markdown("### Options")
    include_variants = st.checkbox("Include full variant list", value=False)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.68rem;color:#8b949e'>"
        "Install PDF support:<br>"
        "<code>pip3 install reportlab</code></div>",
        unsafe_allow_html=True
    )

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# Clinical Report Generator")
st.markdown(
    "<div style='color:#8b949e;font-size:0.9rem;margin-bottom:24px'>"
    "Per-patient genomic summary | Biomarker annotation | Therapy suggestions | PDF / HTML / CSV export"
    "</div>",
    unsafe_allow_html=True
)

# ── Load data ─────────────────────────────────────────────────────────────────

mut_df = load_mutation_data()
tmb_df = load_tmb_data()

if mut_df is None:
    st.warning("No mutation data found. Run python3 tmb_data_download.py first.")
    st.stop()

gene_col, sample_col, variant_col, hgvsp_col = get_cols(mut_df)
all_samples = sorted(mut_df[sample_col].unique().tolist())

# ── Sample selector ───────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Select Patient</div>', unsafe_allow_html=True)

col_sel, col_info = st.columns([2, 1])
with col_sel:
    selected_sample = st.selectbox("Patient / Sample ID", all_samples)
with col_info:
    n_muts = len(mut_df[mut_df[sample_col] == selected_sample])
    st.markdown(
        f"<div class='report-card' style='margin-top:28px'>"
        f"<div class='report-label'>Total mutations in sample</div>"
        f"<div class='report-value'>{n_muts}</div></div>",
        unsafe_allow_html=True
    )

# ── Build report ──────────────────────────────────────────────────────────────

with st.spinner("Building report..."):
    report = build_report_data(
        selected_sample, mut_df, tmb_df,
        gene_col, sample_col, variant_col, hgvsp_col
    )

# ── Report preview ────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Report Preview</div>', unsafe_allow_html=True)

# Top metrics row
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
tmb_color_hex = {"High":"#f85149","Medium":"#e3b341","Low":"#3fb950"}.get(report["tmb_category"],"#58a6ff")

for col_w, val, label, color in zip(
    [mc1, mc2, mc3, mc4, mc5],
    [
        f"{report['tmb_val']} mut/Mb",
        report["tmb_category"],
        str(report["mut_count"]),
        f"{report['indel_fraction']}%",
        report["msi_status"].split(" ")[0],
    ],
    ["TMB Score","TMB Category","Coding Mutations","Indel Fraction","MSI Status"],
    [tmb_color_hex, tmb_color_hex, "#58a6ff", "#58a6ff", "#e3b341"],
):
    col_w.markdown(
        f"<div class='report-card'>"
        f"<div class='report-label'>{label}</div>"
        f"<div class='report-value' style='color:{color}'>{val}</div></div>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Biomarkers section
st.markdown('<div class="section-header">Detected Biomarkers</div>', unsafe_allow_html=True)

if report["biomarkers"]:
    for b in report["biomarkers"]:
        label      = f"{b['gene']} {b['protein']}".strip()
        lv_color   = "#3fb950" if b["level"] in ("1","2") else "#8b949e"
        th_color   = {"Targeted":"#58a6ff","Immunotherapy":"#3fb950","Prognostic":"#8b949e"}.get(b["therapy"],"#8b949e")
        st.markdown(
            f"<div class='report-card'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<div>"
            f"<span style='font-family:IBM Plex Mono;font-size:1rem;color:#e6edf3;font-weight:700'>{label}</span>"
            f"&nbsp;&nbsp;"
            f"<span style='font-size:0.75rem;color:#8b949e'>{b['variant_type']}</span>"
            f"</div>"
            f"<span style='font-family:IBM Plex Mono;font-size:0.75rem;font-weight:700;"
            f"color:{lv_color};background:{lv_color}20;padding:2px 10px;border-radius:10px'>"
            f"Level {b['level']}</span>"
            f"</div>"
            f"<div style='margin-top:8px;display:flex;gap:24px'>"
            f"<div><div class='report-label'>Therapy Type</div>"
            f"<div style='font-size:0.85rem;color:{th_color}'>{b['therapy']}</div></div>"
            f"<div><div class='report-label'>Suggested Drug</div>"
            f"<div style='font-size:0.85rem;color:#e6edf3'>{b['drug']}</div></div>"
            f"</div></div>",
            unsafe_allow_html=True
        )
else:
    st.info("No actionable biomarkers detected for this sample.")

st.markdown("<br>", unsafe_allow_html=True)

# Clinical interpretation
st.markdown('<div class="section-header">Clinical Interpretation</div>', unsafe_allow_html=True)
for i, interp in enumerate(report["interpretations"], 1):
    st.markdown(
        f"<div class='report-card'>"
        f"<span style='font-family:IBM Plex Mono;color:#58a6ff;font-size:0.8rem'>{i}.</span> "
        f"<span style='color:#c9d1d9;font-size:0.9rem'>{interp}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Variant breakdown
st.markdown('<div class="section-header">Variant Classification Breakdown</div>', unsafe_allow_html=True)
if report["var_counts"]:
    vc_df = pd.DataFrame(list(report["var_counts"].items()), columns=["Variant Type","Count"])
    vc_df = vc_df.sort_values("Count", ascending=False)
    st.dataframe(vc_df.reset_index(drop=True), use_container_width=True)

# Full variant list
if include_variants:
    st.markdown('<div class="section-header">Full Variant List</div>', unsafe_allow_html=True)
    s_coding = mut_df[
        (mut_df[sample_col] == selected_sample) &
        (mut_df[variant_col].isin(TMB_COUNTED_VARIANTS))
    ]
    display_cols = [c for c in [gene_col, variant_col, hgvsp_col, "Chromosome",
                                 "Start_Position", "t_depth", "t_alt_count"] if c and c in s_coding.columns]
    st.dataframe(s_coding[display_cols].reset_index(drop=True), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Export buttons ────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Export Report</div>', unsafe_allow_html=True)

safe_id  = selected_sample.replace("/","_").replace(" ","_")
date_str = datetime.now().strftime("%Y%m%d")

ecol1, ecol2, ecol3 = st.columns(3)

with ecol1:
    if export_pdf_enabled:
        pdf_bytes = export_pdf(report)
        if pdf_bytes:
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"clinical_report_{safe_id}_{date_str}.pdf",
                mime="application/pdf",
            )
        else:
            st.warning("PDF export requires reportlab. Run: pip3 install reportlab")

with ecol2:
    if export_html_enabled:
        html_bytes = export_html(report)
        st.download_button(
            label="Download HTML Report",
            data=html_bytes,
            file_name=f"clinical_report_{safe_id}_{date_str}.html",
            mime="text/html",
        )

with ecol3:
    if export_csv_enabled:
        csv_bytes = export_csv(report)
        st.download_button(
            label="Download CSV Report",
            data=csv_bytes,
            file_name=f"clinical_report_{safe_id}_{date_str}.csv",
            mime="text/csv",
        )

# Batch export
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Batch Export</div>', unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:0.82rem;color:#8b949e;margin-bottom:12px'>"
    "Generate reports for all samples at once as a combined CSV.</div>",
    unsafe_allow_html=True
)

if st.button("Generate batch report for all samples"):
    batch_rows = []
    batch_prog = st.progress(0, text="Generating batch report...")
    for idx, sid in enumerate(all_samples):
        try:
            r = build_report_data(sid, mut_df, tmb_df, gene_col, sample_col, variant_col, hgvsp_col)
            biomarker_str = "; ".join(
                f"{b['gene']} {b['protein']}".strip() for b in r["biomarkers"]
            ) or "None"
            drug_str = "; ".join(
                b["drug"] for b in r["biomarkers"] if b["level"] in ("1","2")
            ) or "None"
            batch_rows.append({
                "sample_id":          sid,
                "TMB (mut/Mb)":       r["tmb_val"],
                "TMB Category":       r["tmb_category"],
                "Coding Mutations":   r["mut_count"],
                "Indel Fraction (%)": r["indel_fraction"],
                "MSI Status":         r["msi_status"],
                "Biomarkers":         biomarker_str,
                "Suggested Drugs":    drug_str,
            })
        except Exception:
            pass
        batch_prog.progress((idx + 1) / len(all_samples))

    batch_prog.empty()
    batch_df = pd.DataFrame(batch_rows)
    st.dataframe(batch_df, use_container_width=True)
    st.download_button(
        "Download Batch Report CSV",
        batch_df.to_csv(index=False).encode(),
        f"batch_report_{date_str}.csv",
        "text/csv"
    )

st.markdown(
    "<div style='margin-top:24px;padding:14px 18px;background:#161b22;"
    "border:1px solid #30363d;border-radius:8px;font-size:0.75rem;color:#8b949e'>"
    "<b>Disclaimer:</b> This report is generated from computational analysis of somatic mutation data "
    "and is intended for research and educational purposes only. "
    "It does not constitute a clinical diagnosis or treatment recommendation. "
    "All findings must be validated by a certified clinical laboratory and interpreted by a "
    "qualified healthcare professional before any clinical decisions are made."
    "</div>",
    unsafe_allow_html=True
)