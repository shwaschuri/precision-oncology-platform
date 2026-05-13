"""
Machine Learning Features Page
Immunotherapy response | Mutation clustering | Tumor subtype classification
Explainable AI | Neoantigen prediction | Multi-omics integration
Requires: pip install scikit-learn shap umap-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from io import StringIO

st.set_page_config(page_title="ML Features", page_icon=None, layout="wide")

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
.pred-card {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:16px 20px; margin-bottom:10px;
}
div[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

EXOME_SIZE_MB = 38.0

TMB_COUNTED = [
    "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
    "In_Frame_Del","In_Frame_Ins","Splice_Site","Translation_Start_Site","Nonstop_Mutation",
]

INDEL_TYPES = ["Frame_Shift_Del","Frame_Shift_Ins","In_Frame_Del","In_Frame_Ins"]

DRIVER_GENES = [
    "TP53","KRAS","PIK3CA","PTEN","APC","BRAF","EGFR","CDH1","RB1",
    "BRCA1","BRCA2","MYC","CDKN2A","STK11","KEAP1","NFE2L2","SMAD4",
    "MLH1","MSH2","MSH6","POLE","VHL","IDH1","FBXW7","CTNNB1",
    "MET","ALK","RET","ERBB2","FGFR1","FGFR2","FGFR3","NF1",
    "RNF43","ARID1A","ARID2","KMT2D","KMT2C","SETD2","BAP1",
]

# HLA supertypes used for neoantigen binding estimation
HLA_SUPERTYPES = {
    "HLA-A*02:01": {"supertype": "A2",   "freq": 0.44},
    "HLA-A*03:01": {"supertype": "A3",   "freq": 0.15},
    "HLA-A*24:02": {"supertype": "A24",  "freq": 0.19},
    "HLA-B*07:02": {"supertype": "B7",   "freq": 0.22},
    "HLA-B*44:02": {"supertype": "B44",  "freq": 0.18},
    "HLA-B*35:01": {"supertype": "B62",  "freq": 0.12},
}

# Amino acid hydrophobicity scale (Kyte-Doolittle)
AA_HYDROPHOBICITY = {
    "A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
    "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
    "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2,
}

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

# ── Feature matrix builder ────────────────────────────────────────────────────

@st.cache_data
def build_feature_matrix(mut_df_hash, tmb_df_hash):
    """
    Build a per-sample numeric feature matrix for ML.
    Features:
      - TMB
      - Indel fraction
      - Binary mutation flags for each driver gene
      - Variant type proportions (missense%, nonsense%, frameshift%, splice%)
    """
    mut_df = load_mutation_data()
    tmb_df = load_tmb_data()
    if mut_df is None:
        return None

    gene_col, sample_col, variant_col, _ = get_cols(mut_df)
    df_coding = mut_df[mut_df[variant_col].isin(TMB_COUNTED)]
    samples   = mut_df[sample_col].unique()
    rows      = []

    for sample in samples:
        s     = df_coding[df_coding[sample_col] == sample]
        n_tot = len(s)

        # TMB
        tmb_row = tmb_df[tmb_df["sample_id"] == sample] if tmb_df is not None else pd.DataFrame()
        tmb_val = float(tmb_row["TMB"].values[0]) if not tmb_row.empty else (n_tot / EXOME_SIZE_MB)

        # Indel fraction
        n_indels   = s[s[variant_col].isin(INDEL_TYPES)].shape[0]
        indel_frac = n_indels / n_tot if n_tot > 0 else 0.0

        # Variant type proportions
        v_counts = s[variant_col].value_counts(normalize=True).to_dict()

        # Gene mutation flags
        genes_mutated = set(s[gene_col].unique())

        row = {"sample_id": sample, "TMB": tmb_val, "indel_fraction": indel_frac}
        row["pct_missense"]   = v_counts.get("Missense_Mutation", 0)
        row["pct_nonsense"]   = v_counts.get("Nonsense_Mutation", 0)
        row["pct_frameshift"] = sum(v_counts.get(t, 0) for t in INDEL_TYPES)
        row["pct_splice"]     = v_counts.get("Splice_Site", 0)
        row["n_mutations"]    = n_tot

        for gene in DRIVER_GENES:
            row[f"mut_{gene}"] = 1 if gene in genes_mutated else 0

        rows.append(row)

    feature_df = pd.DataFrame(rows).set_index("sample_id")
    feature_df = feature_df.fillna(0)
    return feature_df

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ML Features")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.68rem;color:#8b949e'>"
        "Required packages:<br>"
        "<code>pip3 install scikit-learn shap umap-learn</code><br><br>"
        "All models run on your local data from the data/ folder.</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("### Global Options")
    random_seed = st.number_input("Random seed", value=42, step=1)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# Machine Learning Features")
st.markdown(
    "<div style='color:#8b949e;font-size:0.9rem;margin-bottom:24px'>"
    "Immunotherapy response | Mutation clustering | Tumor subtype classification | "
    "Explainable AI | Neoantigen prediction | Multi-omics integration"
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

with st.spinner("Building feature matrix..."):
    feature_df = build_feature_matrix(str(len(mut_df)), str(len(tmb_df) if tmb_df is not None else 0))

if feature_df is None or feature_df.empty:
    st.error("Could not build feature matrix from available data.")
    st.stop()

n_samples  = len(feature_df)
n_features = len(feature_df.columns)

# Summary metrics
mc1, mc2, mc3 = st.columns(3)
for col_w, val, label in zip(
    [mc1, mc2, mc3],
    [n_samples, n_features, len(DRIVER_GENES)],
    ["Samples", "Features", "Driver Genes"]
):
    col_w.markdown(
        f"<div class='stat-card'><div class='stat-value'>{val}</div>"
        f"<div class='stat-label'>{label}</div></div>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

(tab_immuno, tab_cluster, tab_subtype,
 tab_xai, tab_neoag, tab_multiomics) = st.tabs([
    "Immunotherapy Response",
    "Mutation Clustering",
    "Tumor Subtype",
    "Explainable AI",
    "Neoantigen Prediction",
    "Multi-omics Integration",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: IMMUNOTHERAPY RESPONSE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

with tab_immuno:
    st.markdown('<div class="section-header">Immunotherapy Response Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.82rem;color:#8b949e;margin-bottom:16px'>"
        "Predicts likelihood of immunotherapy response using a Random Forest trained on "
        "TMB, indel fraction, MSI-associated features, and driver gene mutation status. "
        "Labels are derived from established biomarker thresholds (TMB >= 10, indel >= 15%, "
        "or MMR gene mutations) rather than clinical outcome data, since no survival labels "
        "are available in the current dataset."
        "</div>",
        unsafe_allow_html=True
    )

    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.calibration import CalibratedClassifierCV
        import warnings
        warnings.filterwarnings("ignore")

        # Build proxy labels: 1 = likely responder, 0 = likely non-responder
        mmr_genes  = ["mut_MLH1","mut_MSH2","mut_MSH6","mut_PMS2","mut_POLE"]
        mmr_cols   = [c for c in mmr_genes if c in feature_df.columns]
        mmr_hit    = feature_df[mmr_cols].sum(axis=1) >= 1 if mmr_cols else pd.Series(0, index=feature_df.index)

        labels = (
            (feature_df["TMB"] >= 10) |
            (feature_df["indel_fraction"] >= 0.15) |
            mmr_hit
        ).astype(int)

        X = feature_df.values
        y = labels.values

        n_pos  = y.sum()
        n_neg  = len(y) - n_pos

        ic1, ic2, ic3 = st.columns(3)
        for col_w, val, label in zip(
            [ic1, ic2, ic3],
            [len(y), n_pos, n_neg],
            ["Total Samples","Predicted Responders","Predicted Non-responders"]
        ):
            col_w.markdown(
                f"<div class='stat-card'><div class='stat-value'>{val}</div>"
                f"<div class='stat-label'>{label}</div></div>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="section-header">Model Settings</div>', unsafe_allow_html=True)
            model_type  = st.selectbox("Model", ["Random Forest","Gradient Boosting"], key="immuno_model")
            n_estimators = st.slider("Number of trees", 50, 500, 100, step=50, key="immuno_trees")
            run_cv      = st.checkbox("Run cross-validation", value=True, key="immuno_cv")

        with col_right:
            st.markdown('<div class="section-header">Label Definition</div>', unsafe_allow_html=True)
            tmb_thresh  = st.slider("TMB threshold", 5.0, 20.0, 10.0, step=0.5, key="immuno_tmb")
            indel_thresh = st.slider("Indel fraction threshold", 0.05, 0.40, 0.15, step=0.05, key="immuno_indel")

            labels = (
                (feature_df["TMB"] >= tmb_thresh) |
                (feature_df["indel_fraction"] >= indel_thresh) |
                mmr_hit
            ).astype(int)
            y = labels.values

        if st.button("Train model", type="primary", key="immuno_train"):
            with st.spinner("Training..."):
                scaler = StandardScaler()
                X_sc   = scaler.fit_transform(X)

                if model_type == "Random Forest":
                    clf = RandomForestClassifier(
                        n_estimators=n_estimators, random_state=random_seed,
                        class_weight="balanced", n_jobs=-1
                    )
                else:
                    clf = GradientBoostingClassifier(
                        n_estimators=n_estimators, random_state=random_seed
                    )

                clf.fit(X_sc, y)
                proba = clf.predict_proba(X_sc)[:, 1]

                result_df = feature_df[["TMB","indel_fraction"]].copy()
                result_df["response_probability"] = proba.round(3)
                result_df["prediction"] = (proba >= 0.5).astype(int)
                result_df["label"] = result_df["prediction"].map({1:"Likely Responder",0:"Likely Non-responder"})
                result_df = result_df.reset_index()

                # Probability distribution
                st.markdown('<div class="section-header">Response Probability Distribution</div>', unsafe_allow_html=True)
                fig_prob = go.Figure()
                for lbl, color in [("Likely Responder","#3fb950"),("Likely Non-responder","#f85149")]:
                    sub = result_df[result_df["label"] == lbl]
                    fig_prob.add_trace(go.Histogram(
                        x=sub["response_probability"], name=lbl,
                        marker_color=color, opacity=0.75, nbinsx=20,
                        hovertemplate=f"{lbl}<br>Prob: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
                    ))
                fig_prob.update_layout(
                    barmode="overlay",
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                    font_family="IBM Plex Mono",
                    xaxis=dict(title="Response Probability", color="#8b949e", gridcolor="#21262d"),
                    yaxis=dict(title="Count", color="#8b949e", gridcolor="#21262d"),
                    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
                    margin=dict(l=0,r=0,t=10,b=0), height=300,
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                # TMB vs probability scatter
                st.markdown('<div class="section-header">TMB vs Response Probability</div>', unsafe_allow_html=True)
                fig_sc = go.Figure(go.Scatter(
                    x=result_df["TMB"],
                    y=result_df["response_probability"],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=result_df["response_probability"],
                        colorscale=[[0,"#f85149"],[0.5,"#e3b341"],[1,"#3fb950"]],
                        showscale=True,
                        colorbar=dict(title="Prob", tickfont=dict(color="#8b949e"), title_font=dict(color="#8b949e")),
                        line=dict(color="#0d1117",width=1),
                    ),
                    hovertemplate="<b>%{customdata}</b><br>TMB: %{x:.2f}<br>Prob: %{y:.3f}<extra></extra>",
                    customdata=result_df["sample_id"],
                ))
                fig_sc.add_hline(y=0.5, line_dash="dash", line_color="#8b949e",
                                 annotation_text="threshold 0.5", annotation_font_color="#8b949e")
                fig_sc.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                    font_family="IBM Plex Mono",
                    xaxis=dict(title="TMB (mut/Mb)", color="#8b949e", gridcolor="#21262d"),
                    yaxis=dict(title="Response Probability", color="#8b949e", gridcolor="#21262d"),
                    margin=dict(l=0,r=0,t=10,b=0), height=320,
                )
                st.plotly_chart(fig_sc, use_container_width=True)

                if run_cv and len(y) >= 6 and y.sum() >= 2 and (len(y) - y.sum()) >= 2:
                    cv  = StratifiedKFold(n_splits=min(5, y.sum()), shuffle=True, random_state=random_seed)
                    auc = cross_val_score(clf, X_sc, y, cv=cv, scoring="roc_auc")
                    st.markdown(
                        f"<div style='font-size:0.82rem;color:#8b949e;margin-top:8px'>"
                        f"Cross-validation AUC: "
                        f"<b style='color:#58a6ff'>{auc.mean():.3f} +/- {auc.std():.3f}</b> "
                        f"({len(auc)} folds)</div>",
                        unsafe_allow_html=True
                    )

                with st.expander("Full prediction table"):
                    st.dataframe(result_df.sort_values("response_probability", ascending=False).reset_index(drop=True), use_container_width=True)
                    st.download_button(
                        "Download predictions CSV",
                        result_df.to_csv(index=False).encode(),
                        "immunotherapy_predictions.csv","text/csv"
                    )

    except ImportError:
        st.info("Install required packages: pip3 install scikit-learn")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: MUTATION CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

with tab_cluster:
    st.markdown('<div class="section-header">Mutation Clustering</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.82rem;color:#8b949e;margin-bottom:16px'>"
        "Clusters samples based on their mutation profiles using K-Means or hierarchical "
        "clustering. Visualised via PCA (always available) or UMAP (requires umap-learn). "
        "Each cluster represents a group of samples with similar mutation landscapes."
        "</div>",
        unsafe_allow_html=True
    )

    try:
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        import warnings
        warnings.filterwarnings("ignore")

        col_cl1, col_cl2 = st.columns(2)
        with col_cl1:
            cluster_method = st.selectbox("Clustering method", ["K-Means","Hierarchical"], key="cl_method")
            n_clusters     = st.slider("Number of clusters (k)", 2, min(8, n_samples-1), 3, key="cl_k")
        with col_cl2:
            viz_method     = st.selectbox("Visualisation", ["PCA","UMAP (requires umap-learn)"], key="cl_viz")
            color_by       = st.selectbox("Colour by", ["Cluster","TMB","indel_fraction"], key="cl_color")

        if st.button("Run clustering", type="primary", key="cl_run"):
            with st.spinner("Clustering samples..."):
                scaler = StandardScaler()
                X_sc   = scaler.fit_transform(feature_df.values)

                if cluster_method == "K-Means":
                    model   = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
                    labels  = model.fit_predict(X_sc)
                else:
                    model   = AgglomerativeClustering(n_clusters=n_clusters)
                    labels  = model.fit_predict(X_sc)

                sil = silhouette_score(X_sc, labels) if n_clusters > 1 and len(set(labels)) > 1 else 0.0

                # Dimensionality reduction
                pca     = PCA(n_components=2, random_state=random_seed)
                coords  = pca.fit_transform(X_sc)
                var_exp = pca.explained_variance_ratio_

                use_umap = "UMAP" in viz_method
                if use_umap:
                    try:
                        import umap
                        reducer = umap.UMAP(n_components=2, random_state=random_seed)
                        coords  = reducer.fit_transform(X_sc)
                        axis_labels = ("UMAP-1","UMAP-2")
                    except ImportError:
                        st.warning("umap-learn not installed. Falling back to PCA. Install with: pip3 install umap-learn")
                        use_umap    = False
                        axis_labels = (f"PC1 ({var_exp[0]*100:.1f}%)", f"PC2 ({var_exp[1]*100:.1f}%)")
                else:
                    axis_labels = (f"PC1 ({var_exp[0]*100:.1f}%)", f"PC2 ({var_exp[1]*100:.1f}%)")

                plot_df = pd.DataFrame({
                    "x":          coords[:,0],
                    "y":          coords[:,1],
                    "cluster":    labels.astype(str),
                    "sample_id":  feature_df.index,
                    "TMB":        feature_df["TMB"].values,
                    "indel_fraction": feature_df["indel_fraction"].values,
                })

                CLUSTER_COLORS = ["#58a6ff","#f85149","#3fb950","#e3b341",
                                   "#d2a8ff","#f0883e","#bc8cff","#ff7b72"]

                st.markdown(
                    f"<div style='font-size:0.82rem;color:#8b949e;margin-bottom:12px'>"
                    f"Silhouette score: <b style='color:#58a6ff'>{sil:.3f}</b> "
                    f"(range -1 to 1; higher = better separation)</div>",
                    unsafe_allow_html=True
                )

                fig_cl = go.Figure()
                if color_by == "Cluster":
                    for ci in sorted(plot_df["cluster"].unique()):
                        sub   = plot_df[plot_df["cluster"] == ci]
                        color = CLUSTER_COLORS[int(ci) % len(CLUSTER_COLORS)]
                        fig_cl.add_trace(go.Scatter(
                            x=sub["x"], y=sub["y"], mode="markers",
                            name=f"Cluster {ci}",
                            marker=dict(size=9, color=color, line=dict(color="#0d1117",width=1)),
                            hovertemplate="<b>%{customdata}</b><br>Cluster: " + ci +
                                          "<br>TMB: %{text:.2f}<extra></extra>",
                            customdata=sub["sample_id"],
                            text=sub["TMB"],
                        ))
                else:
                    col_vals = plot_df[color_by]
                    fig_cl.add_trace(go.Scatter(
                        x=plot_df["x"], y=plot_df["y"], mode="markers",
                        marker=dict(
                            size=9,
                            color=col_vals,
                            colorscale=[[0,"#161b22"],[0.5,"#58a6ff"],[1,"#f85149"]],
                            showscale=True,
                            colorbar=dict(title=color_by, tickfont=dict(color="#8b949e"), title_font=dict(color="#8b949e")),
                            line=dict(color="#0d1117",width=1),
                        ),
                        hovertemplate="<b>%{customdata}</b><br>" + color_by + ": %{text:.3f}<extra></extra>",
                        customdata=plot_df["sample_id"],
                        text=col_vals,
                    ))

                fig_cl.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                    font_family="IBM Plex Mono",
                    xaxis=dict(title=axis_labels[0], color="#8b949e", gridcolor="#21262d"),
                    yaxis=dict(title=axis_labels[1], color="#8b949e", gridcolor="#21262d"),
                    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
                    margin=dict(l=0,r=0,t=10,b=0), height=420,
                )
                st.plotly_chart(fig_cl, use_container_width=True)

                # Cluster profile
                st.markdown('<div class="section-header">Cluster Profiles</div>', unsafe_allow_html=True)
                plot_df["cluster_int"] = labels
                profile = plot_df.groupby("cluster_int").agg(
                    n_samples=("sample_id","count"),
                    median_TMB=("TMB","median"),
                    mean_indel=("indel_fraction","mean"),
                ).reset_index()
                profile.columns = ["Cluster","Samples","Median TMB","Mean Indel Frac"]
                st.dataframe(profile, use_container_width=True)

                with st.expander("Full cluster assignments"):
                    out = plot_df[["sample_id","cluster","TMB","indel_fraction"]].reset_index(drop=True)
                    st.dataframe(out, use_container_width=True)
                    st.download_button("Download cluster CSV", out.to_csv(index=False).encode(), "clusters.csv","text/csv")

    except ImportError:
        st.info("Install required packages: pip3 install scikit-learn")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: TUMOR SUBTYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

with tab_subtype:
    st.markdown('<div class="section-header">Tumor Subtype Classification</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.82rem;color:#8b949e;margin-bottom:16px'>"
        "Classifies samples into molecular subtypes based on mutation profile similarity. "
        "Subtypes are derived by unsupervised clustering then labelled using dominant "
        "biomarker patterns (e.g. POLE-ultramutator, MSI-H, KRAS-driven, TP53-dominant). "
        "A Random Forest then learns to predict subtype from mutation features."
        "</div>",
        unsafe_allow_html=True
    )

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.decomposition import PCA
        from sklearn.model_selection import cross_val_score
        import warnings
        warnings.filterwarnings("ignore")

        col_st1, col_st2 = st.columns(2)
        with col_st1:
            n_subtypes = st.slider("Number of subtypes", 2, min(6, n_samples-1), 3, key="st_k")
        with col_st2:
            st_n_trees = st.slider("Random Forest trees", 50, 300, 100, step=50, key="st_trees")

        if st.button("Classify subtypes", type="primary", key="st_run"):
            with st.spinner("Classifying tumor subtypes..."):
                scaler = StandardScaler()
                X_sc   = scaler.fit_transform(feature_df.values)

                # Step 1: unsupervised clustering to derive subtype labels
                km     = KMeans(n_clusters=n_subtypes, random_state=random_seed, n_init=10)
                raw_cl = km.fit_predict(X_sc)

                # Step 2: assign biologically meaningful labels to clusters
                def label_cluster(cluster_id, feat_df, cluster_assignments):
                    members   = feat_df[cluster_assignments == cluster_id]
                    med_tmb   = members["TMB"].median()
                    med_indel = members["indel_fraction"].median()
                    pole_frac = members.get("mut_POLE", pd.Series(0)).mean() if "mut_POLE" in members.columns else 0
                    tp53_frac = members.get("mut_TP53",  pd.Series(0)).mean() if "mut_TP53"  in members.columns else 0
                    kras_frac = members.get("mut_KRAS",  pd.Series(0)).mean() if "mut_KRAS"  in members.columns else 0
                    egfr_frac = members.get("mut_EGFR",  pd.Series(0)).mean() if "mut_EGFR"  in members.columns else 0
                    mlh1_frac = members.get("mut_MLH1",  pd.Series(0)).mean() if "mut_MLH1"  in members.columns else 0

                    if med_tmb > 20 or pole_frac > 0.3:
                        return "Ultramutator (POLE)"
                    if med_tmb > 10 and med_indel > 0.15 and mlh1_frac > 0.1:
                        return "MSI-H / dMMR"
                    if med_tmb > 10:
                        return "TMB-High"
                    if kras_frac > 0.3:
                        return "KRAS-driven"
                    if egfr_frac > 0.2:
                        return "EGFR-driven"
                    if tp53_frac > 0.5:
                        return "TP53-dominant"
                    return f"Subtype {cluster_id + 1}"

                feat_vals = feature_df.values
                subtype_labels = np.array([label_cluster(ci, feature_df, raw_cl) for ci in raw_cl])

                # Step 3: train RF to predict subtype
                le  = LabelEncoder()
                y_e = le.fit_transform(subtype_labels)
                clf = RandomForestClassifier(n_estimators=st_n_trees, random_state=random_seed,
                                             class_weight="balanced", n_jobs=-1)
                clf.fit(X_sc, y_e)
                proba = clf.predict_proba(X_sc)

                # PCA for visualisation
                pca    = PCA(n_components=2, random_state=random_seed)
                coords = pca.fit_transform(X_sc)
                var_exp = pca.explained_variance_ratio_

                plot_df = pd.DataFrame({
                    "x":       coords[:,0],
                    "y":       coords[:,1],
                    "subtype": subtype_labels,
                    "sample":  feature_df.index,
                    "TMB":     feature_df["TMB"].values,
                })

                SUBTYPE_COLORS = {
                    "Ultramutator (POLE)": "#f0883e",
                    "MSI-H / dMMR":        "#3fb950",
                    "TMB-High":            "#58a6ff",
                    "KRAS-driven":         "#e3b341",
                    "EGFR-driven":         "#d2a8ff",
                    "TP53-dominant":       "#f85149",
                }

                fig_st = go.Figure()
                for stype in plot_df["subtype"].unique():
                    sub   = plot_df[plot_df["subtype"] == stype]
                    color = SUBTYPE_COLORS.get(stype, "#8b949e")
                    fig_st.add_trace(go.Scatter(
                        x=sub["x"], y=sub["y"], mode="markers",
                        name=stype,
                        marker=dict(size=9, color=color, line=dict(color="#0d1117",width=1)),
                        hovertemplate="<b>%{customdata}</b><br>Subtype: " + stype +
                                      "<br>TMB: %{text:.2f}<extra></extra>",
                        customdata=sub["sample"],
                        text=sub["TMB"],
                    ))
                fig_st.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                    font_family="IBM Plex Mono",
                    xaxis=dict(title=f"PC1 ({var_exp[0]*100:.1f}%)", color="#8b949e", gridcolor="#21262d"),
                    yaxis=dict(title=f"PC2 ({var_exp[1]*100:.1f}%)", color="#8b949e", gridcolor="#21262d"),
                    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
                    margin=dict(l=0,r=0,t=10,b=0), height=420,
                )
                st.plotly_chart(fig_st, use_container_width=True)

                # Subtype profiles
                st.markdown('<div class="section-header">Subtype Profiles</div>', unsafe_allow_html=True)
                plot_df["TMB"]          = feature_df["TMB"].values
                plot_df["indel_frac"]   = feature_df["indel_fraction"].values
                profile = plot_df.groupby("subtype").agg(
                    n=("sample","count"),
                    median_TMB=("TMB","median"),
                    mean_indel=("indel_frac","mean"),
                ).reset_index()
                profile.columns = ["Subtype","Samples","Median TMB","Mean Indel Frac"]
                st.dataframe(profile.sort_values("Median TMB", ascending=False), use_container_width=True)

                # Feature importance bar
                st.markdown('<div class="section-header">Top Predictive Features</div>', unsafe_allow_html=True)
                imp_df = pd.DataFrame({
                    "feature":    feature_df.columns,
                    "importance": clf.feature_importances_,
                }).sort_values("importance", ascending=False).head(15)

                fig_imp = go.Figure(go.Bar(
                    x=imp_df["importance"], y=imp_df["feature"],
                    orientation="h", marker_color="#58a6ff",
                    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
                ))
                fig_imp.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                    font_family="IBM Plex Mono",
                    xaxis=dict(title="Feature Importance", color="#8b949e", gridcolor="#21262d"),
                    yaxis=dict(autorange="reversed", color="#8b949e"),
                    margin=dict(l=0,r=0,t=10,b=0), height=380,
                )
                st.plotly_chart(fig_imp, use_container_width=True)

                out = plot_df[["sample","subtype","TMB","indel_frac"]].rename(
                    columns={"sample":"sample_id","indel_frac":"indel_fraction"})
                with st.expander("Full subtype assignments"):
                    st.dataframe(out.reset_index(drop=True), use_container_width=True)
                    st.download_button("Download subtypes CSV", out.to_csv(index=False).encode(), "subtypes.csv","text/csv")

    except ImportError:
        st.info("Install required packages: pip3 install scikit-learn")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: EXPLAINABLE AI
# ═══════════════════════════════════════════════════════════════════════════════

with tab_xai:
    st.markdown('<div class="section-header">Explainable AI -- SHAP Feature Importance</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.82rem;color:#8b949e;margin-bottom:16px'>"
        "Uses SHAP (SHapley Additive exPlanations) to explain which features drive each "
        "prediction. SHAP values measure the marginal contribution of each feature to the "
        "model output, enabling both global (population-level) and local (per-patient) explanations."
        "</div>",
        unsafe_allow_html=True
    )

    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import warnings
        warnings.filterwarnings("ignore")

        col_x1, col_x2 = st.columns(2)
        with col_x1:
            xai_target = st.selectbox(
                "Explain model for",
                ["Immunotherapy Response","TMB-High vs Low"],
                key="xai_target"
            )
        with col_x2:
            n_shap_samples = st.slider("Samples to explain", 5, min(50, n_samples), min(20, n_samples), key="xai_n")

        if st.button("Compute SHAP values", type="primary", key="xai_run"):
            with st.spinner("Computing SHAP values (this may take 20-60 seconds)..."):
                scaler = StandardScaler()
                X_sc   = scaler.fit_transform(feature_df.values)

                if xai_target == "Immunotherapy Response":
                    mmr_genes = ["mut_MLH1","mut_MSH2","mut_MSH6","mut_PMS2","mut_POLE"]
                    mmr_cols  = [c for c in mmr_genes if c in feature_df.columns]
                    mmr_hit   = feature_df[mmr_cols].sum(axis=1) >= 1 if mmr_cols else pd.Series(0, index=feature_df.index)
                    y = ((feature_df["TMB"] >= 10) | (feature_df["indel_fraction"] >= 0.15) | mmr_hit).astype(int).values
                else:
                    y = (feature_df["TMB"] >= 10).astype(int).values

                clf = RandomForestClassifier(n_estimators=100, random_state=random_seed,
                                             class_weight="balanced", n_jobs=-1)
                clf.fit(X_sc, y)

                explainer   = shap.TreeExplainer(clf)
                X_explain   = X_sc[:n_shap_samples]
                shap_values = explainer.shap_values(X_explain)

                # Handle all SHAP output formats:
                # list of arrays (old sklearn RF), 3D ndarray (new shap), 2D ndarray (binary)
                if isinstance(shap_values, list):
                    sv = np.array(shap_values[1])
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    sv = shap_values[:, :, 1]
                else:
                    sv = np.array(shap_values)
                if sv.ndim == 1:
                    sv = sv.reshape(1, -1)

                feat_names  = list(feature_df.columns)
                mean_abs    = np.abs(sv).mean(axis=0).ravel()
                shap_df     = pd.DataFrame({
                    "feature":    feat_names,
                    "mean_shap":  mean_abs,
                }).sort_values("mean_shap", ascending=False).head(20)

                # Global feature importance
                st.markdown('<div class="section-header">Global Feature Importance (mean |SHAP|)</div>', unsafe_allow_html=True)
                fig_shap = go.Figure(go.Bar(
                    x=shap_df["mean_shap"],
                    y=shap_df["feature"],
                    orientation="h",
                    marker=dict(
                        color=shap_df["mean_shap"],
                        colorscale=[[0,"#21262d"],[0.5,"#58a6ff"],[1,"#f0883e"]],
                        showscale=False,
                    ),
                    hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
                ))
                fig_shap.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                    font_family="IBM Plex Mono",
                    xaxis=dict(title="Mean |SHAP value|", color="#8b949e", gridcolor="#21262d"),
                    yaxis=dict(autorange="reversed", color="#8b949e"),
                    margin=dict(l=0,r=0,t=10,b=0), height=440,
                )
                st.plotly_chart(fig_shap, use_container_width=True)

                # SHAP beeswarm (dot plot per feature per sample)
                st.markdown('<div class="section-header">SHAP Value Distribution (Beeswarm)</div>', unsafe_allow_html=True)
                top_feats = shap_df["feature"].head(10).tolist()
                top_idx   = [feat_names.index(f) for f in top_feats]
                sv_top    = sv[:, top_idx]
                x_top     = X_explain[:, top_idx]

                fig_bee = go.Figure()
                for fi, feat in enumerate(top_feats):
                    shap_col = sv_top[:, fi]
                    feat_col = x_top[:, fi]
                    fig_bee.add_trace(go.Scatter(
                        x=shap_col,
                        y=[feat] * len(shap_col),
                        mode="markers",
                        marker=dict(
                            size=7,
                            color=feat_col,
                            colorscale=[[0,"#58a6ff"],[1,"#f85149"]],
                            opacity=0.7,
                            line=dict(color="#0d1117",width=0.5),
                        ),
                        hovertemplate=f"<b>{feat}</b><br>SHAP: %{{x:.4f}}<br>Feature value: %{{text:.3f}}<extra></extra>",
                        text=feat_col,
                        showlegend=False,
                    ))
                fig_bee.add_vline(x=0, line_dash="dash", line_color="#30363d")
                fig_bee.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                    font_family="IBM Plex Mono",
                    xaxis=dict(title="SHAP value", color="#8b949e", gridcolor="#21262d"),
                    yaxis=dict(color="#8b949e", tickfont=dict(size=11)),
                    margin=dict(l=0,r=0,t=10,b=0), height=380,
                )
                st.plotly_chart(fig_bee, use_container_width=True)
                st.markdown(
                    "<div style='font-size:0.72rem;color:#8b949e'>"
                    "Blue = low feature value | Red = high feature value | "
                    "Right of 0 = pushes toward positive class</div>",
                    unsafe_allow_html=True
                )

                # Per-sample waterfall (first sample)
                st.markdown('<div class="section-header">Per-Sample Explanation (Sample 0)</div>', unsafe_allow_html=True)
                s0_shap = sv[0]
                s0_df   = pd.DataFrame({
                    "feature": feat_names,
                    "shap":    s0_shap,
                }).sort_values("shap", key=abs, ascending=False).head(15)

                fig_wf = go.Figure(go.Bar(
                    x=s0_df["shap"],
                    y=s0_df["feature"],
                    orientation="h",
                    marker_color=["#3fb950" if v > 0 else "#f85149" for v in s0_df["shap"]],
                    hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
                ))
                fig_wf.add_vline(x=0, line_color="#30363d")
                fig_wf.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                    font_family="IBM Plex Mono",
                    title=dict(
                        text=f"Explanation for: {feature_df.index[0]}",
                        font=dict(size=11, color="#8b949e")
                    ),
                    xaxis=dict(title="SHAP value", color="#8b949e", gridcolor="#21262d"),
                    yaxis=dict(autorange="reversed", color="#8b949e"),
                    margin=dict(l=0,r=0,t=40,b=0), height=400,
                )
                st.plotly_chart(fig_wf, use_container_width=True)
                st.markdown(
                    "<div style='font-size:0.72rem;color:#8b949e'>"
                    "Green = pushes toward responder | Red = pushes toward non-responder</div>",
                    unsafe_allow_html=True
                )

    except ImportError:
        st.markdown(
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;"
            "padding:16px 20px;font-size:0.85rem;color:#8b949e'>"
            "<b style='color:#e3b341'>SHAP not installed.</b><br>"
            "Install it with: <code style='color:#58a6ff'>pip3 install shap</code><br><br>"
            "SHAP (SHapley Additive exPlanations) provides model-agnostic explanations "
            "that work with any ML model. It decomposes each prediction into feature contributions "
            "based on Shapley values from cooperative game theory."
            "</div>",
            unsafe_allow_html=True
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: NEOANTIGEN PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

with tab_neoag:
    st.markdown('<div class="section-header">Neoantigen Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.82rem;color:#8b949e;margin-bottom:16px'>"
        "Estimates neoantigen burden per sample using missense mutation count, "
        "protein-level changes, and a hydrophobicity-based binding affinity proxy. "
        "True neoantigen prediction requires peptide-MHC binding tools such as NetMHCpan. "
        "This module provides a fast computational approximation based on available MAF data."
        "</div>",
        unsafe_allow_html=True
    )

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        hla_allele = st.selectbox(
            "HLA allele (population frequency estimate)",
            list(HLA_SUPERTYPES.keys()),
            key="neo_hla"
        )
    with col_n2:
        min_depth = st.slider("Min read depth filter", 0, 50, 10, key="neo_depth")
        vaf_min   = st.slider("Min variant allele frequency", 0.0, 0.5, 0.05, step=0.01, key="neo_vaf")

    if st.button("Estimate neoantigen burden", type="primary", key="neo_run"):
        with st.spinner("Estimating neoantigen burden..."):

            # Filter to missense mutations (most likely to generate neoantigens)
            df_missense = mut_df[
                (mut_df[variant_col] == "Missense_Mutation")
            ].copy()

            # Apply depth filter if column available
            if "t_depth" in df_missense.columns:
                df_missense = df_missense[
                    pd.to_numeric(df_missense["t_depth"], errors="coerce").fillna(0) >= min_depth
                ]

            # Apply VAF filter if possible
            if "t_alt_count" in df_missense.columns and "t_depth" in df_missense.columns:
                t_alt   = pd.to_numeric(df_missense["t_alt_count"], errors="coerce").fillna(0)
                t_depth = pd.to_numeric(df_missense["t_depth"],     errors="coerce").fillna(1)
                vaf     = t_alt / t_depth.replace(0, 1)
                df_missense = df_missense[vaf >= vaf_min]

            # Neoantigen score per mutation:
            # Proxy for immunogenicity = protein change hydrophobicity difference
            # (mutant peptide more hydrophobic = better MHC binding proxy)
            def hydro_score(hgvsp: str) -> float:
                if not isinstance(hgvsp, str) or len(hgvsp) < 4:
                    return 0.5
                try:
                    # Format: p.X123Y or p.Xxx123Yyy
                    clean = hgvsp.replace("p.","")
                    # Single-letter variant
                    if len(clean) >= 3 and clean[0].isupper() and clean[-1].isupper():
                        ref_aa = clean[0]
                        mut_aa = clean[-1]
                        ref_h  = AA_HYDROPHOBICITY.get(ref_aa, 0.0)
                        mut_h  = AA_HYDROPHOBICITY.get(mut_aa, 0.0)
                        delta  = mut_h - ref_h
                        # Normalise to 0-1 range
                        return min(max((delta + 9) / 18, 0.0), 1.0)
                except Exception:
                    pass
                return 0.5

            hla_freq = HLA_SUPERTYPES[hla_allele]["freq"]

            if hgvsp_col and hgvsp_col in df_missense.columns:
                df_missense["neo_score"] = df_missense[hgvsp_col].apply(hydro_score)
            else:
                df_missense["neo_score"] = 0.5

            # Adjust by HLA frequency (common alleles present neoantigens to more patients)
            df_missense["neo_score"] *= hla_freq

            # Aggregate per sample
            neo_agg = df_missense.groupby(sample_col).agg(
                n_missense=        (variant_col, "count"),
                mean_neo_score=    ("neo_score",  "mean"),
                sum_neo_score=     ("neo_score",  "sum"),
                top_gene=          (gene_col,     lambda x: x.value_counts().index[0] if len(x) > 0 else ""),
            ).reset_index().rename(columns={sample_col:"sample_id"})

            neo_agg["neo_score_norm"] = (neo_agg["sum_neo_score"] / neo_agg["sum_neo_score"].max()).round(3)
            neo_agg["tier"] = pd.cut(
                neo_agg["neo_score_norm"],
                bins=[0, 0.33, 0.66, 1.0],
                labels=["Low","Medium","High"],
                include_lowest=True
            )
            neo_agg = neo_agg.sort_values("neo_score_norm", ascending=False)

            # Summary metrics
            nc1, nc2, nc3 = st.columns(3)
            for col_w, val, label in zip(
                [nc1, nc2, nc3],
                [len(neo_agg), round(neo_agg["n_missense"].mean(), 1), round(neo_agg["neo_score_norm"].median(), 3)],
                ["Samples Analysed","Mean Missense Mutations","Median Neo Score"]
            ):
                col_w.markdown(
                    f"<div class='stat-card'><div class='stat-value'>{val}</div>"
                    f"<div class='stat-label'>{label}</div></div>",
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Neo score bar
            tier_colors = neo_agg["tier"].astype(str).map(
                {"Low":"#58a6ff","Medium":"#e3b341","High":"#f85149"}
            ).tolist()
            fig_neo = go.Figure(go.Bar(
                x=neo_agg["sample_id"].str[:16],
                y=neo_agg["neo_score_norm"],
                marker_color=tier_colors,
                hovertemplate=(
                    "<b>%{x}</b><br>Neo score: %{y:.3f}<br>"
                    "Missense muts: %{customdata[0]}<br>"
                    "Top gene: %{customdata[1]}<extra></extra>"
                ),
                customdata=neo_agg[["n_missense","top_gene"]].values,
            ))
            fig_neo.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                font_family="IBM Plex Mono",
                xaxis=dict(showticklabels=False, title="Samples (sorted by neo score)", color="#8b949e"),
                yaxis=dict(title="Neoantigen Score (normalised)", color="#8b949e", gridcolor="#21262d"),
                margin=dict(l=0,r=0,t=10,b=0), height=300,
            )
            st.plotly_chart(fig_neo, use_container_width=True)

            # Neo score vs missense scatter
            st.markdown('<div class="section-header">Missense Mutations vs Neoantigen Score</div>', unsafe_allow_html=True)
            fig_ns = go.Figure(go.Scatter(
                x=neo_agg["n_missense"],
                y=neo_agg["neo_score_norm"],
                mode="markers",
                marker=dict(
                    size=9,
                    color=neo_agg["neo_score_norm"],
                    colorscale=[[0,"#161b22"],[0.5,"#58a6ff"],[1,"#f85149"]],
                    showscale=True,
                    colorbar=dict(title="Neo Score", tickfont=dict(color="#8b949e"), title_font=dict(color="#8b949e")),
                    line=dict(color="#0d1117",width=1),
                ),
                hovertemplate="<b>%{customdata}</b><br>Missense: %{x}<br>Neo score: %{y:.3f}<extra></extra>",
                customdata=neo_agg["sample_id"],
            ))
            fig_ns.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                font_family="IBM Plex Mono",
                xaxis=dict(title="Number of Missense Mutations", color="#8b949e", gridcolor="#21262d"),
                yaxis=dict(title="Neoantigen Score", color="#8b949e", gridcolor="#21262d"),
                margin=dict(l=0,r=0,t=10,b=0), height=320,
            )
            st.plotly_chart(fig_ns, use_container_width=True)

            with st.expander("Full neoantigen table"):
                st.dataframe(neo_agg.reset_index(drop=True), use_container_width=True)
                st.download_button(
                    "Download neoantigen CSV",
                    neo_agg.to_csv(index=False).encode(),
                    "neoantigen_scores.csv","text/csv"
                )

            st.markdown(
                "<div style='margin-top:16px;padding:12px 16px;background:#161b22;"
                "border:1px solid #30363d;border-radius:8px;font-size:0.75rem;color:#8b949e'>"
                "<b>Note:</b> This is a computational approximation using hydrophobicity as a proxy for "
                "MHC-I binding affinity. For clinical or research-grade neoantigen prediction, use "
                "dedicated tools such as NetMHCpan, pVACseq, or neoepiscope with phased VCF input "
                "and patient-specific HLA typing."
                "</div>",
                unsafe_allow_html=True
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: MULTI-OMICS INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

with tab_multiomics:
    st.markdown('<div class="section-header">Multi-omics Integration</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.82rem;color:#8b949e;margin-bottom:16px'>"
        "Integrates mutation data with additional omics layers. "
        "Upload RNA expression, CNV, or methylation data below and the page will "
        "align them with your mutation matrix by sample ID, then run joint dimensionality "
        "reduction and correlation analysis across layers."
        "</div>",
        unsafe_allow_html=True
    )

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.cross_decomposition import CCA
        import warnings
        warnings.filterwarnings("ignore")

        # ── Omics uploader ─────────────────────────────────────────────────────

        st.markdown('<div class="section-header">Upload Additional Omics Layer</div>', unsafe_allow_html=True)

        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            omics_file = st.file_uploader(
                "Upload omics CSV (rows = samples, columns = features, first column = sample_id)",
                type=["csv","tsv","txt"],
                key="mo_upload"
            )
        with col_u2:
            omics_type = st.selectbox(
                "Omics layer type",
                ["RNA Expression","Copy Number Variation","Methylation","Proteomics","Other"],
                key="mo_type"
            )

        # ── Simulate omics data if none uploaded ───────────────────────────────

        st.markdown('<div class="section-header">Integration Analysis</div>', unsafe_allow_html=True)

        if omics_file is None:
            st.info(
                "No omics file uploaded. Showing a simulated RNA expression layer "
                "derived from mutation features as a demonstration. "
                "Upload a real CSV to run integration on your data."
            )
            # Simulate RNA-seq-like data correlated with mutation features
            np.random.seed(random_seed)
            rna_features = [f"GENE_{i:04d}" for i in range(50)]
            rna_matrix   = np.random.randn(n_samples, 50)

            # Make first 10 features correlated with TMB
            tmb_vals = feature_df["TMB"].values
            for i in range(10):
                rna_matrix[:, i] += tmb_vals * 0.3 + np.random.randn(n_samples) * 0.5

            omics_df = pd.DataFrame(
                rna_matrix,
                index=feature_df.index,
                columns=rna_features
            )
            omics_label = "Simulated RNA Expression"
        else:
            sep = "\t" if omics_file.name.endswith((".tsv",".txt")) else ","
            raw_omics = pd.read_csv(omics_file, sep=sep, index_col=0)

            # Align to mutation samples
            common = feature_df.index.intersection(raw_omics.index)
            if len(common) == 0:
                st.error(
                    f"No matching sample IDs between mutation data and uploaded omics file. "
                    f"Mutation samples look like: {list(feature_df.index[:3])}. "
                    f"Your file has: {list(raw_omics.index[:3])}"
                )
                st.stop()

            omics_df    = raw_omics.loc[common]
            omics_label = omics_type
            st.success(f"Aligned {len(common)} samples across mutation and {omics_type} data.")

        # ── Joint PCA ──────────────────────────────────────────────────────────

        st.markdown('<div class="section-header">Joint Dimensionality Reduction (PCA)</div>', unsafe_allow_html=True)

        mut_common  = feature_df.loc[omics_df.index]
        scaler_mut  = StandardScaler()
        scaler_oms  = StandardScaler()
        X_mut = scaler_mut.fit_transform(mut_common.fillna(0).values)
        X_oms = scaler_oms.fit_transform(omics_df.fillna(0).values)

        # PCA on each layer separately
        pca_mut = PCA(n_components=min(5, X_mut.shape[1]), random_state=random_seed)
        pca_oms = PCA(n_components=min(5, X_oms.shape[1]), random_state=random_seed)
        pc_mut  = pca_mut.fit_transform(X_mut)
        pc_oms  = pca_oms.fit_transform(X_oms)

        # Concatenate for joint embedding
        X_joint  = np.hstack([pc_mut, pc_oms])
        pca_joint = PCA(n_components=2, random_state=random_seed)
        coords    = pca_joint.fit_transform(X_joint)
        var_exp   = pca_joint.explained_variance_ratio_

        tmb_colors = mut_common["TMB"].values
        fig_joint  = go.Figure(go.Scatter(
            x=coords[:,0], y=coords[:,1],
            mode="markers",
            marker=dict(
                size=9,
                color=tmb_colors,
                colorscale=[[0,"#161b22"],[0.3,"#58a6ff"],[0.7,"#e3b341"],[1,"#f85149"]],
                showscale=True,
                colorbar=dict(title="TMB", tickfont=dict(color="#8b949e"), title_font=dict(color="#8b949e")),
                line=dict(color="#0d1117",width=1),
            ),
            hovertemplate="<b>%{customdata}</b><br>TMB: %{text:.2f}<extra></extra>",
            customdata=omics_df.index,
            text=tmb_colors,
        ))
        fig_joint.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
            font_family="IBM Plex Mono",
            title=dict(text=f"Joint PCA: Mutation + {omics_label}", font=dict(size=12,color="#8b949e")),
            xaxis=dict(title=f"PC1 ({var_exp[0]*100:.1f}%)", color="#8b949e", gridcolor="#21262d"),
            yaxis=dict(title=f"PC2 ({var_exp[1]*100:.1f}%)", color="#8b949e", gridcolor="#21262d"),
            margin=dict(l=0,r=0,t=40,b=0), height=400,
        )
        st.plotly_chart(fig_joint, use_container_width=True)

        # ── Cross-layer correlation ────────────────────────────────────────────

        st.markdown('<div class="section-header">Cross-Layer Correlation</div>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.78rem;color:#8b949e;margin-bottom:12px'>"
            "Pearson correlation between mutation features (TMB, indel fraction, "
            "top driver gene flags) and the first principal component of the omics layer.</div>",
            unsafe_allow_html=True
        )

        omics_pc1 = pc_oms[:, 0]
        corr_feats = ["TMB","indel_fraction","pct_missense","pct_nonsense","pct_frameshift","pct_splice"]
        corr_feats += [f"mut_{g}" for g in ["TP53","KRAS","EGFR","BRAF","PIK3CA","PTEN","APC","POLE"]]
        corr_feats  = [f for f in corr_feats if f in mut_common.columns]

        corr_rows = []
        for feat in corr_feats:
            vals = mut_common[feat].fillna(0).values
            if vals.std() > 0:
                r = np.corrcoef(vals, omics_pc1)[0, 1]
                corr_rows.append({"feature": feat, "correlation": round(r, 3)})

        if corr_rows:
            corr_df = pd.DataFrame(corr_rows).sort_values("correlation", key=abs, ascending=False)
            fig_corr = go.Figure(go.Bar(
                x=corr_df["correlation"],
                y=corr_df["feature"],
                orientation="h",
                marker_color=["#3fb950" if v > 0 else "#f85149" for v in corr_df["correlation"]],
                hovertemplate="<b>%{y}</b><br>r = %{x:.3f}<extra></extra>",
            ))
            fig_corr.add_vline(x=0, line_color="#30363d")
            fig_corr.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
                font_family="IBM Plex Mono",
                xaxis=dict(title=f"Pearson r with {omics_label} PC1", color="#8b949e", gridcolor="#21262d"),
                yaxis=dict(autorange="reversed", color="#8b949e"),
                margin=dict(l=0,r=0,t=10,b=0), height=400,
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # ── Variance explained ─────────────────────────────────────────────────

        st.markdown('<div class="section-header">Variance Explained per Layer</div>', unsafe_allow_html=True)
        layers = ["Mutation","Omics","Joint"]
        var_vals = [
            pca_mut.explained_variance_ratio_[:2].sum() * 100,
            pca_oms.explained_variance_ratio_[:2].sum() * 100,
            var_exp[:2].sum() * 100,
        ]
        fig_ve = go.Figure(go.Bar(
            x=layers, y=var_vals,
            marker_color=["#58a6ff","#3fb950","#f0883e"],
            hovertemplate="<b>%{x}</b><br>Variance explained (PC1+2): %{y:.1f}%<extra></extra>",
        ))
        fig_ve.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
            font_family="IBM Plex Mono",
            xaxis=dict(title="Omics Layer", color="#8b949e"),
            yaxis=dict(title="% Variance Explained (PC1+PC2)", color="#8b949e", gridcolor="#21262d"),
            margin=dict(l=0,r=0,t=10,b=0), height=260,
        )
        st.plotly_chart(fig_ve, use_container_width=True)

        # ── Export ─────────────────────────────────────────────────────────────

        joint_out = pd.DataFrame({
            "sample_id":   omics_df.index,
            "joint_PC1":   coords[:,0],
            "joint_PC2":   coords[:,1],
            "mutation_PC1":pc_mut[:,0],
            "omics_PC1":   pc_oms[:,0],
            "TMB":         mut_common["TMB"].values,
        })
        with st.expander("Download integration results"):
            st.dataframe(joint_out.reset_index(drop=True), use_container_width=True)
            st.download_button(
                "Download integration CSV",
                joint_out.to_csv(index=False).encode(),
                "multiomics_integration.csv","text/csv"
            )

    except ImportError:
        st.info("Install required packages: pip3 install scikit-learn")