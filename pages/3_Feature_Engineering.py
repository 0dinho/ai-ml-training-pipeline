"""Feature Engineering page — optional preprocessing step before training.

Supports two modes:
  1. Preprocessing step: Apply dimensionality reduction BEFORE training
     (output replaces X_train/X_val/X_test via the _fe session keys).
  2. Visualization only: Run reduction for visualization without feeding
     into training (used for t-SNE and optionally UMAP/NMF).

Techniques available (tabbed layout):
  - PCA  (with optional variance-threshold mode + scree plot)
  - LDA  (supervised tasks only)
  - NMF  (non-negative data only)
  - t-SNE (visualization only — cannot transform new data)
  - UMAP (requires the `umap-learn` package)
  - Feature Transforms (polynomial, log/sqrt, quantile binning)

This page is **optional** — skipping it passes X_train unchanged to Training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Feature Engineering", page_icon="⚙️", layout="wide")

# ── Theme constants ───────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]

SUPERVISED_TASKS = {"binary_classification", "multiclass_classification", "regression"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Feature Engineering Status")
    if st.session_state.get("feature_engineering_applied"):
        orig_n = st.session_state.get("_fe_orig_n_features", "?")
        new_n = len(st.session_state.get("feature_names_fe", []))
        technique = st.session_state.get("fe_technique", "unknown")
        st.success(f"Applied ({technique}): {orig_n} → {new_n} features")
    else:
        st.info("No feature engineering applied yet.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚙️ Feature Engineering")
st.caption("Optional — apply dimensionality reduction or feature transforms before training.")

# ── Guard clause ─────────────────────────────────────────────────────────────
if "X_train" not in st.session_state:
    st.warning("Please complete the **Preprocessing** step first.")
    st.stop()

X_train: np.ndarray = st.session_state["X_train"]
X_val: np.ndarray | None = st.session_state.get("X_val")
X_test: np.ndarray | None = st.session_state.get("X_test")
feature_names: list[str] = st.session_state.get(
    "feature_names", [f"f{i}" for i in range(X_train.shape[1])]
)
task_type: str = st.session_state.get("task_type", "binary_classification")
y_train = st.session_state.get("y_train")

n_samples, n_features = X_train.shape
is_supervised_task = task_type in SUPERVISED_TASKS

st.metric("Current feature count", n_features)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: 2D / 3D scatter of reduced data
# ═══════════════════════════════════════════════════════════════════════════════

def _scatter_reduced(
    X_reduced: np.ndarray,
    labels: np.ndarray | None,
    title: str,
    label_name: str = "class",
    dim_names: list[str] | None = None,
) -> go.Figure:
    """Return a Plotly 2D or 3D scatter figure for reduced data."""
    n_dims = X_reduced.shape[1]
    names = dim_names or [f"Dim {i+1}" for i in range(n_dims)]

    if labels is not None:
        labels_str = labels.astype(str)
        unique_labels = np.unique(labels_str)
        color_map = {lbl: COLORS[i % len(COLORS)] for i, lbl in enumerate(unique_labels)}
        colors_arr = [color_map[l] for l in labels_str]
    else:
        colors_arr = COLORS[0]
        labels_str = None

    if n_dims >= 3:
        fig = go.Figure(
            data=go.Scatter3d(
                x=X_reduced[:, 0],
                y=X_reduced[:, 1],
                z=X_reduced[:, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    color=colors_arr if isinstance(colors_arr, list) else COLORS[0],
                    opacity=0.8,
                ),
                text=labels_str,
                name="",
            )
        )
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=names[0],
                yaxis_title=names[1],
                zaxis_title=names[2],
            ),
            **PLOTLY_LAYOUT,
        )
    else:
        if labels is not None:
            df_scatter = pd.DataFrame(
                {names[0]: X_reduced[:, 0], names[1]: X_reduced[:, 1], label_name: labels_str}
            )
            fig = px.scatter(
                df_scatter, x=names[0], y=names[1], color=label_name,
                color_discrete_sequence=COLORS, title=title,
            )
        else:
            df_scatter = pd.DataFrame({names[0]: X_reduced[:, 0], names[1]: X_reduced[:, 1]})
            fig = px.scatter(
                df_scatter, x=names[0], y=names[1],
                color_discrete_sequence=COLORS, title=title,
            )
        fig.update_layout(**PLOTLY_LAYOUT)

    return fig


def _project_to_2d_for_display(X: np.ndarray) -> np.ndarray:
    """Project high-dimensional data to 2D via PCA for display purposes."""
    from sklearn.decomposition import PCA as _PCA
    pca_disp = _PCA(n_components=2)
    return pca_disp.fit_transform(X)


def _store_fe_result(
    X_tr_new: np.ndarray,
    X_vl_new: np.ndarray | None,
    X_te_new: np.ndarray | None,
    new_feature_names: list[str],
    technique: str,
    transformer,
) -> None:
    """Persist feature-engineered arrays and metadata into session state."""
    st.session_state["_fe_orig_n_features"] = n_features
    st.session_state["X_train_fe"] = X_tr_new
    st.session_state["X_val_fe"] = X_vl_new
    st.session_state["X_test_fe"] = X_te_new
    st.session_state["feature_names_fe"] = new_feature_names
    st.session_state["feature_engineering_applied"] = True
    st.session_state["fe_transformer"] = transformer
    st.session_state["fe_technique"] = technique


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Dimensionality Reduction Techniques
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Dimensionality Reduction")

tab_pca, tab_lda, tab_cca, tab_nmf, tab_tsne, tab_umap, tab_transforms = st.tabs(
    ["PCA", "LDA", "CCA", "NMF", "t-SNE", "UMAP", "Feature Transforms"]
)


# ─────────────────────────────────────────────────────────────────────────────
# PCA Tab
# ─────────────────────────────────────────────────────────────────────────────
with tab_pca:
    st.subheader("Principal Component Analysis")

    max_components_pca = min(n_features, n_samples - 1, 50)
    default_components_pca = min(10, n_features)

    use_var_threshold = st.checkbox(
        "Use explained variance threshold instead of fixed n_components",
        key="pca_use_var_threshold",
    )

    if use_var_threshold:
        var_threshold = st.slider(
            "Explained variance threshold",
            min_value=0.80, max_value=0.99, value=0.95, step=0.01,
            key="pca_var_threshold",
            help="Find the smallest n_components that explains at least this fraction of variance.",
        )
        pca_n_components_display = None  # determined after fitting
    else:
        pca_n_components = st.slider(
            "n_components",
            min_value=1, max_value=max_components_pca, value=default_components_pca,
            key="pca_n_components",
        )

    pca_mode = st.radio(
        "Mode",
        ["Use as preprocessing step", "Visualization only"],
        key="pca_mode",
        horizontal=True,
    )

    if st.button("Apply PCA", type="primary", key="btn_pca"):
        with st.spinner("Fitting PCA..."):
            try:
                from sklearn.decomposition import PCA

                X_tr_f = X_train.astype(float)

                if use_var_threshold:
                    # Fit full PCA first, then find cutoff
                    n_fit = min(n_features, n_samples - 1)
                    pca_full = PCA(n_components=n_fit)
                    pca_full.fit(X_tr_f)
                    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
                    n_keep = int(np.searchsorted(cumvar, var_threshold) + 1)
                    n_keep = max(1, min(n_keep, n_fit))
                    threshold_val = st.session_state.get("pca_var_threshold", 0.95)
                    st.info(
                        f"Variance threshold {threshold_val:.0%} → keeping **{n_keep}** components "
                        f"(explains {cumvar[n_keep - 1]:.2%} variance)."
                    )
                    pca = PCA(n_components=n_keep)
                else:
                    n_keep = st.session_state.get("pca_n_components", default_components_pca)
                    # Also fit full PCA for scree plot
                    n_fit = min(n_features, n_samples - 1)
                    pca_full = PCA(n_components=n_fit)
                    pca_full.fit(X_tr_f)
                    pca = PCA(n_components=n_keep)

                pca.fit(X_tr_f)
                X_tr_pca = pca.transform(X_tr_f)
                X_vl_pca = pca.transform(X_val.astype(float)) if X_val is not None else None
                X_te_pca = pca.transform(X_test.astype(float)) if X_test is not None else None

                pca_feat_names = [f"PC{i+1}" for i in range(n_keep)]

                # ── Scree plot ────────────────────────────────────────────────
                evr = pca_full.explained_variance_ratio_
                n_plot = min(len(evr), 30)
                cumvar_plot = np.cumsum(evr[:n_plot])
                comp_labels = [f"PC{i+1}" for i in range(n_plot)]

                fig_scree = go.Figure()
                fig_scree.add_bar(
                    x=comp_labels,
                    y=evr[:n_plot],
                    name="Individual variance",
                    marker_color=COLORS[1],
                )
                fig_scree.add_scatter(
                    x=comp_labels,
                    y=cumvar_plot,
                    mode="lines+markers",
                    name="Cumulative variance",
                    line=dict(color=COLORS[0], width=2),
                    yaxis="y2",
                )
                fig_scree.update_layout(
                    title="Scree Plot",
                    xaxis_title="Component",
                    yaxis_title="Explained Variance Ratio",
                    yaxis2=dict(
                        title="Cumulative Variance",
                        overlaying="y",
                        side="right",
                        range=[0, 1.05],
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_scree, width='stretch')

                # ── Component loadings table ──────────────────────────────────
                loadings = pd.DataFrame(
                    pca.components_.T,
                    index=feature_names[:n_features],
                    columns=pca_feat_names,
                )
                with st.expander("Component loadings (top features per component)", expanded=False):
                    n_top = min(10, n_features)
                    rows = []
                    for comp_col in pca_feat_names[:min(5, n_keep)]:
                        top_feats = loadings[comp_col].abs().nlargest(n_top).index.tolist()
                        for feat in top_feats:
                            rows.append({
                                "Component": comp_col,
                                "Feature": feat,
                                "Loading": round(loadings.loc[feat, comp_col], 4),
                            })
                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

                if pca_mode == "Use as preprocessing step":
                    _store_fe_result(X_tr_pca, X_vl_pca, X_te_pca, pca_feat_names, "PCA", pca)
                    st.success(
                        f"PCA applied as preprocessing step! "
                        f"Features: {n_features} → {n_keep}"
                    )
                else:
                    st.session_state["reduction_output"] = X_tr_pca
                    st.success(f"PCA visualization computed ({n_keep} components).")

                st.session_state["_pca_loadings"] = loadings
                st.rerun()

            except Exception as exc:
                st.error(f"PCA failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# LDA Tab
# ─────────────────────────────────────────────────────────────────────────────
with tab_lda:
    st.subheader("Linear Discriminant Analysis")

    if task_type not in ("binary_classification", "multiclass_classification"):
        st.warning(
            "LDA is only available for **binary** or **multiclass classification** tasks. "
            f"Current task: `{task_type}`."
        )
    elif y_train is None:
        st.warning("y_train is not available in session state. Please re-run Preprocessing.")
    else:
        n_classes = len(np.unique(y_train))
        max_lda_components = min(n_classes - 1, n_features)

        if max_lda_components < 1:
            st.warning("Not enough classes to run LDA (need at least 2 classes).")
        else:
            if max_lda_components == 1:
                lda_n_components = 1
                st.info("Only 2 classes detected → LDA will produce 1 component.")
            else:
                lda_n_components = st.slider(
                    "n_components",
                    min_value=1,
                    max_value=max_lda_components,
                    value=min(2, max_lda_components),
                    key="lda_n_components",
                    help=f"Max {max_lda_components} (= n_classes - 1).",
                )

            st.info("LDA always acts as a **preprocessing step** (transforms train/val/test).")

            if st.button("Apply LDA", type="primary", key="btn_lda"):
                with st.spinner("Fitting LDA..."):
                    try:
                        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

                        lda = LinearDiscriminantAnalysis(n_components=lda_n_components)
                        X_tr_f = X_train.astype(float)
                        lda.fit(X_tr_f, y_train)

                        X_tr_lda = lda.transform(X_tr_f)
                        X_vl_lda = lda.transform(X_val.astype(float)) if X_val is not None else None
                        X_te_lda = lda.transform(X_test.astype(float)) if X_test is not None else None

                        lda_feat_names = [f"LD{i+1}" for i in range(lda_n_components)]

                        # ── 2D scatter of LDA projection ──────────────────────
                        if lda_n_components >= 2:
                            fig_lda = _scatter_reduced(
                                X_tr_lda[:, :2],
                                y_train,
                                title="LDA 2D Projection (training set)",
                                label_name="class",
                                dim_names=["LD1", "LD2"],
                            )
                            st.plotly_chart(fig_lda, width='stretch')
                        elif lda_n_components == 1:
                            df_lda1d = pd.DataFrame({"LD1": X_tr_lda[:, 0], "class": y_train.astype(str)})
                            fig_lda = px.histogram(
                                df_lda1d, x="LD1", color="class",
                                barmode="overlay",
                                color_discrete_sequence=COLORS,
                                title="LDA 1D Projection (training set)",
                            )
                            fig_lda.update_layout(**PLOTLY_LAYOUT)
                            st.plotly_chart(fig_lda, width='stretch')

                        _store_fe_result(
                            X_tr_lda, X_vl_lda, X_te_lda,
                            lda_feat_names, "LDA", lda,
                        )
                        st.success(
                            f"LDA applied! Features: {n_features} → {lda_n_components}"
                        )
                        st.rerun()

                    except Exception as exc:
                        st.error(f"LDA failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CCA Tab
# ─────────────────────────────────────────────────────────────────────────────
with tab_cca:
    st.subheader("Canonical Correlation Analysis (CCA)")
    st.caption(
        "CCA finds linear combinations of X and y that are maximally correlated. "
        "Supervised tasks only — requires a continuous or integer target."
    )

    _cca_supervised = task_type in SUPERVISED_TASKS
    if not _cca_supervised:
        st.info("CCA requires a supervised task (classification or regression). "
                "Switch task type on the Upload page.")
    elif y_train is None:
        st.info("No target labels found in session state. Run Preprocessing first.")
    else:
        _cca_max = min(n_features, int(np.unique(y_train).size) if task_type != "regression" else n_features, 10)
        cca_n = st.slider("n_components", 1, max(_cca_max, 1), min(2, _cca_max), key="cca_n")

        _cca_mode = st.radio(
            "Mode",
            ["Use as preprocessing step", "Visualization only"],
            key="cca_mode",
            horizontal=True,
        )

        if st.button("Apply CCA", type="primary", key="btn_cca"):
            with st.spinner("Fitting CCA..."):
                try:
                    from sklearn.cross_decomposition import CCA

                    cca = CCA(n_components=cca_n, max_iter=500)
                    # CCA needs y as a 2D array
                    _y_2d = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
                    _y_2d = _y_2d.astype(float)
                    cca.fit(X_train, _y_2d)

                    X_cca_train = cca.transform(X_train)
                    cca_feat_names = [f"CCA{i+1}" for i in range(cca_n)]

                    # 2D scatter
                    if cca_n >= 2:
                        fig_cca = _scatter_reduced(
                            X_cca_train, y_train, cca_feat_names,
                            title="CCA 2D Projection (training set)",
                        )
                        st.plotly_chart(fig_cca, width='stretch')
                    else:
                        st.line_chart(X_cca_train[:, 0])

                    # X-loadings table (x_rotations_)
                    if hasattr(cca, "x_rotations_"):
                        with st.expander("X Loadings (x_rotations_)", expanded=False):
                            loadings_df = pd.DataFrame(
                                cca.x_rotations_,
                                index=feature_names[:cca.x_rotations_.shape[0]],
                                columns=cca_feat_names,
                            )
                            st.dataframe(loadings_df.round(4), width='stretch')

                    if _cca_mode == "Use as preprocessing step":
                        X_cca_val = cca.transform(X_val) if X_val is not None else None
                        X_cca_test = cca.transform(X_test) if X_test is not None else None
                        _store_fe_result(
                            X_cca_train, X_cca_val, X_cca_test,
                            cca_feat_names, "CCA", cca,
                        )
                        st.success(f"CCA applied! Features: {n_features} → {cca_n}")
                    else:
                        st.session_state["reduction_output"] = X_cca_train
                        st.info("CCA result stored for visualization only (not used in training).")
                except Exception as exc:
                    st.error(f"CCA failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# NMF Tab
# ─────────────────────────────────────────────────────────────────────────────
with tab_nmf:
    st.subheader("Non-negative Matrix Factorization")

    has_negatives = bool(np.any(X_train < 0))
    if has_negatives:
        st.warning(
            "Negative values detected in X_train. NMF requires non-negative data. "
            "Results may be unreliable or fail."
        )

    max_nmf_components = min(n_features, 20)
    nmf_n_components = st.slider(
        "n_components",
        min_value=1, max_value=max_nmf_components,
        value=min(5, max_nmf_components),
        key="nmf_n_components",
    )
    nmf_max_iter = st.slider(
        "max_iter", min_value=100, max_value=500, value=200, step=50,
        key="nmf_max_iter",
    )
    nmf_mode = st.radio(
        "Mode",
        ["Use as preprocessing step", "Visualization only"],
        key="nmf_mode",
        horizontal=True,
    )

    if st.button("Apply NMF", type="primary", key="btn_nmf"):
        with st.spinner("Fitting NMF..."):
            try:
                from sklearn.decomposition import NMF

                X_tr_f = X_train.astype(float)
                # Clip negatives to 0 to avoid NMF errors
                X_tr_nn = np.clip(X_tr_f, 0, None)
                X_vl_nn = np.clip(X_val.astype(float), 0, None) if X_val is not None else None
                X_te_nn = np.clip(X_test.astype(float), 0, None) if X_test is not None else None

                nmf = NMF(
                    n_components=nmf_n_components,
                    max_iter=nmf_max_iter,
                    random_state=42,
                )
                X_tr_nmf = nmf.fit_transform(X_tr_nn)
                X_vl_nmf = nmf.transform(X_vl_nn) if X_vl_nn is not None else None
                X_te_nmf = nmf.transform(X_te_nn) if X_te_nn is not None else None

                nmf_feat_names = [f"NMF{i+1}" for i in range(nmf_n_components)]

                # ── Component heatmap ─────────────────────────────────────────
                comp_df = pd.DataFrame(
                    nmf.components_,
                    index=nmf_feat_names,
                    columns=feature_names[:n_features],
                )
                n_feat_show = min(n_features, 30)
                fig_nmf = px.imshow(
                    comp_df.iloc[:, :n_feat_show],
                    color_continuous_scale="RdBu_r",
                    title="NMF Component Matrix (first 30 features)",
                    aspect="auto",
                )
                fig_nmf.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_nmf, width='stretch')

                if nmf_mode == "Use as preprocessing step":
                    _store_fe_result(
                        X_tr_nmf, X_vl_nmf, X_te_nmf,
                        nmf_feat_names, "NMF", nmf,
                    )
                    st.success(f"NMF applied! Features: {n_features} → {nmf_n_components}")
                else:
                    st.session_state["reduction_output"] = X_tr_nmf
                    st.success(f"NMF visualization computed ({nmf_n_components} components).")

                st.rerun()

            except Exception as exc:
                st.error(f"NMF failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE Tab
# ─────────────────────────────────────────────────────────────────────────────
with tab_tsne:
    st.subheader("t-SNE (Visualization Only)")
    st.info(
        "t-SNE cannot transform new data — it will **NOT** be used as a preprocessing step. "
        "Results are stored for visualization only."
    )

    tsne_n_components = st.radio(
        "n_components",
        [2, 3],
        key="tsne_n_components",
        horizontal=True,
    )
    tsne_perplexity = st.slider(
        "perplexity", min_value=5, max_value=50, value=30,
        key="tsne_perplexity",
    )
    tsne_n_iter = st.slider(
        "n_iter", min_value=250, max_value=1000, value=500, step=50,
        key="tsne_n_iter",
    )

    if st.button("Run t-SNE (visualization only)", type="primary", key="btn_tsne"):
        with st.spinner("Running t-SNE (this may take a while)..."):
            try:
                from sklearn.manifold import TSNE

                tsne = TSNE(
                    n_components=tsne_n_components,
                    perplexity=tsne_perplexity,
                    n_iter=tsne_n_iter,
                    random_state=42,
                )
                X_tsne = tsne.fit_transform(X_train.astype(float))

                tsne_feat_names = [f"tSNE{i+1}" for i in range(tsne_n_components)]

                # ── Scatter ───────────────────────────────────────────────────
                labels_for_plot = y_train if is_supervised_task and y_train is not None else None
                fig_tsne = _scatter_reduced(
                    X_tsne,
                    labels_for_plot,
                    title="t-SNE Projection (training set)",
                    label_name="class",
                    dim_names=tsne_feat_names,
                )
                st.plotly_chart(fig_tsne, width='stretch')

                # Store only for visualization — do NOT set X_train_fe
                st.session_state["reduction_output"] = X_tsne
                st.success("t-SNE complete. Result stored for visualization only.")
                st.rerun()

            except Exception as exc:
                st.error(f"t-SNE failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# UMAP Tab
# ─────────────────────────────────────────────────────────────────────────────
with tab_umap:
    st.subheader("UMAP")

    try:
        import umap as _umap_check  # noqa: F401
        umap_available = True
    except ImportError:
        umap_available = False
        st.warning(
            "UMAP is not installed. Install it with: `pip install umap-learn`"
        )

    if umap_available:
        umap_n_components = st.slider(
            "n_components", min_value=2, max_value=10, value=2,
            key="umap_n_components",
        )
        umap_n_neighbors = st.slider(
            "n_neighbors", min_value=5, max_value=50, value=15,
            key="umap_n_neighbors",
        )
        umap_min_dist = st.slider(
            "min_dist", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
            key="umap_min_dist",
        )
        umap_mode = st.radio(
            "Mode",
            ["Use as preprocessing step", "Visualization only"],
            key="umap_mode",
            horizontal=True,
        )

        if st.button("Apply UMAP", type="primary", key="btn_umap"):
            with st.spinner("Fitting UMAP..."):
                try:
                    import umap

                    reducer = umap.UMAP(
                        n_components=umap_n_components,
                        n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist,
                        random_state=42,
                    )
                    X_tr_f = X_train.astype(float)
                    X_tr_umap = reducer.fit_transform(X_tr_f)
                    X_vl_umap = reducer.transform(X_val.astype(float)) if X_val is not None else None
                    X_te_umap = reducer.transform(X_test.astype(float)) if X_test is not None else None

                    umap_feat_names = [f"UMAP{i+1}" for i in range(umap_n_components)]

                    # ── 2D scatter ────────────────────────────────────────────
                    labels_for_plot = y_train if is_supervised_task and y_train is not None else None
                    fig_umap = _scatter_reduced(
                        X_tr_umap[:, :2] if umap_n_components >= 2 else X_tr_umap,
                        labels_for_plot,
                        title="UMAP 2D Projection (training set)",
                        label_name="class",
                        dim_names=umap_feat_names[:2],
                    )
                    st.plotly_chart(fig_umap, width='stretch')

                    if umap_mode == "Use as preprocessing step":
                        _store_fe_result(
                            X_tr_umap, X_vl_umap, X_te_umap,
                            umap_feat_names, "UMAP", reducer,
                        )
                        st.success(
                            f"UMAP applied! Features: {n_features} → {umap_n_components}"
                        )
                    else:
                        st.session_state["reduction_output"] = X_tr_umap
                        st.success(f"UMAP visualization computed ({umap_n_components} components).")

                    st.rerun()

                except Exception as exc:
                    st.error(f"UMAP failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature Transforms Tab (from original page — kept and enhanced)
# ─────────────────────────────────────────────────────────────────────────────
with tab_transforms:
    st.subheader("Feature Transforms")
    st.caption(
        "These transforms can be used standalone or stacked on top of a "
        "dimensionality reduction technique already applied above."
    )

    # ── Polynomial Features ──────────────────────────────────────────────────
    with st.expander("Polynomial Features (interactions + squares)", expanded=False):
        poly_degree = st.slider("Degree", 2, 3, 2, key="poly_degree")
        poly_interaction_only = st.checkbox(
            "Interaction terms only (no squares)", key="poly_interact"
        )
        poly_enabled = st.checkbox("Enable polynomial features", key="poly_enabled")

        if poly_enabled and n_features > 20:
            st.warning(
                f"Your dataset has {n_features} features. Polynomial expansion may create "
                "many features and slow training significantly. Consider reducing features first."
            )

    # ── Log / Sqrt Transforms ─────────────────────────────────────────────────
    with st.expander("Log / Sqrt transforms on skewed columns", expanded=False):
        numerical_feature_names = [
            fn for fn in feature_names
            if not fn.endswith("_cat") and not fn.endswith("_oh")
        ]

        skewness_map: dict[str, float] = {}
        df_preview = pd.DataFrame(X_train, columns=feature_names)
        for fn in numerical_feature_names:
            try:
                skewness_map[fn] = float(df_preview[fn].skew())
            except Exception:
                pass

        skewed_cols = [fn for fn, sk in skewness_map.items() if abs(sk) > 1.0]
        transform_choices: dict[str, str] = {}

        if skewed_cols:
            st.caption(f"{len(skewed_cols)} columns with |skewness| > 1.0 detected.")
            for fn in skewed_cols:
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.write(f"`{fn}` — skewness: {skewness_map[fn]:.2f}")
                with col2:
                    choice = st.selectbox(
                        "Transform",
                        ["none", "log1p", "sqrt"],
                        key=f"transform_{fn}",
                    )
                    transform_choices[fn] = choice
        else:
            st.success("No highly skewed columns detected (|skewness| <= 1.0).")

    # ── Quantile Binning ─────────────────────────────────────────────────────
    with st.expander("Quantile Binning", expanded=False):
        bins_enabled_cols: dict[str, int] = {}
        for fn in numerical_feature_names[:10]:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"`{fn}`")
            with col2:
                n_bins = st.slider("Bins", 2, 10, 4, key=f"bins_{fn}")
            with col3:
                if st.checkbox("Enable", key=f"bin_en_{fn}"):
                    bins_enabled_cols[fn] = n_bins

    st.divider()

    # Determine input arrays: use FE output if already applied, else raw
    def _get_current_arrays():
        if st.session_state.get("feature_engineering_applied") and "X_train_fe" in st.session_state:
            return (
                st.session_state["X_train_fe"],
                st.session_state.get("X_val_fe"),
                st.session_state.get("X_test_fe"),
                st.session_state.get("feature_names_fe", feature_names),
            )
        return X_train, X_val, X_test, feature_names

    if st.button("Apply Feature Transforms", type="primary", key="btn_transforms"):
        with st.spinner("Applying transformations..."):
            try:
                X_tr_cur, X_vl_cur, X_te_cur, fn_cur = _get_current_arrays()

                X_tr = X_tr_cur.copy().astype(float)
                X_vl_t = X_vl_cur.copy().astype(float) if X_vl_cur is not None else None
                X_te_t = X_te_cur.copy().astype(float) if X_te_cur is not None else None
                fn_list = list(fn_cur)

                df_tr = pd.DataFrame(X_tr, columns=fn_list)
                df_vl = pd.DataFrame(X_vl_t, columns=fn_list) if X_vl_t is not None else None
                df_te = pd.DataFrame(X_te_t, columns=fn_list) if X_te_t is not None else None

                # -- Log / Sqrt transforms --
                effective_transforms = st.session_state.get("_transform_choices_snapshot", transform_choices)
                for fn, choice in transform_choices.items():
                    if fn not in df_tr.columns:
                        continue
                    if choice == "log1p":
                        for df_ in [d for d in [df_tr, df_vl, df_te] if d is not None]:
                            df_[fn] = np.log1p(df_[fn].clip(lower=0))
                    elif choice == "sqrt":
                        for df_ in [d for d in [df_tr, df_vl, df_te] if d is not None]:
                            df_[fn] = np.sqrt(df_[fn].clip(lower=0))

                # -- Quantile binning --
                for fn, nb in bins_enabled_cols.items():
                    if fn not in df_tr.columns:
                        continue
                    try:
                        bin_col = f"{fn}_bin"
                        _, edges = pd.cut(df_tr[fn], bins=nb, retbins=True)
                        for df_ in [d for d in [df_tr, df_vl, df_te] if d is not None]:
                            df_[bin_col] = (
                                pd.cut(df_[fn], bins=edges, labels=False, include_lowest=True)
                                .fillna(0)
                                .astype(float)
                            )
                    except Exception:
                        pass

                # -- Polynomial features --
                poly_enabled_flag = st.session_state.get("poly_enabled", False)
                if poly_enabled_flag:
                    try:
                        from sklearn.preprocessing import PolynomialFeatures

                        poly = PolynomialFeatures(
                            degree=st.session_state.get("poly_degree", 2),
                            interaction_only=st.session_state.get("poly_interact", False),
                            include_bias=False,
                        )
                        X_tr_poly = poly.fit_transform(df_tr.values)
                        poly_names = [f"poly_{i}" for i in range(X_tr_poly.shape[1])]
                        df_tr = pd.DataFrame(X_tr_poly, columns=poly_names)
                        if df_vl is not None:
                            df_vl = pd.DataFrame(poly.transform(df_vl.values), columns=poly_names)
                        if df_te is not None:
                            df_te = pd.DataFrame(poly.transform(df_te.values), columns=poly_names)
                    except Exception as e:
                        st.warning(f"Polynomial features failed: {e}")

                new_fn = df_tr.columns.tolist()
                technique_label = st.session_state.get("fe_technique", "transforms")
                if technique_label not in ("PCA", "LDA", "NMF", "UMAP"):
                    technique_label = "feature_transforms"
                else:
                    technique_label = f"{technique_label}+transforms"

                _store_fe_result(
                    df_tr.values,
                    df_vl.values if df_vl is not None else None,
                    df_te.values if df_te is not None else None,
                    new_fn,
                    technique_label,
                    None,
                )
                orig_n = st.session_state.get("_fe_orig_n_features", n_features)
                st.success(
                    f"Feature transforms applied! "
                    f"Features: {orig_n} → {df_tr.shape[1]}"
                )
                st.rerun()

            except Exception as exc:
                st.error(f"Feature transforms failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Results Preview
# ═══════════════════════════════════════════════════════════════════════════════
fe_applied = st.session_state.get("feature_engineering_applied", False)
reduction_output = st.session_state.get("reduction_output")

if fe_applied or reduction_output is not None:
    st.divider()
    st.header("Results Preview")

    orig_n = st.session_state.get("_fe_orig_n_features", n_features)

    if fe_applied and "X_train_fe" in st.session_state:
        X_fe = st.session_state["X_train_fe"]
        fn_fe = st.session_state.get("feature_names_fe", [])
        technique = st.session_state.get("fe_technique", "—")

        c1, c2, c3 = st.columns(3)
        c1.metric("Original features", orig_n)
        c2.metric("After reduction/transforms", X_fe.shape[1], delta=X_fe.shape[1] - orig_n)
        c3.metric("Technique", technique)

        # ── 2D scatter of engineered features ────────────────────────────────
        if X_fe.shape[1] >= 2:
            if X_fe.shape[1] == 2:
                X_display_2d = X_fe
                dim_names_2d = fn_fe[:2]
            else:
                X_display_2d = _project_to_2d_for_display(X_fe)
                dim_names_2d = ["PCA1 (proj)", "PCA2 (proj)"]

            labels_for_plot = y_train if is_supervised_task and y_train is not None else None
            fig_2d = _scatter_reduced(
                X_display_2d, labels_for_plot,
                title=f"2D View of Engineered Features ({technique})",
                label_name="class",
                dim_names=dim_names_2d,
            )
            st.plotly_chart(fig_2d, width='stretch')

        # ── 3D scatter if available ───────────────────────────────────────────
        if X_fe.shape[1] >= 3:
            with st.expander("3D scatter view", expanded=False):
                labels_for_plot = y_train if is_supervised_task and y_train is not None else None
                fig_3d = _scatter_reduced(
                    X_fe[:, :3], labels_for_plot,
                    title=f"3D View of Engineered Features ({technique})",
                    label_name="class",
                    dim_names=fn_fe[:3],
                )
                st.plotly_chart(fig_3d, width='stretch')

        # ── Component loadings (if PCA) ───────────────────────────────────────
        if "_pca_loadings" in st.session_state and technique in ("PCA", "PCA+transforms"):
            with st.expander("PCA Component Loadings", expanded=False):
                st.dataframe(
                    st.session_state["_pca_loadings"].round(4),
                    width='stretch',
                )

        # ── Data preview table ────────────────────────────────────────────────
        with st.expander("Preview transformed training data (first 10 rows)", expanded=False):
            st.dataframe(
                pd.DataFrame(X_fe[:10], columns=fn_fe),
                width='stretch',
            )

    elif reduction_output is not None:
        # Visualization-only mode (t-SNE or others)
        X_vis = reduction_output
        st.info("Showing visualization-only result (not used for training).")
        c1, c2 = st.columns(2)
        c1.metric("Visualization dimensions", X_vis.shape[1])
        c2.metric("Original features", orig_n)

        labels_for_plot = y_train if is_supervised_task and y_train is not None else None
        if X_vis.shape[1] >= 2:
            vis_dim_names = [f"Dim{i+1}" for i in range(X_vis.shape[1])]
            fig_vis = _scatter_reduced(
                X_vis[:, :3] if X_vis.shape[1] >= 3 else X_vis,
                labels_for_plot,
                title="Visualization Projection (training set)",
                label_name="class",
                dim_names=vis_dim_names[:3],
            )
            st.plotly_chart(fig_vis, width='stretch')

    # ── Clear button ──────────────────────────────────────────────────────────
    if st.button("Clear Feature Engineering", key="btn_clear_fe"):
        for key in (
            "X_train_fe", "X_val_fe", "X_test_fe", "feature_names_fe",
            "feature_engineering_applied", "fe_transformer", "fe_technique",
            "_fe_orig_n_features", "_pca_loadings", "reduction_output",
        ):
            st.session_state.pop(key, None)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Navigation
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()

nav1, nav_mid, nav2 = st.columns([1, 1, 1])

with nav1:
    st.page_link("pages/2_Preprocessing.py", label="← Back to Preprocessing")

with nav_mid:
    if st.button("Skip — use features as-is →", type="secondary", key="btn_skip_fe"):
        for key in (
            "X_train_fe", "X_val_fe", "X_test_fe", "feature_names_fe",
            "feature_engineering_applied", "fe_transformer", "fe_technique",
            "_fe_orig_n_features", "_pca_loadings", "reduction_output",
        ):
            st.session_state.pop(key, None)
        st.rerun()

with nav2:
    st.page_link("pages/4_Training.py", label="Continue to Training →")
