import streamlit as st
import numpy as np
import pandas as pd

from src.pipelines.training import (
    get_default_params,
    run_optuna_tuning,
    save_model,
    log_to_mlflow,
    train_model,
    get_training_summary,
    normalize_task_type,
    is_supervised,
    create_model_adapter,
    MODEL_DISPLAY_NAMES,
    CLUSTERING_MODEL_TYPES,
    ANOMALY_MODEL_TYPES,
    REDUCTION_MODEL_TYPES,
    CLASSIFICATION_MODEL_TYPES,
    REGRESSION_MODEL_TYPES,
)

st.set_page_config(page_title="Training", page_icon="🏋️", layout="wide")

UNSUPERVISED_TASKS = {"clustering", "dimensionality_reduction", "anomaly_detection"}

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Dataset Status")
    if "df" in st.session_state:
        st.write(f"**File:** {st.session_state.get('uploaded_filename', '—')}")
        shape = st.session_state["df"].shape
        st.write(f"**Shape:** {shape[0]} rows × {shape[1]} cols")
        if st.session_state.get("target_column"):
            st.write(f"**Target:** {st.session_state['target_column']}")
        if st.session_state.get("task_type"):
            st.write(f"**Task:** {st.session_state['task_type']}")
    else:
        st.info("No dataset loaded yet.")

# ── Guard clause ─────────────────────────────────────────────────────────────
st.title("🏋️ Training")

if "X_train" not in st.session_state:
    st.warning(
        "Please preprocess your data on the **Preprocessing** page first."
    )
    st.stop()

# Resolve task type
task_type: str = normalize_task_type(st.session_state.get("task_type", "binary_classification"))
supervised = is_supervised(task_type)

# Resolve feature arrays — prefer Feature Engineering outputs if available
if st.session_state.get("feature_engineering_applied") and "X_train_fe" in st.session_state:
    X_train: np.ndarray = st.session_state["X_train_fe"]
    X_val_raw = st.session_state.get("X_val_fe")
    X_test_raw = st.session_state.get("X_test_fe")
    feature_names: list[str] = st.session_state.get("feature_names_fe", [])
    st.info("Using feature-engineered data from the previous step.")
else:
    X_train = st.session_state["X_train"]
    X_val_raw = st.session_state.get("X_val")
    X_test_raw = st.session_state.get("X_test")
    feature_names = st.session_state.get("feature_names", [f"f{i}" for i in range(X_train.shape[1])])

X_val: np.ndarray | None = X_val_raw
X_test: np.ndarray | None = X_test_raw

# y arrays — None for unsupervised
if supervised:
    _y_train = st.session_state.get("y_train")
    _y_val = st.session_state.get("y_val")
    _y_test = st.session_state.get("y_test")
    y_train = np.asarray(_y_train) if _y_train is not None else None
    y_val = np.asarray(_y_val) if _y_val is not None else None
    y_test = np.asarray(_y_test) if _y_test is not None else None
    if y_train is None:
        st.warning("y_train is missing. Please re-run Preprocessing.")
        st.stop()
else:
    y_train = None
    y_val = None
    y_test = None

target_column: str = st.session_state.get("target_column", "")
st.metric("Features", X_train.shape[1])

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Model Selection (conditional on task_type)
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Model Selection")

# ── Binary / Multiclass Classification ─────────────────────────────────────────
if task_type in ("binary_classification", "multiclass_classification"):
    cols = st.columns(4)
    use_rf  = cols[0].checkbox("Random Forest",      value=True,  key="use_rf")
    use_xgb = cols[1].checkbox("XGBoost",            value=True,  key="use_xgb")
    use_lr  = cols[2].checkbox("Logistic Regression",value=False, key="use_lr")
    use_svm = cols[3].checkbox("SVM",                value=False, key="use_svm")

    selected_models: list[str] = []
    if use_rf:  selected_models.append("random_forest")
    if use_xgb: selected_models.append("xgboost")
    if use_lr:  selected_models.append("logistic_regression")
    if use_svm: selected_models.append("svm")

# ── Regression ──────────────────────────────────────────────────────────────
elif task_type == "regression":
    cols = st.columns(4)
    use_rf  = cols[0].checkbox("Random Forest",    value=True,  key="use_rf")
    use_xgb = cols[1].checkbox("XGBoost",          value=True,  key="use_xgb")
    use_lr  = cols[2].checkbox("Linear Regression",value=False, key="use_lr")
    use_ridge = cols[3].checkbox("Ridge",          value=False, key="use_ridge")
    cols2 = st.columns(4)
    use_svr   = cols2[0].checkbox("SVR",           value=False, key="use_svr")
    use_gb    = cols2[1].checkbox("Gradient Boost",value=False, key="use_gb")
    use_knn   = cols2[2].checkbox("KNN",           value=False, key="use_knn_r")

    selected_models = []
    if use_rf:    selected_models.append("random_forest")
    if use_xgb:   selected_models.append("xgboost")
    if use_lr:    selected_models.append("linear_regression")
    if use_ridge: selected_models.append("ridge")
    if use_svr:   selected_models.append("svr")
    if use_gb:    selected_models.append("gradient_boosting")
    if use_knn:   selected_models.append("knn")

# ── Clustering ───────────────────────────────────────────────────────────────
elif task_type == "clustering":
    cols = st.columns(5)
    use_km  = cols[0].checkbox("KMeans",            value=True,  key="use_kmeans")
    use_ms  = cols[1].checkbox("Mean Shift",        value=False, key="use_mean_shift")
    use_db  = cols[2].checkbox("DBSCAN",            value=False, key="use_dbscan")
    use_agg = cols[3].checkbox("Agglomerative",     value=False, key="use_agg")
    use_gmm = cols[4].checkbox("Gaussian Mixture",  value=False, key="use_gmm")

    selected_models = []
    if use_km:  selected_models.append("kmeans")
    if use_ms:  selected_models.append("mean_shift")
    if use_db:  selected_models.append("dbscan")
    if use_agg: selected_models.append("agglomerative")
    if use_gmm: selected_models.append("gaussian_mixture")

# ── Anomaly Detection ─────────────────────────────────────────────────────────
elif task_type == "anomaly_detection":
    cols = st.columns(4)
    use_if  = cols[0].checkbox("Isolation Forest",     value=True,  key="use_iforest")
    use_oc  = cols[1].checkbox("One-Class SVM",        value=False, key="use_ocsvm")
    use_lof = cols[2].checkbox("LOF",                  value=False, key="use_lof")
    use_ee  = cols[3].checkbox("Elliptic Envelope",    value=False, key="use_ee")

    selected_models = []
    if use_if:  selected_models.append("isolation_forest")
    if use_oc:  selected_models.append("one_class_svm")
    if use_lof: selected_models.append("local_outlier_factor")
    if use_ee:  selected_models.append("elliptic_envelope")

# ── Dimensionality Reduction ──────────────────────────────────────────────────
elif task_type == "dimensionality_reduction":
    cols = st.columns(4)
    use_pca  = cols[0].checkbox("PCA",          value=True,  key="use_pca")
    use_umap = cols[1].checkbox("UMAP",         value=False, key="use_umap")
    use_tsne = cols[2].checkbox("t-SNE",        value=False, key="use_tsne")
    use_svd  = cols[3].checkbox("Truncated SVD",value=False, key="use_tsvd")

    selected_models = []
    if use_pca:  selected_models.append("pca")
    if use_umap: selected_models.append("umap")
    if use_tsne: selected_models.append("tsne")
    if use_svd:  selected_models.append("truncated_svd")

else:
    st.error(f"Unknown task type: `{task_type}`")
    st.stop()

if not selected_models:
    st.info("Select at least one model above.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Cross-Validation Config (supervised only)
# ═══════════════════════════════════════════════════════════════════════════════
cv_folds = 5
if supervised:
    st.header("Cross-Validation")
    cv_folds = st.slider("CV folds", min_value=2, max_value=10, value=5, key="cv_folds")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Per-Model Hyperparameter Controls
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Hyperparameters")

model_configs: dict[str, dict] = {}

# ── Random Forest ─────────────────────────────────────────────────────────────
if "random_forest" in selected_models:
    with st.expander("Random Forest", expanded=True):
        auto_rf = st.toggle("Auto Tune with Optuna", key="auto_rf")
        if auto_rf:
            rf_trials = st.number_input("Optuna trials", 5, 200, 50, key="rf_trials")
            rf_timeout = st.number_input("Timeout (sec)", 10, 600, 300, key="rf_timeout")
            model_configs["random_forest"] = {
                "auto_tune": True, "n_trials": int(rf_trials), "timeout": int(rf_timeout),
            }
        else:
            c1, c2 = st.columns(2)
            with c1:
                rf_n_est   = st.slider("n_estimators", 50, 500, 100, step=50, key="rf_n_est")
                rf_max_depth = st.slider("max_depth", 3, 30, 10, key="rf_max_depth")
                rf_min_split = st.slider("min_samples_split", 2, 20, 5, key="rf_min_split")
            with c2:
                rf_min_leaf = st.slider("min_samples_leaf", 1, 10, 2, key="rf_min_leaf")
                rf_max_feat = st.selectbox("max_features", ["sqrt", "log2", "None"], key="rf_max_feat")
            model_configs["random_forest"] = {
                "auto_tune": False,
                "params": {
                    "n_estimators": rf_n_est,
                    "max_depth": rf_max_depth,
                    "min_samples_split": rf_min_split,
                    "min_samples_leaf": rf_min_leaf,
                    "max_features": None if rf_max_feat == "None" else rf_max_feat,
                },
            }

# ── XGBoost ───────────────────────────────────────────────────────────────────
if "xgboost" in selected_models:
    with st.expander("XGBoost", expanded=True):
        auto_xgb = st.toggle("Auto Tune with Optuna", key="auto_xgb")
        if auto_xgb:
            xgb_trials  = st.number_input("Optuna trials", 5, 200, 50, key="xgb_trials")
            xgb_timeout = st.number_input("Timeout (sec)", 10, 600, 300, key="xgb_timeout")
            model_configs["xgboost"] = {
                "auto_tune": True, "n_trials": int(xgb_trials), "timeout": int(xgb_timeout),
            }
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                xgb_n_est    = st.slider("n_estimators", 50, 500, 100, step=50, key="xgb_n_est")
                xgb_max_depth= st.slider("max_depth", 3, 15, 6, key="xgb_max_depth")
                xgb_lr       = st.slider("learning_rate", 0.01, 0.3, 0.1, step=0.01, key="xgb_lr")
            with c2:
                xgb_subsample  = st.slider("subsample", 0.5, 1.0, 0.8, step=0.05, key="xgb_subsample")
                xgb_colsample  = st.slider("colsample_bytree", 0.5, 1.0, 0.8, step=0.05, key="xgb_colsample")
                xgb_min_child  = st.slider("min_child_weight", 1, 10, 3, key="xgb_min_child")
            with c3:
                xgb_gamma  = st.slider("gamma", 0.0, 5.0, 0.0, step=0.1, key="xgb_gamma")
                xgb_alpha  = st.slider("reg_alpha", 0.0, 10.0, 0.0, step=0.5, key="xgb_alpha")
                xgb_lambda = st.slider("reg_lambda", 0.0, 10.0, 1.0, step=0.5, key="xgb_lambda")
            model_configs["xgboost"] = {
                "auto_tune": False,
                "params": {
                    "n_estimators": xgb_n_est,
                    "max_depth": xgb_max_depth,
                    "learning_rate": xgb_lr,
                    "subsample": xgb_subsample,
                    "colsample_bytree": xgb_colsample,
                    "min_child_weight": xgb_min_child,
                    "gamma": xgb_gamma,
                    "reg_alpha": xgb_alpha,
                    "reg_lambda": xgb_lambda,
                },
            }

# ── Logistic Regression ───────────────────────────────────────────────────────
if "logistic_regression" in selected_models:
    with st.expander("Logistic Regression", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            lr_C       = st.number_input("C (regularization)", 0.001, 100.0, 1.0, key="lr_C")
            lr_penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"], key="lr_penalty")
        with c2:
            lr_solver  = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"], key="lr_solver")
            lr_max_iter= st.slider("Max iterations", 100, 2000, 200, key="lr_max_iter")
        model_configs["logistic_regression"] = {
            "auto_tune": False,
            "params": {"C": lr_C, "penalty": lr_penalty, "solver": lr_solver, "max_iter": lr_max_iter},
        }

# ── SVM (classification) ──────────────────────────────────────────────────────
if "svm" in selected_models:
    with st.expander("SVM", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            svm_C      = st.number_input("C", 0.01, 100.0, 1.0, key="svm_C")
            svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svm_kernel")
        with c2:
            svm_gamma  = st.selectbox("Gamma", ["scale", "auto"], key="svm_gamma")
        model_configs["svm"] = {
            "auto_tune": False,
            "params": {"C": svm_C, "kernel": svm_kernel, "gamma": svm_gamma, "probability": True},
        }

# ── Linear Regression ─────────────────────────────────────────────────────────
if "linear_regression" in selected_models:
    model_configs["linear_regression"] = {"auto_tune": False, "params": {}}

# ── Ridge ─────────────────────────────────────────────────────────────────────
if "ridge" in selected_models:
    with st.expander("Ridge", expanded=True):
        ridge_alpha = st.number_input("Alpha", 0.001, 1000.0, 1.0, key="ridge_alpha")
        model_configs["ridge"] = {"auto_tune": False, "params": {"alpha": ridge_alpha}}

# ── SVR ───────────────────────────────────────────────────────────────────────
if "svr" in selected_models:
    with st.expander("SVR", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            svr_C      = st.number_input("C", 0.01, 100.0, 1.0, key="svr_C")
            svr_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="svr_kernel")
        with c2:
            svr_epsilon = st.number_input("Epsilon", 0.001, 10.0, 0.1, key="svr_epsilon")
        model_configs["svr"] = {
            "auto_tune": False,
            "params": {"C": svr_C, "kernel": svr_kernel, "epsilon": svr_epsilon},
        }

# ── Gradient Boosting ─────────────────────────────────────────────────────────
if "gradient_boosting" in selected_models:
    with st.expander("Gradient Boosting", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            gb_n_est    = st.slider("n_estimators", 50, 500, 100, step=50, key="gb_n_est")
            gb_max_depth= st.slider("max_depth", 2, 10, 3, key="gb_max_depth")
        with c2:
            gb_lr       = st.slider("learning_rate", 0.01, 0.3, 0.1, step=0.01, key="gb_lr")
            gb_subsample= st.slider("subsample", 0.5, 1.0, 0.8, step=0.05, key="gb_subsample")
        model_configs["gradient_boosting"] = {
            "auto_tune": False,
            "params": {
                "n_estimators": gb_n_est, "max_depth": gb_max_depth,
                "learning_rate": gb_lr, "subsample": gb_subsample,
            },
        }

# ── KNN (regression) ─────────────────────────────────────────────────────────
if "knn" in selected_models:
    with st.expander("KNN", expanded=True):
        knn_k = st.slider("n_neighbors", 1, 30, 5, key="knn_k")
        model_configs["knn"] = {"auto_tune": False, "params": {"n_neighbors": knn_k}}

# ── KMeans ────────────────────────────────────────────────────────────────────
if "kmeans" in selected_models:
    with st.expander("KMeans", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            km_k     = st.slider("n_clusters", 2, 20, 3, key="km_k")
            km_init  = st.selectbox("Init", ["k-means++", "random"], key="km_init")
        with c2:
            km_n_init = st.slider("n_init", 5, 20, 10, key="km_n_init")
            km_max_iter = st.slider("max_iter", 100, 500, 300, key="km_max_iter")
        model_configs["kmeans"] = {
            "auto_tune": False,
            "params": {"n_clusters": km_k, "init": km_init, "n_init": km_n_init, "max_iter": km_max_iter},
        }

# ── Mean Shift ───────────────────────────────────────────────────────────────
if "mean_shift" in selected_models:
    with st.expander("Mean Shift", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            ms_bandwidth = st.number_input(
                "bandwidth (0 = auto-estimate)", min_value=0.0, value=0.0,
                step=0.1, key="ms_bandwidth",
                help="Leave 0 to let sklearn estimate bandwidth from data.",
            )
        with c2:
            ms_bin_seeding = st.checkbox("bin_seeding (faster for large data)", value=False, key="ms_bin_seeding")
        model_configs["mean_shift"] = {
            "auto_tune": False,
            "params": {
                "bandwidth": ms_bandwidth if ms_bandwidth > 0 else None,
                "bin_seeding": ms_bin_seeding,
            },
        }

# ── DBSCAN ────────────────────────────────────────────────────────────────────
if "dbscan" in selected_models:
    with st.expander("DBSCAN", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            db_eps    = st.number_input("eps", 0.01, 10.0, 0.5, key="db_eps")
        with c2:
            db_min_samples = st.slider("min_samples", 2, 50, 5, key="db_min_samples")
        model_configs["dbscan"] = {
            "auto_tune": False,
            "params": {"eps": db_eps, "min_samples": db_min_samples},
        }

# ── Agglomerative ─────────────────────────────────────────────────────────────
if "agglomerative" in selected_models:
    with st.expander("Agglomerative Clustering", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            agg_k      = st.slider("n_clusters", 2, 20, 3, key="agg_k")
        with c2:
            agg_linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key="agg_linkage")
        model_configs["agglomerative"] = {
            "auto_tune": False,
            "params": {"n_clusters": agg_k, "linkage": agg_linkage},
        }

# ── Gaussian Mixture ─────────────────────────────────────────────────────────
if "gaussian_mixture" in selected_models:
    with st.expander("Gaussian Mixture Model", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            gmm_k         = st.slider("n_components", 2, 20, 3, key="gmm_k")
        with c2:
            gmm_cov       = st.selectbox("Covariance type", ["full", "tied", "diag", "spherical"], key="gmm_cov")
        model_configs["gaussian_mixture"] = {
            "auto_tune": False,
            "params": {"n_components": gmm_k, "covariance_type": gmm_cov},
        }

# ── Isolation Forest ─────────────────────────────────────────────────────────
if "isolation_forest" in selected_models:
    with st.expander("Isolation Forest", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            if_n_est     = st.slider("n_estimators", 50, 500, 100, step=50, key="if_n_est")
            if_contamination = st.slider("contamination", 0.01, 0.5, 0.1, step=0.01, key="if_cont")
        with c2:
            if_max_feat  = st.selectbox("max_features", ["1.0", "auto", "sqrt"], key="if_max_feat")
        model_configs["isolation_forest"] = {
            "auto_tune": False,
            "params": {
                "n_estimators": if_n_est,
                "contamination": if_contamination,
                "max_features": float(if_max_feat) if if_max_feat == "1.0" else if_max_feat,
            },
        }

# ── One-Class SVM ─────────────────────────────────────────────────────────────
if "one_class_svm" in selected_models:
    with st.expander("One-Class SVM", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            oc_kernel = st.selectbox("Kernel", ["rbf", "poly", "sigmoid"], key="oc_kernel")
            oc_nu     = st.slider("nu", 0.01, 0.99, 0.1, step=0.01, key="oc_nu")
        with c2:
            oc_gamma  = st.selectbox("Gamma", ["scale", "auto"], key="oc_gamma")
        model_configs["one_class_svm"] = {
            "auto_tune": False,
            "params": {"kernel": oc_kernel, "nu": oc_nu, "gamma": oc_gamma},
        }

# ── Local Outlier Factor ───────────────────────────────────────────────────────
if "local_outlier_factor" in selected_models:
    with st.expander("Local Outlier Factor", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            lof_n       = st.slider("n_neighbors", 5, 50, 20, key="lof_n")
            lof_cont    = st.slider("contamination", 0.01, 0.5, 0.1, step=0.01, key="lof_cont")
        with c2:
            lof_algo    = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"], key="lof_algo")
        model_configs["local_outlier_factor"] = {
            "auto_tune": False,
            "params": {"n_neighbors": lof_n, "contamination": lof_cont, "algorithm": lof_algo},
        }

# ── Elliptic Envelope ─────────────────────────────────────────────────────────
if "elliptic_envelope" in selected_models:
    with st.expander("Elliptic Envelope", expanded=True):
        ee_cont = st.slider("contamination", 0.01, 0.5, 0.1, step=0.01, key="ee_cont")
        model_configs["elliptic_envelope"] = {
            "auto_tune": False,
            "params": {"contamination": ee_cont},
        }

# ── PCA ───────────────────────────────────────────────────────────────────────
if "pca" in selected_models:
    with st.expander("PCA", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            pca_n = st.slider("n_components", 2, min(50, X_train.shape[1]), 2, key="pca_n")
        with c2:
            pca_whiten = st.checkbox("Whiten", value=False, key="pca_whiten")
        model_configs["pca"] = {
            "auto_tune": False,
            "params": {"n_components": pca_n, "whiten": pca_whiten},
        }

# ── UMAP ──────────────────────────────────────────────────────────────────────
if "umap" in selected_models:
    with st.expander("UMAP", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            umap_n = st.slider("n_components", 2, 10, 2, key="umap_n")
            umap_neighbors = st.slider("n_neighbors", 5, 100, 15, key="umap_neighbors")
        with c2:
            umap_min_dist = st.slider("min_dist", 0.0, 0.99, 0.1, step=0.05, key="umap_min_dist")
            umap_metric   = st.selectbox("Metric", ["euclidean", "cosine", "manhattan"], key="umap_metric")
        model_configs["umap"] = {
            "auto_tune": False,
            "params": {
                "n_components": umap_n, "n_neighbors": umap_neighbors,
                "min_dist": umap_min_dist, "metric": umap_metric,
            },
        }

# ── t-SNE ─────────────────────────────────────────────────────────────────────
if "tsne" in selected_models:
    with st.expander("t-SNE", expanded=True):
        st.caption("⚠️ t-SNE cannot project new data points — prediction page will warn you.")
        c1, c2 = st.columns(2)
        with c1:
            tsne_n     = st.slider("n_components", 2, 3, 2, key="tsne_n")
            tsne_perp  = st.slider("perplexity", 5, 100, 30, key="tsne_perp")
        with c2:
            tsne_lr    = st.number_input("learning_rate", 10.0, 1000.0, 200.0, key="tsne_lr")
            tsne_iter  = st.slider("n_iter", 250, 5000, 1000, step=250, key="tsne_iter")
        model_configs["tsne"] = {
            "auto_tune": False,
            "params": {
                "n_components": tsne_n, "perplexity": tsne_perp,
                "learning_rate": tsne_lr, "n_iter": tsne_iter,
            },
        }

# ── Truncated SVD ─────────────────────────────────────────────────────────────
if "truncated_svd" in selected_models:
    with st.expander("Truncated SVD (LSA)", expanded=True):
        svd_n = st.slider("n_components", 2, min(100, X_train.shape[1] - 1), 2, key="tsvd_n")
        model_configs["truncated_svd"] = {
            "auto_tune": False,
            "params": {"n_components": svd_n},
        }

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Train Models
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Train Models")

if st.button("🏋️ Train Models", type="primary"):
    results = []

    for model_type in selected_models:
        cfg = model_configs[model_type]
        display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)

        st.subheader(f"Training {display_name}")

        try:
            # Step 1: Optuna tuning (if enabled — supervised only)
            if cfg.get("auto_tune") and supervised:
                st.write("Running Optuna hyperparameter search...")
                tune_bar = st.progress(0, text="Tuning...")

                def tune_callback(trial_num, total, score):
                    pct = trial_num / total
                    tune_bar.progress(pct, text=f"Trial {trial_num}/{total} — best score: {score:.4f}")

                params = run_optuna_tuning(
                    model_type, task_type,
                    X_train, y_train,
                    cv_folds=cv_folds,
                    n_trials=cfg["n_trials"],
                    timeout=cfg["timeout"],
                    callback=tune_callback,
                )
                tune_bar.progress(1.0, text="Tuning complete!")
                st.write(f"Best params: `{params}`")
            else:
                params = cfg.get("params", get_default_params(model_type, task_type))

            # Step 2: Train model
            train_bar = st.progress(0, text="Training...")

            result = train_model(
                model_type, task_type, params,
                X_train, y_train,
                X_val if supervised else None,
                y_val if supervised else None,
                cv_folds=cv_folds,
            )
            train_bar.progress(1.0, text="Training complete!")

            # Step 3: Save model artifact
            artifact_path = save_model(
                result, task_type=task_type,
                feature_names=feature_names, target_column=target_column,
            )
            st.write(f"Model saved to `{artifact_path}`")

            # Step 4: MLflow logging (best-effort)
            run_id = log_to_mlflow(result)
            if run_id:
                result.mlflow_run_id = run_id
                st.write(f"MLflow run: `{run_id}`")
            else:
                st.caption("MLflow logging skipped (server unavailable).")

            results.append(result)
            st.success(f"{display_name} training complete!")

        except Exception as e:
            st.error(f"{display_name} failed: {e}")
            continue

    # Step 5: Store in session state
    if not results:
        st.error("All models failed to train. Check the errors above.")
        st.stop()

    st.session_state["training_results"] = results
    st.session_state["trained_models"] = {r.model_type: r.model for r in results}
    summary = get_training_summary(results)
    st.session_state["training_summary"] = summary

    # Build ModelAdapters for all trained models
    result_adapters = {
        r.model_type: create_model_adapter(r, task_type, feature_names, target_column)
        for r in results
    }
    st.session_state["training_adapters"] = result_adapters

    # Best model adapter
    best_name = summary.get("best_model", "")
    best_result = next((r for r in results if r.model_name == best_name), results[0])
    st.session_state["training_adapter"] = result_adapters.get(best_result.model_type)

    # Store task-specific outputs
    if task_type == "clustering":
        for r in results:
            adapter = result_adapters.get(r.model_type)
            if adapter is not None:
                try:
                    st.session_state["cluster_labels"] = adapter.predict(X_train)
                except Exception:
                    pass
                break
    elif task_type == "dimensionality_reduction":
        for r in results:
            adapter = result_adapters.get(r.model_type)
            if adapter is not None:
                try:
                    st.session_state["reduction_output"] = adapter.transform(X_train)
                except Exception:
                    pass
                break
    elif task_type == "anomaly_detection":
        for r in results:
            adapter = result_adapters.get(r.model_type)
            if adapter is not None:
                try:
                    st.session_state["anomaly_scores"] = adapter.decision_scores(X_train)
                except Exception:
                    pass
                break

    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Results Summary (after training)
# ═══════════════════════════════════════════════════════════════════════════════
if "training_results" in st.session_state:
    st.header("Results Summary")

    results = st.session_state["training_results"]
    summary = st.session_state["training_summary"]

    # Metrics comparison table
    comparison_df = pd.DataFrame(summary["comparison"])
    st.dataframe(comparison_df, width='stretch', hide_index=True)

    # Best model callout
    st.success(
        f"**Best model:** {summary['best_model']} — "
        f"{summary['primary_metric']}: {summary['best_score']:.4f}"
    )

    # Per-model details
    for r in results:
        with st.expander(f"{r.model_name} — Details"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Parameters:**")
                st.json(r.params)
            with c2:
                st.write("**Metrics:**")
                for k, v in r.metrics.items():
                    if isinstance(v, float):
                        st.write(f"- {k}: {v:.4f}")
                    else:
                        st.write(f"- {k}: {v}")
                if r.cv_mean is not None:
                    st.write(f"- CV mean: {r.cv_mean:.4f} (±{r.cv_std:.4f})")
                st.write(f"- Training time: {r.training_time:.2f}s")
                if r.artifact_path:
                    st.write(f"- Artifact: `{r.artifact_path}`")
                if r.mlflow_run_id:
                    st.write(f"- MLflow run: `{r.mlflow_run_id}`")
                else:
                    st.caption("MLflow: not logged")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Navigation
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
nav1, nav2 = st.columns(2)
with nav1:
    st.page_link("pages/3_Feature_Engineering.py", label="← Back to Feature Engineering")
with nav2:
    if "training_results" in st.session_state:
        st.page_link("pages/5_Results.py", label="Continue to Results →")
