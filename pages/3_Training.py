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
)

st.set_page_config(page_title="Training", page_icon="🏋️", layout="wide")

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

if "X_train" not in st.session_state or "y_train" not in st.session_state:
    st.warning(
        "Please preprocess your data on the **Preprocessing** page first."
    )
    st.stop()

X_train: np.ndarray = st.session_state["X_train"]
X_val: np.ndarray = st.session_state["X_val"]
X_test: np.ndarray = st.session_state["X_test"]
y_train = np.asarray(st.session_state["y_train"])
y_val = np.asarray(st.session_state["y_val"])
y_test = np.asarray(st.session_state["y_test"])
task_type: str = st.session_state.get("task_type", "classification")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Model Selection
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Model Selection")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    use_rf = st.checkbox("Random Forest", value=True, key="use_rf")
with col_m2:
    use_xgb = st.checkbox("XGBoost", value=True, key="use_xgb")
with col_m3:
    use_nn = st.checkbox("Neural Network", value=False, key="use_nn")

selected_models: list[str] = []
if use_rf:
    selected_models.append("random_forest")
if use_xgb:
    selected_models.append("xgboost")
if use_nn:
    selected_models.append("neural_network")

if not selected_models:
    st.info("Select at least one model above.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Cross-Validation Config
# ═══════════════════════════════════════════════════════════════════════════════
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
                rf_n_est = st.slider("n_estimators", 50, 500, 100, step=50, key="rf_n_est")
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
            xgb_trials = st.number_input("Optuna trials", 5, 200, 50, key="xgb_trials")
            xgb_timeout = st.number_input("Timeout (sec)", 10, 600, 300, key="xgb_timeout")
            model_configs["xgboost"] = {
                "auto_tune": True, "n_trials": int(xgb_trials), "timeout": int(xgb_timeout),
            }
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                xgb_n_est = st.slider("n_estimators", 50, 500, 100, step=50, key="xgb_n_est")
                xgb_max_depth = st.slider("max_depth", 3, 15, 6, key="xgb_max_depth")
                xgb_lr = st.slider("learning_rate", 0.01, 0.3, 0.1, step=0.01, key="xgb_lr")
            with c2:
                xgb_subsample = st.slider("subsample", 0.5, 1.0, 0.8, step=0.05, key="xgb_subsample")
                xgb_colsample = st.slider("colsample_bytree", 0.5, 1.0, 0.8, step=0.05, key="xgb_colsample")
                xgb_min_child = st.slider("min_child_weight", 1, 10, 3, key="xgb_min_child")
            with c3:
                xgb_gamma = st.slider("gamma", 0.0, 5.0, 0.0, step=0.1, key="xgb_gamma")
                xgb_alpha = st.slider("reg_alpha", 0.0, 10.0, 0.0, step=0.5, key="xgb_alpha")
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

# ── Neural Network ────────────────────────────────────────────────────────────
if "neural_network" in selected_models:
    with st.expander("Neural Network", expanded=True):
        auto_nn = st.toggle("Auto Tune with Optuna", key="auto_nn")
        if auto_nn:
            nn_trials = st.number_input("Optuna trials", 5, 200, 50, key="nn_trials")
            nn_timeout = st.number_input("Timeout (sec)", 10, 600, 300, key="nn_timeout")
            model_configs["neural_network"] = {
                "auto_tune": True, "n_trials": int(nn_trials), "timeout": int(nn_timeout),
            }
        else:
            c1, c2 = st.columns(2)
            with c1:
                nn_n_layers = st.slider("Hidden layers", 1, 3, 2, key="nn_n_layers")
                nn_layers = []
                for i in range(nn_n_layers):
                    size = st.slider(
                        f"Layer {i+1} size", 16, 256, 64 if i == 0 else 32,
                        step=16, key=f"nn_layer_{i}",
                    )
                    nn_layers.append(size)
                nn_activation = st.selectbox(
                    "Activation", ["relu", "tanh", "leaky_relu"], key="nn_activation",
                )
            with c2:
                nn_dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05, key="nn_dropout")
                nn_lr = st.slider("Learning rate", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f", key="nn_lr")
                nn_batch = st.selectbox("Batch size", [16, 32, 64, 128], index=1, key="nn_batch")
                nn_epochs = st.slider("Epochs", 10, 200, 50, key="nn_epochs")
                nn_patience = st.slider("Early stopping patience", 3, 20, 5, key="nn_patience")
            model_configs["neural_network"] = {
                "auto_tune": False,
                "params": {
                    "hidden_layers": nn_layers,
                    "activation": nn_activation,
                    "dropout": nn_dropout,
                    "learning_rate": nn_lr,
                    "batch_size": nn_batch,
                    "epochs": nn_epochs,
                    "early_stopping_patience": nn_patience,
                },
            }

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Train Models
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Train Models")

if st.button("🏋️ Train Models", type="primary"):
    results = []

    for model_type in selected_models:
        cfg = model_configs[model_type]
        display_name = {
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
            "neural_network": "Neural Network",
        }[model_type]

        st.subheader(f"Training {display_name}")

        # Step 1: Optuna tuning (if enabled)
        if cfg["auto_tune"]:
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
            params = cfg["params"]

        # Step 2: Train model
        train_bar = st.progress(0, text="Training...")

        if model_type == "neural_network":
            def progress_cb(epoch, total, loss):
                pct = epoch / total
                train_bar.progress(pct, text=f"Epoch {epoch}/{total} — loss: {loss:.4f}")
        else:
            progress_cb = None

        result = train_model(
            model_type, task_type, params,
            X_train, y_train, X_val, y_val,
            cv_folds=cv_folds,
            progress_callback=progress_cb,
        )
        train_bar.progress(1.0, text="Training complete!")

        # Step 3: Save model artifact
        artifact_path = save_model(result)
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

    # Step 5: Store in session state
    st.session_state["training_results"] = results
    st.session_state["trained_models"] = {r.model_type: r.model for r in results}
    st.session_state["training_summary"] = get_training_summary(results)
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
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

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
                    st.write(f"- {k}: {v:.4f}")
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
    st.page_link("pages/2_Preprocessing.py", label="← Back to Preprocessing")
with nav2:
    if "training_results" in st.session_state:
        st.page_link("pages/4_Results.py", label="Continue to Results →")
