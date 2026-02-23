# Roadmap

Progress tracker for the MLOps AutoML Platform. Each phase builds on the previous one.

---

## Phase 1 — Project Setup
> Foundation: repository structure, dependencies, and tooling.

- [x] Create project directory structure (`src/`, `pages/`, `components/`, `configs/`, `docker/`, etc.)
- [x] Initialize Git repository
- [x] Initialize DVC
- [x] Set up virtual environment and `requirements.txt`
- [x] Plan the app as a multi-page Streamlit app
- [x] Create README.md and ROADMAP.md

---

## Phase 2 — Dataset Upload & Exploration (Streamlit Page 1)
> Let users upload data and understand it before modeling.

- [x] File uploader supporting CSV, Excel, JSON
- [x] Auto-detect column types (numerical, categorical, datetime, text)
- [x] Data preview table with pagination
- [x] Auto-generate EDA report:
  - [x] Distribution plots for numerical columns
  - [x] Missing values heatmap
  - [x] Correlation matrix
  - [x] Class balance chart for the target column
- [x] Target column selector
- [x] Task type selector (classification / regression)
- [x] Store dataset and metadata in session state

---

## Phase 3 — Data Preprocessing Pipeline (Streamlit Page 2)
> Configurable preprocessing with smart defaults.

- [x] Per-column configuration UI:
  - [x] Imputation strategy (mean, median, mode, drop, constant)
  - [x] Encoding method (one-hot, label, ordinal, target encoding)
  - [x] Scaling method (standard, min-max, robust, none)
- [x] "Auto Preprocess" button with smart defaults
- [x] Before/after data preview
- [x] Train/validation/test split ratio configuration
- [x] Build and persist sklearn preprocessing pipeline
- [x] Version processed dataset with DVC

---

## Phase 4 — Model Training (Streamlit Page 3)
> Train multiple algorithms with optional auto-tuning.

- [x] Model selection: Random Forest, XGBoost, Neural Network
- [x] Per-model hyperparameter controls (sliders, inputs)
- [x] "Auto Tune with Optuna" toggle per model
- [x] Cross-validation with configurable folds
- [x] Live training progress bar and log output in the UI
- [x] Automatic MLflow experiment logging (params, metrics, artifacts)
- [x] Support training multiple models in one session

---

## Phase 5 — Results & Metrics Dashboard (Streamlit Page 4)
> Compare models and promote the best one.

- [x] Metrics summary table (side-by-side comparison)
  - [x] Classification: accuracy, precision, recall, F1, AUC-ROC
  - [x] Regression: MSE, RMSE, MAE, R²
- [x] Visualizations:
  - [x] Confusion matrix
  - [x] ROC curve
  - [x] Feature importance chart
  - [x] Learning curves
  - [x] Residual plots (regression)
- [x] MLflow experiment history table (sortable, filterable)
- [x] One-click "Promote to Registry" button for the best model

---

## Phase 6 — Prediction & Inference (Streamlit Page 5)
> Use the trained model on new data.

- [x] Batch prediction: upload a new CSV
- [x] Single prediction: manual row input form (auto-generated from training schema)
- [x] Display predictions with confidence scores / probabilities
- [x] Download predictions as CSV
- [x] Load model from MLflow registry

---

## Phase 7 — Model Serving API
> Production-ready API for real-time inference.

- [x] FastAPI app with `/predict` endpoint
- [x] `/predict/batch` endpoint for multiple rows
- [x] `/health` health check endpoint
- [x] Pydantic input validation (schema auto-generated from training columns)
- [x] Load active model from MLflow Model Registry
- [x] Prometheus metrics instrumentation (request count, latency, prediction distribution)
- [x] Dockerfile for the API service

---

## Phase 8 — Monitoring & Drift Detection (Streamlit Page 6)
> Track model health and detect when retraining is needed.

- [x] Prometheus metrics collection:
  - [x] Request count
  - [x] Prediction latency (p50, p95, p99)
  - [x] Prediction distribution over time
- [x] Data drift detection:
  - [x] Compare new data distributions to training data
  - [x] Statistical tests (KS test, PSI, chi-squared)
  - [x] Drift alerts displayed in UI
- [x] Grafana dashboard with pre-built panels
- [x] Native Streamlit monitoring page as fallback

---

## Phase 9 — Automated Retraining
> Close the loop: detect drift, retrain, redeploy.

- [x] "Retrain Now" button in the UI (triggered when drift is detected)
- [x] Scheduled retraining via background job (APScheduler or cron)
- [x] New experiment auto-logged to MLflow
- [x] Auto-compare new model vs. current production model
- [x] Optional auto-promotion if new model performs better

---

## Phase 10 — Dockerization & Deployment
> One command to run everything.

- [ ] `docker-compose.yml` with all services:
  - [ ] Streamlit app
  - [ ] FastAPI API
  - [ ] MLflow tracking server
  - [ ] Prometheus
  - [ ] Grafana
- [ ] `Dockerfile.streamlit`
- [ ] `Dockerfile.api`
- [ ] GitHub Actions CI/CD:
  - [ ] Lint (`flake8`, `black`)
  - [ ] Test (`pytest`)
  - [ ] Build Docker image
  - [ ] Deploy to VPS on merge to `main`
- [ ] Production environment configuration
- [ ] Deploy and verify on VPS
