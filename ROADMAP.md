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

- [ ] Per-column configuration UI:
  - [ ] Imputation strategy (mean, median, mode, drop, constant)
  - [ ] Encoding method (one-hot, label, ordinal, target encoding)
  - [ ] Scaling method (standard, min-max, robust, none)
- [ ] "Auto Preprocess" button with smart defaults
- [ ] Before/after data preview
- [ ] Train/validation/test split ratio configuration
- [ ] Build and persist sklearn preprocessing pipeline
- [ ] Version processed dataset with DVC

---

## Phase 4 — Model Training (Streamlit Page 3)
> Train multiple algorithms with optional auto-tuning.

- [ ] Model selection: Random Forest, XGBoost, Neural Network
- [ ] Per-model hyperparameter controls (sliders, inputs)
- [ ] "Auto Tune with Optuna" toggle per model
- [ ] Cross-validation with configurable folds
- [ ] Live training progress bar and log output in the UI
- [ ] Automatic MLflow experiment logging (params, metrics, artifacts)
- [ ] Support training multiple models in one session

---

## Phase 5 — Results & Metrics Dashboard (Streamlit Page 4)
> Compare models and promote the best one.

- [ ] Metrics summary table (side-by-side comparison)
  - [ ] Classification: accuracy, precision, recall, F1, AUC-ROC
  - [ ] Regression: MSE, RMSE, MAE, R²
- [ ] Visualizations:
  - [ ] Confusion matrix
  - [ ] ROC curve
  - [ ] Feature importance chart
  - [ ] Learning curves
  - [ ] Residual plots (regression)
- [ ] MLflow experiment history table (sortable, filterable)
- [ ] One-click "Promote to Registry" button for the best model

---

## Phase 6 — Prediction & Inference (Streamlit Page 5)
> Use the trained model on new data.

- [ ] Batch prediction: upload a new CSV
- [ ] Single prediction: manual row input form (auto-generated from training schema)
- [ ] Display predictions with confidence scores / probabilities
- [ ] Download predictions as CSV
- [ ] Load model from MLflow registry

---

## Phase 7 — Model Serving API
> Production-ready API for real-time inference.

- [ ] FastAPI app with `/predict` endpoint
- [ ] `/predict/batch` endpoint for multiple rows
- [ ] `/health` health check endpoint
- [ ] Pydantic input validation (schema auto-generated from training columns)
- [ ] Load active model from MLflow Model Registry
- [ ] Prometheus metrics instrumentation (request count, latency, prediction distribution)
- [ ] Dockerfile for the API service

---

## Phase 8 — Monitoring & Drift Detection (Streamlit Page 6)
> Track model health and detect when retraining is needed.

- [ ] Prometheus metrics collection:
  - [ ] Request count
  - [ ] Prediction latency (p50, p95, p99)
  - [ ] Prediction distribution over time
- [ ] Data drift detection:
  - [ ] Compare new data distributions to training data
  - [ ] Statistical tests (KS test, PSI, chi-squared)
  - [ ] Drift alerts displayed in UI
- [ ] Grafana dashboard with pre-built panels
- [ ] Native Streamlit monitoring page as fallback

---

## Phase 9 — Automated Retraining
> Close the loop: detect drift, retrain, redeploy.

- [ ] "Retrain Now" button in the UI (triggered when drift is detected)
- [ ] Scheduled retraining via background job (APScheduler or cron)
- [ ] New experiment auto-logged to MLflow
- [ ] Auto-compare new model vs. current production model
- [ ] Optional auto-promotion if new model performs better

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
