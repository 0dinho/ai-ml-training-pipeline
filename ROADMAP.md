# Roadmap

Progress tracker for the MLOps AutoML Platform (expanded edition).
Each phase builds on the previous one.

---

## Phase 1 — Project Setup
> Foundation: repository structure, dependencies, and tooling.

- [x] Create project directory structure
- [x] Initialize Git repository
- [x] Initialize DVC
- [x] Set up virtual environment and `requirements.txt`
- [x] Plan the app as a multi-page Streamlit app
- [x] Create README.md and ROADMAP.md
- [x] **[NEW]** Restructure `src/` to separate supervised vs unsupervised model modules
- [x] **[NEW]** Add `umap-learn`, `shap`, `scipy`, `scikit-learn >= 1.4` to `requirements.txt`
- [x] **[NEW]** Update `configs/` with task-type-aware default config files

---

## Phase 2 — Dataset Upload & Exploration (Page 1)
> Let users upload data and understand it before choosing a task.

- [x] File uploader supporting CSV, Excel, JSON
- [x] Auto-detect column types (numerical, categorical, datetime, text)
- [x] Data preview table with pagination
- [x] Auto-generate EDA report (distributions, missing values, correlation, class balance)
- [x] Target column selector
- [x] Store dataset and metadata in session state
- [x] **[NEW]** Expanded task type selector:
  - [x] Regression
  - [x] Binary Classification
  - [x] Multiclass Classification
  - [x] Clustering
  - [x] Dimensionality Reduction
  - [x] Anomaly Detection
- [x] **[NEW]** Hide target column selector for fully unsupervised tasks (clustering, anomaly detection, dimensionality reduction)
- [x] **[NEW]** Show task-relevant EDA hints (e.g. class balance for classification, inlier/outlier ratio hint for anomaly detection)

---

## Phase 3 — Data Cleaning & Preprocessing (Page 2)
> Robust cleaning pipeline before any modeling.

- [x] Per-column imputation strategy
- [x] Encoding and scaling controls
- [x] "Auto Preprocess" button with smart defaults
- [x] Before/after data preview
- [x] Train/validation/test split configuration
- [x] Persist sklearn pipeline with DVC versioning
- [x] **[NEW]** Data quality diagnostics panel:
  - [x] Missing value ratio per column (visual bar chart)
  - [x] Duplicate row count and preview
  - [x] Constant / near-constant column detection
  - [x] Outlier detection summary (IQR / Z-score flagging)
- [x] **[NEW]** Cleaning actions:
  - [x] Drop or flag duplicates
  - [x] Low-variance filter (configurable threshold)
  - [x] High-correlation filter (drop one of correlated pair above threshold)
  - [ ] Value range validation (configurable min/max bounds per numerical column)
  - [ ] Regex-based validation for text/categorical columns
  - [x] KNN imputation as an additional strategy
- [x] **[NEW]** Disable train/val/test split UI for fully unsupervised tasks

---

## Phase 4 — Feature Engineering (Page 3) **[NEW PAGE]**
> Dimensionality reduction and feature transformation as a configurable step.

- [x] Technique selector:
  - [x] PCA — Principal Component Analysis
  - [x] LDA — Linear Discriminant Analysis (supervised tasks only)
  - [ ] CCA — Canonical Correlation Analysis
  - [x] NMF — Non-negative Matrix Factorization
  - [x] t-SNE (visualization only, not fed into training)
  - [x] UMAP (visualization + optionally fed into training)
- [x] Configure: number of components or explained variance threshold
- [x] Explained variance plot (scree plot for PCA)
- [x] Component loadings / feature contribution table
- [x] 2D / 3D scatter plot of transformed data (colored by target if supervised)
- [x] Toggle: "Use as preprocessing step before training" vs. "Run as the main task"
- [x] Persist transformed dataset and fitted transformer

---

## Phase 5 — Model Training (Page 4)
> Train the right algorithms for each task type.

- [x] Model selection UI
- [x] Per-model hyperparameter controls
- [x] "Auto Tune with Optuna" toggle
- [x] Cross-validation with configurable folds
- [x] Live training progress and logs
- [x] Automatic MLflow experiment logging
- [x] Multi-model session support
- [x] **[NEW]** Supervised — Regression models:
  - [x] Linear Regression, Ridge, Lasso, ElasticNet
  - [x] Decision Tree Regressor
  - [x] Random Forest Regressor
  - [x] XGBoost Regressor
  - [x] SVR (Support Vector Regressor)
  - [x] kNN Regressor
  - [x] Neural Network Regressor (MLP)
- [x] **[NEW]** Supervised — Classification models:
  - [x] Logistic Regression
  - [x] Naive Bayes (Gaussian, Multinomial, Bernoulli)
  - [x] SVM / SVC (linear and RBF kernel)
  - [x] Decision Tree Classifier
  - [x] Random Forest Classifier
  - [x] XGBoost Classifier
  - [x] kNN Classifier
  - [x] Neural Network Classifier (MLP / Softmax)
- [x] **[NEW]** Unsupervised — Clustering models:
  - [x] K-Means (with elbow method + silhouette scan for K)
  - [ ] Mean Shift
  - [x] Hierarchical / Agglomerative Clustering (configurable linkage)
  - [x] DBSCAN (eps and min_samples controls)
  - [x] Expectation-Maximization / Gaussian Mixture Model
- [x] **[NEW]** Unsupervised — Anomaly Detection models:
  - [x] Isolation Forest
  - [x] One-Class SVM
  - [ ] Autoencoder (PyTorch, reconstruction error threshold)
- [x] **[NEW]** Task-aware MLflow tags (`task_type`, `model_family`)

---

## Phase 6 — Results & Metrics Dashboard (Page 5)
> Evaluate and compare models with task-appropriate metrics and charts.

- [x] Supervised metrics table (accuracy, F1, RMSE, R², etc.)
- [x] Confusion matrix, ROC curve, feature importance, learning curves
- [x] MLflow experiment history
- [x] One-click "Promote to Registry"
- [x] **[NEW]** Unified metrics table that adapts to task type
- [x] **[NEW]** Classification metrics: Accuracy, Precision, Recall, F1, AUC-ROC, Log Loss, MCC
- [x] **[NEW]** Regression metrics: MSE, RMSE, MAE, R², Adjusted R²
- [x] **[NEW]** Clustering metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Inertia
- [x] **[NEW]** Anomaly detection metrics: contamination rate, anomaly score distribution, Precision@k
- [ ] **[NEW]** New visualizations:
  - [x] Cluster scatter plot with centroid overlays (2D projection if needed)
  - [x] Dendrogram for Hierarchical Clustering
  - [x] Elbow / silhouette curve (clustering)
  - [x] 2D/3D embedding plot (dimensionality reduction as main task)
  - [x] Anomaly score histogram with decision threshold line
  - [x] Precision-Recall curve (binary classification)
  - [x] SHAP summary plot and beeswarm chart (supervised models)
  - [x] Residual plot and actual-vs-predicted scatter (regression)

---

## Phase 7 — Prediction & Inference (Page 6)
> Use the trained model on new data.

- [x] Batch prediction via CSV upload
- [x] Single-row manual input form
- [x] Confidence scores
- [x] Download predictions as CSV
- [x] Load model from MLflow registry
- [x] **[NEW]** Task-aware prediction output:
  - [x] Classification → class label + class probabilities
  - [x] Regression → predicted value + optional confidence interval
  - [x] Clustering → cluster ID assignment for new points
  - [x] Anomaly Detection → anomaly flag + anomaly score per row
  - [x] Dimensionality Reduction → transformed coordinates

---

## Phase 8 — Model Serving API (FastAPI)
> Production-ready API with task-aware responses.

- [x] `/predict` single-row endpoint
- [x] `/predict/batch` multi-row endpoint
- [x] `/health` health check
- [x] Pydantic validation from training schema
- [x] Load active model from MLflow Model Registry
- [x] Prometheus metrics instrumentation
- [x] Dockerfile
- [x] **[NEW]** Task-aware response schema (classification, regression, clustering, anomaly)
- [x] **[NEW]** `/model/info` endpoint returning active model metadata and task type
- [ ] **[NEW]** Autoencoder serving: encode input → compute reconstruction error → return anomaly decision

---

## Phase 9 — Monitoring & Drift Detection (Page 7)
> Track model health across all task types.

- [x] Prometheus metrics (request count, latency, prediction distribution)
- [x] Data drift detection (KS test, PSI, chi-squared)
- [x] Drift alerts in UI
- [x] Grafana dashboards
- [x] **[NEW]** Concept drift monitoring for supervised models (prediction distribution shift over time)
- [x] **[NEW]** Cluster drift: detect when new data points fall outside existing cluster boundaries
- [x] **[NEW]** Anomaly rate tracking over time (sudden spike = potential concept drift)
- [x] **[NEW]** Per-feature drift score breakdown in the UI

---

## Phase 10 — Automated Retraining (Page 8)
> Close the loop for all task types.

- [x] "Retrain Now" button
- [x] Scheduled retraining via APScheduler
- [x] New experiments auto-logged to MLflow
- [x] Auto-compare new vs. current production model
- [x] Optional auto-promotion if new model is better
- [x] **[NEW]** Task-aware comparison logic:
  - [x] Supervised: compare primary metric (F1, RMSE, etc.)
  - [x] Clustering: compare silhouette score on validation data
  - [x] Anomaly: compare anomaly score distribution stability

---

## Phase 11 — Dockerization & Deployment
> One command to run the full stack.

- [x] `docker-compose.yml` with all services
- [x] `Dockerfile.streamlit` and `Dockerfile.api`
- [x] GitHub Actions CI/CD (lint, test, build, deploy)
- [x] `.env.example`
- [x] **[NEW]** Update Docker builds to include new dependencies (`umap-learn`, `shap`, etc.)
- [ ] **[NEW]** Verify end-to-end flow for each task type in CI (integration tests per task)
- [x] **[NEW]** Add task-type smoke tests to `pytest` suite
