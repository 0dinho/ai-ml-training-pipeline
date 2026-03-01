# MLOps AutoML Platform

A production-grade, end-to-end machine learning platform covering the full spectrum of supervised and unsupervised ML — with automated preprocessing, feature engineering, training, experiment tracking, model serving, and monitoring, all accessible through an interactive Streamlit UI.

Upload a dataset, select your task type, configure your pipeline, train and compare models, serve predictions via API, and monitor for drift — no code required.

---

## Supported Task Types

| Category | Tasks |
|---|---|
| **Supervised — Regression** | Linear / Polynomial Regression, Ridge, Lasso, ElasticNet, SVR, Decision Tree, Random Forest, XGBoost, kNN, Neural Network |
| **Supervised — Binary Classification** | Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, XGBoost, kNN, Neural Network |
| **Supervised — Multiclass Classification** | Same as binary + One-vs-Rest / One-vs-One wrappers, Softmax Neural Network |
| **Unsupervised — Clustering** | K-Means, Mean Shift, Hierarchical (Agglomerative), DBSCAN, Expectation-Maximization (GMM) |
| **Unsupervised — Dimensionality Reduction** | PCA, LDA, CCA, NMF, t-SNE, UMAP |
| **Unsupervised — Anomaly Detection** | Isolation Forest, One-Class SVM, Autoencoder |

---

## Features

### Page 1 — Dataset Upload & Exploration
- Upload CSV, Excel, or JSON files
- Auto-detect column types: numerical, categorical, datetime, text
- Interactive EDA: distributions, missing value heatmap, correlation matrix, class balance
- **Task type selector**: regression, binary classification, multiclass classification, clustering, dimensionality reduction, anomaly detection
- Target column selector (hidden for fully unsupervised tasks)
- Dataset summary stats stored in session state

### Page 2 — Data Cleaning & Preprocessing
- **Quality diagnostics**: missing value ratio per column, duplicate row detection, constant/near-constant columns
- **Cleaning actions**:
  - Drop or impute missing values (mean, median, mode, constant, KNN imputation)
  - Remove duplicate rows
  - Fix incorrect values via rule-based filters (configurable min/max bounds, regex for text)
  - Low-variance filter (threshold configurable)
  - High-correlation filter (drop one of a correlated pair above threshold)
- Per-column encoding: one-hot, label, ordinal, target encoding
- Per-column scaling: standard, min-max, robust, none
- "Auto Clean" button with smart defaults
- Before/after data preview
- Train / validation / test split (configurable ratios; split disabled for unsupervised tasks)
- Persist sklearn pipeline with DVC versioning

### Page 3 — Feature Engineering
- **Dimensionality reduction as a preprocessing step** (distinct from the standalone task):
  - PCA — Principal Component Analysis
  - LDA — Linear Discriminant Analysis (supervised tasks only)
  - CCA — Canonical Correlation Analysis
  - NMF — Non-negative Matrix Factorization
- Configure number of components / variance threshold
- Visualize explained variance and component loadings
- Toggle: apply reduction before training or use as the main task
- Transformed dataset preview

### Page 4 — Model Training
- **Algorithm catalog by task type** (see Supported Task Types above)
- Per-model hyperparameter controls (sliders, dropdowns, numeric inputs)
- "Auto Tune with Optuna" toggle per model
- Cross-validation with configurable folds (supervised tasks)
- Elbow method / silhouette scan for K selection (clustering)
- Live training progress bar and log output in the UI
- Automatic MLflow experiment logging: params, metrics, artifacts, tags
- Train multiple models in one session for side-by-side comparison

### Page 5 — Results & Metrics Dashboard
- **Supervised metrics**:
  - Regression: MSE, RMSE, MAE, R², Adjusted R²
  - Classification: Accuracy, Precision, Recall, F1, AUC-ROC, Log Loss, MCC
- **Unsupervised metrics**:
  - Clustering: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Inertia
  - Anomaly Detection: Precision@k, contamination rate, anomaly score distribution
- **Visualizations**:
  - Confusion matrix, ROC curve, Precision-Recall curve (classification)
  - Residual plots, actual vs. predicted (regression)
  - Cluster scatter plots with centroid overlays (clustering)
  - 2D/3D embedding plots (dimensionality reduction)
  - Anomaly score histogram with threshold line (anomaly detection)
  - Feature importance, SHAP summary, learning curves (supervised)
- MLflow experiment history: sortable, filterable, diff-able across runs
- One-click "Promote to Registry" for the best model

### Page 6 — Prediction & Inference
- **Batch prediction**: upload a new CSV, get predictions + confidence scores
- **Single prediction**: auto-generated input form from training schema
- Download predictions as CSV
- Clustering: assign new points to existing clusters
- Anomaly detection: score new observations and flag anomalies
- Load active model from MLflow Model Registry

### Page 7 — Model Serving API
- FastAPI `/predict` endpoint (single row)
- `/predict/batch` endpoint (multiple rows)
- `/health` health check
- Pydantic input validation auto-generated from training schema
- Task-aware response: class + probability (classification), value (regression), cluster ID (clustering), anomaly flag + score (anomaly detection)
- Prometheus metrics: request count, latency, prediction distribution
- Dockerfile for the API service

### Page 8 — Monitoring & Drift Detection
- Prometheus metrics: request count, latency (p50/p95/p99), prediction distribution
- **Data drift detection** (new data vs. training baseline):
  - KS test (numerical), Chi-squared test (categorical), PSI
- **Concept drift** indicators for supervised models
- Grafana pre-built dashboards
- Native Streamlit monitoring page as fallback
- Drift alerts displayed in the UI

### Page 9 — Automated Retraining
- "Retrain Now" button (triggered from monitoring page on drift alert)
- Scheduled retraining via APScheduler
- New run auto-logged to MLflow
- Auto-comparison: new model vs. current production model
- Optional auto-promotion if new model outperforms

---

## Tech Stack

| Layer | Tools |
|---|---|
| UI | Streamlit (multi-page app) |
| ML — Supervised | Scikit-learn, XGBoost, PyTorch |
| ML — Unsupervised | Scikit-learn (KMeans, DBSCAN, GMM, IsolationForest, One-Class SVM), PyTorch (Autoencoder) |
| Dimensionality Reduction | Scikit-learn (PCA, LDA, NMF), UMAP-learn |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API Serving | FastAPI, Uvicorn |
| Monitoring | Prometheus, Grafana |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## Project Structure

```
MLOps/
├── app.py                          # Streamlit entrypoint
├── pages/
│   ├── 1_Upload_Explore.py
│   ├── 2_Cleaning_Preprocessing.py
│   ├── 3_Feature_Engineering.py
│   ├── 4_Training.py
│   ├── 5_Results.py
│   ├── 6_Prediction.py
│   ├── 7_Monitoring.py
│   └── 8_Retraining.py            # (was integrated in monitoring)
├── src/
│   ├── pipelines/
│   │   ├── cleaning.py            # Data quality + cleaning logic
│   │   ├── preprocessing.py       # Encoding, scaling, splitting
│   │   ├── feature_engineering.py # PCA, LDA, CCA, NMF, UMAP
│   │   └── training.py            # Model dispatch by task type
│   ├── models/
│   │   ├── supervised/
│   │   │   ├── classifiers.py
│   │   │   └── regressors.py
│   │   └── unsupervised/
│   │       ├── clustering.py
│   │       ├── reduction.py
│   │       └── anomaly.py
│   ├── evaluation/
│   │   ├── supervised_metrics.py
│   │   └── unsupervised_metrics.py
│   ├── utils/
│   │   ├── schema.py
│   │   ├── mlflow_helpers.py
│   │   └── session.py
│   └── monitoring/
│       ├── drift.py
│       └── alerting.py
├── components/                     # Reusable Streamlit UI components
├── configs/                        # YAML/JSON config files
├── docker/
│   ├── Dockerfile.streamlit
│   ├── Dockerfile.api
│   ├── grafana/dashboards/
│   └── prometheus/
├── tests/
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
├── artifacts/
├── notebooks/
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .github/workflows/
└── README.md
```

---

## Prerequisites

- Python 3.10+ (Conda recommended)
- Docker & Docker Compose
- Git

## Quick Start (Local Development)

```bash
git clone https://github.com/<your-username>/MLOps.git
cd MLOps

conda create -n mlops python=3.11 -y
conda activate mlops

pip install -r requirements.txt

dvc init

mlflow server --host 0.0.0.0 --port 5001 &

streamlit run app.py
```

App available at `http://localhost:8501`.

## Deployment (Docker Compose)

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Streamlit App | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| FastAPI Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## API Usage

### Health check
```bash
curl http://localhost:8000/health
```

### Single prediction (classification)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 5.1, "feature_2": 3.5, "feature_3": 1.4}'
```

### Batch prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"feature_1": 5.1, "feature_2": 3.5}, {"feature_1": 6.2, "feature_2": 2.8}]'
```

Response shape adapts to task type:
- **Classification**: `{ "prediction": "class_A", "probabilities": { "class_A": 0.87, ... } }`
- **Regression**: `{ "prediction": 42.3 }`
- **Clustering**: `{ "cluster_id": 2 }`
- **Anomaly Detection**: `{ "is_anomaly": true, "anomaly_score": -0.34 }`

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5001` |
| `MODEL_REGISTRY_NAME` | MLflow registered model name | `automl-model` |
| `API_HOST` | FastAPI host | `0.0.0.0` |
| `API_PORT` | FastAPI port | `8000` |
| `PROMETHEUS_PORT` | Prometheus metrics port | `9090` |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'Add my feature'`)
4. Push (`git push origin feature/my-feature`)
5. Open a Pull Request

## License

MIT