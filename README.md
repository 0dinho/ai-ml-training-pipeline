# MLOps AutoML Platform

An end-to-end machine learning platform covering supervised and unsupervised ML — automated preprocessing, feature engineering, training, experiment tracking, model serving, and monitoring, all through an interactive Streamlit UI.

Upload a dataset, select your task type, configure your pipeline, train and compare models, serve predictions via API, and monitor for drift — no code required.

---

## Supported Task Types & Models

### Supervised

| Task | Models |
|---|---|
| **Binary Classification** | Random Forest, XGBoost, Logistic Regression, Gaussian NB, Bernoulli NB, SVM (RBF), SVM (Linear), Decision Tree, k-NN |
| **Multiclass Classification** | Random Forest, XGBoost, Logistic Regression, Gaussian NB, Bernoulli NB, SVM (RBF), SVM (Linear), Decision Tree, k-NN |
| **Regression** | Random Forest, XGBoost, Linear Regression, Ridge, Lasso, Elastic Net, Decision Tree, SVR, k-NN, Gradient Boosting |

### Unsupervised

| Task | Models |
|---|---|
| **Clustering** | K-Means, Mean Shift, DBSCAN, Agglomerative, Gaussian Mixture |
| **Anomaly Detection** | Isolation Forest, One-Class SVM, Local Outlier Factor, Elliptic Envelope |
| **Dimensionality Reduction** | PCA, UMAP, t-SNE, Truncated SVD |

---

## Features

### Page 1 — Dataset Upload & Exploration
- Upload CSV, Excel, or JSON files
- Auto-detect column types (numerical, categorical, datetime, text)
- Interactive EDA: distributions, missing values, correlation matrix, class balance
- Task type auto-detection with manual override
- Target column selector (hidden for unsupervised tasks)

### Page 2 — Preprocessing
- **Quality diagnostics**: missing value ratio, duplicate rows, near-constant columns, outlier detection
- **Imputation**: mean, median, mode, constant, KNN
- **Encoding**: one-hot, ordinal, target encoding
- **Scaling**: standard, min-max, robust
- Low-variance filter, high-correlation filter
- Train / validation / test split (stratified for classification; disabled for unsupervised)
- Persists sklearn pipeline + schema to `artifacts/`

### Page 3 — Feature Engineering
- **Dimensionality reduction**: PCA, LDA (supervised only), NMF, t-SNE, UMAP
- **Feature transforms**: polynomial, log, sqrt, quantile binning
- Explained variance visualization and component loadings
- Transformed dataset preview

### Page 4 — Model Training
- Select one or more models from the task-specific catalog
- Per-model hyperparameter controls (sliders, dropdowns)
- Optional Optuna hyperparameter tuning per model
- Cross-validation with configurable folds (supervised)
- Real-time progress with per-model status indicators
- Automatic MLflow experiment logging (params, metrics, artifacts)

### Page 5 — Results & Metrics
- **Classification**: accuracy, precision, recall, F1, ROC-AUC, MCC, confusion matrix, ROC curves
- **Regression**: MSE, RMSE, MAE, R²
- **Clustering**: silhouette, Davies-Bouldin, Calinski-Harabasz, elbow curve, centroid overlays, dendrogram
- **Anomaly Detection**: anomaly ratio, score distribution
- **Dimensionality Reduction**: explained variance, 2D scatter
- SHAP beeswarm + bar charts (tree-based models)
- Learning curves, model comparison table
- Promote best model to session for prediction/serving

### Page 6 — Prediction & Inference
- **Single prediction**: auto-generated form from training schema
- **Batch prediction**: upload CSV, download results
- Task-aware output: probabilities (classification), cluster labels, anomaly scores, reduced coordinates
- Export predictions as CSV

### Page 7 — Monitoring & Retraining
- **API health**: live status from FastAPI `/health` endpoint
- **Prometheus metrics**: request count, latency (p50/p95/p99), prediction distribution
- **Data drift detection**: KS test (numerical), Chi-squared (categorical), PSI
- **Concept drift monitoring**: prediction distribution comparison over time
- **Retraining**: one-click retrain, scheduled retraining via APScheduler, auto-comparison with current model, optional auto-promotion

---

## Tech Stack

| Layer | Tools |
|---|---|
| UI | Streamlit |
| ML | scikit-learn, XGBoost |
| Dimensionality Reduction | scikit-learn (PCA, LDA, NMF), umap-learn |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API | FastAPI, Uvicorn, Pydantic |
| Monitoring | Prometheus, Grafana |
| Containerization | Docker, Docker Compose |

---

## Project Structure

```
MLOps/
├── app.py                            # Streamlit entrypoint
├── pages/
│   ├── 1_Upload_Explore.py
│   ├── 2_Preprocessing.py
│   ├── 3_Feature_Engineering.py
│   ├── 4_Training.py
│   ├── 5_Results.py
│   ├── 6_Prediction.py
│   └── 7_Monitoring.py
├── src/
│   ├── models/
│   │   ├── adapter.py                # ModelAdapter — unified inference wrapper
│   │   ├── classifiers.py            # Classification model factories
│   │   ├── regressors.py             # Regression model factories
│   │   ├── clustering.py             # ClusteringAdapter
│   │   ├── anomaly.py                # AnomalyAdapter
│   │   └── reduction.py              # ReductionAdapter
│   ├── pipelines/
│   │   ├── preprocessing.py          # Encoding, scaling, splitting
│   │   ├── training.py               # Model dispatch, training, MLflow logging
│   │   └── retraining.py             # Automated retraining orchestrator
│   ├── evaluation/
│   │   ├── metrics.py                # Unified metrics for all 6 task types
│   │   └── plots.py                  # Plotly visualization helpers
│   ├── monitoring/
│   │   └── drift.py                  # KS, Chi-squared, PSI drift tests
│   └── utils/
│       └── data_utils.py             # Dataset loading, column detection, EDA
├── api/
│   ├── main.py                       # FastAPI app (health, predict, batch)
│   ├── schemas.py                    # Pydantic request/response models
│   ├── metrics.py                    # Prometheus metric definitions
│   └── model_loader.py              # MLflow / disk model loading
├── configs/
│   ├── config.yaml                   # Global defaults
│   └── tasks/                        # Per-task-type configs (6 YAML files)
├── docker/
│   ├── Dockerfile.streamlit
│   ├── Dockerfile.api
│   ├── Dockerfile.mlflow
│   ├── prometheus/prometheus.yml
│   └── grafana/                      # Provisioning + dashboards
├── tests/                            # pytest suite
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
├── artifacts/                        # Saved pipelines, models, schemas
├── docker-compose.yml
├── retraining_scheduler.py           # Standalone APScheduler process
├── requirements.txt
└── .env.example
```

---

## Quick Start

### Local Development

```bash
git clone https://github.com/<your-username>/MLOps.git
cd MLOps

conda create -n mlops python=3.11 -y
conda activate mlops

pip install -r requirements.txt

# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5001 &

# Launch the app
streamlit run app.py
```

App at [http://localhost:8501](http://localhost:8501), MLflow UI at [http://localhost:5001](http://localhost:5001).

### Docker Compose

```bash
cp .env.example .env    # adjust values if needed
docker compose up --build
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

### Single prediction
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

Response adapts to task type:
- **Classification**: `{ "prediction": "class_A", "probabilities": { "class_A": 0.87, ... } }`
- **Regression**: `{ "prediction": 42.3 }`
- **Clustering**: `{ "cluster_id": 2 }`
- **Anomaly Detection**: `{ "is_anomaly": true, "anomaly_score": -0.34 }`

---

## Environment Variables

See [`.env.example`](.env.example) for all configurable variables:

| Variable | Description | Default |
|---|---|---|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://mlflow:5000` |
| `MLFLOW_MODEL_NAME` | Registered model name in MLflow | `automl_model` |
| `GRAFANA_PASSWORD` | Grafana admin password | `admin` |
| `RETRAIN_INTERVAL_HOURS` | Hours between scheduled retraining runs | `24` |
| `RETRAIN_IMPROVEMENT_THRESHOLD` | Min metric improvement to auto-promote | `0.01` |
| `RETRAIN_AUTO_PROMOTE` | Auto-promote better models | `true` |

---

## License

MIT
