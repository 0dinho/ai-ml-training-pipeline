# MLOps AutoML Platform

A production-grade, end-to-end machine learning platform with automated training, experiment tracking, model versioning, serving, and monitoring — all accessible through an interactive Streamlit UI.

Upload a dataset, configure preprocessing, train multiple models, compare results, serve predictions via API, and monitor for drift — no code required.

## Features

### Dataset Upload & Exploration
- Upload CSV, Excel, or JSON files
- Auto-detect column types (numerical, categorical, datetime, text)
- Interactive EDA: distributions, missing values heatmap, correlation matrix, class balance
- Select target column and task type (classification / regression)

### Data Preprocessing
- Per-column configuration: imputation, encoding, scaling
- One-click "Auto Preprocess" with smart defaults
- Before/after data preview
- Configurable train/validation/test split
- Dataset versioning with DVC

### Model Training
- Algorithms: Random Forest, XGBoost, Neural Network
- Hyperparameter tuning via sliders or automated with Optuna
- Live training progress and logs in the UI
- Cross-validation with configurable folds
- Every run tracked automatically in MLflow

### Results Dashboard
- Side-by-side model comparison (accuracy, F1, RMSE, R², etc.)
- Confusion matrix, ROC curve, feature importance, learning curves
- Full MLflow experiment history
- One-click model promotion to the registry

### Prediction & Inference
- Batch prediction via CSV upload
- Single-row manual input
- Confidence scores
- Downloadable prediction results

### Model Serving API
- FastAPI `/predict` endpoint
- Auto-generated Pydantic validation from training schema
- Loads the active registered model from MLflow

### Monitoring & Drift Detection
- Request count, latency, prediction distribution over time
- Data drift detection (new data vs. training distribution)
- Alerts displayed in the UI
- Prometheus metrics + Grafana dashboards

### Automated Retraining
- Trigger retraining from the UI when drift is detected
- Scheduled retraining via background jobs
- New experiments auto-logged to MLflow

## Tech Stack

| Layer | Tools |
|---|---|
| UI | Streamlit (multi-page app) |
| ML | Scikit-learn, XGBoost, PyTorch |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API Serving | FastAPI, Uvicorn |
| Monitoring | Prometheus, Grafana |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |

## Project Structure

```
MLOps/
├── app.py                     # Streamlit entrypoint
├── pages/                     # Streamlit multi-page app
│   ├── 1_Upload_Explore.py
│   ├── 2_Preprocessing.py
│   ├── 3_Training.py
│   ├── 4_Results.py
│   ├── 5_Prediction.py
│   └── 6_Monitoring.py
├── src/
│   ├── pipelines/             # Data & training pipelines
│   ├── models/                # Model definitions & wrappers
│   ├── utils/                 # Helpers (metrics, schemas, etc.)
│   └── monitoring/            # Drift detection & alerting
├── components/                # Reusable Streamlit UI components
├── configs/                   # YAML/JSON config files
├── docker/
│   ├── Dockerfile.streamlit
│   ├── Dockerfile.api
│   ├── grafana/dashboards/
│   └── prometheus/
├── tests/                     # Unit and integration tests
├── data/
│   ├── raw/                   # Original uploaded datasets
│   ├── processed/             # Preprocessed datasets
│   └── predictions/           # Prediction outputs
├── artifacts/                 # Trained models, scalers, encoders
├── notebooks/                 # Exploration notebooks
├── docker-compose.yml
├── requirements.txt
├── .github/workflows/         # CI/CD pipelines
└── README.md
```

## Prerequisites

- Python 3.10+ (Conda recommended)
- Docker & Docker Compose
- Git

## Quick Start (Local Development)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/MLOps.git
cd MLOps
```

### 2. Create a Conda environment

```bash
conda create -n mlops python=3.11 -y
conda activate mlops
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize DVC (already done if you cloned the repo)

```bash
dvc init
```

### 5. Start MLflow tracking server

```bash
mlflow server --host 0.0.0.0 --port 5000 &
```

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Deployment (Docker Compose)

Spin up the full stack with a single command:

```bash
docker-compose up --build
```

This starts:

| Service | URL |
|---|---|
| Streamlit App | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| FastAPI Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

To stop all services:

```bash
docker-compose down
```

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

## CI/CD

GitHub Actions workflows handle:

- **Linting**: `flake8` and `black` checks on every push
- **Testing**: `pytest` runs the full test suite
- **Build**: Docker image build and push to registry
- **Deploy**: Auto-deploy to VPS on merge to `main`

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |
| `MODEL_REGISTRY_NAME` | MLflow registered model name | `automl-model` |
| `API_HOST` | FastAPI host | `0.0.0.0` |
| `API_PORT` | FastAPI port | `8000` |
| `PROMETHEUS_PORT` | Prometheus metrics port | `9090` |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## License

MIT
