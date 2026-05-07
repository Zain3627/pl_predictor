# Premier League Predictor

A machine learning project that predicts Premier League match outcomes and final league standings using historical match data. Orchestrated with [ZenML](https://zenml.io/), tracked with [MLflow](https://mlflow.org/), and deployed on AWS EC2 with automated weekly evaluation.

## Overview

This project fetches historical Premier League match data and upcoming fixture data from external APIs, engineers features from match statistics, trains a classification model, predicts the results of remaining fixtures to produce a projected final league table, and evaluates live predictions against real results on a weekly schedule.

**Supported models** (configurable in `src/config.py`):
- Logistic Regression
- Random Forest
- XGBoost (default)

---

## Project Structure

```
pl_predictor/
│
├── run_fetch_data_pipeline.py        # Entry point – fetch & clean data
├── run_make_predictions.py           # Entry point – train, evaluate, predict
├── run_live_evaluation_pipeline.py   # Entry point – evaluate live predictions
├── streamlit_app.py                  # Web UI
│
├── Dockerfile.pipelines              # Docker image for all three pipelines
│
├── scripts/
│   ├── setup_zenml.sh                # Registers ZenML stack at container runtime
│   ├── pipelineswrapper.sh           # Runs all three pipelines sequentially
│   └── start_mlflow_server.sh        # Starts MLflow server on EC2 (port 60300)
│
├── pipelines/                        # ZenML pipeline definitions
│   ├── data_fetching.py
│   ├── make_predictions.py
│   └── live_evaluation.py
│
├── steps/                            # ZenML step definitions
│   ├── fetch_data.py
│   ├── clean_data.py
│   ├── upload_data.py
│   ├── ingest_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict_model.py
│   ├── data_preparation.py
│   └── evaluate_live.py
│
├── src/                              # Core business logic
│   ├── config.py
│   ├── data_extraction.py
│   ├── data_cleaning.py
│   ├── data_upload.py
│   ├── data_ingest.py
│   ├── model_train.py
│   ├── model_evaluation.py
│   ├── model_predict.py
│   ├── live_evaluation.py
│   └── .env                          # Credentials (not committed)
│
└── data/                             # Local CSV artefacts
```

---

## Pipelines

### 1. Data Fetching (`run_fetch_data_pipeline.py`)
```
fetch_data → clean_data → upload_data
```
Scrapes historical match CSVs and upcoming fixtures, engineers rolling features, and uploads to PostgreSQL.

### 2. Make Predictions (`run_make_predictions.py`)
```
ingest_data → train_model → evaluate_model → predict_model → upload_data
```
Trains the classifier, logs metrics to MLflow, sets the `@champion` alias on the best model, predicts upcoming fixtures, and projects the final league table.

### 3. Live Evaluation (`run_live_evaluation_pipeline.py`)
```
prepare_data → evaluate_model
```
Loads previously predicted vs actual results for played matches and computes accuracy, precision, recall, and F1. Used by the automated cron job to decide whether retraining is needed.

---

## Tech Stack

- **Python 3.11**
- **ZenML 0.93.3** – ML pipeline orchestration & dashboard (port 8080)
- **MLflow 3.10.0** – Experiment tracking, model registry & UI (port 60300)
- **scikit-learn / XGBoost** – Model training & evaluation
- **pandas / NumPy** – Data manipulation
- **psycopg2** – PostgreSQL connectivity
- **Pydantic** – Configuration validation
- **Streamlit** – Web UI
- **Supabase (PostgreSQL)** – Cloud database for processed data & MLflow backend
- **AWS ECR** – Docker image registry
- **AWS EC2** – Cloud deployment
- **AWS S3** – MLflow artifact storage
- **Docker** – Containerisation

---

## Local Development Setup

### 1. Clone and create virtual environment
```bash
git clone <repo-url>
cd pl_predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials
Create `src/.env`:
```
DB_USER=...
DB_PASSWORD=...
DB_HOST=...
DB_PORT=...
DB_NAME=...
MLFLOW_TRACKING_URI=postgresql://<user>:<password>@<host>:<port>/<dbname>
```

### 3. Initialise ZenML
```bash
zenml init
zenml experiment-tracker register mlflow_tracker \
  --flavor=mlflow \
  --tracking_uri="${MLFLOW_TRACKING_URI}"
zenml stack register pipeline_stack -o default -a default -e mlflow_tracker --set
```

### 4. Run pipelines
```bash
python run_fetch_data_pipeline.py
python run_make_predictions.py
python run_live_evaluation_pipeline.py
```

### 5. Launch dashboards (optional)
```bash
# ZenML dashboard
zenml up --port 8080

# MLflow UI
bash scripts/start_mlflow_server.sh
```

---

## Docker Deployment

### Build the image
```bash
docker build -f Dockerfile.pipelines -t pl_predictor_pipelines .
```

### Run locally
```bash
docker run \
  --env-file src/.env \
  -p 8080:8080 \
  pl_predictor_pipelines
```

The container:
1. Registers the ZenML stack (reads `MLFLOW_TRACKING_URI` from env)
2. Starts the ZenML dashboard server on port 8080
3. Runs all three pipelines sequentially via `pipelineswrapper.sh`

Override CMD to run a single pipeline:
```bash
docker run --env-file src/.env pl_predictor_pipelines \
  python run_live_evaluation_pipeline.py
```

---

## AWS Deployment

### Push image to ECR
```bash
# Authenticate
aws ecr get-login-password --region eu-north-1 \
  | docker login --username AWS --password-stdin \
    <account-id>.dkr.ecr.eu-north-1.amazonaws.com

# Tag and push
docker tag pl_predictor_pipelines:latest \
  <account-id>.dkr.ecr.eu-north-1.amazonaws.com/pl_predictor_pipelines:latest

docker push <account-id>.dkr.ecr.eu-north-1.amazonaws.com/pl_predictor_pipelines:latest
```

### Run on EC2
```bash
# Authenticate Docker to ECR on the instance
aws ecr get-login-password --region eu-north-1 \
  | sudo docker login --username AWS --password-stdin \
    <account-id>.dkr.ecr.eu-north-1.amazonaws.com

# Pull and run
sudo docker pull <account-id>.dkr.ecr.eu-north-1.amazonaws.com/pl_predictor_pipelines:latest

sudo docker run \
  --env-file ~/src/.env \
  -p 8080:8080 \
  <account-id>.dkr.ecr.eu-north-1.amazonaws.com/pl_predictor_pipelines:latest
```

EC2 Security Group must have inbound TCP rules open for ports **8080** (ZenML) and **60300** (MLflow).

---

## Automated Weekly Evaluation (Cron Job)

A Python cron script (`~/cron_job.py` on EC2) runs every Tuesday at 1 AM. It:
1. Re-authenticates to ECR
2. Pulls the latest image
3. Runs the live evaluation pipeline and parses the accuracy
4. If accuracy < 0.5, automatically triggers a full retraining pipeline run

```bash
# Add to crontab (on EC2)
export TERM=xterm-256color
crontab -e
# Add:
0 1 * * 2 python3 /home/ec2-user/cron_job.py >> /home/ec2-user/pipeline.log 2>&1
```

---

## MLflow Server on EC2

The MLflow tracking server runs on EC2 with PostgreSQL as the backend store and S3 for artifact storage.

```bash
# One-time setup
python3.11 -m venv ~/mlflow-env
source ~/mlflow-env/bin/activate
pip install mlflow==3.10.0 psycopg2-binary boto3

# Start server
bash ~/start_mlflow_server.sh
```

Dashboard available at: `http://<ec2-public-ip>:60300`

**Required AWS permissions for EC2 instance role:**
- `AmazonEC2ContainerRegistryReadOnly` – pull images from ECR
- S3 read/write on the artifacts bucket – for MLflow artifact storage

---

## Model Registry

Models are registered in MLflow under the name `pl-predictor`. After each training run:
- A new version is registered and tagged with its test accuracy
- The version with the highest accuracy across all runs receives the `@champion` alias
- `predict_model` always loads `models:/pl-predictor@champion`
