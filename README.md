# Premier League Predictor

A machine learning project that predicts Premier League match outcomes and final league standings using historical match data. Built with [ZenML](https://zenml.io/) pipelines and tracked with [MLflow](https://mlflow.org/).

## Overview

This project fetches historical Premier League match data (2023–2026 seasons) and upcoming fixture data from external APIs, engineers features from match statistics, trains a classification model, and predicts the results of remaining fixtures to produce a projected final league table.

**Supported models** (configurable in `src/config.py`):
- Logistic Regression
- Random Forest
- XGBoost (default)

## Project Structure

```
pl_predictor/
│
├── run_fetch_data_pipeline.py      # Entry point – runs the data fetching pipeline
├── run_make_predictions.py         # Entry point – runs the predictions pipeline
│
├── pipelines/                      # ZenML pipeline definitions
│   ├── data_fetching.py            #   Fetch → Clean → Upload pipeline
│   └── make_predictions.py         #   Ingest → Train → Evaluate → Predict pipeline
│
├── steps/                          # ZenML step definitions (thin wrappers around src/)
│   ├── fetch_data.py               #   Fetches raw data from APIs
│   ├── clean_data.py               #   Cleans and feature-engineers the data
│   ├── upload_data.py              #   Uploads processed data to PostgreSQL
│   ├── ingest_data.py              #   Reads processed data from PostgreSQL
│   ├── train_model.py              #   Trains the selected classifier
│   ├── evaluate_model.py           #   Evaluates model (Accuracy, Precision, F1)
│   └── predict_model.py            #   Predicts upcoming fixtures & builds league table
│
├── src/                            # Core business logic
│   ├── config.py                   #   Model configuration (Pydantic)
│   ├── data_extraction.py          #   Loads match & fixture data from APIs
│   ├── data_cleaning.py            #   Feature engineering & train/test split
│   ├── data_upload.py              #   Uploads DataFrames to PostgreSQL (Supabase)
│   ├── data_ingest.py              #   Reads clean data from PostgreSQL (Supabase)
│   ├── model_train.py              #   Model training logic
│   ├── model_evaluation.py         #   Evaluation strategies (Strategy pattern)
│   ├── model_predict.py            #   Match prediction & league table projection
│   └── .env                        #   Database credentials (not committed)
│
├── data/                           # Local CSV artefacts
│   ├── previous_matches.csv
│   ├── upcoming_fixtures.csv
│   ├── home_snapshot.csv
│   ├── away_snapshot.csv
│   ├── averaged_previous_matches.csv
│   ├── fixtures_no_teams.csv
│   ├── fixtures_team_ids.csv
│   ├── predicted_league_table.csv
│   └── X_train.csv
│
├── .gitignore
└── LICENSE
```

## Pipelines

### 1. Data Fetching (`run_fetch_data_pipeline.py`)

```
fetch_data → clean_data → upload_data
```

| Step | Description |
|------|-------------|
| **fetch_data** | Pulls historical match CSVs from football-data.co.uk and upcoming fixtures from the Fantasy Premier League API. Also builds the current league table. |
| **clean_data** | Drops irrelevant columns, computes rolling averages (goals, shots, corners, clean sheets, points), creates interaction features, and performs a train/test split. |
| **upload_data** | Writes processed DataFrames to a PostgreSQL database (Supabase). |

### 2. Make Predictions (`run_make_predictions.py`)

```
ingest_data → train_model → evaluate_model → predict_model
```

| Step | Description |
|------|-------------|
| **ingest_data** | Reads the cleaned datasets back from PostgreSQL. |
| **train_model** | Trains the model specified in `src/config.py` (default: XGBoost). Logged with MLflow. |
| **evaluate_model** | Evaluates on the test set using Accuracy, Precision, and F1 Score. Metrics are logged to MLflow. |
| **predict_model** | Predicts outcomes for remaining fixtures and produces the projected final league table. |

## Tech Stack

- **Python 3.10**
- **ZenML** – ML pipeline orchestration
- **MLflow** – Experiment tracking
- **scikit-learn / XGBoost** – Model training & evaluation
- **pandas / NumPy** – Data manipulation
- **psycopg2** – PostgreSQL connectivity
- **Pydantic** – Configuration validation
- **Supabase (PostgreSQL)** – Cloud database for storing processed data

## Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd pl_predictor
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install zenml mlflow scikit-learn xgboost pandas numpy psycopg2-binary pydantic python-dotenv requests
   ```

4. **Configure ZenML & MLflow**
   ```bash
   zenml init
   zenml integration install mlflow -y
   zenml experiment-tracker register mlflow_tracker --flavor=mlflow
   zenml stack register ml_stack -a default -o default -e mlflow_tracker --set
   ```

5. **Set up environment variables**  
   Create `src/.env` with your Supabase/PostgreSQL credentials:
   ```
   user=<db_user>
   password=<db_password>
   host=<db_host>
   port=<db_port>
   dbname=<db_name>
   ```

## Usage

**Fetch and prepare data:**
```bash
python run_fetch_data_pipeline.py
```

**Train model and predict:**
```bash
python run_make_predictions.py
```

The predicted league table is saved to `data/predicted_league_table.csv`.
