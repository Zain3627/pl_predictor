#!/bin/bash
source ~/mlflow-env/bin/activate

MLFLOW_ALLOWED_HOSTS="*" mlflow server \
  --backend-store-uri "postgresql://postgres.qecyiiptraezorjgihdz:12456nshassz@aws-1-eu-north-1.pooler.supabase.com:5432/postgres" \
  --default-artifact-root s3://pl-predictor-mlflow-artifacts/mlruns \
  --host 0.0.0.0 \
  --port 60300 \
  --allowed-hosts "*"
