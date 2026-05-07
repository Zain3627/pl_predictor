# Plan: Cloud Hosting, Scheduling, Monitoring & CI/CD for PL Predictor Pipelines

## Context

The project already has a Streamlit app running on EC2 via Docker/ECR and Supabase as the data store. The three ML pipelines (`fetch_data`, `make_predictions`, `live_evaluation`) currently run only on the local dev machine because of:
- Hardcoded absolute paths (`/mnt/localdisk/...`) in three `src/` files
- MLflow backed by a local SQLite file (`mlflow.db`) that can't persist across containers
- ZenML stack only initialized on the local machine

The goal is to containerize the pipelines, run them from EC2 on a schedule, expose MLflow metrics as a monitoring dashboard, and wire up CI/CD via GitHub Actions.

---

## Phase 2: MLflow Backend → Supabase PostgreSQL

**Why:** The current SQLite `mlflow.db` is a local file. Containers can't share it. Moving to Supabase PostgreSQL makes MLflow accessible from any container and persists across EC2 restarts.

### 2.1 Create S3 bucket for MLflow artifacts

Create `pl-predictor-mlflow-artifacts` in the AWS console (same region as EC2). MLflow stores trained model binaries here (currently written to local `mlruns/`). The EC2 instance profile needs `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on this bucket.

### 2.2 Update MLFLOW_TRACKING_URI

In `src/.env` change:
```
# Before
MLFLOW_TRACKING_URI=sqlite:////mnt/localdisk/.../mlflow.db

# After
MLFLOW_TRACKING_URI=postgresql+psycopg2://DB_USER:DB_PASSWORD@DB_HOST:5432/DB_NAME?sslmode=require
```

MLflow will auto-create its own tables (`experiments`, `runs`, `metrics`, etc.) in Supabase the first time it connects — no manual DDL needed.

### 2.3 Update start_mlflow_server.sh

```bash
mlflow ui \
  --backend-store-uri "postgresql+psycopg2://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/${DB_NAME}?sslmode=require" \
  --default-artifact-root "s3://pl-predictor-mlflow-artifacts/mlflow" \
  --host 0.0.0.0 \
  --port 5000
```

**Important:** Pipelines connect directly to PostgreSQL via the `MLFLOW_TRACKING_URI` env var — the MLflow server process does NOT need to be running for pipelines to execute. It's only needed for the UI.

### 2.4 Verify locally before containerizing

Run `run_make_predictions.py` locally with the new PostgreSQL URI. Confirm runs appear when you launch `mlflow ui`. Then run `run_live_evaluation_pipeline.py` and confirm live eval metrics appear.

---

## Phase 3: Containerize Pipelines

### 3.1 ZenML inside the container

ZenML requires a stack to be initialized before any pipeline step runs. The stack must reference the MLflow experiment tracker. A setup script handles this at container startup:

**New file: `setup_zenml.sh`**
```bash
#!/bin/bash
set -e
zenml init
zenml experiment-tracker register mlflow_tracker \
  --flavor=mlflow \
  --tracking_uri="${MLFLOW_TRACKING_URI}"
zenml stack register pipeline_stack -o default -a default -e mlflow_tracker --set
echo "ZenML stack ready"
```

### 3.2 New file: `Dockerfile.pipelines`

Single image, three pipelines — differentiated at runtime by overriding CMD:

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY steps/ steps/
COPY pipelines/ pipelines/
COPY run_fetch_data_pipeline.py run_make_predictions.py run_live_evaluation_pipeline.py setup_zenml.sh ./
RUN chmod +x setup_zenml.sh

ENTRYPOINT ["bash", "-c", "./setup_zenml.sh && exec python \"$@\"", "--"]
CMD ["run_fetch_data_pipeline.py"]
```

### 3.3 Run commands (same image, different CMD)

```bash
docker run --rm --env-file pipeline.env <ECR>/pl-predictor-pipelines:latest run_fetch_data_pipeline.py
docker run --rm --env-file pipeline.env <ECR>/pl-predictor-pipelines:latest run_make_predictions.py
docker run --rm --env-file pipeline.env <ECR>/pl-predictor-pipelines:latest run_live_evaluation_pipeline.py
```

`pipeline.env` contains `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `MLFLOW_TRACKING_URI`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `DATA_DIR` (optional, defaults to `/tmp`).

---

## Phase 4: Scheduling on EC2 (cron)

**Recommendation: EC2 cron over AWS EventBridge + ECS.** The EC2 instance already exists. Cron adds zero cost and zero new AWS resource types. ECS adds 4-6 new resource types (cluster, task definitions, networking, IAM roles) for no practical benefit at this scale.

### 4.1 New file: `run_pipeline.sh` (deploy to EC2)

```bash
#!/bin/bash
set -e
source /etc/pl-predictor.env   # secrets file, chmod 600, owned by ec2-user
PIPELINE=$1
ECR="<account>.dkr.ecr.<region>.amazonaws.com/pl-predictor-pipelines"

aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin "$ECR"
docker pull "${ECR}:latest"

case "$PIPELINE" in
  fetch)    SCRIPT="run_fetch_data_pipeline.py" ;;
  predict)  SCRIPT="run_make_predictions.py" ;;
  evaluate) SCRIPT="run_live_evaluation_pipeline.py" ;;
  *) echo "Unknown pipeline: $PIPELINE"; exit 1 ;;
esac

docker run --rm \
  -e DB_USER -e DB_PASSWORD -e DB_HOST -e DB_PORT -e DB_NAME \
  -e MLFLOW_TRACKING_URI -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION \
  "${ECR}:latest" "$SCRIPT"

docker image prune -f   # clean up old layers
```

### 4.2 Cron schedule (EC2 crontab, times UTC)

```cron
# Fetch data every Friday 08:00 — ahead of weekend fixtures
0 8 * * 5  /home/ec2-user/pl-pipelines/run_pipeline.sh fetch >> /var/log/pl-fetch.log 2>&1

# Make predictions every Friday 09:00 — after fresh data is in Supabase
0 9 * * 5  /home/ec2-user/pl-pipelines/run_pipeline.sh predict >> /var/log/pl-predict.log 2>&1

# Live evaluation every Monday 08:00 — after weekend results are in
0 8 * * 1  /home/ec2-user/pl-pipelines/run_pipeline.sh evaluate >> /var/log/pl-evaluate.log 2>&1
```

---

## Phase 5: Monitoring

**Primary tool: MLflow UI** — already fully integrated. Every `run_live_evaluation_pipeline.py` run logs `accuracy`, `precision`, `recall`, `f1_score` as MLflow metrics. MLflow's UI shows these as a time-series chart across runs automatically — this covers 100% of the model monitoring need without extra tooling.

### 5.1 Access the MLflow UI from EC2

Do NOT open port 5000 publicly. Use an SSH tunnel:

```bash
# On your local machine:
ssh -L 5000:localhost:5000 ec2-user@<ec2-ip>

# Then on EC2 (over SSH):
source /etc/pl-predictor.env
mlflow ui \
  --backend-store-uri "${MLFLOW_TRACKING_URI}" \
  --default-artifact-root "s3://pl-predictor-mlflow-artifacts/mlflow" \
  --host 127.0.0.1 --port 5000
```

Open `http://localhost:5000` in browser. Navigate to the `live_evaluation` experiment to see accuracy over time. The model registry shows all trained versions and which one holds the `@champion` alias.

### 5.2 Optional: always-on access with nginx basic auth

If you want the dashboard accessible without SSH:
- Install nginx on EC2
- Add HTTP Basic Auth to a proxy pointing at `localhost:5000`
- Run `mlflow ui` as a systemd service
- Open port 443 (HTTPS) in the EC2 security group with an IP allowlist

Skip this initially — SSH tunnel is sufficient for a solo project.

### 5.3 CloudWatch for infrastructure health

Add one CloudWatch alarm: `CPUUtilization > 80%` for 10 minutes → SNS email alert. The EC2 instance already emits this metric. No code changes needed.

---

## Phase 6: CI/CD with GitHub Actions

### 6.1 New file: `.github/workflows/ci.yml`

Runs on every push and PR. Verifies all imports compile (catches broken code without needing DB access):

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  check-imports:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: |
          python -c "from src.data_cleaning import DataCleaning"
          python -c "from src.data_extraction import DataExtraction"
          python -c "from src.model_train import ModelTrain"
          python -c "from src.model_predict import ModelPredict"
          python -c "from src.model_evaluation import Accuracy"
          python -c "from src.live_evaluation import Accuracy"
```

### 6.2 New file: `.github/workflows/deploy.yml`

Runs on push to `main` only. Builds pipeline image → pushes to ECR → pulls on EC2 → smoke test:

```yaml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  build-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}
      - id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      - name: Build and push
        env:
          REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          TAG: ${{ github.sha }}
        run: |
          docker build -f Dockerfile.pipelines \
            -t $REGISTRY/pl-predictor-pipelines:$TAG \
            -t $REGISTRY/pl-predictor-pipelines:latest .
          docker push $REGISTRY/pl-predictor-pipelines:$TAG
          docker push $REGISTRY/pl-predictor-pipelines:latest

  deploy:
    needs: build-push
    runs-on: ubuntu-latest
    steps:
      - name: Pull image on EC2 + smoke test
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ec2-user
          key: ${{ secrets.EC2_SSH_PRIVATE_KEY }}
          script: |
            set -e
            source /etc/pl-predictor.env
            ECR="${{ secrets.ECR_REGISTRY }}/pl-predictor-pipelines"
            aws ecr get-login-password --region $AWS_DEFAULT_REGION \
              | docker login --username AWS --password-stdin "$ECR"
            docker pull "${ECR}:latest"
            # Smoke test: verify container starts and env vars present
            docker run --rm \
              -e DB_HOST -e MLFLOW_TRACKING_URI \
              "${ECR}:latest" \
              run_fetch_data_pipeline.py \
              python -c "
            import os
            assert os.getenv('DB_HOST'), 'DB_HOST missing'
            assert os.getenv('MLFLOW_TRACKING_URI'), 'MLFLOW_TRACKING_URI missing'
            from src.data_cleaning import DataCleaning
            print('Smoke test passed')
            "
```

### 6.3 GitHub Actions secrets required

Add these in repo Settings → Secrets → Actions:
```
AWS_ACCESS_KEY_ID          IAM user with ECR push permissions
AWS_SECRET_ACCESS_KEY
AWS_REGION                 e.g. us-east-1
ECR_REGISTRY               e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com
EC2_HOST                   Elastic IP of the EC2 instance
EC2_SSH_PRIVATE_KEY        private key for ec2-user

DB_USER / DB_PASSWORD / DB_HOST / DB_PORT / DB_NAME
MLFLOW_TRACKING_URI        the PostgreSQL URI
```

---

## Implementation Order

| Week | Tasks |
|------|-------|
| 1 | Phase 1: fix paths + rename env vars. Test all 3 pipelines locally. |
| 1 | Phase 2: migrate MLflow to Supabase PostgreSQL. Verify runs appear in `mlflow ui`. |
| 2 | Phase 3: write `Dockerfile.pipelines` + `setup_zenml.sh`. Build and test locally. Push to ECR manually. |
| 2 | Phase 4: deploy `run_pipeline.sh` to EC2. Test each pipeline manually via the script. Add crontab. |
| 3 | Phase 5: verify MLflow UI via SSH tunnel after first cron run. |
| 3 | Phase 6: create GitHub Actions workflows. Add secrets. Push to main and verify full CI/CD chain. |

## Verification

1. **Pipelines work in container:** `docker run ... run_live_evaluation_pipeline.py` exits 0, new run appears in MLflow
2. **Scheduling works:** Set a cron entry 2 minutes in the future, wait, confirm log entry and MLflow run
3. **Monitoring works:** Open `http://localhost:5000` via SSH tunnel, see accuracy trend across runs in `live_evaluation` experiment
4. **CI/CD works:** Push a trivial commit to `main`, watch GitHub Actions build → deploy → smoke test all pass

## New Files to Create

- `Dockerfile.pipelines`
- `setup_zenml.sh`
- `run_pipeline.sh` (deployed to EC2, not committed with secrets)
- `.github/workflows/ci.yml`
- `.github/workflows/deploy.yml`

## Files to Modify

- `src/.env` — rename 5 keys + new PostgreSQL `MLFLOW_TRACKING_URI`
- `start_mlflow_server.sh` — PostgreSQL backend + S3 artifact root
