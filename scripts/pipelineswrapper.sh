#!/bin/bash
set -euo pipefail

python run_fetch_data_pipeline.py
python run_make_predictions.py
python run_live_evaluation_pipeline.py