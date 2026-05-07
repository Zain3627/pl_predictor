#!/bin/bash
set -e
zenml experiment-tracker register mlflow_tracker \
--flavor=mlflow \
--tracking_uri="${MLFLOW_TRACKING_URI}"
zenml stack register pipeline_stack -o default -a default -e mlflow_tracker --set
zenml up --port 8080 --ip-address 0.0.0.0
echo "ZenML stack ready"