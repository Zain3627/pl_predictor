import pandas as pd
import numpy as np
from zenml import step
from zenml.logger import get_logger
logger = get_logger(__name__)

from src.model_evaluation import Accuracy, Precision, Recall, F1Score
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated, Tuple

import mlflow
from mlflow import MlflowClient
from zenml.client import Client
from dotenv import load_dotenv
import os

load_dotenv('src/.env')
experiment_tracker = Client().active_stack.experiment_tracker

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
        model: ClassifierMixin,
        X_test: pd.DataFrame,
        Y_test: pd.Series,
        model_version: int
) -> Tuple[
    Annotated[float, 'Accuracy'],
    Annotated[float, 'Precision'],
    Annotated[float, 'Recall'],
    Annotated[float, 'F1Score']
]:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info('Predicting on test set')
        predicted_Y_test = np.array(model.predict(X_test))
        true_Y_test = np.array(Y_test)

        accuracy = Accuracy().evaluate(true_Y_test, predicted_Y_test)
        precision = Precision().evaluate(true_Y_test, predicted_Y_test)
        recall = Recall().evaluate(true_Y_test, predicted_Y_test)
        f1_score = F1Score().evaluate(true_Y_test, predicted_Y_test)

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1_score)

        # Tag the registered model version with its accuracy for comparison
        client = MlflowClient()
        client.set_model_version_tag("pl-predictor", str(model_version), "accuracy", str(accuracy))
        logger.info(f'Tagged pl-predictor v{model_version} with accuracy={accuracy:.4f}')

        # Find the version with the highest accuracy across all registered versions
        all_versions = client.search_model_versions("name='pl-predictor'")
        scored = [v for v in all_versions if v.tags.get("accuracy")]
        best = max(scored, key=lambda v: float(v.tags["accuracy"]))
        client.set_registered_model_alias("pl-predictor", "champion", best.version)
        logger.info(f'Champion set to v{best.version} (accuracy={best.tags["accuracy"]})')

        logger.info(f'Evaluation — Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1_score}')
        return accuracy, precision, recall, f1_score

    except Exception as e:
        logger.error(e)
        raise e
