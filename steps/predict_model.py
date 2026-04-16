from typing import Annotated, Tuple

import pandas as pd
from zenml.logger import get_logger
logger = get_logger(__name__)
from zenml import step

from src.model_predict import ModelPredict
from sklearn.base import ClassifierMixin

import mlflow
from zenml.client import Client
from dotenv import load_dotenv
import os

load_dotenv('src/.env')
experiment_tracker = Client().active_stack.experiment_tracker

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

@step(experiment_tracker=experiment_tracker.name)
def predict_model(trained_model: ClassifierMixin, fixtures: pd.DataFrame, team_ids_df: pd.DataFrame, league_table: pd.DataFrame) -> Tuple[
        Annotated[pd.DataFrame, "Predicted league table"],
        Annotated[pd.DataFrame, "Predicted fixtures with team IDs and predictions"]
    ]:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            model = mlflow.pyfunc.load_model("models:/pl-predictor@champion")
            logger.info('Loaded champion model from MLflow registry')
        except Exception:
            logger.warning('No champion model found in registry — using current run model as fallback')
            model = trained_model

        predictor = ModelPredict()
        league_table, predicted_with_team_ids = predictor.predict_matches(model, fixtures, team_ids_df, league_table)
        logger.info('Completed model prediction step')
        return league_table, predicted_with_team_ids
    except Exception as e:
        raise e
