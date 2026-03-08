from typing import Annotated, Tuple

import pandas as pd
from zenml.logger import get_logger
logger = get_logger(__name__)
from zenml import step

from src.model_predict import ModelPredict

from sklearn.base import ClassifierMixin
from src.model_predict import ModelPredict

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def predict_model(trained_model: ClassifierMixin, fixtures: pd.DataFrame, team_ids_df: pd.DataFrame, league_table: pd.DataFrame) -> Tuple[
        Annotated [pd.DataFrame, "Predicted league table"],
        Annotated [pd.DataFrame, "Predicted fixtures with team IDs and predictions"]
    ]:
    """
    Method to predict upcoming fixtures using a trained model
        
    Args:
    trained_model: ClassifierMixin trained model object
    fixtures: pd.DataFrame upcoming fixtures to predict
    team_ids_df: pd.DataFrame team IDs for upcoming fixtures
    league_table: pd.DataFrame league table for the current season

    Returns: 
    pd.DataFrame: Predicted league table
    pd.DataFrame: Predicted fixtures with team IDs and predictions

    """
    try:
        predictor = ModelPredict()
        league_table, predicted_with_team_ids = predictor.predict_matches(trained_model, fixtures, team_ids_df, league_table)
        logger.info('Completed model prediction step')
        return league_table, predicted_with_team_ids
    except Exception as e:
        raise e