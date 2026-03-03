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
def predict_model(trained_model: ClassifierMixin, fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Method to predict upcoming fixtures using a trained model
        
    Args:
    trained_model: ClassifierMixin trained model object
    fixtures: pd.DataFrame upcoming fixtures to predict

    Returns: 
    pd.DataFrame: Predicted fixtures with points
    """
    try:
        predictor = ModelPredict()
        predicted_fixtures = predictor.predict_matches(trained_model, fixtures)
        logger.info('Completed model prediction step')
        return predicted_fixtures
    except Exception as e:
        raise e