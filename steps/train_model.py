import pandas as pd
from zenml.logger import get_logger
from zenml import step
logger = get_logger(__name__)

from src.model_train import ModelTrain
from sklearn.base import ClassifierMixin
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame, Y_train: pd.Series) -> ClassifierMixin:
    """
    Method to train model that is choosen in the config file
        
    Args:
    X_train:pd.DataFrame training features for matches from 2023 to 2026
    Y_train:pd.Series target variable for matches from 2023 to 2026

    Returns: 
    trained_model: ClassifierMixin trained model object
    """
    try:
        trainer = ModelTrain()
        trained_model = trainer.train(X_train, Y_train)
        mlflow.sklearn.autolog()
        logger.info('Completed model training step')
        return trained_model
    except Exception as e:
        raise e