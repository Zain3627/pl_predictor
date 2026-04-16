import pandas as pd
from zenml.logger import get_logger
from zenml import step
logger = get_logger(__name__)

from src.model_train import ModelTrain
from sklearn.base import ClassifierMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client
from dotenv import load_dotenv
import os

load_dotenv('src/.env')
experiment_tracker = Client().active_stack.experiment_tracker

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame, Y_train: pd.Series) -> Tuple[
    Annotated[ClassifierMixin, 'model'],
    Annotated[int, 'model_version']
]:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.autolog(log_models=True)
        trainer = ModelTrain()
        trained_model = trainer.train(X_train, Y_train)
        mlflow.log_param('model_name', trained_model.__class__.__name__)
        mlflow.log_param('training_samples', len(X_train))
        mlflow.log_param('n_features', X_train.shape[1])
        for param, value in trained_model.get_params().items():
            mlflow.log_param(param, value)

        run_id = mlflow.active_run().info.run_id
        registered = mlflow.register_model(f"runs:/{run_id}/model", "pl-predictor")
        version = int(registered.version)
        logger.info(f'Model registered as pl-predictor version {version}')
        return trained_model, version
    except Exception as e:
        raise e
