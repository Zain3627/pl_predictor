import pandas as pd
import numpy as np
from zenml import step
from zenml.logger import get_logger
logger = get_logger(__name__)

from src.model_evaluation import Accuracy, Precision, Recall, F1Score
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated, Tuple

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
        model: ClassifierMixin,
        X_test: pd.DataFrame,
        Y_test: pd.Series
) -> Tuple [
    Annotated[float, 'Accuracy'],
    Annotated[float, 'Precision'], 
    Annotated[float, 'F1Score']
    ]:
    """
    Step to evaluate the model performance on the test set using classification metrics
    
    Args:
    model: ClassifierMixin trained model object
    X_test:pd.DataFrame testing features for matches from 2023 to 2026
    Y_test:pd.Series target variable for matches from 2023 to 2026

    Returns:
    accuracy: float Accuracy value
    precision: float Precision value
    f1_score: float F1 Score value
    """
    try:
        logger.info('Predicting on test set')
        predicted_Y_test = np.array(model.predict(X_test))
        true_Y_test = np.array(Y_test)

        accuracy_evaluator = Accuracy()
        precision_evaluator = Precision()
        f1_score_evaluator = F1Score()

        accuracy = accuracy_evaluator.evaluate(predicted_Y_test, true_Y_test)
        precision = precision_evaluator.evaluate(predicted_Y_test, true_Y_test)
        f1_score = f1_score_evaluator.evaluate(predicted_Y_test, true_Y_test)

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('f1_score', f1_score)

        logger.info(f'Model evaluation completed with Accuracy: {accuracy}, Precision: {precision}, F1 Score: {f1_score}')
        return accuracy, precision, f1_score
    
    except Exception as e:
        logger.error(e)
        raise e