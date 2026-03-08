import pandas as pd
import numpy as np
from zenml import step
from zenml.logger import get_logger
logger = get_logger(__name__)

from src.live_evaluation import Accuracy, Precision, Recall, F1Score
from typing_extensions import Annotated, Tuple

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
        final_result: np.ndarray,
        current_predictions: np.ndarray
) -> Tuple [
    Annotated[float, 'Accuracy'],
    Annotated[float, 'Precision'], 
    Annotated[float, 'Recall'],
    Annotated[float, 'F1Score']
    ]:
    """
    Step to evaluate the model performance on the test set using classification metrics
    
    Args:
    final_result: np.ndarray actual target values for matches from 2023 to 2026
    current_predictions: np.ndarray predicted target values for matches from 2023 to 2026

    Returns:
    accuracy: float Accuracy value
    precision: float Precision value
    recall: float Recall value
    f1_score: float F1 Score value
    """
    try:
        if len(final_result) == 0 or len(current_predictions) == 0:
            logger.info('No matches have been played yet for the current season. Skipping evaluation.')
            return 0.0, 0.0, 0.0, 0.0
        logger.info('Evaluating on new played matches set')

        accuracy_evaluator = Accuracy()
        precision_evaluator = Precision()
        recall_evaluator = Recall()
        f1_score_evaluator = F1Score()

        accuracy = accuracy_evaluator.evaluate(final_result, current_predictions)
        precision = precision_evaluator.evaluate(final_result, current_predictions)
        recall = recall_evaluator.evaluate(final_result, current_predictions)
        f1_score = f1_score_evaluator.evaluate(final_result, current_predictions)

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1_score)

        logger.info(f'Model evaluation completed with Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')
        return accuracy, precision, recall, f1_score

    except Exception as e:
        logger.error(e)
        raise e