import pandas as pd
import numpy as np

from zenml.logger import get_logger
logger = get_logger(__name__)

from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelEvaluation(ABC):

    @abstractmethod
    def evaluate(self, final_result: np.ndarray, current_predictions: np.ndarray) -> float:
        """
        Abstract method for the strategy pattern to evaluate the model performance on the test set.
        
        Args:
        final_result:np.ndarray actual target values for matches from 2023 to 2026
        current_predictions:np.ndarray predicted target values for matches from 2023 to 2026

        Returns:
        evaluation_metric: float evaluation metric value
        """
        pass

class Accuracy(ModelEvaluation):
    """
    Class to evaluate the model performance using Accuracy
    """
    def evaluate(self, final_result: np.ndarray, current_predictions: np.ndarray) -> float:
        try:
            logger.info('Evaluating model using Accuracy')
            accuracy = accuracy_score(final_result, current_predictions)
            logger.info(f'Accuracy: {accuracy}')

            # pd.DataFrame({
            #     'predicted': current_predictions,
            #     'true': final_result
            # }).to_csv('predicted_vs_true.csv', index=False)
            return accuracy
        
        except Exception as e:
            logger.error(e)
            raise e

class Precision(ModelEvaluation):
    """
    Class to evaluate the model performance using Precision (weighted)
    """
    def evaluate(self, final_result: np.ndarray, current_predictions: np.ndarray) -> float:
        try:
            logger.info('Evaluating model using Precision')
            precision = precision_score(final_result, current_predictions, average='weighted', zero_division=0)
            logger.info(f'Precision: {precision}')
            return precision
        
        except Exception as e:
            logger.error(e)
            raise e

class Recall(ModelEvaluation):
    """
    Class to evaluate the model performance using Recall (weighted)
    """
    def evaluate(self, final_result: np.ndarray, current_predictions: np.ndarray) -> float:
        try:
            logger.info('Evaluating model using Recall')
            recall = recall_score(final_result, current_predictions, average='weighted', zero_division=0)
            logger.info(f'Recall: {recall}')
            return recall
        
        except Exception as e:
            logger.error(e)
            raise e

class F1Score(ModelEvaluation):
    """
    Class to evaluate the model performance using F1 Score (weighted)
    """
    def evaluate(self, final_result: np.ndarray, current_predictions: np.ndarray) -> float:
        try:
            logger.info('Evaluating model using F1 Score')
            f1 = f1_score(final_result, current_predictions, average='weighted', zero_division=0)
            logger.info(f'F1 Score: {f1}')
            return f1
        
        except Exception as e:
            logger.error(e)
            raise e