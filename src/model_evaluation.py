import pandas as pd
import numpy as np

from zenml.logger import get_logger
logger = get_logger(__name__)

from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelEvaluation(ABC):

    @abstractmethod
    def evaluate(self, predicted_Y_test: np.ndarray, true_Y_test: np.ndarray) -> float:
        """
        Abstract method for the strategy pattern to evaluate the model performance on the test set.
        
        Args:
        predicted_Y_test:np.ndarray predicted target values for matches from 2023 to 2026
        true_Y_test:np.ndarray target variable for matches from 2023 to 2026

        Returns:
        evaluation_metric: float evaluation metric value
        """
        pass

class Accuracy(ModelEvaluation):
    """
    Class to evaluate the model performance using Accuracy
    """
    def evaluate(self, predicted_Y_test: np.ndarray, true_Y_test: np.ndarray) -> float:
        try:
            logger.info('Evaluating model using Accuracy')
            accuracy = accuracy_score(true_Y_test, predicted_Y_test)
            logger.info(f'Accuracy: {accuracy}')

            pd.DataFrame({
                'predicted': predicted_Y_test,
                'true': true_Y_test
            }).to_csv('predicted_vs_true.csv', index=False)
            return accuracy
        
        except Exception as e:
            logger.error(e)
            raise e

class Precision(ModelEvaluation):
    """
    Class to evaluate the model performance using Precision (weighted)
    """
    def evaluate(self, predicted_Y_test: np.ndarray, true_Y_test: np.ndarray) -> float:
        try:
            logger.info('Evaluating model using Precision')
            precision = precision_score(true_Y_test, predicted_Y_test, average='weighted', zero_division=0)
            logger.info(f'Precision: {precision}')
            return precision
        
        except Exception as e:
            logger.error(e)
            raise e

class Recall(ModelEvaluation):
    """
    Class to evaluate the model performance using Recall (weighted)
    """
    def evaluate(self, predicted_Y_test: np.ndarray, true_Y_test: np.ndarray) -> float:
        try:
            logger.info('Evaluating model using Recall')
            recall = recall_score(true_Y_test, predicted_Y_test, average='weighted', zero_division=0)
            logger.info(f'Recall: {recall}')
            return recall
        
        except Exception as e:
            logger.error(e)
            raise e

class F1Score(ModelEvaluation):
    """
    Class to evaluate the model performance using F1 Score (weighted)
    """
    def evaluate(self, predicted_Y_test: np.ndarray, true_Y_test: np.ndarray) -> float:
        try:
            logger.info('Evaluating model using F1 Score')
            f1 = f1_score(true_Y_test, predicted_Y_test, average='weighted', zero_division=0)
            logger.info(f'F1 Score: {f1}')
            return f1
        
        except Exception as e:
            logger.error(e)
            raise e