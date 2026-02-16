import pandas as pd
from zenml.logger import get_logger
logger = get_logger(__name__)

from abc import ABC, abstractmethod

from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error


class ModelEvaluation(ABC):

    @abstractmethod
    def evaluate(self, predicted_Y_test: pd.DataFrame, true_Y_test: pd.Series) -> float:
        """
        Abstract method for the strategy pattern to evaluate the model performance on the test set.
        
        Args:
        predicted_Y_test:pd.DataFrame predicted target values for matches from 2023 to 2026
        true_Y_test:pd.Series target variable for matches from 2023 to 2026

        Returns:
        evaluation_metric: float evaluation metric value
        """
        pass

class MSE(ModelEvaluation):
    """
    Class to evaluate the model performance using Mean Squared Error
    """
    def evaluate(self, predicted_Y_test: pd.DataFrame, true_Y_test: pd.Series) -> float:
        try:
            logger.info('Evaluating model using Mean Squared Error')
            mse = mean_squared_error(true_Y_test, predicted_Y_test)
            logger.info(f'Mean Squared Error: {mse}')
            return mse
        
        except Exception as e:
            logger.error(e)
            raise e

class RMSE(ModelEvaluation):
    """
    Class to evaluate the model performance using Root Mean Squared Error
    """
    def evaluate(self, predicted_Y_test: pd.DataFrame, true_Y_test: pd.Series) -> float:
        try:
            logger.info('Evaluating model using Root Mean Squared Error')
            rmse = root_mean_squared_error(true_Y_test, predicted_Y_test)
            logger.info(f'Root Mean Squared Error: {rmse}')
            return rmse
        
        except Exception as e:
            logger.error(e)
            raise e
        
class R2Score(ModelEvaluation):
    """
    Class to evaluate the model performance using R2 Score
    """
    def evaluate(self, predicted_Y_test: pd.DataFrame, true_Y_test: pd.Series) -> float:
        try:
            logger.info('Evaluating model using R2 Score')
            r2 = r2_score(true_Y_test, predicted_Y_test)
            logger.info(f'R2 Score: {r2}')
            return r2
        
        except Exception as e:
            logger.error(e)
            raise e

class Accuracy(ModelEvaluation):
    """
    Class to evaluate the model performance using Accuracy
    """
    def evaluate(self, predicted_Y_test: pd.DataFrame, true_Y_test: pd.Series) -> float:
        try:
            logger.info('Evaluating model using Accuracy')
            accuracy = (predicted_Y_test == true_Y_test).mean()
            logger.info(f'Accuracy: {accuracy}')
            return accuracy
        
        except Exception as e:
            logger.error(e)
            raise e