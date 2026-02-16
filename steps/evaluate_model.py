import pandas as pd
from zenml import step
from zenml.logger import get_logger
logger = get_logger(__name__)

from src.model_evaluation import MSE, RMSE, R2Score, Accuracy
from sklearn.base import RegressorMixin
from typing_extensions import Annotated, Tuple

@step
def evaluate_model(
        model: RegressorMixin,
        X_test: pd.DataFrame,
        Y_test: pd.Series
) -> Tuple [
    Annotated[float, 'Accuracy'],
    Annotated[float, 'RMSE'], 
    Annotated[float, 'R2Score']
    ]:
    """
    Step to evaluate the model performance on the test set using RMSE and R2 Score
    
    Args:
    model: RegressorMixin trained model object
    X_test:pd.DataFrame testing features for matches from 2023 to 2026
    Y_test:pd.Series target variable for matches from 2023 to 2026

    Returns:
    rmse: float Root Mean Squared Error value
    r2_score: float R2 Score value
    """
    try:
        logger.info('Predicting on test set')
        predicted_Y_test = model.predict(X_test)

        accuracy_evaluator = Accuracy()
        rmse_evaluator = RMSE()
        r2_score_evaluator = R2Score()

        accuracy = accuracy_evaluator.evaluate(predicted_Y_test, Y_test)
        rmse = rmse_evaluator.evaluate(predicted_Y_test, Y_test)
        r2_score = r2_score_evaluator.evaluate(predicted_Y_test, Y_test)

        logger.info(f'Model evaluation completed with Accuracy: {accuracy}, RMSE: {rmse}, R2 Score: {r2_score}')
        return accuracy,rmse, r2_score
    
    except Exception as e:
        logger.error(e)
        raise e