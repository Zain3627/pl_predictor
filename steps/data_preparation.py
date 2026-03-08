import pandas as pd
import numpy as np
from zenml import step
from zenml.logger import get_logger
logger = get_logger(__name__)
from typing import Tuple

from src.data_preparation import DataPreparation

@step
def prepare_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Step for preparing data for current results live evaluation.

    Args:
    None

    Returns:
    final_result:np.ndarray - Array containing the actual results for the fixtures that have been played
    current_predictions:np.ndarray - Array containing the predicted results for the fixtures that have been played
    """
    try:
        data_preparer = DataPreparation()
        final_result, current_predictions = data_preparer.prepare_data()
        logger.info("Completed data preparation step")
        return final_result, current_predictions
    except Exception as e:
        logger.error(e)
        raise e