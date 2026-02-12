import pandas as pd
import numpy as np

from zenml import pipeline, step
from zenml.logger import get_logger
from src.data_cleaning import DataCleaning
logger = get_logger(__name__)

@step
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data loaded from API

    Args: pd.DataFrame: Dataframe containing the data from the API for all previous matches from 2023 to 2026
    
    Returns: pd.DataFrame: Dataframe containing clean data for all previous matches from 2023 to 2026
    """
    try:
        cleaner = DataCleaning()
        cleaner.clean_data(data)
        logger.info("Completed cleaning step")
        return data
    except Exception as e:
        logger.error(e)
        raise e