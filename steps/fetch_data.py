import pandas as pd
from src.data_extraction import DataExtraction

from zenml import pipeline, step
from zenml.logger import get_logger
from typing import Tuple

logger = get_logger(__name__)

@step
def fetch_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Method to load data from API endpoint and return a dataframe 

    Args: None
    
    Returns: 
    pd.DataFrame: Dataframe containing the data from the API for all previous matches from 2023 to 2026
    pd.DataFrame: Dataframe containing the data from the API for upcoming fixtures for the 2026 season
    """
    try:
        extract = DataExtraction()
        previous_matches, fixtures_df = extract.load_api()
        logger.info("Loaded data from API")
        return previous_matches, fixtures_df
    except Exception as e:
        logger.error(e)
        raise e
