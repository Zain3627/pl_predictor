from src.data_extraction import DataExtraction

from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def fetch_data():
    """
    Method to load data from API endpoint and return a dataframe 

    Args: None
    
    Returns: pd.DataFrame: Dataframe containing the data from the API for all previous matches from 2023 to 2026
    """
    try:
        ingest = DataExtraction()
        previous_matches = ingest.load_api()
        logger.info("Loaded data from API")
        return previous_matches
    except Exception as e:
        logger.error(e)
        raise e
