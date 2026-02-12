from zenml import pipeline, step
from zenml.logger import get_logger
from steps.fetch_data import fetch_data
from steps.clean_data import clean_data

logger = get_logger(__name__)

@pipeline(enable_cache=False)
def data_fetching():
    """
    Pipeline to load data from API endpoint

    Args: None
    
    Returns: None    
    """
    try:
        previous_matches, fixtures_df = fetch_data()
        previous_matches = clean_data(previous_matches, fixtures_df)
    except Exception as e:
        logger.error(e)
        raise e