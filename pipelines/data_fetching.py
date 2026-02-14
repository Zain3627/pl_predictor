from zenml import pipeline, step
from zenml.logger import get_logger
from steps.fetch_data import fetch_data
from steps.clean_data import clean_data
from steps.upload_data import upload_data

logger = get_logger(__name__)

@pipeline(enable_cache=False)
def data_fetching():
    """
    Pipeline to load data from API endpoint, clean data and upload to database

    Args: None
    
    Returns: None    
    """
    try:
        previous_matches, fixtures_df = fetch_data()
        X_train, X_test, Y_train, Y_test, fixtures = clean_data(previous_matches, fixtures_df)
        upload_data(X_train, X_test, Y_train, Y_test, fixtures)
    except Exception as e:
        logger.error(e)
        raise e