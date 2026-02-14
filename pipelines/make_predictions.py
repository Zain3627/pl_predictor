from zenml import pipeline
from steps.ingest_data import ingest_data

from zenml.logger import get_logger

logger = get_logger(__name__)

@pipeline(enable_cache=False)
def make_predictions():
    """
    Pipeline to load clean data from database, make predictions and upload to database

    Args: None
    
    Returns: None    
    """
    try:
        ingest_data()
    except Exception as e:
        logger.error(e)
        raise e