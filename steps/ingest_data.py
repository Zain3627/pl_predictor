from src.data_extraction import DataExtraction

from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def ingest_data():
    """
    Docstring for ingest_data
    """
    try:
        ingest = DataExtraction()
        previous_matches = ingest.load_api()
        logger.info("Loaded data from API")
        return previous_matches
    except Exception as e:
        logger.error(e)
        raise e
