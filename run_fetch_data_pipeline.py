from pipelines.data_fetching import data_fetching
from zenml.logger import get_logger
logger = get_logger(__name__)

if __name__ == "__main__":
    try:
        fetch = data_fetching()
        logger.info("Data fetching pipeline executed successfully")        
    except Exception as e:
        logger.error(e)
