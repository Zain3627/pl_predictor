from pipelines.live_evaluation import live_evaluation
from zenml.logger import get_logger
logger = get_logger(__name__)

if __name__ == "__main__":
    try:
        live_evaluation()
        logger.info("Live evaluation pipeline executed successfully")        
    except Exception as e:
        logger.error(e)
