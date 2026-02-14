from pipelines.make_predictions import make_predictions

from zenml.logger import get_logger
logger = get_logger(__name__)

if __name__ == '__main__':
    try: 
        make_predictions()
        logger.info('Make predictions pipeline executed successfully')
    except Exception as e:
        logger.error(e)