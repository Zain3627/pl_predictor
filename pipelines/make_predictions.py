from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.train_model import train_model
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
        X_train, X_test, Y_train, Y_test, fixtures = ingest_data()
        train_model(X_train, Y_train)
    except Exception as e:
        logger.error(e)
        raise e