from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)

@pipeline(enable_cache=False)
def live_evaluation():
    """
    Pipeline to load the current predictions and evaluate against actual results for the fixtures that have been played

    Args: None
    
    Returns: None    
    """
    try:
        pass
    except Exception as e:
        logger.error(e)
        raise e