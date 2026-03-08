from zenml import pipeline, step
from zenml.logger import get_logger
from steps.data_preparation import prepare_data
from steps.evaluate_live import evaluate_model
logger = get_logger(__name__)

@pipeline(enable_cache=False)
def live_evaluation():
    """
    Pipeline to load the current predictions and evaluate against actual results for the fixtures that have been played

    Args: None
    
    Returns: None    
    """
    try:
        final_result, current_predictions = prepare_data()
        accuracy, precision, recall, f1_score = evaluate_model(final_result, current_predictions)
    except Exception as e:
        logger.error(e)
        raise e