from zenml.logger import get_logger
from zenml import pipeline, step
from src.table_upload import TableUpload
import pandas as pd

logger = get_logger(__name__)

@step
def upload_data(predicted_fixtures: pd.DataFrame) -> None:
    """
    Step to upload predicted league table to database

    Args: 
    predicted_fixtures:pd.DataFrame predicted final league standings (team, total_points)
    
    Returns: None
    """
    try:
        uploader = TableUpload()
        uploader.table_upload(predicted_fixtures)
        logger.info("Completed upload step")
    except Exception as e:
        logger.error(e)
        raise e