from zenml.logger import get_logger
from zenml import pipeline, step
from src.table_upload import TableUpload
import pandas as pd

logger = get_logger(__name__)

@step
def upload_data(predicted_fixtures: pd.DataFrame, predicted_with_team_ids:pd.DataFrame) -> None:
    """
    Step to upload predicted league table to database

    Args: 
    predicted_fixtures:pd.DataFrame predicted final league standings (team, total_points)
    predicted_with_team_ids: pd.DataFrame predicted fixtures with team IDs and predictions (HomeTeam, AwayTeam, predictions)

    Returns: None
    """
    try:
        uploader = TableUpload()
        uploader.table_upload(predicted_fixtures, predicted_with_team_ids)
        logger.info("Completed upload step")
    except Exception as e:
        logger.error(e)
        raise e