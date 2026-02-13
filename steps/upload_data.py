from zenml.logger import get_logger
from zenml import pipeline, step
from src.data_upload import DataUpload
import pandas as pd

logger = get_logger(__name__)

@step
def upload_data(X_train:pd.DataFrame, X_test:pd.DataFrame, Y_train:pd.Series, Y_test:pd.Series, fixtures:pd.DataFrame) -> None:
    """
    Method to upload cleaned data to database

    Args: 
    X_train:pd.DataFrame training features for matches from 2023 to 2026
    X_test:pd.DataFrame testing features for matches from 2023 to 2026
    Y_train:pd.Series target variable for matches from 2023 to 2026
    Y_test:pd.Series target variable for matches from 2023 to 2026
    fixtures:pd.DataFrame cleaned data for upcoming fixtures for the 2026 season
    
    Returns: None
    """
    try:
        uploader = DataUpload()
        uploader.upload(X_train, X_test, Y_train, Y_test, fixtures)
        logger.info("Completed upload step")
    except Exception as e:
        logger.error(e)
        raise e