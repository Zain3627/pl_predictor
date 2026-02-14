from zenml import pipeline, step
from src.data_ingest import DataIngest
import pandas as pd

from zenml.logger import get_logger
logger = get_logger(__name__)

@step
def ingest_data() -> None:
    """
    Step that ingests the clean and ready data from database

    Args: None

    Returns:
    X_train:pd.DataFrame training features for matches from 2023 to 2026
    X_test:pd.DataFrame testing features for matches from 2023 to 2026
    Y_train:pd.Series target variable for matches from 2023 to 2026
    Y_test:pd.Series target variable for matches from 2023 to 2026
    fixtures:pd.DataFrame cleaned data for upcoming fixtures for the 2026 season        
    """
    try:
        ingestor = DataIngest()
        ingestor.ingest()
        logger.info("Completed ingest step")
        return None
    except Exception as e:
        logger.error(e)
        raise e
    