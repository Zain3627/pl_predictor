from zenml import pipeline, step
from src.data_ingest import DataIngest
from typing_extensions import Annotated, Tuple
import pandas as pd

from zenml.logger import get_logger
logger = get_logger(__name__)

@step
def ingest_data() -> Tuple[
        Annotated[pd.DataFrame,'X_train'], 
        Annotated[pd.DataFrame,'X_test'],
        Annotated[pd.Series,'Y_train'],
        Annotated[pd.Series,'Y_test'],
        Annotated[pd.DataFrame,'fixtures'],
        ]:
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
        X_train, X_test, Y_train, Y_test, fixtures = ingestor.ingest()
        logger.info("Completed ingest step")
        return X_train, X_test, Y_train, Y_test, fixtures
    except Exception as e:
        logger.error(e)
        raise e
    