import pandas as pd
import numpy as np
from typing_extensions import Annotated
from typing import Tuple

from zenml import pipeline, step
from zenml.logger import get_logger
from src.data_cleaning import DataCleaning
logger = get_logger(__name__)

@step
def clean_data(data: pd.DataFrame, fixtures: pd.DataFrame) -> Tuple[
        Annotated[pd.DataFrame,'X_train'], 
        Annotated[pd.DataFrame,'X_test'],
        Annotated[pd.Series,'Y_train'],
        Annotated[pd.Series,'Y_test'],
        Annotated[pd.DataFrame,'fixtures'],
        ]:
    """
    Clean raw data loaded from API

    Args: 
    data:pd.DataFrame: Dataframe containing the data from the API for all previous matches from 2023 to 2026
    fixtures:pd.DataFrame: Dataframe containing the data from the API for upcoming fixtures for the 2026 season
    
    Returns: 
    X_train:pd.DataFrame training features for matches from 2023 to 2026
    X_test:pd.DataFrame testing features for matches from 2023 to 2026
    Y_train:pd.Series target variable for matches from 2023 to 2026
    Y_test:pd.Series target variable for matches from 2023 to 2026
    fixtures:pd.DataFrame cleaned data for upcoming fixtures for the 2026 season
    """
    try:
        cleaner = DataCleaning()
        X_train, X_test, Y_train, Y_test, fixtures = cleaner.clean_data(data, fixtures)
        logger.info("Completed cleaning step")
        return X_train, X_test, Y_train, Y_test, fixtures
    except Exception as e:
        logger.error(e)
        raise e