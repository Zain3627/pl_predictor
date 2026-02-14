import pandas as pd
from zenml.logger import get_logger
from zenml import step
logger = get_logger(__name__)

from src.model_train import ModelTrain

@step
def train_model(X_train: pd.DataFrame, Y_train: pd.Series) -> None:
    """
    Method to train model that is choosen in the config file
        
    Args:
    X_train:pd.DataFrame training features for matches from 2023 to 2026
    X_test:pd.DataFrame testing features for matches from 2023 to 2026
    Y_train:pd.Series target variable for matches from 2023 to 2026
    Y_test:pd.Series target variable for matches from 2023 to 2026
    fixtures:pd.DataFrame cleaned data for upcoming fixtures for the 2026 season   

    Returns: None
    """
    try:
        trainer = ModelTrain()
        trainer.train(X_train, Y_train)
        logger.info('Completed model training step')
        return
    except Exception as e:
        raise e