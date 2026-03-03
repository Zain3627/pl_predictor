import pandas as pd
from zenml.logger import get_logger
logger = get_logger(__name__)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.base import ClassifierMixin

from src.config import ModelNameConfig

class ModelTrain:
    def __init__(self):
        pass

    def train(self, X_train: pd.DataFrame, Y_train: pd.Series) -> ClassifierMixin:
        """
        Train model that is choosen in the config file
        
        Args:
        X_train:pd.DataFrame training features for matches from 2023 to 2026
        Y_train:pd.Series target variable for matches from 2023 to 2026

        Returns: 
        trained_model: ClassifierMixin trained model object
        """   
         
        try:
            config = ModelNameConfig()
            logger.info('Training model')
            model = config.model_name
            if model == 'logistic_regression':
                trained_model = LogisticRegression(max_iter=1000)

            elif model == 'random_forest':
                trained_model = RandomForestClassifier()

            elif model == 'xgboost':
                trained_model = XGBClassifier()

            else :
                raise ValueError(f"Model {model} is not supported")
            
            trained_model.fit(X_train, Y_train)

            logger.info('Model trained successfully')
            return trained_model
        except Exception as e:
            logger.error(e)
            raise e
    