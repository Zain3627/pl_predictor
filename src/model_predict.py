import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin

from zenml.logger import get_logger
logger = get_logger(__name__)

from abc import ABC, abstractmethod

class ModelPredict():
    """
    Class to predict the upcoming fixtures and complete the pl table.
    """
    def predict_matches(self, model: ClassifierMixin, fixtures: pd.DataFrame) -> pd.DataFrame:
        """
        Method to predict the upcoming matches and output the final league standings.

        Args: 
        trained_model: ClassifierMixin trained model object
        fixtures: pd.DataFrame upcoming fixtures to predict

        Returns:
        pd.DataFrame: Predicted table        
        """
        try:
            logger.info('Predicting upcoming fixtures')
            feature_order = ['HTP', 'ATP', 'home_avg_FTHG', 'home_avg_HS', 'home_avg_HST', 
                 'home_avg_HC', 'home_avg_HP', 'home_avg_HCS', 'away_avg_FTAG', 
                 'away_avg_AS', 'away_avg_AST', 'away_avg_AC', 'away_avg_AP', 
                 'away_avg_ACS', 'H_golas/shot', 'A_goals/shot', 'home_avg_HP_squared', 
                 'away_avg_AP_squared', 'diff_avg_FTHG', 'diff_avg_points', 
                 'diff_avg_total_points', 'diff_avg_CS']
            fixtures = fixtures[feature_order]
            predictions = model.predict(fixtures)
            fixtures['predicted_points'] = predictions
            fixtures[fixtures['predicted_points']==2] =  3
            logger.info('Predictions made successfully')
            fixtures.to_csv('predicted_fixtures.csv', index=False)
            return fixtures
        except Exception as e:
            logger.error(f'Error predicting fixtures: {e}')
            raise e