from typing import Annotated, Tuple

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
    def predict_matches(self, model: ClassifierMixin, fixtures: pd.DataFrame, team_ids_df: pd.DataFrame, league_table: pd.DataFrame) -> Tuple[
        Annotated [pd.DataFrame, "Predicted league table"],
        Annotated [pd.DataFrame, "Predicted fixtures with team IDs and predictions"]
    ]:
        """
        Method to predict the upcoming matches and output the final league standings.

        Args: 
        trained_model: ClassifierMixin trained model object
        fixtures: pd.DataFrame upcoming fixtures to predict
        team_ids_df: pd.DataFrame team IDs for upcoming fixtures
        league_table: pd.DataFrame league table for the current season
        
        Returns:
        pd.DataFrame: Predicted table        
        pd.DataFrame: Predicted fixtures with team IDs and predictions
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

            predicted_with_team_ids = pd.concat([team_ids_df.reset_index(drop=True), pd.Series(predictions, name='predictions')], axis=1)

            fixtures['predicted_points'] = predictions
            logger.info('Predictions made successfully')
            fixtures.to_csv('predicted_fixtures.csv', index=False)

            league_table = league_table.set_index('team')['total_points'].to_dict()
            # print('-'*50 , league_table, '-'*50)
            team_ids_df['predictions'] = predictions

            for _, row in team_ids_df.iterrows():
                home, away, result = row['HomeTeam'], row['AwayTeam'], row['predictions']
                if result == 0:
                    league_table[home] += 3
                elif result == 2:
                    league_table[away] += 3
                elif result == 1:
                    league_table[home] += 1
                    league_table[away] += 1
            
            league_table = pd.DataFrame(
                list(league_table.items()), columns=['team', 'total_points']
            ).sort_values('total_points', ascending=False).reset_index(drop=True)

            league_table.to_csv('/mnt/localdisk/Projects/Python/pl_predictor/data/predicted_league_table.csv', index=False)

            return league_table, predicted_with_team_ids
        except Exception as e:
            logger.error(f'Error predicting fixtures: {e}')
            raise e