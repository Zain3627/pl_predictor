import numpy as np
import pandas as pd
from zenml.logger import get_logger

logger = get_logger(__name__)

class DataCleaning:
    """
    Class used for cleaning raw data
    """
    def __init__(self):
        pass

    def clean_data(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data
        
        Args: data:pd.DataFrame

        returns: data:pd.DataFrame cleaned data for matches from 2023 to 2026
        """
        data.drop([
            'Div', 'Date', 'Time', 'Referee', 'HTHG', 'HTAG', 'HTR', 'HY', 'AY', 'HTHG', 'HTAG'
        ], axis=1, inplace=True)
        data = data[['season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC' ]]
        data.to_csv("/run/media/zain/Local Disk/Projects/Python/pl_predictor/data/cleaned_previous_matches.csv", index=False)

        # add home and away columns
        data['HP'] = data['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
        data['AP'] = data['FTR'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))
        data['HCS'] = data['FTAG'].apply(lambda x : 1 if x == 0 else 0)
        data['ACS'] = data['FTHG'].apply(lambda x : 1 if x == 0 else 0)

        home_team_features = ['FTHG','HS', 'HST', 'HC', 'HP' ,'HCS']
        for col in home_team_features:
            data[f'home_avg_{col}'] = (
                data.groupby(['season','HomeTeam'])[col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            ).fillna(data[col])

        away_team_features = ['FTAG','AS', 'AST', 'AC', 'AP', 'ACS']
        for col in away_team_features:
            data[f'away_avg_{col}'] = (
                data.groupby(['season','AwayTeam'])[col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            ).fillna(data[col])

        data['H_golas/shot'] = data['home_avg_FTHG'] / data['home_avg_HS']
        data['A_goals/shot'] = data['away_avg_FTAG'] / data['away_avg_AS'] 
        data['home_avg_HP_squared'] = data['home_avg_HP'] ** 2
        data['away_avg_AP_squared'] = data['away_avg_AP'] ** 2

        data.drop([
                    'FTHG','FTAG','HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HP' ,'HCS', 'AP', 'ACS'
                    ], axis=1, inplace=True)
        
        snapshot = pd.concat([
            data
            .groupby(["season", "HomeTeam"])
            .tail(1)
            .reset_index(drop=True),
            data
            .groupby(["season", "AwayTeam"])
            .tail(1)
            .reset_index(drop=True)
        ])
        snapshot = snapshot[snapshot['season'] == '2026']
        snapshot.to_csv("/run/media/zain/Local Disk/Projects/Python/pl_predictor/data/snapshot_previous_matches.csv", index=False)
        data.to_csv("/run/media/zain/Local Disk/Projects/Python/pl_predictor/data/averaged_previous_matches.csv", index=False)
      
        data.drop(['season', 'HomeTeam', 'AwayTeam'], axis=1, inplace=True)       
        return data
