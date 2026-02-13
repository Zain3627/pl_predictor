import numpy as np
import pandas as pd
from zenml.logger import get_logger
from sklearn.model_selection import train_test_split
from typing import Tuple
from typing_extensions import Annotated

logger = get_logger(__name__)

class DataCleaning:
    """
    Class used for cleaning raw data
    """
    def __init__(self):
        pass

    def clean_data(self, data:pd.DataFrame, fixtures:pd.DataFrame) -> Tuple[
        Annotated[pd.DataFrame,'X_train'], 
        Annotated[pd.DataFrame,'X_test'],
        Annotated[pd.Series,'Y_train'],
        Annotated[pd.Series,'Y_test'],
        Annotated[pd.DataFrame,'fixtures'],
        ]:
        """
        Clean raw data
        
        Args: 
        data:pd.DataFrame raw data for matches from 2023 to 2026
        fixtures:pd.DataFrame raw data for upcoming fixtures for the 2026 season

        Returns: 
        X_train:pd.DataFrame training features for matches from 2023 to 2026
        X_test:pd.DataFrame testing features for matches from 2023 to 2026
        Y_train:pd.Series target variable for matches from 2023 to 2026
        Y_test:pd.Series target variable for matches from 2023 to 2026
        fixtures:pd.DataFrame cleaned data for upcoming fixtures for the 2026 season
        """
        data.drop([
            'Div', 'Date', 'Time', 'Referee', 'HTHG', 'HTAG', 'HTR', 'HY', 'AY', 'HTHG', 'HTAG'
        ], axis=1, inplace=True)
        data = data[['season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC' ]]

        # add home and away columns
        data['HP'] = data['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
        data['AP'] = data['FTR'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))
        data['HTP'] = data.groupby(['season','HomeTeam'])['HP'].cumsum() - data['HP']
        data['ATP'] = data.groupby(['season','AwayTeam'])['AP'].cumsum() - data['AP']
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
        
        home_snapshot = (data
            .groupby(["season", "HomeTeam"])
            .tail(1)
            .reset_index(drop=True))
        home_snapshot = home_snapshot[home_snapshot['season'] == 2026]
        home_snapshot.drop([
            'away_avg_FTAG', 'away_avg_AS', 'away_avg_AST', 'away_avg_AC', 'away_avg_AP', 'away_avg_ACS', 'A_goals/shot', 'away_avg_AP_squared', 'season', 'AwayTeam','FTR', 'ATP'
        ], axis=1, inplace=True)

        away_snapshot = (data
            .groupby(["season", "AwayTeam"])
            .tail(1)
            .reset_index(drop=True))
        away_snapshot = away_snapshot[away_snapshot['season'] == 2026]
        away_snapshot.drop([
            'home_avg_FTHG', 'home_avg_HS', 'home_avg_HST', 'home_avg_HC', 'home_avg_HP', 'home_avg_HCS', 'H_golas/shot', 'home_avg_HP_squared', 'season', 'HomeTeam', 'FTR', 'HTP'
        ], axis=1, inplace=True)

        home_snapshot.to_csv("/run/media/zain/Local Disk/Projects/Python/pl_predictor/data/home_snapshot.csv", index=False)
        away_snapshot.to_csv("/run/media/zain/Local Disk/Projects/Python/pl_predictor/data/away_snapshot.csv", index=False)
        data.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)       
        
        # fixtures data
        teams = {
            1 : 'Arsenal', 2 : 'Aston Villa', 3 : 'Burnley', 4 : 'Bournemouth', 5 : 'Brentford', 6 : 'Brighton', 7 : 'Chelsea', 8 : 'Crystal Palace', 
            9 : 'Everton', 10 : 'Fulham', 11 : 'Leeds', 12 : 'Liverpool', 13 : 'Man City', 14 : 'Man United', 15 : 'Newcastle', 
            16 : 'Nott\'m Forest', 17 : 'Sunderland', 18 : 'Tottenham', 19 : 'West Ham', 20 : 'Wolves'
        }
        fixtures = fixtures[[
            'team_h', 'team_a'
        ]]
        fixtures['HomeTeam'] = fixtures['team_h'].map(teams)
        fixtures['AwayTeam'] = fixtures['team_a'].map(teams)
        fixtures.drop(['team_h', 'team_a'], axis=1, inplace=True)
        
        fixtures = fixtures.merge(home_snapshot, how='left', left_on='HomeTeam', right_on='HomeTeam')
        fixtures = fixtures.merge(away_snapshot, how='left', left_on='AwayTeam', right_on='AwayTeam')
        fixtures.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)

        data.to_csv("/run/media/zain/Local Disk/Projects/Python/pl_predictor/data/averaged_previous_matches.csv", index=False)
        Y = data['FTR']
        Y = Y.map({'H': 0, 'D': 1, 'A': 2})
        X = data.drop('FTR', axis=1)
        # X is the previous matches and fixtures is the upcomming fixtures
        # make differential features for home and away teams
        X['diff_avg_FTHG'] = X['home_avg_FTHG'] - X['away_avg_FTAG']
        fixtures['diff_avg_FTHG'] = fixtures['home_avg_FTHG'] - fixtures['away_avg_FTAG']
        X['diff_avg_points'] = X['home_avg_HP'] - X['away_avg_AP']
        fixtures['diff_avg_points'] = fixtures['home_avg_HP'] - fixtures['away_avg_AP']
        X['diff_avg_total_points'] = X['HTP'] - X['ATP']
        fixtures['diff_avg_total_points'] = fixtures['HTP'] - fixtures['ATP']
        X['diff_avg_CS'] = X['home_avg_HCS'] - X['away_avg_ACS']
        fixtures['diff_avg_CS'] = fixtures['home_avg_HCS'] - fixtures['away_avg_ACS']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=27, stratify=X['season'])
        X_train.drop('season', axis=1, inplace=True)
        X_test.drop('season', axis=1, inplace=True)
        X_train.to_csv("/run/media/zain/Local Disk/Projects/Python/pl_predictor/data/X_train.csv", index=False)
        fixtures.to_csv("/run/media/zain/Local Disk/Projects/Python/pl_predictor/data/upcoming_fixtures.csv", index=False, encoding='utf-8')
        return X_train, X_test, Y_train, Y_test, fixtures
