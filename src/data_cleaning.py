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

        team_stats= map()
        
        rolling_features.to_csv("/run/media/zain/Local Disk/Projects/Python/pl_predictor/data/averaged_previous_matches.csv", index=False)
        return rolling_features
