import pandas as pd
import numpy as np
import requests
from zenml.logger import get_logger

logger = get_logger(__name__)

class DataExtraction:
    """
    Class used for loading data from endpoint
    """
    def __init__(self):
        pass

    def load_api(self) -> pd.DataFrame:
        """
        Method to load data from API endpoint and return a dataframe 

        Args: None

        Returns: pd.DataFrame: Dataframe containing the data from the API for all previous matches from 2023 to 2026
        """
        try:
            season_urls = {
                '2023' : "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
                '2024' : 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
                '2025' : 'https://www.football-data.co.uk/mmz4281/2425/E0.csv',
                '2026' : 'https://www.football-data.co.uk/mmz4281/2526/E0.csv'
            }

            previous_matches = []
            for season,url in season_urls.items():
                response = requests.get(url, timeout=10)
                from io import StringIO
                content = response.content.decode('utf-8-sig')
                season_data = pd.read_csv(StringIO(content))
                season_data['season'] = season
                previous_matches.append(season_data)
                
            previous_matches = pd.concat(previous_matches, ignore_index=True)
            previous_matches.to_csv("/run/media/zain/Local Disk/Projects/Python/pl_predictor/data/previous_matches.csv", index=False, encoding='utf-8')
            return previous_matches
        except Exception as e:
            logger.error(e)
            raise e