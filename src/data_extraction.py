import pandas as pd
import numpy as np
import requests

class DataExtraction:
    def __init__(self):
        pass

    def load_api(self) -> pd.DataFrame:
        season_urls = {
            '2023' : "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
            '2024' : 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
            '2025' : 'https://www.football-data.co.uk/mmz4281/2425/E0.csv',
            '2026' : 'https://www.football-data.co.uk/mmz4281/2526/E0.csv'
        }

        previous_matches = []
        for season,url in season_urls.items():
            try:
                response = requests.get(url, timeout=10)
                from io import StringIO
                season_data = pd.read_csv(StringIO(response.text))
                previous_matches.append(season_data)
            except Exception as e:
                raise e

        return previous_matches
    