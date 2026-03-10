import pandas as pd
import numpy as np
import requests
from typing import Tuple
from zenml.logger import get_logger
import psycopg2
from dotenv import load_dotenv
import os

logger = get_logger(__name__)

class DataPreparation:
    """
    Class used for preparing data for current results live evaluation
    """
    def __init__(self):
        pass

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method for preparing data for current results live evaluation.

        Args:
        None

        Returns:
        final_result:np.ndarray - Array containing the actual results for the fixtures that have been played
        current_predictions:np.ndarray - Array containing the predicted results for the fixtures that have been played
        """
        try:
            url = 'https://www.football-data.co.uk/mmz4281/2526/E0.csv'
            response = requests.get(url, timeout=10)
            from io import StringIO
            content = response.content.decode('utf-8-sig')
            season_data = pd.read_csv(StringIO(content))
            season_data = season_data[['HomeTeam', 'AwayTeam', 'FTR']]
            season_data['FTR'] = season_data['FTR'].map({'H': 0, 'D': 1, 'A': 2})

            load_dotenv()

            # Fetch variables
            USER = os.getenv("user")
            PASSWORD = os.getenv("password")
            HOST = os.getenv("host")
            PORT = os.getenv("port")
            DBNAME = os.getenv("dbname")
            
            # Connect to the database
            connection = psycopg2.connect(
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME,
            )
            logger.info("Database connection successful")
                
            # Fetch the data
            with connection.cursor() as cursor:
                
                # Fetch predicted_fixtures table
                cursor.execute('SELECT * FROM "predicted_fixtures" ORDER BY row_id')
                predicted_fixtures = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
                predicted_fixtures = predicted_fixtures.drop(columns=['row_id'])
                logger.info(f"Fetched predicted_fixtures: {len(predicted_fixtures)} rows")

            connection.close()

            predicted_fixtures = predicted_fixtures[['HomeTeam', 'AwayTeam', 'predictions']]
            # Build a dataframe of finished matches that we have predictions for
            finished_with_predictions = predicted_fixtures.merge(
                season_data,
                on=['HomeTeam', 'AwayTeam'],
                how='inner'
            )
            final_result = np.array([])
            current_predictions = np.array([])
            if not finished_with_predictions.empty:
                finished_with_predictions.to_csv("/mnt/localdisk/Projects/Python/pl_predictor/data/finished_with_predictions.csv", index=False, encoding='utf-8')
                final_result = np.array(finished_with_predictions['FTR'])
                current_predictions = np.array(finished_with_predictions['predictions'])

            return final_result, current_predictions
        except Exception as e:
            logger.error(e)
            raise e