import pandas as pd
import numpy as np

import psycopg2
from dotenv import load_dotenv
import os

from typing import Tuple
from typing_extensions import Annotated
from zenml.logger import get_logger

logger = get_logger(__name__)

class DataIngest:
    def __init__(self):
        pass

    def ingest(self) -> Tuple[
        Annotated[pd.DataFrame,'X_train'], 
        Annotated[pd.DataFrame,'X_test'],
        Annotated[pd.Series,'Y_train'],
        Annotated[pd.Series,'Y_test'],
        Annotated[pd.DataFrame,'fixtures'],
        ]:
        """
        Method that ingests the clean and ready data from database

        Args: None

        Returns:
        X_train:pd.DataFrame training features for matches from 2023 to 2026
        X_test:pd.DataFrame testing features for matches from 2023 to 2026
        Y_train:pd.Series target variable for matches from 2023 to 2026
        Y_test:pd.Series target variable for matches from 2023 to 2026
        fixtures:pd.DataFrame cleaned data for upcoming fixtures for the 2026 season        
        """
        load_dotenv()

        # Fetch variables
        USER = os.getenv("user")
        PASSWORD = os.getenv("password")
        HOST = os.getenv("host")
        PORT = os.getenv("port")
        DBNAME = os.getenv("dbname")
        
        # Connect to the database
        try:
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
                # Fetch training features
                cursor.execute('SELECT * FROM "previous_fixtures_train" ORDER BY row_id')
                X_train = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
                X_train = X_train.drop(columns=['row_id'])
                logger.info(f"Fetched X_train: {len(X_train)} rows")
                
                # Fetch testing features
                cursor.execute('SELECT * FROM "previous_fixtures_test" ORDER BY row_id')
                X_test = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
                X_test = X_test.drop(columns=['row_id'])
                logger.info(f"Fetched X_test: {len(X_test)} rows")
                
                # Fetch training target
                cursor.execute('SELECT * FROM "train_score" ORDER BY row_id')
                Y_train_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
                Y_train = Y_train_df.drop(columns=['row_id']).squeeze()
                logger.info(f"Fetched Y_train: {len(Y_train)} rows")
                
                # Fetch testing target
                cursor.execute('SELECT * FROM "test_score" ORDER BY row_id')
                Y_test_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
                Y_test = Y_test_df.drop(columns=['row_id']).squeeze()
                logger.info(f"Fetched Y_test: {len(Y_test)} rows")
                
                # Fetch upcoming fixtures
                cursor.execute('SELECT * FROM "upcoming_fixtures" ORDER BY row_id')
                fixtures = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
                fixtures = fixtures.drop(columns=['row_id'])
                logger.info(f"Fetched fixtures: {len(fixtures)} rows")
            
            connection.close()
            logger.info("Database connection closed")
            return X_train, X_test, Y_train, Y_test, fixtures

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise e