import pandas as pd
import numpy as np

import psycopg2
from dotenv import load_dotenv
import os

from typing import Tuple
from zenml.logger import get_logger

logger = get_logger(__name__)

class DataIngest:
    def __init__(self):
        pass

    def ingest(self):
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
                
            with connection:
                with connection.cursor() as cursor:
                    pass

            connection.close()
            logger.info("Database connection closed")

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise e
        return