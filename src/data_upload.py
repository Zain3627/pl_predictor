import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

from zenml.logger import get_logger
logger = get_logger(__name__)


class DataUpload:
    def __init__(self):
        pass

    def upload(self, X_train:pd.DataFrame, X_test:pd.DataFrame, Y_train:pd.Series, Y_test:pd.Series, fixtures:pd.DataFrame) -> None:
        
        # Load environment variables from .env
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
            
            # Create a cursor to execute SQL queries
            # cursor = connection.cursor()
            
            # # Example query
            # cursor.execute("SELECT NOW();")
            # result = cursor.fetchone()
            # print("Current Time:", result)

            # # Close the cursor and connection
            # cursor.close()
            connection.close()
            logger.info("Database connection closed")

        except Exception as e:
            raise e
        return