import pandas as pd
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv
import os

from zenml.logger import get_logger
logger = get_logger(__name__)


class TableUpload:
    def __init__(self):
        pass

    def table_upload(self, predicted_fixtures: pd.DataFrame, predicted_with_team_ids: pd.DataFrame) -> None:
        """
        Upload the predicted league table to the database.

        Args:
        predicted_fixtures:pd.DataFrame predicted final league standings (team, total_points)
        predicted_with_team_ids: pd.DataFrame predicted fixtures with team IDs and predictions (HomeTeam, AwayTeam, predictions)

        Returns: None
        """
        load_dotenv()

        USER = os.getenv("user")
        PASSWORD = os.getenv("password")
        HOST = os.getenv("host")
        PORT = os.getenv("port")
        DBNAME = os.getenv("dbname")

        try:
            connection = psycopg2.connect(
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME,
            )
            logger.info("Database connection successful")

            tables = {
                "predicted_league_table": predicted_fixtures.copy(),
                "predicted_fixtures": predicted_with_team_ids.copy(),
            }

            def upload_table(cursor, table_name: str, data: pd.DataFrame) -> None:
                data.insert(0, 'row_id', range(len(data)))
                cursor.execute(f'DROP TABLE IF EXISTS "{table_name}";')

                columns = list(data.columns)
                column_definitions = ['"row_id" BIGINT PRIMARY KEY']
                for col in columns[1:]:
                    if pd.api.types.is_integer_dtype(data[col]):
                        pg_type = "BIGINT"
                    elif pd.api.types.is_float_dtype(data[col]):
                        pg_type = "DOUBLE PRECISION"
                    else:
                        pg_type = "TEXT"
                    column_definitions.append(f'"{col}" {pg_type}')

                cursor.execute(
                    f'CREATE TABLE "{table_name}" ({", ".join(column_definitions)})'
                )
                logger.info(f"Created table {table_name}")

                rows = list(data.itertuples(index=False, name=None))
                quoted_columns = ', '.join(f'"{col}"' for col in columns)
                extras.execute_values(
                    cursor,
                    f'INSERT INTO "{table_name}" ({quoted_columns}) VALUES %s',
                    rows
                )
                logger.info(f"Inserted {len(data)} rows into {table_name}")

            with connection:
                with connection.cursor() as cursor:
                    for table_name, data in tables.items():
                        upload_table(cursor, table_name, data)

            connection.close()
            logger.info("All tables uploaded successfully")

        except Exception as e:
            logger.error(f"Error uploading predicted league table: {e}")
            raise e