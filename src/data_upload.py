import pandas as pd
import psycopg2
from psycopg2 import extras
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

            table_map = {
                "previous_fixtures_train": X_train,
                "previous_fixtures_test": X_test,
                "train_score": Y_train,
                "test_score": Y_test,
                "upcoming_fixtures": fixtures,
            }

            def _series_to_df(series: pd.Series) -> pd.DataFrame:
                column_name = series.name if series.name else "value"
                return series.to_frame(name=column_name)
            def _infer_pg_type(series: pd.Series) -> str:
                if pd.api.types.is_integer_dtype(series):
                    return "BIGINT"
                elif pd.api.types.is_float_dtype(series):
                    return "DOUBLE PRECISION"
                elif pd.api.types.is_bool_dtype(series):
                    return "BOOLEAN"
                elif pd.api.types.is_datetime64_any_dtype(series):
                    return "TIMESTAMP"
                else:
                    return "TEXT"
                
            with connection:
                with connection.cursor() as cursor:
                    for table_name, data in table_map.items():
                        if isinstance(data, pd.Series):
                            data = _series_to_df(data)

                        if data.empty:
                            logger.info("Skipped insert into %s (empty dataset)", table_name)
                            continue
                        
                        # Add row_id to preserve order
                        data = data.copy()
                        data.insert(0, 'row_id', range(len(data)))
                        
                        try:
                            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}";')
                            
                            # Create table with row_id as primary key
                            columns = list(data.columns)
                            column_definitions = []
                            for col in columns:
                                if col == 'row_id':
                                    column_definitions.append('"row_id" BIGINT PRIMARY KEY')
                                else:
                                    col_type = _infer_pg_type(data[col])
                                    column_definitions.append(f'"{col}" {col_type}')
                            
                            create_sql = f'CREATE TABLE "{table_name}" ({", ".join(column_definitions)})'
                            cursor.execute(create_sql)
                            logger.info(f"Created table {table_name}")
                            
                        except Exception as e:
                            logger.error(f"Error creating table {table_name}: {e}")
                            connection.rollback()
                            continue
                            
                        rows = data.itertuples(index=False, name=None)
                        quoted_columns = ', '.join(f'"{col}"' for col in columns)
                        insert_sql = f'INSERT INTO "{table_name}" ({quoted_columns}) VALUES %s'
                        extras.execute_values(cursor, insert_sql, rows)
                        logger.info(f"Inserted {len(data)} rows into {table_name}")

            connection.close()
            logger.info("Database connection closed")

        except Exception as e:
            raise e
        return