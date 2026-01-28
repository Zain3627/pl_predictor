from zenml import pipeline, step
from zenml.logger import get_logger

from steps.ingest_data import ingest_data

@pipeline
def data_fetching():
    """
    Docstring for data_fetching
    
    """

    previous_matches = ingest_data()