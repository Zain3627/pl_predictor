from zenml import pipeline, step
from zenml.logger import get_logger

from steps.fetch_data import fetch_data

@pipeline
def data_fetching():
    """
    Pipeline to load data from API endpoint
    Args: None
    Returns: None    
    """

    previous_matches = fetch_data()