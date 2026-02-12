import numpy as np
import pandas as pd
from zenml.logger import get_logger

logger = get_logger(__name__)

class DataCleaning:
    """
    Docstring for DataCleaning
    """
    def __init__(self):
        pass

    def clean_data(self, data:pd.DataFrame) -> pd.DataFrame:
        return data