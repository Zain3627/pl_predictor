from pydantic import BaseModel


class ModelNameConfig(BaseModel):
    """
    Model Configurations
    model_name options: linear_regression, random_forest, xgboost
    """
    
    model_name: str = "xgboost"