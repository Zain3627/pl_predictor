from pydantic import BaseModel


class ModelNameConfig(BaseModel):
    """
    Model Configurations
    model_name options: logistic_regression, random_forest, xgboost
    """
    
    model_name: str = "xgboost"