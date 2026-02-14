from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """
    Model Configurations
    model_name options: linear_regression, random_forest, xgboost
    """
    
    model_name: str = "linear_regression"
    
