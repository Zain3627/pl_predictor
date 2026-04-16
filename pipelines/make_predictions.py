from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.predict_model import predict_model
from steps.upload_table import upload_data
from zenml.logger import get_logger

logger = get_logger(__name__)

@pipeline(enable_cache=False)
def make_predictions():
    try:
        X_train, X_test, Y_train, Y_test, fixtures, team_ids_df, league_table = ingest_data()
        model, model_version = train_model(X_train, Y_train)
        accuracy, precision, recall, f1_score = evaluate_model(model, X_test, Y_test, model_version)
        league_table, predicted_with_team_ids = predict_model(model, fixtures, team_ids_df, league_table)
        upload_data(league_table, predicted_with_team_ids)
    except Exception as e:
        logger.error(e)
        raise e
