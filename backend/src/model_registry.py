import argparse
import os   
from utils import Log
from dotenv import load_dotenv
load_dotenv()
import mlflow
from mlflow.tracking import MlflowClient

logger = Log(__file__).get_logger()
logger.info("Starting Model Registry")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, required=True, choices=["Val_loss", "Val_acc"], default="Val_loss")
    parser.add_argument("--alias", type=str, default="Production")
    args = parser.parse_args()


    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")

    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
    experiment_ids = dict(mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME))['experiment_id']

    client = MlflowClient()
    best_runs = client.search_runs(experiment_ids, order_by=[f"metrics.{args.metric} DESC"])[-1]
    logger.info(f"Best run: {best_runs.info.run_id}")

    name = best_runs.data.params["model_name"] 

    try:
        client.create_registered_model(name)
    except:
        pass
    
    run_id = best_runs.info.run_id
    model_uri = f'runs:/{run_id}/model'
    mv = client.create_model_version(name, model_uri, run_id)
    logger.info(f"Registered model: {name}, version: {mv.version}")
    client.set_registered_model_alias(name, args.alias, mv.version)

