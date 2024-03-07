import argparse
import os   
import json
from dataclasses import asdict
import mlflow
from mlflow.tracking import MlflowClient
from utils import Logger, AppPath
from config.serve_config import BaseServeConfig
from dotenv import load_dotenv
load_dotenv()

LOGGER = Logger(__file__)
LOGGER.log.info("Starting Model Registry")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="raw_data", 
                        help="Name of the config file")
    parser.add_argument("--filter_string", type=str, default="", 
                        help="Filter string for searching runs in MLflow tracking server")
    parser.add_argument("--best_metric", type=str, choices=["best_val_loss", "best_val_acc"], default="best_val_loss", 
                        help="Metric for selecting the best model")
    parser.add_argument("--model_alias", type=str, default="Production", 
                        help="Alias tag of the model. Help to identify the model in the model registry.")
    args = parser.parse_args()


    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    LOGGER.log.info(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")

    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
    experiment_ids = dict(mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME))['experiment_id']

    client = MlflowClient()
    try:
        best_runs = client.search_runs(experiment_ids, 
                                       filter_string=args.filter_string,
                                       order_by=[f"metrics.{args.best_metric} DESC"]
                                       )[-1]
    except:
        LOGGER.log.info("No runs found")
        exit(0)
    
    LOGGER.log.info(f"Best run: {best_runs.info.run_id}")

    model_name = best_runs.data.params["model_name"] 

    try:
        client.create_registered_model(model_name)
    except:
        pass
    run_id = best_runs.info.run_id
    model_uri = f'runs:/{run_id}/model'
    mv = client.create_model_version(model_name, model_uri, run_id)
    LOGGER.log.info(f"Registered model: {model_name}, version: {mv.version}")
    client.set_registered_model_alias(model_name, args.model_alias, mv.version)

    server_config = BaseServeConfig(config_name=args.config_name,
                                    model_name=model_name, 
                                    model_alias=args.model_alias)

    path_save_cfg = AppPath.SERVE_CONFIG / f"{args.config_name}.json"
    with open(path_save_cfg, 'w+') as f:
        json.dump(asdict(server_config), f, indent=4)
        
    LOGGER.log.info(f"Config saved to {args.config_name}.json")

    LOGGER.log.info(f"Model {model_name} registered with alias {args.model_alias} and version {mv.version}")
    LOGGER.log.info("Model Registry completed")

