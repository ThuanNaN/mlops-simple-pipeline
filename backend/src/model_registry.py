import argparse
from dataclasses import asdict
import os   
import json
from utils import Log, DataPath
from dotenv import load_dotenv
load_dotenv()
from config.serve_config import ServeConfig
import mlflow
from mlflow.tracking import MlflowClient


logger = Log(__file__).get_logger()
logger.info("Starting Model Registry")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_tag", type=str, default="raw_data")
    parser.add_argument("--filter_string", type=str, default="")
    parser.add_argument("--metric", type=str, choices=["val_loss", "val_acc"], default="val_loss")
    parser.add_argument("--alias", type=str, default="Production")
    args = parser.parse_args()


    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")

    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
    experiment_ids = dict(mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME))['experiment_id']

    client = MlflowClient()
    try:
        best_runs = client.search_runs(experiment_ids, 
                                       filter_string=args.filter_string,
                                       order_by=[f"metrics.{args.metric} DESC"]
                                       )[-1]
    except:
        logger.info("No runs found")
        exit(0)
    
    logger.info(f"Best run: {best_runs.info.run_id}")

    model_name = best_runs.data.params["model_name"] 

    try:
        client.create_registered_model(model_name)
    except:
        pass
    
    run_id = best_runs.info.run_id
    model_uri = f'runs:/{run_id}/model'
    mv = client.create_model_version(model_name, model_uri, run_id)
    logger.info(f"Registered model: {model_name}, version: {mv.version}")
    client.set_registered_model_alias(model_name, args.alias, mv.version)

    server_config = ServeConfig(config_tag=args.config_tag,
                                model_name=model_name, 
                                model_alias=args.alias)

    path_save_cfg = DataPath.CONFIG_DIR / f"{args.config_tag}.json"
    with open(path_save_cfg, 'w+') as f:
        json.dump(asdict(server_config), f, indent=4)
        
    logger.info(f"Config saved to {args.config_tag}.json")

    logger.info(f"Model {model_name} registered with alias {args.alias} and version {mv.version}")
    logger.info("Model Registry completed")

