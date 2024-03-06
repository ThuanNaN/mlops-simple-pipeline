import os
import json
import torch
from PIL import Image
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
from torch.nn import functional as F
from config import ServeConfig
from utils import CatDog_Data, DataPath, save_cache
from logger import Logger

from dotenv import load_dotenv
load_dotenv()

LOGGER = Logger(__file__, log_file="model_serving.log")
LOGGER.log.info("Starting Model Serving")

class ResponsePrediction(BaseModel):
    probs: list = []
    best_prob: float = -1.0
    predicted_id: int = -1
    predicted_class: str = "unknown"
    predictor_name: str = "unknown"
    predictor_alias: str = "unknown"

class ModelServing:
    def __init__(self, serve_config: ServeConfig):
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[os.getenv("Frontend_URL")],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.serve_config = serve_config
        self.load_model()

        @self.app.get("/")
        async def health():
            return "OK"

        @self.app.post("/predict", response_model=ResponsePrediction, tags=["Predict"])
        async def predict(file_upload: UploadFile = File(...)):
            LOGGER.log.info(f"Received image: {file_upload.filename}")
            try: 
                pil_img = Image.open(file_upload.file)
                LOGGER.save_requests(pil_img, file_upload.filename)
            except Exception as e:
                LOGGER.log.error(f"Error: {e}")

            if pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB')

            transformed_image = CatDog_Data.test_transform(pil_img).unsqueeze(0)
            output = self.loaded_model(transformed_image.to(self.device)).detach().cpu()
            probs, best_prob, predicted_id, predicted_class = self.output2pred(output)

            LOGGER.log_model(self.serve_config.model_name, self.serve_config.model_alias)
            LOGGER.log_response(best_prob, predicted_id, predicted_class)

            torch.cuda.empty_cache()
            save_cache(file_upload.filename, 
                        DataPath.CAPTURED_DATA_DIR, 
                        self.serve_config.model_name, 
                        self.serve_config.model_alias, 
                        probs,
                        best_prob, 
                        predicted_id, 
                        predicted_class)

            return ResponsePrediction(probs=probs,
                                      best_prob = best_prob,
                                      predicted_id=predicted_id, 
                                      predicted_class=predicted_class, 
                                      predictor_name=self.serve_config.model_name, 
                                      predictor__alias=self.serve_config.model_alias)


        @self.app.middleware("http")
        async def log_requests(request, call_next):
            response = await call_next(request)
            LOGGER.log.info(
                f"{request.client.host} - \"{request.method} {request.url.path} {request.scope['http_version']}\" {response.status_code}")
            return response


    def load_model(self):
        try:
            MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()
            model_info = client.get_model_version_by_alias(
                name=self.serve_config.model_name, alias=self.serve_config.model_alias)
            self.loaded_model = mlflow.pytorch.load_model(
                model_info.source, map_location=self.device)
            LOGGER.log.info(f"Model {self.serve_config.model_name} loaded")
        except Exception as e:
            LOGGER.log.info(f"Load model failed")
            LOGGER.log.info(f"Error: {e}")

    def output2pred(self, output):
        probabilities = F.softmax(output, dim=1)
        best_prob = torch.max(probabilities, 1)[0].item()
        predicted_id = torch.max(probabilities, 1)[1].item()
        predicted_class = CatDog_Data.id2class[predicted_id]
        return probabilities.squeeze().tolist(), round(best_prob, 6), predicted_id, predicted_class

    def run(self, host, port):
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    host = os.getenv("HOST")
    port = int(os.getenv("API_PORT"))
    config_name = os.getenv("MODEL_CONFIG")

    with open(DataPath.CONFIG_DIR / f"{config_name}.json", "r") as f:
        config = json.load(f)
    serve_config = ServeConfig(**config)
    LOGGER.log.info(f"Serve config: {serve_config}")

    model_serving = ModelServing(serve_config)
    model_serving.run(host=host, port=port)