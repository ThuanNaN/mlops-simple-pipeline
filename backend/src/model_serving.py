import os   
import uvicorn
from dotenv import load_dotenv
load_dotenv()
from utils import CatDog_Data, Log, DataPath
from config import ServeConfig
from PIL import Image
import torch
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

logger = Log(__file__).get_logger(log_file="model_serving.log")
logger.info("Starting Model Serving")


class Response(BaseModel):
    id: int
    label: str


class ModelServing:
    def __init__(self):
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[os.getenv("Frontend_URL")],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

        @self.app.get("/")
        async def health():
            return "OK"

        @self.app.post("/predict")
        async def predict(file_upload: UploadFile = File(...)):
            pil_img = Image.open(file_upload.file)

            if pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB')
            
            logger.info(f"Received image: {file_upload.filename}")
            ModelServing.log_request(pil_img, file_upload.filename)

            transformed_image = CatDog_Data.test_transform(pil_img).unsqueeze(0)
            output = self.loaded_model(transformed_image.to(self.device)).detach().cpu()
            predicted_id = torch.max(output, 1)[1].item()
            pred_class = CatDog_Data.id2class[predicted_id]
            torch.cuda.empty_cache()
            return Response(id=predicted_id, label=pred_class)

    def load_model(self):
        try:
            MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()
            model_info = client.get_model_version_by_alias(name=ServeConfig.model_name, alias=ServeConfig.model_alias)
            self.loaded_model = mlflow.pytorch.load_model(model_info.source, map_location=self.device)
            logger.info(f"Model {ServeConfig.model_name} loaded")
        except Exception as e:
            logger.info(f"Load model failed")
            logger.info(f"Error: {e}")

    def run(self, host, port):
        uvicorn.run(self.app, host=host, port=port)
    
    @staticmethod
    def log_request(image, image_name):
        path_save = f"{DataPath.CAPTURED_DATA_DIR}/{image_name}"
        logger.info(f"Save image to {path_save}")
        image.save(path_save)
    
    @staticmethod
    def log_response(response):
        pass


if __name__ == "__main__":
    host = os.getenv("API_HOST")
    port = int(os.getenv("API_PORT"))

    model_serving = ModelServing()
    model_serving.run(host=host, port=port)
