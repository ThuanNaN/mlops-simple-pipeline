import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from PIL import Image
import torch
from torch.nn import functional as F
import torchvision
from utils import AppPath, Logger, save_cache
from config.data_config import CatDog_Data
from dotenv import load_dotenv
load_dotenv()

LOGGER = Logger(__file__, log_file="predictor.log")
LOGGER.log.info("Starting Model Serving")

class Predictor:
    def __init__(self, model_name: str, model_alias: str, device: str = "cpu"):
        self.model_name = model_name
        self.model_alias = model_alias
        self.device = device
        self.load_model()
        self.create_transform()

    async def predict(self, image, image_name):
        pil_img = Image.open(image)
        LOGGER.save_requests(pil_img, image_name)

        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')

        transformed_image = self.transforms_(pil_img).unsqueeze(0)        
        output = await self.model_inference(transformed_image)
        probs, best_prob, predicted_id, predicted_class = self.output2pred(output)

        LOGGER.log_model(self.model_name, self.model_alias)
        LOGGER.log_response(best_prob, predicted_id, predicted_class)

        torch.cuda.empty_cache()
        save_cache(image_name, 
                    AppPath.CAPTURED_DATA_DIR,
                    self.model_name, 
                    self.model_alias, 
                    probs,
                    best_prob, 
                    predicted_id, 
                    predicted_class)
        return {"probs":probs,
                "best_prob": best_prob,
                "predicted_id": predicted_id, 
                "predicted_class": predicted_class, 
                "predictor_name": self.model_name, 
                "predictor_alias": self.model_alias}

    async def model_inference(self, input):
        input = input.to(self.device)
        with torch.no_grad():
            output = self.loaded_model(input.to(self.device)).cpu()
        return output    

    def load_model(self):
        try:
            self.loaded_model = torchvision.models.resnet50()
            in_features = self.loaded_model.fc.in_features
            self.loaded_model.fc = torch.nn.Linear(in_features, CatDog_Data.n_classes)
            self.loaded_model.to(self.device)
            self.loaded_model.eval()

            self.id2class = CatDog_Data.id2label
            self.class2id = CatDog_Data.label2id
            self.mean = CatDog_Data.mean
            self.std = CatDog_Data.std
            self.img_size = CatDog_Data.img_size

        except Exception as e:
            LOGGER.log.info(f"Load model failed")
            LOGGER.log.info(f"Error: {e}")
            return None

    def create_transform(self):
        self.transforms_ = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.img_size, self.img_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def output2pred(self, output):
        probabilities = F.softmax(output, dim=1)
        best_prob = torch.max(probabilities, 1)[0].item()
        predicted_id = torch.max(probabilities, 1)[1].item()
        predicted_class = self.id2class[predicted_id]
        return probabilities.squeeze().tolist(), round(best_prob, 6), predicted_id, predicted_class
