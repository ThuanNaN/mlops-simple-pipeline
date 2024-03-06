import os
from pathlib import Path
from dataclasses import dataclass
import torchvision

ROOT_DIR = Path(__file__).parent.parent

@dataclass
class DataPath:
    SOUCE_DIR = ROOT_DIR / "src"
    CONFIG_DIR = SOUCE_DIR / "config"
    DATA_DIR = ROOT_DIR / "data_source"

    RAW_DATA_DIR = DATA_DIR / "catdog_raw"
    COLLECTED_DATA_DIR = DATA_DIR / "collected"
    CAPTURED_DATA_DIR = DATA_DIR / "captured"
    TRAIN_DATA_DIR = DATA_DIR / "train_data"
    CACHE_DIR = DATA_DIR / "cache"


DataPath.CAPTURED_DATA_DIR.mkdir(parents=True, exist_ok=True)
DataPath.COLLECTED_DATA_DIR.mkdir(parents=True, exist_ok=True)
DataPath.TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
DataPath.CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CatDog_Data:
    n_classes = 2
    img_size = 224
    classes = ['cat', 'dog']
    id2class = {0: 'Cat', 1: 'Dog'}
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])


def seed_everything(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_cache(image_name, image_path, predictor_name, predictor__alias, probs, best_prob, pred_id, pred_class):
    cache_path = f"{DataPath.CACHE_DIR}/predicted_cache.csv"
    cache_exists = os.path.isfile(cache_path)
    with open(cache_path, "a") as f:
        if not cache_exists:
            f.write("Image_name, Image_path, Predictor_name, Predictor_alias, Probabilities , Best_prob, Predicted_id, Predicted_class\n")
        f.write(f"{image_name},{image_path},{predictor_name},{predictor__alias}, {probs},{best_prob},{pred_id},{pred_class}\n")

