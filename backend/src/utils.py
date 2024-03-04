import logging
import sys
from dataclasses import dataclass
import torchvision
from pathlib import Path

class DataPath:
    ROOT_DIR = Path(__file__).parent.parent
    SOUCE_DIR = ROOT_DIR / "src"
    DATA_DIR = ROOT_DIR / "data_source"

    RAW_DATA_DIR = DATA_DIR / "catdog_raw"
    COLLECTED_DATA_DIR = DATA_DIR / "collected"
    CAPTURED_DATA_DIR = DATA_DIR / "captured"
    TRAIN_DATA_DIR = DATA_DIR / "train_data"

DataPath.CAPTURED_DATA_DIR.mkdir(parents=True, exist_ok=True)
DataPath.COLLECTED_DATA_DIR.mkdir(parents=True, exist_ok=True)
DataPath.TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)


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

class Log:
    log: logging.Logger = None

    def __init__(self, name="", log_level=logging.INFO) -> None:
        if Log.log == None:
            Log.log = self._init_logger(name, log_level)

    def _init_logger(self, name, log_level):
        logger = logging.getLogger(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(log_level)
        return logger

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

    