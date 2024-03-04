import logging
from logging.handlers import RotatingFileHandler
import sys
from dataclasses import dataclass
import torchvision
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
LOG_DIR = ROOT_DIR / "logs"

@dataclass
class DataPath:
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
    def __init__(self, name="") -> None:
        self.logger = logging.getLogger(name)
    
    def get_logger(self, log_level=logging.INFO, log_file=None):
        self.log_level = log_level
        self.logger.setLevel(log_level)
        self.init_formatter()
        if log_file is not None:
            self._add_file_hander(LOG_DIR / log_file)
        else:
            self._add_stream_hander()
        return self.logger
    
    def init_formatter(self):
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    def _add_stream_hander(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(stream_handler)

    def _add_file_hander(self, log_file):
        file_handler = RotatingFileHandler(log_file, maxBytes=2000, backupCount=10)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)


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

    