import sys
import logging
from logging.handlers import RotatingFileHandler
from utils import DataPath, ROOT_DIR

LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class Logger:
    def __init__(self, name="", log_level=logging.INFO, log_file=None) -> None:
        self.log = logging.getLogger(name)
        self.get_logger(log_level, log_file)
    
    def get_logger(self, log_level, log_file):
        self.log.setLevel(log_level)
        self._init_formatter()
        if log_file is not None:
            self._add_file_hander(LOG_DIR / log_file)
        else:
            self._add_stream_hander()
    
    def _init_formatter(self):
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    def _add_stream_hander(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self.formatter)
        self.log.addHandler(stream_handler)

    def _add_file_hander(self, log_file):
        file_handler = RotatingFileHandler(log_file, maxBytes=10000, backupCount=10)
        file_handler.setFormatter(self.formatter)
        self.log.addHandler(file_handler)

    def save_requests(self, image, image_name):
        path_save = f"{DataPath.CAPTURED_DATA_DIR}/{image_name}"
        self.log.info(f"Save image to {path_save}")
        image.save(path_save)

    def log_model(self, predictor_name, predictor__alias):
        self.log.info(f"Predictor name: {predictor_name} -  Predictor alias: {predictor__alias}")

    def log_response(self, pred_prob, pred_id, pred_class):
        self.log.info(f"Predicted Prob: {pred_prob} -  Predicted ID: {pred_id} -  Predicted Class: {pred_class}")

