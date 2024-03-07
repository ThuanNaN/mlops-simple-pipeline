import sys
import logging

class Logger:
    def __init__(self, name="", log_level=logging.INFO) -> None:
        self.log = logging.getLogger(name)
        self.log.setLevel(log_level)
        self._init_formatter()
        self._add_stream_hander()
    
    def _init_formatter(self):
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    def _add_stream_hander(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self.formatter)
        self.log.addHandler(stream_handler)