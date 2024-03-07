from dataclasses import dataclass

@dataclass
class BaseServeConfig:
    config_name: str = "raw_data"
    model_name: str = "resnet_18"
    model_alias: str = "Production"

