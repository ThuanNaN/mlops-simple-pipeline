from dataclasses import dataclass

@dataclass
class ServeConfig:
    config_tag: str = "raw_data"
    model_name: str = "resnet_18"
    model_alias: str = "Production"

