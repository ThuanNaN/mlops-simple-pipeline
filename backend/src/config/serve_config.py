from dataclasses import dataclass

@dataclass
class ServeConfig:
    model_name: str = "resnet18"
    model_alias: str = "Production"