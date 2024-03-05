from torchvision.models import mobilenet_v2, mobilenet_v3_small
import torch.nn as nn

def create_mobilenet(n_classes: int = 2, model_name = "mobilenet_v2", load_pretrained: bool = False):
    if model_name == "mobilenet_v2":
        if load_pretrained:
            model = mobilenet_v2(weights=mobilenet_v2.MobileNetV2_Weights.IMAGENET1K_V1)
        else:
            model = mobilenet_v2()
    elif model_name == "mobilenet_v3_small":
        if load_pretrained:
            model = mobilenet_v3_small(weights=mobilenet_v3_small.MobileNetV3_Small_Weights.IMAGENET1K)
        else:
            model = mobilenet_v3_small()
    else:
        raise ValueError("Invalid model name. [mobilenet_v2, mobilenet_v3_small]")
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, n_classes)
    return model

