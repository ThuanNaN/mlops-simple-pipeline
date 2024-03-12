from dataclasses import dataclass
import torchvision

@dataclass
class CatDog_Data:
    n_classes = 2
    img_size = 224
    classes = ['cat', 'dog']
    id2label = {0: 'Cat', 1: 'Dog'}
    label2id = {'Cat': 0, 'Dog': 1}
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