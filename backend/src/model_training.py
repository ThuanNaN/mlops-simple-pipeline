import argparse
import torchvision
from modeling import create_resnet, Trainer
from utils import DataPath, CatDog_Data, Log, seed_everything

logger = Log(__file__).log
logger.info("Starting Model Training")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_version", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    seed_everything(args.seed)

    train_data = torchvision.datasets.ImageFolder(
        root=DataPath.TRAIN_DATA_DIR/args.data_version/"train",
        transform=CatDog_Data.train_transform
    )
    val_data = torchvision.datasets.ImageFolder(
        root=DataPath.TRAIN_DATA_DIR/args.data_version/"val",
        transform=CatDog_Data.test_transform
    )

    test_data = torchvision.datasets.ImageFolder(
        root=DataPath.TRAIN_DATA_DIR/args.data_version/"test",
        transform=CatDog_Data.test_transform
    )

    model = create_resnet(n_classes=CatDog_Data.n_classes, model_name=args.model_name, load_pretrained=False)

    mlflow_log_pamrams = {
        "data_version": args.data_version,
        "model": model.__class__.__name__,
        "model_name": args.model_name,
        "n_epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "lr": args.lr,
        "device": args.device,
        "seed": args.seed
    }
    logger.info(f"Model training params: {mlflow_log_pamrams}")
    
    trainer = Trainer(model=model,
                      num_epochs = args.epochs,
                      learning_rate = args.lr,
                      weight_decay = args.weight_decay,
                      train_data=train_data,
                      val_data=val_data,
                      batch_size=args.batch_size,   
                      device=args.device,
                      mlflow_log_pamrams=mlflow_log_pamrams,
                      verbose=True)

    trainer.train()

