import argparse
import torchvision
from modeling import Trainer, create_mobilenet, create_resnet
from utils import DataPath, CatDog_Data, Log, seed_everything

logger = Log(__file__).get_logger()
logger.info("Starting Model Training")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_version", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resnet_18",
                        choices=["resnet_18", "resnet_34", "mobilenet_v2", "mobilenet_v3_small"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--best_model_metric", type=str, default="val_loss", 
                        choices=["val_loss", "val_acc"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    seed_everything(args.seed)

    try:
        data_path = DataPath.TRAIN_DATA_DIR/args.data_version
        assert data_path.exists()
    except AssertionError:
        logger.error(f"Data version: {args.data_version} not found.")
        raise FileNotFoundError(f"Data version: {args.data_version} not found.")

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

    model_prefix = args.model_name.split("_")[0]
    if model_prefix == "resnet":
        model = create_resnet(n_classes=CatDog_Data.n_classes, model_name=args.model_name)
    elif model_prefix == "mobilenet":
        model = create_mobilenet(n_classes=CatDog_Data.n_classes, model_name=args.model_name)
    
    mlflow_log_pamrams = {
        "data_version": args.data_version,
        "model": model.__class__.__name__,
        "model_name": args.model_name,
        "n_epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "lr": args.lr,
        "best_model_metric": args.best_model_metric,
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
                      best_model_metric=args.best_model_metric,
                      device=args.device,
                      mlflow_log_pamrams=mlflow_log_pamrams,
                      verbose=True)

    trainer.train()

    logger.info(f"Model Training Completed. Model: {args.model_name}, Data: {args.data_version}")


