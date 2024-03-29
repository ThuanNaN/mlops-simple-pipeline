import argparse
import torchvision
from modeling import Trainer, create_mobilenet, create_resnet
from utils import AppPath, Logger, seed_everything
from config.data_config import CatDog_Data


LOGGER = Logger(__file__)
LOGGER.log.info("Starting Model Training")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_version", type=str, required=True, 
                        help="Version/directory to be used for training")
    parser.add_argument("--model_name", type=str, default="resnet_18",
                        choices=["resnet_18", "resnet_34", "mobilenet_v2", "mobilenet_v3_small"],
                        help="Model to be used for training")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate for optimizer")
    parser.add_argument("--best_model_metric", type=str, default="val_loss", 
                        choices=["val_loss", "val_acc"],
                        help="Metric for selecting the best model to logging to MLflow")
    parser.add_argument("--device", type=str, default="cpu", 
                        choices=["cuda", "cpu"],
                        help="Device to be used for training")
    parser.add_argument("--seed", type=int, default=43, 
                        help="Seed for reproducibility")
    args = parser.parse_args()
    seed_everything(args.seed)

    try:
        data_path = AppPath.TRAIN_DATA_DIR/args.data_version
        assert data_path.exists()
    except AssertionError:
        LOGGER.log.error(f"Data version: {args.data_version} not found.")
        raise FileNotFoundError(f"Data version: {args.data_version} not found.")

    train_data = torchvision.datasets.ImageFolder(
        root=AppPath.TRAIN_DATA_DIR/args.data_version/"train",
        transform=CatDog_Data.train_transform
    )
    val_data = torchvision.datasets.ImageFolder(
        root=AppPath.TRAIN_DATA_DIR/args.data_version/"val",
        transform=CatDog_Data.test_transform
    )

    test_data = torchvision.datasets.ImageFolder(
        root=AppPath.TRAIN_DATA_DIR/args.data_version/"test",
        transform=CatDog_Data.test_transform
    )

    model_prefix = args.model_name.split("_")[0]
    if model_prefix == "resnet":
        model = create_resnet(n_classes=CatDog_Data.n_classes, model_name=args.model_name)
    elif model_prefix == "mobilenet":
        model = create_mobilenet(n_classes=CatDog_Data.n_classes, model_name=args.model_name)
    
    mlflow_log_tags = {
        "data_version": args.data_version,
        "id2label": CatDog_Data.id2label,
        "label2id": CatDog_Data.label2id,
    }
    LOGGER.log.info(f"Model training tasg: {mlflow_log_tags}")

    mlflow_log_pamrams = {
        "model": model.__class__.__name__,
        "model_name": args.model_name,
        "n_epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "lr": args.lr,
        "best_model_metric": args.best_model_metric,
        "device": args.device,
        "seed": args.seed,
        "n_classes": CatDog_Data.n_classes,
        "image_size": CatDog_Data.img_size,
        "image_mean": CatDog_Data.mean,
        "image_std": CatDog_Data.std,
    }
    LOGGER.log.info(f"Model training params: {mlflow_log_pamrams}")
    
    trainer = Trainer(model=model,
                      num_epochs = args.epochs,
                      learning_rate = args.lr,
                      weight_decay = args.weight_decay,
                      train_data=train_data,
                      val_data=val_data,
                      batch_size=args.batch_size,   
                      best_model_metric=args.best_model_metric,
                      device=args.device,
                      mlflow_log_tags=mlflow_log_tags,
                      mlflow_log_pamrams=mlflow_log_pamrams,
                      verbose=True)

    trainer.train()
    LOGGER.log.info(f"Model Training Completed. Model: {args.model_name}, Data: {args.data_version}")


