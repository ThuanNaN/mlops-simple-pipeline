
## 1. Start MLflow server
```bash
make mlflow_up
```

## 2. Training

### 2.1 First round
```bash
python src/data_processing.py --version v1.0

python src/model_training.py --data_version v1.0 --model_name resnet18

python src/model_registry.py --metric val_loss --alias Production
```

### 2.2 Serving trained model
```bash
make serving_up
```

### 2.3 Add more add and re-train model

```bash
python src/data_processing.py --merge_collected --version v1.1

python src/model_training.py --data_version v1.1 --model_name resnet18

python src/model_registry.py --metric val_loss --alias Production
```

### 2.4 Restart to choice the best model for serving
```bash
make serving_restart
```

## 3. Turn on/off the system
```bash
make all_down
```

```bash
make all_up
```