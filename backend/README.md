# Install packages

```bash
pip3 install -r requirements.txt
```

## 1. Start MLflow server

```bash
make mlflow_up
```

## 2. Training

### 2.1 First round

Split the raw data into train/val/test folder and tagging to v1.0

```bash
python3 src/data_processing.py --version v1.0
```

Train model "resnet_18" with the data version 1.0. The model will be logged to MLflow.

```bash
python3 src/model_training.py --data_version v1.0 --model_name resnet_18 --device cpu
```

Registry the model trained to MLflow by compared the metric "val_loss", tagging "Production" and save config file in /src/config/raw_data.json

```bash
python3 src/model_registry.py --best_metric best_val_loss --model_alias Production --config_name raw_data
```

### 2.2 Serving trained model

Retrieve model stored in mlflow server from "model_name" and "model_alias" then deploy to API

```bash
make model_name=resnet_18 model_alias=Production port=5000 serving_up
```

### 2.3 Add more data and re-train model

Merge labeled data from /data_source/collected/ with raw_data and split into train/val/test folder. Tagging the version as well as the folder name to v1.1

```bash
python3 src/data_processing.py --merge_collected --version v1.1
```

Train model with new dataset and log to MLflow.

```bash
python3 src/model_training.py --data_version v1.1 --model_name resnet_18 --device cpu
```

Retrieve models are trained on dataset version v1.1 and use the metric to choose the best model. After that, tag alias to "Challenger" and log into MLflow. Save config file in /src/config/add_collect.json

```bash
python3 src/model_registry.py --filter_string "tags.data_version LIKE 'v1.1'" --best_metric best_val_loss --model_alias Challenger  --config_name add_collect
```

### 2.4 Restart to change config of new model for serving

Restart the container to pull model which have "model_config" == "add_collect" for new serving

```bash
make serving_down
make model_name=resnet_18 model_alias=Challenger port=5000 serving_up
```

## 3. Turn on/off the system

Turn on/off both MLflow and serving containers

```bash
make all_down
```

```bash
make all_up
```
