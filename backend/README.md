## Install packages
```bash
pip3 install -r requirements.txt
```

## 1. Start MLflow server
```bash
make mlflow_up
```

## 2. Training

### 2.1 First round
```bash
python src/data_processing.py --version v1.0

python src/model_training.py --data_version v1.0 --model_name resnet_18 --device cuda

python src/model_registry.py --metric val_loss --config_tag raw_data --alias Production
```

### 2.2 Serving trained model
```bash
make model_config=raw_data serving_up
```

### 2.3 Add more data and re-train model
Merge labled data from /data_source/collected/ 
```bash
python src/data_processing.py --merge_collected --version v1.1

python src/model_training.py --data_version v1.1 --model_name resnet_18 --device cuda

python src/model_registry.py --filter_string "tags.Dataset_version LIKE 'v1.1'" --config_tag add_collect --alias Challenger 
```

### 2.4 Restart to choice the best model for serving
```bash
make serving_down
make model_config=add_collect serving_up
```

## 3. Turn on/off the system
```bash
make all_down
```

```bash
make all_up
```
