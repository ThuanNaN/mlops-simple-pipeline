
## Start MLflow server
```bash
make mlflow_up
```


## First cycle
```bash
python src/data_processing.py --version v1.0

python src/model_training.py --data_version v1.0 --model_name resnet18

python src/model_registry.py --metric val_loss --alias Production
```

```bash
make serving_up
```

## Second cycle

```bash
python src/data_processing.py --merge_collected --version v1.1

python src/model_training.py --data_version v1.1 --model_name resnet18

python src/model_registry.py --metric val_loss --alias Production
```


```bash
make serving_restart
```

## Shutdown all containers
```bash
make teardown
```
