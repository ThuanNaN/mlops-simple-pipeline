
mlflow_up:
	docker compose -f ./docker/mlflow/docker-compose.yml up -d

mlflow_down:
	docker compose -f ./docker/mlflow/docker-compose.yml down

serving_up:
	MODEL_NAME=${model_name} MODEL_ALIAS=${model_alias} PORT=${port} DEVICE=cpu docker compose -f ./docker/model_serving/docker-compose.yml up -d

serving_down:
	MODEL_NAME="" MODEL_ALIAS="" PORT=5000 DEVICE="" docker compose -f ./docker/model_serving/docker-compose.yml down

all_up:
	make mlflow_up
	make model_name=resnet_18 model_alias=Production port=5000 serving_up

all_down:
	make mlflow_down
	make serving_down

rebuild_serving:
	make serving_down
	docker rmi model_serving-model_serving
	make model_name=resnet_18 model_alias=Production port=5000 serving_up
