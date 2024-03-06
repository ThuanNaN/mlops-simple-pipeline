
mlflow_up:
	docker compose -f ./docker/mlflow/docker-compose.yml up -d

mlflow_down:
	docker compose -f ./docker/mlflow/docker-compose.yml down

serving_up:
	MODEL_CONFIG=$(model_config) PORT=$(port) docker compose -f ./docker/model_serving/docker-compose.yml up -d

serving_down:
	MODEL_CONFIG="" PORT="" docker compose -f ./docker/model_serving/docker-compose.yml down

serving_restart:
	make serving_down
	make model_config=raw_data port=6000 serving_up

all_up:
	make mlflow_up
	make model_config=raw_data port=6000 serving_up

all_down:
	make mlflow_down
	make serving_down

rebuild_serving:
	make serving_down
	docker rmi model_serving-model_serving
	make model_config=raw_data port=6000 serving_up
