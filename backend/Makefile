
mlflow_up:
	docker compose -f ./docker/mlflow/docker-compose.yml up -d

mlflow_down:
	docker compose -f ./docker/mlflow/docker-compose.yml down

serving_up:
	docker compose -f ./docker/model_serving/docker-compose.yml up -d

serving_down:
	docker compose -f ./docker/model_serving/docker-compose.yml down

serving_restart:
	docker compose -f ./docker/model_serving/docker-compose.yml restart

teardown:
	make mlflow_down
	make serving_down

rebuild_serving:
	make serving_down
	docker rmi model_serving-model_serving
	make serving_up 