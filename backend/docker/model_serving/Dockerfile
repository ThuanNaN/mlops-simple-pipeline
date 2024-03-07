FROM python:3.11-slim

RUN apt-get update

WORKDIR /backend

RUN pip install --upgrade pip

COPY docker/model_serving/requirements.txt /backend/

RUN pip install --no-cache-dir -r /backend/requirements.txt

COPY /app /backend/app
WORKDIR /backend/app