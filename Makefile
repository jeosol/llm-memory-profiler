# test cuda
# python -c "import torch; print('Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

NAME=llm-memory-profiler

venv:
	uv venv --python 3.12.0 ~/.venv/${NAME}

source-env:
	echo "source ~/.venv/${NAME}/bin/activate"

install:
	uv pip install -r requirements.txt

deactivate:
	deactivate

HASHLEN=9
COMMIT_ID=$(shell git --work-tree=./ --git-dir=./.git rev-parse --short=${HASHLEN} HEAD)

appversion:
	echo ${COMMIT_ID}

run-estimator:
	python ./src/memory_profiler/estimator.py

run-smoke-test:
	python ./src/test_smoke.py
#
# docker business
# 
build:
	docker build -t ${NAME_}:${COMMIT_ID} .

build-image: build

run:
	docker run \
	--network=host \
	--env RABBITMQ_DEFAULT_USER=${RABBITMQ_DEFAULT_USER} \
	--env RABBITMQ_DEFAULT_PASS=${RABBITMQ_DEFAULT_PASS} \
	--env RABBITMQ_DEFAULT_HOSTNAME=${RABBITMQ_DEFAULT_HOSTNAME} \
	--rm -it --name ${NAME_} -p 80:80 ${NAME_}:${COMMIT_ID}
cbuild:
	docker-compose build
cup:
	docker-compose up
cdown:
	docker-compose down
log:
	docker logs ${NAME} -f

# on a browser http://192.168.99.100/items/5?q=somequery or http://127.0.0.1/items/5?q=somequery

#run-tests:
	curl http://localhost
	curl http://127.0.0.1/items/5?q=somequery

REGISTRY_NAME=tdba
IMAGE_HOST_REGISTRY=titan:8080/aiml
tag-image:
	docker tag docker.io/library/${NAME}:${COMMIT_ID} ${IMAGE_HOST_REGISTRY}/${NAME_}:${COMMIT_ID}

push-image:
	@echo "Pushing image ${NAME_}:${COMMIT_ID} to digital ocean registry ..."
	docker push ${IMAGE_HOST_REGISTRY}/${NAME_}:${COMMIT_ID}

build-tag-push-image:
	make build-image
	make tag-image
	make push-image

#interactive-api-docs:
#Now you can go to http://192.168.99.100/docs or http://127.0.0.1/docs (or equivalent, using your Docker host).

#Alternative API docs¶

#And you can also go to http://192.168.99.100/redoc or http://127.0.0.1/redoc (or equivalent, using your Docker host).
