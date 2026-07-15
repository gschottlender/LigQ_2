.PHONY: start build stop status logs pull init-data cli

start:
	./docker/ligq.sh start

build:
	./docker/ligq.sh build

stop:
	./docker/ligq.sh stop

status:
	./docker/ligq.sh status

logs:
	./docker/ligq.sh logs

pull:
	./docker/ligq.sh pull

init-data:
	./docker/ligq.sh init-data

cli:
	./docker/ligq.sh cli $(ARGS)
