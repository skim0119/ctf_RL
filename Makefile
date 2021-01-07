
define sc2
		docker run -it --rm \
			-u `id -u`:`id -g` \
			--gpus all \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			--network=host \
			-v ${PWD}:${PWD} \
			-w ${PWD} \
			-v /etc/passwd:/etc/passwd \
			-v /dev:/dev \
			-e HOME=${HOME} \
			-v ~/StarCraftII:${PWD}/StarCraftII \
			tf2-sc2 \
			$(1)
endef


#Can't be the same name as a directory
image:
	docker build --rm -t tf2-sc2 -f docker/tf2-sc2/Dockerfile docker/tf2-sc2

sc2:
	xhost +local:docker
	$(call sc2,bash)

tensorboard:
	docker run -it --rm \
		--memory="2G" \
		--network=host \
		-v ${PWD}:${PWD} \
		-w ${PWD} \
		tf2-sc2 \
		tensorboard --logdir=logs

clean:
	rm -r images/ logs/ models/

.PHONY: install build
