define sc2-rl
		docker run -it --rm \
			-u `id -u`:`id -g` \
			--gpus all \
			--network=host \
			-v ${PWD}:${PWD} \
			-w ${PWD} \
			sc2-rl \
			$(1)
endef

image:
	docker build --rm -t raide-rl -f docker/raide-rl/Dockerfile docker/raide-rl
	docker build --rm -t tf23 -f docker/tf23/Dockerfile docker/tf23
	docker build --rm -t sc2-rl -f docker/sc2-rl/Dockerfile docker/sc2-rl

dev:
	$(call sc2-rl,bash)
