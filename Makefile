
IMG ?= us-east4-docker.pkg.dev/akurata-offsite/util/tts-server:v0.0.2
.PHONY: build
build:
	docker buildx ls | grep -q "container-builder" || docker buildx create --name container-builder --driver docker-container --bootstrap
	docker buildx use container-builder
	docker buildx build --push --platform linux/arm64 -t $(IMG) .
	docker buildx rm container-builder