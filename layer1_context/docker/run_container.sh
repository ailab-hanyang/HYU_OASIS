#!/usr/bin/env bash
# Run the HYU_RefAV vLLM container on the H100 server.
# Bind-mounts preserve host absolute paths so relative symlinks inside the
# workspace (data/datasets/sensor, tracker_downloads, scenario_mining_downloads,
# output) resolve correctly.

set -euo pipefail

HOST_HOME=/home/ailab/AILabDataset
IMAGE=${IMAGE:-hyu-refav-vllm:latest}
NAME=${NAME:-layer1_vllm}

docker run --rm -it \
  --gpus all \
  --shm-size=32g \
  --ipc=host \
  -v ${HOST_HOME}/03_Shared_Repository/minwon/HYU_RefAV:${HOST_HOME}/03_Shared_Repository/minwon/HYU_RefAV \
  -v ${HOST_HOME}/01_Open_Dataset/08_Argoverse2:${HOST_HOME}/01_Open_Dataset/08_Argoverse2:ro \
  -v ${HOST_HOME}/03_Shared_Repository/jeongwoo/HYU_RefAV/output:${HOST_HOME}/03_Shared_Repository/jeongwoo/HYU_RefAV/output \
  -v ${HOST_HOME}/03_Shared_Repository/01_CheckPoint:${HOST_HOME}/03_Shared_Repository/01_CheckPoint:ro \
  -w ${HOST_HOME}/03_Shared_Repository/minwon/HYU_RefAV \
  --name ${NAME} \
  ${IMAGE} \
  bash
