FROM vllm/vllm-openai:latest

ENTRYPOINT []

ARG WORKSPACE=/home/ailab/AILabDataset/03_Shared_Repository/minwon/HYU_RefAV
WORKDIR ${WORKSPACE}
ENV PYTHONPATH=${WORKSPACE}

RUN ln -sf "$(command -v python3)" /usr/local/bin/python

COPY layer1_context/docker/requirements.docker.txt /tmp/requirements.docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements.docker.txt

CMD ["/bin/bash"]
