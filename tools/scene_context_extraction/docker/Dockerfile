FROM vllm/vllm-openai:latest

ENTRYPOINT []

ARG WORKSPACE=/workspace/HYU_OASIS
WORKDIR ${WORKSPACE}
ENV PYTHONPATH=${WORKSPACE}

RUN ln -sf "$(command -v python3)" /usr/local/bin/python

COPY tools/scene_context_extraction/docker/requirements.docker.txt /tmp/requirements.docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements.docker.txt

CMD ["/bin/bash"]
