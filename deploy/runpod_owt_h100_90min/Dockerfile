FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/runpod_owt_h100_90min/requirements.txt /app/deploy/runpod_owt_h100_90min/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /app/deploy/runpod_owt_h100_90min/requirements.txt

COPY src /app/src
COPY tokenizer_params/owt_vocab.txt /app/tokenizer_params/owt_vocab.txt
COPY tokenizer_params/owt_merges.txt /app/tokenizer_params/owt_merges.txt
COPY experiments/owt_h100_90min/config.json /app/experiments/owt_h100_90min/config.json
COPY experiments/owt_h100_90min/owt_h100_90min.pt /app/experiments/owt_h100_90min/owt_h100_90min.pt
COPY deploy/runpod_owt_h100_90min/handler.py /app/deploy/runpod_owt_h100_90min/handler.py

CMD ["python3", "-u", "/app/deploy/runpod_owt_h100_90min/handler.py"]
