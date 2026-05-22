FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY deploy/runpod_owt_h100_90min/requirements.txt /app/deploy/runpod_owt_h100_90min/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /app/deploy/runpod_owt_h100_90min/requirements.txt

COPY src /app/src
COPY experiments/owt_h100_90min/config.json /app/experiments/owt_h100_90min/config.json
COPY deploy/runpod_owt_h100_90min/handler.py /app/deploy/runpod_owt_h100_90min/handler.py

CMD ["python3", "-u", "/app/deploy/runpod_owt_h100_90min/handler.py"]
