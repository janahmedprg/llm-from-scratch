# RunPod Serverless Docker Deployment: owt_h100_90min

This folder packages `experiments/owt_h100_90min/owt_h100_90min.pt` as a RunPod Serverless Docker worker.

## Files

- `Dockerfile`: builds the CUDA/PyTorch worker image.
- `handler.py`: RunPod Serverless handler that loads the model and serves inference.
- `requirements.txt`: Python dependencies used inside the image.
- `Dockerfile.dockerignore`: keeps caches and training data out of the Docker build context.
- `deploy_endpoint.py`: optional helper to create a RunPod template and endpoint after you push the image.
- `test_input.json`: local RunPod SDK test payload.

## Build and Push

Build from the repo root so the Dockerfile can copy `src`, tokenizer files, config, and the checkpoint:

```bash
docker build --platform linux/amd64 \
  -f deploy/runpod_owt_h100_90min/Dockerfile \
  -t docker.io/YOUR_USER/owt-h100-90min:latest .

docker push docker.io/YOUR_USER/owt-h100-90min:latest
```

The checkpoint is baked into the image, so the image will be large.

## Create the Serverless Endpoint

```bash
export RUNPOD_API_KEY=...
python3 deploy/runpod_owt_h100_90min/deploy_endpoint.py \
  --image docker.io/YOUR_USER/owt-h100-90min:latest
```

The helper creates a private template and a GPU Serverless endpoint. It defaults to one RTX 4090 worker, with zero warm workers.

## Request Shape

```json
{
  "input": {
    "prompt": "Once upon a time",
    "max_new_tokens": 80,
    "temperature": 0.8,
    "top_p": 0.95,
    "seed": 42
  }
}
```

RunPod returns:

```json
{
  "output": {
    "prompt": "...",
    "completion": "...",
    "text": "...",
    "generated_tokens": 80,
    "device": "cuda"
  }
}
```

## Local Handler Test

Install the worker dependencies into an environment with PyTorch, then run:

```bash
cd deploy/runpod_owt_h100_90min
PYTHONPATH=../.. python3 handler.py
```

The RunPod SDK will use `test_input.json` from this directory.
