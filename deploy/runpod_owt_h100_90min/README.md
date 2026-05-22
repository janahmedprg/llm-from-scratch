# RunPod Serverless Docker Deployment: owt_h100_90min

This folder packages `experiments/owt_h100_90min/owt_h100_90min.pt` as a RunPod Serverless Docker worker.

## Files

- `Dockerfile`: builds the CUDA 12.8/PyTorch worker image.
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
  -t docker.io/janahmed/owt-h100-90min:latest .

docker push docker.io/janahmed/owt-h100-90min:latest
```

The checkpoint and tokenizer files are loaded from RunPod's Hugging Face model cache when `MODEL_ID` is set, so the image does not bake in `owt_h100_90min.pt` or `tokenizer_params`.

## Push Model to Hugging Face

Create a Hugging Face model repo, then upload the checkpoint, config, and tokenizer files:

```bash
hf repo create janahmed/owt-h100-90min --type model --private

hf upload janahmed/owt-h100-90min \
  experiments/owt_h100_90min/owt_h100_90min.pt \
  owt_h100_90min.pt

hf upload janahmed/owt-h100-90min \
  experiments/owt_h100_90min/config.json \
  config.json

hf upload janahmed/owt-h100-90min \
  tokenizer_params/owt_vocab.txt \
  owt_vocab.txt

hf upload janahmed/owt-h100-90min \
  tokenizer_params/owt_merges.txt \
  owt_merges.txt
```

In the RunPod endpoint settings, set the cached model field to `janahmed/owt-h100-90min`. For private repos, also provide a Hugging Face access token in RunPod.

## Create the Serverless Endpoint

```bash
export RUNPOD_API_KEY=...
python3 deploy/runpod_owt_h100_90min/deploy_endpoint.py \
  --image docker.io/janahmed/owt-h100-90min:latest
```

The helper defaults `MODEL_ID` to `janahmed/owt-h100-90min`. Override it with `--model-id` if you publish a different model repo.
If a template with the same name already exists, use a fresh name so the new image and `MODEL_ID` environment are applied:

```bash
python3 deploy/runpod_owt_h100_90min/deploy_endpoint.py \
  --image docker.io/janahmed/owt-h100-90min:latest \
  --name owt-h100-90min-v2
```

The helper creates a private template and a GPU Serverless endpoint, with zero warm workers and a 300 second idle timeout. By default, it accepts the local aliases `AMPERE_16` and `AMPERE_24`, then expands them to the concrete GPU type ids required by RunPod's REST API.

Default GPU fallback:

- `GpuGroup.AMPERE_16`: NVIDIA RTX A4000, NVIDIA RTX 4000 Ada, NVIDIA RTX 2000 Ada
- `GpuGroup.AMPERE_24`: NVIDIA RTX A4500, NVIDIA RTX A5000, NVIDIA GeForce RTX 3090

```bash
python3 deploy/runpod_owt_h100_90min/deploy_endpoint.py \
  --image docker.io/janahmed/owt-h100-90min:latest \
  --gpu AMPERE_16 \
  --gpu AMPERE_24
```

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
