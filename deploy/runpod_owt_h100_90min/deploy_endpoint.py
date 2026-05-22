import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


RUNPOD_REST_URL = "https://rest.runpod.io/v1"
DEFAULT_GPU_GROUP_IDS = ["AMPERE_16", "AMPERE_24"]


def post_json(path: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{RUNPOD_REST_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RunPod API returned HTTP {exc.code}: {details}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a RunPod Serverless template and endpoint for owt_h100_90min.")
    parser.add_argument("--image", required=True, help="Pushed container image, for example docker.io/user/owt-h100:latest.")
    parser.add_argument("--name", default="owt-h100-90min", help="Base name for the template and endpoint.")
    parser.add_argument(
        "--gpu",
        action="append",
        help="GPU group/type id. Repeat for fallback GPUs. Defaults to AMPERE_16, then AMPERE_24.",
    )
    parser.add_argument("--workers-min", type=int, default=0)
    parser.add_argument("--workers-max", type=int, default=1)
    parser.add_argument("--idle-timeout", type=int, default=10)
    parser.add_argument("--execution-timeout-ms", type=int, default=600_000)
    parser.add_argument("--container-disk-gb", type=int, default=20)
    parser.add_argument("--model-id", help="Hugging Face model repo id, for example USER/owt-h100-90min.")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        print("Set RUNPOD_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    gpu_group_ids = args.gpu or DEFAULT_GPU_GROUP_IDS

    env = {}
    if args.model_id:
        env["MODEL_ID"] = args.model_id

    template_payload = {
        "name": f"{args.name}-template",
        "imageName": args.image,
        "category": "NVIDIA",
        "isServerless": True,
        "isPublic": False,
        "containerDiskInGb": args.container_disk_gb,
        "volumeInGb": 0,
        "volumeMountPath": "/workspace",
        "dockerEntrypoint": [],
        "dockerStartCmd": [],
        "env": env,
        "ports": [],
        "readme": "Serverless worker for the owt_h100_90min TransformerLM checkpoint.",
    }
    template = post_json("/templates", template_payload, args.api_key)
    template_id = template["id"]

    endpoint_payload = {
        "name": args.name,
        "templateId": template_id,
        "computeType": "GPU",
        "gpuTypeIds": gpu_group_ids,
        "gpuCount": 1,
        "workersMin": args.workers_min,
        "workersMax": args.workers_max,
        "idleTimeout": args.idle_timeout,
        "executionTimeoutMs": args.execution_timeout_ms,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "allowedCudaVersions": ["12.8", "13.0"],
    }
    endpoint = post_json("/endpoints", endpoint_payload, args.api_key)

    print(json.dumps({"template": template, "endpoint": endpoint}, indent=2))
    print(f"\nEndpoint API URL: https://api.runpod.ai/v2/{endpoint['id']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
