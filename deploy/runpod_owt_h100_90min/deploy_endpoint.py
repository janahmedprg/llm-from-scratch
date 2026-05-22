import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


RUNPOD_REST_URL = "https://rest.runpod.io/v1"
GPU_GROUP_TYPE_IDS = {
    "AMPERE_16": [
        "NVIDIA RTX A4000",
        "NVIDIA RTX 4000 Ada Generation",
        "NVIDIA RTX 2000 Ada Generation",
    ],
    "AMPERE_24": [
        "NVIDIA RTX A4500",
        "NVIDIA RTX A5000",
        "NVIDIA GeForce RTX 3090",
    ],
}
DEFAULT_GPU_GROUP_IDS = ["AMPERE_16", "AMPERE_24"]
DEFAULT_MODEL_ID = "janahmed/owt-h100-90min"


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


def get_json(path: str, api_key: str) -> Any:
    request = urllib.request.Request(
        f"{RUNPOD_REST_URL}{path}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RunPod API returned HTTP {exc.code}: {details}") from exc


def find_template_by_name(name: str, api_key: str) -> dict[str, Any] | None:
    templates = get_json("/templates", api_key)
    if isinstance(templates, dict):
        templates = templates.get("templates") or templates.get("data") or []

    for template in templates:
        if template.get("name") == name:
            return template
    return None


def resolve_gpu_type_ids(gpu_ids: list[str]) -> list[str]:
    gpu_type_ids = []
    for gpu_id in gpu_ids:
        gpu_type_ids.extend(GPU_GROUP_TYPE_IDS.get(gpu_id, [gpu_id]))
    return gpu_type_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a RunPod Serverless template and endpoint for owt_h100_90min.")
    parser.add_argument("--image", required=True, help="Pushed container image, for example docker.io/user/owt-h100:latest.")
    parser.add_argument("--name", default="owt-h100-90min", help="Base name for the template and endpoint.")
    parser.add_argument(
        "--gpu",
        action="append",
        help="GPU group alias or REST GPU type id. Repeat for fallback GPUs. Defaults to AMPERE_16, then AMPERE_24.",
    )
    parser.add_argument("--workers-min", type=int, default=0)
    parser.add_argument("--workers-max", type=int, default=1)
    parser.add_argument("--idle-timeout", type=int, default=300)
    parser.add_argument("--execution-timeout-ms", type=int, default=100_000)
    parser.add_argument("--container-disk-gb", type=int, default=20)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model repo id.")
    parser.add_argument("--reuse-template", action="store_true", help="Use an existing template with the same name.")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        print("Set RUNPOD_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    gpu_type_ids = resolve_gpu_type_ids(args.gpu or DEFAULT_GPU_GROUP_IDS)

    env = {"MODEL_ID": args.model_id}

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
    template = find_template_by_name(template_payload["name"], args.api_key)
    if template is None:
        template = post_json("/templates", template_payload, args.api_key)
    elif args.reuse_template:
        print(f"Using existing template: {template['id']}", file=sys.stderr)
    else:
        print(
            f"Template {template_payload['name']!r} already exists. "
            "Pass --reuse-template to use it, or pass --name to create a fresh template.",
            file=sys.stderr,
        )
        return 2
    template_id = template["id"]

    endpoint_payload = {
        "name": args.name,
        "templateId": template_id,
        "computeType": "GPU",
        "gpuTypeIds": gpu_type_ids,
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
