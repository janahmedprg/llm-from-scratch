import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import runpod
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer.tokenizer import Tokenizer
from src.decoding import decode
from src.models.rope import RotaryPositionalEmbedding
from src.models.transformer_lm import TransformerLM


MODEL_DIR = REPO_ROOT / "experiments" / "owt_h100_90min"
CACHED_MODEL_ID = os.environ.get("MODEL_ID") or os.environ.get("MODEL_NAME") or os.environ.get("HF_MODEL_ID")
HF_CACHE_ROOT = Path(os.environ.get("HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub"))
LOCAL_CONFIG_PATH = MODEL_DIR / "config.json"
LOCAL_CHECKPOINT_PATH = MODEL_DIR / "owt_h100_90min.pt"
LOCAL_VOCAB_PATH = REPO_ROOT / "tokenizer_params" / "owt_vocab.txt"
LOCAL_MERGES_PATH = REPO_ROOT / "tokenizer_params" / "owt_merges.txt"
SPECIAL_TOKENS = ["<|endoftext|>"]
MAX_REQUEST_TOKENS = int(os.environ.get("MAX_REQUEST_TOKENS", "256"))

MODEL: TransformerLM | None = None
TOKENIZER: Tokenizer | None = None
EOS_TOKEN_ID: int | None = None
CONFIG: dict[str, Any] | None = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_cached_model_dir(model_id: str) -> Path | None:
    cache_name = model_id.replace("/", "--")
    snapshots_dir = HF_CACHE_ROOT / f"models--{cache_name}" / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshots = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    return snapshots[-1] if snapshots else None


def _require_paths(paths: tuple[Path, ...]) -> None:
    missing_paths = [str(path) for path in paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Required model artifact(s) missing: "
            + ", ".join(missing_paths)
            + ". Set MODEL_ID to a RunPod cached Hugging Face repo, or provide explicit *_PATH env vars."
        )


def _resolve_model_paths() -> tuple[Path, Path, Path, Path]:
    config_path = Path(os.environ.get("MODEL_CONFIG_PATH", LOCAL_CONFIG_PATH))
    checkpoint_path = Path(os.environ.get("MODEL_CHECKPOINT_PATH", LOCAL_CHECKPOINT_PATH))
    vocab_path = Path(os.environ.get("TOKENIZER_VOCAB_PATH", LOCAL_VOCAB_PATH))
    merges_path = Path(os.environ.get("TOKENIZER_MERGES_PATH", LOCAL_MERGES_PATH))

    if CACHED_MODEL_ID:
        cached_model_dir = _find_cached_model_dir(CACHED_MODEL_ID)
        if cached_model_dir is not None:
            config_path = cached_model_dir / "config.json"
            checkpoint_path = cached_model_dir / "owt_h100_90min.pt"
            vocab_path = cached_model_dir / "owt_vocab.txt"
            merges_path = cached_model_dir / "owt_merges.txt"
            print(f"Using cached Hugging Face model from {cached_model_dir}", flush=True)
        else:
            raise FileNotFoundError(f"Cached Hugging Face model {CACHED_MODEL_ID!r} not found under {HF_CACHE_ROOT}")

    _require_paths((config_path, checkpoint_path, vocab_path, merges_path))
    return config_path, checkpoint_path, vocab_path, merges_path


def _load_model() -> tuple[TransformerLM, Tokenizer, int, dict[str, Any]]:
    global MODEL, TOKENIZER, EOS_TOKEN_ID, CONFIG

    if MODEL is not None and TOKENIZER is not None and EOS_TOKEN_ID is not None and CONFIG is not None:
        return MODEL, TOKENIZER, EOS_TOKEN_ID, CONFIG

    config_path, checkpoint_path, vocab_path, merges_path = _resolve_model_paths()

    CONFIG = _read_json(config_path)
    TOKENIZER = Tokenizer.from_files(vocab_path, merges_path, SPECIAL_TOKENS)
    EOS_TOKEN_ID = TOKENIZER.encode("<|endoftext|>")[0]

    rope = RotaryPositionalEmbedding(
        theta=CONFIG["theta"],
        d_k=CONFIG["d_model"] // CONFIG["num_heads"],
        max_seq_len=CONFIG["context_length"],
        device=DEVICE,
    )

    MODEL = TransformerLM(
        vocab_size=CONFIG["vocab_size"],
        context_length=CONFIG["context_length"],
        num_layers=CONFIG["num_layers"],
        d_model=CONFIG["d_model"],
        num_heads=CONFIG["num_heads"],
        d_ff=CONFIG["d_ff"],
        rope=rope,
        device=DEVICE,
    ).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.eval()

    return MODEL, TOKENIZER, EOS_TOKEN_ID, CONFIG


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def handler(event: dict[str, Any]) -> dict[str, Any]:
    job_input = event.get("input") or {}
    prompt = str(job_input.get("prompt", ""))
    max_new_tokens = _bounded_int(job_input.get("max_new_tokens"), default=80, minimum=1, maximum=MAX_REQUEST_TOKENS)
    temperature = _bounded_float(job_input.get("temperature"), default=0.8, minimum=0.0, maximum=5.0)
    top_p = _bounded_float(job_input.get("top_p"), default=0.95, minimum=0.01, maximum=1.0)
    include_prompt = bool(job_input.get("include_prompt", True))

    seed = job_input.get("seed")
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model, tokenizer, eos_token_id, config = _load_model()
    prompt_tokens = tokenizer.encode(prompt) or [eos_token_id]

    generated_tokens = decode(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        context_length=config["context_length"],
        device=DEVICE,
    )

    generated_text = tokenizer.decode(generated_tokens)
    completion_text = tokenizer.decode(generated_tokens[len(prompt_tokens) :])

    return {
        "prompt": prompt,
        "completion": completion_text,
        "text": generated_text if include_prompt else completion_text,
        "generated_tokens": len(generated_tokens) - len(prompt_tokens),
        "device": DEVICE,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
