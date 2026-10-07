import os
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image

T = TypeVar("T")
ImageInput = Image.Image | np.ndarray | str | Path
NO_CUDA_MESSAGE = (
    "CUDA is not available. You are probably on the login node, or the "
    "job was submitted without --gres=gpu:1. Run inside `srun --gres=gpu:1` "
    "or an sbatch job. device='cpu' is only meant for tests."
)


def resolve_device(device: str | None) -> str:
    if device == "cpu":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(NO_CUDA_MESSAGE)
    if device in (None, "cuda"):
        return "cuda:0"
    return device


def as_texts(texts: str | Sequence[str]) -> list[str]:
    items = [texts] if isinstance(texts, str) else list(texts)
    if not items:
        raise ValueError("expected at least one text")
    return items


def as_images(images: ImageInput | Sequence[ImageInput]) -> list[Image.Image]:
    if isinstance(images, (Image.Image, np.ndarray, str, Path)):
        images = [images]
    items = [to_pil(item) for item in images]
    if not items:
        raise ValueError("expected at least one image")
    return items


def to_pil(item: ImageInput) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, np.ndarray):
        if item.dtype != np.uint8 or item.ndim != 3 or item.shape[2] != 3:
            raise ValueError(
                "arrays must be HWC uint8 with 3 channels, got "
                f"{item.dtype} {item.shape}"
            )
        return Image.fromarray(item)
    if isinstance(item, (str, Path)):
        with Image.open(item) as opened:
            return opened.convert("RGB")
    raise TypeError(f"unsupported image type: {type(item).__name__}")


def chunks(items: list[T], size: int) -> list[list[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def resolve_snapshot(model: str) -> str:
    """Disable TF32 so the forward is true fp32, then find the weights."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hub_dir = os.path.join(os.environ["HF_HOME"], "hub")
    return snapshot_download(model, cache_dir=hub_dir)


def report_load(model: str, path: str, device: str, dtype: object) -> None:
    print(
        f"[qwen_embed] loaded {model} snapshot={path} "
        f"device={device} dtype={dtype}",
        flush=True,
    )


def pool_last_token(
    hidden: torch.Tensor, attention_mask: torch.Tensor
) -> np.ndarray:
    """L2-normalized hidden state of the last real token, any padding."""
    last = attention_mask.shape[1] - 1 - attention_mask.flip(1).argmax(1)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    pooled = torch.nn.functional.normalize(hidden[rows, last], dim=-1)
    return pooled.float().cpu().numpy()
