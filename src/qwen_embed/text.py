import functools
from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from qwen_embed._common import (
    as_texts,
    chunks,
    pool_last_token,
    report_load,
    resolve_device,
    resolve_snapshot,
)

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
MAX_LENGTH = 8192


@functools.cache
def load(model: str, device: str) -> tuple[AutoTokenizer, AutoModel]:
    path = resolve_snapshot(model)
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    net = AutoModel.from_pretrained(path, dtype=torch.float32)
    net = net.to(device).eval()
    report_load(model, path, device, net.dtype)
    return tokenizer, net


def embed_text(
    texts: str | Sequence[str],
    *,
    instruction: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
    model: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Return (B, D) float32 unit vectors, one per text.

    instruction=None embeds the raw text (document side). Otherwise
    every text becomes "Instruct: {instruction}\\nQuery:{text}".
    """
    items = as_texts(texts)
    if instruction is not None:
        items = [f"Instruct: {instruction}\nQuery:{t}" for t in items]
    tokenizer, net = load(model, resolve_device(device))
    return np.concatenate(
        [embed_batch(tokenizer, net, b) for b in chunks(items, batch_size)]
    )


@torch.inference_mode()
def embed_batch(
    tokenizer: AutoTokenizer, net: AutoModel, texts: list[str]
) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(net.device)
    hidden = net(**inputs).last_hidden_state
    return pool_last_token(hidden, inputs["attention_mask"])
