import functools
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from qwen_embed._common import (
    ImageInput,
    as_images,
    chunks,
    pool_last_token,
    report_load,
    resolve_device,
    resolve_snapshot,
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_INSTRUCTION = "Represent the user's input."


@functools.cache
def load(model: str, device: str) -> tuple[AutoProcessor, AutoModel]:
    path = resolve_snapshot(model)
    processor = AutoProcessor.from_pretrained(path)
    processor.tokenizer.padding_side = "left"
    net = AutoModel.from_pretrained(path, dtype=torch.float32)
    net = net.to(device).eval()
    report_load(model, path, device, net.dtype)
    return processor, net


def embed_image(
    images: ImageInput | Sequence[ImageInput],
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    batch_size: int = 8,
    device: str | None = None,
    model: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Return (B, D) float32 unit vectors, one per image.

    instruction is the system prompt; the default is the model's own.
    """
    items = as_images(images)
    processor, net = load(model, resolve_device(device))
    return np.concatenate(
        [
            embed_batch(processor, net, batch, instruction)
            for batch in chunks(items, batch_size)
        ]
    )


@torch.inference_mode()
def embed_batch(
    processor: AutoProcessor,
    net: AutoModel,
    images: list[Image.Image],
    instruction: str,
) -> np.ndarray:
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": [{"type": "image"}]},
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=[prompt] * len(images),
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(net.device)
    hidden = net(**inputs).last_hidden_state
    return pool_last_token(hidden, inputs["attention_mask"])
