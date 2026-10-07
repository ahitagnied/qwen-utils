import functools
from collections.abc import Sequence

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen_embed._common import (
    ImageInput,
    as_images,
    chunks,
    report_load,
    resolve_device,
    resolve_snapshot,
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


@functools.cache
def load(
    model: str, device: str
) -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    path = resolve_snapshot(model)
    processor = AutoProcessor.from_pretrained(path)
    processor.tokenizer.padding_side = "left"
    net = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.float32
    )
    net = net.to(device).eval()
    report_load(model, path, device, net.dtype)
    return processor, net


def caption_image(
    images: ImageInput | Sequence[ImageInput],
    prompt: str,
    *,
    max_new_tokens: int = 128,
    batch_size: int = 8,
    device: str | None = None,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """Greedily answer prompt about each image; one string per image."""
    items = as_images(images)
    processor, net = load(model, resolve_device(device))
    return [
        caption
        for batch in chunks(items, batch_size)
        for caption in caption_batch(
            processor, net, batch, prompt, max_new_tokens
        )
    ]


@torch.inference_mode()
def caption_batch(
    processor: AutoProcessor,
    net: AutoModelForImageTextToText,
    images: list[Image.Image],
    prompt: str,
    max_new_tokens: int,
) -> list[str]:
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=[text] * len(images),
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(net.device)
    output = net.generate(
        **inputs,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=max_new_tokens,
    )
    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    captions = processor.batch_decode(new_tokens, skip_special_tokens=True)
    return [caption.strip() for caption in captions]
