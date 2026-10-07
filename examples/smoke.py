import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from qwen_embed import (
    caption,
    caption_image,
    embed_image,
    embed_text,
    image,
    text,
)

CALLS = 50


def scene(color: str, box: tuple[int, int, int, int]) -> Image.Image:
    canvas = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(canvas).rectangle(box, fill=color)
    return canvas


images = [
    scene("red", (40, 40, 120, 120)),
    np.asarray(scene("blue", (130, 100, 220, 200))),
]
texts = ["a red block in the top left", "a blue block on the right"]
prompt = "This image shows a goal state. Describe it in one sentence."

print("device:", torch.cuda.get_device_name(0))
start = time.perf_counter()
for _ in range(CALLS):
    text_vectors = embed_text(texts)
    image_vectors = embed_image(images)
captions = caption_image(images, prompt)
for _ in range(CALLS - 1):
    assert caption_image(images, prompt, max_new_tokens=8) is not None
print(f"{3 * CALLS} calls in {time.perf_counter() - start:.1f}s")

print(
    "embed_text  ",
    text_vectors.shape,
    text_vectors.dtype,
    np.linalg.norm(text_vectors, axis=1),
)
print(
    "embed_image ",
    image_vectors.shape,
    image_vectors.dtype,
    np.linalg.norm(image_vectors, axis=1),
)
print("caption_image", len(captions), captions)
print(
    "model loads:",
    {
        module.__name__: module.load.cache_info().misses
        for module in (text, image, caption)
    },
)
