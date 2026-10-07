"""Frozen Qwen text/image embeddings and image captions.

Every function returns a batch, even for a single non-list input.
"""

import os

os.environ.setdefault("HF_HOME", "/scratch/ad158/qwen")

import gc  # noqa: E402

import torch  # noqa: E402

from qwen_embed import caption, image, text  # noqa: E402
from qwen_embed.caption import caption_image  # noqa: E402
from qwen_embed.image import embed_image  # noqa: E402
from qwen_embed.text import embed_text  # noqa: E402

__all__ = ["caption_image", "embed_image", "embed_text", "unload"]


def unload() -> None:
    """Drop every cached model and free its GPU memory."""
    for module in (text, image, caption):
        module.load.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()
