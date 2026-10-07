from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from qwen_embed import caption_image, embed_image, embed_text
from qwen_embed._common import (
    as_images,
    as_texts,
    pool_last_token,
    resolve_device,
)


@pytest.fixture
def array() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (32, 48, 3), np.uint8)


def test_text_coercion() -> None:
    assert as_texts("a") == ["a"]
    assert as_texts(("a", "b")) == ["a", "b"]
    with pytest.raises(ValueError):
        as_texts([])


def test_image_coercion(array: np.ndarray, tmp_path: Path) -> None:
    path = tmp_path / "x.png"
    Image.fromarray(array).save(path)
    rgba = Image.fromarray(array).convert("RGBA")
    for single in (array, rgba, path, str(path)):
        (image,) = as_images(single)
        assert image.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(image), array)
    assert len(as_images([array, rgba, path, str(path)])) == 4


@pytest.mark.parametrize(
    "bad", [np.zeros((4, 4, 3), np.float32), np.zeros((4, 4), np.uint8)]
)
def test_rejects_bad_arrays(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="HWC uint8"):
        as_images(bad)


def test_pooling_picks_last_real_token() -> None:
    hidden = torch.randn(2, 4, 8)
    left = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    right = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    pooled_left = pool_last_token(hidden, left)
    pooled_right = pool_last_token(hidden, right)
    assert pooled_left.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(pooled_left, axis=1), 1, 1e-6)
    expected_left = torch.nn.functional.normalize(hidden[:, 3], dim=-1)
    expected_right = torch.nn.functional.normalize(
        hidden[[0, 1], [2, 1]], dim=-1
    )
    np.testing.assert_array_equal(pooled_left, expected_left.numpy())
    np.testing.assert_array_equal(pooled_right, expected_right.numpy())


def test_cpu_is_allowed_explicitly() -> None:
    assert resolve_device("cpu") == "cpu"


def test_missing_cuda_raises(
    monkeypatch: pytest.MonkeyPatch, array: np.ndarray
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for device in (None, "cuda", "cuda:0"):
        with pytest.raises(RuntimeError, match="--gres=gpu"):
            resolve_device(device)
    with pytest.raises(RuntimeError, match="login node"):
        embed_text("a")
    with pytest.raises(RuntimeError, match="login node"):
        embed_image(array)
    with pytest.raises(RuntimeError, match="login node"):
        caption_image(array, "Describe.")
