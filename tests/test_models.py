import numpy as np
import pytest
import torch
from PIL import Image

from qwen_embed import caption, caption_image, embed_image, embed_text, image
from qwen_embed import text as text_module

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs a GPU and weights"
    ),
]

TEXTS = [f"a red block {i} cm left of the {'bowl ' * i}" for i in range(10)]
PROMPT = "Describe this image in one sentence."


@pytest.fixture(scope="module")
def images() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    sizes = [(64, 64), (96, 128), (128, 96), (224, 224), (80, 160)]
    arrays = [rng.integers(0, 256, (*s, 3), np.uint8) for s in sizes]
    return arrays + arrays[:4]


def assert_unit_rows(vectors: np.ndarray, count: int) -> None:
    assert vectors.dtype == np.float32
    assert vectors.ndim == 2 and vectors.shape[0] == count
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, 1e-5)


def test_embed_text() -> None:
    vectors = embed_text(TEXTS)
    assert_unit_rows(vectors, len(TEXTS))
    assert embed_text("one").shape == (1, vectors.shape[1])
    assert embed_text(TEXTS).tobytes() == vectors.tobytes()
    np.testing.assert_allclose(
        embed_text(TEXTS, batch_size=1),
        embed_text(TEXTS, batch_size=8),
        atol=1e-5,
    )


def test_embed_text_instruction_changes_vectors() -> None:
    plain = embed_text(TEXTS[:2])
    instructed = embed_text(TEXTS[:2], instruction="Find the goal state")
    assert not np.allclose(plain, instructed)


def test_embed_image(images: list[np.ndarray], tmp_path) -> None:
    vectors = embed_image(images)
    assert_unit_rows(vectors, len(images))
    path = tmp_path / "x.png"
    Image.fromarray(images[0]).save(path)
    for single in (images[0], Image.fromarray(images[0]), path, str(path)):
        assert embed_image(single, batch_size=1).tobytes() == (
            embed_image(images[0], batch_size=1).tobytes()
        )
    assert embed_image(images).tobytes() == vectors.tobytes()
    np.testing.assert_allclose(
        embed_image(images, batch_size=1),
        embed_image(images, batch_size=8),
        atol=1e-5,
    )


def test_caption_image(images: list[np.ndarray]) -> None:
    captions = caption_image(images[:3], PROMPT, max_new_tokens=24)
    assert len(captions) == 3
    assert all(isinstance(c, str) and c for c in captions)
    assert caption_image(images[0], PROMPT, max_new_tokens=24) == captions[:1]
    assert caption_image(images[:3], PROMPT, max_new_tokens=24) == captions
    assert (
        caption_image(images[:3], PROMPT, max_new_tokens=24, batch_size=1)
        == captions
    )


@pytest.mark.parametrize("module", [text_module, image, caption])
def test_models_load_once(module, images: list[np.ndarray]) -> None:
    call = {
        text_module: lambda: embed_text(TEXTS[:2]),
        image: lambda: embed_image(images[:2]),
        caption: lambda: caption_image(images[:1], PROMPT, max_new_tokens=4),
    }[module]
    call()
    misses = module.load.cache_info().misses
    for _ in range(3):
        call()
    assert module.load.cache_info().misses == misses
    assert module.load.cache_info().currsize == 1
