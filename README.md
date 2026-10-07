# qwen-utils

```python
from qwen_embed import embed_text, embed_image, caption_image, unload
```

| function | model | output |
| --- | --- | --- |
| `embed_text` | Qwen/Qwen3-Embedding-0.6B | `(B, 1024)` float32, unit norm |
| `embed_image` | Qwen/Qwen3-VL-Embedding-2B | `(B, 2048)` float32, unit norm |
| `caption_image` | Qwen/Qwen3-VL-2B-Instruct | `list[str]`, greedy |

Text and image vectors come from different models, so don't compare them.

## Examples

```python
v = embed_text(["a red block in the top left", "a blue block on the right"])
# (2, 1024)

v = embed_text("one caption")
# (1, 1024), single input still returns a batch

v = embed_text(texts, instruction="Find the goal state")
# each text becomes "Instruct: ...\nQuery:<text>"
```

```python
img = Image.open("frames/000.png")
arr = np.zeros((256, 256, 3), np.uint8)  # HWC uint8

v = embed_image([img, arr, "frames/001.png"])
# (3, 2048)
```

```python
c = caption_image([img], "This image shows a goal state. Describe it.")
# ['A red square is positioned at the top-left corner of a white background.']

c = caption_image(img, "Describe it.", max_new_tokens=32)
# list of 1
```

All three take `batch_size`, `device`, `model`. `unload()` frees the GPU.

## Setup

```bash
export HF_HOME=/scratch/ad158/qwen
export UV_PROJECT_ENVIRONMENT=/scratch/ad158/qwen/venv
export UV_CACHE_DIR=/scratch/ad158/uv-cache
uv sync
```

Weights live in `$HF_HOME/hub`. Set `HF_HUB_OFFLINE=1` on compute nodes.

## Running

Batch job, with all calls in one script so models load once:

```bash
sbatch slurm/run.sbatch my_script.py
```

Interactive:

```bash
srun -p commons --gres=gpu:1 --mem=64G --pty bash
uv run python
```

From another project: `uv pip install -e /path/to/qwen-utils`.

Notes:
- Needs a GPU. On the login node it raises an error.
- Models load on first call and stay cached for the process.
- Don't load these inside DDP training. Precompute vectors, save with
  `np.save`, and load the arrays in training.
- All three models take about 20 GB together. On a 24 GB GPU, call
  `unload()` between steps.

## Tests

```bash
uv run pytest -m "not gpu"    # no GPU needed
uv run pytest                 # on a GPU node
```
