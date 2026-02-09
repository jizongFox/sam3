# Docker Usage

This repo includes a `Dockerfile` that builds a runnable image with a small CLI entrypoint for **single-image segmentation with a text prompt**.

The container entrypoint is:

```bash
python -m sam3.cli.segment_image
```

It accepts:

```bash
sam3-cli <image_path> <prompt> <save_path>
```

## Prereqs

- Docker installed
- For GPU usage: NVIDIA driver + NVIDIA Container Toolkit (`docker run --gpus all` must work)
- Access to the gated Hugging Face model repo: `facebook/sam3`

The provided image uses CUDA 12.8 (`pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`).

## Build

From the repo root:

```bash
docker build -t sam3-cli .
```

## Run

Mount a host folder to `/data` for input/output files.

GPU run (recommended):

```bash
docker run --rm --gpus all \
  -v "$PWD":/data \
  sam3-cli \
  /data/input.jpg "a dog" /data/output.png
```

CPU run (slow):

```bash
docker run --rm \
  -v "$PWD":/data \
  sam3-cli \
  /data/input.jpg "a dog" /data/output.png --device cpu
```

## Output Formats

By default the CLI saves a **cutout** image.

- `--mode cutout` (default): saves an RGBA image where alpha is the union of predicted masks (use `.png`)
- `--mode mask`: saves a binary mask image (`L` / grayscale)
- `--mode overlay`: saves an RGB image with a red overlay on the predicted region (`--alpha` controls opacity)

Example:

```bash
docker run --rm --gpus all -v "$PWD":/data sam3-cli \
  /data/input.jpg "a dog" /data/mask.png --mode mask
```

## Model Downloads, HF Tokens, and “Put the Model on Disk”

SAM3 checkpoints are downloaded from Hugging Face via `huggingface_hub` when the model is first constructed.

Why a token is sometimes needed:
- The checkpoint repo `facebook/sam3` is gated; Hugging Face requires authentication to download it.

How to avoid a token at runtime:
- Download once, then reuse the on-disk cache.

### Recommended: persist the Hugging Face cache

The image sets `HF_HOME=/cache/huggingface`. If you mount a host cache directory there, downloads persist across runs.

First run (uses token to populate the cache):

```bash
docker run --rm --gpus all \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -v "$HOME/.cache/huggingface":/cache/huggingface \
  -v "$PWD":/data \
  sam3-cli \
  /data/input.jpg "a dog" /data/output.png
```

Later runs (no token; uses cached files):

```bash
docker run --rm --gpus all \
  -e HF_HUB_OFFLINE=1 \
  -v "$HOME/.cache/huggingface":/cache/huggingface \
  -v "$PWD":/data \
  sam3-cli \
  /data/input.jpg "a dog" /data/output.png
```

### Optional: prefetch checkpoints without running inference

You can prefetch the checkpoint files into the mounted cache by overriding the entrypoint:

```bash
docker run --rm \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -v "$HOME/.cache/huggingface":/cache/huggingface \
  --entrypoint python \
  sam3-cli \
  -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam3', 'config.json'); hf_hub_download('facebook/sam3', 'sam3.pt'); print('done')"
```

## CLI Reference

Print help:

```bash
docker run --rm sam3-cli --help
```

Current flags:
- `--device {auto,cuda,cpu}`
- `--confidence-threshold <float>`
- `--mode {cutout,overlay,mask}`
- `--alpha <float>` (overlay only)
- `--compile` (enables `torch.compile`)

## Troubleshooting

- `You need access to the Hugging Face model repo` / 401 / 403 errors
  - Request access to `facebook/sam3` on Hugging Face, then use a token once to populate the cache.
- `docker run --gpus all` fails
  - Install/configure the NVIDIA Container Toolkit and ensure your NVIDIA driver is working on the host.
- Output is fully transparent / empty
  - Try lowering `--confidence-threshold` (e.g. `0.3`) or adjust the prompt.
