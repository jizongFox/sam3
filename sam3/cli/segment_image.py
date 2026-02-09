# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw
from dataclasses import dataclass

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


@dataclass
class Config:
    """Segment an image using SAM3 with a text prompt."""
    image_path: str
    prompt: str
    save_path: str
    device: str = "auto"
    confidence_threshold: float = 0.5
    mode: Literal["cutout", "overlay", "mask", "boundary"] = "overlay"
    alpha: float = 0.55
    compile: bool = False


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _union_mask_from_state(state: dict) -> np.ndarray:
    if "masks" not in state:
        raise RuntimeError("SAM3 output state did not contain 'masks'.")

    masks = state["masks"]
    if isinstance(masks, torch.Tensor):
        masks_t = masks
    else:
        masks_t = torch.as_tensor(masks)

    if masks_t.numel() == 0:
        h = int(state["original_height"])
        w = int(state["original_width"])
        return np.zeros((h, w), dtype=np.bool_)

    if masks_t.ndim == 4:
        # (N, 1, H, W)
        masks_t = masks_t[:, 0]
    elif masks_t.ndim == 3:
        # (N, H, W)
        pass
    else:
        raise RuntimeError(f"Unexpected mask tensor shape: {tuple(masks_t.shape)}")

    union = masks_t.any(dim=0)
    return union.detach().cpu().numpy().astype(np.bool_)


def _save_mask(mask: np.ndarray, save_path: Path) -> None:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img.save(save_path)


def _save_cutout(image_rgb: "PILImage", mask: np.ndarray, save_path: Path) -> None:
    from PIL import Image

    rgba = image_rgb.convert("RGBA")
    alpha = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    rgba.putalpha(alpha)

    # JPEG cannot store alpha; fall back to black background.
    if save_path.suffix.lower() in {".jpg", ".jpeg"}:
        bg = Image.new("RGB", rgba.size, (0, 0, 0))
        bg.paste(image_rgb.convert("RGB"), mask=alpha)
        bg.save(save_path)
        return

    rgba.save(save_path)


def _save_boundary(image_rgb: "PILImage", mask: np.ndarray, save_path: Path, alpha: float) -> None:
    # Find contours using OpenCV
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image
    result = image_rgb.convert("RGBA")

    # Draw contours
    for contour in contours:
        # Convert contour to points for PIL drawing
        points = [(int(point[0][0]), int(point[0][1])) for point in contour]
        if len(points) > 2:
            # Draw red boundary with specified alpha
            boundary_alpha = int(max(0.0, min(1.0, alpha)) * 255)
            # Create a temporary image for the boundary
            temp_img = Image.new("RGBA", result.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(temp_img)
            draw.line(points + [points[0]], fill=(255, 0, 0, boundary_alpha), width=4)
            # Composite the boundary onto the result
            result = Image.alpha_composite(result, temp_img)

    result.convert("RGB").save(save_path)


def _save_overlay(image_rgb: "PILImage", mask: np.ndarray, save_path: Path, alpha: float) -> None:
    base = image_rgb.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    overlay_alpha = (mask.astype(np.uint8) * int(max(0.0, min(1.0, alpha)) * 255)).astype(np.uint8)
    overlay.putalpha(Image.fromarray(overlay_alpha, mode="L"))
    out = Image.alpha_composite(base, overlay).convert("RGB")
    out.save(save_path)


def main() -> int:
    args = tyro.cli(Config)

    image_path = Path(args.image_path)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)

    # Help Hugging Face caches stay writable in containers.
    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/cache/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        model = build_sam3_image_model(device=device, eval_mode=True, compile=args.compile)
    except Exception as e:
        msg = ("Failed to build/load SAM3 model. If this is a Hugging Face auth error, "
               "set HUGGINGFACE_HUB_TOKEN (or mount your HF cache) and ensure you have access "
               "to facebook/sam3. Original error: ")
        raise RuntimeError(msg + repr(e)) from e

    processor = Sam3Processor(model=model, device=device, confidence_threshold=float(args.confidence_threshold), )

    image = Image.open(image_path).convert("RGB")
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt=str(args.prompt), state=state)

    mask = _union_mask_from_state(state)

    if args.mode == "mask":
        _save_mask(mask, save_path)
    elif args.mode == "overlay":
        _save_overlay(image, mask, save_path, alpha=float(args.alpha))
    elif args.mode == "boundary":
        _save_boundary(image, mask, save_path, alpha=float(args.alpha))
    else:
        _save_cutout(image, mask, save_path)

    return 0


if __name__ == "__main__":
    main()
