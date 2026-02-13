"""Utilities for converting RGB/RGBA sprites into indexed palettes."""
from __future__ import annotations

from typing import Iterable, Sequence

from PIL import Image

from .palette_ops import ColorTuple


def _build_palette_image(colors: Sequence[ColorTuple]) -> Image.Image:
    if len(colors) > 256:
        raise ValueError("Palettes are limited to 256 colors for indexed PNGs")
    palette_image = Image.new("P", (1, 1))
    flat = []
    for color in colors:
        flat.extend(color)
    padding = 256 - len(colors)
    if padding > 0:
        flat.extend([0, 0, 0] * padding)
    palette_image.putpalette(flat)
    return palette_image


def quantize_image(
    image: Image.Image,
    *,
    palette: Sequence[ColorTuple] | None = None,
    max_colors: int = 256,
    dither: bool = True,
) -> Image.Image:
    """Return an indexed copy of ``image``.

    ``palette`` enforces a specific palette ordering. Otherwise Pillow's adaptive
    quantizer is used, capped by ``max_colors``.
    """

    dither_flag = Image.FLOYDSTEINBERG if dither else Image.NONE
    if palette:
        base = image.convert("RGB")
        palette_image = _build_palette_image(palette[:256])
        quantized = base.quantize(palette=palette_image, dither=dither_flag)
    else:
        base = image.convert("RGBA")
        quantized = base.quantize(
            colors=max_colors,
            method=Image.FASTOCTREE,
            dither=dither_flag,
        )
    return quantized
