"""Palette/index remapping helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image


ColorTuple = Tuple[int, int, int]


@dataclass(slots=True)
class PaletteInfo:
    """Lightweight snapshot of an indexed image palette."""

    colors: List[ColorTuple]
    transparent_index: int | None

    @property
    def size(self) -> int:
        return len(self.colors)


class PaletteError(RuntimeError):
    """Raised when palette processing fails."""


def ensure_indexed(image: Image.Image) -> Image.Image:
    """Return an indexed (mode "P") copy of ``image``.

    Pillow sometimes converts PNGs to RGB when transparency exceeds 256 colors.
    For now we error out so the CLI can give actionable feedback.
    """

    if image.mode != "P":
        raise PaletteError(
            "Expected indexed PNG (mode 'P'). Convert/quantize before processing."
        )
    return image


def extract_palette(image: Image.Image, *, include_unused: bool = False) -> PaletteInfo:
    """Return palette entries and transparent index from ``image``."""

    ensure_indexed(image)
    palette = image.getpalette()
    if not palette:
        raise PaletteError("Image does not contain palette data")
    colors: List[ColorTuple] = []
    for i in range(0, len(palette), 3):
        colors.append((palette[i], palette[i + 1], palette[i + 2]))
    if not include_unused:
        used = image.getcolors(maxcolors=65_536)
        if used:
            max_index = max(idx for _count, idx in used)
            colors = colors[: max_index + 1]
    trans_index = image.info.get("transparency")
    if isinstance(trans_index, bytes):
        # convert byte array (per-index alpha) into first fully transparent index
        try:
            trans_index = next(i for i, alpha in enumerate(trans_index) if alpha == 0)
        except StopIteration:
            trans_index = None
    elif isinstance(trans_index, tuple):
        trans_index = trans_index[0]
    elif not isinstance(trans_index, int):
        trans_index = None
    return PaletteInfo(colors=colors, transparent_index=trans_index)


def read_act_palette(path: Path) -> List[ColorTuple]:
    """Load an Adobe ACT palette file (<=256 colors)."""

    data = path.read_bytes()
    if len(data) % 3 != 0:
        raise PaletteError("ACT palette length must be divisible by 3")
    colors: List[ColorTuple] = []
    for i in range(0, min(len(data), 256 * 3), 3):
        colors.append((data[i], data[i + 1], data[i + 2]))
    return colors
