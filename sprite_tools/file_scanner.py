"""Directory scanning helpers for batch sprite processing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

_IMAGE_EXTENSIONS = {
    ".png",
    ".bmp",
    ".gif",
    ".ase",
    ".aseprite",
    ".jpg",
    ".jpeg",
    ".pcx",
    ".tga",
    ".webp",
}


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTENSIONS


@dataclass(slots=True)
class ScanOptions:
    roots: Sequence[Path]
    recursive: bool = True
    allowed_exts: Iterable[str] | None = None


def iter_image_files(options: ScanOptions) -> Iterator[Path]:
    """Yield files matching extensions under the given roots."""

    allowed = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (options.allowed_exts or _IMAGE_EXTENSIONS)
    }
    for root in options.roots:
        root = root.expanduser()
        candidates = root.rglob("*") if options.recursive else root.glob("*")
        for path in candidates:
            if path.is_file() and path.suffix.lower() in allowed:
                yield path
