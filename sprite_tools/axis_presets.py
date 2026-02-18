from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class AxisPreset:
    preset_id: str
    name: str
    x: int
    y: int
    description: str = ""
    author: str = ""
    schema_version: int = 1


def _normalize_preset_id(raw_id: str, fallback_stem: str) -> str:
    seed = str(raw_id or fallback_stem).strip().lower()
    slug = re.sub(r"[^a-z0-9_-]+", "-", seed).strip("-")
    return slug or "axis-preset"


def load_axis_presets_from_directory(directory: Path) -> tuple[List[AxisPreset], List[str]]:
    presets: List[AxisPreset] = []
    warnings: List[str] = []
    if not directory.exists() or not directory.is_dir():
        return presets, warnings

    for path in sorted(directory.glob("*.axis.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{path.name}: invalid JSON ({exc})")
            continue

        if not isinstance(payload, dict):
            warnings.append(f"{path.name}: root must be an object")
            continue

        name = str(payload.get("name", "")).strip()
        if not name:
            warnings.append(f"{path.name}: missing required 'name'")
            continue

        try:
            x = int(payload.get("x"))
            y = int(payload.get("y"))
        except Exception:  # noqa: BLE001
            warnings.append(f"{path.name}: x and y must be integers")
            continue

        schema_version_raw = payload.get("schema_version", 1)
        try:
            schema_version = int(schema_version_raw)
        except Exception:  # noqa: BLE001
            schema_version = 1

        preset = AxisPreset(
            preset_id=_normalize_preset_id(str(payload.get("id", "")), path.stem),
            name=name,
            x=x,
            y=y,
            description=str(payload.get("description", "") or ""),
            author=str(payload.get("author", "") or ""),
            schema_version=max(1, schema_version),
        )
        presets.append(preset)

    return presets, warnings
