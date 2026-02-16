from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal

ProjectMode = Literal["managed", "linked"]

PROJECT_SCHEMA_VERSION = 1
PROJECT_MANIFEST_NAME = "project.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_name(name: str) -> str:
    filtered = "".join(ch for ch in name if ch.isalnum() or ch in {"-", "_", "."}).strip("._")
    return filtered or "sprite"


@dataclass
class ProjectSpriteEntry:
    sprite_id: str
    source_path: str
    load_mode: Literal["detect", "preserve"]
    source_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sprite_id": self.sprite_id,
            "source_path": self.source_path,
            "load_mode": self.load_mode,
            "source_hash": self.source_hash,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProjectSpriteEntry":
        sprite_id = str(payload.get("sprite_id", "")).strip()
        source_path = str(payload.get("source_path", "")).strip()
        load_mode = str(payload.get("load_mode", "detect")).strip().lower()
        mode: Literal["detect", "preserve"] = "preserve" if load_mode == "preserve" else "detect"
        source_hash = str(payload.get("source_hash", "")).strip().lower()
        if not sprite_id or not source_path:
            raise ValueError("Invalid sprite entry: missing sprite_id/source_path")
        return cls(sprite_id=sprite_id, source_path=source_path, load_mode=mode, source_hash=source_hash)


@dataclass
class ProjectManifest:
    schema_version: int
    project_name: str
    project_mode: ProjectMode
    created_at: str
    updated_at: str
    sprites: List[ProjectSpriteEntry]
    settings: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_name": self.project_name,
            "project_mode": self.project_mode,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_index": [entry.to_dict() for entry in self.sprites],
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProjectManifest":
        schema_version = int(payload.get("schema_version", PROJECT_SCHEMA_VERSION))
        project_name = str(payload.get("project_name", "SpriteTools Project")).strip() or "SpriteTools Project"
        mode_raw = str(payload.get("project_mode", "managed")).strip().lower()
        project_mode: ProjectMode = "linked" if mode_raw == "linked" else "managed"
        created_at = str(payload.get("created_at", "")).strip() or _utc_now_iso()
        updated_at = str(payload.get("updated_at", "")).strip() or created_at
        source_entries = payload.get("source_index", [])
        sprites: List[ProjectSpriteEntry] = []
        if isinstance(source_entries, list):
            for entry in source_entries:
                if isinstance(entry, dict):
                    sprites.append(ProjectSpriteEntry.from_dict(entry))
        settings = payload.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}
        return cls(
            schema_version=schema_version,
            project_name=project_name,
            project_mode=project_mode,
            created_at=created_at,
            updated_at=updated_at,
            sprites=sprites,
            settings=settings,
        )


@dataclass
class ProjectPaths:
    root: Path
    manifest: Path
    metadata: Path
    sprites_metadata: Path
    sources_sprites: Path
    cache: Path
    backups: Path
    exports_renders: Path


class ProjectService:
    def resolve_paths(self, project_root: Path) -> ProjectPaths:
        root = project_root.resolve()
        metadata = root / "metadata"
        backups = root / "backups"
        return ProjectPaths(
            root=root,
            manifest=root / PROJECT_MANIFEST_NAME,
            metadata=metadata,
            sprites_metadata=metadata / "sprites.json",
            sources_sprites=root / "sources" / "sprites",
            cache=root / "cache",
            backups=backups,
            exports_renders=root / "exports" / "renders",
        )

    def create_project(self, project_root: Path, project_name: str | None = None, mode: ProjectMode = "managed") -> tuple[ProjectPaths, ProjectManifest]:
        paths = self.resolve_paths(project_root)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.metadata.mkdir(parents=True, exist_ok=True)
        paths.sources_sprites.mkdir(parents=True, exist_ok=True)
        (paths.cache / "thumbs").mkdir(parents=True, exist_ok=True)
        (paths.cache / "analysis").mkdir(parents=True, exist_ok=True)
        (paths.backups / "snapshots").mkdir(parents=True, exist_ok=True)
        (paths.backups / "autosave").mkdir(parents=True, exist_ok=True)
        paths.exports_renders.mkdir(parents=True, exist_ok=True)

        name = (project_name or paths.root.stem or "SpriteTools Project").strip()
        now = _utc_now_iso()
        manifest = ProjectManifest(
            schema_version=PROJECT_SCHEMA_VERSION,
            project_name=name,
            project_mode=mode,
            created_at=now,
            updated_at=now,
            sprites=[],
            settings={"output_dir": str(paths.exports_renders.relative_to(paths.root)).replace("\\", "/")},
        )
        self.save_manifest(paths, manifest)
        self.save_sprite_metadata(paths, {})
        return paths, manifest

    def open_project(self, project_path: Path) -> tuple[ProjectPaths, ProjectManifest]:
        manifest_path = project_path
        if manifest_path.is_dir():
            manifest_path = manifest_path / PROJECT_MANIFEST_NAME
        if manifest_path.name != PROJECT_MANIFEST_NAME:
            raise ValueError("Project file must be project.json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Project manifest not found: {manifest_path}")

        paths = self.resolve_paths(manifest_path.parent)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = ProjectManifest.from_dict(payload)
        return paths, manifest

    def save_manifest(self, paths: ProjectPaths, manifest: ProjectManifest) -> None:
        manifest.updated_at = _utc_now_iso()
        data = manifest.to_dict()
        paths.manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_sprite_metadata(self, paths: ProjectPaths) -> Dict[str, Dict[str, Any]]:
        if not paths.sprites_metadata.exists():
            return {}
        payload = json.loads(paths.sprites_metadata.read_text(encoding="utf-8"))
        sprites = payload.get("sprites", {})
        if not isinstance(sprites, dict):
            return {}
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, value in sprites.items():
            if isinstance(key, str) and isinstance(value, dict):
                normalized[key] = value
        return normalized

    def save_sprite_metadata(self, paths: ProjectPaths, sprites: Dict[str, Dict[str, Any]]) -> None:
        payload = {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "updated_at": _utc_now_iso(),
            "sprites": sprites,
        }
        paths.sprites_metadata.parent.mkdir(parents=True, exist_ok=True)
        paths.sprites_metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def import_managed_sprite(self, paths: ProjectPaths, source: Path) -> Path:
        source = source.resolve()
        target_dir = paths.sources_sprites
        target_dir.mkdir(parents=True, exist_ok=True)
        base = _safe_name(source.stem)
        suffix = source.suffix.lower() or ".png"
        candidate = target_dir / f"{base}{suffix}"
        counter = 2
        while candidate.exists() and not self._same_file(candidate, source):
            candidate = target_dir / f"{base}_{counter}{suffix}"
            counter += 1
        if not candidate.exists():
            shutil.copy2(source, candidate)
        return candidate

    def hash_file(self, path: Path) -> str:
        digest = hashlib.sha1()
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def source_path_for_entry(self, paths: ProjectPaths, entry: ProjectSpriteEntry, project_mode: ProjectMode) -> Path:
        source = Path(entry.source_path)
        if project_mode == "managed":
            return (paths.root / source).resolve()
        if source.is_absolute():
            return source
        return (paths.root / source).resolve()

    def _same_file(self, left: Path, right: Path) -> bool:
        try:
            return left.resolve() == right.resolve()
        except Exception:  # noqa: BLE001
            return False
