from __future__ import annotations

import json
import hashlib
import re
import sys
import time
from datetime import datetime, timezone
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Literal, Sequence, Tuple

import logging
import os

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import (
    QAbstractListModel,
    QEvent,
    QItemSelection,
    QItemSelectionModel,
    QMimeData,
    QModelIndex,
    QPoint,
    QPointF,
    QRect,
    QSize,
    QSettings,
    Qt,
    Signal,
    QTimer,
)
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent, QShortcut, QBitmap, QImage, QRegion, QPolygon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QKeySequenceEdit,
    QPushButton,
    QGroupBox,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QSizePolicy,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except Exception:  # noqa: BLE001
    QOpenGLWidget = None

from sprite_tools.file_scanner import ScanOptions, iter_image_files, is_supported_image
from sprite_tools.palette_ops import PaletteError, PaletteInfo, extract_palette, read_act_palette
from sprite_tools.project_system import PROJECT_MANIFEST_NAME, ProjectManifest, ProjectPaths, ProjectService, ProjectSpriteEntry
from sprite_tools.quantization import quantize_image
from sprite_tools.processing import ProcessOptions, process_image_object, process_sprite, write_act

ColorTuple = Tuple[int, int, int]

logger = logging.getLogger(__name__)
DEBUG_LOG_PATH: Path | None = None
_EXCEPTION_HOOK_INSTALLED = False
_PALETTE_CLIPBOARD_MIME = "application/x-spritetools-palette+json"
_SPRITE_BASE_NAME_ROLE = Qt.ItemDataRole.UserRole + 1
_GROUP_ID_ROLE = Qt.ItemDataRole.UserRole + 2
_SPRITE_ICON_SIG_ROLE = Qt.ItemDataRole.UserRole + 3
_TIMELINE_ITEM_KIND_ROLE = Qt.ItemDataRole.UserRole + 12
_TIMELINE_FRAME_LABEL_ROLE = Qt.ItemDataRole.UserRole + 13
_TIMELINE_FRAME_DURATION_ROLE = Qt.ItemDataRole.UserRole + 14
_TIMELINE_FRAME_NUMBER_ROLE = Qt.ItemDataRole.UserRole + 15
_TIMELINE_FRAME_NAME_ROLE = Qt.ItemDataRole.UserRole + 16
_BROWSER_REBUILD_WARN_MS = 40.0
_BROWSER_INPLACE_WARN_MS = 20.0
_SPRITE_ICON_CACHE_LIMIT = 8192
_BROWSER_DEBOUNCE_FAST_MS = 20
_BROWSER_DEBOUNCE_MEDIUM_MS = 35
_BROWSER_DEBOUNCE_SLOW_MS = 60
_BROWSER_INPLACE_CHUNK_SMALL = 400
_BROWSER_INPLACE_CHUNK_MEDIUM = 250
_BROWSER_INPLACE_CHUNK_LARGE = 150
_EVENT_TYPE_CLOSE = int(QEvent.Type.Close)
_EVENT_TYPE_WHEEL = int(QEvent.Type.Wheel)
_PROJECT_FOLDER_SUFFIX = ".spto"
_PROJECT_LEGACY_FOLDER_SUFFIX = ".spritetools"
_RECENT_PROJECTS_LIMIT = 8
_GROUP_COLOR_PRESETS: List[ColorTuple] = [
    (220, 120, 120),
    (120, 180, 240),
    (140, 220, 150),
    (235, 190, 120),
    (180, 140, 230),
    (120, 210, 210),
    (240, 150, 210),
    (200, 200, 120),
]


def _setup_debug_logging() -> None:
    global DEBUG_LOG_PATH, _EXCEPTION_HOOK_INSTALLED
    if not os.environ.get("SPRITETOOLS_DEBUG"):
        logger.addHandler(logging.NullHandler())
        return
    log_name = os.environ.get("SPRITETOOLS_DEBUG_LOG", "sprite_tools_debug.log")
    log_path = Path(log_name)
    if not log_path.is_absolute():
        log_path = Path(__file__).resolve().parents[2] / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # remove existing file handlers to avoid duplicates
    root_logger.handlers = [h for h in root_logger.handlers if not isinstance(h, logging.FileHandler)]
    root_logger.addHandler(handler)
    DEBUG_LOG_PATH = log_path
    root_logger.info("SpriteTools debug logging enabled at %s", log_path)
    if not _EXCEPTION_HOOK_INSTALLED:
        previous_hook = sys.excepthook

        def _logging_excepthook(exc_type, exc_value, exc_traceback, _prev=previous_hook):
            root_logger.error(
                "Unhandled exception",
                exc_info=(exc_type, exc_value, exc_traceback),
            )
            _prev(exc_type, exc_value, exc_traceback)

        sys.excepthook = _logging_excepthook
        _EXCEPTION_HOOK_INSTALLED = True


_setup_debug_logging()


@dataclass
class PaletteClipboardPayload:
    colors: List[ColorTuple | None]
    alphas: List[int]


def _extract_palette_alphas_from_image_info(image: Image.Image, count: int) -> List[int]:
    alphas = [255] * max(0, count)
    transparency = image.info.get("transparency")
    if isinstance(transparency, int):
        if 0 <= transparency < len(alphas):
            alphas[transparency] = 0
    elif isinstance(transparency, (bytes, bytearray, list, tuple)):
        for idx in range(min(len(alphas), len(transparency))):
            try:
                alphas[idx] = max(0, min(255, int(transparency[idx])))
            except (TypeError, ValueError):
                alphas[idx] = 255
    return alphas


def _parse_gpl_palette(path: Path) -> List[ColorTuple]:
    colors: List[ColorTuple] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("GIMP Palette") or line.startswith("Name:") or line.startswith("Columns:"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        colors.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
        if len(colors) >= 256:
            break
    if not colors:
        raise PaletteError("GPL file did not contain any valid colors")
    return colors


def _parse_jasc_pal_palette(path: Path) -> List[ColorTuple]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if len(lines) < 4 or lines[0].upper() != "JASC-PAL":
        raise PaletteError("Unsupported .pal format (expected JASC-PAL)")
    colors: List[ColorTuple] = []
    for line in lines[3:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        colors.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
        if len(colors) >= 256:
            break
    if not colors:
        raise PaletteError("JASC-PAL file did not contain any valid colors")
    return colors


def _parse_hex_palette_text(path: Path) -> List[ColorTuple]:
    colors: List[ColorTuple] = []
    pattern = re.compile(r"#?([0-9a-fA-F]{6})")
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.search(line)
        if not match:
            continue
        value = match.group(1)
        colors.append((int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)))
        if len(colors) >= 256:
            break
    if not colors:
        raise PaletteError("Text palette did not contain any #RRGGBB colors")
    return colors


def _load_palette_from_source(path: Path, mode: Literal["detect", "preserve"]) -> tuple[List[ColorTuple], List[int], str]:
    ext = path.suffix.lower()
    if ext in {".act", ".gpl", ".pal", ".txt", ".hex"}:
        if ext == ".act":
            colors = read_act_palette(path)
        elif ext == ".gpl":
            colors = _parse_gpl_palette(path)
        elif ext == ".pal":
            colors = _parse_jasc_pal_palette(path)
        else:
            colors = _parse_hex_palette_text(path)
        alphas = [255] * len(colors)
        logger.debug("Loaded palette file path=%s format=%s colors=%s", path.name, ext, len(colors))
        return colors[:256], alphas[:256], f"file:{ext}"

    if not is_supported_image(path):
        raise PaletteError(f"Unsupported palette source: {path.suffix}")

    with Image.open(path) as source:
        image = source.copy()
    if mode == "preserve":
        if image.mode != "P":
            raise PaletteError("Preserve mode requires indexed image (mode 'P')")
        palette = extract_palette(image, include_unused=True)
        colors = palette.colors[:256]
        alphas = _extract_palette_alphas_from_image_info(image, len(colors))
        logger.debug(
            "Loaded palette from image preserve path=%s colors=%s raw_palette_entries=%s",
            path.name,
            len(colors),
            len(image.getpalette() or []) // 3,
        )
        return colors, alphas, "image:preserve"

    # detect
    if image.mode == "P":
        palette = extract_palette(image)
        colors = palette.colors[:256]
        alphas = _extract_palette_alphas_from_image_info(image, len(colors))
        logger.debug("Loaded palette from indexed image detect path=%s colors=%s", path.name, len(colors))
        return colors, alphas, "image:detect-indexed"

    rgba = image.convert("RGBA")
    quantized = quantize_image(rgba, max_colors=256, dither=False)
    palette = extract_palette(quantized)
    colors = palette.colors[:256]
    alphas = _extract_palette_alphas_from_image_info(quantized, len(colors))
    logger.debug("Loaded palette from image detect-quantized path=%s colors=%s", path.name, len(colors))
    return colors, alphas, "image:detect-quantized"


def _set_palette_clipboard(payload: PaletteClipboardPayload) -> None:
    clipboard = QApplication.clipboard()
    mime = QMimeData()
    serializable_colors: List[List[int] | None] = []
    for color in payload.colors:
        if color is None:
            serializable_colors.append(None)
        else:
            serializable_colors.append([int(color[0]), int(color[1]), int(color[2])])
    packet = {
        "version": 1,
        "colors": serializable_colors,
        "alphas": [int(max(0, min(255, alpha))) for alpha in payload.alphas],
    }
    encoded = json.dumps(packet).encode("utf-8")
    mime.setData(_PALETTE_CLIPBOARD_MIME, encoded)
    text_lines: List[str] = []
    for color, alpha in zip(payload.colors, payload.alphas):
        if color is None:
            text_lines.append("-")
        elif alpha < 255:
            text_lines.append(f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}{alpha:02X}")
        else:
            text_lines.append(f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}")
    mime.setText("\n".join(text_lines))
    clipboard.setMimeData(mime)
    logger.debug("Clipboard set palette entries=%s", len(payload.colors))


def _get_palette_clipboard() -> PaletteClipboardPayload | None:
    clipboard = QApplication.clipboard()
    mime = clipboard.mimeData()
    if mime is None:
        return None

    if mime.hasFormat(_PALETTE_CLIPBOARD_MIME):
        try:
            packet = json.loads(bytes(mime.data(_PALETTE_CLIPBOARD_MIME)).decode("utf-8"))
            colors_raw = packet.get("colors", [])
            alphas_raw = packet.get("alphas", [])
            colors: List[ColorTuple | None] = []
            alphas: List[int] = []
            for idx, value in enumerate(colors_raw):
                if value is None:
                    colors.append(None)
                elif isinstance(value, list) and len(value) >= 3:
                    colors.append((int(value[0]), int(value[1]), int(value[2])))
                else:
                    continue
                alpha = int(alphas_raw[idx]) if idx < len(alphas_raw) else 255
                alphas.append(max(0, min(255, alpha)))
            if colors:
                logger.debug("Clipboard parsed app payload entries=%s", len(colors))
                return PaletteClipboardPayload(colors=colors, alphas=alphas)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Clipboard app payload parse failed: %s", exc)

    text = clipboard.text().strip()
    if not text:
        return None
    colors: List[ColorTuple | None] = []
    alphas: List[int] = []
    for token in re.split(r"[\r\n,;\s]+", text):
        if not token:
            continue
        if token == "-":
            colors.append(None)
            alphas.append(255)
            continue
        value = token[1:] if token.startswith("#") else token
        if len(value) == 6 and re.fullmatch(r"[0-9a-fA-F]{6}", value):
            colors.append((int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)))
            alphas.append(255)
        elif len(value) == 8 and re.fullmatch(r"[0-9a-fA-F]{8}", value):
            colors.append((int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)))
            alphas.append(int(value[6:8], 16))
    if colors:
        logger.debug("Clipboard parsed text payload entries=%s", len(colors))
        return PaletteClipboardPayload(colors=colors, alphas=alphas)
    return None


@dataclass
class SpriteRecord:
    path: Path
    pixmap: QPixmap
    palette: List[ColorTuple]
    slot_bindings: Dict[int, int]
    indexed_image: Image.Image
    load_mode: Literal["detect", "preserve"] = "detect"
    group_id: str | None = None
    local_overrides: Dict[int, Tuple[ColorTuple, int]] = field(default_factory=dict)
    offset_x: int = 0  # Per-sprite X offset
    offset_y: int = 0  # Per-sprite Y offset
    canvas_width: int = 304
    canvas_height: int = 224
    canvas_override_enabled: bool = False


@dataclass
class PaletteGroup:
    group_id: str
    mode: Literal["detect", "preserve"]
    signature: str
    display_name: str = ""
    color: ColorTuple = (120, 120, 120)
    member_keys: set[str] = field(default_factory=set)
    detect_palette_colors: List[ColorTuple] = field(default_factory=list)
    detect_palette_alphas: List[int] = field(default_factory=list)
    detect_palette_slot_ids: List[int] = field(default_factory=list)
    canvas_width: int = 304
    canvas_height: int = 224
    offset_x: int = 0
    offset_y: int = 0
    canvas_override_enabled: bool = False


@dataclass
class AnimationFrameEntry:
    sprite_key: str
    duration_frames: int = 1
    gap_before_frames: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sprite_key": str(self.sprite_key),
            "duration_frames": max(1, int(self.duration_frames)),
            "gap_before_frames": max(0, int(self.gap_before_frames)),
            "notes": str(self.notes or ""),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AnimationFrameEntry":
        sprite_key = str(payload.get("sprite_key", "")).strip()
        if not sprite_key:
            raise ValueError("Animation frame entry missing sprite_key")
        duration_frames = max(1, int(payload.get("duration_frames", 1)))
        gap_before_frames = max(0, int(payload.get("gap_before_frames", 0)))
        notes = str(payload.get("notes", "") or "")
        return cls(
            sprite_key=sprite_key,
            duration_frames=duration_frames,
            gap_before_frames=gap_before_frames,
            notes=notes,
        )


@dataclass
class AnimationTag:
    tag_id: str
    name: str
    state_label: str = ""
    frames: List[AnimationFrameEntry] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.tag_id),
            "name": str(self.name),
            "state_label": str(self.state_label or ""),
            "frames": [frame.to_dict() for frame in self.frames],
            "notes": str(self.notes or ""),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AnimationTag":
        tag_id = str(payload.get("id", "")).strip()
        name = str(payload.get("name", "")).strip()
        if not tag_id or not name:
            raise ValueError("Animation tag missing id/name")
        state_label = str(payload.get("state_label", "") or "")
        notes = str(payload.get("notes", "") or "")
        frames_raw = payload.get("frames", [])
        frames: List[AnimationFrameEntry] = []
        if isinstance(frames_raw, list):
            for frame_payload in frames_raw:
                if not isinstance(frame_payload, dict):
                    continue
                try:
                    frames.append(AnimationFrameEntry.from_dict(frame_payload))
                except Exception:  # noqa: BLE001
                    continue
        return cls(tag_id=tag_id, name=name, state_label=state_label, frames=frames, notes=notes)


@dataclass
class HistoryEntry:
    label: str
    state: Dict[str, Any]


@dataclass
class HistoryField:
    capture: Callable[[], Any]
    apply: Callable[[Any], None]


def _process_preview_request(source: Image.Image, options: ProcessOptions) -> tuple[Image.Image, PaletteInfo]:
    return process_image_object(source, options)


class HistoryManager:
    def __init__(self) -> None:
        self._fields: Dict[str, HistoryField] = {}
        self._history: List[HistoryEntry] = []
        self._index = -1
        self._restoring = False
        self._ready = False
        self._last_undo_label: str | None = None
        self._last_redo_label: str | None = None

    def register_field(self, name: str, capture: Callable[[], Any], apply: Callable[[Any], None]) -> None:
        self._fields[name] = HistoryField(capture=capture, apply=apply)

    @property
    def is_restoring(self) -> bool:
        return self._restoring

    def reset(self, label: str = "reset") -> None:
        if not self._fields:
            return
        snapshot = self._capture_state()
        self._history = [HistoryEntry(label=label, state=snapshot)]
        self._index = 0
        self._ready = True
        logger.debug("History reset label=%s fields=%s entries=%s index=%s", label, list(self._fields.keys()), len(self._history), self._index)

    def record(
        self,
        label: str,
        *,
        force: bool = False,
        include_fields: Sequence[str] | None = None,
    ) -> bool:
        if not self._ready or self._restoring:
            logger.debug(
                "History record skipped label=%s ready=%s restoring=%s force=%s",
                label,
                self._ready,
                self._restoring,
                force,
            )
            return False
        snapshot = self._capture_state(include_fields=include_fields)
        if not force and self._history and snapshot == self._history[self._index].state:
            logger.debug("History record dedup label=%s index=%s entries=%s", label, self._index, len(self._history))
            return False
        if self._index < len(self._history) - 1:
            self._history = self._history[: self._index + 1]
            logger.debug("History record truncated future branch label=%s new_len=%s", label, len(self._history))
        self._history.append(HistoryEntry(label=label, state=snapshot))
        self._index += 1
        logger.debug(
            "History record added label=%s force=%s fields=%s index=%s entries=%s",
            label,
            force,
            list(include_fields) if include_fields is not None else "all",
            self._index,
            len(self._history),
        )
        return True

    def undo(self) -> bool:
        if not self.can_undo:
            logger.debug("History undo skipped can_undo=%s index=%s entries=%s", self.can_undo, self._index, len(self._history))
            return False
        undone_label = self._history[self._index].label
        self._index -= 1
        applied_label = self._history[self._index].label
        self._last_undo_label = undone_label
        logger.debug(
            "History undo apply undo_label=%s apply_label=%s index=%s entries=%s",
            undone_label,
            applied_label,
            self._index,
            len(self._history),
        )
        self._apply_state(self._history[self._index].state)
        return True

    def redo(self) -> bool:
        if not self.can_redo:
            logger.debug("History redo skipped can_redo=%s index=%s entries=%s", self.can_redo, self._index, len(self._history))
            return False
        self._index += 1
        applied_label = self._history[self._index].label
        self._last_redo_label = applied_label
        logger.debug("History redo apply label=%s index=%s entries=%s", applied_label, self._index, len(self._history))
        self._apply_state(self._history[self._index].state)
        return True

    @property
    def last_undo_label(self) -> str | None:
        return self._last_undo_label

    @property
    def last_redo_label(self) -> str | None:
        return self._last_redo_label

    @property
    def can_undo(self) -> bool:
        return self._ready and self._index > 0

    @property
    def can_redo(self) -> bool:
        return self._ready and self._index >= 0 and self._index < len(self._history) - 1

    def _capture_state(self, include_fields: Sequence[str] | None = None) -> Dict[str, Any]:
        if include_fields is None:
            return {name: field.capture() for name, field in self._fields.items()}
        selected = [name for name in include_fields if name in self._fields]
        return {name: self._fields[name].capture() for name in selected}

    def _apply_state(self, state: Dict[str, Any]) -> None:
        self._restoring = True
        try:
            for name, value in state.items():
                field = self._fields.get(name)
                if field is not None:
                    field.apply(value)
        finally:
            self._restoring = False
@dataclass
class DropDecision:
    insert_row: int | None
    hint_row: int | None
    hint_mode: Literal["before", "after", "swap", "none"]
    reason: str
    swap_row: int | None = None


class PaletteItemDelegate(QStyledItemDelegate):
    """Custom delegate that keeps the swatch color intact when selected."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.show_index_labels = True
        self.show_grid_lines = False
        self.local_rows: set[int] = set()
        self.usage_counts: Dict[int, int] = {}
        self.show_usage_badge = False
        self.merge_source_rows: set[int] = set()
        self.merge_destination_row: int | None = None
        self.merge_candidate_risk: Dict[int, str] = {}
        self.swatch_inset = 6
        self._last_selection_debug_key: tuple[Any, ...] | None = None

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        color = index.data(Qt.UserRole)
        alpha_raw = index.data(Qt.UserRole + 1)
        alpha = 255 if alpha_raw is None else max(0, min(255, int(alpha_raw)))
        is_empty = color is None

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(option.rect, QColor(42, 42, 42, 0))

        # Canonical swatch rect: single source of truth for color tile, local marker, and selection border.
        inset = max(1, int(self.swatch_inset))
        swatch_side = min(option.rect.width(), option.rect.height()) - (inset * 2)
        swatch_side = max(12, swatch_side)
        swatch_rect = QRect(
            option.rect.x() + (option.rect.width() - swatch_side) // 2,
            option.rect.y() + (option.rect.height() - swatch_side) // 2,
            swatch_side,
            swatch_side,
        )
        swatch_rect = swatch_rect.intersected(option.rect.adjusted(1, 1, -1, -1))

        # Draw swatch checkerboard + color directly to avoid Qt icon layout offset mismatches.
        tile = max(2, swatch_rect.width() // 4)
        light = QColor(200, 200, 200)
        dark = QColor(150, 150, 150)
        painter.setPen(Qt.PenStyle.NoPen)
        for y in range(swatch_rect.top(), swatch_rect.bottom() + 1, tile):
            for x in range(swatch_rect.left(), swatch_rect.right() + 1, tile):
                use_light = (((x - swatch_rect.left()) // tile) + ((y - swatch_rect.top()) // tile)) % 2 == 0
                painter.setBrush(light if use_light else dark)
                painter.drawRect(
                    x,
                    y,
                    min(tile, swatch_rect.right() - x + 1),
                    min(tile, swatch_rect.bottom() - y + 1),
                )

        if is_empty:
            painter.fillRect(swatch_rect, QColor(60, 60, 60, 220))
        else:
            painter.fillRect(swatch_rect, QColor(color[0], color[1], color[2], alpha))

        # Subtle swatch edge for pixel-crisp boundaries.
        painter.setPen(QPen(QColor(30, 30, 30, 210), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(swatch_rect)
        painter.restore()
        
        if self.show_grid_lines:
            painter.save()
            pen = QPen(QColor(120, 120, 120, 180))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(option.rect.adjusted(0, 0, -1, -1))
            painter.restore()

        if self.show_index_labels:
            # Draw index number on the swatch
            painter.save()
            index_text = str(index.row())
            font = painter.font()
            font.setPointSize(8)
            font.setBold(True)
            painter.setFont(font)
            text_rect = option.rect.adjusted(2, 2, -2, -2)
            
            if is_empty:
                painter.setPen(QColor(120, 120, 120))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, index_text)
            else:
                painter.setPen(QColor(255, 255, 255))
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    offset_rect = text_rect.adjusted(dx, dy, dx, dy)
                    painter.drawText(offset_rect, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, index_text)
                painter.setPen(QColor(0, 0, 0))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, index_text)
            painter.restore()

        if index.row() in self.local_rows and not is_empty:
            painter.save()
            marker = QRect(swatch_rect.right() - 10, swatch_rect.top() + 2, 9, 9)
            painter.setPen(QPen(QColor(20, 20, 20, 220), 1))
            painter.setBrush(QColor(255, 215, 0, 220))
            painter.drawEllipse(marker)
            painter.restore()

        if self.show_usage_badge and not is_empty:
            usage = int(self.usage_counts.get(index.row(), 0))
            if usage > 0:
                painter.save()
                badge_text = str(usage)
                font = painter.font()
                font.setPointSize(7)
                font.setBold(True)
                painter.setFont(font)
                metrics = painter.fontMetrics()
                badge_w = max(14, metrics.horizontalAdvance(badge_text) + 6)
                badge_h = max(12, metrics.height() + 2)
                badge_rect = QRect(
                    swatch_rect.right() - badge_w + 1,
                    swatch_rect.bottom() - badge_h + 1,
                    badge_w,
                    badge_h,
                )
                painter.setPen(QPen(QColor(18, 18, 18, 230), 1))
                painter.setBrush(QColor(30, 30, 30, 220))
                painter.drawRoundedRect(badge_rect, 3, 3)
                painter.setPen(QColor(230, 230, 230))
                painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)
                painter.restore()

        if not is_empty:
            risk_level = self.merge_candidate_risk.get(index.row())
            if risk_level in {"safe", "risky"}:
                painter.save()
                risk_color = {
                    "safe": QColor(96, 210, 120),
                    "risky": QColor(255, 96, 96),
                }[risk_level]
                risk_rect = swatch_rect.adjusted(-3, -3, 3, 3).intersected(option.rect.adjusted(1, 1, -1, -1))
                risk_pen = QPen(risk_color)
                risk_pen.setWidth(2)
                painter.setPen(risk_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(risk_rect)
                painter.restore()

            is_source = index.row() in self.merge_source_rows
            is_destination = self.merge_destination_row == index.row()
            if is_source or is_destination:
                painter.save()
                border_color = QColor(86, 182, 255) if is_source else QColor(196, 120, 255)
                role_rect = swatch_rect.adjusted(-2, -2, 2, 2).intersected(option.rect.adjusted(1, 1, -1, -1))
                role_pen = QPen(border_color)
                role_pen.setWidth(4)
                painter.setPen(role_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(role_rect)
                painter.restore()
        
        if not is_selected:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        outer_rect = swatch_rect.adjusted(-1, -1, 1, 1).intersected(option.rect.adjusted(1, 1, -1, -1))
        outer_pen = QPen(QColor(92, 92, 92))
        outer_pen.setWidth(1)
        painter.setPen(outer_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(outer_rect)
        inner_rect = swatch_rect
        highlight_pen = QPen(QColor(232, 232, 232))
        highlight_pen.setWidth(1)
        painter.setPen(highlight_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(inner_rect)

        if logger.isEnabledFor(logging.DEBUG):
            debug_key = (
                index.row(),
                option.rect.x(), option.rect.y(), option.rect.width(), option.rect.height(),
                swatch_rect.x(), swatch_rect.y(), swatch_rect.width(), swatch_rect.height(),
                outer_rect.x(), outer_rect.y(), outer_rect.width(), outer_rect.height(),
            )
            if debug_key != self._last_selection_debug_key:
                self._last_selection_debug_key = debug_key
                logger.debug(
                    "Palette selection geometry row=%s cell=(%s,%s %sx%s) swatch=(%s,%s %sx%s) outer=(%s,%s %sx%s)",
                    index.row(),
                    option.rect.x(), option.rect.y(), option.rect.width(), option.rect.height(),
                    swatch_rect.x(), swatch_rect.y(), swatch_rect.width(), swatch_rect.height(),
                    outer_rect.x(), outer_rect.y(), outer_rect.width(), outer_rect.height(),
                )
        painter.restore()


def _empty_slot_pixmap(size: int = 32) -> QPixmap:
    """Create a pixmap for empty palette slots."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(60, 60, 60))  # Dark gray background
    painter = QPainter(pixmap)
    # Draw checkerboard pattern
    painter.setPen(Qt.PenStyle.NoPen)
    checker_size = size // 4
    for row in range(4):
        for col in range(4):
            if (row + col) % 2 == 0:
                painter.setBrush(QColor(70, 70, 70))
            else:
                painter.setBrush(QColor(50, 50, 50))
            painter.drawRect(col * checker_size, row * checker_size, checker_size, checker_size)
    painter.end()
    return pixmap


def _color_pixmap(color: ColorTuple, size: int = 32, alpha: int = 255) -> QPixmap:
    """Create a color swatch pixmap with alpha visualization via checkerboard."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    
    # Draw checkerboard background (like empty slot but lighter)
    painter.setPen(Qt.PenStyle.NoPen)
    checker_size = size // 4
    for row in range(4):
        for col in range(4):
            if (row + col) % 2 == 0:
                painter.setBrush(QColor(200, 200, 200))  # Light gray
            else:
                painter.setBrush(QColor(150, 150, 150))  # Darker gray
            painter.drawRect(col * checker_size, row * checker_size, checker_size, checker_size)
    
    # Draw color with alpha on top
    color_with_alpha = QColor(*color, alpha)
    painter.fillRect(0, 0, size, size, color_with_alpha)
    painter.end()
    
    return pixmap


def _compute_forced_cell_size(viewport_width: int, columns: int, gap: int, *, min_cell: int = 16, max_cell: int = 96) -> int:
    cols = max(1, int(columns))
    safe_gap = max(0, int(gap))
    available = max(1, int(viewport_width))
    required_gap = (cols - 1) * safe_gap
    cell = (available - required_gap) // cols
    cell = max(min_cell, min(max_cell, cell))
    while (cols * cell) + required_gap > available and cell > min_cell:
        cell -= 1
    return cell


class PaletteModel(QAbstractListModel):
    palette_changed = Signal(list, str)

    def __init__(self) -> None:
        super().__init__()
        self.colors: List[ColorTuple | None] = [None] * 256  # Always 256 slots
        self.alphas: List[int] = [255] * 256  # Alpha values for each slot (255 = opaque)
        self.cell_size = 48
        self.slot_ids: List[int] = list(range(256))
        self.last_index_remap: Dict[int, int] | None = None

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent and parent.isValid():
            return 0
        return 256  # Always 256 slots

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # type: ignore[override]
        if not index.isValid() or not (0 <= index.row() < 256):
            return None
        color = self.colors[index.row()]
        if role == Qt.DecorationRole:
            if color is None:
                return QIcon(_empty_slot_pixmap())
            alpha = self.alphas[index.row()]
            return QIcon(_color_pixmap(color, alpha=alpha))
        if role == Qt.UserRole:
            return color
        if role == Qt.UserRole + 1:  # Alpha role
            return self.alphas[index.row()]
        if role == Qt.SizeHintRole:
            return QSize(self.cell_size, self.cell_size)
        return None

    def set_cell_size(self, size: int) -> None:
        self.cell_size = max(16, min(128, int(size)))
        self.layoutChanged.emit()

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:  # type: ignore[override]
        base_flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDropEnabled
        if index.isValid():
            return base_flags | Qt.ItemIsDragEnabled
        return base_flags

    def supportedDropActions(self) -> Qt.DropActions:  # type: ignore[override]
        return Qt.MoveAction

    def moveRows(  # type: ignore[override]
        self,
        sourceParent: QModelIndex,
        sourceRow: int,
        count: int,
        destinationParent: QModelIndex,
        destinationChild: int,
    ) -> bool:
        logger.debug(
            "moveRows sourceParent=%s sourceRow=%s count=%s destParent=%s destChild=%s",
            sourceParent,
            sourceRow,
            count,
            destinationParent,
            destinationChild,
        )
        if count != 1:
            logger.debug("moveRows aborted: count %s", count)
            return False
        if sourceParent.isValid() or destinationParent.isValid():
            logger.debug("moveRows aborted: non-root parents")
            return False
        if not (0 <= sourceRow < 256 and 0 <= destinationChild <= 256):
            logger.debug("moveRows aborted: invalid indices source=%s dest=%s", sourceRow, destinationChild)
            return False
        if destinationChild > 256:
            destinationChild = 256
        if destinationChild in (sourceRow, sourceRow + 1):
            logger.debug("moveRows no-op (source=%s dest=%s)", sourceRow, destinationChild)
            return False
        
        self.beginMoveRows(QModelIndex(), sourceRow, sourceRow, QModelIndex(), destinationChild)
        color = self.colors[sourceRow]
        alpha = self.alphas[sourceRow]
        insert_row = destinationChild if destinationChild <= sourceRow else destinationChild - 1
        
        # Swap the colors and alphas
        self.colors[sourceRow] = self.colors[insert_row]
        self.colors[insert_row] = color
        self.alphas[sourceRow] = self.alphas[insert_row]
        self.alphas[insert_row] = alpha
        
        self.endMoveRows()
        logger.debug("moveRows success source=%s insert_row=%s", sourceRow, insert_row)
        self.last_index_remap = {idx: idx for idx in range(256)}
        self.last_index_remap[sourceRow] = insert_row
        self.last_index_remap[insert_row] = sourceRow
        self.palette_changed.emit([c for c in self.colors if c is not None], "reorder")
        return True

    def swap_rows(self, row_a: int, row_b: int) -> bool:
        if row_a == row_b:
            logger.debug("swap_rows no-op row=%s", row_a)
            return False
        if not (0 <= row_a < 256 and 0 <= row_b < 256):
            logger.debug("swap_rows invalid rows row_a=%s row_b=%s", row_a, row_b)
            return False
        self.colors[row_a], self.colors[row_b] = self.colors[row_b], self.colors[row_a]
        self.alphas[row_a], self.alphas[row_b] = self.alphas[row_b], self.alphas[row_a]
        index_a = self.index(row_a)
        index_b = self.index(row_b)
        roles = [Qt.DecorationRole, Qt.UserRole]
        self.dataChanged.emit(index_a, index_a, roles)
        self.dataChanged.emit(index_b, index_b, roles)
        logger.debug("swap_rows success row_a=%s row_b=%s", row_a, row_b)
        self.last_index_remap = {idx: idx for idx in range(256)}
        self.last_index_remap[row_a] = row_b
        self.last_index_remap[row_b] = row_a
        self.palette_changed.emit([c for c in self.colors if c is not None], "swap")
        return True

    def set_colors(
        self,
        colors: Sequence[ColorTuple],
        *,
        slots: Sequence[int] | None = None,
        alphas: Sequence[int] | None = None,
        reason: str = "refresh",
        emit_signal: bool = True,
    ) -> None:
        self.beginResetModel()
        # Always maintain 256 slots
        self.colors = [None] * 256
        self.alphas = [255] * 256  # Reset all alphas to opaque
        if slots is not None:
            # Place colors at their slot positions
            for i, (color, slot_id) in enumerate(zip(colors, slots)):
                if 0 <= slot_id < 256:
                    self.colors[slot_id] = color
                    if alphas is not None and i < len(alphas):
                        self.alphas[slot_id] = max(0, min(255, int(alphas[i])))
            self.slot_ids = list(range(256))
        else:
            # Place colors sequentially, rest are None
            for i, color in enumerate(colors[:256]):
                self.colors[i] = color
                if alphas is not None and i < len(alphas):
                    self.alphas[i] = max(0, min(255, int(alphas[i])))
            self.slot_ids = list(range(256))
        self.endResetModel()

        self.last_index_remap = None

        if emit_signal:
            self.palette_changed.emit([c for c in self.colors if c is not None], reason)

    def update_color(self, row: int, color: ColorTuple, alpha: int = 255) -> None:
        if not (0 <= row < 256):
            return
        self.colors[row] = color
        self.alphas[row] = alpha
        index = self.index(row)
        self.dataChanged.emit(index, index, [Qt.DecorationRole, Qt.UserRole])
        logger.debug("Model update_color row=%s color=%s alpha=%s", row, color, alpha)
        self.last_index_remap = None
        self.palette_changed.emit([c for c in self.colors if c is not None], "edit")

    def set_slot(self, row: int, color: ColorTuple | None, alpha: int = 255) -> None:
        if not (0 <= row < 256):
            return
        self.colors[row] = color
        self.alphas[row] = max(0, min(255, int(alpha)))
        index = self.index(row)
        self.dataChanged.emit(index, index, [Qt.DecorationRole, Qt.UserRole])
        logger.debug("Model set_slot row=%s color=%s alpha=%s", row, color, alpha)

    def slot_map(self) -> List[int]:
        if len(self.slot_ids) != len(self.colors):
            self.slot_ids = list(range(len(self.colors)))
        return list(self.slot_ids)

    def set_slot_map(self, slots: Sequence[int]) -> None:
        self.slot_ids = list(slots[: len(self.colors)])
        if len(self.slot_ids) < len(self.colors):
            self.slot_ids.extend(range(len(self.slot_ids), len(self.colors)))

    def merge_rows(self, source_row: int, target_row: int) -> bool:
        """Merge source_row into target_row, removing source_row."""
        if source_row == target_row:
            logger.debug("merge_rows no-op (same row=%s)", source_row)
            return False
        if not (0 <= source_row < len(self.colors) and 0 <= target_row < len(self.colors)):
            logger.debug("merge_rows invalid rows source=%s target=%s", source_row, target_row)
            return False
        
        # Remove source row
        self.beginRemoveRows(QModelIndex(), source_row, source_row)
        self.colors.pop(source_row)
        if self.slot_ids:
            self.slot_ids.pop(source_row)
        self.endRemoveRows()
        
        logger.debug("merge_rows success source=%s target=%s", source_row, target_row)
        self.palette_changed.emit(list(self.colors), "merge")
        return True

    @staticmethod
    def _shift_block(values: List[Any], start: int, end: int, target_start: int) -> List[Any]:
        block = values[start : end + 1]
        remaining = values[:start] + values[end + 1 :]
        insert_at = target_start
        return remaining[:insert_at] + block + remaining[insert_at:]

    def shift_block(self, start: int, end: int, target_start: int) -> bool:
        if not (0 <= start <= end < 256):
            logger.debug("shift_block invalid range start=%s end=%s", start, end)
            return False
        block_len = end - start + 1
        if not (0 <= target_start <= 256 - block_len):
            logger.debug("shift_block invalid target_start=%s block_len=%s", target_start, block_len)
            return False
        if target_start == start:
            logger.debug("shift_block no-op start=%s", start)
            return False

        logger.debug(
            "shift_block start=%s end=%s target_start=%s block_len=%s",
            start,
            end,
            target_start,
            block_len,
        )
        self.beginResetModel()
        old_positions = list(range(256))
        values = list(range(256))
        self.colors = self._shift_block(self.colors, start, end, target_start)
        self.alphas = self._shift_block(self.alphas, start, end, target_start)
        new_positions = self._shift_block(values, start, end, target_start)
        self.endResetModel()
        self.last_index_remap = {old_idx: new_positions.index(old_idx) for old_idx in old_positions}
        self.palette_changed.emit([c for c in self.colors if c is not None], "shift")
        return True


class PaletteListView(QListView):
    palette_changed = Signal(list)
    selection_changed = Signal()
    # Keep swap decisions confined to the center of a cell to avoid surprise swaps.
    _BEFORE_ZONE_MAX = 0.45
    _AFTER_ZONE_MIN = 0.55

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setViewMode(QListView.IconMode)
        self.setIconSize(QSize(32, 32))
        self.setResizeMode(QListView.ResizeMode.Adjust)
        # Snap mode lets users drag icons while keeping them aligned to the grid.
        self.setMovement(QListView.Movement.Snap)
        self.setSpacing(6)
        self.setWrapping(True)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        # We paint our own indicator so disable Qt's default lines to avoid duplicates.
        self.setDropIndicatorShown(False)
        # Enable multi-selection with Ctrl+Click
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._delegate = PaletteItemDelegate(self)
        self.setItemDelegate(self._delegate)
        self.model_obj = PaletteModel()
        self.setModel(self.model_obj)
        self._last_change_reason = "init"
        self._pending_decision: DropDecision | None = None
        self._drag_hint_decision: DropDecision | None = None
        self._active_drag_row: int | None = None
        
        # Animation support for selected items
        self._animation_value = 0.0
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(50)  # 20 FPS
        self._animation_timer.timeout.connect(self._animate_selection)
        self._animation_timer.start()
        
        logger.debug(
            "PaletteListView dragDropMode=%s movement=%s defaultAction=%s",
            self.dragDropMode(),
            self.movement(),
            self.defaultDropAction(),
        )
        self.model_obj.palette_changed.connect(self._handle_model_change)
        self.doubleClicked.connect(self._edit_index)
    
    def _animate_selection(self) -> None:
        """Animate the selection highlight with a pulsing effect."""
        import math
        self._animation_value += 0.1
        if self._animation_value > 2 * math.pi:
            self._animation_value = 0.0
        # Trigger repaint of selected items only
        for index in self.selectedIndexes():
            self.update(index)

    def drawDropIndicator(self, painter) -> None:  # type: ignore[override]
        # Default indicator is disabled in favor of the custom overlay in paintEvent.
        return

    def set_colors(
        self,
        colors: Sequence[ColorTuple],
        *,
        slots: Sequence[int] | None = None,
        alphas: Sequence[int] | None = None,
        emit_signal: bool = True,
    ) -> None:
        self.model_obj.set_colors(colors, slots=slots, alphas=alphas, reason="refresh", emit_signal=emit_signal)
        if emit_signal:
            self._last_change_reason = "refresh"

    def _edit_index(self, index: QModelIndex) -> None:
        color = self.model_obj.data(index, Qt.UserRole)
        raw_alpha = self.model_obj.data(index, Qt.UserRole + 1)
        alpha = 255 if raw_alpha is None else max(0, min(255, int(raw_alpha)))
        
        if color is None:
            # Empty slot - allow adding a new color
            dialog = QColorDialog(QColor(255, 255, 255, 255), self)
            dialog.setWindowTitle(f"Add color to index {index.row()}")
            dialog.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
            dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            result = dialog.currentColor()
            if not result.isValid():
                return
            new_color = (result.red(), result.green(), result.blue())
            new_alpha = result.alpha()
            logger.debug("PaletteListView add color to empty slot row=%s color=%s alpha=%s", index.row(), new_color, new_alpha)
            self.model_obj.update_color(index.row(), new_color, new_alpha)
            return
        
        if not isinstance(color, tuple):
            return
        
        # Create QColor with current alpha
        start = QColor(*color, alpha)
        dialog = QColorDialog(start, self)
        dialog.setWindowTitle(f"Edit color at index {index.row()}")
        dialog.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
        dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        result = dialog.currentColor()
        if not result.isValid():
            return
        updated = (result.red(), result.green(), result.blue())
        updated_alpha = result.alpha()
        logger.debug("PaletteListView edit row=%s new_color=%s alpha=%s", index.row(), updated, updated_alpha)
        self.model_obj.update_color(index.row(), updated, updated_alpha)

    def _handle_model_change(self, colors: List[ColorTuple], reason: str) -> None:
        self._last_change_reason = reason
        logger.debug("PaletteListView model change reason=%s count=%s", reason, len(colors))
        self.palette_changed.emit(list(colors))

    @property
    def last_change_reason(self) -> str:
        return self._last_change_reason

    def selectionChanged(self, selected: QItemSelection, deselected: QItemSelection) -> None:  # type: ignore[override]
        super().selectionChanged(selected, deselected)
        self.selection_changed.emit()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        """Handle keyboard events including Ctrl+V for pasting hex colors."""
        if event.matches(QKeySequence.StandardKey.Copy):
            self._copy_selected_palette()
            return
        if event.matches(QKeySequence.StandardKey.Paste):
            if self._paste_palette_payload():
                return
            self._paste_hex_color()
            return
        super().keyPressEvent(event)

    def set_show_index_labels(self, enabled: bool) -> None:
        self._delegate.show_index_labels = bool(enabled)
        self.viewport().update()

    def set_show_grid_lines(self, enabled: bool) -> None:
        self._delegate.show_grid_lines = bool(enabled)
        self.viewport().update()

    def set_cell_size(self, size: int) -> None:
        self.model_obj.set_cell_size(size)
        icon = max(8, int(size) - 12)
        self.setIconSize(QSize(icon, icon))
        self.viewport().update()

    def set_swatch_inset(self, inset: int) -> None:
        self._delegate.swatch_inset = max(1, int(inset))
        self.viewport().update()

    def measure_first_row_columns(self, *, sample_limit: int = 128) -> int:
        model = self.model_obj
        total = model.rowCount()
        if total <= 0:
            return 0
        first = self.visualRect(model.index(0))
        if not first.isValid() or first.width() <= 0:
            return 0
        first_y = first.y()
        columns = 1
        max_rows = min(total, max(1, int(sample_limit)))
        for row in range(1, max_rows):
            rect = self.visualRect(model.index(row))
            if not rect.isValid() or rect.width() <= 0:
                continue
            if abs(rect.y() - first_y) <= 1:
                columns += 1
                continue
            if rect.y() > first_y:
                break
        return columns

    def set_local_rows(self, rows: Sequence[int]) -> None:
        self._delegate.local_rows = {int(row) for row in rows if 0 <= int(row) < 256}
        self.viewport().update()

    def set_usage_counts(self, counts: Dict[int, int], *, show_badge: bool = True) -> None:
        normalized: Dict[int, int] = {}
        for row, count in counts.items():
            row_int = int(row)
            if 0 <= row_int < 256:
                normalized[row_int] = max(0, int(count))
        self._delegate.usage_counts = normalized
        self._delegate.show_usage_badge = bool(show_badge)
        self.viewport().update()

    def set_merge_roles(self, source_rows: Sequence[int], destination_row: int | None) -> None:
        self._delegate.merge_source_rows = {int(row) for row in source_rows if 0 <= int(row) < 256}
        self._delegate.merge_destination_row = (
            int(destination_row)
            if destination_row is not None and 0 <= int(destination_row) < 256
            else None
        )
        logger.debug(
            "PaletteListView merge roles source_count=%s destination=%s sources=%s",
            len(self._delegate.merge_source_rows),
            self._delegate.merge_destination_row,
            sorted(self._delegate.merge_source_rows)[:16],
        )
        self.viewport().update()

    def set_merge_candidate_risk(self, risk_map: Dict[int, str]) -> None:
        normalized: Dict[int, str] = {}
        for row, level in risk_map.items():
            row_int = int(row)
            if 0 <= row_int < 256 and level in {"safe", "risky"}:
                normalized[row_int] = level
        self._delegate.merge_candidate_risk = normalized
        logger.debug(
            "PaletteListView merge candidate risk updated rows=%s sample=%s",
            len(normalized),
            dict(list(normalized.items())[:16]),
        )
        self.viewport().update()

    def _copy_selected_palette(self) -> None:
        selected_rows = sorted({idx.row() for idx in self.selectedIndexes()})
        if not selected_rows:
            return
        colors: List[ColorTuple | None] = []
        alphas: List[int] = []
        for row in selected_rows:
            colors.append(self.model_obj.colors[row])
            alphas.append(self.model_obj.alphas[row])
        _set_palette_clipboard(PaletteClipboardPayload(colors=colors, alphas=alphas))
        logger.debug("Palette copy selected_rows=%s count=%s", selected_rows[:20], len(selected_rows))

    def _paste_palette_payload(self) -> bool:
        payload = _get_palette_clipboard()
        if payload is None or not payload.colors:
            return False
        current = self.currentIndex()
        start_row = current.row() if current.isValid() else (self.selectedIndexes()[0].row() if self.selectedIndexes() else 0)
        applied = 0
        for offset, color in enumerate(payload.colors):
            row = start_row + offset
            if row >= 256:
                break
            alpha = payload.alphas[offset] if offset < len(payload.alphas) else 255
            self.model_obj.set_slot(row, color, alpha)
            applied += 1
        if applied:
            self.model_obj.palette_changed.emit([c for c in self.model_obj.colors if c is not None], "paste")
            selection = self.selectionModel()
            if selection:
                selection.clearSelection()
                for row in range(start_row, start_row + applied):
                    idx = self.model_obj.index(row)
                    selection.select(idx, QItemSelectionModel.SelectionFlag.Select)
                self.setCurrentIndex(self.model_obj.index(start_row))
            logger.debug("Palette paste payload start_row=%s applied=%s", start_row, applied)
        return applied > 0
    
    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        """Handle right-click context menu for palette indices."""
        # Import here to access parent window
        from PySide6.QtWidgets import QMenu
        
        index = self.indexAt(event.pos())
        if not index.isValid():
            return
        
        row = index.row()
        color = self.model_obj.colors[row]
        
        menu = QMenu(self)
        
        if color is not None:
            # Action to show sprites using this index
            show_sprites_action = menu.addAction(f"Show Sprites Using Index {row}")
            show_sprites_action.triggered.connect(lambda: self._show_sprites_using_index(row))
        
        menu.exec(event.globalPos())
    
    def _show_sprites_using_index(self, index: int) -> None:
        """Show a dialog listing sprites that use the specified palette index."""
        # We need to access the parent window's sprite records
        # This will be called from the menu action, so we emit a signal
        # that the parent window can connect to
        if not hasattr(self, '_sprites_using_index_signal'):
            # Create a signal on first use
            self._sprites_using_index_signal = Signal(int)
        
        # For simplicity, we'll call a method directly if we can find the window
        window = self.window()
        if hasattr(window, '_show_sprites_using_index_dialog'):
            window._show_sprites_using_index_dialog(index)
    
    def _paste_hex_color(self) -> None:
        """Paste hex color from clipboard to selected palette index."""
        selected_indexes = self.selectedIndexes()
        if not selected_indexes:
            logger.debug("Paste: No palette index selected")
            return
        
        clipboard = QApplication.clipboard()
        text = clipboard.text().strip()
        
        if not text:
            logger.debug("Paste: Clipboard is empty")
            return
        
        # Parse hex color (supports #RRGGBB, RRGGBB, #RGB, RGB formats)
        color = self._parse_hex_color(text)
        if color is None:
            logger.debug(f"Paste: Invalid hex color format: {text}")
            return
        
        # Apply color to first selected index
        index = selected_indexes[0]
        row = index.row()
        self.model_obj.update_color(row, color)
        logger.info(f"Pasted hex color {text} -> RGB{color} to index {row}")
    
    @staticmethod
    def _parse_hex_color(text: str) -> ColorTuple | None:
        """Parse hex color string to RGB tuple."""
        text = text.strip()
        
        # Remove # prefix if present
        if text.startswith('#'):
            text = text[1:]
        
        # Support 3-digit shorthand (#RGB -> #RRGGBB)
        if len(text) == 3:
            text = ''.join([c * 2 for c in text])
        
        # Validate 6-digit hex
        if len(text) != 6:
            return None
        
        try:
            r = int(text[0:2], 16)
            g = int(text[2:4], 16)
            b = int(text[4:6], 16)
            return (r, g, b)
        except ValueError:
            return None

    def _log_drag_event(self, label: str, event, *, extra: str = "") -> None:
        md = event.mimeData() if hasattr(event, "mimeData") else None
        formats = md.formats() if md else []
        if hasattr(event, "position"):
            point = event.position().toPoint()
        elif hasattr(event, "pos"):
            point = event.pos()
        else:
            point = None
        index = self.indexAt(point) if point else QModelIndex()
        point_repr = (point.x(), point.y()) if point else None
        logger.debug(
            "PaletteListView %s action=%s accepted=%s at_row=%s point=%s formats=%s source_is_self=%s %s",
            label,
            event.dropAction() if hasattr(event, "dropAction") else None,
            event.isAccepted(),
            index.row() if index.isValid() else None,
            point_repr,
            list(formats),
            event.source() is self if hasattr(event, "source") else None,
            extra,
        )

    def _event_point(self, event) -> QPoint | None:
        if hasattr(event, "position"):
            return event.position().toPoint()
        if hasattr(event, "pos"):
            return event.pos()
        return None

    def _horizontal_flow(self) -> bool:
        return self.flow() == QListView.Flow.LeftToRight

    def _distance_to_rect(self, point: QPoint, rect: QRect) -> int:
        dx = 0
        if point.x() < rect.left():
            dx = rect.left() - point.x()
        elif point.x() > rect.right():
            dx = point.x() - rect.right()
        dy = 0
        if point.y() < rect.top():
            dy = rect.top() - point.y()
        elif point.y() > rect.bottom():
            dy = point.y() - rect.bottom()
        return dx + dy

    def _compute_drop_decision(self, point: QPoint | None) -> DropDecision | None:
        if point is None:
            return None
        count = self.model_obj.rowCount()
        viewport_rect = self.viewport().rect()
        if viewport_rect.isNull():
            return None
        clamped = QPoint(
            min(max(point.x(), viewport_rect.left()), viewport_rect.right() - 1),
            min(max(point.y(), viewport_rect.top()), viewport_rect.bottom() - 1),
        )
        outside = (clamped.x() != point.x()) or (clamped.y() != point.y())
        horizontal_flow = self._horizontal_flow()
        if count == 0:
            return DropDecision(0, None, "swap", f"empty-list outside={outside}")
        rects: List[tuple[int, QRect]] = []
        for row in range(count):
            index = self.model_obj.index(row)
            cell_rect = self.visualRect(index)
            if cell_rect.isValid():
                rects.append((row, cell_rect))
        if not rects:
            return DropDecision(0, None, "none", "no-rects")
        spacing = self.spacing()
        first_rect = rects[0][1]
        last_rect = rects[-1][1]
        if horizontal_flow:
            if clamped.y() < first_rect.top() - spacing:
                return DropDecision(0, rects[0][0], "before", f"before-first outside={outside}")
            if clamped.y() > last_rect.bottom() + spacing:
                return DropDecision(count, rects[-1][0], "after", f"after-last outside={outside}")
        else:
            if clamped.x() < first_rect.left() - spacing:
                return DropDecision(0, rects[0][0], "before", f"before-first outside={outside}")
            if clamped.x() > last_rect.right() + spacing:
                return DropDecision(count, rects[-1][0], "after", f"after-last outside={outside}")
        index = self.indexAt(clamped)
        if index.isValid():
            rect = self.visualRect(index)
            span = max(1, rect.width() if horizontal_flow else rect.height())
            delta = (clamped.x() - rect.left()) if horizontal_flow else (clamped.y() - rect.top())
            ratio = max(0.0, min(1.0, delta / span))
            before_cutoff = self._BEFORE_ZONE_MAX
            after_cutoff = self._AFTER_ZONE_MIN
            if ratio < before_cutoff:
                insert_row = max(0, min(index.row(), count))
                return DropDecision(insert_row, index.row(), "before", f"index={index.row()} zone=before ratio={ratio:.2f} outside={outside}")
            if ratio > after_cutoff:
                insert_row = max(0, min(index.row() + 1, count))
                return DropDecision(insert_row, index.row(), "after", f"index={index.row()} zone=after ratio={ratio:.2f} outside={outside}")
            return DropDecision(
                index.row(),
                index.row(),
                "swap",
                f"index={index.row()} zone=swap ratio={ratio:.2f} outside={outside}",
                swap_row=index.row(),
            )
        closest_row, closest_rect = min(rects, key=lambda item: self._distance_to_rect(clamped, item[1]))
        if horizontal_flow:
            if clamped.x() < closest_rect.left():
                return DropDecision(closest_row, closest_row, "before", f"gap-left row={closest_row} outside={outside}")
            if clamped.x() > closest_rect.right():
                insert_row = max(0, min(closest_row + 1, count))
                return DropDecision(insert_row, closest_row, "after", f"gap-right row={closest_row} outside={outside}")
            if clamped.y() < closest_rect.top():
                return DropDecision(closest_row, closest_row, "before", f"gap-above row={closest_row} outside={outside}")
            if clamped.y() > closest_rect.bottom():
                insert_row = max(0, min(closest_row + 1, count))
                return DropDecision(insert_row, closest_row, "after", f"gap-below row={closest_row} outside={outside}")
        else:
            if clamped.y() < closest_rect.top():
                return DropDecision(closest_row, closest_row, "before", f"gap-above row={closest_row} outside={outside}")
            if clamped.y() > closest_rect.bottom():
                insert_row = max(0, min(closest_row + 1, count))
                return DropDecision(insert_row, closest_row, "after", f"gap-below row={closest_row} outside={outside}")
            if clamped.x() < closest_rect.left():
                return DropDecision(closest_row, closest_row, "before", f"gap-left row={closest_row} outside={outside}")
            if clamped.x() > closest_rect.right():
                insert_row = max(0, min(closest_row + 1, count))
                return DropDecision(insert_row, closest_row, "after", f"gap-right row={closest_row} outside={outside}")
        insert_row = max(0, min(closest_row + 1, count))
        return DropDecision(insert_row, closest_row, "after", f"fallback row={closest_row} outside={outside}")

    def _decision_debug(self, decision: DropDecision | None) -> str:
        if decision is None:
            return "decision=None"
        return (
            "decision(insert={insert} hint_row={hint} mode={mode} swap={swap} reason={reason})".format(
                insert=decision.insert_row,
                hint=decision.hint_row,
                mode=decision.hint_mode,
                swap=decision.swap_row,
                reason=decision.reason,
            )
        )

    def _update_drag_hint(self, decision: DropDecision | None) -> None:
        if self._drag_hint_decision == decision:
            return
        self._drag_hint_decision = decision
        self.viewport().update()

    def _clear_drag_hint(self) -> None:
        if self._drag_hint_decision is None and self._pending_decision is None:
            return
        self._drag_hint_decision = None
        self._pending_decision = None
        self.viewport().update()

    def _indicator_geometry(self, decision: DropDecision | None) -> tuple[str, QRect | tuple[QPoint, QPoint]] | None:
        if decision is None:
            return None
        viewport_rect = self.viewport().rect()
        count = self.model_obj.rowCount()
        if count == 0 and decision.insert_row == 0:
            size = self.gridSize()
            rect = QRect(QPoint(viewport_rect.left() + 4, viewport_rect.top() + 4), size)
            return "swap", rect.intersected(viewport_rect)
        if count == 0:
            return None
        horizontal_flow = self._horizontal_flow()
        if decision.hint_mode == "swap":
            row = decision.hint_row
            if row is None or not (0 <= row < count):
                return None
            rect = self.visualRect(self.model_obj.index(row))
            return "swap", rect.intersected(viewport_rect)
        if decision.hint_mode not in {"before", "after"}:
            return None
        if decision.insert_row is None:
            return None
        line = self._gap_line_points(decision.insert_row, viewport_rect, horizontal_flow)
        if line is None:
            return None
        return "line", line

    def _gap_line_points(
        self, insert_row: int, viewport_rect: QRect, horizontal_flow: bool
    ) -> tuple[QPoint, QPoint] | None:
        count = self.model_obj.rowCount()
        if count == 0:
            return None
        spacing = max(2, self.spacing())
        if insert_row <= 0:
            first_rect = self.visualRect(self.model_obj.index(0))
            if not first_rect.isValid():
                return None
            if horizontal_flow:
                x = first_rect.left() - spacing // 2
                start = QPoint(x, max(first_rect.top(), viewport_rect.top()))
                end = QPoint(x, min(first_rect.bottom(), viewport_rect.bottom()))
            else:
                y = first_rect.top() - spacing // 2
                start = QPoint(max(first_rect.left(), viewport_rect.left()), y)
                end = QPoint(min(first_rect.right(), viewport_rect.right()), y)
            return start, end
        if insert_row >= count:
            last_rect = self.visualRect(self.model_obj.index(count - 1))
            if not last_rect.isValid():
                return None
            if horizontal_flow:
                x = last_rect.right() + spacing // 2
                start = QPoint(x, max(last_rect.top(), viewport_rect.top()))
                end = QPoint(x, min(last_rect.bottom(), viewport_rect.bottom()))
            else:
                y = last_rect.bottom() + spacing // 2
                start = QPoint(max(last_rect.left(), viewport_rect.left()), y)
                end = QPoint(min(last_rect.right(), viewport_rect.right()), y)
            return start, end
        prev_rect = self.visualRect(self.model_obj.index(insert_row - 1))
        next_rect = self.visualRect(self.model_obj.index(insert_row))
        if not prev_rect.isValid() or not next_rect.isValid():
            return None
        if horizontal_flow:
            same_row = abs(prev_rect.center().y() - next_rect.center().y()) <= max(prev_rect.height(), next_rect.height())
            if same_row:
                x = int(round((prev_rect.right() + next_rect.left()) / 2))
                top = max(min(prev_rect.top(), next_rect.top()), viewport_rect.top())
                bottom = min(max(prev_rect.bottom(), next_rect.bottom()), viewport_rect.bottom())
                return QPoint(x, top), QPoint(x, bottom)
            y = int(round((prev_rect.bottom() + next_rect.top()) / 2))
            left = viewport_rect.left()
            right = viewport_rect.right()
            return QPoint(left, y), QPoint(right, y)
        same_column = abs(prev_rect.center().x() - next_rect.center().x()) <= max(prev_rect.width(), next_rect.width())
        if same_column:
            y = int(round((prev_rect.bottom() + next_rect.top()) / 2))
            left = max(min(prev_rect.left(), next_rect.left()), viewport_rect.left())
            right = min(max(prev_rect.right(), next_rect.right()), viewport_rect.right())
            return QPoint(left, y), QPoint(right, y)
        x = int(round((prev_rect.right() + next_rect.left()) / 2))
        top = viewport_rect.top()
        bottom = viewport_rect.bottom()
        return QPoint(x, top), QPoint(x, bottom)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        self._log_drag_event("dragEnter(before)", event)
        super().dragEnterEvent(event)
        self._log_drag_event("dragEnter(after)", event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        point = self._event_point(event)
        decision = self._compute_drop_decision(point)
        extra = self._decision_debug(decision)
        self._log_drag_event("dragMove(before)", event, extra=extra)
        super().dragMoveEvent(event)
        if decision and (decision.insert_row is not None or decision.swap_row is not None):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()
        self._pending_decision = decision
        self._update_drag_hint(decision)
        self._log_drag_event("dragMove(after)", event, extra=extra)

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        self._log_drag_event("dragLeave(before)", event)
        super().dragLeaveEvent(event)
        self._log_drag_event("dragLeave(after)", event)
        self._clear_drag_hint()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        point = self._event_point(event)
        decision = self._compute_drop_decision(point) or self._pending_decision
        if decision is None:
            count = self.model_obj.rowCount()
            decision = DropDecision(count, count - 1 if count else None, "after", "fallback:end")
        selected_rows = sorted({idx.row() for idx in self.selectedIndexes()})
        source_row = self._active_drag_row if self._active_drag_row is not None else (selected_rows[0] if selected_rows else None)
        extra = f"{self._decision_debug(decision)} selected={selected_rows} drag_row={self._active_drag_row}"
        self._log_drag_event("drop(before)", event, extra=extra)
        moved = False
        new_selection_row: int | None = None
        if source_row is not None:
            if decision.swap_row is not None and decision.swap_row != source_row:
                moved = self.model_obj.swap_rows(source_row, decision.swap_row)
                if moved:
                    new_selection_row = decision.swap_row
            else:
                target_row = decision.insert_row if decision.insert_row is not None else self.model_obj.rowCount()
                if target_row not in (source_row, source_row + 1):
                    moved = self.model_obj.moveRows(
                        QModelIndex(), source_row, 1, QModelIndex(), target_row
                    )
                    if moved:
                        new_selection_row = target_row if target_row <= source_row else target_row - 1
                else:
                    logger.debug(
                        "PaletteListView drop no-op source=%s target=%s",
                        source_row,
                        target_row,
                    )
        else:
            logger.debug("PaletteListView drop ignored: no selection")
        event.setDropAction(Qt.DropAction.MoveAction)
        event.accept()
        self._log_drag_event("drop(after)", event, extra=f"{extra} moved={moved}")
        if moved and new_selection_row is not None:
            self._select_row(new_selection_row)
        self._clear_drag_hint()
        self._active_drag_row = None
        self._pending_decision = None
        
        # Force viewport update to clear any lingering drag visuals
        self.viewport().update()
        self.setState(QAbstractItemView.State.NoState)

    def startDrag(self, supportedActions: Qt.DropActions) -> None:  # type: ignore[override]
        rows = [idx.row() for idx in self.selectedIndexes()]
        logger.debug(
            "PaletteListView startDrag rows=%s actions=%s (%s)",
            rows,
            getattr(supportedActions, "name", supportedActions),
            type(supportedActions).__name__,
        )
        self._active_drag_row = rows[0] if rows else None
        super().startDrag(supportedActions)
        
        # Clean up after drag completes (whether dropped or cancelled)
        logger.debug("PaletteListView startDrag completed, cleaning up state")
        self._active_drag_row = None
        self._pending_decision = None
        self._clear_drag_hint()
        self.setState(QAbstractItemView.State.NoState)
        self.viewport().update()

    def _select_row(self, row: int | None) -> None:
        if row is None:
            return
        if not (0 <= row < self.model_obj.rowCount()):
            return
        selection = self.selectionModel()
        if not selection:
            return
        index = self.model_obj.index(row)
        selection.setCurrentIndex(index, QItemSelectionModel.ClearAndSelect)
        self.setCurrentIndex(index)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        geometry = self._indicator_geometry(self._drag_hint_decision)
        if not geometry:
            return
        mode, payload = geometry
        painter = QPainter(self.viewport())
        if mode == "swap":
            rect = payload if isinstance(payload, QRect) else QRect()
            pen = QPen(QColor(255, 255, 255, 200))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 255, 255, 60))
            painter.drawRoundedRect(rect, 6, 6)
        else:
            start, end = payload if isinstance(payload, tuple) else (QPoint(), QPoint())
            pen = QPen(QColor(255, 255, 255, 230))
            pen.setWidth(4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(start, end)


class PreviewCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._overlay_layers: List[tuple[int, QRegion, QColor]] = []
        self._placeholder_text = "Select an image to preview"
        self.setMinimumSize(200, 200)

    def set_display_pixmap(self, pixmap: QPixmap | None) -> None:
        if pixmap is None:
            self._pixmap = None
            self.update()
            return
        if self._pixmap is not None and self._pixmap.cacheKey() == pixmap.cacheKey():
            return
        self._pixmap = pixmap
        self.update()

    def _pixmap_top_left(self) -> QPoint:
        pixmap = self._pixmap
        if pixmap is None:
            return QPoint(0, 0)
        return QPoint((self.width() - pixmap.width()) // 2, (self.height() - pixmap.height()) // 2)

    def _overlay_bounds(self, layers: Sequence[tuple[int, QRegion, QColor]]) -> QRect:
        pixmap = self._pixmap
        if pixmap is None or not layers:
            return QRect()
        origin = self._pixmap_top_left()
        bounds = QRect()
        for _layer_key, region, _color in layers:
            region_rect = region.boundingRect()
            if region_rect.isEmpty():
                continue
            translated = QRect(region_rect)
            translated.translate(origin)
            bounds = translated if bounds.isNull() else bounds.united(translated)
        return bounds

    def set_overlay_layers(self, layers: Sequence[tuple[int, QRegion, QColor]]) -> None:
        previous_layers = self._overlay_layers
        normalized: List[tuple[int, QRegion, QColor]] = []
        for layer_key, region, color in layers:
            if region.isEmpty():
                continue
            normalized.append((int(layer_key), region, QColor(color)))
        self._overlay_layers = normalized
        if self._pixmap is None:
            self.update()
            return
        previous_bounds = self._overlay_bounds(previous_layers)
        current_bounds = self._overlay_bounds(self._overlay_layers)
        dirty = previous_bounds.united(current_bounds)
        if dirty.isNull() or dirty.isEmpty():
            self.update()
            return
        self.update(dirty.adjusted(-2, -2, 2, 2).intersected(self.rect()))

    def paintEvent(self, event) -> None:  # type: ignore[override]
        event_rect = event.rect() if hasattr(event, "rect") else self.rect()
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.fillRect(event_rect, Qt.GlobalColor.transparent)

        if self._pixmap is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder_text)
            painter.end()
            return

        pixmap = self._pixmap
        origin = self._pixmap_top_left()
        px = origin.x()
        py = origin.y()
        bounds = QRect(px, py, pixmap.width(), pixmap.height())
        exposed = bounds.intersected(event_rect)
        if exposed.isEmpty():
            painter.end()
            return

        source = QRect(exposed.x() - px, exposed.y() - py, exposed.width(), exposed.height())
        painter.drawPixmap(exposed, pixmap, source)

        if self._overlay_layers:
            for _layer_key, region, color in self._overlay_layers:
                translated = QRegion(region)
                translated.translate(px, py)
                translated = translated.intersected(QRegion(bounds))
                translated = translated.intersected(QRegion(exposed))
                if translated.isEmpty():
                    continue
                painter.save()
                painter.setClipRegion(translated)
                painter.fillRect(exposed, color)
                painter.restore()

        painter.end()


class PreviewPane(QWidget):
    zoomChanged = Signal(float)
    drag_offset_changed = Signal(int, int)  # (dx, dy) in pixels
    drag_started = Signal()
    drag_finished = Signal(int, int)  # total delta for this drag interaction
    panning_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.title = QLabel("Preview")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label = PreviewCanvas()
        self.scroll_area = QScrollArea()
        self._gpu_viewport_enabled = False
        if QOpenGLWidget is not None:
            try:
                self.scroll_area.setViewport(QOpenGLWidget())
                self._gpu_viewport_enabled = True
            except Exception:  # noqa: BLE001
                logger.debug("PreviewPane OpenGL viewport unavailable; using default viewport", exc_info=True)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.viewport().installEventFilter(self)
        self.image_label.installEventFilter(self)
        self._source_pixmap: QPixmap | None = None
        self._static_source_pixmap: QPixmap | None = None
        self._scaled_base_pixmap: QPixmap | None = None
        self._scaled_base_size = QSize(0, 0)
        self._scaled_static_pixmap: QPixmap | None = None
        self._scaled_static_size = QSize(0, 0)
        self._overlay_layers: List[tuple[int, QRegion, QColor]] = []
        self._overlay_source_size = QSize(0, 0)
        self._overlay_scaled_region_cache: Dict[tuple[int, int, int, int, int], QRegion] = {}
        self._backdrop_pixmap: QPixmap | None = None
        self._backdrop_key: tuple[int, int, int | None, int | None, int | None, int | None] | None = None
        self._checker_tile_pixmap: QPixmap | None = None
        self._drag_backdrop_rgba: tuple[int, int, int, int] | None = None
        self._zoom = 1.0
        self._min_zoom = 0.25
        self._max_zoom = 8.0
        self._transform_mode = Qt.TransformationMode.FastTransformation
        self._pan_active = False
        self._pan_margin_px = 96
        self._pan_last_pos = QPointF(0, 0)
        self._pan_pending_dx = 0.0
        self._pan_pending_dy = 0.0
        self._pan_apply_timer = QTimer(self)
        self._pan_apply_timer.setInterval(8)
        self._pan_apply_timer.timeout.connect(self._flush_pan_delta)
        self._drag_mode = False
        self._drag_active = False
        self._drag_start_pos = QPointF(0, 0)
        self._drag_accum_dx = 0
        self._drag_accum_dy = 0
        self._drag_visual_dx = 0
        self._drag_visual_dy = 0
        layout.addWidget(self.title)
        layout.addWidget(self.scroll_area, 1)
    
    def set_drag_mode(self, enabled: bool) -> None:
        """Enable or disable drag mode for repositioning sprites."""
        self._drag_mode = enabled
        if enabled:
            self.image_label.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.image_label.unsetCursor()
            self._drag_active = False

    def set_pixmap(self, pixmap: QPixmap | None, *, reset_zoom: bool = False) -> None:
        if pixmap is None:
            self.image_label.set_display_pixmap(None)
            self.image_label.set_overlay_layers([])
            self._source_pixmap = None
            self._static_source_pixmap = None
            self._scaled_base_pixmap = None
            self._scaled_base_size = QSize(0, 0)
            self._scaled_static_pixmap = None
            self._scaled_static_size = QSize(0, 0)
            self._overlay_layers = []
            self._overlay_source_size = QSize(0, 0)
            self._overlay_scaled_region_cache = {}
            self._backdrop_pixmap = None
            self._backdrop_key = None
            self._drag_visual_dx = 0
            self._drag_visual_dy = 0
            if reset_zoom:
                self._zoom = 1.0
            self.image_label.resize(QSize(200, 200))
            return

        if self._source_pixmap is not None and self._source_pixmap.cacheKey() == pixmap.cacheKey():
            if reset_zoom:
                self._zoom = 1.0
                self._apply_scaled_pixmap()
            return

        self._source_pixmap = pixmap
        self._scaled_base_pixmap = None
        self._scaled_base_size = QSize(0, 0)
        self._overlay_scaled_region_cache = {}
        self._backdrop_pixmap = None
        self._backdrop_key = None
        self._drag_visual_dx = 0
        self._drag_visual_dy = 0
        if reset_zoom:
            self._zoom = 1.0
        self._apply_scaled_pixmap()

    def set_static_pixmap(self, pixmap: QPixmap | None) -> None:
        if pixmap is None:
            if self._static_source_pixmap is None:
                return
            self._static_source_pixmap = None
            self._scaled_static_pixmap = None
            self._scaled_static_size = QSize(0, 0)
            self._backdrop_pixmap = None
            self._backdrop_key = None
            self._apply_scaled_pixmap()
            return

        if self._static_source_pixmap is not None and self._static_source_pixmap.cacheKey() == pixmap.cacheKey():
            return
        self._static_source_pixmap = pixmap
        self._scaled_static_pixmap = None
        self._scaled_static_size = QSize(0, 0)
        self._backdrop_pixmap = None
        self._backdrop_key = None
        self._apply_scaled_pixmap()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        self._apply_scaled_pixmap()
        super().resizeEvent(event)

    def set_overlay_layers(self, layers: Sequence[tuple[int, QRegion, QColor]], source_size: QSize) -> None:
        normalized: List[tuple[int, QRegion, QColor]] = []
        for layer_key, region, color in layers:
            if region.isEmpty():
                continue
            normalized.append((int(layer_key), region, QColor(color)))
        self._overlay_layers = normalized
        if self._overlay_source_size != QSize(source_size):
            self._overlay_scaled_region_cache = {}
        self._overlay_source_size = QSize(source_size)
        self._push_overlay_layers_to_canvas()

    def _scaled_overlay_region(self, layer_key: int, region: QRegion, target_size: QSize) -> QRegion:
        source_w = max(1, int(self._overlay_source_size.width()))
        source_h = max(1, int(self._overlay_source_size.height()))
        target_w = max(1, int(target_size.width()))
        target_h = max(1, int(target_size.height()))
        cache_key = (int(layer_key), source_w, source_h, target_w, target_h)
        cached = self._overlay_scaled_region_cache.get(cache_key)
        if cached is not None:
            return cached

        sx = target_w / source_w
        sy = target_h / source_h
        scaled = QRegion()
        for rect in region:
            x = int(rect.x() * sx)
            y = int(rect.y() * sy)
            x2 = int((rect.x() + rect.width()) * sx)
            y2 = int((rect.y() + rect.height()) * sy)
            w = max(1, x2 - x)
            h = max(1, y2 - y)
            scaled = scaled.united(QRegion(x, y, w, h))

        self._overlay_scaled_region_cache[cache_key] = scaled
        return scaled

    def _push_overlay_layers_to_canvas(self) -> None:
        if self._drag_active or not self._overlay_layers:
            self.image_label.set_overlay_layers([])
            return
        if self._overlay_source_size.isEmpty() or self._scaled_base_pixmap is None:
            self.image_label.set_overlay_layers([])
            return

        target_size = self._scaled_base_pixmap.size()
        display_overlay_layers: List[tuple[int, QRegion, QColor]] = []
        for layer_key, source_region, color in self._overlay_layers:
            scaled_region = self._scaled_overlay_region(layer_key, source_region, target_size)
            if scaled_region.isEmpty():
                continue
            display_overlay_layers.append((layer_key, scaled_region, color))
        self.image_label.set_overlay_layers(display_overlay_layers)

    def eventFilter(self, obj, event):  # type: ignore[override]
        etype = event.type()
        if obj in (self.scroll_area.viewport(), self.image_label):
            if etype == QEvent.Type.Wheel and isinstance(event, QWheelEvent):
                if self._handle_wheel_zoom(event):
                    event.accept()
                    return True
            if isinstance(event, QMouseEvent):
                # Handle drag mode for sprite repositioning
                if self._drag_mode:
                    if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:
                        self._start_drag(event)
                        return True
                    if etype == QEvent.Type.MouseMove and self._drag_active:
                        self._drag_sprite(event)
                        return True
                    if etype == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.RightButton:
                        self._end_drag()
                        return True
                
                # Handle pan mode (left or middle mouse button when not in drag mode)
                if etype == QEvent.Type.MouseButtonPress and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
                    self._start_pan(event)
                    return True
                if etype == QEvent.Type.MouseMove and self._pan_active:
                    self._pan(event)
                    return True
                if etype == QEvent.Type.MouseButtonRelease and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
                    self._end_pan()
                    return True
            if self._pan_active and etype in (QEvent.Type.Leave, QEvent.Type.HoverLeave):
                self._end_pan()
                return True
        return super().eventFilter(obj, event)
    
    def _start_drag(self, event: QMouseEvent) -> None:
        """Start dragging sprite for repositioning."""
        self._drag_active = True
        self._drag_accum_dx = 0
        self._drag_accum_dy = 0
        self._drag_visual_dx = 0
        self._drag_visual_dy = 0
        self.drag_started.emit()
        if hasattr(event, 'position'):
            self._drag_start_pos = event.position()
        else:
            self._drag_start_pos = QPointF(event.pos())
    
    def _drag_sprite(self, event: QMouseEvent) -> None:
        """Handle sprite dragging to reposition."""
        if not self._drag_active:
            return
        
        if hasattr(event, 'position'):
            current_pos = event.position()
        else:
            current_pos = QPointF(event.pos())
        
        delta = current_pos - self._drag_start_pos
        
        # Emit offset change in actual pixels (accounting for zoom)
        if self._zoom > 0:
            dx = int(delta.x() / self._zoom)
            dy = int(delta.y() / self._zoom)
            
            if dx != 0 or dy != 0:
                self._drag_accum_dx += dx
                self._drag_accum_dy += dy
                self._drag_visual_dx += dx
                self._drag_visual_dy += dy
                self.drag_offset_changed.emit(dx, dy)
                self._apply_scaled_pixmap()
                self._drag_start_pos = current_pos
    
    def _end_drag(self) -> None:
        """End dragging sprite."""
        self._drag_active = False
        self.drag_finished.emit(self._drag_accum_dx, self._drag_accum_dy)
        self._drag_accum_dx = 0
        self._drag_accum_dy = 0

    def is_dragging(self) -> bool:
        return self._drag_active

    def current_zoom(self) -> float:
        return float(self._zoom)

    def wheelEvent(self, event):  # type: ignore[override]
        if not isinstance(event, QWheelEvent) or not self._handle_wheel_zoom(event):
            super().wheelEvent(event)

    def _apply_scaled_pixmap(self) -> None:
        if not self._source_pixmap:
            return
        target_width = max(1, int(self._source_pixmap.width() * self._zoom))
        target_height = max(1, int(self._source_pixmap.height() * self._zoom))
        target_size = QSize(target_width, target_height)
        if self._scaled_base_pixmap is None or self._scaled_base_size != target_size:
            self._scaled_base_pixmap = self._source_pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                self._transform_mode,
            )
            self._scaled_base_size = target_size

        static_pixmap: QPixmap | None = None
        if self._static_source_pixmap is not None:
            if self._scaled_static_pixmap is None or self._scaled_static_size != target_size:
                self._scaled_static_pixmap = self._static_source_pixmap.scaled(
                    target_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    self._transform_mode,
                )
                self._scaled_static_size = target_size
            static_pixmap = self._scaled_static_pixmap

        display_pixmap = self._scaled_base_pixmap
        if display_pixmap is None:
            return

        use_solid_drag_backdrop = self._drag_active and self._drag_backdrop_rgba is not None
        backdrop = self._ensure_backdrop(display_pixmap.size(), solid_rgba=self._drag_backdrop_rgba if use_solid_drag_backdrop else None)

        if self._drag_active and (self._drag_visual_dx != 0 or self._drag_visual_dy != 0):
            shifted = QPixmap(display_pixmap.size())
            shifted.fill(Qt.GlobalColor.transparent)
            painter = QPainter(shifted)
            offset_x = int(self._drag_visual_dx * self._zoom)
            offset_y = int(self._drag_visual_dy * self._zoom)
            painter.drawPixmap(offset_x, offset_y, display_pixmap)
            painter.end()
            display_pixmap = shifted

        if backdrop is not None:
            composed = QPixmap(backdrop)
            painter = QPainter(composed)
            if static_pixmap is not None:
                painter.drawPixmap(0, 0, static_pixmap)
            painter.drawPixmap(0, 0, display_pixmap)
            painter.end()
            display_pixmap = composed
        elif static_pixmap is not None:
            composed = QPixmap(display_pixmap.size())
            composed.fill(Qt.GlobalColor.transparent)
            painter = QPainter(composed)
            painter.drawPixmap(0, 0, static_pixmap)
            painter.drawPixmap(0, 0, display_pixmap)
            painter.end()
            display_pixmap = composed

        self.image_label.set_display_pixmap(display_pixmap)
        self._push_overlay_layers_to_canvas()
        viewport_size = self.scroll_area.viewport().size()
        padded_width = max(display_pixmap.width(), viewport_size.width() + (self._pan_margin_px * 2))
        padded_height = max(display_pixmap.height(), viewport_size.height() + (self._pan_margin_px * 2))
        self.image_label.resize(QSize(max(1, padded_width), max(1, padded_height)))

    def _ensure_backdrop(self, size: QSize, solid_rgba: tuple[int, int, int, int] | None = None) -> QPixmap | None:
        if size.width() <= 0 or size.height() <= 0:
            return None
        key = (
            int(size.width()),
            int(size.height()),
            None if solid_rgba is None else int(solid_rgba[0]),
            None if solid_rgba is None else int(solid_rgba[1]),
            None if solid_rgba is None else int(solid_rgba[2]),
            None if solid_rgba is None else int(solid_rgba[3]),
        )
        if self._backdrop_pixmap is not None and self._backdrop_key == key:
            return self._backdrop_pixmap

        backdrop = QPixmap(size)
        painter = QPainter(backdrop)
        if solid_rgba is not None:
            painter.fillRect(QRect(0, 0, size.width(), size.height()), QColor(*solid_rgba))
        else:
            checker_tile = self._checker_tile_pixmap
            if checker_tile is None:
                checker_tile = QPixmap(16, 16)
                tile_painter = QPainter(checker_tile)
                light = QColor(200, 200, 200, 255)
                dark = QColor(150, 150, 150, 255)
                tile_painter.fillRect(QRect(0, 0, 16, 16), light)
                tile_painter.fillRect(QRect(8, 0, 8, 8), dark)
                tile_painter.fillRect(QRect(0, 8, 8, 8), dark)
                tile_painter.end()
                self._checker_tile_pixmap = checker_tile
            painter.drawTiledPixmap(QRect(0, 0, size.width(), size.height()), checker_tile)
        painter.end()
        self._backdrop_pixmap = backdrop
        self._backdrop_key = key
        return backdrop

    def set_drag_backdrop_rgba(self, rgba: tuple[int, int, int, int] | None) -> None:
        self._drag_backdrop_rgba = rgba
        self._backdrop_pixmap = None
        self._backdrop_key = None

    def set_scaling_mode(self, mode: Qt.TransformationMode) -> None:
        if self._transform_mode == mode:
            return
        self._transform_mode = mode
        self._scaled_base_pixmap = None
        self._scaled_base_size = QSize(0, 0)
        self._backdrop_pixmap = None
        self._backdrop_key = None
        self._apply_scaled_pixmap()

    def _handle_wheel_zoom(self, event: QWheelEvent) -> bool:
        if self._source_pixmap is None:
            return False
        delta = event.angleDelta().y()
        if delta == 0:
            return False
        factor = 1.1 if delta > 0 else 0.9
        new_zoom = max(self._min_zoom, min(self._max_zoom, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 0.01:
            return True
        self._zoom = new_zoom
        self.zoomChanged.emit(self._zoom)
        self._apply_scaled_pixmap()
        return True

    def _start_pan(self, event: QMouseEvent) -> None:
        if self._source_pixmap is None:
            return
        self._pan_active = True
        self.panning_changed.emit(True)
        self._pan_pending_dx = 0.0
        self._pan_pending_dy = 0.0
        target_interval = 8 if self._zoom <= 2.0 else 12 if self._zoom <= 4.0 else 16
        self._pan_apply_timer.setInterval(target_interval)
        if not self._pan_apply_timer.isActive():
            self._pan_apply_timer.start()
        self._pan_last_pos = event.globalPosition()
        self.scroll_area.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)

    def _pan(self, event: QMouseEvent) -> None:
        if not self._pan_active:
            return
        delta = event.globalPosition() - self._pan_last_pos
        self._pan_last_pos = event.globalPosition()
        self._pan_pending_dx += float(delta.x())
        self._pan_pending_dy += float(delta.y())

    def _flush_pan_delta(self) -> None:
        if not self._pan_active:
            if self._pan_apply_timer.isActive():
                self._pan_apply_timer.stop()
            return
        dx = int(self._pan_pending_dx)
        dy = int(self._pan_pending_dy)
        if dx == 0 and dy == 0:
            return
        self._pan_pending_dx -= float(dx)
        self._pan_pending_dy -= float(dy)
        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()
        if dx:
            hbar.setValue(int(max(hbar.minimum(), min(hbar.maximum(), hbar.value() - dx))))
        if dy:
            vbar.setValue(int(max(vbar.minimum(), min(vbar.maximum(), vbar.value() - dy))))

    def _end_pan(self) -> None:
        if not self._pan_active:
            return
        self._flush_pan_delta()
        self._pan_apply_timer.stop()
        self._pan_pending_dx = 0.0
        self._pan_pending_dy = 0.0
        self._pan_active = False
        self.panning_changed.emit(False)
        self.scroll_area.viewport().unsetCursor()


class LoadedImagesPanel(QWidget):
    def __init__(
        self,
        on_selection_change,
        on_browser_change: Callable[[], None] | None = None,
        on_browser_settings_change: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_browser_change = on_browser_change
        self._on_browser_settings_change = on_browser_settings_change
        self._browser_change_timer = QTimer(self)
        self._browser_change_timer.setSingleShot(True)
        self._browser_change_timer.setInterval(_BROWSER_DEBOUNCE_MEDIUM_MS)
        self._browser_change_timer.timeout.connect(self._emit_browser_change)
        self._zoom_drag_active = False
        self._pending_zoom_drag_commit = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        mode_row.addWidget(QLabel("Load Mode:"))
        self.load_mode_combo = QComboBox()
        self.load_mode_combo.addItem("Detect Indexes", "detect")
        self.load_mode_combo.addItem("Preserve Indexes", "preserve")
        self.load_mode_combo.setToolTip(
            "Detect Indexes: build a unified palette across newly loaded sprites.\n"
            "Preserve Indexes: keep each newly loaded indexed PNG palette/index mapping as-is.\n"
            "This setting applies only when importing new sprites."
        )
        mode_row.addWidget(self.load_mode_combo, 1)
        layout.addLayout(mode_row)

        browser_row = QHBoxLayout()
        browser_row.setSpacing(4)
        browser_row.addWidget(QLabel("View:"))
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItem("List", "list")
        self.display_mode_combo.addItem("Thumbnails", "thumbnails")
        browser_row.addWidget(self.display_mode_combo)

        browser_row.addWidget(QLabel("Sort:"))
        self.sort_mode_combo = QComboBox()
        self.sort_mode_combo.addItem("Added", "added")
        self.sort_mode_combo.addItem("Name", "name")
        self.sort_mode_combo.addItem("Group", "group")
        self.sort_mode_combo.addItem("Path", "path")
        browser_row.addWidget(self.sort_mode_combo)

        browser_row.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Name/path (Ctrl+F, F3)")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setMinimumWidth(180)
        browser_row.addWidget(self.search_edit)

        self.group_marker_mode_combo = QComboBox()
        self.group_marker_mode_combo.addItem("Text Color", "text")
        self.group_marker_mode_combo.addItem("Colored Square", "square")
        self.group_marker_mode_combo.addItem("Text + Square", "both")

        self.group_square_thickness_spin = QSpinBox()
        self.group_square_thickness_spin.setRange(1, 10)
        self.group_square_thickness_spin.setValue(3)
        self.group_square_thickness_spin.setFixedWidth(72)
        self.group_square_thickness_spin.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.group_square_padding_spin = QSpinBox()
        self.group_square_padding_spin.setRange(0, 16)
        self.group_square_padding_spin.setValue(1)
        self.group_square_padding_spin.setFixedWidth(72)
        self.group_square_padding_spin.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.group_square_fill_alpha_spin = QSpinBox()
        self.group_square_fill_alpha_spin.setRange(0, 255)
        self.group_square_fill_alpha_spin.setValue(175)
        self.group_square_fill_alpha_spin.setFixedWidth(72)
        self.group_square_fill_alpha_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.group_square_fill_alpha_spin.setToolTip("Colored square fill opacity (0-255)")

        self.scroll_speed_spin = QSpinBox()
        self.scroll_speed_spin.setRange(1, 8)
        self.scroll_speed_spin.setValue(3)
        self.scroll_speed_spin.setFixedWidth(72)
        self.scroll_speed_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.scroll_speed_spin.setToolTip("Thumbnail wheel scroll speed multiplier")

        browser_row.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(32, 240)
        self.zoom_slider.setSingleStep(4)
        self.zoom_slider.setPageStep(12)
        self.zoom_slider.setValue(64)
        self.zoom_slider.setFixedWidth(120)
        browser_row.addWidget(self.zoom_slider)
        self.zoom_value_label = QLabel("64")
        self.zoom_value_label.setMinimumWidth(30)
        browser_row.addWidget(self.zoom_value_label)

        self.float_groups_button = QPushButton("Float Groups")
        self.float_groups_button.setCheckable(True)
        self.float_groups_button.setToolTip("Open Groups panel in a floating window")
        browser_row.addWidget(self.float_groups_button)

        self.float_sprites_button = QPushButton("Float Sprites")
        self.float_sprites_button.setCheckable(True)
        self.float_sprites_button.setToolTip("Open Sprites panel in a floating window")
        browser_row.addWidget(self.float_sprites_button)

        browser_row.addStretch(1)
        self._zoom_by_view: Dict[str, int] = {
            "list": int(self.zoom_slider.value()),
            "thumbnails": int(self.zoom_slider.value()),
        }
        self._last_view_mode = "list"
        self._group_panel_floating = False
        self._sprite_panel_floating = False

        self.group_float_dialog = QDialog(self)
        self.group_float_dialog.setWindowTitle("Groups")
        self.group_float_dialog.setModal(False)
        self.group_float_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.group_float_dialog.installEventFilter(self)
        self.group_float_dialog_layout = QVBoxLayout()
        self.group_float_dialog.setLayout(self.group_float_dialog_layout)
        self.group_float_dialog_layout.setContentsMargins(4, 4, 4, 4)
        self.group_float_dialog_layout.setSpacing(4)

        self.sprite_float_dialog = QDialog(self)
        self.sprite_float_dialog.setWindowTitle("Sprites")
        self.sprite_float_dialog.setModal(False)
        self.sprite_float_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.sprite_float_dialog.installEventFilter(self)
        self.sprite_float_dialog_layout = QVBoxLayout()
        self.sprite_float_dialog.setLayout(self.sprite_float_dialog_layout)
        self.sprite_float_dialog_layout.setContentsMargins(4, 4, 4, 4)
        self.sprite_float_dialog_layout.setSpacing(4)

        self.browser_settings_dialog = QDialog(self)
        self.browser_settings_dialog.setWindowTitle("Sprite Browser Settings")
        self.browser_settings_dialog.setModal(False)
        settings_root = QVBoxLayout(self.browser_settings_dialog)
        settings_root.setContentsMargins(8, 8, 8, 8)
        settings_root.setSpacing(6)
        settings_form = QFormLayout()
        settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_form.addRow("Group Mark:", self.group_marker_mode_combo)
        settings_form.addRow("Border:", self.group_square_thickness_spin)
        settings_form.addRow("Pad:", self.group_square_padding_spin)
        settings_form.addRow("Fill:", self.group_square_fill_alpha_spin)
        settings_form.addRow("Scroll:", self.scroll_speed_spin)
        settings_root.addLayout(settings_form)
        settings_buttons = QHBoxLayout()
        settings_buttons.addStretch(1)
        self.browser_settings_close_button = QPushButton("Close")
        settings_buttons.addWidget(self.browser_settings_close_button)
        settings_root.addLayout(settings_buttons)
        
        self.clear_button = QPushButton("Clear Sprites")
        self.clear_button.setToolTip("Remove all currently loaded sprites from the workspace")
        self.clear_button.setMaximumHeight(24)
        self.count_label = QLabel("No files loaded")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.project_info_label = QLabel("Workspace: Unsaved")
        self.project_info_label.setWordWrap(True)
        self.project_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.project_info_label.setToolTip("No active project")

        group_label = QLabel("Groups")
        group_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.group_hint_label = QLabel("Tip: Hover group to preview members • Double-click to rename")
        self.group_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.group_hint_label.setStyleSheet("color: #9aa0a6; font-size: 10px;")
        self.group_list = QListWidget()
        self.group_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.group_list.setToolTip("Loaded sprite groups")
        self.group_list.setMouseTracking(True)

        group_actions = QHBoxLayout()
        self.new_group_button = QPushButton("New Group")
        self.assign_group_button = QPushButton("Assign")
        self.group_color_button = QPushButton("Color")
        self.rename_group_button = QPushButton("Rename")
        self.detach_group_button = QPushButton("Detach")
        self.auto_group_button = QPushButton("Auto")
        self.new_group_button.setToolTip("Create a new group from selected sprites")
        self.assign_group_button.setToolTip("Assign selected sprites to selected group")
        self.group_color_button.setToolTip("Customize selected group color")
        self.rename_group_button.setToolTip("Rename selected group")
        self.detach_group_button.setToolTip("Detach selected sprites into individual groups")
        self.auto_group_button.setToolTip("Auto-assign selected sprites by signature")
        group_actions.addWidget(self.new_group_button)
        group_actions.addWidget(self.assign_group_button)
        group_actions.addWidget(self.group_color_button)
        group_actions.addWidget(self.rename_group_button)
        group_actions.addWidget(self.detach_group_button)
        group_actions.addWidget(self.auto_group_button)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.setMovement(QListView.Movement.Static)
        self.list_widget.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list_widget.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        self.group_section = QWidget()
        group_section_layout = QVBoxLayout(self.group_section)
        group_section_layout.setContentsMargins(0, 0, 0, 0)
        group_section_layout.setSpacing(4)

        self.sprite_section = QWidget()
        sprite_section_layout = QVBoxLayout(self.sprite_section)
        sprite_section_layout.setContentsMargins(0, 0, 0, 0)
        sprite_section_layout.setSpacing(4)

        group_section_layout.addWidget(group_label)
        group_section_layout.addWidget(self.group_hint_label)
        group_section_layout.addWidget(self.group_list, 1)
        group_section_layout.addLayout(group_actions)

        sprite_label = QLabel("Sprites")
        sprite_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sprite_section_layout.addWidget(sprite_label)
        sprite_section_layout.addLayout(browser_row)
        sprite_section_layout.addWidget(self.list_widget, 1)

        self.group_sprite_splitter = QSplitter(Qt.Orientation.Vertical)
        self.group_sprite_splitter.addWidget(self.group_section)
        self.group_sprite_splitter.addWidget(self.sprite_section)
        self.group_sprite_splitter.setChildrenCollapsible(False)
        self.group_sprite_splitter.setStretchFactor(0, 1)
        self.group_sprite_splitter.setStretchFactor(1, 3)
        self.group_sprite_splitter.setSizes([190, 430])

        layout.addWidget(self.clear_button)
        layout.addWidget(self.project_info_label)
        layout.addWidget(self.count_label)
        layout.addWidget(self.group_sprite_splitter, 1)
        self.list_widget.currentItemChanged.connect(on_selection_change)
        self.list_widget.viewport().installEventFilter(self)
        self.display_mode_combo.currentIndexChanged.connect(self._on_browser_controls_changed)
        self.sort_mode_combo.currentIndexChanged.connect(self._on_browser_controls_changed)
        self.group_marker_mode_combo.currentIndexChanged.connect(self._on_browser_controls_changed)
        self.group_square_thickness_spin.valueChanged.connect(self._on_browser_controls_changed)
        self.group_square_padding_spin.valueChanged.connect(self._on_browser_controls_changed)
        self.group_square_fill_alpha_spin.valueChanged.connect(self._on_browser_controls_changed)
        self.scroll_speed_spin.valueChanged.connect(self._on_scroll_speed_changed)
        self.zoom_slider.valueChanged.connect(self._on_browser_controls_changed)
        self.zoom_slider.sliderPressed.connect(self._on_zoom_slider_pressed)
        self.zoom_slider.sliderReleased.connect(self._on_zoom_slider_released)
        self.float_groups_button.toggled.connect(self._set_group_panel_floating)
        self.float_sprites_button.toggled.connect(self._set_sprite_panel_floating)
        self.search_edit.textChanged.connect(self._on_search_text_changed)
        self.browser_settings_close_button.clicked.connect(self.browser_settings_dialog.close)

        self.search_focus_shortcut = QShortcut(QKeySequence.StandardKey.Find, self)
        self.search_focus_shortcut.activated.connect(self._focus_search)
        self.search_next_shortcut = QShortcut(QKeySequence("F3"), self)
        self.search_next_shortcut.activated.connect(lambda: self.jump_search(forward=True))
        self.search_prev_shortcut = QShortcut(QKeySequence("Shift+F3"), self)
        self.search_prev_shortcut.activated.connect(lambda: self.jump_search(forward=False))
        self.search_clear_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self.search_edit)
        self.search_clear_shortcut.activated.connect(self._clear_search)
        self._apply_browser_mode()

    def set_loaded_count(self, count: int) -> None:
        if count <= 0:
            self.count_label.setText("No files loaded")
        else:
            self.count_label.setText(f"{count} file(s) loaded")

    def selected_load_mode(self) -> Literal["detect", "preserve"]:
        data = self.load_mode_combo.currentData()
        return "preserve" if data == "preserve" else "detect"

    def browser_view_mode(self) -> str:
        return str(self.display_mode_combo.currentData() or "list")

    def browser_sort_mode(self) -> str:
        return str(self.sort_mode_combo.currentData() or "added")

    def browser_zoom(self) -> int:
        return int(self.zoom_slider.value())

    def browser_zoom_for_mode(self, view_mode: str) -> int:
        key = "thumbnails" if view_mode == "thumbnails" else "list"
        return int(self._zoom_by_view.get(key, int(self.zoom_slider.value())))

    def browser_group_marker_mode(self) -> str:
        return str(self.group_marker_mode_combo.currentData() or "text")

    def browser_group_square_thickness(self) -> int:
        return int(self.group_square_thickness_spin.value())

    def browser_group_square_padding(self) -> int:
        return int(self.group_square_padding_spin.value())

    def browser_group_square_fill_alpha(self) -> int:
        return int(self.group_square_fill_alpha_spin.value())

    def browser_scroll_speed(self) -> int:
        return int(self.scroll_speed_spin.value())

    def browser_search_text(self) -> str:
        return self.search_edit.text().strip()

    def _clamp_browser_zoom(self, zoom: int) -> int:
        return max(int(self.zoom_slider.minimum()), min(int(self.zoom_slider.maximum()), int(zoom)))

    def apply_browser_settings(
        self,
        *,
        view_mode: str,
        sort_mode: str,
        zoom: int,
        list_zoom: int | None = None,
        thumbnails_zoom: int | None = None,
        group_marker_mode: str,
        group_square_thickness: int,
        group_square_padding: int,
        group_square_fill_alpha: int,
        scroll_speed: int,
    ) -> None:
        view_index = self.display_mode_combo.findData(view_mode)
        if view_index < 0:
            view_index = self.display_mode_combo.findData("list")
        sort_index = self.sort_mode_combo.findData(sort_mode)
        if sort_index < 0:
            sort_index = self.sort_mode_combo.findData("added")
        marker_index = self.group_marker_mode_combo.findData(group_marker_mode)
        if marker_index < 0:
            marker_index = self.group_marker_mode_combo.findData("text")

        blocked_display = self.display_mode_combo.blockSignals(True)
        blocked_sort = self.sort_mode_combo.blockSignals(True)
        blocked_marker = self.group_marker_mode_combo.blockSignals(True)
        blocked_thickness = self.group_square_thickness_spin.blockSignals(True)
        blocked_padding = self.group_square_padding_spin.blockSignals(True)
        blocked_fill_alpha = self.group_square_fill_alpha_spin.blockSignals(True)
        blocked_scroll_speed = self.scroll_speed_spin.blockSignals(True)
        blocked_zoom = self.zoom_slider.blockSignals(True)
        self.display_mode_combo.setCurrentIndex(max(0, view_index))
        self.sort_mode_combo.setCurrentIndex(max(0, sort_index))
        self.group_marker_mode_combo.setCurrentIndex(max(0, marker_index))
        self.group_square_thickness_spin.setValue(
            max(self.group_square_thickness_spin.minimum(), min(self.group_square_thickness_spin.maximum(), int(group_square_thickness)))
        )
        self.group_square_padding_spin.setValue(
            max(self.group_square_padding_spin.minimum(), min(self.group_square_padding_spin.maximum(), int(group_square_padding)))
        )
        self.group_square_fill_alpha_spin.setValue(
            max(
                self.group_square_fill_alpha_spin.minimum(),
                min(self.group_square_fill_alpha_spin.maximum(), int(group_square_fill_alpha)),
            )
        )
        self.scroll_speed_spin.setValue(
            max(self.scroll_speed_spin.minimum(), min(self.scroll_speed_spin.maximum(), int(scroll_speed)))
        )
        fallback_zoom = self._clamp_browser_zoom(int(zoom))
        self._zoom_by_view = {
            "list": self._clamp_browser_zoom(int(list_zoom) if list_zoom is not None else fallback_zoom),
            "thumbnails": self._clamp_browser_zoom(
                int(thumbnails_zoom) if thumbnails_zoom is not None else fallback_zoom
            ),
        }
        active_view_mode = self.browser_view_mode()
        active_zoom = self.browser_zoom_for_mode(active_view_mode)
        self.zoom_slider.setValue(active_zoom)
        self._last_view_mode = active_view_mode
        self.display_mode_combo.blockSignals(blocked_display)
        self.sort_mode_combo.blockSignals(blocked_sort)
        self.group_marker_mode_combo.blockSignals(blocked_marker)
        self.group_square_thickness_spin.blockSignals(blocked_thickness)
        self.group_square_padding_spin.blockSignals(blocked_padding)
        self.group_square_fill_alpha_spin.blockSignals(blocked_fill_alpha)
        self.scroll_speed_spin.blockSignals(blocked_scroll_speed)
        self.zoom_slider.blockSignals(blocked_zoom)
        self._apply_browser_mode()

    def _on_browser_controls_changed(self) -> None:
        sender_obj = self.sender()
        current_view_mode = self.browser_view_mode()
        if sender_obj is self.display_mode_combo:
            previous_view_mode = "thumbnails" if self._last_view_mode == "thumbnails" else "list"
            self._zoom_by_view[previous_view_mode] = self._clamp_browser_zoom(int(self.zoom_slider.value()))
            target_zoom = self.browser_zoom_for_mode(current_view_mode)
            blocked_zoom = self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(self._clamp_browser_zoom(target_zoom))
            self.zoom_slider.blockSignals(blocked_zoom)
            self._last_view_mode = "thumbnails" if current_view_mode == "thumbnails" else "list"
        elif sender_obj is self.zoom_slider:
            view_key = "thumbnails" if current_view_mode == "thumbnails" else "list"
            self._zoom_by_view[view_key] = self._clamp_browser_zoom(int(self.zoom_slider.value()))
            self._last_view_mode = view_key
        else:
            self._last_view_mode = "thumbnails" if current_view_mode == "thumbnails" else "list"
        self._apply_browser_mode()
        if sender_obj is self.zoom_slider and self._zoom_drag_active:
            self._pending_zoom_drag_commit = True
            return
        if self._on_browser_settings_change is not None:
            self._on_browser_settings_change()
        self._schedule_browser_change_emit()

    def _on_scroll_speed_changed(self) -> None:
        if self._on_browser_settings_change is not None:
            self._on_browser_settings_change()

    def _on_zoom_slider_pressed(self) -> None:
        self._zoom_drag_active = True

    def _on_zoom_slider_released(self) -> None:
        self._zoom_drag_active = False
        if not self._pending_zoom_drag_commit:
            return
        self._pending_zoom_drag_commit = False
        if self._on_browser_settings_change is not None:
            self._on_browser_settings_change()
        self._schedule_browser_change_emit()

    def _focus_search(self) -> None:
        self.search_edit.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.search_edit.selectAll()

    def _clear_search(self) -> None:
        if self.search_edit.text():
            self.search_edit.clear()

    def _on_search_text_changed(self, _text: str) -> None:
        self.apply_browser_search_filter()

    def _item_matches_search(self, item: QListWidgetItem, needle: str) -> bool:
        text = (item.text() or "").lower()
        tooltip = (item.toolTip() or "").lower()
        return needle in text or needle in tooltip

    def _visible_rows(self) -> List[int]:
        rows: List[int] = []
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is not None and not item.isHidden():
                rows.append(row)
        return rows

    def jump_search(self, *, forward: bool) -> None:
        visible_rows = self._visible_rows()
        if not visible_rows:
            return
        current_row = self.list_widget.currentRow()
        if current_row < 0:
            target_row = visible_rows[0] if forward else visible_rows[-1]
            self.list_widget.setCurrentRow(target_row)
            self.list_widget.scrollToItem(self.list_widget.item(target_row))
            return

        if forward:
            for row in visible_rows:
                if row > current_row:
                    self.list_widget.setCurrentRow(row)
                    self.list_widget.scrollToItem(self.list_widget.item(row))
                    return
            target_row = visible_rows[0]
        else:
            for row in reversed(visible_rows):
                if row < current_row:
                    self.list_widget.setCurrentRow(row)
                    self.list_widget.scrollToItem(self.list_widget.item(row))
                    return
            target_row = visible_rows[-1]
        self.list_widget.setCurrentRow(target_row)
        self.list_widget.scrollToItem(self.list_widget.item(target_row))

    def apply_browser_search_filter(self) -> None:
        needle = self.browser_search_text().lower()
        list_widget = self.list_widget
        selected_before = list_widget.currentItem()
        selected_before_key = selected_before.data(Qt.ItemDataRole.UserRole) if selected_before is not None else None

        updates_enabled = list_widget.updatesEnabled()
        list_widget.setUpdatesEnabled(False)
        try:
            for row in range(list_widget.count()):
                item = list_widget.item(row)
                if item is None:
                    continue
                match = True if not needle else self._item_matches_search(item, needle)
                item.setHidden(not match)
                if not match:
                    item.setSelected(False)

            visible_rows = self._visible_rows()
            if not visible_rows:
                list_widget.clearSelection()
                list_widget.setCurrentItem(None)
            else:
                current_item = list_widget.currentItem()
                if current_item is None or current_item.isHidden():
                    preferred_row = None
                    if selected_before_key:
                        for row in visible_rows:
                            candidate = list_widget.item(row)
                            if candidate is not None and candidate.data(Qt.ItemDataRole.UserRole) == selected_before_key:
                                preferred_row = row
                                break
                    if preferred_row is None:
                        preferred_row = visible_rows[0]
                    list_widget.setCurrentRow(preferred_row)
                    target = list_widget.item(preferred_row)
                    if target is not None:
                        list_widget.scrollToItem(target)
        finally:
            list_widget.setUpdatesEnabled(updates_enabled)
            list_widget.viewport().update()

    def _schedule_browser_change_emit(self) -> None:
        count = self.list_widget.count()
        if count >= 1200:
            interval = _BROWSER_DEBOUNCE_SLOW_MS
        elif count >= 300:
            interval = _BROWSER_DEBOUNCE_MEDIUM_MS
        else:
            interval = _BROWSER_DEBOUNCE_FAST_MS
        if self._browser_change_timer.interval() != interval:
            self._browser_change_timer.setInterval(interval)
        self._browser_change_timer.start()

    def _emit_browser_change(self) -> None:
        if self._on_browser_change is not None:
            self._on_browser_change()

    def open_browser_settings_dialog(self) -> None:
        if self.browser_settings_dialog.isVisible():
            self.browser_settings_dialog.raise_()
            self.browser_settings_dialog.activateWindow()
            return
        self.browser_settings_dialog.show()
        self.browser_settings_dialog.raise_()
        self.browser_settings_dialog.activateWindow()

    def _set_group_panel_floating(self, enabled: bool) -> None:
        if enabled:
            if self._group_panel_floating:
                return
            self.group_section.setParent(None)
            self.group_float_dialog_layout.addWidget(self.group_section)
            self.group_float_dialog.resize(520, 340)
            self.group_float_dialog.show()
            self.group_float_dialog.raise_()
            self.group_float_dialog.activateWindow()
            self._group_panel_floating = True
            return

        if not self._group_panel_floating:
            return
        self.group_section.setParent(None)
        self.group_sprite_splitter.insertWidget(0, self.group_section)
        self.group_float_dialog.hide()
        self._group_panel_floating = False

    def _set_sprite_panel_floating(self, enabled: bool) -> None:
        if enabled:
            if self._sprite_panel_floating:
                return
            self.sprite_section.setParent(None)
            self.sprite_float_dialog_layout.addWidget(self.sprite_section)
            self.sprite_float_dialog.resize(620, 460)
            self.sprite_float_dialog.show()
            self.sprite_float_dialog.raise_()
            self.sprite_float_dialog.activateWindow()
            self._sprite_panel_floating = True
            return

        if not self._sprite_panel_floating:
            return
        self.sprite_section.setParent(None)
        self.group_sprite_splitter.insertWidget(1, self.sprite_section)
        self.sprite_float_dialog.hide()
        self._sprite_panel_floating = False

    def _apply_browser_mode(self) -> None:
        zoom = self.browser_zoom()
        self.zoom_value_label.setText(str(zoom))
        view_mode = self.browser_view_mode()
        if view_mode == "thumbnails":
            tile = max(48, min(int(self.zoom_slider.maximum()), zoom))
            cell_w = tile + 44
            cell_h = tile + 40
            self.zoom_slider.setEnabled(True)
            self.group_marker_mode_combo.setEnabled(True)
            marker_has_square = self.group_marker_mode_combo.currentData() in ("square", "both")
            self.group_square_thickness_spin.setEnabled(marker_has_square)
            self.group_square_padding_spin.setEnabled(marker_has_square)
            self.group_square_fill_alpha_spin.setEnabled(marker_has_square)
            self.scroll_speed_spin.setEnabled(True)
            self.list_widget.setViewMode(QListView.ViewMode.IconMode)
            self.list_widget.setFlow(QListView.Flow.LeftToRight)
            self.list_widget.setWrapping(True)
            self.list_widget.setWordWrap(False)
            self.list_widget.setSpacing(8)
            self.list_widget.setLayoutMode(QListView.LayoutMode.SinglePass)
            self.list_widget.setIconSize(QSize(tile, tile))
            self.list_widget.setGridSize(QSize(cell_w, cell_h))
            self.list_widget.setAlternatingRowColors(False)
        else:
            list_icon = max(28, min(int(self.zoom_slider.maximum()), zoom))
            list_row_h = max(36, list_icon + 10)
            self.zoom_slider.setEnabled(True)
            self.group_marker_mode_combo.setEnabled(False)
            self.group_square_thickness_spin.setEnabled(False)
            self.group_square_padding_spin.setEnabled(False)
            self.group_square_fill_alpha_spin.setEnabled(False)
            self.scroll_speed_spin.setEnabled(False)
            self.list_widget.setViewMode(QListView.ViewMode.ListMode)
            self.list_widget.setFlow(QListView.Flow.TopToBottom)
            self.list_widget.setWrapping(False)
            self.list_widget.setWordWrap(False)
            self.list_widget.setSpacing(2)
            self.list_widget.setLayoutMode(QListView.LayoutMode.SinglePass)
            self.list_widget.setIconSize(QSize(list_icon, list_icon))
            self.list_widget.setGridSize(QSize(0, list_row_h))
            self.list_widget.setAlternatingRowColors(True)

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # type: ignore[override]
        try:
            event_type = int(event.type())
        except Exception:  # noqa: BLE001
            return super().eventFilter(watched, event)

        group_dialog = getattr(self, "group_float_dialog", None)
        sprite_dialog = getattr(self, "sprite_float_dialog", None)
        groups_button = getattr(self, "float_groups_button", None)
        sprites_button = getattr(self, "float_sprites_button", None)
        group_floating = bool(getattr(self, "_group_panel_floating", False))
        sprite_floating = bool(getattr(self, "_sprite_panel_floating", False))
        list_widget = getattr(self, "list_widget", None)
        list_viewport = list_widget.viewport() if list_widget is not None else None

        if watched is group_dialog and event_type == _EVENT_TYPE_CLOSE:
            if group_floating and groups_button is not None:
                blocked = groups_button.blockSignals(True)
                groups_button.setChecked(False)
                groups_button.blockSignals(blocked)
                self._set_group_panel_floating(False)
            return False

        if watched is sprite_dialog and event_type == _EVENT_TYPE_CLOSE:
            if sprite_floating and sprites_button is not None:
                blocked = sprites_button.blockSignals(True)
                sprites_button.setChecked(False)
                sprites_button.blockSignals(blocked)
                self._set_sprite_panel_floating(False)
            return False

        if watched is list_viewport and event_type == _EVENT_TYPE_WHEEL and isinstance(event, QWheelEvent):
            if bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                delta_y = event.angleDelta().y()
                if delta_y != 0:
                    steps = max(1, abs(delta_y) // 120)
                    direction = 1 if delta_y > 0 else -1
                    self.zoom_slider.setValue(self.zoom_slider.value() + (direction * steps * self.zoom_slider.singleStep()))
                    event.accept()
                    return True
            elif self.browser_view_mode() == "thumbnails":
                multiplier = max(1, self.browser_scroll_speed())
                vertical_scrollbar = self.list_widget.verticalScrollBar()
                pixel_delta_y = event.pixelDelta().y()
                if pixel_delta_y != 0:
                    vertical_scrollbar.setValue(vertical_scrollbar.value() - (pixel_delta_y * multiplier))
                    event.accept()
                    return True

                delta_y = event.angleDelta().y()
                if delta_y != 0:
                    steps = max(1, abs(delta_y) // 120)
                    direction = 1 if delta_y > 0 else -1
                    pixels_per_notch = 36 * multiplier
                    vertical_scrollbar.setValue(vertical_scrollbar.value() - (direction * steps * pixels_per_notch))
                    event.accept()
                    return True
        return super().eventFilter(watched, event)

    def set_project_info(
        self,
        *,
        name: str | None,
        mode: str | None,
        root: Path | None,
        dirty: bool = False,
        last_saved_text: str | None = None,
        last_saved_kind: str | None = None,
        recovery_source_text: str | None = None,
    ) -> None:
        if not name or root is None:
            self.project_info_label.setText("Workspace: Unsaved")
            self.project_info_label.setToolTip("No active project")
            return
        mode_text = (mode or "managed").strip().lower()
        friendly_mode = "Managed" if mode_text == "managed" else ("Linked" if mode_text == "linked" else mode_text.title())
        root_name = root.name or root.as_posix()
        state_text = "Unsaved changes" if dirty else "Saved"
        self.project_info_label.setText(f"Project: {name} ({friendly_mode})\nState: {state_text} • Folder: {root_name}")
        tooltip = f"{root}\nState: {state_text}"
        if last_saved_text:
            kind = (last_saved_kind or "Saved").strip()
            tooltip += f"\n{kind}: {last_saved_text}"
        if recovery_source_text:
            tooltip += f"\nRecovery source: {recovery_source_text}"
        self.project_info_label.setToolTip(tooltip)


class FloatingPaletteWindow(QWidget):
    def __init__(self, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._window_index = index
        self._settings = QSettings("SpriteTools", "SpriteTools")
        self._settings_prefix = f"floating_palette/{self._window_index}"
        self._is_loading_settings = False
        self._layout_in_progress = False
        self._shown_once = False
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._apply_layout_options)
        self._supported_drop_suffixes = {
            ".act", ".gpl", ".pal", ".txt", ".hex", ".png", ".bmp", ".gif", ".jpg", ".jpeg"
        }
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(f"Palette Window #{index}")
        self.resize(460, 520)
        self.setAcceptDrops(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        controls = QHBoxLayout()
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self._load_palette)
        controls.addWidget(self.load_button)

        self.clear_palette_btn = QPushButton("Clear Currently Loaded Palette")
        self.clear_palette_btn.clicked.connect(self._clear_loaded_palette)
        controls.addWidget(self.clear_palette_btn)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Detect", "detect")
        self.mode_combo.addItem("Preserve", "preserve")
        self.mode_combo.setToolTip("For image imports: Detect builds palette, Preserve keeps source indices")
        self.mode_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.mode_combo.setMinimumWidth(110)
        controls.addWidget(self.mode_combo)

        controls.addSpacing(8)
        controls.addWidget(QLabel("Columns"))
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(1, 32)
        self.columns_spin.setValue(8)
        self.columns_spin.valueChanged.connect(self._apply_layout_options)
        self.columns_spin.setFixedWidth(54)
        controls.addWidget(self.columns_spin)

        controls.addWidget(QLabel("Zoom"))
        self.zoom_spin = QSpinBox()
        self.zoom_spin.setRange(16, 96)
        self.zoom_spin.setValue(42)
        self.zoom_spin.valueChanged.connect(self._apply_layout_options)
        self.zoom_spin.setFixedWidth(58)
        controls.addWidget(self.zoom_spin)

        controls.addWidget(QLabel("Gap"))
        self.spacing_spin = QSpinBox()
        self.spacing_spin.setRange(-8, 20)
        self.spacing_spin.setValue(6)
        self.spacing_spin.valueChanged.connect(self._apply_layout_options)
        self.spacing_spin.setFixedWidth(50)
        controls.addWidget(self.spacing_spin)
        controls.addStretch(1)
        root.addLayout(controls)

        force_cols_row = QHBoxLayout()
        self.force_columns_check = QCheckBox("Force palette view columns")
        self.force_columns_check.setChecked(True)
        self.force_columns_check.toggled.connect(self._apply_layout_options)
        force_cols_row.addWidget(self.force_columns_check)
        force_cols_row.addStretch(1)
        root.addLayout(force_cols_row)

        toggles = QHBoxLayout()
        self.show_indices_check = QCheckBox("Indices")
        self.show_indices_check.setChecked(True)
        self.show_indices_check.toggled.connect(self._apply_layout_options)
        toggles.addWidget(self.show_indices_check)

        self.grid_lines_check = QCheckBox("Grid")
        self.grid_lines_check.setChecked(False)
        self.grid_lines_check.toggled.connect(self._apply_layout_options)
        toggles.addWidget(self.grid_lines_check)

        self.count_label = QLabel("0 colors")
        toggles.addStretch(1)
        toggles.addWidget(self.count_label)
        root.addLayout(toggles)

        self.palette_list = PaletteListView(self)
        self.palette_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.palette_list.palette_changed.connect(self._on_palette_changed)
        root.addWidget(self.palette_list, 1)

        self._load_view_settings()
        self._layout_timer.start(0)

    def _settings_key(self, name: str) -> str:
        return f"{self._settings_prefix}/{name}"

    def _get_pref_bool(self, name: str, default: bool) -> bool:
        value = self._settings.value(self._settings_key(name), default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _get_pref_int(self, name: str, default: int) -> int:
        value = self._settings.value(self._settings_key(name), default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _load_view_settings(self) -> None:
        self._is_loading_settings = True
        controls = [
            self.columns_spin,
            self.zoom_spin,
            self.spacing_spin,
            self.force_columns_check,
            self.show_indices_check,
            self.grid_lines_check,
        ]
        blocked_states = [control.blockSignals(True) for control in controls]
        try:
            self.columns_spin.setValue(max(1, min(32, self._get_pref_int("columns", self.columns_spin.value()))))
            self.zoom_spin.setValue(max(16, min(96, self._get_pref_int("zoom", self.zoom_spin.value()))))
            self.spacing_spin.setValue(max(-8, min(20, self._get_pref_int("gap", self.spacing_spin.value()))))
            self.force_columns_check.setChecked(self._get_pref_bool("force_columns", self.force_columns_check.isChecked()))
            self.show_indices_check.setChecked(self._get_pref_bool("show_indices", self.show_indices_check.isChecked()))
            self.grid_lines_check.setChecked(self._get_pref_bool("show_grid", self.grid_lines_check.isChecked()))
            mode_value = str(self._settings.value(self._settings_key("load_mode"), self.mode_combo.currentData() or "detect")).strip().lower()
            mode_index = self.mode_combo.findData("preserve" if mode_value == "preserve" else "detect")
            self.mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        finally:
            for control, blocked in zip(controls, blocked_states):
                control.blockSignals(blocked)
            self._is_loading_settings = False

        geometry = self._settings.value(self._settings_key("geometry"))
        if geometry is not None:
            try:
                self.restoreGeometry(geometry)
            except Exception:  # noqa: BLE001
                logger.debug("Floating palette failed to restore geometry window=%s", self._window_index, exc_info=True)

        logger.debug(
            "Floating palette settings loaded window=%s cols=%s zoom=%s gap=%s force=%s idx=%s grid=%s mode=%s",
            self._window_index,
            self.columns_spin.value(),
            self.zoom_spin.value(),
            self.spacing_spin.value(),
            self.force_columns_check.isChecked(),
            self.show_indices_check.isChecked(),
            self.grid_lines_check.isChecked(),
            self.mode_combo.currentData(),
        )

    def _save_view_settings(self) -> None:
        self._settings.setValue(self._settings_key("columns"), int(self.columns_spin.value()))
        self._settings.setValue(self._settings_key("zoom"), int(self.zoom_spin.value()))
        self._settings.setValue(self._settings_key("gap"), int(self.spacing_spin.value()))
        self._settings.setValue(self._settings_key("force_columns"), bool(self.force_columns_check.isChecked()))
        self._settings.setValue(self._settings_key("show_indices"), bool(self.show_indices_check.isChecked()))
        self._settings.setValue(self._settings_key("show_grid"), bool(self.grid_lines_check.isChecked()))
        self._settings.setValue(self._settings_key("load_mode"), str(self.mode_combo.currentData() or "detect"))
        self._settings.setValue(self._settings_key("geometry"), self.saveGeometry())

    def _selected_mode(self) -> Literal["detect", "preserve"]:
        return "preserve" if self.mode_combo.currentData() == "preserve" else "detect"

    def _apply_layout_options(self) -> None:
        if self._is_loading_settings or self._layout_in_progress:
            return
        self._layout_in_progress = True
        try:
            gap = self.spacing_spin.value()
            cols = self.columns_spin.value()
            force_columns = self.force_columns_check.isChecked()
            effective_gap = max(0, gap)
            tightness = max(0, -gap)
            if force_columns:
                available_width = max(120, self.palette_list.viewport().width())
                if cols > 1:
                    max_gap_for_cols = max(0, (available_width - (cols * 16)) // (cols - 1))
                    if effective_gap > max_gap_for_cols:
                        logger.debug(
                            "Floating palette gap clamped window=%s requested_gap=%s clamped_gap=%s cols=%s viewport_w=%s",
                            self._window_index,
                            effective_gap,
                            max_gap_for_cols,
                            cols,
                            available_width,
                        )
                        effective_gap = max_gap_for_cols
                cell = _compute_forced_cell_size(available_width, cols, effective_gap)
                self.palette_list.setSpacing(effective_gap)
                self.palette_list.set_cell_size(cell)
                realized_cols = self.palette_list.measure_first_row_columns()
                if realized_cols and realized_cols != cols:
                    attempts = 0
                    while realized_cols < cols and cell > 16 and attempts < 40:
                        cell -= 1
                        self.palette_list.set_cell_size(cell)
                        realized_cols = self.palette_list.measure_first_row_columns()
                        attempts += 1
                    while realized_cols > cols and cell < 96 and attempts < 80:
                        cell += 1
                        self.palette_list.set_cell_size(cell)
                        new_realized = self.palette_list.measure_first_row_columns()
                        attempts += 1
                        if new_realized < cols:
                            cell -= 1
                            self.palette_list.set_cell_size(cell)
                            realized_cols = self.palette_list.measure_first_row_columns()
                            break
                        realized_cols = new_realized
                    logger.debug(
                        "Floating palette exact-fit correction window=%s target_cols=%s realized_cols=%s cell=%s attempts=%s",
                        self._window_index,
                        cols,
                        realized_cols,
                        cell,
                        attempts,
                    )
                if self.zoom_spin.value() != cell:
                    self.zoom_spin.blockSignals(True)
                    self.zoom_spin.setValue(cell)
                    self.zoom_spin.blockSignals(False)
            else:
                available_width = self.palette_list.viewport().width()
                cell = self.zoom_spin.value()
            self.zoom_spin.setEnabled(not force_columns)
            self.palette_list.setSpacing(effective_gap)
            self.palette_list.set_cell_size(cell)
            self.palette_list.set_swatch_inset(max(1, 6 - tightness))
            self.palette_list.set_show_index_labels(self.show_indices_check.isChecked())
            self.palette_list.set_show_grid_lines(self.grid_lines_check.isChecked())
            if force_columns:
                self.palette_list.setMinimumWidth(220)
            else:
                width_hint = cols * (cell + effective_gap) + 30
                self.palette_list.setMinimumWidth(width_hint)
            logger.debug(
                "Floating palette apply layout window=%s force_cols=%s cols=%s cell=%s gap=%s effective_gap=%s tightness=%s viewport_w=%s show_idx=%s grid=%s",
                self._window_index,
                force_columns,
                cols,
                cell,
                gap,
                effective_gap,
                tightness,
                available_width,
                self.show_indices_check.isChecked(),
                self.grid_lines_check.isChecked(),
            )
            self._save_view_settings()
        finally:
            self._layout_in_progress = False

    def _update_count_label(self) -> None:
        count = sum(1 for color in self.palette_list.model_obj.colors if color is not None)
        rows = max(1, (count + max(1, self.columns_spin.value()) - 1) // max(1, self.columns_spin.value()))
        self.count_label.setText(f"{count} colors • {rows} rows")

    def _on_palette_changed(self, _colors: List[ColorTuple]) -> None:
        self._update_count_label()

    def _clear_loaded_palette(self) -> None:
        self.palette_list.set_colors([], slots=[], alphas=[], emit_signal=False)
        self._update_count_label()
        self.setWindowTitle(f"Palette Window #{self._window_index}")
        logger.debug("Floating palette cleared window=%s", self._window_index)

    def _load_palette(self) -> None:
        file_path, _filter = QFileDialog.getOpenFileName(
            self,
            "Load Palette Source",
            str(Path.cwd()),
            "Palette/Image (*.act *.gpl *.pal *.txt *.hex *.png *.bmp *.gif *.jpg *.jpeg)",
        )
        if not file_path:
            return
        path = Path(file_path)
        self._load_palette_from_path(path)

    def _load_palette_from_path(self, path: Path) -> None:
        mode = self._selected_mode()
        try:
            colors, alphas, source = _load_palette_from_source(path, mode)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Floating palette load failed path=%s mode=%s err=%s", path.name, mode, exc)
            QMessageBox.warning(self, "Load Palette", f"Failed to load palette:\n{exc}")
            return

        self.palette_list.set_colors(colors, slots=list(range(len(colors))), alphas=alphas, emit_signal=False)
        self._update_count_label()
        self.setWindowTitle(f"Palette Window #{self._window_index} - {path.name}")
        logger.debug(
            "Floating palette loaded window=%s path=%s mode=%s source=%s colors=%s",
            self._window_index,
            path.name,
            mode,
            source,
            len(colors),
        )

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        if hasattr(self, "force_columns_check") and self.force_columns_check.isChecked():
            self._layout_timer.start(16)
        super().resizeEvent(event)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._shown_once:
            self._shown_once = True
            self._layout_timer.start(0)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_view_settings()
        super().closeEvent(event)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        for url in urls:
            local = Path(url.toLocalFile())
            if local.is_file() and local.suffix.lower() in self._supported_drop_suffixes:
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        loaded = 0
        for url in urls:
            local = Path(url.toLocalFile())
            if not local.is_file() or local.suffix.lower() not in self._supported_drop_suffixes:
                continue
            self._load_palette_from_path(local)
            loaded += 1
        if loaded:
            logger.debug("Floating palette drop loaded window=%s files=%s", self._window_index, loaded)
            event.acceptProposedAction()
        else:
            event.ignore()


class MergeOperationDialog(QDialog):
    def __init__(self, owner: "SpriteToolsWindow") -> None:
        super().__init__(owner)
        self._owner = owner
        self._source_rows: set[int] = set()
        self._destination_row: int | None = None
        self._scope_records: List[SpriteRecord] = []
        self._source_after_color: ColorTuple = (255, 0, 255)
        self._quick_assign_mouse_active = False
        self._hover_palette_row: int | None = None
        self._destination_blink_on = False
        self._destination_blink_phase = 0.0
        self._destination_blink_timer = QTimer(self)
        self._destination_blink_timer.setInterval(30)
        self._destination_blink_timer.timeout.connect(self._tick_destination_blink)
        self._view_columns = max(1, min(32, self._owner._get_pref_int("merge/view_columns", 8)))
        self._view_force_columns = self._owner._get_pref_bool("merge/view_force_columns", True)
        self._view_zoom = max(16, min(96, self._owner._get_pref_int("merge/view_zoom", 42)))
        self._view_gap = max(-8, min(20, self._owner._get_pref_int("merge/view_gap", 6)))
        self._view_show_indices = self._owner._get_pref_bool("merge/view_show_indices", True)
        self._view_show_grid = self._owner._get_pref_bool("merge/view_show_grid", True)
        self.setModal(True)
        self.setWindowTitle("Merge Indexes (Source → Destination)")
        self.resize(1160, 700)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Apply To"))
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("Global", "global")
        self.scope_combo.addItem("Group", "group")
        self.scope_combo.addItem("Local", "local")
        self.scope_combo.setToolTip("Where merge changes are applied: all sprites, current group, or current sprite")
        self.scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        controls.addWidget(self.scope_combo)

        self.view_settings_btn = QPushButton("View")
        self.view_settings_btn.setToolTip("Configure merge palette swatch view")
        self.view_settings_btn.clicked.connect(self._open_view_settings)
        controls.addWidget(self.view_settings_btn)

        controls.addWidget(QLabel("Source Swatches After Merge"))
        self.source_after_combo = QComboBox()
        self.source_after_combo.addItem("Fill gaps (compact palette)", "compact_shift")
        self.source_after_combo.addItem("Preserve source swatch colors", "preserve_colors")
        self.source_after_combo.addItem("Recolor source swatches", "recolor_sources")
        self.source_after_combo.setToolTip("Controls what happens to Source swatches after pixel remap")
        self.source_after_combo.currentIndexChanged.connect(self._on_source_after_mode_changed)
        controls.addWidget(self.source_after_combo)

        self.source_after_color_btn = QPushButton("Pick color")
        self.source_after_color_btn.clicked.connect(self._pick_source_after_color)
        self.source_after_color_btn.setVisible(False)
        controls.addWidget(self.source_after_color_btn)

        controls.addStretch(1)
        self.selection_label = QLabel("Source: none | Destination: none")
        controls.addWidget(self.selection_label)
        root.addLayout(controls)

        quick_row = QHBoxLayout()
        self.quick_assign_check = QCheckBox("Quick assign mode")
        self.quick_assign_check.setChecked(True)
        quick_row.addWidget(self.quick_assign_check)
        self.quick_assign_hint_label = QLabel(
            "L="
            "<span style='color:rgb(86,182,255)'><b>Source</b></span> "
            "| R="
            "<span style='color:rgb(196,120,255)'><b>Destination</b></span> "
            "| M=Clear swatch"
        )
        self.quick_assign_hint_label.setTextFormat(Qt.TextFormat.RichText)
        quick_row.addWidget(self.quick_assign_hint_label)
        quick_row.addStretch(1)
        root.addLayout(quick_row)

        role_row = QHBoxLayout()
        self.tag_source_btn = QPushButton("Tag Selected as Source")
        self.tag_source_btn.setStyleSheet(
            "QPushButton { border: 1px solid rgb(86,182,255); color: rgb(86,182,255); font-weight: 600; }"
        )
        self.tag_source_btn.clicked.connect(self._mark_selected_as_sources)
        role_row.addWidget(self.tag_source_btn)
        self.tag_destination_btn = QPushButton("Set Current as Destination")
        self.tag_destination_btn.setStyleSheet(
            "QPushButton { border: 1px solid rgb(196,120,255); color: rgb(196,120,255); font-weight: 600; }"
        )
        self.tag_destination_btn.clicked.connect(self._set_current_as_destination)
        role_row.addWidget(self.tag_destination_btn)
        self.clear_roles_btn = QPushButton("Clear Source/Destination")
        self.clear_roles_btn.clicked.connect(self._clear_source_destination)
        role_row.addWidget(self.clear_roles_btn)
        self.clear_all_btn = QPushButton("Clear All Selections")
        self.clear_all_btn.clicked.connect(self._clear_all_selections)
        role_row.addWidget(self.clear_all_btn)
        self.clear_after_apply_check = QCheckBox("Clear selection after Apply Merge")
        self.clear_after_apply_check.setChecked(self._owner._get_pref_bool("merge/clear_selection_after_apply", True))
        self.clear_after_apply_check.toggled.connect(
            lambda checked: self._owner._set_pref("merge/clear_selection_after_apply", bool(checked))
        )
        role_row.addWidget(self.clear_after_apply_check)
        self.apply_button = QPushButton("Apply Merge")
        self.apply_button.clicked.connect(self._apply_merge)
        self.apply_button.setEnabled(False)
        role_row.addWidget(self.apply_button)
        role_row.addStretch(1)
        root.addLayout(role_row)

        self.role_legend_label = QLabel(
            "<b>Role colors:</b> "
            "<span style='color:rgb(86,182,255)'><b>Source</b></span> | "
            "<span style='color:rgb(196,120,255)'><b>Destination</b></span>"
        )
        self.role_legend_label.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self.role_legend_label)

        content_split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(content_split, 1)

        self.tabs = QTabWidget()
        content_split.addWidget(self.tabs)

        main_tab = QWidget()
        main_layout = QVBoxLayout(main_tab)
        main_layout.setContentsMargins(6, 6, 6, 6)
        self.palette_view = PaletteListView(main_tab)
        self.palette_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.palette_view.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.palette_view.setDragEnabled(False)
        self.palette_view.setAcceptDrops(False)
        self.palette_view.selection_changed.connect(self._on_palette_selection_changed)
        self.palette_view.viewport().installEventFilter(self)
        self.palette_view.setMouseTracking(True)
        self.palette_view.viewport().setMouseTracking(True)
        self._apply_palette_view_options()
        main_layout.addWidget(self.palette_view, 1)
        self.tabs.addTab(main_tab, "Palette")

        groups_tab = QWidget()
        groups_layout = QVBoxLayout(groups_tab)
        groups_layout.setContentsMargins(6, 6, 6, 6)
        self.group_list = QListWidget()
        groups_layout.addWidget(self.group_list, 1)
        self.tabs.addTab(groups_tab, "Group Impact")

        sprites_tab = QWidget()
        sprites_layout = QVBoxLayout(sprites_tab)
        sprites_layout.setContentsMargins(6, 6, 6, 6)
        self.sprite_list = QListWidget()
        self.sprite_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.sprite_list.itemSelectionChanged.connect(self._on_sprite_pick_changed)
        self.sprite_list.setIconSize(QSize(40, 40))
        sprites_layout.addWidget(self.sprite_list, 1)
        self.tabs.addTab(sprites_tab, "Sprite Impact")

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(6, 6, 6, 6)
        preview_layout.setSpacing(6)
        self.preview_title_label = QLabel("Preview (right panel, main-style checkerboard)")
        preview_layout.addWidget(self.preview_title_label)

        preview_split = QSplitter()
        self.preview_sprite_list = QListWidget()
        self.preview_sprite_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.preview_sprite_list.itemSelectionChanged.connect(self._on_sprite_pick_changed)
        self.preview_sprite_list.setIconSize(QSize(40, 40))
        preview_split.addWidget(self.preview_sprite_list)

        self.preview_pane = PreviewPane()
        self.preview_pane.title.setVisible(False)
        if hasattr(self._owner, "filter_combo"):
            mode = Qt.TransformationMode.FastTransformation if self._owner.filter_combo.currentIndex() == 0 else Qt.TransformationMode.SmoothTransformation
            self.preview_pane.set_scaling_mode(mode)
        preview_split.addWidget(self.preview_pane)
        preview_split.setStretchFactor(0, 1)
        preview_split.setStretchFactor(1, 3)
        preview_layout.addWidget(preview_split, 1)
        content_split.addWidget(preview_panel)
        content_split.setStretchFactor(0, 2)
        content_split.setStretchFactor(1, 1)
        content_split.setSizes([760, 380])

        self.impact_label = QLabel("Tag Source and Destination indexes")
        self.impact_label.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self.impact_label)
        self.risk_legend_label = QLabel(
            "<b>Risk legend:</b> "
            "<span style='color:#60d278'>Green</span> = selecting this Destination will <b>not lose pixel data</b>; "
            "<span style='color:#ff6060'>Red</span> = selecting this Destination <b>will lose pixel data</b>"
        )
        self.risk_legend_label.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self.risk_legend_label)

        actions = QHBoxLayout()
        actions.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        actions.addWidget(close_btn)
        root.addLayout(actions)

        self._binding_shortcuts: List[QShortcut] = []
        self._setup_key_bindings()
        self._load_palette_snapshot()
        self._refresh_views()

    def _setup_key_bindings(self) -> None:
        for shortcut in self._binding_shortcuts:
            shortcut.setParent(None)
        self._binding_shortcuts.clear()

        binding_map: Dict[str, tuple[str, Callable[[], None]]] = {
            "merge.apply": ("A", self._on_apply_shortcut),
            "merge.tag_source": ("S", self._on_tag_source_shortcut),
            "merge.tag_destination": ("D", self._on_tag_destination_shortcut),
            "merge.clear_roles": ("C", self._on_clear_roles_shortcut),
            "merge.clear_all": ("Shift+C", self._on_clear_all_shortcut),
            "merge.scope_global": ("Alt+1", self._on_scope_global_shortcut),
            "merge.scope_group": ("Alt+2", self._on_scope_group_shortcut),
            "merge.scope_local": ("Alt+3", self._on_scope_local_shortcut),
            "merge.view_settings": ("V", self._on_view_settings_shortcut),
            "merge.close": ("Escape", self._on_close_shortcut),
        }
        for action, (default_sequence, callback) in binding_map.items():
            bindings = self._owner._get_key_bindings(action, default_sequence)
            for binding in bindings:
                sequence = str(binding.get("shortcut", "")).strip()
                if not sequence:
                    continue
                is_global = bool(binding.get("global", False))
                context = Qt.ShortcutContext.ApplicationShortcut if is_global else Qt.ShortcutContext.WidgetWithChildrenShortcut
                shortcut = QShortcut(QKeySequence(sequence), self)
                shortcut.setContext(context)
                shortcut.activated.connect(callback)
                self._binding_shortcuts.append(shortcut)
                logger.debug("Merge dialog binding action=%s sequence=%s global=%s", action, sequence, is_global)

    def _on_apply_shortcut(self) -> None:
        if self.apply_button.isEnabled():
            self._apply_merge()

    def _on_tag_source_shortcut(self) -> None:
        self._mark_selected_as_sources()

    def _on_tag_destination_shortcut(self) -> None:
        self._set_current_as_destination()

    def _on_clear_roles_shortcut(self) -> None:
        self._clear_source_destination()

    def _on_clear_all_shortcut(self) -> None:
        self._clear_all_selections()

    def _set_scope_by_key(self, scope_key: str) -> None:
        index = self.scope_combo.findData(scope_key)
        if index >= 0 and index != self.scope_combo.currentIndex():
            self.scope_combo.setCurrentIndex(index)

    def _on_scope_global_shortcut(self) -> None:
        self._set_scope_by_key("global")

    def _on_scope_group_shortcut(self) -> None:
        self._set_scope_by_key("group")

    def _on_scope_local_shortcut(self) -> None:
        self._set_scope_by_key("local")

    def _on_view_settings_shortcut(self) -> None:
        self._open_view_settings()

    def _on_close_shortcut(self) -> None:
        self.reject()

    def _scope(self) -> Literal["global", "group", "local"]:
        data = self.scope_combo.currentData()
        if data == "group":
            return "group"
        if data == "local":
            return "local"
        return "global"

    def _load_palette_snapshot(self) -> None:
        self.palette_view.set_colors(
            self._owner.palette_colors,
            slots=self._owner._palette_slot_ids,
            alphas=self._owner.palette_alphas,
            emit_signal=False,
        )

    def _current_palette_row(self) -> int | None:
        current = self.palette_view.currentIndex()
        if current.isValid():
            return int(current.row())
        selected = self.palette_view.selectedIndexes()
        if selected:
            return int(selected[0].row())
        return None

    def _selection_payload(self) -> Tuple[List[int], int] | None:
        if self._destination_row is None:
            return None
        sources = sorted(self._source_rows)
        if not sources:
            return None
        return sources, int(self._destination_row)

    def _mark_selected_as_sources(self) -> None:
        rows = sorted({idx.row() for idx in self.palette_view.selectedIndexes()})
        if not rows:
            QMessageBox.information(self, "Source", "Select one or more indexes first.")
            return
        for row in rows:
            if self._destination_row is not None and row == self._destination_row:
                continue
            self._source_rows.add(int(row))
        self._refresh_views()

    def _set_current_as_destination(self) -> None:
        row = self._current_palette_row()
        if row is None:
            QMessageBox.information(self, "Destination", "Select one index as destination.")
            return
        self._destination_row = int(row)
        if self._destination_row in self._source_rows:
            self._source_rows.remove(self._destination_row)
        self._refresh_views()

    def _clear_source_destination(self) -> None:
        self._source_rows.clear()
        self._destination_row = None
        self._refresh_views()

    def _clear_all_selections(self) -> None:
        self._source_rows.clear()
        self._destination_row = None
        self.palette_view.clearSelection()
        self.preview_sprite_list.clearSelection()
        self.sprite_list.clearSelection()
        self._refresh_views()

    def _on_scope_changed(self, _index: int) -> None:
        scope = self._scope()
        if scope != "global" and self.source_after_combo.currentData() == "compact_shift":
            self.source_after_combo.setCurrentIndex(self.source_after_combo.findData("preserve_colors"))
        self._refresh_views()

    def _on_source_after_mode_changed(self, _index: int) -> None:
        mode = self.source_after_combo.currentData()
        self.source_after_color_btn.setVisible(mode == "recolor_sources")

    def _pick_source_after_color(self) -> None:
        current = QColor(*self._source_after_color)
        color = QColorDialog.getColor(current, self, "Pick Source Swatch Color")
        if not color.isValid():
            return
        self._source_after_color = (color.red(), color.green(), color.blue())
        self.source_after_color_btn.setStyleSheet(
            f"background-color: rgb({color.red()},{color.green()},{color.blue()});"
        )

    def _on_palette_selection_changed(self) -> None:
        self._refresh_views()

    def _apply_palette_view_options(self) -> None:
        gap = max(-8, int(self._view_gap))
        cols = max(1, int(self._view_columns))
        effective_gap = max(0, gap)
        tightness = max(0, -gap)
        force_columns = bool(self._view_force_columns)
        if force_columns:
            viewport_width = max(120, self.palette_view.viewport().width())
            cell = _compute_forced_cell_size(viewport_width, cols, effective_gap)
            self.palette_view.setSpacing(effective_gap)
            self.palette_view.set_cell_size(cell)
            realized_cols = self.palette_view.measure_first_row_columns()
            if realized_cols and realized_cols != cols:
                attempts = 0
                while realized_cols < cols and cell > 16 and attempts < 40:
                    cell -= 1
                    self.palette_view.set_cell_size(cell)
                    realized_cols = self.palette_view.measure_first_row_columns()
                    attempts += 1
                while realized_cols > cols and cell < 96 and attempts < 80:
                    cell += 1
                    self.palette_view.set_cell_size(cell)
                    new_realized = self.palette_view.measure_first_row_columns()
                    attempts += 1
                    if new_realized < cols:
                        cell -= 1
                        self.palette_view.set_cell_size(cell)
                        realized_cols = self.palette_view.measure_first_row_columns()
                        break
                    realized_cols = new_realized
                logger.debug(
                    "Merge view exact-fit correction target_cols=%s realized_cols=%s cell=%s attempts=%s",
                    cols,
                    realized_cols,
                    cell,
                    attempts,
                )
            self._view_zoom = cell
        else:
            viewport_width = self.palette_view.viewport().width()
            cell = max(16, min(96, int(self._view_zoom)))
            realized_cols = self.palette_view.measure_first_row_columns()
        self.palette_view.setSpacing(effective_gap)
        self.palette_view.set_cell_size(cell)
        self.palette_view.set_swatch_inset(max(1, 6 - tightness))
        self.palette_view.set_show_index_labels(bool(self._view_show_indices))
        self.palette_view.set_show_grid_lines(bool(self._view_show_grid))
        if force_columns:
            self.palette_view.setMinimumWidth(220)
        else:
            self.palette_view.setMinimumWidth(cols * (cell + effective_gap) + 30)
        final_realized_cols = self.palette_view.measure_first_row_columns()
        logger.debug(
            "Merge dialog view applied force_cols=%s cols=%s realized_cols=%s zoom=%s gap=%s effective_gap=%s tightness=%s viewport_w=%s idx=%s grid=%s",
            force_columns,
            cols,
            final_realized_cols,
            cell,
            gap,
            effective_gap,
            tightness,
            viewport_width,
            self._view_show_indices,
            self._view_show_grid,
        )

    def _save_palette_view_options(self) -> None:
        self._owner._set_pref("merge/view_columns", int(self._view_columns))
        self._owner._set_pref("merge/view_force_columns", bool(self._view_force_columns))
        self._owner._set_pref("merge/view_zoom", int(self._view_zoom))
        self._owner._set_pref("merge/view_gap", int(self._view_gap))
        self._owner._set_pref("merge/view_show_indices", bool(self._view_show_indices))
        self._owner._set_pref("merge/view_show_grid", bool(self._view_show_grid))

    def _open_view_settings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Merge Palette View Settings")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)

        row = QHBoxLayout()
        row.addWidget(QLabel("Columns"))
        cols_spin = QSpinBox()
        cols_spin.setRange(1, 32)
        cols_spin.setValue(self._view_columns)
        cols_spin.setFixedWidth(58)
        row.addWidget(cols_spin)
        row.addWidget(QLabel("Zoom"))
        zoom_spin = QSpinBox()
        zoom_spin.setRange(16, 96)
        zoom_spin.setValue(self._view_zoom)
        zoom_spin.setFixedWidth(58)
        row.addWidget(zoom_spin)
        row.addWidget(QLabel("Gap"))
        gap_spin = QSpinBox()
        gap_spin.setRange(-8, 20)
        gap_spin.setValue(self._view_gap)
        gap_spin.setFixedWidth(58)
        row.addWidget(gap_spin)
        row.addStretch(1)
        layout.addLayout(row)

        toggles = QHBoxLayout()
        show_indices = QCheckBox("Indices")
        show_indices.setChecked(self._view_show_indices)
        toggles.addWidget(show_indices)
        show_grid = QCheckBox("Grid")
        show_grid.setChecked(self._view_show_grid)
        toggles.addWidget(show_grid)
        force_cols = QCheckBox("Force palette view columns")
        force_cols.setChecked(self._view_force_columns)
        toggles.addWidget(force_cols)
        toggles.addStretch(1)
        layout.addLayout(toggles)

        def apply_settings() -> None:
            self._view_columns = cols_spin.value()
            self._view_force_columns = force_cols.isChecked()
            self._view_zoom = zoom_spin.value()
            self._view_gap = gap_spin.value()
            self._view_show_indices = show_indices.isChecked()
            self._view_show_grid = show_grid.isChecked()
            zoom_spin.setEnabled(not self._view_force_columns)
            self._apply_palette_view_options()
            self._save_palette_view_options()
            if self._view_force_columns:
                zoom_spin.blockSignals(True)
                zoom_spin.setValue(int(self._view_zoom))
                zoom_spin.blockSignals(False)

        cols_spin.valueChanged.connect(lambda _v: apply_settings())
        zoom_spin.valueChanged.connect(lambda _v: apply_settings())
        gap_spin.valueChanged.connect(lambda _v: apply_settings())
        show_indices.toggled.connect(lambda _v: apply_settings())
        show_grid.toggled.connect(lambda _v: apply_settings())
        force_cols.toggled.connect(lambda _v: apply_settings())

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        apply_settings()
        dialog.exec()

    def _on_sprite_pick_changed(self) -> None:
        self._refresh_preview_from_state()

    def _tick_destination_blink(self) -> None:
        if self._hover_palette_row is None:
            self._destination_blink_timer.stop()
            self._destination_blink_on = False
            return
        self._destination_blink_on = True
        self._destination_blink_phase += float(getattr(self._owner, "_animation_speed", 0.15))
        import math
        if self._destination_blink_phase > 2 * math.pi:
            self._destination_blink_phase -= 2 * math.pi
        self._refresh_preview_from_state()

    def _set_hover_palette_row(self, row: int | None) -> None:
        if self._hover_palette_row == row:
            return
        self._hover_palette_row = row
        self._update_destination_blink_state()
        self._refresh_preview_from_state()

    def _update_destination_blink_state(self) -> None:
        highlight_enabled = bool(getattr(self._owner, "highlight_checkbox", None) and self._owner.highlight_checkbox.isChecked())
        should_blink = self._hover_palette_row is not None
        if should_blink and highlight_enabled and not self._destination_blink_timer.isActive():
            self._destination_blink_on = True
            self._destination_blink_phase = 0.0
            self._destination_blink_timer.start()
            logger.debug("Merge dialog hover highlight started hover_row=%s", self._hover_palette_row)
            self._refresh_preview_from_state()
        elif (not should_blink or not highlight_enabled) and self._destination_blink_timer.isActive():
            self._destination_blink_timer.stop()
            self._destination_blink_on = False
            logger.debug("Merge dialog hover highlight stopped hover_row=%s", self._hover_palette_row)
            self._refresh_preview_from_state()

    def _refresh_preview_from_state(self) -> None:
        selection = self._selection_payload()
        if selection is None:
            self._refresh_preview(None, None, self._scope())
            return
        sources, destination = selection
        self._refresh_preview(sources, destination, self._scope())

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self.palette_view.viewport():
            if event.type() == QEvent.Type.MouseMove and isinstance(event, QMouseEvent):
                hover_index = self.palette_view.indexAt(event.pos())
                self._set_hover_palette_row(int(hover_index.row()) if hover_index.isValid() else None)
            elif event.type() in (QEvent.Type.Leave, QEvent.Type.HoverLeave):
                self._set_hover_palette_row(None)
            elif event.type() == QEvent.Type.Resize and self._view_force_columns:
                self._apply_palette_view_options()

        if obj is self.palette_view.viewport() and self.quick_assign_check.isChecked():
            if event.type() == QEvent.Type.MouseButtonPress and isinstance(event, QMouseEvent):
                index = self.palette_view.indexAt(event.pos())
                if not index.isValid():
                    return super().eventFilter(obj, event)
                row = int(index.row())
                if event.button() == Qt.MouseButton.LeftButton:
                    self._quick_assign_mouse_active = True
                    if self._destination_row == row:
                        self._destination_row = None
                        self._source_rows.add(row)
                    elif row in self._source_rows:
                        self._source_rows.remove(row)
                    else:
                        self._source_rows.add(row)
                    self._refresh_views()
                    self._update_destination_blink_state()
                    return True
                if event.button() == Qt.MouseButton.RightButton:
                    self._quick_assign_mouse_active = True
                    if self._destination_row == row:
                        self._destination_row = None
                    else:
                        self._destination_row = row
                        if row in self._source_rows:
                            self._source_rows.remove(row)
                    self._refresh_views()
                    self._update_destination_blink_state()
                    return True
                if event.button() == Qt.MouseButton.MiddleButton:
                    self._quick_assign_mouse_active = True
                    if row in self._source_rows:
                        self._source_rows.remove(row)
                    if self._destination_row == row:
                        self._destination_row = None
                    self._refresh_views()
                    self._update_destination_blink_state()
                    return True
            if self._quick_assign_mouse_active and event.type() == QEvent.Type.MouseMove:
                return True
            if self._quick_assign_mouse_active and event.type() == QEvent.Type.MouseButtonRelease:
                self._quick_assign_mouse_active = False
                return True
            if event.type() == QEvent.Type.ContextMenu:
                return True
        return super().eventFilter(obj, event)

    def _refresh_preview(
        self,
        sources: Sequence[int] | None,
        destination: int | None,
        scope: Literal["global", "group", "local"],
    ) -> None:
        selected_items = self.preview_sprite_list.selectedItems()
        preview_records: List[SpriteRecord] = []
        if selected_items:
            for item in selected_items:
                key = item.data(Qt.ItemDataRole.UserRole)
                record = self._owner.sprite_records.get(str(key)) if key else None
                if record is not None:
                    preview_records.append(record)
        if not preview_records:
            current = self._owner._current_record()
            if current is not None:
                preview_records = [current]

        selected_record = preview_records[0] if preview_records else None
        if selected_record is None:
            self.preview_pane.set_pixmap(None, reset_zoom=False)
            return
        self.preview_title_label.setText(f"Previewing: {selected_record.path.name}")
        pixmap = self._owner._build_merge_preview_pixmap(
            sources,
            destination,
            scope=scope,
            record=selected_record,
            highlight_index=self._hover_palette_row,
            blink_highlight=self._destination_blink_on,
            blink_phase=self._destination_blink_phase,
        )
        self.preview_pane.set_pixmap(pixmap, reset_zoom=False)

    def _refresh_views(self) -> None:
        scope = self._scope()
        records = self._owner._merge_scope_records(scope)
        self._scope_records = list(records)
        self._refresh_preview_sprite_picker(records)
        usage_counts = self._owner._usage_counts_for_records(records)
        self.palette_view.set_usage_counts(usage_counts, show_badge=True)
        self.palette_view.set_merge_roles(sorted(self._source_rows), self._destination_row)
        risk_map = self._owner._compute_destination_risk_levels(sorted(self._source_rows), records)
        self.palette_view.set_merge_candidate_risk(risk_map)

        selection = self._selection_payload()
        if selection is None:
            source_text = sorted(self._source_rows)
            source_value = source_text if source_text else "none"
            destination_value = self._destination_row if self._destination_row is not None else "none"
            self.selection_label.setText(
                "<span style='color:rgb(86,182,255)'><b>Source</b></span>: "
                f"{source_value} | "
                "<span style='color:rgb(196,120,255)'><b>Destination</b></span>: "
                f"{destination_value}"
            )
            if self._source_rows:
                safe_count = sum(1 for value in risk_map.values() if value == "safe")
                risky_count = sum(1 for value in risk_map.values() if value == "risky")
                self.impact_label.setText(
                    f"<b>Apply To</b>=<b>{scope}</b> sprites={len(records)} | "
                    f"Destination hints: "
                    f"<span style='color:#60d278'>will not lose pixel data={safe_count}</span> "
                    f"<span style='color:#ff6060'>will lose pixel data={risky_count}</span>"
                )
            else:
                self.impact_label.setText(f"<b>Apply To</b>=<b>{scope}</b> sprites={len(records)} | Tag Source and Destination")
            self.apply_button.setEnabled(False)
            self._refresh_group_list([], None)
            self._refresh_sprite_list([], None)
            self._refresh_preview(None, None, scope)
            self._update_destination_blink_state()
            return

        sources, destination = selection
        self.selection_label.setText(
            "<span style='color:rgb(86,182,255)'><b>Source</b></span>: "
            f"{sources} | "
            "<span style='color:rgb(196,120,255)'><b>Destination</b></span>: "
            f"{destination}"
        )
        impact = self._owner._analyze_merge_impact_for_records(sources, destination, records)
        safe_count = sum(1 for value in risk_map.values() if value == "safe")
        risky_count = sum(1 for value in risk_map.values() if value == "risky")
        lose_text = (
            "<span style='color:#ff6060'><b>Current selection: THIS MERGE WILL LOSE PIXEL DATA</b></span>"
            if impact["risky_sprites"] > 0
            else "<span style='color:#60d278'><b>Current selection: THIS MERGE WILL NOT LOSE PIXEL DATA</b></span>"
        )
        self.impact_label.setText(
            f"{lose_text} | <b>Apply To</b>=<b>{scope}</b> affected={impact['affected_sprites']}/{impact['total_sprites']} | "
            f"<span style='color:#60d278'>will not lose pixel data={safe_count}</span> "
            f"<span style='color:#ff6060'>will lose pixel data={risky_count}</span>"
        )
        self._refresh_group_list(records, impact)
        self._refresh_sprite_list(records, impact)
        self._refresh_preview(sources, destination, scope)
        self._update_destination_blink_state()
        self.apply_button.setEnabled(True)

    def _refresh_group_list(self, records: Sequence[SpriteRecord], impact: Dict[str, Any] | None) -> None:
        self.group_list.clear()
        if not records:
            return
        group_map: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "affected": 0, "risky": 0})
        affected_keys = set(impact.get("affected_keys", [])) if impact else set()
        risky_keys = set(impact.get("risky_keys", [])) if impact else set()
        for record in records:
            group_id = record.group_id or "Ungrouped"
            key = str(record.path)
            group_map[group_id]["total"] += 1
            if key in affected_keys:
                group_map[group_id]["affected"] += 1
            if key in risky_keys:
                group_map[group_id]["risky"] += 1
        for group_id in sorted(group_map.keys()):
            data = group_map[group_id]
            self.group_list.addItem(
                f"{group_id}: total={data['total']} affected={data['affected']} risk={data['risky']}"
            )

    def _refresh_sprite_list(self, records: Sequence[SpriteRecord], impact: Dict[str, Any] | None) -> None:
        self.sprite_list.clear()
        if not records:
            return
        affected_keys = set(impact.get("affected_keys", [])) if impact else set()
        risky_keys = set(impact.get("risky_keys", [])) if impact else set()
        usage_by_key = impact.get("source_hits_by_key", {}) if impact else {}
        for record in records:
            key = str(record.path)
            hits = usage_by_key.get(key, 0)
            marker = "RISK" if key in risky_keys else ("AFFECT" if key in affected_keys else "SAFE")
            item = QListWidgetItem(f"[{marker}] {record.path.name} source_hits={hits}")
            item.setData(Qt.ItemDataRole.UserRole, record.path.as_posix())
            icon_pixmap = record.pixmap.scaled(
                QSize(40, 40), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            item.setIcon(QIcon(icon_pixmap))
            if marker == "RISK":
                item.setBackground(QColor(100, 30, 30, 120))
            elif marker == "AFFECT":
                item.setBackground(QColor(90, 90, 35, 110))
            else:
                item.setBackground(QColor(25, 60, 25, 90))
            self.sprite_list.addItem(item)

    def _apply_merge(self) -> None:
        selection = self._selection_payload()
        if selection is None:
            return
        sources, destination = selection
        scope = self._scope()
        source_after_mode = str(self.source_after_combo.currentData() or "preserve_colors")
        applied = self._owner._apply_source_destination_merge(
            sources,
            destination,
            scope=scope,
            source_after_mode=source_after_mode,
            source_after_color=self._source_after_color,
        )
        if not applied:
            QMessageBox.warning(self, "Merge", "Merge operation did not apply.")
            return
        self._load_palette_snapshot()
        if self.clear_after_apply_check.isChecked():
            logger.debug("Merge dialog apply completed; clearing selection after apply")
            self._clear_all_selections()
        else:
            self._refresh_views()

    def _refresh_preview_sprite_picker(self, records: Sequence[SpriteRecord]) -> None:
        selected_keys = {item.data(Qt.ItemDataRole.UserRole) for item in self.preview_sprite_list.selectedItems()}
        self.preview_sprite_list.blockSignals(True)
        self.preview_sprite_list.clear()
        for record in records:
            item = QListWidgetItem(record.path.name)
            item.setData(Qt.ItemDataRole.UserRole, record.path.as_posix())
            icon_pixmap = record.pixmap.scaled(
                QSize(40, 40), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            item.setIcon(QIcon(icon_pixmap))
            self.preview_sprite_list.addItem(item)
            if record.path.as_posix() in selected_keys:
                item.setSelected(True)
        if self.preview_sprite_list.count() and not self.preview_sprite_list.selectedItems():
            current = self._owner._current_record()
            if current is not None:
                for row in range(self.preview_sprite_list.count()):
                    item = self.preview_sprite_list.item(row)
                    if item and item.data(Qt.ItemDataRole.UserRole) == current.path.as_posix():
                        item.setSelected(True)
                        break
        self.preview_sprite_list.blockSignals(False)


class KeybindingsDialog(QDialog):
    def __init__(self, entries: Sequence[Dict[str, Any]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(920, 620)

        self._rows: List[Dict[str, Any]] = []
        for raw in entries:
            row = dict(raw)
            row["bindings"] = self._normalize_binding_list(raw.get("bindings", []))
            row["default_bindings"] = self._normalize_binding_list(raw.get("default_bindings", []))
            self._rows.append(row)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(QLabel("Create as many bindings as needed per action. Each binding can be Global or Window scope."))

        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Action", "Current Bindings"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)

        root.addWidget(QLabel("Bindings for selected action"))
        self.binding_table = QTableWidget(0, 2, self)
        self.binding_table.setHorizontalHeaderLabels(["Shortcut", "Scope"])
        self.binding_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.binding_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.binding_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.binding_table.verticalHeader().setVisible(False)
        self.binding_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.binding_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        root.addWidget(self.binding_table)

        assign_row = QHBoxLayout()
        assign_row.addWidget(QLabel("Shortcut:"))
        self.sequence_edit = QKeySequenceEdit(self)
        assign_row.addWidget(self.sequence_edit, 1)
        self.global_check = QCheckBox("Global")
        self.global_check.setToolTip("Global: works app-wide; Window: active only in the main window/dialog context")
        assign_row.addWidget(self.global_check)
        self.add_binding_button = QPushButton("Add Binding")
        self.update_binding_button = QPushButton("Update Selected")
        self.remove_binding_button = QPushButton("Remove Selected")
        self.clear_button = QPushButton("Clear Input")
        assign_row.addWidget(self.add_binding_button)
        assign_row.addWidget(self.update_binding_button)
        assign_row.addWidget(self.remove_binding_button)
        assign_row.addWidget(self.clear_button)
        root.addLayout(assign_row)

        self.capture_status_label = QLabel("")
        root.addWidget(self.capture_status_label)

        actions_row = QHBoxLayout()
        self.import_button = QPushButton("Import...")
        self.export_button = QPushButton("Export...")
        self.reset_button = QPushButton("Reset Selected Action")
        self.reset_all_button = QPushButton("Reset All")
        actions_row.addWidget(self.import_button)
        actions_row.addWidget(self.export_button)
        actions_row.addWidget(self.reset_button)
        actions_row.addWidget(self.reset_all_button)
        actions_row.addStretch(1)
        root.addLayout(actions_row)

        footer_row = QHBoxLayout()
        self.dirty_status_label = QLabel("")
        footer_row.addWidget(self.dirty_status_label)
        footer_row.addStretch(1)
        self.save_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")
        footer_row.addWidget(self.save_button)
        footer_row.addWidget(self.cancel_button)
        root.addLayout(footer_row)

        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.binding_table.itemSelectionChanged.connect(self._on_binding_selection_changed)
        self.add_binding_button.clicked.connect(self._add_binding)
        self.update_binding_button.clicked.connect(self._update_binding)
        self.remove_binding_button.clicked.connect(self._remove_binding)
        self.clear_button.clicked.connect(self._clear_sequence_edit)
        self.import_button.clicked.connect(self._import_bindings)
        self.export_button.clicked.connect(self._export_bindings)
        self.reset_button.clicked.connect(self._reset_selected)
        self.reset_all_button.clicked.connect(self._reset_all)
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        self._initial_bindings = self.bindings()
        self._has_unsaved_changes = False
        self._refresh_items()

    @staticmethod
    def _normalize_shortcut_text(value: str) -> str:
        sequence = QKeySequence(str(value).strip())
        return sequence.toString(QKeySequence.SequenceFormat.NativeText).strip()

    def _normalize_binding_list(self, raw_bindings: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        candidate_list: List[Any]
        if isinstance(raw_bindings, dict):
            candidate_list = [raw_bindings]
        elif isinstance(raw_bindings, list):
            candidate_list = list(raw_bindings)
        elif isinstance(raw_bindings, str):
            candidate_list = [raw_bindings]
        else:
            candidate_list = []

        for item in candidate_list:
            shortcut_text = ""
            is_global = False
            if isinstance(item, dict):
                shortcut_text = self._normalize_shortcut_text(str(item.get("shortcut", "")))
                is_global = bool(item.get("global", False))
            else:
                shortcut_text = self._normalize_shortcut_text(str(item))
            if not shortcut_text:
                continue
            if any(existing["shortcut"].lower() == shortcut_text.lower() for existing in normalized):
                continue
            normalized.append({"shortcut": shortcut_text, "global": is_global})
        return normalized

    def bindings(self) -> Dict[str, Dict[str, Any]]:
        output: Dict[str, Dict[str, Any]] = {}
        for row in self._rows:
            action_id = str(row.get("id", "") or "").strip()
            if not action_id:
                continue
            output[action_id] = {
                "bindings": [
                    {"shortcut": str(binding.get("shortcut", "")).strip(), "global": bool(binding.get("global", False))}
                    for binding in self._normalize_binding_list(row.get("bindings", []))
                ]
            }
        return output

    def _update_dirty_state(self) -> None:
        self._has_unsaved_changes = self.bindings() != self._initial_bindings
        if self._has_unsaved_changes:
            self.dirty_status_label.setText("Unsaved shortcut changes")
            self.save_button.setEnabled(True)
        else:
            self.dirty_status_label.setText("No unsaved shortcut changes")
            self.save_button.setEnabled(False)

    def reject(self) -> None:
        if not self._has_unsaved_changes:
            super().reject()
            return
        reply = QMessageBox.question(
            self,
            "Unsaved Shortcuts",
            "You have unsaved shortcut changes. Save before closing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            self.accept()
            return
        if reply == QMessageBox.StandardButton.Discard:
            super().reject()

    def _selected_action_id(self) -> str | None:
        row_index = self.table.currentRow()
        if row_index < 0:
            return None
        item = self.table.item(row_index, 0)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value if value else None

    def _selected_row(self) -> Dict[str, Any] | None:
        action_id = self._selected_action_id()
        if not action_id:
            return None
        for row in self._rows:
            if str(row.get("id", "")) == action_id:
                return row
        return None

    def _selected_binding_index(self) -> int | None:
        row_index = self.binding_table.currentRow()
        return row_index if row_index >= 0 else None

    def _refresh_items(self) -> None:
        selected_id = self._selected_action_id()
        self.table.setRowCount(0)
        for row_index, row in enumerate(self._rows):
            action_id = str(row.get("id", "") or "")
            bindings = self._normalize_binding_list(row.get("bindings", []))
            defaults = self._normalize_binding_list(row.get("default_bindings", []))
            self.table.insertRow(row_index)

            action_item = QTableWidgetItem(str(row.get("label", "")))
            action_item.setData(Qt.ItemDataRole.UserRole, action_id)
            if bindings != defaults:
                action_item.setToolTip("Custom bindings")
            self.table.setItem(row_index, 0, action_item)

            if bindings:
                inline_parts = [
                    f"{binding.get('shortcut', '')} [{'G' if bool(binding.get('global', False)) else 'W'}]"
                    for binding in bindings
                ]
                inline_text = " | ".join(part for part in inline_parts if part.strip())
            else:
                inline_parts = []
                inline_text = "(unassigned)"
            bindings_item = QTableWidgetItem(inline_text)
            if inline_parts:
                bindings_item.setToolTip("\n".join(inline_parts))
            self.table.setItem(row_index, 1, bindings_item)

            if selected_id and selected_id == action_id:
                self.table.selectRow(row_index)

        if self.table.rowCount() and not self.table.selectionModel().hasSelection():
            self.table.selectRow(0)
        self._refresh_binding_items()
        self._update_dirty_state()

    def _refresh_binding_items(self) -> None:
        row = self._selected_row()
        selected_binding = self._selected_binding_index()
        self.binding_table.setRowCount(0)
        if row is None:
            self.add_binding_button.setEnabled(False)
            self.update_binding_button.setEnabled(False)
            self.remove_binding_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            return

        bindings = self._normalize_binding_list(row.get("bindings", []))
        row["bindings"] = bindings
        for index, binding in enumerate(bindings):
            self.binding_table.insertRow(index)
            shortcut_item = QTableWidgetItem(str(binding.get("shortcut", "")))
            scope_item = QTableWidgetItem("Global" if bool(binding.get("global", False)) else "Window")
            self.binding_table.setItem(index, 0, shortcut_item)
            self.binding_table.setItem(index, 1, scope_item)
        if selected_binding is not None and 0 <= selected_binding < self.binding_table.rowCount():
            self.binding_table.selectRow(selected_binding)
        elif self.binding_table.rowCount() > 0:
            self.binding_table.selectRow(0)

        self.add_binding_button.setEnabled(True)
        self.update_binding_button.setEnabled(self.binding_table.rowCount() > 0)
        self.remove_binding_button.setEnabled(self.binding_table.rowCount() > 0)
        self.reset_button.setEnabled(True)
        self._on_binding_selection_changed()

    def _on_selection_changed(self) -> None:
        self.capture_status_label.setText("")
        self._refresh_binding_items()

    def _on_binding_selection_changed(self) -> None:
        row = self._selected_row()
        binding_index = self._selected_binding_index()
        if row is None or binding_index is None:
            self.sequence_edit.blockSignals(True)
            self.sequence_edit.setKeySequence(QKeySequence())
            self.sequence_edit.blockSignals(False)
            self.global_check.blockSignals(True)
            self.global_check.setChecked(False)
            self.global_check.blockSignals(False)
            return
        bindings = self._normalize_binding_list(row.get("bindings", []))
        if binding_index < 0 or binding_index >= len(bindings):
            return
        selected = bindings[binding_index]
        self.sequence_edit.blockSignals(True)
        self.sequence_edit.setKeySequence(QKeySequence(str(selected.get("shortcut", ""))))
        self.sequence_edit.blockSignals(False)
        self.global_check.blockSignals(True)
        self.global_check.setChecked(bool(selected.get("global", False)))
        self.global_check.blockSignals(False)

    def _clear_sequence_edit(self) -> None:
        self.sequence_edit.clear()

    def _all_bindings_by_shortcut(self) -> Dict[str, str]:
        used: Dict[str, str] = {}
        for row in self._rows:
            action_id = str(row.get("id", "") or "").strip()
            for binding in self._normalize_binding_list(row.get("bindings", [])):
                shortcut = str(binding.get("shortcut", "")).strip().lower()
                if shortcut:
                    used[shortcut] = action_id
        return used

    def _label_for_action(self, action_id: str) -> str:
        for row in self._rows:
            if str(row.get("id", "")) == action_id:
                return str(row.get("label", action_id))
        return action_id

    def _add_binding(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        shortcut = self._normalize_shortcut_text(self.sequence_edit.keySequence().toString(QKeySequence.SequenceFormat.NativeText))
        if not shortcut:
            QMessageBox.warning(self, "Shortcut", "Shortcut cannot be empty.")
            return

        action_id = str(row.get("id", "") or "")
        used = self._all_bindings_by_shortcut()
        owner = used.get(shortcut.lower())
        if owner and owner != action_id:
            QMessageBox.warning(
                self,
                "Shortcut Conflict",
                f"'{shortcut}' is already assigned to {self._label_for_action(owner)}.",
            )
            return

        bindings = self._normalize_binding_list(row.get("bindings", []))
        if any(str(binding.get("shortcut", "")).lower() == shortcut.lower() for binding in bindings):
            QMessageBox.warning(self, "Shortcut", "This action already has that shortcut.")
            return
        bindings.append({"shortcut": shortcut, "global": bool(self.global_check.isChecked())})
        row["bindings"] = bindings
        self.capture_status_label.setText(f"Added binding {shortcut} to {row.get('label', 'Action')}.")
        self._refresh_items()

    def _update_binding(self) -> None:
        row = self._selected_row()
        binding_index = self._selected_binding_index()
        if row is None or binding_index is None:
            return
        shortcut = self._normalize_shortcut_text(self.sequence_edit.keySequence().toString(QKeySequence.SequenceFormat.NativeText))
        if not shortcut:
            QMessageBox.warning(self, "Shortcut", "Shortcut cannot be empty.")
            return

        action_id = str(row.get("id", "") or "")
        used = self._all_bindings_by_shortcut()
        owner = used.get(shortcut.lower())
        bindings = self._normalize_binding_list(row.get("bindings", []))
        existing_for_self = None
        for idx, binding in enumerate(bindings):
            if str(binding.get("shortcut", "")).lower() == shortcut.lower():
                existing_for_self = idx
                break
        if owner and owner != action_id:
            QMessageBox.warning(
                self,
                "Shortcut Conflict",
                f"'{shortcut}' is already assigned to {self._label_for_action(owner)}.",
            )
            return
        if existing_for_self is not None and existing_for_self != binding_index:
            QMessageBox.warning(self, "Shortcut", "This action already has that shortcut.")
            return
        if binding_index < 0 or binding_index >= len(bindings):
            return

        bindings[binding_index] = {"shortcut": shortcut, "global": bool(self.global_check.isChecked())}
        row["bindings"] = bindings
        self.capture_status_label.setText(f"Updated binding {shortcut} on {row.get('label', 'Action')}.")
        self._refresh_items()

    def _remove_binding(self) -> None:
        row = self._selected_row()
        binding_index = self._selected_binding_index()
        if row is None or binding_index is None:
            return
        bindings = self._normalize_binding_list(row.get("bindings", []))
        if binding_index < 0 or binding_index >= len(bindings):
            return
        removed = bindings.pop(binding_index)
        row["bindings"] = bindings
        self.capture_status_label.setText(f"Removed {removed.get('shortcut', '')} from {row.get('label', 'Action')}.")
        self._refresh_items()

    def _export_bindings(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Keyboard Shortcuts",
            "spritetools-shortcuts.json",
            "JSON Files (*.json)",
        )
        if not path:
            return
        payload = {
            "version": 2,
            "app": "SpriteTools",
            "bindings": self.bindings(),
        }
        try:
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Export Shortcuts", f"Failed to export bindings:\n{exc}")
            return
        self.capture_status_label.setText(f"Exported shortcuts to {Path(path).name}.")

    def _import_bindings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Keyboard Shortcuts",
            "",
            "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Import Shortcuts", f"Failed to read bindings file:\n{exc}")
            return

        source = loaded.get("bindings", loaded) if isinstance(loaded, dict) else None
        if not isinstance(source, dict):
            QMessageBox.warning(self, "Import Shortcuts", "Invalid file format.")
            return

        known_ids = {str(row.get("id", "")) for row in self._rows}
        incoming: Dict[str, List[Dict[str, Any]]] = {}
        for action_id, raw_value in source.items():
            action_key = str(action_id).strip()
            if not action_key or action_key not in known_ids:
                continue
            parsed: List[Dict[str, Any]] = []
            if isinstance(raw_value, dict) and isinstance(raw_value.get("bindings"), list):
                parsed = self._normalize_binding_list(raw_value.get("bindings", []))
            elif isinstance(raw_value, dict) and isinstance(raw_value.get("shortcuts"), (list, str)):
                shortcuts = raw_value.get("shortcuts", [])
                if isinstance(shortcuts, str):
                    shortcuts = [shortcuts]
                parsed = self._normalize_binding_list(
                    [{"shortcut": value, "global": bool(raw_value.get("global", False))} for value in shortcuts]
                )
            elif isinstance(raw_value, str):
                parsed = self._normalize_binding_list([raw_value])
            if parsed:
                incoming[action_key] = parsed

        if not incoming:
            QMessageBox.information(self, "Import Shortcuts", "No matching valid bindings were found.")
            return

        used: Dict[str, str] = {}
        for row in self._rows:
            action_id = str(row.get("id", "") or "")
            source_bindings = incoming.get(action_id, self._normalize_binding_list(row.get("bindings", [])))
            for binding in source_bindings:
                shortcut = str(binding.get("shortcut", "")).strip().lower()
                if not shortcut:
                    continue
                existing = used.get(shortcut)
                if existing and existing != action_id:
                    QMessageBox.warning(
                        self,
                        "Import Shortcuts",
                        f"Conflict detected: '{binding.get('shortcut', '')}' is assigned to both {self._label_for_action(existing)} and {self._label_for_action(action_id)}.",
                    )
                    return
                used[shortcut] = action_id

        for row in self._rows:
            action_id = str(row.get("id", "") or "")
            if action_id in incoming:
                row["bindings"] = incoming[action_id]

        self._refresh_items()
        self.capture_status_label.setText(
            f"Imported bindings for {len(incoming)} action{'s' if len(incoming) != 1 else ''} from {Path(path).name}."
        )

    def _reset_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        row["bindings"] = self._normalize_binding_list(row.get("default_bindings", []))
        self._refresh_items()

    def _reset_all(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset All Shortcuts",
            "Reset all shortcuts to default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        for row in self._rows:
            row["bindings"] = self._normalize_binding_list(row.get("default_bindings", []))
        self._refresh_items()


class TimelineRulerWidget(QWidget):
    playheadScrubbed = Signal(float)
    rangeScrubbed = Signal(int, int)
    rangeDragStarted = Signal()
    rangeDragFinished = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame_widths: List[int] = []
        self._frame_durations: List[int] = []
        self._duration_unit_px = 20
        self._zoom_percent = 100
        self._scroll_x = 0
        self._playhead_frame: float | None = None
        self._scrubbing = False
        self._range_in_frame = 0
        self._range_out_frame = 1
        self._range_drag_mode: Literal["none", "in", "out"] = "none"
        self.setMinimumHeight(30)
        self.setMaximumHeight(34)

    def set_timeline_layout(
        self,
        frame_widths: Sequence[int],
        frame_durations: Sequence[int],
        duration_unit_px: int,
        zoom_percent: int,
    ) -> None:
        self._frame_widths = [max(1, int(width)) for width in frame_widths]
        self._frame_durations = [max(1, int(value)) for value in frame_durations]
        if len(self._frame_durations) != len(self._frame_widths):
            self._frame_durations = [max(1, int(round(width / max(1, duration_unit_px)))) for width in self._frame_widths]
        self._duration_unit_px = max(1, int(duration_unit_px))
        self._zoom_percent = max(50, min(220, int(zoom_percent)))
        self.update()

    def set_scroll_x(self, value: int) -> None:
        clamped = max(0, int(value))
        if clamped == self._scroll_x:
            return
        self._scroll_x = clamped
        self.update()

    def set_playhead_frame(self, frame_position: float | None) -> None:
        normalized = None if frame_position is None else max(0.0, float(frame_position))
        if normalized == self._playhead_frame:
            return
        self._playhead_frame = normalized
        self.update()

    def set_range_frames(self, in_frame: int, out_frame: int) -> None:
        in_value = max(0, int(in_frame))
        out_value = max(in_value + 1, int(out_frame))
        if in_value == self._range_in_frame and out_value == self._range_out_frame:
            return
        self._range_in_frame = in_value
        self._range_out_frame = out_value
        self.update()

    def _total_timeline_frames(self) -> int:
        total = 0
        for duration in self._frame_durations:
            total += max(1, int(duration))
        return max(1, int(total))

    def _x_from_frame(self, frame_position: int) -> int:
        return int(round((max(0, int(frame_position)) * self._duration_unit_px) - self._scroll_x))

    def _try_start_range_drag(self, event: QMouseEvent) -> bool:
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return False
        pointer_x = int(round(float(event.position().x())))
        in_x = self._x_from_frame(self._range_in_frame)
        out_x = self._x_from_frame(self._range_out_frame)
        threshold = max(10, min(18, int(self._duration_unit_px // 2) + 4))
        d_in = abs(pointer_x - in_x)
        d_out = abs(pointer_x - out_x)
        if d_in > threshold and d_out > threshold:
            return False
        self._range_drag_mode = "in" if d_in <= d_out else "out"
        self.rangeDragStarted.emit()
        return True

    def _is_near_range_handle(self, event: QMouseEvent) -> bool:
        pointer_x = int(round(float(event.position().x())))
        in_x = self._x_from_frame(self._range_in_frame)
        out_x = self._x_from_frame(self._range_out_frame)
        threshold = max(10, min(18, int(self._duration_unit_px // 2) + 4))
        return abs(pointer_x - in_x) <= threshold or abs(pointer_x - out_x) <= threshold

    def _emit_range_drag(self, event: QMouseEvent) -> None:
        dragged_frame = int(round(self._frame_position_from_event(event)))
        if self._range_drag_mode == "in":
            in_value = max(0, min(dragged_frame, self._range_out_frame - 1))
            out_value = max(in_value + 1, self._range_out_frame)
        elif self._range_drag_mode == "out":
            out_value = max(1, dragged_frame)
            out_value = max(out_value, self._range_in_frame + 1)
            in_value = min(self._range_in_frame, out_value - 1)
        else:
            return
        self.rangeScrubbed.emit(int(in_value), int(out_value))

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rect = self.rect()
        top_band_height = 10
        ruler_top = rect.top() + top_band_height
        if ruler_top >= rect.bottom():
            ruler_top = rect.top() + 1

        painter.fillRect(rect, QColor(26, 26, 26))
        painter.fillRect(QRect(rect.left(), rect.top(), rect.width(), top_band_height), QColor(33, 33, 33))
        painter.setPen(QPen(QColor(56, 56, 56), 1))
        painter.drawLine(rect.left(), ruler_top, rect.right(), ruler_top)
        baseline_y = rect.bottom() - 1
        painter.setPen(QPen(QColor(78, 78, 78), 1))
        painter.drawLine(rect.left(), baseline_y, rect.right(), baseline_y)

        if not self._frame_widths:
            painter.end()
            return

        major_tick_every = max(1, 4)
        label_every_frame = self._zoom_percent >= 150
        world_x = 0
        cumulative_frames = 0
        text_y = ruler_top + 11

        for frame_index, frame_width in enumerate(self._frame_widths):
            screen_left = world_x - self._scroll_x
            screen_right = (world_x + frame_width) - self._scroll_x
            frame_units = self._frame_durations[frame_index] if frame_index < len(self._frame_durations) else max(1, int(round(frame_width / max(1, self._duration_unit_px))))

            if screen_right >= rect.left() and screen_left <= rect.right():
                painter.setPen(QPen(QColor(60, 60, 60), 1))
                painter.drawLine(int(screen_left), ruler_top, int(screen_left), baseline_y)

                step_px = frame_width / float(frame_units)
                painter.setPen(QPen(QColor(108, 108, 108), 1))
                for i in range(1, frame_units):
                    tick_x = int(round(screen_left + (i * step_px)))
                    tick_top = ruler_top + (6 if ((cumulative_frames + i) % major_tick_every) else 2)
                    painter.drawLine(tick_x, tick_top, tick_x, baseline_y)

            cumulative_frames += frame_units
            if label_every_frame or (cumulative_frames % major_tick_every == 0):
                label_x = int(screen_right + 2)
                if rect.left() <= label_x <= rect.right() - 22:
                    painter.setPen(QPen(QColor(175, 175, 175), 1))
                    painter.drawText(label_x, text_y, f"{cumulative_frames}f")

            world_x += frame_width

        if self._playhead_frame is not None:
            playhead_x = int(round((self._playhead_frame * self._duration_unit_px) - self._scroll_x))
            if rect.left() <= playhead_x <= rect.right():
                painter.setPen(QPen(QColor(255, 96, 96), 2))
                painter.drawLine(playhead_x, ruler_top, playhead_x, baseline_y)

        in_x = self._x_from_frame(self._range_in_frame)
        out_x = self._x_from_frame(self._range_out_frame)
        if rect.left() <= in_x <= rect.right():
            painter.setPen(QPen(QColor(110, 210, 130), 2))
            painter.drawLine(in_x, ruler_top, in_x, baseline_y)
            painter.setBrush(QColor(110, 210, 130))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(QPolygon([QPoint(in_x - 4, 1), QPoint(in_x + 4, 1), QPoint(in_x, 7)]))
        if rect.left() <= out_x <= rect.right():
            painter.setPen(QPen(QColor(255, 186, 90), 2))
            painter.drawLine(out_x, ruler_top, out_x, baseline_y)
            painter.setBrush(QColor(255, 186, 90))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(QPolygon([QPoint(out_x - 4, 1), QPoint(out_x + 4, 1), QPoint(out_x, 7)]))

        painter.setPen(QPen(QColor(92, 92, 92), 1))
        left_edge = -self._scroll_x
        right_edge = world_x - self._scroll_x
        painter.drawLine(int(left_edge), ruler_top, int(left_edge), baseline_y)
        painter.drawLine(int(right_edge), ruler_top, int(right_edge), baseline_y)
        painter.end()

    def _frame_position_from_event(self, event: QMouseEvent) -> float:
        x = float(event.position().x())
        world_x = x + float(self._scroll_x)
        return max(0.0, world_x / float(max(1, self._duration_unit_px)))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            if self._try_start_range_drag(event):
                self._emit_range_drag(event)
                event.accept()
                return
        if event.button() == Qt.MouseButton.LeftButton:
            self._scrubbing = True
            self.playheadScrubbed.emit(self._frame_position_from_event(event))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._range_drag_mode != "none" and bool(event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)):
            self._emit_range_drag(event)
            event.accept()
            return
        if self._range_drag_mode == "none" and not self._scrubbing:
            if self._is_near_range_handle(event):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.unsetCursor()
        if self._scrubbing and bool(event.buttons() & Qt.MouseButton.LeftButton):
            self.playheadScrubbed.emit(self._frame_position_from_event(event))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._range_drag_mode != "none" and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._emit_range_drag(event)
            self._range_drag_mode = "none"
            self.rangeDragFinished.emit()
            self.unsetCursor()
            event.accept()
            return
        if self._scrubbing and event.button() == Qt.MouseButton.LeftButton:
            self._scrubbing = False
            self.playheadScrubbed.emit(self._frame_position_from_event(event))
            event.accept()
            return
        super().mouseReleaseEvent(event)


class AnimationTimelineItemDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        kind = str(index.data(_TIMELINE_ITEM_KIND_ROLE) or "")
        rect = option.rect.adjusted(0, 0, -1, -1)
        if not rect.isValid():
            return

        selected = bool(option.state & QStyle.StateFlag.State_Selected)

        if kind == "gap":
            painter.save()
            painter.fillRect(rect, QColor(24, 24, 24, 0))
            painter.setPen(QPen(QColor(58, 58, 58), 1, Qt.PenStyle.DotLine))
            painter.drawRect(rect)
            painter.restore()
            return

        painter.save()
        body_color = QColor(54, 74, 96) if selected else QColor(48, 48, 48)
        border_color = QColor(170, 210, 255) if selected else QColor(96, 96, 96)
        painter.fillRect(rect, body_color)
        painter.setPen(QPen(border_color, 1))
        painter.drawRect(rect)

        top_band_h = max(8, min(18, rect.height() // 4))
        band_rect = QRect(rect.left() + 1, rect.top() + 1, max(1, rect.width() - 1), top_band_h)
        painter.fillRect(band_rect, QColor(52, 145, 158, 170 if selected else 135))

        frame_num_value = int(index.data(_TIMELINE_FRAME_NUMBER_ROLE) or 0)
        if frame_num_value > 0:
            top_num_font = painter.font()
            top_num_font.setBold(False)
            top_num_font.setPointSizeF(max(6.0, min(8.0, float(top_band_h) * 0.55)))
            painter.setFont(top_num_font)
            painter.setPen(QPen(QColor(216, 236, 246), 1))
            number_text = f"{frame_num_value}"
            painter.drawText(
                band_rect.adjusted(4, 0, -4, 0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                number_text,
            )

        icon_data = index.data(Qt.ItemDataRole.DecorationRole)
        icon = icon_data if isinstance(icon_data, QIcon) else None
        content_rect = rect.adjusted(6, 4, -6, -4)
        name_font = painter.font()
        name_font.setBold(False)
        name_font.setPointSizeF(max(6.0, min(8.5, float(rect.height()) * 0.10)))
        duration_font = painter.font()
        duration_font.setBold(False)
        duration_font.setPointSizeF(max(6.0, min(8.0, float(rect.height()) * 0.095)))
        label_fm = name_font
        duration_fm = duration_font
        painter.setFont(label_fm)
        info_line1_h = max(9, painter.fontMetrics().height())
        painter.setFont(duration_fm)
        info_line2_h = max(9, painter.fontMetrics().height())
        info_gap = max(1, min(4, rect.height() // 24))
        name_lines_h = (info_line1_h * 2) + info_gap
        info_total_h = name_lines_h + info_gap + info_line2_h
        info_rect = QRect(
            content_rect.left(),
            max(content_rect.top(), content_rect.bottom() - info_total_h + 1),
            max(1, content_rect.width()),
            max(1, info_total_h),
        )
        visual_rect = QRect(
            content_rect.left(),
            max(content_rect.top(), band_rect.bottom() + 4),
            max(1, content_rect.width()),
            max(1, info_rect.top() - max(content_rect.top(), band_rect.bottom() + 4) - 3),
        )

        if icon is not None and not visual_rect.isEmpty():
            base_thumb = min(max(18, int(option.decorationSize.width())), max(18, int(option.decorationSize.height())))
            if base_thumb <= 0:
                base_thumb = 42
            thumb_size = max(20, min(base_thumb, max(20, visual_rect.height() - 2)))
            thumb_rect = QRect(
                visual_rect.center().x() - (thumb_size // 2),
                visual_rect.center().y() - (thumb_size // 2),
                thumb_size,
                thumb_size,
            )
            thumb_pix = icon.pixmap(thumb_rect.size())
            painter.save()
            painter.setClipRect(visual_rect)
            painter.fillRect(thumb_rect, QColor(36, 36, 36))
            if not thumb_pix.isNull():
                draw_x = thumb_rect.left() + (thumb_rect.width() - thumb_pix.width()) // 2
                draw_y = thumb_rect.top() + (thumb_rect.height() - thumb_pix.height()) // 2
                painter.drawPixmap(draw_x, draw_y, thumb_pix)
            painter.setPen(QPen(QColor(112, 112, 112), 1))
            painter.drawRect(thumb_rect)
            painter.restore()

        frame_name = str(index.data(_TIMELINE_FRAME_NAME_ROLE) or "").strip()
        duration_value = str(index.data(_TIMELINE_FRAME_DURATION_ROLE) or "").strip()
        if not frame_name:
            frame_name = str(index.data(_TIMELINE_FRAME_LABEL_ROLE) or "").strip()
        if not frame_name:
            frame_name = str(index.data(Qt.ItemDataRole.DisplayRole) or "").strip().splitlines()[0]
        if not duration_value:
            duration_value = ""
        painter.setFont(name_font)
        line1 = painter.fontMetrics().elidedText(frame_name, Qt.TextElideMode.ElideRight, max(10, info_rect.width() - 2))
        painter.setFont(duration_font)
        line2 = painter.fontMetrics().elidedText(duration_value, Qt.TextElideMode.ElideRight, max(10, info_rect.width() - 2))
        name_rect = QRect(info_rect.left(), info_rect.top(), info_rect.width(), name_lines_h)
        line2_rect = QRect(info_rect.left(), info_rect.top() + name_lines_h + info_gap, info_rect.width(), info_line2_h)

        painter.setPen(QPen(QColor(224, 224, 224), 1))
        painter.setFont(name_font)
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, line1)
        painter.setPen(QPen(QColor(196, 196, 196), 1))
        painter.setFont(duration_font)
        painter.drawText(line2_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, line2)
        painter.restore()


class SpriteToolsWindow(QMainWindow):
    previewFutureDone = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SpriteTools GUI (prototype)")
        self.resize(1200, 700)
        self.sprite_records: Dict[str, SpriteRecord] = {}
        self.palette_colors: List[ColorTuple] = []
        self.palette_alphas: List[int] = []
        self._palette_slot_ids: List[int] = []
        self._detect_palette_colors: List[ColorTuple] = []
        self._detect_palette_alphas: List[int] = []
        self._detect_palette_slot_ids: List[int] = []
        self._palette_groups: Dict[str, PaletteGroup] = {}
        self._animation_tags: Dict[str, AnimationTag] = {}
        self._group_key_to_id: Dict[str, str] = {}
        self._next_group_id = 1
        self._slot_color_lookup: Dict[ColorTuple, List[int]] = {}
        self._next_slot_id = 0
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._handle_preview_timer_timeout)
        self._preview_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spritetools-preview")
        self._preview_future: Future[tuple[Image.Image, PaletteInfo]] | None = None
        self._preview_pending_request: Dict[str, Any] | None = None
        self._preview_active_request: Dict[str, Any] | None = None
        self._preview_request_serial = 0
        self.previewFutureDone.connect(self._drain_preview_result)
        self._preview_pixmap_timer = QTimer(self)
        self._preview_pixmap_timer.setSingleShot(True)
        self._preview_pixmap_timer.timeout.connect(self._flush_preview_pixmap_update)
        self._load_queue: deque[Path] = deque()
        self._load_mode_in_progress: Literal["detect", "preserve"] | None = None
        self._load_mode_overrides: Dict[str, Literal["detect", "preserve"]] = {}
        self._load_added_count = 0
        self._load_failed_count = 0
        self._load_total_count = 0
        self._load_had_selection = False
        self._load_starting_count = 0
        self._load_selected_key_before: str | None = None
        self._is_loading_sprites = False
        self._detect_batch_queue: deque[tuple[str, str, Any]] = deque()
        self._detect_batch_total = 0
        self._detect_batch_done = 0
        self._detect_batch_timer = QTimer(self)
        self._detect_batch_timer.setInterval(0)
        self._detect_batch_timer.timeout.connect(self._process_detect_batch_tick)
        self._load_timer = QTimer(self)
        self._load_timer.setInterval(0)
        self._load_timer.timeout.connect(self._process_load_queue_tick)
        self._animation_preview_timer = QTimer(self)
        self._animation_preview_timer.setSingleShot(True)
        self._animation_preview_timer.timeout.connect(self._advance_animation_preview)
        self._autosave_debounce_timer = QTimer(self)
        self._autosave_debounce_timer.setSingleShot(True)
        self._autosave_debounce_timer.timeout.connect(self._perform_autosave)
        self._autosave_periodic_timer = QTimer(self)
        self._autosave_periodic_timer.setInterval(60_000)
        self._autosave_periodic_timer.timeout.connect(self._perform_autosave)
        self._autosave_periodic_timer.start()
        self._autosave_dirty = False
        self._autosave_enabled = True
        self._autosave_recovery_mode = False
        self._autosave_last_error_ts = 0.0
        self._autosave_last_status_ts = 0.0
        self._autosave_roll_limit = 12
        self._last_project_saved_text: str | None = None
        self._last_project_saved_kind: str | None = None
        self._recovery_source_text: str | None = None
        self._main_palette_layout_timer = QTimer(self)
        self._main_palette_layout_timer.setSingleShot(True)
        self._main_palette_layout_timer.timeout.connect(self._apply_main_palette_layout_options)
        
        # Timer for smooth fade animation on selected color in preview
        self._highlight_animation_timer = QTimer(self)
        self._highlight_animation_timer.setInterval(30)  # ~33 FPS for smooth animation
        self._highlight_animation_timer.timeout.connect(self._update_highlight_animation)
        self._highlight_animation_phase = 0.0
        self._overlay_timer_interval_ms = 30
        
        # Customizable overlay settings
        self._selected_overlay_color = (255, 255, 255)
        self._selected_overlay_alpha_min = 77
        self._selected_overlay_alpha_max = 255
        self._selected_animation_speed = 0.15
        self._selected_highlight_animation_phase = 0.0

        self._hover_overlay_color = (255, 255, 255)
        self._hover_overlay_alpha_min = 77
        self._hover_overlay_alpha_max = 255
        self._hover_animation_speed = 0.15
        self._hover_highlight_animation_phase = 0.0

        self._overlay_show_both = False

        # Legacy aliases (used by merge preview and older helper paths)
        self._overlay_color = self._selected_overlay_color
        self._overlay_alpha_min = self._selected_overlay_alpha_min
        self._overlay_alpha_max = self._selected_overlay_alpha_max
        self._animation_speed = self._selected_animation_speed
        self._preview_hover_palette_row: int | None = None
        self._preview_bg_transparent_enabled = False
        self._preview_background_indices: set[int] = set()
        self._preview_view_mode: Literal["sprite_edit", "animation_assist"] = "sprite_edit"
        self._preview_animation_follow_selection = False
        self._preview_onion_source_mode: Literal["timeline", "sprite_list"] = "timeline"
        self._preview_onion_enabled = False
        self._preview_onion_prev_count = 2
        self._preview_onion_next_count = 2
        self._preview_onion_base_alpha = 110
        self._preview_onion_prev_alpha = 110
        self._preview_onion_next_alpha = 110
        self._preview_onion_prev_tint = (255, 120, 120)
        self._preview_onion_next_tint = (120, 170, 255)
        self._preview_onion_tint_strength = 90
        self._preview_onion_sprite_list_scope: Literal["all", "selected"] = "all"
        self._preview_onion_sprite_list_wrap = False
        self._active_offset_drag_mode: Literal["none", "global", "group", "individual"] = "none"
        self._offset_drag_live = False
        self._offset_drag_target_key: str | None = None
        self._offset_drag_group_id: str | None = None
        self._floating_palette_windows: List[FloatingPaletteWindow] = []
        self._floating_palette_window_counter = 0
        self._main_palette_columns = 8
        self._main_palette_force_columns = True
        self._main_palette_zoom = 42
        self._main_palette_gap = 6
        self._main_palette_show_indices = True
        self._main_palette_show_grid = False
        self._main_palette_show_usage_badge = False
        self._main_palette_layout_in_progress = False
        self._last_main_palette_layout_signature: tuple[Any, ...] | None = None
        self._merge_dialog: MergeOperationDialog | None = None
        self._hover_group_id: str | None = None
        self._used_index_cache: Dict[str, set[int]] = {}
        self._last_browser_sort_mode: str = "added"
        self._last_browser_view_mode: str = "list"
        self._sprite_icon_cache: OrderedDict[tuple[Any, ...], QIcon] = OrderedDict()
        self._inplace_refresh_timer = QTimer(self)
        self._inplace_refresh_timer.setSingleShot(True)
        self._inplace_refresh_timer.setInterval(0)
        self._inplace_refresh_timer.timeout.connect(self._continue_inplace_icon_refresh)
        self._inplace_refresh_active = False
        self._inplace_refresh_next_row = 0
        self._inplace_refresh_total = 0
        self._inplace_refresh_refreshed = 0
        self._inplace_refresh_start_ts = 0.0
        self._inplace_refresh_reason = "style-change"
        self._inplace_refresh_view_mode = "list"
        self._inplace_refresh_sort_mode = "added"
        self._animation_preview_tag_id: str | None = None
        self._animation_preview_frame_index = 0
        self._animation_preview_subframe_step = 0
        self._animation_preview_displayed_frame_index = -1
        self._animation_preview_visible_sprite_key: str | None = None
        self._animation_preview_timeline_position = 0
        self._animation_preview_current_duration_frames = 1
        self._animation_last_preview_size = QSize(0, 0)
        self._animation_preview_pixmap_cache: OrderedDict[tuple[Any, ...], QPixmap] = OrderedDict()
        self._animation_preview_index_cache: OrderedDict[tuple[Any, ...], tuple[int, int, bytes]] = OrderedDict()
        self._animation_assist_layer_cache: OrderedDict[tuple[Any, ...], tuple[QPixmap, QPixmap | None]] = OrderedDict()
        self._assist_interaction_refresh_pending = False
        self._assist_interaction_refresh_timer = QTimer(self)
        self._assist_interaction_refresh_timer.setSingleShot(True)
        self._assist_interaction_refresh_timer.timeout.connect(self._flush_assist_interaction_refresh)
        self._process_options_debug_next_log_at: Dict[str, float] = {}
        self._animation_frame_list_syncing = False
        self._animation_timeline_zoom = 100
        self._animation_timeline_in_frame = 0
        self._animation_timeline_out_frame = 1
        self._animation_timeline_range_syncing = False
        self._animation_timeline_range_history_before: tuple[int, int] | None = None
        self._animation_timeline_range_history_commit_timer = QTimer(self)
        self._animation_timeline_range_history_commit_timer.setSingleShot(True)
        self._animation_timeline_range_history_commit_timer.setInterval(250)
        self._animation_timeline_range_history_commit_timer.timeout.connect(self._commit_animation_timeline_range_history)
        self._animation_timeline_should_restore_visible = False
        self._is_app_closing = False
        self._animation_playhead_frame: float | None = None
        self._animation_manual_drag_active = False
        self._animation_manual_drag_source_row = -1
        self._animation_manual_drag_target_row = -1
        self._animation_manual_drag_selected_rows: List[int] = []
        self._animation_manual_drag_mode: Literal["none", "move", "resize"] = "none"
        self._animation_manual_drag_resize_edge: Literal["none", "left", "right"] = "none"
        self._animation_manual_drag_start_x = 0
        self._animation_manual_drag_resize_left_row = -1
        self._animation_manual_drag_resize_right_row = -1
        self._animation_manual_drag_resize_left_duration = 1
        self._animation_manual_drag_resize_right_duration = 1
        self._animation_manual_drag_changed = False
        self._animation_manual_drag_gap_target_timeline_position: int | None = None
        self._animation_manual_drag_session_seq = 0
        self._animation_manual_drag_session_id = 0
        self._settings = QSettings("SpriteTools", "SpriteTools")
        self._project_service = ProjectService()
        self._project_paths: ProjectPaths | None = None
        self._project_manifest: ProjectManifest | None = None
        self._pending_project_sprite_metadata: Dict[str, Dict[str, Any]] | None = None
        
        self._history_manager: HistoryManager | None = None
        self._pending_palette_index_remap: Dict[int, int] | None = None
        self.setAcceptDrops(True)

        self._main_splitter = QSplitter()

        self.images_panel = LoadedImagesPanel(
            self._on_selection_changed,
            on_browser_change=self._on_sprite_browser_controls_changed,
            on_browser_settings_change=self._save_sprite_browser_settings,
        )
        self.images_panel.list_widget.installEventFilter(self)
        self.images_panel.clear_button.clicked.connect(self._clear_all_sprites)

        self.action_import_sprites = QAction("Import Sprites...", self)
        self.action_new_project = QAction("New Project...", self)
        self.action_open_project = QAction("Open Project...", self)
        self.action_save_project = QAction("Save", self)
        self.action_save_project_as = QAction("Save Project As...", self)
        self.action_keyboard_shortcuts = QAction("Keyboard Shortcuts...", self)
        self.action_rename_group = QAction("Rename Selected Group", self)
        self.action_sprite_browser_settings = QAction("Sprite Browser Settings...", self)
        self.action_animation_timeline = QAction("Animation Timeline...", self)
        self.action_animation_play_pause = QAction("Animation Play/Pause", self)
        self.action_animation_seek_left_1f = QAction("Animation Seek 1f Left", self)
        self.action_animation_seek_right_1f = QAction("Animation Seek 1f Right", self)
        self.action_exit = QAction("Exit", self)
        self._recent_projects_menu = None
        self._action_shortcut_registry: List[tuple[str, QAction, str]] = [
            ("file/import_sprites", self.action_import_sprites, "Ctrl+I"),
            ("file/new_project", self.action_new_project, "Ctrl+N"),
            ("file/open_project", self.action_open_project, "Ctrl+O"),
            ("file/save_project", self.action_save_project, "Ctrl+S"),
            ("file/save_project_as", self.action_save_project_as, "Ctrl+Shift+S"),
            ("edit/keyboard_shortcuts", self.action_keyboard_shortcuts, "Ctrl+Alt+K"),
            ("groups/rename", self.action_rename_group, "F2"),
            ("view/sprite_browser_settings", self.action_sprite_browser_settings, "Ctrl+Alt+B"),
            ("view/animation_timeline", self.action_animation_timeline, "Ctrl+Alt+T"),
            ("animation/play_pause", self.action_animation_play_pause, "Space"),
            ("animation/seek_left_1f", self.action_animation_seek_left_1f, "Left"),
            ("animation/seek_right_1f", self.action_animation_seek_right_1f, "Right"),
            ("file/exit", self.action_exit, "Ctrl+Q"),
        ]

        self.action_import_sprites.triggered.connect(self._prompt_and_load)
        self.action_new_project.triggered.connect(self._new_project)
        self.action_open_project.triggered.connect(self._open_project)
        self.action_save_project.triggered.connect(self._save_project)
        self.action_save_project_as.triggered.connect(self._save_project_as)
        self.action_keyboard_shortcuts.triggered.connect(self._open_keybindings_dialog)
        self.action_rename_group.triggered.connect(self._rename_selected_group)
        self.action_sprite_browser_settings.triggered.connect(self._open_sprite_browser_settings)
        self.action_animation_timeline.triggered.connect(self._open_animation_timeline_dialog)
        self.action_animation_play_pause.triggered.connect(self._toggle_animation_play_pause_shortcut)
        self.action_animation_seek_left_1f.triggered.connect(lambda: self._seek_animation_by_delta(-1))
        self.action_animation_seek_right_1f.triggered.connect(lambda: self._seek_animation_by_delta(1))
        self.action_exit.triggered.connect(self.close)
        for _binding_key, action, _default_shortcut in self._action_shortcut_registry:
            self.addAction(action)

        self.images_panel.load_mode_combo.currentIndexChanged.connect(self._on_load_mode_changed)
        self.images_panel.new_group_button.clicked.connect(self._create_group_from_selected_sprites)
        self.images_panel.assign_group_button.clicked.connect(self._assign_selected_sprites_to_selected_group)
        self.images_panel.group_color_button.clicked.connect(self._set_selected_group_color)
        self.images_panel.rename_group_button.clicked.connect(self._rename_selected_group)
        self.images_panel.detach_group_button.clicked.connect(self._detach_selected_sprites_to_individual_groups)
        self.images_panel.auto_group_button.clicked.connect(self._auto_assign_selected_sprites_to_signature_groups)
        self.images_panel.group_list.itemDoubleClicked.connect(self._on_group_item_double_clicked)
        self.images_panel.group_list.itemEntered.connect(self._on_group_item_entered)
        self.images_panel.group_list.viewport().installEventFilter(self)
        self._refresh_project_info_banner()
        self._update_project_action_buttons()

        self.palette_panel = QWidget()
        palette_layout = QVBoxLayout(self.palette_panel)
        palette_layout.setContentsMargins(6, 6, 6, 6)
        palette_layout.setSpacing(6)
        palette_label = QLabel("Detected Colors")
        palette_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Hex color display for selected palette color
        self.selected_color_label = QLabel("No color selected")
        self.selected_color_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selected_color_label.setStyleSheet(
            "QLabel { "
            "background-color: #2b2b2b; "
            "color: #ffffff; "
            "padding: 8px; "
            "border: 2px solid #555; "
            "border-radius: 4px; "
            "font-size: 11pt; "
            "font-weight: bold; "
            "}"
        )
        self.selected_color_label.setMinimumHeight(40)
        
        self.palette_list = PaletteListView()
        self.palette_list.palette_changed.connect(self._on_palette_changed)
        self.palette_list.selection_changed.connect(self._on_palette_selection_changed)
        self.palette_list.viewport().installEventFilter(self)
        self.palette_list.setMouseTracking(True)
        self.palette_list.viewport().setMouseTracking(True)
        self._apply_main_palette_layout_options()
        palette_layout.addWidget(palette_label)
        palette_layout.addWidget(self.selected_color_label)
        palette_layout.addWidget(self.palette_list, 1)
        
        # Palette actions
        palette_actions = QHBoxLayout()
        self.merge_button = QPushButton("Merge Mode")
        self.merge_button.clicked.connect(self._open_merge_operation_dialog)
        self.merge_button.setEnabled(True)
        self.merge_button.setToolTip("Select multiple indexes, then set current index as Destination")
        palette_actions.addWidget(self.merge_button)
        self.open_palette_window_btn = QPushButton("Palette Window")
        self.open_palette_window_btn.clicked.connect(self._open_floating_palette_window)
        self.open_palette_window_btn.setToolTip("Open a floating palette inspector window")
        palette_actions.addWidget(self.open_palette_window_btn)
        self.main_palette_view_btn = QPushButton("View")
        self.main_palette_view_btn.clicked.connect(self._open_main_palette_view_settings)
        self.main_palette_view_btn.setToolTip("Configure main palette grid display")
        palette_actions.addWidget(self.main_palette_view_btn)
        self.animation_timeline_btn = QPushButton("Animation")
        self.animation_timeline_btn.clicked.connect(self._open_animation_timeline_dialog)
        self.animation_timeline_btn.setToolTip("Open animation timeline editor")
        palette_actions.addWidget(self.animation_timeline_btn)
        self.set_local_override_btn = QPushButton("Set Local")
        self.set_local_override_btn.setToolTip("Mark selected palette indices as local-only for current sprite")
        self.set_local_override_btn.clicked.connect(self._set_selected_indices_local_override)
        palette_actions.addWidget(self.set_local_override_btn)
        self.clear_local_override_btn = QPushButton("Clear Local")
        self.clear_local_override_btn.setToolTip("Clear local-only mark for selected palette indices")
        self.clear_local_override_btn.clicked.connect(self._clear_selected_indices_local_override)
        palette_actions.addWidget(self.clear_local_override_btn)
        palette_actions.addStretch(1)
        palette_actions.addWidget(QLabel("Shift"))
        self.shift_steps_spin = QSpinBox()
        self.shift_steps_spin.setRange(1, 255)
        self.shift_steps_spin.setValue(1)
        self.shift_steps_spin.setFixedWidth(56)
        self.shift_steps_spin.setToolTip("Shift selected contiguous index block by N slots")
        palette_actions.addWidget(self.shift_steps_spin)
        self.shift_left_button = QPushButton("←")
        self.shift_left_button.setFixedWidth(32)
        self.shift_left_button.setToolTip("Shift selected index block left")
        self.shift_left_button.clicked.connect(lambda: self._shift_selected_indices("left"))
        palette_actions.addWidget(self.shift_left_button)
        self.shift_right_button = QPushButton("→")
        self.shift_right_button.setFixedWidth(32)
        self.shift_right_button.setToolTip("Shift selected index block right")
        self.shift_right_button.clicked.connect(lambda: self._shift_selected_indices("right"))
        palette_actions.addWidget(self.shift_right_button)
        palette_layout.addLayout(palette_actions)
        
        # Auto-merge group
        auto_merge_group = QGroupBox("Auto-Merge Similar Colors")
        auto_merge_layout = QVBoxLayout(auto_merge_group)
        auto_merge_layout.setContentsMargins(6, 6, 6, 6)
        auto_merge_layout.setSpacing(4)
        
        tolerance_row = QHBoxLayout()
        tolerance_row.addWidget(QLabel("Tolerance:"))
        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(0, 255)
        self.tolerance_spin.setValue(10)
        self.tolerance_spin.setToolTip("Color distance threshold for auto-merge (0-255)")
        tolerance_row.addWidget(self.tolerance_spin)
        auto_merge_layout.addLayout(tolerance_row)
        
        self.auto_merge_button = QPushButton("Auto-Merge")
        self.auto_merge_button.clicked.connect(self._auto_merge_colors)
        auto_merge_layout.addWidget(self.auto_merge_button)
        
        palette_layout.addWidget(auto_merge_group)

        size_group = QGroupBox("Export Size")
        size_layout = QFormLayout(size_group)
        self._size_layout = size_layout
        size_layout.setContentsMargins(6, 6, 6, 6)
        self.output_size_mode = QComboBox()
        self.output_size_mode.addItem("Match sprite size", "native")
        self.output_size_mode.addItem("Custom canvas", "custom")
        self.output_size_mode.setCurrentIndex(1)
        self.output_size_mode.currentIndexChanged.connect(self._handle_size_mode_change)
        size_layout.addRow("Mode", self.output_size_mode)
        self.canvas_scope_combo = QComboBox()
        self.canvas_scope_combo.addItem("Combined", "combined")
        self.canvas_scope_combo.addItem("Global", "global")
        self.canvas_scope_combo.addItem("Per Group", "group")
        self.canvas_scope_combo.addItem("Local", "local")
        self.canvas_scope_combo.setCurrentIndex(0)
        self.canvas_scope_combo.currentIndexChanged.connect(self._handle_canvas_scope_change)
        size_layout.addRow("Scope", self.canvas_scope_combo)
        self.canvas_global_widget = QWidget()
        dims_row = QHBoxLayout(self.canvas_global_widget)
        dims_row.setContentsMargins(0, 0, 0, 0)
        dims_row.setSpacing(4)
        self.canvas_width_spin = QSpinBox()
        self.canvas_width_spin.setRange(1, 4096)
        self.canvas_width_spin.setValue(304)
        self.canvas_width_spin.valueChanged.connect(self._handle_canvas_dimension_change)
        self.canvas_height_spin = QSpinBox()
        self.canvas_height_spin.setRange(1, 4096)
        self.canvas_height_spin.setValue(224)
        self.canvas_height_spin.valueChanged.connect(self._handle_canvas_dimension_change)
        dims_row.addWidget(QLabel("W"))
        dims_row.addWidget(self.canvas_width_spin)
        dims_row.addWidget(QLabel("H"))
        dims_row.addWidget(self.canvas_height_spin)
        size_layout.addRow("Global", self.canvas_global_widget)
        self.canvas_group_widget = QWidget()
        group_dims_row = QHBoxLayout(self.canvas_group_widget)
        group_dims_row.setContentsMargins(0, 0, 0, 0)
        group_dims_row.setSpacing(4)
        self.group_canvas_width_spin = QSpinBox()
        self.group_canvas_width_spin.setRange(1, 4096)
        self.group_canvas_width_spin.setValue(304)
        self.group_canvas_width_spin.valueChanged.connect(self._handle_group_canvas_dimension_change)
        self.group_canvas_height_spin = QSpinBox()
        self.group_canvas_height_spin.setRange(1, 4096)
        self.group_canvas_height_spin.setValue(224)
        self.group_canvas_height_spin.valueChanged.connect(self._handle_group_canvas_dimension_change)
        group_dims_row.addWidget(QLabel("W"))
        group_dims_row.addWidget(self.group_canvas_width_spin)
        group_dims_row.addWidget(QLabel("H"))
        group_dims_row.addWidget(self.group_canvas_height_spin)
        size_layout.addRow("Per Group", self.canvas_group_widget)
        self.canvas_local_widget = QWidget()
        local_dims_row = QHBoxLayout(self.canvas_local_widget)
        local_dims_row.setContentsMargins(0, 0, 0, 0)
        local_dims_row.setSpacing(4)
        self.local_canvas_width_spin = QSpinBox()
        self.local_canvas_width_spin.setRange(1, 4096)
        self.local_canvas_width_spin.setValue(304)
        self.local_canvas_width_spin.valueChanged.connect(self._handle_local_canvas_dimension_change)
        self.local_canvas_height_spin = QSpinBox()
        self.local_canvas_height_spin.setRange(1, 4096)
        self.local_canvas_height_spin.setValue(224)
        self.local_canvas_height_spin.valueChanged.connect(self._handle_local_canvas_dimension_change)
        local_dims_row.addWidget(QLabel("W"))
        local_dims_row.addWidget(self.local_canvas_width_spin)
        local_dims_row.addWidget(QLabel("H"))
        local_dims_row.addWidget(self.local_canvas_height_spin)
        size_layout.addRow("Local", self.canvas_local_widget)
        palette_layout.addWidget(size_group)

        fill_group = QGroupBox("Canvas Fill")
        fill_layout = QFormLayout(fill_group)
        fill_layout.setContentsMargins(6, 6, 6, 6)
        self.fill_mode_combo = QComboBox()
        self.fill_mode_combo.addItem("Palette color", "palette")
        self.fill_mode_combo.addItem("Transparent", "transparent")
        self.fill_mode_combo.currentIndexChanged.connect(self._handle_fill_mode_change)
        fill_layout.addRow("Mode", self.fill_mode_combo)
        self.fill_index_spin = QSpinBox()
        self.fill_index_spin.setRange(0, 255)
        self.fill_index_spin.setValue(0)
        self.fill_index_spin.valueChanged.connect(self._handle_fill_index_change)
        fill_layout.addRow("Fill index", self.fill_index_spin)
        self.fill_preview_label = QLabel("—")
        self.fill_preview_label.setMinimumWidth(80)
        fill_layout.addRow("Preview", self.fill_preview_label)
        self.background_indices_btn = QPushButton("Background Indexes...")
        self.background_indices_btn.setToolTip("Select palette indexes treated as background for preview options")
        self.background_indices_btn.clicked.connect(self._open_background_index_selector)
        fill_layout.addRow("Background", self.background_indices_btn)
        palette_layout.addWidget(fill_group)
        
        # Global sprite offset controls
        offset_group = QGroupBox("Sprite Offset")
        offset_layout = QVBoxLayout(offset_group)
        offset_layout.setContentsMargins(6, 6, 6, 6)
        offset_layout.setSpacing(4)

        offset_scope_row = QHBoxLayout()
        offset_scope_row.addWidget(QLabel("Scope:"))
        self.offset_scope_combo = QComboBox()
        self.offset_scope_combo.addItem("Combined", "combined")
        self.offset_scope_combo.addItem("Global", "global")
        self.offset_scope_combo.addItem("Per Group", "group")
        self.offset_scope_combo.addItem("Local", "local")
        self.offset_scope_combo.setCurrentIndex(0)
        self.offset_scope_combo.currentIndexChanged.connect(self._handle_offset_scope_change)
        offset_scope_row.addWidget(self.offset_scope_combo)
        offset_scope_row.addStretch(1)
        offset_layout.addLayout(offset_scope_row)
        
        # Global offset controls
        self.global_offset_widget = QWidget()
        global_offset_row = QHBoxLayout(self.global_offset_widget)
        global_offset_row.setContentsMargins(0, 0, 0, 0)
        global_offset_row.addWidget(QLabel("Global:"))
        global_offset_row.addWidget(QLabel("X:"))
        self.global_offset_x_spin = QSpinBox()
        self.global_offset_x_spin.setRange(-4096, 4096)
        self.global_offset_x_spin.setValue(0)
        self.global_offset_x_spin.setToolTip("Global X offset applied to all sprites")
        self.global_offset_x_spin.valueChanged.connect(self._handle_global_offset_change)
        global_offset_row.addWidget(self.global_offset_x_spin)
        global_offset_row.addWidget(QLabel("Y:"))
        self.global_offset_y_spin = QSpinBox()
        self.global_offset_y_spin.setRange(-4096, 4096)
        self.global_offset_y_spin.setValue(0)
        self.global_offset_y_spin.setToolTip("Global Y offset applied to all sprites")
        self.global_offset_y_spin.valueChanged.connect(self._handle_global_offset_change)
        global_offset_row.addWidget(self.global_offset_y_spin)
        self.global_drag_mode_btn = QPushButton("Global Drag")
        self.global_drag_mode_btn.setCheckable(True)
        self.global_drag_mode_btn.setAutoExclusive(False)
        self.global_drag_mode_btn.setToolTip("Drag sprite in viewport to change GLOBAL offset")
        self.global_drag_mode_btn.clicked.connect(lambda: self._handle_drag_mode_button_click("global"))
        global_offset_row.addWidget(self.global_drag_mode_btn)
        offset_layout.addWidget(self.global_offset_widget)

        self.group_offset_widget = QWidget()
        group_offset_row = QHBoxLayout(self.group_offset_widget)
        group_offset_row.setContentsMargins(0, 0, 0, 0)
        group_offset_row.addWidget(QLabel("Per Group:"))
        group_offset_row.addWidget(QLabel("X:"))
        self.group_offset_x_spin = QSpinBox()
        self.group_offset_x_spin.setRange(-4096, 4096)
        self.group_offset_x_spin.setValue(0)
        self.group_offset_x_spin.setToolTip("Group-level X offset")
        self.group_offset_x_spin.valueChanged.connect(self._handle_group_offset_change)
        group_offset_row.addWidget(self.group_offset_x_spin)
        group_offset_row.addWidget(QLabel("Y:"))
        self.group_offset_y_spin = QSpinBox()
        self.group_offset_y_spin.setRange(-4096, 4096)
        self.group_offset_y_spin.setValue(0)
        self.group_offset_y_spin.setToolTip("Group-level Y offset")
        self.group_offset_y_spin.valueChanged.connect(self._handle_group_offset_change)
        group_offset_row.addWidget(self.group_offset_y_spin)
        self.group_drag_mode_btn = QPushButton("Group Drag")
        self.group_drag_mode_btn.setCheckable(True)
        self.group_drag_mode_btn.setAutoExclusive(False)
        self.group_drag_mode_btn.setToolTip("Drag sprite in viewport to change GROUP offset")
        self.group_drag_mode_btn.clicked.connect(lambda: self._handle_drag_mode_button_click("group"))
        group_offset_row.addWidget(self.group_drag_mode_btn)
        offset_layout.addWidget(self.group_offset_widget)
        
        # Per-sprite offset controls
        self.local_offset_widget = QWidget()
        per_sprite_offset_row = QHBoxLayout(self.local_offset_widget)
        per_sprite_offset_row.setContentsMargins(0, 0, 0, 0)
        per_sprite_offset_row.addWidget(QLabel("Local:"))
        per_sprite_offset_row.addWidget(QLabel("X:"))
        self.sprite_offset_x_spin = QSpinBox()
        self.sprite_offset_x_spin.setRange(-4096, 4096)
        self.sprite_offset_x_spin.setValue(0)
        self.sprite_offset_x_spin.setToolTip("Per-sprite X offset")
        self.sprite_offset_x_spin.valueChanged.connect(self._handle_sprite_offset_change)
        per_sprite_offset_row.addWidget(self.sprite_offset_x_spin)
        per_sprite_offset_row.addWidget(QLabel("Y:"))
        self.sprite_offset_y_spin = QSpinBox()
        self.sprite_offset_y_spin.setRange(-4096, 4096)
        self.sprite_offset_y_spin.setValue(0)
        self.sprite_offset_y_spin.setToolTip("Per-sprite Y offset")
        self.sprite_offset_y_spin.valueChanged.connect(self._handle_sprite_offset_change)
        per_sprite_offset_row.addWidget(self.sprite_offset_y_spin)
        self.individual_drag_mode_btn = QPushButton("Individual Drag")
        self.individual_drag_mode_btn.setCheckable(True)
        self.individual_drag_mode_btn.setAutoExclusive(False)
        self.individual_drag_mode_btn.setToolTip("Drag sprite in viewport to change INDIVIDUAL sprite offset")
        self.individual_drag_mode_btn.clicked.connect(lambda: self._handle_drag_mode_button_click("individual"))
        per_sprite_offset_row.addWidget(self.individual_drag_mode_btn)
        offset_layout.addWidget(self.local_offset_widget)
        
        palette_layout.addWidget(offset_group)

        self.animation_timeline_dialog = QDialog(self)
        self.animation_timeline_dialog.setWindowTitle("Animation Timeline")
        self.animation_timeline_dialog.setModal(False)
        self.animation_timeline_dialog.resize(980, 520)
        animation_layout = QVBoxLayout(self.animation_timeline_dialog)
        animation_layout.setContentsMargins(8, 8, 8, 8)
        animation_layout.setSpacing(6)

        self.animation_tag_list = QListWidget()
        self.animation_tag_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.animation_tag_list.setToolTip("Animation tags")
        animation_layout.addWidget(self.animation_tag_list, 1)

        animation_actions = QHBoxLayout()
        self.animation_tag_new_btn = QPushButton("New")
        self.animation_tag_delete_btn = QPushButton("Delete")
        self.animation_assign_selected_btn = QPushButton("Assign Sel")
        self.animation_clear_frames_btn = QPushButton("Clear")
        animation_actions.addWidget(self.animation_tag_new_btn)
        animation_actions.addWidget(self.animation_tag_delete_btn)
        animation_actions.addWidget(self.animation_assign_selected_btn)
        animation_actions.addWidget(self.animation_clear_frames_btn)
        animation_layout.addLayout(animation_actions)

        preview_row = QHBoxLayout()
        self.animation_play_btn = QPushButton("Play")
        self.animation_play_btn.setCheckable(True)
        self.animation_loop_check = QCheckBox("Loop")
        self.animation_loop_check.setChecked(True)
        preview_row.addWidget(self.animation_play_btn)
        preview_row.addWidget(self.animation_loop_check)
        preview_row.addWidget(QLabel("FPS:"))
        self.animation_fps_spin = QSpinBox()
        self.animation_fps_spin.setRange(1, 60)
        self.animation_fps_spin.setValue(12)
        self.animation_fps_spin.setFixedWidth(60)
        preview_row.addWidget(self.animation_fps_spin)
        preview_row.addWidget(QLabel("In:"))
        self.animation_in_spin = QSpinBox()
        self.animation_in_spin.setRange(0, 99999)
        self.animation_in_spin.setValue(0)
        self.animation_in_spin.setFixedWidth(70)
        preview_row.addWidget(self.animation_in_spin)
        preview_row.addWidget(QLabel("Out:"))
        self.animation_out_spin = QSpinBox()
        self.animation_out_spin.setRange(1, 100000)
        self.animation_out_spin.setValue(1)
        self.animation_out_spin.setFixedWidth(70)
        preview_row.addWidget(self.animation_out_spin)
        preview_row.addStretch(1)
        animation_layout.addLayout(preview_row)

        self.animation_status_label = QLabel("No tag selected")
        animation_layout.addWidget(self.animation_status_label)

        self.animation_timeline_ruler = TimelineRulerWidget()
        self.animation_timeline_ruler.setToolTip("Timeline ruler (frames)")
        animation_layout.addWidget(self.animation_timeline_ruler)

        self.animation_frame_list = QListWidget()
        self.animation_frame_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.animation_frame_list.setToolTip("Frames in selected animation tag")
        self.animation_frame_list.setViewMode(QListView.ViewMode.ListMode)
        self.animation_frame_list.setFlow(QListView.Flow.LeftToRight)
        self.animation_frame_list.setWrapping(False)
        self.animation_frame_list.setUniformItemSizes(False)
        self.animation_frame_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.animation_frame_list.setMovement(QListView.Movement.Snap)
        self.animation_frame_list.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.animation_frame_list.setDragEnabled(False)
        self.animation_frame_list.setAcceptDrops(False)
        self.animation_frame_list.viewport().setAcceptDrops(False)
        self.animation_frame_list.setDropIndicatorShown(False)
        self.animation_frame_list.setDragDropOverwriteMode(False)
        self.animation_frame_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.animation_frame_list.setAutoScroll(True)
        self.animation_frame_list.setAutoScrollMargin(24)
        self.animation_frame_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.animation_frame_list.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.animation_frame_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.animation_frame_list.setIconSize(QSize(56, 56))
        self.animation_frame_list.setMouseTracking(True)
        self.animation_frame_list.viewport().setMouseTracking(True)
        self.animation_frame_list.setMinimumHeight(76)
        self.animation_frame_list.setSpacing(0)
        self.animation_frame_list.setStyleSheet(
            "QListWidget { background-color: #1f1f1f; border: 1px solid #3f3f3f; }"
            "QListWidget::item { border: 0px; background: transparent; padding: 0px; margin: 0px; }"
            "QListWidget::item:selected { border: 1px solid #7fb6ff; background-color: rgba(80,120,160,140); }"
        )
        self._animation_timeline_item_delegate = AnimationTimelineItemDelegate(self.animation_frame_list)
        self.animation_frame_list.setItemDelegate(self._animation_timeline_item_delegate)
        self._animation_timeline_marker = QWidget(self.animation_frame_list.viewport())
        self._animation_timeline_marker.setVisible(False)
        self._animation_timeline_marker.setStyleSheet("background-color: #7fb6ff;")
        self._animation_timeline_marker.setFixedWidth(2)
        self._animation_timeline_drop_zone = QWidget(self.animation_frame_list.viewport())
        self._animation_timeline_drop_zone.setVisible(False)
        self._animation_timeline_drop_zone.setStyleSheet("background-color: rgba(127,182,255,70); border: 1px solid #7fb6ff;")
        animation_layout.addWidget(self.animation_frame_list)

        frame_edit_row = QHBoxLayout()
        frame_edit_row.addWidget(QLabel("Dur:"))
        self.animation_frame_duration_spin = QSpinBox()
        self.animation_frame_duration_spin.setRange(1, 120)
        self.animation_frame_duration_spin.setValue(1)
        self.animation_frame_duration_spin.setFixedWidth(70)
        frame_edit_row.addWidget(self.animation_frame_duration_spin)
        self.animation_prewarm_btn = QPushButton("Prewarm")
        self.animation_prewarm_btn.setToolTip("Precompute processed timeline frames for smoother playback")
        frame_edit_row.addWidget(self.animation_prewarm_btn)
        self.animation_frame_delete_btn = QPushButton("Delete Frame")
        self.animation_frame_delete_btn.setToolTip("Delete selected frame(s) from this animation tag")
        frame_edit_row.addWidget(self.animation_frame_delete_btn)
        self.animation_frame_move_left_btn = QPushButton("←")
        self.animation_frame_move_left_btn.setToolTip("Move selected frame left")
        frame_edit_row.addWidget(self.animation_frame_move_left_btn)
        self.animation_frame_move_right_btn = QPushButton("→")
        self.animation_frame_move_right_btn.setToolTip("Move selected frame right")
        frame_edit_row.addWidget(self.animation_frame_move_right_btn)
        frame_edit_row.addWidget(QLabel("Zoom:"))
        self.animation_timeline_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.animation_timeline_zoom_slider.setRange(50, 220)
        self.animation_timeline_zoom_slider.setSingleStep(5)
        self.animation_timeline_zoom_slider.setPageStep(10)
        self.animation_timeline_zoom_slider.setFixedWidth(150)
        frame_edit_row.addWidget(self.animation_timeline_zoom_slider)
        self.animation_timeline_zoom_value_label = QLabel("100%")
        self.animation_timeline_zoom_value_label.setMinimumWidth(42)
        frame_edit_row.addWidget(self.animation_timeline_zoom_value_label)
        frame_edit_row.addStretch(1)
        animation_layout.addLayout(frame_edit_row)

        timeline_footer = QHBoxLayout()
        timeline_footer.addStretch(1)
        self.animation_timeline_close_btn = QPushButton("Close")
        timeline_footer.addWidget(self.animation_timeline_close_btn)
        animation_layout.addLayout(timeline_footer)

        self.animation_tag_list.currentItemChanged.connect(self._on_animation_tag_selection_changed)
        self.animation_tag_new_btn.clicked.connect(self._create_animation_tag)
        self.animation_tag_delete_btn.clicked.connect(self._delete_selected_animation_tag)
        self.animation_assign_selected_btn.clicked.connect(self._assign_selected_sprites_to_animation_tag)
        self.animation_clear_frames_btn.clicked.connect(self._clear_selected_animation_tag_frames)
        self.animation_play_btn.toggled.connect(self._toggle_animation_preview)
        self.animation_fps_spin.valueChanged.connect(self._on_animation_preview_fps_changed)
        self.animation_in_spin.valueChanged.connect(self._on_animation_timeline_range_changed)
        self.animation_out_spin.valueChanged.connect(self._on_animation_timeline_range_changed)
        self.animation_frame_list.currentItemChanged.connect(self._on_animation_frame_selection_changed)
        self.animation_frame_list.model().rowsMoved.connect(self._on_animation_frame_rows_moved)
        self.animation_frame_list.horizontalScrollBar().valueChanged.connect(self.animation_timeline_ruler.set_scroll_x)
        self.animation_timeline_ruler.playheadScrubbed.connect(self._on_animation_timeline_ruler_scrubbed)
        self.animation_timeline_ruler.rangeScrubbed.connect(self._on_animation_timeline_ruler_range_scrubbed)
        self.animation_timeline_ruler.rangeDragStarted.connect(self._on_animation_timeline_range_edit_started)
        self.animation_timeline_ruler.rangeDragFinished.connect(self._on_animation_timeline_range_edit_finished)
        self.animation_frame_duration_spin.valueChanged.connect(self._on_animation_frame_duration_changed)
        self.animation_prewarm_btn.clicked.connect(self._prewarm_selected_animation_tag_preview)
        self.animation_frame_delete_btn.clicked.connect(self._delete_selected_animation_frames)
        self.animation_frame_move_left_btn.clicked.connect(lambda: self._move_selected_animation_frame(-1))
        self.animation_frame_move_right_btn.clicked.connect(lambda: self._move_selected_animation_frame(1))
        self.animation_timeline_zoom_slider.valueChanged.connect(self._on_animation_timeline_zoom_changed)
        self.animation_timeline_close_btn.clicked.connect(self.animation_timeline_dialog.close)
        self.animation_timeline_dialog.finished.connect(self._on_animation_timeline_dialog_finished)

        self.animation_frame_delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self.animation_frame_list)
        self.animation_frame_delete_shortcut.activated.connect(self._delete_selected_animation_frames)
        self.animation_frame_list.viewport().installEventFilter(self)

        self.animation_timeline_zoom_slider.setValue(self._animation_timeline_zoom)

        self._refresh_animation_tag_list()
        
        self._update_canvas_inputs()
        self._update_fill_preview()

        self.output_dir = Path.cwd() / "sprite_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_processed_indexed: Image.Image | None = None
        self._last_index_data: List[int] | None = None
        self._overlay_region_by_index: Dict[int, QRegion] = {}
        self._overlay_region_cache_shape: tuple[int, int] | None = None
        self._overlay_region_cache_token: Any = None
        self._overlay_region_multi_cache: OrderedDict[tuple[Any, int, int, int], QRegion] = OrderedDict()
        self._last_preview_rgba: Image.Image | None = None
        self._preview_base_pixmap: QPixmap | None = None
        self._last_palette_info: PaletteInfo | None = None
        self._reset_zoom_next = True
        self._palette_index_lookup: Dict[ColorTuple, List[int]] = {}
        self._preview_pan_active = False
        self._last_pan_overlay_refresh_ts = 0.0
        self._pan_overlay_refresh_interval_s = 0.033
        self._overlay_compose_ms_ema = 0.0
        self._overlay_compose_samples = 0
        self._overlay_perf_log_every = 30
        self._overlay_alpha_quantum = 8
        self._last_overlay_alpha_signature: tuple[Any, ...] | None = None
        self._last_preview_render_signature: tuple[Any, ...] | None = None

        self.preview_container = QWidget()
        preview_layout = QVBoxLayout(self.preview_container)
        preview_layout.setContentsMargins(6, 6, 6, 6)
        preview_layout.setSpacing(6)
        self.preview_panel = PreviewPane()
        self.preview_panel.drag_offset_changed.connect(self._handle_drag_offset_changed)
        self.preview_panel.drag_started.connect(self._handle_drag_started)
        self.preview_panel.drag_finished.connect(self._handle_drag_finished)
        self.preview_panel.panning_changed.connect(self._handle_preview_panning_changed)
        preview_layout.addWidget(self.preview_panel, 1)

        # Single controls row: highlight checkbox, overlay settings button, and scaling
        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("View:"))
        self.preview_context_combo = QComboBox()
        self.preview_context_combo.addItem("Sprite Edit", "sprite_edit")
        self.preview_context_combo.addItem("Animation Assist", "animation_assist")
        self.preview_context_combo.setToolTip("Choose whether preview follows sprite edits or animation timeline assistance")
        self.preview_context_combo.currentIndexChanged.connect(self._on_preview_context_changed)
        controls_row.addWidget(self.preview_context_combo)

        self.preview_animation_follow_selection_check = QCheckBox("Follow Selection")
        self.preview_animation_follow_selection_check.setToolTip(
            "Animation Assist: when enabled, selection changes move onion source/playhead; when disabled, source stays locked"
        )
        self.preview_animation_follow_selection_check.toggled.connect(self._on_preview_animation_follow_selection_changed)
        controls_row.addWidget(self.preview_animation_follow_selection_check)

        controls_row.addWidget(QLabel("Onion Source:"))
        self.preview_onion_source_combo = QComboBox()
        self.preview_onion_source_combo.addItem("Timeline", "timeline")
        self.preview_onion_source_combo.addItem("Sprite List", "sprite_list")
        self.preview_onion_source_combo.setToolTip("Select onion source model")
        self.preview_onion_source_combo.currentIndexChanged.connect(self._on_preview_onion_source_changed)
        controls_row.addWidget(self.preview_onion_source_combo)

        self.preview_onion_enabled_check = QCheckBox("Onion")
        self.preview_onion_enabled_check.setToolTip("Enable onion skin compositing in Animation Assist view")
        self.preview_onion_enabled_check.toggled.connect(self._on_preview_onion_enabled_changed)
        controls_row.addWidget(self.preview_onion_enabled_check)

        self.preview_onion_settings_btn = QPushButton("Onion Settings...")
        self.preview_onion_settings_btn.clicked.connect(self._open_onion_settings)
        controls_row.addWidget(self.preview_onion_settings_btn)

        self.highlight_checkbox = QCheckBox("Highlight selected color")
        self.highlight_checkbox.setChecked(True)
        self.highlight_checkbox.toggled.connect(self._on_highlight_checkbox_toggled)
        controls_row.addWidget(self.highlight_checkbox)
        self.hover_highlight_checkbox = QCheckBox("Highlight hover color")
        self.hover_highlight_checkbox.setChecked(True)
        self.hover_highlight_checkbox.toggled.connect(self._on_hover_highlight_checkbox_toggled)
        controls_row.addWidget(self.hover_highlight_checkbox)
        
        self.overlay_settings_btn = QPushButton("Overlay Settings...")
        self.overlay_settings_btn.clicked.connect(self._open_overlay_settings)
        controls_row.addWidget(self.overlay_settings_btn)
        
        controls_row.addStretch(1)
        controls_row.addWidget(QLabel("Scaling:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("Nearest")
        self.filter_combo.addItem("Bilinear")
        self.filter_combo.currentIndexChanged.connect(self._handle_filter_mode_change)
        controls_row.addWidget(self.filter_combo)
        preview_layout.addLayout(controls_row)

        output_row = QHBoxLayout()
        self.output_dir_button = QPushButton("Choose Output Folder")
        self.output_dir_button.clicked.connect(self._choose_output_dir)
        self.output_dir_label = QLabel(str(self.output_dir))
        self.output_dir_label.setWordWrap(True)
        output_row.addWidget(self.output_dir_button)
        output_row.addWidget(self.output_dir_label, 1)
        preview_layout.addLayout(output_row)

        export_row = QHBoxLayout()
        self.export_selected_btn = QPushButton("Export Selected")
        self.export_selected_btn.clicked.connect(self._export_selected)
        self.export_all_btn = QPushButton("Export All")
        self.export_all_btn.clicked.connect(self._export_all)
        self.export_palette_btn = QPushButton("Export Palette Only")
        self.export_palette_btn.clicked.connect(self._export_palette_only)
        self.export_palette_btn.setToolTip("Export only the palette as .act file")
        export_row.addWidget(self.export_selected_btn)
        export_row.addWidget(self.export_all_btn)
        export_row.addWidget(self.export_palette_btn)
        preview_layout.addLayout(export_row)

        self._main_splitter.addWidget(self.images_panel)
        self._main_splitter.addWidget(self.palette_panel)
        self._main_splitter.addWidget(self.preview_container)
        self._main_splitter.setSizes([250, 500, 450])

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._main_splitter)
        self.setCentralWidget(container)
        self._build_menu_bar()
        self.statusBar().showMessage("Ready")
        if DEBUG_LOG_PATH:
            self.statusBar().showMessage(f"Debug log: {DEBUG_LOG_PATH}", 5000)
        self._update_export_buttons()
        self._update_loaded_count()
        self._setup_history_manager()
        self._install_shortcuts()
        self._load_persistent_ui_state()
        self._reset_history()

    def _get_pref_bool(self, key: str, default: bool) -> bool:
        value = self._settings.value(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _get_pref_int(self, key: str, default: int) -> int:
        value = self._settings.value(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _get_pref_float(self, key: str, default: float) -> float:
        value = self._settings.value(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _set_pref(self, key: str, value: Any) -> None:
        self._settings.setValue(key, value)

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.clear()
        menu_bar.setStyleSheet(
            "QMenuBar { padding: 0px 0px 0px 6px; margin: 0px; }"
            "QMenuBar::item { padding: 1px 7px; margin: 0px; }"
        )
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.action_import_sprites)
        file_menu.addSeparator()
        file_menu.addAction(self.action_new_project)
        file_menu.addAction(self.action_open_project)
        self._recent_projects_menu = file_menu.addMenu("Open &Recent")
        self._recent_projects_menu.aboutToShow.connect(self._populate_recent_projects_menu)
        file_menu.addSeparator()
        file_menu.addAction(self.action_save_project)
        file_menu.addAction(self.action_save_project_as)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addAction(self.action_keyboard_shortcuts)
        edit_menu.addAction(self.action_rename_group)

        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(self.action_sprite_browser_settings)
        view_menu.addAction(self.action_animation_timeline)

    def _populate_recent_projects_menu(self) -> None:
        menu = self._recent_projects_menu
        if menu is None:
            return
        menu.clear()
        recent_paths = self._recent_project_manifest_paths()
        if not recent_paths:
            empty_action = menu.addAction("(No Recent Projects)")
            empty_action.setEnabled(False)
            clear_action = menu.addAction("Clear Recent List")
            clear_action.setEnabled(False)
            return
        for manifest_path in recent_paths:
            root = manifest_path.parent
            label = f"{root.name} — {manifest_path.as_posix()}"
            action = menu.addAction(label)
            action.triggered.connect(lambda checked=False, value=manifest_path: self._open_project_from_path(value))
        menu.addSeparator()
        clear_action = menu.addAction("Clear Recent List")
        clear_action.triggered.connect(self._clear_recent_projects)

    def _clear_recent_projects(self) -> None:
        self._set_recent_project_manifest_paths([])
        self.statusBar().showMessage("Cleared recent project list", 2500)
        self._update_project_action_buttons()

    def _binding_entries_for_dialog(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for binding_key, action, default_shortcut in self._action_shortcut_registry:
            default_norm = KeybindingsDialog._normalize_shortcut_text(default_shortcut)
            entries.append(
                {
                    "id": binding_key,
                    "label": action.text().replace("&", ""),
                    "default_bindings": [{"shortcut": default_norm, "global": False}] if default_norm else [],
                    "bindings": self._get_key_bindings(binding_key, default_shortcut),
                }
            )

        undo_default = QKeySequence(QKeySequence.StandardKey.Undo).toString() or "Ctrl+Z"
        redo_default = QKeySequence(QKeySequence.StandardKey.Redo).toString() or "Ctrl+Y"
        entries.append(
            {
                "id": "edit/undo",
                "label": "Undo",
                "default_bindings": [{"shortcut": KeybindingsDialog._normalize_shortcut_text(undo_default), "global": False}],
                "bindings": self._get_key_bindings("edit/undo", undo_default),
            }
        )
        entries.append(
            {
                "id": "edit/redo",
                "label": "Redo",
                "default_bindings": [{"shortcut": KeybindingsDialog._normalize_shortcut_text(redo_default), "global": False}],
                "bindings": self._get_key_bindings("edit/redo", redo_default),
            }
        )

        extra_bindings: List[tuple[str, str, str]] = [
            ("merge.apply", "Merge Mode: Apply Merge", "A"),
            ("merge.tag_source", "Merge Mode: Tag Selected as Source", "S"),
            ("merge.tag_destination", "Merge Mode: Tag Current as Destination", "D"),
            ("merge.clear_roles", "Merge Mode: Clear Source/Destination", "C"),
            ("merge.clear_all", "Merge Mode: Clear All Selections", "Shift+C"),
            ("merge.scope_global", "Merge Mode: Scope Global", "Alt+1"),
            ("merge.scope_group", "Merge Mode: Scope Group", "Alt+2"),
            ("merge.scope_local", "Merge Mode: Scope Local", "Alt+3"),
            ("merge.view_settings", "Merge Mode: Open View Settings", "V"),
            ("merge.close", "Merge Mode: Close Dialog", "Escape"),
            ("browser/sprites_zoom_in", "Sprite Browser: Zoom In", "Ctrl+="),
            ("browser/sprites_zoom_out", "Sprite Browser: Zoom Out", "Ctrl+-"),
            ("browser/sprites_zoom_reset", "Sprite Browser: Reset Zoom", "Ctrl+0"),
        ]
        for binding_id, label, default_value in extra_bindings:
            default_norm = KeybindingsDialog._normalize_shortcut_text(default_value)
            entries.append(
                {
                    "id": binding_id,
                    "label": label,
                    "default_bindings": [{"shortcut": default_norm, "global": False}] if default_norm else [],
                    "bindings": self._get_key_bindings(binding_id, default_value),
                }
            )
        return entries

    def _open_keybindings_dialog(self) -> None:
        dialog = KeybindingsDialog(self._binding_entries_for_dialog(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        for binding_key, payload in dialog.bindings().items():
            bindings = payload.get("bindings", []) if isinstance(payload, dict) else []
            if not isinstance(bindings, list):
                bindings = []
            normalized_bindings: List[Dict[str, Any]] = []
            for raw in bindings:
                if not isinstance(raw, dict):
                    continue
                shortcut = KeybindingsDialog._normalize_shortcut_text(str(raw.get("shortcut", "")))
                if not shortcut:
                    continue
                normalized_bindings.append({"shortcut": shortcut, "global": bool(raw.get("global", False))})
            self._set_pref(f"bindings/{binding_key}", json.dumps({"bindings": normalized_bindings}))
        self._settings.sync()
        self._apply_configured_shortcuts()
        self.statusBar().showMessage("Keyboard shortcuts updated", 2500)

    def _open_sprite_browser_settings(self) -> None:
        self.images_panel.open_browser_settings_dialog()

    def _open_animation_timeline_dialog(self) -> None:
        if self.animation_timeline_dialog.isVisible():
            self.animation_timeline_dialog.raise_()
            self.animation_timeline_dialog.activateWindow()
            return
        self._animation_timeline_should_restore_visible = True
        self.animation_timeline_dialog.show()
        self.animation_timeline_dialog.raise_()
        self.animation_timeline_dialog.activateWindow()
        self._save_animation_timeline_ui_settings()

    def _on_animation_timeline_dialog_finished(self, _result: int) -> None:
        if self._is_app_closing:
            return
        self._animation_timeline_should_restore_visible = False
        self._save_animation_timeline_ui_settings()

    def _save_window_ui_settings(self) -> None:
        self._set_pref("window/geometry", self.saveGeometry())
        self._set_pref("window/state", self.saveState())
        self._set_pref("window/maximized", bool(self.isMaximized()))
        if hasattr(self, "_main_splitter"):
            sizes = self._main_splitter.sizes()
            self._set_pref("window/main_splitter_sizes", ",".join(str(int(value)) for value in sizes))

    def _load_window_ui_settings(self) -> None:
        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            try:
                self.restoreGeometry(geometry)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to restore main window geometry", exc_info=True)

        state = self._settings.value("window/state")
        if state is not None:
            try:
                self.restoreState(state)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to restore main window state", exc_info=True)

        splitter_raw = str(self._settings.value("window/main_splitter_sizes", "")).strip()
        if splitter_raw and hasattr(self, "_main_splitter"):
            parsed_sizes: List[int] = []
            for token in splitter_raw.split(","):
                try:
                    parsed_sizes.append(max(0, int(token.strip())))
                except ValueError:
                    parsed_sizes.clear()
                    break
            if len(parsed_sizes) == self._main_splitter.count():
                self._main_splitter.setSizes(parsed_sizes)

        if self._get_pref_bool("window/maximized", False):
            self.showMaximized()

    def _save_load_mode_setting(self) -> None:
        self._set_pref("import/load_mode", self.images_panel.selected_load_mode())

    def _save_sprite_browser_settings(self) -> None:
        self._set_pref("browser/sprites_view_mode", self.images_panel.browser_view_mode())
        self._set_pref("browser/sprites_sort_mode", self.images_panel.browser_sort_mode())
        self._set_pref("browser/sprites_zoom", int(self.images_panel.browser_zoom()))
        self._set_pref("browser/sprites_zoom_list", int(self.images_panel.browser_zoom_for_mode("list")))
        self._set_pref("browser/sprites_zoom_thumbnails", int(self.images_panel.browser_zoom_for_mode("thumbnails")))
        self._set_pref("browser/sprites_group_marker_mode", self.images_panel.browser_group_marker_mode())
        self._set_pref("browser/sprites_group_square_thickness", int(self.images_panel.browser_group_square_thickness()))
        self._set_pref("browser/sprites_group_square_padding", int(self.images_panel.browser_group_square_padding()))
        self._set_pref("browser/sprites_group_square_fill_alpha", int(self.images_panel.browser_group_square_fill_alpha()))
        self._set_pref("browser/sprites_scroll_speed", int(self.images_panel.browser_scroll_speed()))

    def _load_load_mode_setting(self) -> None:
        mode_value = str(self._settings.value("import/load_mode", "detect")).strip().lower()
        desired_mode: Literal["detect", "preserve"] = "preserve" if mode_value == "preserve" else "detect"
        combo = self.images_panel.load_mode_combo
        idx = combo.findData(desired_mode)
        if idx < 0:
            idx = 0
        was_blocked = combo.blockSignals(True)
        combo.setCurrentIndex(idx)
        combo.blockSignals(was_blocked)
        logger.debug("Loaded import mode setting mode=%s index=%s", desired_mode, idx)

    def _load_sprite_browser_settings(self) -> None:
        view_mode = str(self._settings.value("browser/sprites_view_mode", "list") or "list").strip().lower()
        sort_mode = str(self._settings.value("browser/sprites_sort_mode", "added") or "added").strip().lower()
        group_marker_mode = str(
            self._settings.value("browser/sprites_group_marker_mode", "text") or "text"
        ).strip().lower()
        group_square_thickness = self._get_pref_int("browser/sprites_group_square_thickness", 3)
        group_square_padding = self._get_pref_int("browser/sprites_group_square_padding", 1)
        group_square_fill_alpha = self._get_pref_int("browser/sprites_group_square_fill_alpha", 175)
        scroll_speed = self._get_pref_int("browser/sprites_scroll_speed", 3)
        zoom = self._get_pref_int("browser/sprites_zoom", 64)
        list_zoom = self._get_pref_int("browser/sprites_zoom_list", zoom)
        thumbnails_zoom = self._get_pref_int("browser/sprites_zoom_thumbnails", zoom)
        self.images_panel.apply_browser_settings(
            view_mode=view_mode,
            sort_mode=sort_mode,
            zoom=zoom,
            list_zoom=list_zoom,
            thumbnails_zoom=thumbnails_zoom,
            group_marker_mode=group_marker_mode,
            group_square_thickness=group_square_thickness,
            group_square_padding=group_square_padding,
            group_square_fill_alpha=group_square_fill_alpha,
            scroll_speed=scroll_speed,
        )
        self._last_browser_sort_mode = self.images_panel.browser_sort_mode()
        self._last_browser_view_mode = self.images_panel.browser_view_mode()
        logger.debug(
            "Loaded sprite browser settings view=%s sort=%s zoom=%s list_zoom=%s thumbs_zoom=%s marker=%s border=%s pad=%s fill=%s scroll=%s",
            view_mode,
            sort_mode,
            zoom,
            list_zoom,
            thumbnails_zoom,
            group_marker_mode,
            group_square_thickness,
            group_square_padding,
            group_square_fill_alpha,
            scroll_speed,
        )

    def _save_persistent_ui_state(self) -> None:
        self._save_preview_ui_settings()
        self._save_load_mode_setting()
        self._save_sprite_browser_settings()
        self._save_window_ui_settings()
        self._save_animation_timeline_ui_settings()
        self._settings.sync()

    def _load_persistent_ui_state(self) -> None:
        self._load_ui_settings()
        self._load_load_mode_setting()
        self._load_sprite_browser_settings()
        self._load_window_ui_settings()
        self._load_animation_timeline_ui_settings()

    def _save_animation_timeline_ui_settings(self) -> None:
        actual_visible = bool(getattr(self, "animation_timeline_dialog", None) is not None and self.animation_timeline_dialog.isVisible())
        self._set_pref("animation/timeline_zoom", int(self._animation_timeline_zoom))
        self._set_pref("animation/timeline_in", int(self._animation_timeline_in_frame))
        self._set_pref("animation/timeline_out", int(self._animation_timeline_out_frame))
        self._set_pref("animation/timeline_visible", bool(self._animation_timeline_should_restore_visible))
        logger.debug(
            "Save timeline UI settings desired_visible=%s actual_visible=%s app_closing=%s",
            self._animation_timeline_should_restore_visible,
            actual_visible,
            self._is_app_closing,
        )
        if hasattr(self, "animation_timeline_dialog"):
            try:
                self._set_pref("window/animation_timeline_geometry", self.animation_timeline_dialog.saveGeometry())
            except Exception:  # noqa: BLE001
                logger.debug("Failed to save animation timeline geometry", exc_info=True)

    def _load_animation_timeline_ui_settings(self) -> None:
        self._animation_timeline_zoom = max(50, min(220, self._get_pref_int("animation/timeline_zoom", self._animation_timeline_zoom)))
        self._animation_timeline_in_frame = max(0, self._get_pref_int("animation/timeline_in", self._animation_timeline_in_frame))
        self._animation_timeline_out_frame = max(1, self._get_pref_int("animation/timeline_out", self._animation_timeline_out_frame))
        if hasattr(self, "animation_timeline_zoom_slider"):
            blocked = self.animation_timeline_zoom_slider.blockSignals(True)
            self.animation_timeline_zoom_slider.setValue(self._animation_timeline_zoom)
            self.animation_timeline_zoom_slider.blockSignals(blocked)
        if hasattr(self, "animation_timeline_zoom_value_label"):
            self.animation_timeline_zoom_value_label.setText(f"{self._animation_timeline_zoom}%")
        self._sync_animation_timeline_range_controls()
        geometry = self._settings.value("window/animation_timeline_geometry")
        if geometry is not None and hasattr(self, "animation_timeline_dialog"):
            try:
                self.animation_timeline_dialog.restoreGeometry(geometry)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to restore animation timeline geometry", exc_info=True)

        should_show_timeline = self._get_pref_bool("animation/timeline_visible", False)
        self._animation_timeline_should_restore_visible = bool(should_show_timeline)
        logger.debug(
            "Load timeline UI settings desired_visible=%s zoom=%s in=%s out=%s",
            self._animation_timeline_should_restore_visible,
            self._animation_timeline_zoom,
            self._animation_timeline_in_frame,
            self._animation_timeline_out_frame,
        )
        if should_show_timeline and hasattr(self, "animation_timeline_dialog"):
            QTimer.singleShot(0, self._restore_animation_timeline_visibility)

    def _restore_animation_timeline_visibility(self) -> None:
        if not hasattr(self, "animation_timeline_dialog"):
            return
        if not self._animation_timeline_should_restore_visible:
            return
        if not self.isVisible():
            QTimer.singleShot(40, self._restore_animation_timeline_visibility)
            return
        if self._is_loading_sprites:
            QTimer.singleShot(40, self._restore_animation_timeline_visibility)
            return
        if self.animation_timeline_dialog.isVisible():
            return
        logger.debug("Restoring animation timeline visibility on startup")
        self._open_animation_timeline_dialog()

    def _get_key_binding(self, action: str, default: str) -> str:
        sequences = [entry.get("shortcut", "") for entry in self._get_key_bindings(action, default)]
        default_text = KeybindingsDialog._normalize_shortcut_text(default)
        return str(sequences[0]).strip() if sequences and str(sequences[0]).strip() else default_text

    def _get_key_bindings(self, action: str, default: str) -> List[Dict[str, Any]]:
        default_text = KeybindingsDialog._normalize_shortcut_text(default)
        raw = self._settings.value(f"bindings/{action}", default_text)

        def _normalize(raw_list: Any) -> List[Dict[str, Any]]:
            entries: List[Dict[str, Any]] = []
            if isinstance(raw_list, dict):
                raw_list = [raw_list]
            if isinstance(raw_list, str):
                raw_list = [raw_list]
            if not isinstance(raw_list, list):
                return entries
            for item in raw_list:
                shortcut = ""
                is_global = False
                if isinstance(item, dict):
                    shortcut = KeybindingsDialog._normalize_shortcut_text(str(item.get("shortcut", "")))
                    is_global = bool(item.get("global", False))
                else:
                    shortcut = KeybindingsDialog._normalize_shortcut_text(str(item))
                if not shortcut:
                    continue
                if any(existing.get("shortcut", "").lower() == shortcut.lower() for existing in entries):
                    continue
                entries.append({"shortcut": shortcut, "global": is_global})
            return entries

        bindings: List[Dict[str, Any]] = []
        if isinstance(raw, dict):
            if isinstance(raw.get("bindings"), list):
                bindings = _normalize(raw.get("bindings", []))
            elif isinstance(raw.get("shortcuts"), (list, str)):
                shortcuts = raw.get("shortcuts", [])
                if isinstance(shortcuts, str):
                    shortcuts = [shortcuts]
                bindings = _normalize([{"shortcut": value, "global": bool(raw.get("global", False))} for value in shortcuts])
        else:
            text = str(raw).strip()
            parsed: Dict[str, Any] | None = None
            if text.startswith("{") and text.endswith("}"):
                try:
                    payload = json.loads(text)
                    if isinstance(payload, dict):
                        parsed = payload
                except Exception:  # noqa: BLE001
                    parsed = None
            if parsed is not None:
                if isinstance(parsed.get("bindings"), list):
                    bindings = _normalize(parsed.get("bindings", []))
                elif isinstance(parsed.get("shortcuts"), (list, str)):
                    shortcuts = parsed.get("shortcuts", [])
                    if isinstance(shortcuts, str):
                        shortcuts = [shortcuts]
                    bindings = _normalize([{"shortcut": value, "global": bool(parsed.get("global", False))} for value in shortcuts])
                elif isinstance(parsed.get("current"), str):
                    current = KeybindingsDialog._normalize_shortcut_text(str(parsed.get("current", "")))
                    secondary = KeybindingsDialog._normalize_shortcut_text(str(parsed.get("secondary", "")))
                    seqs = [current, secondary]
                    bindings = _normalize([{"shortcut": value, "global": bool(parsed.get("global", False))} for value in seqs if value])
            else:
                plain = KeybindingsDialog._normalize_shortcut_text(text)
                if plain:
                    bindings = [{"shortcut": plain, "global": False}]

        if not bindings and default_text:
            bindings = [{"shortcut": default_text, "global": False}]
        return bindings

    def _save_preview_ui_settings(self) -> None:
        self._set_pref("preview/view_mode", str(self._preview_view_mode))
        self._set_pref("preview/animation_follow_selection", bool(self._preview_animation_follow_selection))
        self._set_pref("preview/onion_source_mode", str(self._preview_onion_source_mode))
        self._set_pref("preview/onion_enabled", bool(self._preview_onion_enabled))
        self._set_pref("preview/onion_prev_count", int(self._preview_onion_prev_count))
        self._set_pref("preview/onion_next_count", int(self._preview_onion_next_count))
        self._set_pref("preview/onion_base_alpha", int(self._preview_onion_base_alpha))
        self._set_pref("preview/onion_prev_alpha", int(self._preview_onion_prev_alpha))
        self._set_pref("preview/onion_next_alpha", int(self._preview_onion_next_alpha))
        self._set_pref("preview/onion_prev_tint", f"{self._preview_onion_prev_tint[0]},{self._preview_onion_prev_tint[1]},{self._preview_onion_prev_tint[2]}")
        self._set_pref("preview/onion_next_tint", f"{self._preview_onion_next_tint[0]},{self._preview_onion_next_tint[1]},{self._preview_onion_next_tint[2]}")
        self._set_pref("preview/onion_tint_strength", int(self._preview_onion_tint_strength))
        self._set_pref("preview/onion_sprite_list_scope", str(self._preview_onion_sprite_list_scope))
        self._set_pref("preview/onion_sprite_list_wrap", bool(self._preview_onion_sprite_list_wrap))
        self._set_pref("preview/highlight_enabled", bool(self.highlight_checkbox.isChecked()))
        self._set_pref("preview/highlight_hover_enabled", bool(self.hover_highlight_checkbox.isChecked()))
        self._set_pref("preview/overlay_selected_color", f"{self._selected_overlay_color[0]},{self._selected_overlay_color[1]},{self._selected_overlay_color[2]}")
        self._set_pref("preview/overlay_selected_alpha_min", int(self._selected_overlay_alpha_min))
        self._set_pref("preview/overlay_selected_alpha_max", int(self._selected_overlay_alpha_max))
        self._set_pref("preview/overlay_selected_speed", float(self._selected_animation_speed))
        self._set_pref("preview/overlay_hover_color", f"{self._hover_overlay_color[0]},{self._hover_overlay_color[1]},{self._hover_overlay_color[2]}")
        self._set_pref("preview/overlay_hover_alpha_min", int(self._hover_overlay_alpha_min))
        self._set_pref("preview/overlay_hover_alpha_max", int(self._hover_overlay_alpha_max))
        self._set_pref("preview/overlay_hover_speed", float(self._hover_animation_speed))
        self._set_pref("preview/overlay_show_both", bool(self._overlay_show_both))

        # Legacy keys for backward compatibility with older builds/features.
        self._set_pref("preview/overlay_color", f"{self._selected_overlay_color[0]},{self._selected_overlay_color[1]},{self._selected_overlay_color[2]}")
        self._set_pref("preview/overlay_alpha_min", int(self._selected_overlay_alpha_min))
        self._set_pref("preview/overlay_alpha_max", int(self._selected_overlay_alpha_max))
        self._set_pref("preview/animation_speed", float(self._selected_animation_speed))
        self._set_pref("preview/bg_transparent_enabled", bool(self._preview_bg_transparent_enabled))
        indices_text = ",".join(str(idx) for idx in sorted(self._preview_background_indices))
        self._set_pref("preview/bg_indices", indices_text)

    def _load_ui_settings(self) -> None:
        self._main_palette_columns = max(1, min(32, self._get_pref_int("palette/main_columns", self._main_palette_columns)))
        self._main_palette_force_columns = self._get_pref_bool("palette/main_force_columns", self._main_palette_force_columns)
        self._main_palette_zoom = max(16, min(96, self._get_pref_int("palette/main_zoom", self._main_palette_zoom)))
        self._main_palette_gap = max(-8, min(20, self._get_pref_int("palette/main_gap", self._main_palette_gap)))
        self._main_palette_show_indices = self._get_pref_bool("palette/main_show_indices", self._main_palette_show_indices)
        self._main_palette_show_grid = self._get_pref_bool("palette/main_show_grid", self._main_palette_show_grid)
        self._main_palette_show_usage_badge = self._get_pref_bool("palette/main_show_usage_badge", self._main_palette_show_usage_badge)
        self._apply_main_palette_layout_options()

        legacy_color_raw = str(self._settings.value("preview/overlay_color", "255,255,255"))

        def _parse_rgb(raw: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
            parts = [part.strip() for part in str(raw).split(",")]
            if len(parts) != 3:
                return fallback
            try:
                parsed = tuple(max(0, min(255, int(part))) for part in parts)
                return (parsed[0], parsed[1], parsed[2])
            except ValueError:
                return fallback

        legacy_color = _parse_rgb(legacy_color_raw, self._selected_overlay_color)
        self._selected_overlay_color = _parse_rgb(
            str(self._settings.value("preview/overlay_selected_color", f"{legacy_color[0]},{legacy_color[1]},{legacy_color[2]}")),
            legacy_color,
        )
        self._hover_overlay_color = _parse_rgb(
            str(self._settings.value("preview/overlay_hover_color", f"{self._selected_overlay_color[0]},{self._selected_overlay_color[1]},{self._selected_overlay_color[2]}")),
            self._selected_overlay_color,
        )

        legacy_alpha_min = max(0, min(255, self._get_pref_int("preview/overlay_alpha_min", self._selected_overlay_alpha_min)))
        legacy_alpha_max = max(legacy_alpha_min, min(255, self._get_pref_int("preview/overlay_alpha_max", self._selected_overlay_alpha_max)))
        legacy_speed = max(0.01, min(0.50, self._get_pref_float("preview/animation_speed", self._selected_animation_speed)))

        self._selected_overlay_alpha_min = max(0, min(255, self._get_pref_int("preview/overlay_selected_alpha_min", legacy_alpha_min)))
        self._selected_overlay_alpha_max = max(
            self._selected_overlay_alpha_min,
            min(255, self._get_pref_int("preview/overlay_selected_alpha_max", legacy_alpha_max)),
        )
        self._selected_animation_speed = max(0.01, min(0.50, self._get_pref_float("preview/overlay_selected_speed", legacy_speed)))

        self._hover_overlay_alpha_min = max(0, min(255, self._get_pref_int("preview/overlay_hover_alpha_min", self._selected_overlay_alpha_min)))
        self._hover_overlay_alpha_max = max(
            self._hover_overlay_alpha_min,
            min(255, self._get_pref_int("preview/overlay_hover_alpha_max", self._selected_overlay_alpha_max)),
        )
        self._hover_animation_speed = max(0.01, min(0.50, self._get_pref_float("preview/overlay_hover_speed", self._selected_animation_speed)))

        self._overlay_show_both = self._get_pref_bool("preview/overlay_show_both", self._overlay_show_both)

        # Keep legacy aliases synced.
        self._overlay_color = self._selected_overlay_color
        self._overlay_alpha_min = self._selected_overlay_alpha_min
        self._overlay_alpha_max = self._selected_overlay_alpha_max
        self._animation_speed = self._selected_animation_speed
        self._preview_bg_transparent_enabled = self._get_pref_bool(
            "preview/bg_transparent_enabled",
            self._preview_bg_transparent_enabled,
        )
        bg_indices_raw = str(self._settings.value("preview/bg_indices", ""))
        parsed_indices: set[int] = set()
        for token in re.split(r"[\s,;]+", bg_indices_raw.strip()):
            if not token:
                continue
            try:
                idx = int(token)
            except ValueError:
                continue
            if 0 <= idx <= 255:
                parsed_indices.add(idx)
        self._preview_background_indices = parsed_indices

        view_mode_raw = str(self._settings.value("preview/view_mode", self._preview_view_mode) or self._preview_view_mode).strip().lower()
        self._preview_view_mode = "animation_assist" if view_mode_raw == "animation_assist" else "sprite_edit"
        self._preview_animation_follow_selection = self._get_pref_bool(
            "preview/animation_follow_selection",
            self._preview_animation_follow_selection,
        )
        onion_source_raw = str(
            self._settings.value("preview/onion_source_mode", self._preview_onion_source_mode) or self._preview_onion_source_mode
        ).strip().lower()
        self._preview_onion_source_mode = "sprite_list" if onion_source_raw == "sprite_list" else "timeline"
        self._preview_onion_enabled = self._get_pref_bool("preview/onion_enabled", self._preview_onion_enabled)
        self._preview_onion_prev_count = max(0, min(8, self._get_pref_int("preview/onion_prev_count", self._preview_onion_prev_count)))
        self._preview_onion_next_count = max(0, min(8, self._get_pref_int("preview/onion_next_count", self._preview_onion_next_count)))
        self._preview_onion_base_alpha = max(0, min(220, self._get_pref_int("preview/onion_base_alpha", self._preview_onion_base_alpha)))
        self._preview_onion_prev_alpha = max(
            0,
            min(220, self._get_pref_int("preview/onion_prev_alpha", self._preview_onion_base_alpha)),
        )
        self._preview_onion_next_alpha = max(
            0,
            min(220, self._get_pref_int("preview/onion_next_alpha", self._preview_onion_base_alpha)),
        )
        self._preview_onion_prev_tint = _parse_rgb(
            str(
                self._settings.value(
                    "preview/onion_prev_tint",
                    f"{self._preview_onion_prev_tint[0]},{self._preview_onion_prev_tint[1]},{self._preview_onion_prev_tint[2]}",
                )
            ),
            self._preview_onion_prev_tint,
        )
        self._preview_onion_next_tint = _parse_rgb(
            str(
                self._settings.value(
                    "preview/onion_next_tint",
                    f"{self._preview_onion_next_tint[0]},{self._preview_onion_next_tint[1]},{self._preview_onion_next_tint[2]}",
                )
            ),
            self._preview_onion_next_tint,
        )
        self._preview_onion_tint_strength = max(
            0,
            min(255, self._get_pref_int("preview/onion_tint_strength", self._preview_onion_tint_strength)),
        )
        self._preview_onion_base_alpha = max(self._preview_onion_prev_alpha, self._preview_onion_next_alpha)
        onion_scope_raw = str(
            self._settings.value("preview/onion_sprite_list_scope", self._preview_onion_sprite_list_scope)
            or self._preview_onion_sprite_list_scope
        ).strip().lower()
        self._preview_onion_sprite_list_scope = "selected" if onion_scope_raw == "selected" else "all"
        self._preview_onion_sprite_list_wrap = self._get_pref_bool(
            "preview/onion_sprite_list_wrap",
            self._preview_onion_sprite_list_wrap,
        )

        highlight_enabled = self._get_pref_bool("preview/highlight_enabled", self.highlight_checkbox.isChecked())
        hover_highlight_enabled = self._get_pref_bool("preview/highlight_hover_enabled", self.hover_highlight_checkbox.isChecked())
        self.highlight_checkbox.setChecked(highlight_enabled)
        self.hover_highlight_checkbox.setChecked(hover_highlight_enabled)
        self._sync_preview_context_controls()
        logger.debug(
            "UI settings loaded main_cols=%s main_force=%s main_zoom=%s main_gap=%s highlight_sel=%s highlight_hover=%s selected_overlay=%s/%s-%s@%.2f hover_overlay=%s/%s-%s@%.2f show_both=%s bg_transparent=%s bg_indices=%s view_mode=%s follow_sel=%s onion_source=%s onion_enabled=%s onion_prev=%s onion_next=%s onion_alpha_prev=%s onion_alpha_next=%s onion_tint_prev=%s onion_tint_next=%s onion_tint_strength=%s onion_scope=%s onion_wrap=%s",
            self._main_palette_columns,
            self._main_palette_force_columns,
            self._main_palette_zoom,
            self._main_palette_gap,
            highlight_enabled,
            hover_highlight_enabled,
            self._selected_overlay_color,
            self._selected_overlay_alpha_min,
            self._selected_overlay_alpha_max,
            self._selected_animation_speed,
            self._hover_overlay_color,
            self._hover_overlay_alpha_min,
            self._hover_overlay_alpha_max,
            self._hover_animation_speed,
            self._overlay_show_both,
            self._preview_bg_transparent_enabled,
            sorted(self._preview_background_indices)[:16],
            self._preview_view_mode,
            self._preview_animation_follow_selection,
            self._preview_onion_source_mode,
            self._preview_onion_enabled,
            self._preview_onion_prev_count,
            self._preview_onion_next_count,
            self._preview_onion_prev_alpha,
            self._preview_onion_next_alpha,
            self._preview_onion_prev_tint,
            self._preview_onion_next_tint,
            self._preview_onion_tint_strength,
            self._preview_onion_sprite_list_scope,
            self._preview_onion_sprite_list_wrap,
        )
        if hasattr(self, "background_indices_btn"):
            self._update_background_indices_button_text()

    def _is_animation_assist_view_active(self) -> bool:
        return self._preview_view_mode == "animation_assist"

    def _sync_preview_context_controls(self) -> None:
        if hasattr(self, "preview_context_combo"):
            idx = self.preview_context_combo.findData(self._preview_view_mode)
            if idx >= 0 and idx != self.preview_context_combo.currentIndex():
                blocked = self.preview_context_combo.blockSignals(True)
                self.preview_context_combo.setCurrentIndex(idx)
                self.preview_context_combo.blockSignals(blocked)
        if hasattr(self, "preview_animation_follow_selection_check"):
            blocked_follow = self.preview_animation_follow_selection_check.blockSignals(True)
            self.preview_animation_follow_selection_check.setChecked(bool(self._preview_animation_follow_selection))
            self.preview_animation_follow_selection_check.blockSignals(blocked_follow)
            self.preview_animation_follow_selection_check.setEnabled(self._is_animation_assist_view_active())
        if hasattr(self, "preview_onion_source_combo"):
            source_idx = self.preview_onion_source_combo.findData(self._preview_onion_source_mode)
            if source_idx >= 0 and source_idx != self.preview_onion_source_combo.currentIndex():
                blocked_source = self.preview_onion_source_combo.blockSignals(True)
                self.preview_onion_source_combo.setCurrentIndex(source_idx)
                self.preview_onion_source_combo.blockSignals(blocked_source)
            self.preview_onion_source_combo.setEnabled(self._is_animation_assist_view_active())
        if hasattr(self, "preview_onion_enabled_check"):
            blocked_onion = self.preview_onion_enabled_check.blockSignals(True)
            self.preview_onion_enabled_check.setChecked(bool(self._preview_onion_enabled))
            self.preview_onion_enabled_check.blockSignals(blocked_onion)
            self.preview_onion_enabled_check.setEnabled(self._is_animation_assist_view_active())
        if hasattr(self, "preview_onion_settings_btn"):
            self.preview_onion_settings_btn.setEnabled(self._is_animation_assist_view_active())
        self.highlight_checkbox.setEnabled(True)
        self.hover_highlight_checkbox.setEnabled(True)
        self.overlay_settings_btn.setEnabled(True)

    def _refresh_animation_assist_preview_frame(self) -> None:
        if not self._is_animation_assist_view_active():
            return
        if self._preview_onion_source_mode == "sprite_list":
            record = self._current_record()
            if record is None:
                self._animation_preview_visible_sprite_key = None
                self.preview_panel.set_static_pixmap(None)
                self.preview_panel.set_overlay_layers([], QSize(0, 0))
                self.preview_panel.set_pixmap(None, reset_zoom=False)
                return
            self._set_animation_preview_frame_by_key(record.path.as_posix(), source="assist-refresh-spritelist")
            return
        if self._animation_playhead_frame is not None:
            self._seek_animation_playhead(float(self._animation_playhead_frame), from_user_scrub=False)
            return
        tag = self._selected_animation_tag()
        frame_index = self._selected_animation_frame_index()
        if tag is None or frame_index is None or frame_index < 0 or frame_index >= len(tag.frames):
            self._animation_preview_visible_sprite_key = None
            self.preview_panel.set_static_pixmap(None)
            self.preview_panel.set_overlay_layers([], QSize(0, 0))
            self.preview_panel.set_pixmap(None, reset_zoom=False)
            return
        position = self._animation_frame_start_offset(tag, int(frame_index))
        self._seek_animation_playhead(float(position), from_user_scrub=False)

    def _sprite_list_onion_keys(self, center_key: str) -> tuple[List[str], List[str]]:
        if not self._preview_onion_enabled:
            return [], []
        ordered_keys: List[str] = []
        if self._preview_onion_sprite_list_scope == "selected":
            selected_rows = sorted({index.row() for index in self.images_panel.list_widget.selectedIndexes()})
            for row in selected_rows:
                item = self.images_panel.list_widget.item(row)
                if item is None:
                    continue
                key = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(key, str) and key:
                    ordered_keys.append(key)
        if not ordered_keys:
            ordered_keys = self._ordered_sprite_keys()
        if not ordered_keys:
            return [], []
        try:
            center_index = ordered_keys.index(center_key)
        except ValueError:
            return [], []

        prev_keys: List[str] = []
        next_keys: List[str] = []
        count = len(ordered_keys)

        for step in range(1, max(0, int(self._preview_onion_prev_count)) + 1):
            index = center_index - step
            if index < 0:
                if not self._preview_onion_sprite_list_wrap or count <= 0:
                    break
                index = index % count
            prev_keys.append(ordered_keys[index])

        for step in range(1, max(0, int(self._preview_onion_next_count)) + 1):
            index = center_index + step
            if index >= count:
                if not self._preview_onion_sprite_list_wrap or count <= 0:
                    break
                index = index % count
            next_keys.append(ordered_keys[index])

        return prev_keys, next_keys

    def _timeline_onion_keys(self, tag: AnimationTag, timeline_pos: int) -> tuple[List[str], List[str]]:
        if not self._preview_onion_enabled:
            return [], []
        self._sync_animation_timeline_range_controls(tag)
        total_frames = self._timeline_total_frames_for_tag(tag)
        if total_frames <= 0:
            return [], []

        in_pos = max(0, min(total_frames - 1, int(self._animation_timeline_in_frame)))
        out_pos = max(in_pos + 1, min(total_frames, int(self._animation_timeline_out_frame)))
        span = max(1, out_pos - in_pos)

        def _key_at_position(position: int) -> str | None:
            frame_index, _frame_step, _frame_start = self._animation_timeline_slot_at_position(tag, int(position))
            if frame_index is None or frame_index < 0 or frame_index >= len(tag.frames):
                return None
            return self._resolve_animation_runtime_key(tag.frames[frame_index].sprite_key)

        prev_keys: List[str] = []
        next_keys: List[str] = []

        for step in range(1, max(0, int(self._preview_onion_prev_count)) + 1):
            candidate = timeline_pos - step
            if candidate < in_pos:
                if not self.animation_loop_check.isChecked():
                    break
                candidate = in_pos + ((candidate - in_pos) % span)
            key = _key_at_position(candidate)
            if key is not None:
                prev_keys.append(key)

        for step in range(1, max(0, int(self._preview_onion_next_count)) + 1):
            candidate = timeline_pos + step
            if candidate >= out_pos:
                if not self.animation_loop_check.isChecked():
                    break
                candidate = in_pos + ((candidate - in_pos) % span)
            key = _key_at_position(candidate)
            if key is not None:
                next_keys.append(key)

        return prev_keys, next_keys

    def _compose_animation_assist_pixmap(self, current_key: str, prev_keys: Sequence[str], next_keys: Sequence[str]) -> tuple[QPixmap, QPixmap | None] | None:
        current_record = self.sprite_records.get(current_key)
        if current_record is None:
            return None
        current_pixmap = self._get_animation_preview_pixmap(current_record)
        if current_pixmap is None:
            return None

        if not self._preview_onion_enabled or (not prev_keys and not next_keys):
            return current_pixmap, None

        prev_pixmaps: List[QPixmap] = []
        for key in prev_keys:
            record = self.sprite_records.get(key)
            if record is None:
                continue
            pixmap = self._get_animation_preview_pixmap(record)
            if pixmap is not None:
                prev_pixmaps.append(pixmap)

        next_pixmaps: List[QPixmap] = []
        for key in next_keys:
            record = self.sprite_records.get(key)
            if record is None:
                continue
            pixmap = self._get_animation_preview_pixmap(record)
            if pixmap is not None:
                next_pixmaps.append(pixmap)

        if not prev_pixmaps and not next_pixmaps:
            return current_pixmap, None

        compose_cache_key: tuple[Any, ...] = (
            int(current_pixmap.cacheKey()),
            tuple(int(pixmap.cacheKey()) for pixmap in prev_pixmaps),
            tuple(int(pixmap.cacheKey()) for pixmap in next_pixmaps),
            int(self._preview_onion_prev_alpha),
            int(self._preview_onion_next_alpha),
            tuple(int(channel) for channel in self._preview_onion_prev_tint),
            tuple(int(channel) for channel in self._preview_onion_next_tint),
            int(self._preview_onion_tint_strength),
        )
        cached_layers = self._animation_assist_layer_cache.get(compose_cache_key)
        if cached_layers is not None:
            self._animation_assist_layer_cache.move_to_end(compose_cache_key)
            return cached_layers

        width = current_pixmap.width()
        height = current_pixmap.height()
        for pixmap in prev_pixmaps:
            width = max(width, pixmap.width())
            height = max(height, pixmap.height())
        for pixmap in next_pixmaps:
            width = max(width, pixmap.width())
            height = max(height, pixmap.height())

        target_width = max(1, int(width))
        target_height = max(1, int(height))

        def _draw_centered(
            painter: QPainter,
            pixmap: QPixmap,
            opacity: float,
            tint: tuple[int, int, int] | None = None,
        ) -> None:
            x = int((target_width - pixmap.width()) * 0.5)
            y = int((target_height - pixmap.height()) * 0.5)
            effective_opacity = max(0.0, min(1.0, opacity))
            painter.save()
            painter.setOpacity(effective_opacity)
            painter.drawPixmap(x, y, pixmap)

            tint_strength = max(0.0, min(1.0, float(self._preview_onion_tint_strength) / 255.0))
            if tint is not None and tint_strength > 0.0 and effective_opacity > 0.0:
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceAtop)
                painter.setOpacity(max(0.0, min(1.0, tint_strength * effective_opacity)))
                painter.fillRect(x, y, pixmap.width(), pixmap.height(), QColor(tint[0], tint[1], tint[2]))
            painter.restore()

        def _layer_opacity(distance: int, total: int, base_alpha: int) -> float:
            if total <= 0:
                return 0.0
            base = max(0.0, min(1.0, float(base_alpha) / 255.0))
            weight = float(total - distance + 1) / float(total)
            return max(0.0, min(1.0, base * weight))

        onion_only = QPixmap(target_width, target_height)
        onion_only.fill(Qt.GlobalColor.transparent)
        onion_painter = QPainter(onion_only)
        try:
            total_prev = len(prev_pixmaps)
            for distance, pixmap in reversed(list(enumerate(prev_pixmaps, start=1))):
                _draw_centered(
                    onion_painter,
                    pixmap,
                    _layer_opacity(distance, total_prev, int(self._preview_onion_prev_alpha)),
                    tint=self._preview_onion_prev_tint,
                )

            total_next = len(next_pixmaps)
            for distance, pixmap in reversed(list(enumerate(next_pixmaps, start=1))):
                _draw_centered(
                    onion_painter,
                    pixmap,
                    _layer_opacity(distance, total_next, int(self._preview_onion_next_alpha)),
                    tint=self._preview_onion_next_tint,
                )
        finally:
            onion_painter.end()

        current_only = QPixmap(target_width, target_height)
        current_only.fill(Qt.GlobalColor.transparent)
        current_painter = QPainter(current_only)
        try:
            _draw_centered(current_painter, current_pixmap, 1.0)
        finally:
            current_painter.end()

        layers = (current_only, onion_only)
        self._animation_assist_layer_cache[compose_cache_key] = layers
        if len(self._animation_assist_layer_cache) > 1024:
            self._animation_assist_layer_cache.popitem(last=False)
        return layers

    def _on_preview_context_changed(self, _index: int) -> None:
        data = self.preview_context_combo.currentData() if hasattr(self, "preview_context_combo") else self._preview_view_mode
        mode = "animation_assist" if data == "animation_assist" else "sprite_edit"
        if mode == self._preview_view_mode:
            return
        self._preview_view_mode = mode
        if mode == "sprite_edit":
            if self.animation_play_btn.isChecked():
                self._stop_animation_preview()
            self._schedule_preview_update()
        else:
            self._refresh_animation_assist_preview_frame()
        self._sync_preview_context_controls()
        self._save_preview_ui_settings()

    def _on_preview_animation_follow_selection_changed(self, checked: bool) -> None:
        self._preview_animation_follow_selection = bool(checked)
        self._save_preview_ui_settings()
        if self._preview_animation_follow_selection and self._is_animation_assist_view_active() and not self.animation_play_btn.isChecked():
            self._refresh_animation_assist_preview_frame()

    def _on_preview_onion_source_changed(self, _index: int) -> None:
        data = self.preview_onion_source_combo.currentData() if hasattr(self, "preview_onion_source_combo") else self._preview_onion_source_mode
        self._preview_onion_source_mode = "sprite_list" if data == "sprite_list" else "timeline"
        self._save_preview_ui_settings()
        if self._is_animation_assist_view_active() and not self.animation_play_btn.isChecked():
            self._refresh_animation_assist_preview_frame()

    def _on_preview_onion_enabled_changed(self, checked: bool) -> None:
        self._preview_onion_enabled = bool(checked)
        self._save_preview_ui_settings()
        if self._is_animation_assist_view_active() and not self.animation_play_btn.isChecked():
            self._refresh_animation_assist_preview_frame()

    def _open_onion_settings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Onion Settings")
        dialog.setModal(True)
        layout = QFormLayout(dialog)

        prev_spin = QSpinBox(dialog)
        prev_spin.setRange(0, 8)
        prev_spin.setValue(int(self._preview_onion_prev_count))
        next_spin = QSpinBox(dialog)
        next_spin.setRange(0, 8)
        next_spin.setValue(int(self._preview_onion_next_count))
        prev_alpha_spin = QSpinBox(dialog)
        prev_alpha_spin.setRange(0, 220)
        prev_alpha_spin.setValue(int(self._preview_onion_prev_alpha))
        prev_alpha_spin.setSuffix(" /255")
        next_alpha_spin = QSpinBox(dialog)
        next_alpha_spin.setRange(0, 220)
        next_alpha_spin.setValue(int(self._preview_onion_next_alpha))
        next_alpha_spin.setSuffix(" /255")
        tint_strength_spin = QSpinBox(dialog)
        tint_strength_spin.setRange(0, 255)
        tint_strength_spin.setValue(int(self._preview_onion_tint_strength))
        tint_strength_spin.setSuffix(" /255")

        prev_tint_button = QPushButton(dialog)
        prev_tint_button.setMinimumWidth(110)
        next_tint_button = QPushButton(dialog)
        next_tint_button.setMinimumWidth(110)

        prev_tint = [
            int(self._preview_onion_prev_tint[0]),
            int(self._preview_onion_prev_tint[1]),
            int(self._preview_onion_prev_tint[2]),
        ]
        next_tint = [
            int(self._preview_onion_next_tint[0]),
            int(self._preview_onion_next_tint[1]),
            int(self._preview_onion_next_tint[2]),
        ]

        def _refresh_tint_button(button: QPushButton, rgb: list[int], label: str) -> None:
            button.setText(label)
            button.setStyleSheet(
                f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid #888;"
            )

        def _pick_tint(rgb: list[int], button: QPushButton, label: str) -> None:
            chosen = QColorDialog.getColor(QColor(rgb[0], rgb[1], rgb[2]), dialog, f"Choose {label} Tint")
            if not chosen.isValid():
                return
            rgb[0] = int(chosen.red())
            rgb[1] = int(chosen.green())
            rgb[2] = int(chosen.blue())
            _refresh_tint_button(button, rgb, label)

        _refresh_tint_button(prev_tint_button, prev_tint, "Previous")
        _refresh_tint_button(next_tint_button, next_tint, "Next")
        prev_tint_button.clicked.connect(lambda: _pick_tint(prev_tint, prev_tint_button, "Previous"))
        next_tint_button.clicked.connect(lambda: _pick_tint(next_tint, next_tint_button, "Next"))

        sprite_scope_combo = QComboBox(dialog)
        sprite_scope_combo.addItem("All Loaded Sprites", "all")
        sprite_scope_combo.addItem("Selected Sprites Only", "selected")
        scope_idx = sprite_scope_combo.findData(self._preview_onion_sprite_list_scope)
        if scope_idx >= 0:
            sprite_scope_combo.setCurrentIndex(scope_idx)

        sprite_wrap_check = QCheckBox("Wrap sprite-list neighbors", dialog)
        sprite_wrap_check.setChecked(bool(self._preview_onion_sprite_list_wrap))

        layout.addRow("Previous layers", prev_spin)
        layout.addRow("Next layers", next_spin)
        layout.addRow("Previous alpha", prev_alpha_spin)
        layout.addRow("Next alpha", next_alpha_spin)
        layout.addRow("Previous tint", prev_tint_button)
        layout.addRow("Next tint", next_tint_button)
        layout.addRow("Tint strength", tint_strength_spin)
        layout.addRow("Sprite List scope", sprite_scope_combo)
        layout.addRow("Sprite List wrap", sprite_wrap_check)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=dialog)
        layout.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._preview_onion_prev_count = int(prev_spin.value())
        self._preview_onion_next_count = int(next_spin.value())
        self._preview_onion_prev_alpha = int(prev_alpha_spin.value())
        self._preview_onion_next_alpha = int(next_alpha_spin.value())
        self._preview_onion_base_alpha = max(self._preview_onion_prev_alpha, self._preview_onion_next_alpha)
        self._preview_onion_prev_tint = (int(prev_tint[0]), int(prev_tint[1]), int(prev_tint[2]))
        self._preview_onion_next_tint = (int(next_tint[0]), int(next_tint[1]), int(next_tint[2]))
        self._preview_onion_tint_strength = int(tint_strength_spin.value())
        scope_data = sprite_scope_combo.currentData()
        self._preview_onion_sprite_list_scope = "selected" if scope_data == "selected" else "all"
        self._preview_onion_sprite_list_wrap = bool(sprite_wrap_check.isChecked())
        self._save_preview_ui_settings()
        if self._is_animation_assist_view_active() and not self.animation_play_btn.isChecked():
            self._refresh_animation_assist_preview_frame()

    def _clear_all_sprites(self) -> None:
        """Clear all loaded sprites and reset palette."""
        if not self.sprite_records:
            return
        
        reply = QMessageBox.question(
            self,
            "Clear All Sprites",
            f"Remove all {len(self.sprite_records)} loaded sprite(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._reset_loaded_workspace()
        logger.info("Cleared all sprites")

    def _reset_loaded_workspace(self) -> None:
        self._stop_animation_preview()
        self._cancel_inplace_icon_refresh()
        self._clear_sprite_icon_cache("workspace-reset")
        self._animation_preview_pixmap_cache.clear()
        self._animation_preview_index_cache.clear()
        self._animation_assist_layer_cache.clear()
        self._overlay_region_multi_cache.clear()
        self.sprite_records.clear()
        self._animation_tags = {}
        self._load_mode_overrides.clear()
        self.images_panel.list_widget.clear()
        self.images_panel.group_list.clear()
        self.images_panel.set_loaded_count(0)
        self.palette_colors = []
        self.palette_alphas = []
        self._palette_slot_ids = []
        self._detect_palette_colors = []
        self._detect_palette_alphas = []
        self._detect_palette_slot_ids = []
        self._palette_groups = {}
        self._group_key_to_id = {}
        self._next_group_id = 1
        self._next_slot_id = 0
        self._slot_color_lookup = {}
        self._invalidate_used_index_cache("clear-all")
        self._sync_palette_model()
        self._refresh_group_overview()
        self.preview_panel.set_pixmap(None, reset_zoom=True)
        self.statusBar().clearMessage()
        self._update_export_buttons()
        self._refresh_animation_tag_list()
        self._reset_history()

    def _clear_sprite_icon_cache(self, reason: str) -> None:
        if not self._sprite_icon_cache:
            return
        cleared = len(self._sprite_icon_cache)
        self._sprite_icon_cache.clear()
        logger.debug("Sprite icon cache cleared reason=%s entries=%s", reason, cleared)

    def _prompt_and_load(self) -> None:
        file_dialog = QFileDialog(self, "Select sprite images")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.bmp *.gif *.jpg *.jpeg)")
        if not file_dialog.exec():
            return
        paths = [Path(p) for p in file_dialog.selectedFiles()]
        self._load_images(paths)

    def _new_project(self) -> None:
        default_file = Path.cwd() / f"NewProject{_PROJECT_FOLDER_SUFFIX}"
        target_text, _ = QFileDialog.getSaveFileName(
            self,
            "New Project",
            str(default_file),
            f"SpriteTools Project Folder (*{_PROJECT_FOLDER_SUFFIX});;Legacy Project Folder (*{_PROJECT_LEGACY_FOLDER_SUFFIX})",
        )
        if not target_text:
            return
        project_root = self._normalize_project_root(Path(target_text), selected_is_parent=False)
        if project_root.exists() and any(project_root.iterdir()):
            QMessageBox.warning(
                self,
                "Create Project",
                (
                    "New Project requires an empty target folder for a clean workspace.\n\n"
                    f"Selected folder is not empty:\n{project_root}"
                ),
            )
            return
        if self.sprite_records:
            reply = QMessageBox.question(
                self,
                "Create Project",
                "Creating a new project will clear currently loaded sprites and start clean. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._reset_loaded_workspace()
        try:
            paths, manifest = self._project_service.create_project(project_root, project_name=project_root.stem, mode="managed")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Create Project", f"Failed to create project:\n{exc}")
            return
        self._activate_project(paths, manifest)
        self._remember_recent_project(paths.manifest)
        self._update_project_action_buttons()
        self.statusBar().showMessage(f"Created project: {manifest.project_name}", 5000)

    def _open_project(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open SpriteTools Project",
            str(Path.cwd()),
            f"SpriteTools Project ({PROJECT_MANIFEST_NAME})",
        )
        if not file_path:
            return
        self._open_project_from_path(Path(file_path))

    def _open_recent_project(self) -> None:
        recent_paths = self._recent_project_manifest_paths()
        if not recent_paths:
            QMessageBox.information(self, "Open Recent", "No recent projects found.")
            return

        display_items: List[str] = []
        display_to_path: Dict[str, Path] = {}
        for manifest_path in recent_paths:
            root = manifest_path.parent
            label = f"{root.name} — {manifest_path.as_posix()}"
            display_items.append(label)
            display_to_path[label] = manifest_path

        selected, accepted = QInputDialog.getItem(
            self,
            "Open Recent Project",
            "Recent projects:",
            display_items,
            0,
            False,
        )
        if not accepted or not selected:
            return
        chosen_path = display_to_path.get(selected)
        if chosen_path is None:
            return
        self._open_project_from_path(chosen_path)

    def _open_project_from_path(self, manifest_path: Path) -> None:
        if self.sprite_records:
            reply = QMessageBox.question(
                self,
                "Open Project",
                "Opening a project will clear currently loaded sprites. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        try:
            paths, manifest = self._project_service.open_project(manifest_path)
            sprite_meta = self._project_service.load_sprite_metadata(paths)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Open Project", f"Failed to open project:\n{exc}")
            return

        self._recovery_source_text = None
        manifest, sprite_meta = self._maybe_apply_autosave_recovery(paths, manifest, sprite_meta)

        self._reset_loaded_workspace()
        self._activate_project(paths, manifest)
        self._pending_project_sprite_metadata = sprite_meta
        self._autosave_dirty = False

        resolved_entries, missing_count, relinked_count = self._resolve_project_sources_on_open(paths, manifest)
        existing_paths: List[Path] = []
        for entry, candidate in resolved_entries:
            existing_paths.append(candidate)
            self._load_mode_overrides[candidate.as_posix()] = entry.load_mode
        if existing_paths:
            self._load_images(existing_paths)
        if missing_count:
            suffix = f" (relinked {relinked_count})" if relinked_count else ""
            self.statusBar().showMessage(f"Opened project with {missing_count} missing sprite source(s){suffix}", 5000)
        elif relinked_count:
            self.statusBar().showMessage(f"Opened project and relinked {relinked_count} sprite source(s)", 5000)
        elif not existing_paths:
            self.statusBar().showMessage("Opened project (no sprites indexed yet)", 4000)
        self._autosave_recovery_mode = False
        self._remember_recent_project(paths.manifest)
        self._update_project_action_buttons()

    def _resolve_project_sources_on_open(
        self,
        paths: ProjectPaths,
        manifest: ProjectManifest,
    ) -> tuple[List[tuple[ProjectSpriteEntry, Path]], int, int]:
        resolved_entries: List[tuple[ProjectSpriteEntry, Path]] = []
        missing_entries: List[ProjectSpriteEntry] = []
        for entry in manifest.sprites:
            candidate = self._project_service.source_path_for_entry(paths, entry, manifest.project_mode)
            if candidate.exists() and candidate.is_file():
                resolved_entries.append((entry, candidate))
            else:
                missing_entries.append(entry)
        if not missing_entries:
            return resolved_entries, 0, 0

        relinked_entries, unresolved_count = self._prompt_relink_missing_sources(paths, manifest, missing_entries)
        if relinked_entries:
            resolved_entries.extend(relinked_entries)
            try:
                self._project_service.save_manifest(paths, manifest)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist relinked project entries: %s", exc)
        return resolved_entries, unresolved_count, len(relinked_entries)

    def _prompt_relink_missing_sources(
        self,
        paths: ProjectPaths,
        manifest: ProjectManifest,
        missing_entries: Sequence[ProjectSpriteEntry],
    ) -> tuple[List[tuple[ProjectSpriteEntry, Path]], int]:
        reply = QMessageBox.question(
            self,
            "Missing Sources",
            (
                f"{len(missing_entries)} source sprite(s) are missing.\n"
                "Attempt to relink by scanning a folder for matching filenames?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return [], len(missing_entries)

        folder = QFileDialog.getExistingDirectory(self, "Select folder to scan for missing sources", str(Path.cwd()))
        if not folder:
            return [], len(missing_entries)

        relink_map = self._build_relink_filename_index(Path(folder))
        relinked_entries: List[tuple[ProjectSpriteEntry, Path]] = []
        unresolved = 0
        for entry in missing_entries:
            source_name = Path(entry.source_path).name.lower()
            candidates = relink_map.get(source_name, [])
            if not candidates:
                unresolved += 1
                continue
            selected = sorted(candidates, key=lambda value: (len(value.parts), str(value).lower()))[0]
            if manifest.project_mode == "managed":
                imported = self._project_service.import_managed_sprite(paths, selected)
                resolved_path = imported.resolve()
                entry.source_path = resolved_path.relative_to(paths.root).as_posix()
            else:
                resolved_path = selected.resolve()
                entry.source_path = resolved_path.as_posix()
            try:
                entry.source_hash = self._project_service.hash_file(resolved_path)
            except Exception:  # noqa: BLE001
                entry.source_hash = ""
            relinked_entries.append((entry, resolved_path))
        return relinked_entries, unresolved

    def _build_relink_filename_index(self, root: Path) -> Dict[str, List[Path]]:
        index: Dict[str, List[Path]] = {}
        for dir_path, _dir_names, file_names in os.walk(root):
            for file_name in file_names:
                candidate = Path(dir_path) / file_name
                if not is_supported_image(candidate):
                    continue
                key = file_name.lower()
                bucket = index.setdefault(key, [])
                if len(bucket) < 8:
                    bucket.append(candidate)
        logger.debug("Built relink index root=%s names=%s", root, len(index))
        return index

    def _save_project(self) -> None:
        if self._project_paths is None or self._project_manifest is None:
            QMessageBox.information(
                self,
                "Save Project",
                "No active project. Create or open a project first.",
            )
            return
        if self._write_project_state(self._project_paths, self._project_manifest, context="Save Project"):
            self.statusBar().showMessage(f"Saved project: {self._project_manifest.project_name}", 4000)

    def _save_project_as(self) -> None:
        default_file = (
            self._project_paths.root.with_suffix(_PROJECT_FOLDER_SUFFIX)
            if self._project_paths is not None
            else (Path.cwd() / f"NewProject{_PROJECT_FOLDER_SUFFIX}")
        )
        target_text, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            str(default_file),
            f"SpriteTools Project Folder (*{_PROJECT_FOLDER_SUFFIX});;Legacy Project Folder (*{_PROJECT_LEGACY_FOLDER_SUFFIX})",
        )
        if not target_text:
            return
        target_root = self._normalize_project_root(Path(target_text), selected_is_parent=False)
        if self._project_paths is not None and target_root.resolve() == self._project_paths.root.resolve():
            self._save_project()
            return
        if target_root.exists() and any(target_root.iterdir()):
            reply = QMessageBox.question(
                self,
                "Save Project As",
                f"Target folder already exists and is not empty:\n{target_root}\n\nContinue and overwrite project files?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        active_manifest = self._project_manifest
        mode: Literal["managed", "linked"] = (
            active_manifest.project_mode if active_manifest is not None else "managed"
        )
        name = (active_manifest.project_name if active_manifest is not None else target_root.stem)
        try:
            target_paths, target_manifest = self._project_service.create_project(target_root, project_name=name, mode=mode)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save Project As", f"Failed to initialize target project:\n{exc}")
            return

        if not self._write_project_state(target_paths, target_manifest, context="Save Project As"):
            return

        self._activate_project(target_paths, target_manifest)
        self._remember_recent_project(target_paths.manifest)
        self._update_project_action_buttons()
        self.statusBar().showMessage(f"Saved project copy: {target_manifest.project_name}", 5000)

    def _write_project_state(self, paths: ProjectPaths, manifest: ProjectManifest, context: str) -> bool:
        snapshot = self._build_project_state_snapshot_payload(paths, manifest)
        manifest_payload = snapshot.get("manifest", {})
        sprite_metadata = snapshot.get("sprite_metadata", {})
        try:
            rebuilt_manifest = ProjectManifest.from_dict(manifest_payload if isinstance(manifest_payload, dict) else {})
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, context, f"Failed to build project state:\n{exc}")
            return False

        manifest.sprites = rebuilt_manifest.sprites
        manifest.settings = dict(rebuilt_manifest.settings)

        try:
            self._project_service.save_manifest(paths, manifest)
            if isinstance(sprite_metadata, dict):
                self._project_service.save_sprite_metadata(paths, sprite_metadata)
            else:
                self._project_service.save_sprite_metadata(paths, {})
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, context, f"Failed to save project:\n{exc}")
            return False
        self._autosave_dirty = False
        self._last_project_saved_text = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        self._last_project_saved_kind = "Saved"
        self._refresh_project_info_banner()
        return True

    def _normalize_project_root(self, selected_dir: Path, *, selected_is_parent: bool = True) -> Path:
        candidate = selected_dir
        lower_name = candidate.name.lower()
        has_known_suffix = lower_name.endswith(_PROJECT_FOLDER_SUFFIX) or lower_name.endswith(_PROJECT_LEGACY_FOLDER_SUFFIX)
        if selected_is_parent:
            if has_known_suffix:
                return candidate
            return candidate / f"{candidate.name}{_PROJECT_FOLDER_SUFFIX}"
        if has_known_suffix:
            return candidate
        return candidate.with_name(f"{candidate.name}{_PROJECT_FOLDER_SUFFIX}")

    def _activate_project(self, paths: ProjectPaths, manifest: ProjectManifest) -> None:
        self._project_paths = paths
        self._project_manifest = manifest
        self._load_animation_tags_from_manifest_settings()
        self._autosave_dirty = False
        self._autosave_last_status_ts = 0.0
        updated_dt = self._parse_iso_utc(manifest.updated_at)
        if updated_dt is not None:
            self._last_project_saved_text = updated_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
            self._last_project_saved_kind = "Saved"
        else:
            self._last_project_saved_text = None
            self._last_project_saved_kind = None
        output_value = str(manifest.settings.get("output_dir", "exports/renders")).strip()
        output_candidate = (paths.root / output_value).resolve()
        if not output_candidate.exists():
            output_candidate.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_candidate
        self.output_dir_label.setText(str(self.output_dir))
        load_mode_value = str(manifest.settings.get("load_mode", "detect")).strip().lower()
        mode = "preserve" if load_mode_value == "preserve" else "detect"
        mode_index = self.images_panel.load_mode_combo.findData(mode)
        if mode_index >= 0:
            blocked = self.images_panel.load_mode_combo.blockSignals(True)
            self.images_panel.load_mode_combo.setCurrentIndex(mode_index)
            self.images_panel.load_mode_combo.blockSignals(blocked)
        title = f"SpriteTools GUI (prototype) - {manifest.project_name}"
        self.setWindowTitle(title)
        self._refresh_project_info_banner()
        self._update_project_action_buttons()
        self.statusBar().showMessage(f"Active project: {manifest.project_name}", 3000)

    def _animation_tags_for_manifest_settings(self, runtime_to_source_key: Dict[str, str]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for tag in self._animation_tags.values():
            remapped_frames: List[AnimationFrameEntry] = []
            for frame in tag.frames:
                mapped_key = runtime_to_source_key.get(frame.sprite_key, frame.sprite_key)
                mapped_key_text = str(mapped_key).strip()
                if not mapped_key_text:
                    continue
                remapped_frames.append(
                    AnimationFrameEntry(
                        sprite_key=mapped_key_text,
                        duration_frames=max(1, int(frame.duration_frames)),
                        gap_before_frames=max(0, int(getattr(frame, "gap_before_frames", 0))),
                        notes=str(frame.notes or ""),
                    )
                )
            payload.append(
                AnimationTag(
                    tag_id=tag.tag_id,
                    name=tag.name,
                    state_label=tag.state_label,
                    frames=remapped_frames,
                    notes=tag.notes,
                ).to_dict()
            )
        return payload

    def _load_animation_tags_from_manifest_settings(self) -> None:
        manifest = self._project_manifest
        if manifest is None:
            self._animation_tags = {}
            if hasattr(self, "animation_tag_list"):
                self._refresh_animation_tag_list()
            return
        raw = manifest.settings.get("animation_tags", [])
        tags: Dict[str, AnimationTag] = {}
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                try:
                    parsed = AnimationTag.from_dict(item)
                except Exception:  # noqa: BLE001
                    continue
                tags[parsed.tag_id] = parsed
        self._animation_tags = tags
        logger.debug("Loaded animation tags count=%s", len(self._animation_tags))
        if hasattr(self, "animation_tag_list"):
            self._refresh_animation_tag_list()

    def _refresh_animation_tag_list(self) -> None:
        list_widget = self.animation_tag_list
        current_item = list_widget.currentItem()
        selected_id = str(current_item.data(Qt.ItemDataRole.UserRole)) if current_item is not None else ""
        blocked = list_widget.blockSignals(True)
        list_widget.clear()
        for tag_id in sorted(self._animation_tags.keys(), key=lambda key: self._animation_tags[key].name.lower()):
            tag = self._animation_tags[tag_id]
            label = f"{tag.name} ({len(tag.frames)})"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, tag.tag_id)
            item.setToolTip(f"State: {tag.state_label or '—'}\nFrames: {len(tag.frames)}")
            list_widget.addItem(item)
            if selected_id and tag.tag_id == selected_id:
                list_widget.setCurrentItem(item)
        if list_widget.currentItem() is None and list_widget.count() > 0:
            list_widget.setCurrentRow(0)
        list_widget.blockSignals(blocked)
        self._update_animation_tag_controls()

    def _refresh_animation_frame_list(self) -> None:
        frame_list = self.animation_frame_list
        tag = self._selected_animation_tag()
        self._sync_animation_timeline_range_controls(tag)
        selected_frame_index = self._selected_animation_frame_index()
        zoom_scale = max(0.50, min(2.20, float(self._animation_timeline_zoom) / 100.0))
        icon_side = max(38, min(90, int(round(52 * zoom_scale))))
        icon_size = QSize(icon_side, icon_side)
        frame_list.setIconSize(icon_size)
        duration_unit_px = self._animation_timeline_duration_unit_px()
        frame_widths: List[int] = []
        frame_durations: List[int] = []
        self._animation_frame_list_syncing = True
        blocked = frame_list.blockSignals(True)
        frame_list.clear()
        if tag is not None:
            for index, frame in enumerate(tag.frames):
                runtime_key = self._resolve_animation_runtime_key(frame.sprite_key)
                record = self.sprite_records.get(runtime_key) if runtime_key is not None else None
                frame_name = record.path.name if record is not None else Path(frame.sprite_key).name
                duration = max(1, int(frame.duration_frames))
                gap_before = self._animation_frame_gap_before(frame)
                if gap_before > 0:
                    gap_w = self._animation_timeline_frame_block_width(gap_before)
                    gap_item = QListWidgetItem("")
                    gap_item.setData(_TIMELINE_ITEM_KIND_ROLE, "gap")
                    gap_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    gap_item.setBackground(QColor(0, 0, 0, 0))
                    gap_item.setToolTip(f"Empty timeline space: {gap_before} frame(s)")
                    gap_item.setSizeHint(QSize(gap_w, max(72, int(round(74 * zoom_scale)))))
                    frame_list.addItem(gap_item)
                    frame_widths.append(gap_w)
                    frame_durations.append(gap_before)

                item = QListWidgetItem(f"≡ {index + 1:03d}\n{duration}f")
                item.setData(Qt.ItemDataRole.UserRole, index)
                item.setData(_TIMELINE_ITEM_KIND_ROLE, "frame")
                frame_name_stem = Path(frame_name).stem
                item.setData(_TIMELINE_FRAME_LABEL_ROLE, f"{index + 1} {frame_name_stem}")
                item.setData(_TIMELINE_FRAME_NUMBER_ROLE, int(index + 1))
                item.setData(_TIMELINE_FRAME_NAME_ROLE, frame_name_stem)
                item.setData(_TIMELINE_FRAME_DURATION_ROLE, f"{duration}f")
                item.setToolTip(f"{frame_name}\nDuration: {duration} frame(s)\nGap Before: {gap_before} frame(s)")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsDragEnabled
                    | Qt.ItemFlag.ItemIsDropEnabled
                )
                item.setBackground(QColor(46, 46, 46))
                block_w = self._animation_timeline_frame_block_width(duration)
                block_h = max(72, int(round(74 * zoom_scale)))
                item.setSizeHint(QSize(block_w, block_h))
                frame_widths.append(block_w)
                frame_durations.append(duration)
                preview: QPixmap | None = None
                if record is not None:
                    preview = self._get_animation_preview_pixmap(record)
                icon = self._build_animation_timeline_cell_icon(preview, icon_size)
                if icon is not None:
                    item.setIcon(icon)
                frame_list.addItem(item)
            content_total = self._animation_tag_total_timeline_frames(tag)
            tail_gap = max(0, int(self._timeline_total_frames_for_tag(tag)) - int(content_total))
            if tail_gap > 0:
                tail_gap_item = QListWidgetItem("")
                tail_gap_item.setData(_TIMELINE_ITEM_KIND_ROLE, "gap")
                tail_gap_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                tail_gap_item.setBackground(QColor(0, 0, 0, 0))
                tail_gap_item.setToolTip(f"Timeline end space: {tail_gap} frame(s)")
                tail_gap_item.setSizeHint(QSize(self._animation_timeline_frame_block_width(tail_gap), max(72, int(round(74 * zoom_scale)))))
                frame_list.addItem(tail_gap_item)
                frame_widths.append(self._animation_timeline_frame_block_width(tail_gap))
                frame_durations.append(tail_gap)
        if frame_list.count() > 0:
            if selected_frame_index is not None:
                selected_row = self._timeline_row_for_frame_index(selected_frame_index)
                if selected_row is not None:
                    frame_list.setCurrentRow(selected_row)
            if frame_list.currentItem() is None:
                for row in range(frame_list.count()):
                    item = frame_list.item(row)
                    if item is not None and str(item.data(_TIMELINE_ITEM_KIND_ROLE) or "") == "frame":
                        frame_list.setCurrentRow(row)
                        break
        frame_list.blockSignals(blocked)
        self._animation_frame_list_syncing = False
        self.animation_timeline_ruler.set_timeline_layout(
            frame_widths=frame_widths,
            frame_durations=frame_durations,
            duration_unit_px=duration_unit_px,
            zoom_percent=self._animation_timeline_zoom,
        )
        self.animation_timeline_ruler.set_scroll_x(self.animation_frame_list.horizontalScrollBar().value())
        self.animation_timeline_ruler.set_playhead_frame(self._animation_playhead_frame)
        self.animation_timeline_ruler.set_range_frames(
            int(self._animation_timeline_in_frame),
            int(self._animation_timeline_out_frame),
        )

    def _animation_timeline_duration_unit_px(self) -> int:
        zoom_scale = max(0.50, min(2.20, float(self._animation_timeline_zoom) / 100.0))
        return max(12, int(round(28 * zoom_scale)))

    def _animation_timeline_frame_block_width(self, duration_frames: int) -> int:
        span = max(1, int(duration_frames))
        return max(24, span * self._animation_timeline_duration_unit_px())

    def _build_animation_timeline_cell_icon(
        self,
        preview: QPixmap | None,
        icon_size: QSize,
    ) -> QIcon | None:
        width = max(24, int(icon_size.width()))
        height = max(24, int(icon_size.height()))
        if width <= 0 or height <= 0:
            return None
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        body_rect = QRect(0, 0, width - 1, height - 1)
        painter.setPen(QPen(QColor(96, 96, 96), 1))
        painter.setBrush(QColor(44, 44, 44))
        painter.drawRoundedRect(body_rect, 3, 3)
        painter.setPen(QPen(QColor(126, 126, 126), 1))
        painter.drawLine(0, 0, 0, height - 1)
        painter.drawLine(width - 1, 0, width - 1, height - 1)
        if preview is not None:
            thumb = preview.scaled(QSize(width, height), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            x = (width - thumb.width()) // 2
            y = (height - thumb.height()) // 2
            painter.drawPixmap(x, y, thumb)
            painter.end()
            return QIcon(pixmap)

        painter.fillRect(QRect(1, 1, width - 2, height - 2), QColor(40, 40, 40))
        painter.end()
        return QIcon(pixmap)

    def _selected_animation_frame_index(self) -> int | None:
        item = self.animation_frame_list.currentItem()
        if item is None:
            return None
        if str(item.data(_TIMELINE_ITEM_KIND_ROLE) or "") != "frame":
            return None
        try:
            index = int(item.data(Qt.ItemDataRole.UserRole))
        except (TypeError, ValueError):
            return None
        return index if index >= 0 else None

    def _timeline_row_for_frame_index(self, frame_index: int) -> int | None:
        target = int(frame_index)
        for row in range(self.animation_frame_list.count()):
            item = self.animation_frame_list.item(row)
            if item is None or str(item.data(_TIMELINE_ITEM_KIND_ROLE) or "") != "frame":
                continue
            try:
                value = int(item.data(Qt.ItemDataRole.UserRole))
            except (TypeError, ValueError):
                continue
            if value == target:
                return row
        return None

    def _timeline_item_for_frame_index(self, frame_index: int) -> QListWidgetItem | None:
        row = self._timeline_row_for_frame_index(frame_index)
        if row is None:
            return None
        return self.animation_frame_list.item(row)

    def _selected_animation_frame_rows(self) -> List[int]:
        rows: List[int] = []
        for item in self.animation_frame_list.selectedItems():
            if str(item.data(_TIMELINE_ITEM_KIND_ROLE) or "") != "frame":
                continue
            try:
                rows.append(int(item.data(Qt.ItemDataRole.UserRole)))
            except (TypeError, ValueError):
                continue
        rows = sorted({row for row in rows if row >= 0})
        if rows:
            return rows
        frame_index = self._selected_animation_frame_index()
        return [frame_index] if frame_index is not None else []

    def _selected_animation_tag(self) -> AnimationTag | None:
        item = self.animation_tag_list.currentItem()
        if item is None:
            return None
        tag_id = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not tag_id:
            return None
        return self._animation_tags.get(tag_id)

    def _animation_frame_start_offset(self, tag: AnimationTag, frame_index: int) -> int:
        if frame_index <= 0:
            return max(0, int(getattr(tag.frames[0], "gap_before_frames", 0))) if tag.frames else 0
        elapsed = 0
        for frame in tag.frames[:frame_index]:
            elapsed += max(0, int(getattr(frame, "gap_before_frames", 0)))
            elapsed += max(1, int(frame.duration_frames))
        elapsed += max(0, int(getattr(tag.frames[frame_index], "gap_before_frames", 0)))
        return elapsed

    def _animation_frame_index_at_timeline_position(self, tag: AnimationTag, frame_position: float) -> int | None:
        if not tag.frames:
            return None
        cursor = max(0.0, float(frame_position))
        elapsed = 0.0
        for index, frame in enumerate(tag.frames):
            gap = float(max(0, int(getattr(frame, "gap_before_frames", 0))))
            duration = float(max(1, int(frame.duration_frames)))
            frame_start = elapsed + gap
            frame_end = frame_start + duration
            if cursor < frame_start:
                return max(0, index - 1)
            if frame_start <= cursor < frame_end:
                return index
            elapsed = frame_end
        return len(tag.frames) - 1

    def _animation_frame_gap_before(self, frame: AnimationFrameEntry) -> int:
        return max(0, int(getattr(frame, "gap_before_frames", 0)))

    def _animation_frame_total_span(self, frame: AnimationFrameEntry) -> int:
        return self._animation_frame_gap_before(frame) + max(1, int(frame.duration_frames))

    def _animation_tag_total_timeline_frames(self, tag: AnimationTag | None) -> int:
        if tag is None:
            return 0
        total = 0
        for frame in tag.frames:
            total += self._animation_frame_gap_before(frame)
            total += max(1, int(frame.duration_frames))
        return max(0, int(total))

    def _animation_timeline_slot_at_position(self, tag: AnimationTag, position: int) -> tuple[int | None, int, int]:
        cursor = max(0, int(position))
        elapsed = 0
        for index, frame in enumerate(tag.frames):
            gap = self._animation_frame_gap_before(frame)
            duration = max(1, int(frame.duration_frames))
            frame_start = elapsed + gap
            frame_end = frame_start + duration
            if cursor < frame_start:
                return None, max(0, cursor - elapsed), elapsed
            if frame_start <= cursor < frame_end:
                return index, max(0, cursor - frame_start), frame_start
            elapsed = frame_end
        return None, 0, elapsed

    def _animation_debug_point_context(self, point: QPoint) -> str:
        item = self.animation_frame_list.itemAt(point)
        kind = str(item.data(_TIMELINE_ITEM_KIND_ROLE) or "") if item is not None else "none"
        item_row = self.animation_frame_list.row(item) if item is not None else -1
        frame_index: int | None = None
        if item is not None and kind == "frame":
            try:
                frame_index = int(item.data(Qt.ItemDataRole.UserRole))
            except (TypeError, ValueError):
                frame_index = None
        gap_target_frame_index = self._animation_gap_target_frame_index_at_point(point)
        gap_target_timeline_position = self._animation_gap_target_timeline_position_at_point(point)
        insert_index, marker_x = self._animation_drop_insert_index_from_point(point)
        scroll_x = int(self.animation_frame_list.horizontalScrollBar().value())
        world_x = max(0, int(point.x()) + scroll_x)
        unit_px = max(1, self._animation_timeline_duration_unit_px())
        world_timeline_pos = max(0, int(world_x // unit_px))
        return (
            f"point=({int(point.x())},{int(point.y())}) kind={kind} item_row={item_row} frame_index={frame_index} "
            f"insert_index={insert_index} marker_x={marker_x} scroll_x={scroll_x} world_x={world_x} "
            f"world_timeline_pos={world_timeline_pos} gap_target_frame={gap_target_frame_index} "
            f"gap_target_timeline_pos={gap_target_timeline_position}"
        )

    def _log_animation_timeline_snapshot(self, reason: str) -> None:
        tag = self._selected_animation_tag()
        if tag is None:
            logger.debug("Timeline snapshot reason=%s tag=None", reason)
            return
        entries: list[str] = []
        elapsed = 0
        for index, frame in enumerate(tag.frames):
            gap = self._animation_frame_gap_before(frame)
            duration = max(1, int(frame.duration_frames))
            start = elapsed + gap
            end = start + duration
            entries.append(f"{index}:g{gap} d{duration} [{start},{end}) key={frame.sprite_key}")
            elapsed = end
        logger.debug(
            "Timeline snapshot reason=%s tag=%s in=%s out=%s total=%s playhead=%s frames=%s",
            reason,
            tag.name,
            int(self._animation_timeline_in_frame),
            int(self._animation_timeline_out_frame),
            int(elapsed),
            self._animation_playhead_frame,
            " | ".join(entries) if entries else "<empty>",
        )

    def _animation_frame_row_at_point(self, point: QPoint) -> int | None:
        item = self.animation_frame_list.itemAt(point)
        if item is None:
            return None
        if str(item.data(_TIMELINE_ITEM_KIND_ROLE) or "") != "frame":
            return None
        try:
            frame_index = int(item.data(Qt.ItemDataRole.UserRole))
        except (TypeError, ValueError):
            return None
        return frame_index if frame_index >= 0 else None

    def _animation_frame_edge_hit(self, row: int, point: QPoint) -> Literal["left", "right"] | None:
        if row < 0:
            return None
        item = self._timeline_item_for_frame_index(row)
        if item is None:
            return None
        rect = self.animation_frame_list.visualItemRect(item)
        if not rect.isValid():
            return None
        edge_w = min(14, max(6, int(rect.width() * 0.14)))
        left_edge = QRect(rect.left(), rect.top(), edge_w, rect.height())
        right_edge = QRect(rect.right() - edge_w + 1, rect.top(), edge_w, rect.height())
        if left_edge.contains(point):
            return "left"
        if right_edge.contains(point):
            return "right"
        return None

    def _show_animation_timeline_marker(self, x: int, *, color: str = "#7fb6ff") -> None:
        if not hasattr(self, "_animation_timeline_marker"):
            return
        viewport = self.animation_frame_list.viewport()
        marker_x = max(0, min(int(x), max(0, viewport.width() - 1)))
        self._animation_timeline_marker.setStyleSheet(f"background-color: {color};")
        self._animation_timeline_marker.setGeometry(marker_x, 0, 2, max(1, viewport.height()))
        self._animation_timeline_marker.setVisible(True)
        self._animation_timeline_marker.raise_()

    def _show_animation_timeline_drop_zone(self, rect: QRect, *, color: str = "rgba(127,182,255,70)") -> None:
        if not hasattr(self, "_animation_timeline_drop_zone"):
            return
        viewport = self.animation_frame_list.viewport()
        zone = QRect(rect)
        if zone.width() <= 0 or zone.height() <= 0:
            self._hide_animation_timeline_drop_zone()
            return
        zone.setLeft(max(0, zone.left()))
        zone.setRight(min(max(0, viewport.width() - 1), zone.right()))
        zone.setTop(0)
        zone.setHeight(max(1, viewport.height()))
        self._animation_timeline_drop_zone.setStyleSheet(f"background-color: {color}; border: 1px solid #7fb6ff;")
        self._animation_timeline_drop_zone.setGeometry(zone)
        self._animation_timeline_drop_zone.setVisible(True)
        self._animation_timeline_drop_zone.raise_()

    def _hide_animation_timeline_marker(self) -> None:
        if hasattr(self, "_animation_timeline_marker"):
            self._animation_timeline_marker.setVisible(False)

    def _hide_animation_timeline_drop_zone(self) -> None:
        if hasattr(self, "_animation_timeline_drop_zone"):
            self._animation_timeline_drop_zone.setVisible(False)

    def _animation_gap_rect_at_point(self, point: QPoint) -> QRect | None:
        item = self.animation_frame_list.itemAt(point)
        if item is None:
            return None
        if str(item.data(_TIMELINE_ITEM_KIND_ROLE) or "") != "gap":
            return None
        rect = self.animation_frame_list.visualItemRect(item)
        return rect if rect.isValid() else None

    def _animation_gap_slot_rect_at_point(self, point: QPoint) -> QRect | None:
        gap_rect = self._animation_gap_rect_at_point(point)
        if gap_rect is None:
            return None
        unit_px = max(1, self._animation_timeline_duration_unit_px())
        rel_x = max(0, min(gap_rect.width() - 1, int(point.x()) - gap_rect.left()))
        slot_index = max(0, int(rel_x // unit_px))
        slot_left = gap_rect.left() + (slot_index * unit_px)
        slot_right = min(gap_rect.right(), slot_left + unit_px - 1)
        return QRect(slot_left, gap_rect.top(), max(1, slot_right - slot_left + 1), gap_rect.height())

    def _animation_gap_target_frame_index_at_point(self, point: QPoint) -> int | None:
        item = self.animation_frame_list.itemAt(point)
        if item is None:
            return None
        if str(item.data(_TIMELINE_ITEM_KIND_ROLE) or "") != "gap":
            return None
        start_row = self.animation_frame_list.row(item)
        for row in range(start_row + 1, self.animation_frame_list.count()):
            next_item = self.animation_frame_list.item(row)
            if next_item is None:
                continue
            if str(next_item.data(_TIMELINE_ITEM_KIND_ROLE) or "") != "frame":
                continue
            try:
                frame_index = int(next_item.data(Qt.ItemDataRole.UserRole))
            except (TypeError, ValueError):
                return None
            return frame_index if frame_index >= 0 else None
        return None

    def _animation_gap_target_timeline_position_at_point(self, point: QPoint) -> int | None:
        item = self.animation_frame_list.itemAt(point)
        if item is None:
            return None
        if str(item.data(_TIMELINE_ITEM_KIND_ROLE) or "") != "gap":
            return None
        scroll_x = int(self.animation_frame_list.horizontalScrollBar().value())
        world_x = max(0, int(point.x()) + scroll_x)
        unit_px = max(1, self._animation_timeline_duration_unit_px())
        return max(0, int(world_x // unit_px))

    def _animation_timeline_position_from_point(self, point: QPoint) -> int:
        scroll_x = int(self.animation_frame_list.horizontalScrollBar().value())
        world_x = max(0, int(point.x()) + scroll_x)
        unit_px = max(1, self._animation_timeline_duration_unit_px())
        return max(0, int(world_x // unit_px))

    def _animation_drop_insert_index_from_point(self, point: QPoint) -> tuple[int, int]:
        count = self.animation_frame_list.count()
        viewport = self.animation_frame_list.viewport()
        if count <= 0:
            return 0, 0
        item = self.animation_frame_list.itemAt(point)
        if item is not None:
            kind = str(item.data(_TIMELINE_ITEM_KIND_ROLE) or "")
            rect = self.animation_frame_list.visualItemRect(item)
            if kind == "frame":
                try:
                    frame_index = int(item.data(Qt.ItemDataRole.UserRole))
                except (TypeError, ValueError):
                    frame_index = 0
                before = int(point.x()) < int(rect.center().x())
                return (frame_index if before else frame_index + 1), (rect.left() if before else rect.right() + 1)
            if kind == "gap":
                for row in range(self.animation_frame_list.row(item) + 1, count):
                    next_item = self.animation_frame_list.item(row)
                    if next_item is None:
                        continue
                    if str(next_item.data(_TIMELINE_ITEM_KIND_ROLE) or "") != "frame":
                        continue
                    try:
                        frame_index = int(next_item.data(Qt.ItemDataRole.UserRole))
                    except (TypeError, ValueError):
                        frame_index = 0
                    return frame_index, rect.left()
                marker_x = max(rect.left(), min(int(point.x()), rect.right() + 1))
                frame_total = 0
                for scan_row in range(count):
                    scan_item = self.animation_frame_list.item(scan_row)
                    if scan_item is not None and str(scan_item.data(_TIMELINE_ITEM_KIND_ROLE) or "") == "frame":
                        frame_total += 1
                return frame_total, marker_x
        frame_rows: List[int] = []
        for row in range(count):
            candidate = self.animation_frame_list.item(row)
            if candidate is None:
                continue
            if str(candidate.data(_TIMELINE_ITEM_KIND_ROLE) or "") == "frame":
                frame_rows.append(row)
        if not frame_rows:
            return 0, 0
        first_item = self.animation_frame_list.item(frame_rows[0])
        last_item = self.animation_frame_list.item(frame_rows[-1])
        if first_item is None or last_item is None:
            return 0, 0
        first_rect = self.animation_frame_list.visualItemRect(first_item)
        last_rect = self.animation_frame_list.visualItemRect(last_item)
        if point.x() <= first_rect.left():
            return 0, first_rect.left()
        if point.x() >= last_rect.right():
            return len(frame_rows), last_rect.right() + 1
        return len(frame_rows), min(max(int(point.x()), 0), viewport.width() - 1)

    def _apply_animation_manual_resize_delta(self, row: int, edge: Literal["left", "right"], delta_frames: int) -> bool:
        tag = self._selected_animation_tag()
        if tag is None or row < 0 or row >= len(tag.frames):
            return False
        frame = tag.frames[row]
        duration_origin = max(1, int(self._animation_manual_drag_resize_left_duration))
        gap_origin = max(0, int(self._animation_manual_drag_resize_right_duration))

        if edge == "left":
            shift = int(delta_frames)
            min_shift = -gap_origin
            max_shift = duration_origin - 1
            applied_shift = max(min_shift, min(max_shift, shift))
            new_gap = gap_origin + applied_shift
            new_duration = duration_origin - applied_shift
            changed = (self._animation_frame_gap_before(frame) != new_gap) or (int(frame.duration_frames) != new_duration)
            if not changed:
                return False
            frame.gap_before_frames = new_gap
            frame.duration_frames = new_duration
            logger.debug(
                "Timeline resize-apply row=%s edge=left delta=%s duration:%s->%s gap:%s->%s",
                row,
                delta_frames,
                duration_origin,
                new_duration,
                gap_origin,
                new_gap,
            )
            self.statusBar().showMessage(f"Frame {row + 1}: duration {new_duration}f, gap {new_gap}f", 900)
            return True

        new_duration = max(1, duration_origin + int(delta_frames))
        if int(frame.duration_frames) == new_duration:
            return False
        frame.duration_frames = new_duration
        logger.debug(
            "Timeline resize-apply row=%s edge=right delta=%s duration:%s->%s",
            row,
            delta_frames,
            duration_origin,
            new_duration,
        )
        self.statusBar().showMessage(f"Frame {row + 1}: duration {new_duration}f", 900)
        return True

    def _apply_animation_manual_reorder(
        self,
        source_row: int,
        target_insert_index: int,
        selected_rows: Sequence[int] | None = None,
        gap_target_timeline_position: int | None = None,
    ) -> None:
        tag = self._selected_animation_tag()
        if tag is None:
            return
        total = len(tag.frames)
        if total <= 1:
            return
        src = int(source_row)
        insert_index = int(target_insert_index)
        if src < 0 or src >= total:
            return
        insert_index = max(0, min(total, insert_index))
        total_before = self._animation_tag_total_timeline_frames(tag)

        self._log_animation_timeline_snapshot("reorder-before")

        rows = sorted({int(row) for row in (selected_rows or [src]) if 0 <= int(row) < total})
        if src not in rows:
            rows = [src]
        if not rows:
            return
        moving_includes_last_frame = rows[-1] == (total - 1)
        row_set = set(rows)
        source_hole_span = 0
        source_successor_original_index: int | None = None
        if gap_target_timeline_position is not None:
            elapsed_for_source = 0
            source_anchor = 0
            source_block_end = 0
            first_row = rows[0]
            last_row = rows[-1]
            for idx, frame in enumerate(tag.frames):
                gap = self._animation_frame_gap_before(frame)
                duration = max(1, int(frame.duration_frames))
                frame_start = elapsed_for_source + gap
                frame_end = frame_start + duration
                if idx == first_row:
                    source_anchor = elapsed_for_source
                if idx == last_row:
                    source_block_end = frame_end
                if idx > last_row and idx not in row_set and source_successor_original_index is None:
                    source_successor_original_index = idx
                elapsed_for_source = frame_end
            source_hole_span = max(0, source_block_end - source_anchor)
        if gap_target_timeline_position is None and insert_index >= rows[0] and insert_index <= (rows[-1] + 1):
            logger.debug(
                "Timeline reorder noop source=%s selected=%s insert=%s reason=same-range-no-gap-target",
                source_row,
                rows,
                insert_index,
            )
            return

        moving_frames = [tag.frames[row] for row in rows]
        remaining_frames = [frame for idx, frame in enumerate(tag.frames) if idx not in row_set]
        if (
            gap_target_timeline_position is not None
            and source_hole_span > 0
            and source_successor_original_index is not None
        ):
            successor_remaining_index = sum(1 for idx in range(source_successor_original_index) if idx not in row_set)
            if 0 <= successor_remaining_index < len(remaining_frames):
                successor_frame = remaining_frames[successor_remaining_index]
                successor_frame.gap_before_frames = max(
                    0,
                    self._animation_frame_gap_before(successor_frame) + source_hole_span,
                )
                logger.debug(
                    "Timeline reorder-source-hole-preserve source=%s selected=%s hole_span=%s successor_index=%s successor_key=%s successor_gap=%s",
                    source_row,
                    rows,
                    source_hole_span,
                    successor_remaining_index,
                    successor_frame.sprite_key,
                    self._animation_frame_gap_before(successor_frame),
                )
        removed_before_insert = sum(1 for row in rows if row < insert_index)
        insert_in_remaining = max(0, min(len(remaining_frames), insert_index - removed_before_insert))
        target_total_after = int(total_before)
        if moving_frames:
            moving_span = 0
            for idx, frame in enumerate(moving_frames):
                moving_span += max(1, int(frame.duration_frames))
                if idx > 0:
                    moving_span += self._animation_frame_gap_before(frame)

            if gap_target_timeline_position is None:
                moving_frames[0].gap_before_frames = 0
            else:
                desired = max(0, int(gap_target_timeline_position))
                if moving_includes_last_frame:
                    target_total_after = desired + moving_span
                else:
                    target_total_after = max(int(total_before), desired + moving_span)
                elapsed = 0
                destination_index: int | None = None
                for idx, frame in enumerate(remaining_frames):
                    gap = self._animation_frame_gap_before(frame)
                    frame_start = elapsed + gap
                    if desired <= frame_start:
                        destination_index = idx
                        break
                    elapsed = frame_start + max(1, int(frame.duration_frames))

                if destination_index is None:
                    insert_in_remaining = len(remaining_frames)
                    total_timeline = max(
                        elapsed,
                        int(total_before),
                        int(self._animation_timeline_out_frame),
                        desired + moving_span,
                    )
                    available_tail_gap = max(0, total_timeline - elapsed)
                    if moving_span <= available_tail_gap:
                        block_start_max = max(elapsed, total_timeline - moving_span)
                    else:
                        block_start_max = total_timeline
                    block_start = max(elapsed, min(desired, block_start_max))
                    moving_frames[0].gap_before_frames = max(0, block_start - elapsed)
                    logger.debug(
                        "Timeline reorder-gap-tail source=%s selected=%s desired=%s elapsed=%s out=%s moving_span=%s available_tail=%s block_start=%s lead_gap=%s",
                        source_row,
                        rows,
                        desired,
                        elapsed,
                        total_timeline,
                        moving_span,
                        available_tail_gap,
                        block_start,
                        moving_frames[0].gap_before_frames,
                    )
                else:
                    insert_in_remaining = destination_index
                    destination = remaining_frames[destination_index]
                    destination_gap = self._animation_frame_gap_before(destination)
                    frame_start = elapsed + destination_gap
                    available_gap = max(0, frame_start - elapsed)
                    if moving_span <= available_gap:
                        block_start_max = frame_start - moving_span
                    else:
                        block_start_max = frame_start
                    block_start = max(elapsed, min(desired, block_start_max))
                    left_gap = max(0, block_start - elapsed)
                    right_gap = max(0, frame_start - block_start - moving_span)
                    moving_frames[0].gap_before_frames = left_gap
                    destination.gap_before_frames = right_gap
                    logger.debug(
                        "Timeline reorder-gap-inner source=%s selected=%s desired=%s dest_index=%s elapsed=%s frame_start=%s moving_span=%s available_gap=%s block_start=%s left_gap=%s right_gap=%s overflow=%s",
                        source_row,
                        rows,
                        desired,
                        destination_index,
                        elapsed,
                        frame_start,
                        moving_span,
                        available_gap,
                        block_start,
                        left_gap,
                        right_gap,
                        max(0, moving_span - available_gap),
                    )

        self._record_history("animation-tags-before-reorder", include_fields=["animation_tags"])

        tag.frames = remaining_frames[:insert_in_remaining] + moving_frames + remaining_frames[insert_in_remaining:]

        total_after = self._animation_tag_total_timeline_frames(tag)
        if total_after != target_total_after:
            delta = int(target_total_after - total_after)
            if delta > 0 and tag.frames:
                correction_target = tag.frames[-1]
                correction_target.gap_before_frames = max(0, self._animation_frame_gap_before(correction_target) + delta)
                logger.debug(
                    "Timeline reorder-total-correct source=%s selected=%s total_before=%s target_total=%s total_after=%s delta=%s mode=expand-tail target_key=%s target_gap=%s",
                    source_row,
                    rows,
                    total_before,
                    target_total_after,
                    total_after,
                    delta,
                    correction_target.sprite_key,
                    self._animation_frame_gap_before(correction_target),
                )
            elif delta < 0:
                remaining = -delta
                for frame in reversed(tag.frames):
                    if remaining <= 0:
                        break
                    gap = self._animation_frame_gap_before(frame)
                    if gap <= 0:
                        continue
                    consume = min(gap, remaining)
                    frame.gap_before_frames = gap - consume
                    remaining -= consume
                logger.debug(
                    "Timeline reorder-total-correct source=%s selected=%s total_before=%s target_total=%s total_after=%s delta=%s mode=shrink-gaps remaining=%s",
                    source_row,
                    rows,
                    total_before,
                    target_total_after,
                    total_after,
                    delta,
                    remaining,
                )
        self._refresh_animation_frame_list()
        first_row = max(0, min(insert_in_remaining, self.animation_frame_list.count() - 1))
        first_item_row = self._timeline_row_for_frame_index(first_row)
        if first_item_row is not None:
            self.animation_frame_list.setCurrentRow(first_item_row)
        selection_model = self.animation_frame_list.selectionModel()
        if selection_model is not None:
            selection_model.clearSelection()
            for offset in range(len(moving_frames)):
                row = first_row + offset
                item_row = self._timeline_row_for_frame_index(row)
                if item_row is None:
                    continue
                index = self.animation_frame_list.model().index(item_row, 0)
                selection_model.select(index, QItemSelectionModel.SelectionFlag.Select)
        if not self.animation_play_btn.isChecked():
            self._set_animation_playhead_from_selected_frame()
        self._record_history("animation-tags-reorder", include_fields=["animation_tags"], force=True)
        self._mark_project_dirty("animation-tags-reorder")
        self._log_animation_timeline_snapshot("reorder-after")

    def _apply_animation_manual_gap_drag(self, source_row: int, drag_delta_px: int, selected_rows: Sequence[int] | None = None) -> bool:
        tag = self._selected_animation_tag()
        if tag is None:
            return False
        rows = sorted({int(row) for row in (selected_rows or [source_row]) if 0 <= int(row) < len(tag.frames)})
        if not rows:
            return False
        lead_row = rows[0]
        frame = tag.frames[lead_row]
        unit_px = max(1, self._animation_timeline_duration_unit_px())
        delta_frames = int(round(float(drag_delta_px) / float(unit_px)))
        if delta_frames == 0:
            logger.debug("Timeline gap-drag skipped source=%s delta_px=%s delta_frames=0", source_row, drag_delta_px)
            return False
        origin_gap = self._animation_frame_gap_before(frame)
        new_gap = max(0, origin_gap + delta_frames)
        if new_gap == origin_gap:
            logger.debug(
                "Timeline gap-drag clamped source=%s origin_gap=%s delta_frames=%s",
                source_row,
                origin_gap,
                delta_frames,
            )
            return False
        self._log_animation_timeline_snapshot("gap-drag-before")
        self._record_history("animation-tags-before-gap-drag", include_fields=["animation_tags"])
        frame.gap_before_frames = new_gap
        logger.debug(
            "Timeline gap-drag apply lead_row=%s selected=%s delta_px=%s delta_frames=%s gap:%s->%s",
            lead_row,
            rows,
            drag_delta_px,
            delta_frames,
            origin_gap,
            new_gap,
        )
        self.statusBar().showMessage(f"Frame {lead_row + 1}: gap {new_gap}f", 1200)
        self._refresh_animation_frame_list()
        selected_item_row = self._timeline_row_for_frame_index(lead_row)
        if selected_item_row is not None:
            self.animation_frame_list.setCurrentRow(selected_item_row)
        selection_model = self.animation_frame_list.selectionModel()
        if selection_model is not None:
            selection_model.clearSelection()
            for row in rows:
                item_row = self._timeline_row_for_frame_index(row)
                if item_row is None:
                    continue
                idx = self.animation_frame_list.model().index(item_row, 0)
                selection_model.select(idx, QItemSelectionModel.SelectionFlag.Select)
        if not self.animation_play_btn.isChecked():
            self._set_animation_playhead_from_selected_frame()
        self._record_history("animation-tags-gap-drag", include_fields=["animation_tags"], force=True)
        self._mark_project_dirty("animation-tags-gap-drag")
        self._log_animation_timeline_snapshot("gap-drag-after")
        return True

    def _set_animation_playhead(self, frame_position: float | None) -> None:
        self._animation_playhead_frame = None if frame_position is None else max(0.0, float(frame_position))
        if hasattr(self, "animation_timeline_ruler"):
            self.animation_timeline_ruler.set_playhead_frame(self._animation_playhead_frame)

    def _set_animation_playhead_from_selected_frame(self) -> None:
        tag = self._selected_animation_tag()
        frame_index = self._selected_animation_frame_index()
        if tag is None or frame_index is None or frame_index < 0 or frame_index >= len(tag.frames):
            self._set_animation_playhead(None)
            return
        self._sync_animation_timeline_range_controls(tag)
        pos = self._animation_frame_start_offset(tag, frame_index)
        pos = max(self._animation_timeline_in_frame, min(self._animation_timeline_out_frame - 1, int(pos)))
        self._set_animation_playhead(float(pos))

    def _seek_animation_playhead(self, frame_position: float, *, from_user_scrub: bool = False) -> None:
        tag = self._selected_animation_tag()
        if tag is None or not tag.frames:
            self._set_animation_playhead(None)
            return
        self._sync_animation_timeline_range_controls(tag)
        total_frames = self._timeline_total_frames_for_tag(tag)
        if total_frames <= 0:
            self._set_animation_playhead(None)
            return
        timeline_pos = max(0, min(total_frames - 1, int(frame_position)))
        timeline_pos = max(self._animation_timeline_in_frame, min(self._animation_timeline_out_frame - 1, timeline_pos))
        frame_index, frame_step, frame_start = self._animation_timeline_slot_at_position(tag, timeline_pos)
        self._set_animation_playhead(float(timeline_pos))
        if self.animation_play_btn.isChecked() and from_user_scrub:
            self._stop_animation_preview()

        self._animation_preview_timeline_position = timeline_pos
        if frame_index is None:
            self._set_animation_empty_preview_canvas()
            self._animation_preview_frame_index = 0
            self._animation_preview_subframe_step = 0
            self._animation_preview_displayed_frame_index = -1
            self.animation_status_label.setText(f"{tag.name}: empty frame")
            return

        selected_index = self._selected_animation_frame_index()
        if selected_index != frame_index:
            blocked = self.animation_frame_list.blockSignals(True)
            target_row = self._timeline_row_for_frame_index(frame_index)
            if target_row is not None:
                self.animation_frame_list.setCurrentRow(target_row)
            self.animation_frame_list.blockSignals(blocked)
            self._update_animation_tag_controls()

        frame = tag.frames[frame_index]
        runtime_key = self._resolve_animation_runtime_key(frame.sprite_key)
        if runtime_key is not None:
            self._set_animation_preview_frame_by_key(runtime_key, source="seek-playhead")
        self._animation_preview_frame_index = frame_index
        self._animation_preview_subframe_step = max(0, min(max(0, max(1, int(frame.duration_frames)) - 1), int(frame_step)))
        self._animation_preview_displayed_frame_index = frame_index
        self._set_animation_playhead(float(frame_start + self._animation_preview_subframe_step))
        self._animation_preview_current_duration_frames = max(1, int(frame.duration_frames))

    def _on_animation_timeline_ruler_scrubbed(self, frame_position: float) -> None:
        self._seek_animation_playhead(frame_position, from_user_scrub=True)

    def _next_animation_tag_id(self) -> str:
        next_index = 1
        while True:
            candidate = f"anim_{next_index:03d}"
            if candidate not in self._animation_tags:
                return candidate
            next_index += 1

    def _update_animation_tag_controls(self) -> None:
        tag = self._selected_animation_tag()
        has_tag = tag is not None
        selected_count = len(self.images_panel.list_widget.selectedItems())
        self.animation_tag_delete_btn.setEnabled(has_tag)
        self.animation_assign_selected_btn.setEnabled(has_tag and selected_count > 0)
        self.animation_clear_frames_btn.setEnabled(has_tag and bool(tag.frames if tag else False))
        self.animation_prewarm_btn.setEnabled(has_tag and bool(tag.frames if tag else False))
        self.animation_frame_delete_btn.setEnabled(has_tag and bool(tag.frames if tag else False))
        self._sync_animation_timeline_range_controls(tag)
        self._refresh_animation_frame_list()
        if not has_tag:
            self.animation_status_label.setText("No tag selected")
            self._set_animation_playhead(None)
            if self.animation_play_btn.isChecked():
                self.animation_play_btn.blockSignals(True)
                self.animation_play_btn.setChecked(False)
                self.animation_play_btn.blockSignals(False)
            self.animation_play_btn.setEnabled(False)
            self.animation_frame_duration_spin.setEnabled(False)
            self.animation_frame_move_left_btn.setEnabled(False)
            self.animation_frame_move_right_btn.setEnabled(False)
            return
        self.animation_play_btn.setEnabled(bool(tag.frames))
        frame_index = self._selected_animation_frame_index()
        has_frame = frame_index is not None and frame_index < len(tag.frames)
        self.animation_frame_duration_spin.setEnabled(has_frame)
        self.animation_frame_move_left_btn.setEnabled(has_frame and frame_index is not None and frame_index > 0)
        self.animation_frame_move_right_btn.setEnabled(has_frame and frame_index is not None and frame_index < (len(tag.frames) - 1))
        if has_frame:
            blocked_duration = self.animation_frame_duration_spin.blockSignals(True)
            self.animation_frame_duration_spin.setValue(max(1, int(tag.frames[int(frame_index)].duration_frames)))
            self.animation_frame_duration_spin.blockSignals(blocked_duration)
        elif not self.animation_play_btn.isChecked():
            self._set_animation_playhead(None)
        self.animation_status_label.setText(f"{tag.name}: {len(tag.frames)} frame(s)")

    def _on_animation_frame_selection_changed(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        if not self.animation_play_btn.isChecked():
            self._set_animation_playhead_from_selected_frame()
        self._update_animation_tag_controls()

    def _on_animation_frame_duration_changed(self, value: int) -> None:
        tag = self._selected_animation_tag()
        rows = self._selected_animation_frame_rows()
        if tag is None or not rows:
            return
        clamped = max(1, int(value))
        changed_rows = [row for row in rows if 0 <= row < len(tag.frames) and int(tag.frames[row].duration_frames) != clamped]
        if not changed_rows:
            return
        self._record_history("animation-tags-before-duration", include_fields=["animation_tags"])
        for row in changed_rows:
            tag.frames[row].duration_frames = clamped
        self._refresh_animation_frame_list()
        selection_model = self.animation_frame_list.selectionModel()
        if selection_model is not None:
            selection_model.clearSelection()
            for row in changed_rows:
                item_row = self._timeline_row_for_frame_index(row)
                if item_row is None:
                    continue
                index = self.animation_frame_list.model().index(item_row, 0)
                selection_model.select(index, QItemSelectionModel.SelectionFlag.Select)
            if changed_rows:
                current_row = self._timeline_row_for_frame_index(changed_rows[0])
                if current_row is not None:
                    self.animation_frame_list.setCurrentRow(current_row)
        if not self.animation_play_btn.isChecked():
            self._set_animation_playhead_from_selected_frame()
        self._record_history("animation-tags-duration", include_fields=["animation_tags"], force=True)
        self._mark_project_dirty("animation-tags-duration")

    def _on_animation_timeline_zoom_changed(self, value: int) -> None:
        clamped = max(50, min(220, int(value)))
        if clamped == self._animation_timeline_zoom and self.animation_timeline_zoom_value_label.text() == f"{clamped}%":
            return
        self._animation_timeline_zoom = clamped
        self.animation_timeline_zoom_value_label.setText(f"{clamped}%")
        self._refresh_animation_frame_list()

    def _delete_selected_animation_frames(self) -> None:
        tag = self._selected_animation_tag()
        if tag is None or not tag.frames:
            return
        selected_rows = sorted({index.row() for index in self.animation_frame_list.selectedIndexes()})
        if not selected_rows:
            frame_index = self._selected_animation_frame_index()
            if frame_index is None:
                return
            selected_rows = [frame_index]
        selected_set = set(selected_rows)
        remaining = [frame for row, frame in enumerate(tag.frames) if row not in selected_set]
        if len(remaining) == len(tag.frames):
            return
        self._record_history("animation-tags-before-delete-frames", include_fields=["animation_tags"])
        tag.frames = remaining
        if self._animation_preview_tag_id == tag.tag_id and not tag.frames:
            self._stop_animation_preview()
        self._refresh_animation_frame_list()
        if self.animation_frame_list.count() > 0:
            target_row = max(0, min(selected_rows[0], self.animation_frame_list.count() - 1))
            self.animation_frame_list.setCurrentRow(target_row)
        if not self.animation_play_btn.isChecked():
            self._set_animation_playhead_from_selected_frame()
        self._record_history("animation-tags-delete-frames", include_fields=["animation_tags"], force=True)
        self._mark_project_dirty("animation-tags-delete-frames")

    def _move_selected_animation_frame(self, delta: int) -> None:
        if delta == 0:
            return
        tag = self._selected_animation_tag()
        frame_index = self._selected_animation_frame_index()
        if tag is None or frame_index is None or frame_index < 0 or frame_index >= len(tag.frames):
            return
        new_index = frame_index + int(delta)
        if new_index < 0 or new_index >= len(tag.frames):
            return
        self._record_history("animation-tags-before-move-frame", include_fields=["animation_tags"])
        frame = tag.frames.pop(frame_index)
        tag.frames.insert(new_index, frame)
        self._refresh_animation_frame_list()
        self.animation_frame_list.setCurrentRow(new_index)
        if not self.animation_play_btn.isChecked():
            self._set_animation_playhead_from_selected_frame()
        self._record_history("animation-tags-move-frame", include_fields=["animation_tags"], force=True)
        self._mark_project_dirty("animation-tags-move-frame")

    def _compute_animation_tag_order_from_frame_list(self) -> List[AnimationFrameEntry] | None:
        tag = self._selected_animation_tag()
        if tag is None:
            return None
        old_frames = list(tag.frames)
        if not old_frames:
            return None
        reordered: List[AnimationFrameEntry] = []
        seen: set[int] = set()
        for row in range(self.animation_frame_list.count()):
            item = self.animation_frame_list.item(row)
            if item is None:
                continue
            try:
                old_index = int(item.data(Qt.ItemDataRole.UserRole))
            except (TypeError, ValueError):
                continue
            if old_index < 0 or old_index >= len(old_frames) or old_index in seen:
                continue
            reordered.append(old_frames[old_index])
            seen.add(old_index)
        if len(reordered) != len(old_frames):
            return None
        if all(reordered[i] is old_frames[i] for i in range(len(old_frames))):
            return None
        return reordered

    def _on_animation_frame_rows_moved(self, *args: Any) -> None:
        if self._animation_frame_list_syncing:
            return
        tag = self._selected_animation_tag()
        if tag is None:
            return
        reordered = self._compute_animation_tag_order_from_frame_list()
        if reordered is None:
            return
        self._record_history("animation-tags-before-reorder", include_fields=["animation_tags"])
        tag.frames = reordered
        self._refresh_animation_frame_list()
        if not self.animation_play_btn.isChecked():
            self._set_animation_playhead_from_selected_frame()
        self._record_history("animation-tags-reorder", include_fields=["animation_tags"], force=True)
        self._mark_project_dirty("animation-tags-reorder")

    def _prewarm_selected_animation_tag_preview(self) -> None:
        tag = self._selected_animation_tag()
        if tag is None or not tag.frames:
            return
        warmed = 0
        for frame in tag.frames:
            runtime_key = self._resolve_animation_runtime_key(frame.sprite_key)
            if runtime_key is None:
                continue
            record = self.sprite_records.get(runtime_key)
            if record is None:
                continue
            if self._get_animation_preview_pixmap(record) is not None:
                warmed += 1
        self.statusBar().showMessage(f"Animation prewarm complete: {warmed}/{len(tag.frames)} frame(s)", 2500)

    def _create_animation_tag(self) -> None:
        name, accepted = QInputDialog.getText(self, "New Animation Tag", "Tag name:")
        if not accepted:
            return
        text = str(name).strip()
        if not text:
            return
        tag_id = self._next_animation_tag_id()
        self._record_history("animation-tags-before-create", include_fields=["animation_tags"])
        self._animation_tags[tag_id] = AnimationTag(tag_id=tag_id, name=text)
        self._refresh_animation_tag_list()
        self._record_history("animation-tags-create", include_fields=["animation_tags"], force=True)
        self._mark_project_dirty("animation-tags-create")

    def _delete_selected_animation_tag(self) -> None:
        tag = self._selected_animation_tag()
        if tag is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete Animation Tag",
            f"Delete animation tag '{tag.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._record_history("animation-tags-before-delete", include_fields=["animation_tags"])
        if self._animation_preview_tag_id == tag.tag_id:
            self._stop_animation_preview()
        self._animation_tags.pop(tag.tag_id, None)
        self._refresh_animation_tag_list()
        if not self.animation_play_btn.isChecked():
            self._set_animation_playhead_from_selected_frame()
        self._record_history("animation-tags-delete", include_fields=["animation_tags"], force=True)
        self._mark_project_dirty("animation-tags-delete")

    def _assign_selected_sprites_to_animation_tag(self) -> None:
        tag = self._selected_animation_tag()
        if tag is None:
            return
        selected_keys: List[str] = []
        for item in self.images_panel.list_widget.selectedItems():
            key = item.data(Qt.ItemDataRole.UserRole)
            if key:
                selected_keys.append(str(key))
        if not selected_keys:
            return
        existing = {(frame.sprite_key, frame.duration_frames, frame.notes) for frame in tag.frames}
        keys_to_add: List[str] = []
        for key in selected_keys:
            candidate = (key, 1, "")
            if candidate in existing:
                continue
            keys_to_add.append(key)
            existing.add(candidate)
        added = len(keys_to_add)
        if added <= 0:
            self.statusBar().showMessage(f"Animation tag '{tag.name}' already contains selected sprites", 2500)
            return
        self._record_history("animation-tags-before-assign", include_fields=["animation_tags"])
        for key in keys_to_add:
            tag.frames.append(AnimationFrameEntry(sprite_key=key, duration_frames=1, notes=""))
        self._refresh_animation_tag_list()
        if not self.animation_play_btn.isChecked():
            self._set_animation_playhead_from_selected_frame()
        self._record_history("animation-tags-assign", include_fields=["animation_tags"], force=True)
        self._mark_project_dirty("animation-tags-assign")
        self.statusBar().showMessage(f"Added {added} frame(s) to animation tag '{tag.name}'", 2500)

    def _clear_selected_animation_tag_frames(self) -> None:
        tag = self._selected_animation_tag()
        if tag is None:
            return
        if not tag.frames:
            return
        self._record_history("animation-tags-before-clear", include_fields=["animation_tags"])
        tag.frames = []
        if self._animation_preview_tag_id == tag.tag_id:
            self._stop_animation_preview()
        self._refresh_animation_tag_list()
        if not self.animation_play_btn.isChecked():
            self._set_animation_playhead_from_selected_frame()
        self._record_history("animation-tags-clear", include_fields=["animation_tags"], force=True)
        self._mark_project_dirty("animation-tags-clear")

    def _on_animation_tag_selection_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if self.animation_play_btn.isChecked():
            self._stop_animation_preview()
        else:
            self._set_animation_playhead_from_selected_frame()
        self._update_animation_tag_controls()

    def _on_animation_preview_fps_changed(self, value: int) -> None:
        if not self.animation_play_btn.isChecked():
            return
        self._animation_preview_timer.start(self._animation_preview_step_interval_ms())

    def _timeline_total_frames_for_tag(self, tag: AnimationTag | None) -> int:
        content_total = self._animation_tag_total_timeline_frames(tag) if tag is not None else 0
        return max(1, int(content_total), int(self._animation_timeline_out_frame))

    def _sync_animation_timeline_range_controls(self, tag: AnimationTag | None = None) -> None:
        active_tag = tag if tag is not None else self._selected_animation_tag()
        out_value = max(1, int(self._animation_timeline_out_frame))
        in_value = max(0, min(int(self._animation_timeline_in_frame), out_value - 1))
        out_min = max(1, in_value + 1)
        out_value = max(out_min, out_value)
        self._animation_timeline_in_frame = in_value
        self._animation_timeline_out_frame = out_value

        if hasattr(self, "animation_in_spin") and hasattr(self, "animation_out_spin"):
            self._animation_timeline_range_syncing = True
            in_blocked = self.animation_in_spin.blockSignals(True)
            out_blocked = self.animation_out_spin.blockSignals(True)
            self.animation_out_spin.setMinimum(out_min)
            self.animation_out_spin.setValue(out_value)
            self.animation_in_spin.setMaximum(max(0, out_value - 1))
            self.animation_in_spin.setValue(in_value)
            self.animation_in_spin.blockSignals(in_blocked)
            self.animation_out_spin.blockSignals(out_blocked)
            self._animation_timeline_range_syncing = False
        if hasattr(self, "animation_timeline_ruler"):
            self.animation_timeline_ruler.set_range_frames(in_value, out_value)

    def _on_animation_timeline_range_edit_started(self) -> None:
        if self._history_manager is not None and self._history_manager.is_restoring:
            return
        if self._animation_timeline_range_history_before is None:
            self._animation_timeline_range_history_before = (
                int(self._animation_timeline_in_frame),
                int(self._animation_timeline_out_frame),
            )

    def _on_animation_timeline_range_edit_finished(self) -> None:
        self._commit_animation_timeline_range_history()

    def _queue_animation_timeline_range_history_commit(self) -> None:
        if self._history_manager is not None and self._history_manager.is_restoring:
            return
        if self._animation_timeline_range_history_before is None:
            self._animation_timeline_range_history_before = (
                int(self._animation_timeline_in_frame),
                int(self._animation_timeline_out_frame),
            )
        self._animation_timeline_range_history_commit_timer.start()

    def _clear_animation_timeline_range_history_pending(self) -> None:
        self._animation_timeline_range_history_commit_timer.stop()
        self._animation_timeline_range_history_before = None

    def _commit_animation_timeline_range_history(self) -> None:
        self._animation_timeline_range_history_commit_timer.stop()
        before = self._animation_timeline_range_history_before
        self._animation_timeline_range_history_before = None
        if before is None:
            return
        after = (int(self._animation_timeline_in_frame), int(self._animation_timeline_out_frame))
        if before == after:
            return
        self._record_history(
            "animation-timeline-range",
            include_fields=["animation_timeline_range"],
            force=True,
            mark_dirty=False,
        )

    def _on_animation_timeline_range_changed(self, _value: int) -> None:
        if self._animation_timeline_range_syncing:
            return
        self._on_animation_timeline_range_edit_started()
        new_in = int(self.animation_in_spin.value()) if hasattr(self, "animation_in_spin") else int(self._animation_timeline_in_frame)
        new_out = int(self.animation_out_spin.value()) if hasattr(self, "animation_out_spin") else int(self._animation_timeline_out_frame)
        self._animation_timeline_in_frame = max(0, new_in)
        self._animation_timeline_out_frame = max(1, new_out)
        self._sync_animation_timeline_range_controls()
        self._save_animation_timeline_ui_settings()
        if self.animation_play_btn.isChecked():
            pos = max(self._animation_timeline_in_frame, min(self._animation_timeline_out_frame - 1, int(self._animation_preview_timeline_position)))
            self._seek_animation_playhead(float(pos), from_user_scrub=False)
        else:
            self._set_animation_playhead_from_selected_frame()
        self._refresh_animation_frame_list()
        self._queue_animation_timeline_range_history_commit()

    def _on_animation_timeline_ruler_range_scrubbed(self, in_frame: int, out_frame: int) -> None:
        if self._animation_timeline_range_syncing:
            return
        self._on_animation_timeline_range_edit_started()
        self._animation_timeline_in_frame = max(0, int(in_frame))
        self._animation_timeline_out_frame = max(self._animation_timeline_in_frame + 1, int(out_frame))
        self._sync_animation_timeline_range_controls()
        self._save_animation_timeline_ui_settings()
        if self.animation_play_btn.isChecked():
            pos = max(self._animation_timeline_in_frame, min(self._animation_timeline_out_frame - 1, int(self._animation_preview_timeline_position)))
            self._seek_animation_playhead(float(pos), from_user_scrub=False)
        else:
            self._set_animation_playhead_from_selected_frame()
        self._refresh_animation_frame_list()
        self._queue_animation_timeline_range_history_commit()

    def _animation_preview_step_interval_ms(self) -> int:
        fps = max(1, int(self.animation_fps_spin.value()))
        return max(1, int(round(1000.0 / float(fps))))

    def _toggle_animation_preview(self, enabled: bool) -> None:
        if enabled:
            self._start_animation_preview()
        else:
            self._stop_animation_preview()

    def _toggle_animation_play_pause_shortcut(self) -> None:
        if not hasattr(self, "animation_play_btn"):
            return
        if not self.animation_timeline_dialog.isVisible():
            self._open_animation_timeline_dialog()
        if not self.animation_play_btn.isEnabled():
            return
        self.animation_play_btn.toggle()

    def _seek_animation_by_delta(self, delta: int) -> None:
        if delta == 0:
            return
        tag = self._selected_animation_tag()
        if tag is None or not tag.frames:
            return
        if not self.animation_timeline_dialog.isVisible():
            self._open_animation_timeline_dialog()
        base_pos: int
        if self._animation_playhead_frame is not None:
            base_pos = int(round(float(self._animation_playhead_frame)))
        else:
            selected_index = self._selected_animation_frame_index()
            if selected_index is None:
                base_pos = int(self._animation_preview_timeline_position)
            else:
                base_pos = int(self._animation_frame_start_offset(tag, selected_index))
        self._seek_animation_playhead(float(base_pos + int(delta)), from_user_scrub=True)

    def _start_animation_preview(self) -> None:
        tag = self._selected_animation_tag()
        if tag is None or not tag.frames:
            self.animation_play_btn.blockSignals(True)
            self.animation_play_btn.setChecked(False)
            self.animation_play_btn.blockSignals(False)
            return
        if self._preview_view_mode != "animation_assist":
            self._preview_view_mode = "animation_assist"
            self._sync_preview_context_controls()
            self._save_preview_ui_settings()
        self._last_preview_render_signature = None
        self.preview_panel.set_overlay_layers([], QSize(0, 0))
        self._preview_timer.stop()
        self._preview_pending_request = None
        self._animation_preview_tag_id = tag.tag_id
        self._animation_preview_frame_index = 0
        self._animation_preview_subframe_step = 0
        self._animation_preview_displayed_frame_index = -1
        self._sync_animation_timeline_range_controls(tag)
        self._animation_preview_timeline_position = max(0, int(self._animation_timeline_in_frame))
        self.animation_play_btn.setText("Pause")
        self._set_animation_playhead(float(self._animation_preview_timeline_position))
        self._advance_animation_preview()

    def _stop_animation_preview(self) -> None:
        self._animation_preview_timer.stop()
        self._assist_interaction_refresh_timer.stop()
        self._assist_interaction_refresh_pending = False
        self._animation_preview_tag_id = None
        self._animation_preview_frame_index = 0
        self._animation_preview_subframe_step = 0
        self._animation_preview_displayed_frame_index = -1
        self._animation_preview_visible_sprite_key = None
        self._animation_preview_timeline_position = 0
        self._animation_preview_current_duration_frames = 1
        self._last_preview_render_signature = None
        self.animation_play_btn.blockSignals(True)
        self.animation_play_btn.setChecked(False)
        self.animation_play_btn.blockSignals(False)
        self.animation_play_btn.setText("Play")
        self._set_animation_playhead_from_selected_frame()
        self._schedule_preview_update()

    def _resolve_animation_runtime_key(self, sprite_key: str) -> str | None:
        if sprite_key in self.sprite_records:
            return sprite_key
        if self._project_paths is not None:
            candidate = (self._project_paths.root / sprite_key).resolve().as_posix()
            if candidate in self.sprite_records:
                return candidate
        try:
            candidate = Path(sprite_key).resolve().as_posix()
            if candidate in self.sprite_records:
                return candidate
        except Exception:  # noqa: BLE001
            return None
        return None

    def _animation_preview_cache_key(self, record: SpriteRecord) -> tuple[Any, ...]:
        options = self._build_process_options(
            record,
            output_dir=record.path.parent / "_preview",
            preserve_palette=True,
            debug_context="animation-preview",
        )
        bg_transparency_signature: tuple[int, ...] = ()
        if self._preview_bg_transparent_enabled and self._preview_background_indices:
            bg_transparency_signature = tuple(sorted(int(idx) for idx in self._preview_background_indices))
        local_overrides_sig = tuple(
            (int(index), tuple(map(int, color)), int(alpha))
            for index, (color, alpha) in sorted(record.local_overrides.items(), key=lambda item: int(item[0]))
        )
        return (
            record.path.as_posix(),
            record.load_mode,
            options.canvas_size,
            options.fill_mode,
            int(options.fill_index),
            int(options.offset_x),
            int(options.offset_y),
            bool(self._preview_bg_transparent_enabled),
            bg_transparency_signature,
            local_overrides_sig,
        )

    def _get_animation_preview_pixmap(self, record: SpriteRecord) -> QPixmap | None:
        cache_key = self._animation_preview_cache_key(record)
        cached = self._animation_preview_pixmap_cache.get(cache_key)
        if cached is not None:
            self._animation_preview_pixmap_cache.move_to_end(cache_key)
            return cached
        options = self._build_process_options(
            record,
            output_dir=record.path.parent / "_preview",
            preserve_palette=True,
            debug_context="animation-preview",
        )
        try:
            processed, _palette = self._process_record_image(record, options)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Animation preview process failed key=%s err=%s", record.path.name, exc)
            return None
        processed_index_data: List[int] | None = None
        if processed.mode == "P":
            processed_index_data = list(processed.getdata())
            self._animation_preview_index_cache[cache_key] = (
                int(processed.width),
                int(processed.height),
                bytes(processed_index_data),
            )
            if len(self._animation_preview_index_cache) > 1024:
                self._animation_preview_index_cache.popitem(last=False)
        preview_rgba = self._apply_preview_background_transparency_to_rgba(
            processed.convert("RGBA"),
            processed_index_data,
        )
        qimage = ImageQt(preview_rgba).copy()
        pixmap = QPixmap.fromImage(qimage)
        self._animation_preview_pixmap_cache[cache_key] = pixmap
        if len(self._animation_preview_pixmap_cache) > 2048:
            self._animation_preview_pixmap_cache.popitem(last=False)
        return pixmap

    def _invalidate_animation_preview_pixmap_cache(self, *, refresh_current_frame: bool = False) -> None:
        self._animation_preview_pixmap_cache.clear()
        self._animation_preview_index_cache.clear()
        self._animation_assist_layer_cache.clear()
        if not refresh_current_frame:
            return
        if self.animation_play_btn.isChecked():
            if self._animation_playhead_frame is not None:
                self._seek_animation_playhead(float(self._animation_playhead_frame), from_user_scrub=False)
            else:
                self._set_animation_playhead_from_selected_frame()
            return
        self._set_animation_playhead_from_selected_frame()

    def _set_animation_preview_frame_by_key(self, key: str, *, source: str = "unknown") -> bool:
        assist_active = self._is_animation_assist_view_active() or self.animation_play_btn.isChecked()
        if not assist_active:
            return True

        prev_keys: List[str] = []
        next_keys: List[str] = []
        if self._preview_onion_enabled:
            if self._preview_onion_source_mode == "sprite_list":
                prev_keys, next_keys = self._sprite_list_onion_keys(key)
            else:
                tag = self._selected_animation_tag()
                if tag is not None:
                    timeline_pos = int(self._animation_preview_timeline_position)
                    prev_keys, next_keys = self._timeline_onion_keys(tag, timeline_pos)

        layers = self._compose_animation_assist_pixmap(key, prev_keys, next_keys)
        if layers is None:
            self._animation_preview_visible_sprite_key = None
            self.preview_panel.set_overlay_layers([], QSize(0, 0))
            return False
        current_pixmap, onion_pixmap = layers
        overlay_layers, overlay_source_size, overlay_sig = self._animation_assist_overlay_layers_for_key(key)
        self.preview_panel.set_static_pixmap(onion_pixmap)
        self.preview_panel.set_overlay_layers(overlay_layers, overlay_source_size)
        self.preview_panel.set_pixmap(current_pixmap, reset_zoom=False)
        self._animation_last_preview_size = QSize(current_pixmap.size())
        self._animation_preview_visible_sprite_key = key
        self._last_preview_render_signature = (
            "assist",
            key,
            bool(self._preview_onion_enabled),
            overlay_sig,
        )
        return True

    def _set_animation_empty_preview_canvas(self) -> None:
        if not self._is_animation_assist_view_active() and not self.animation_play_btn.isChecked():
            return
        self._animation_preview_visible_sprite_key = None
        size = QSize(self._animation_last_preview_size)
        if size.width() <= 0 or size.height() <= 0:
            self.preview_panel.set_static_pixmap(None)
            self.preview_panel.set_overlay_layers([], QSize(0, 0))
            self.preview_panel.set_pixmap(None, reset_zoom=False)
            return
        blank = QPixmap(size)
        blank.fill(Qt.GlobalColor.transparent)
        self.preview_panel.set_static_pixmap(None)
        self.preview_panel.set_overlay_layers([], QSize(0, 0))
        self.preview_panel.set_pixmap(blank, reset_zoom=False)

    def _advance_animation_preview(self) -> None:
        tag_id = self._animation_preview_tag_id
        if not tag_id:
            return
        tag = self._animation_tags.get(tag_id)
        if tag is None or not tag.frames:
            self._stop_animation_preview()
            return
        self._sync_animation_timeline_range_controls(tag)
        total_frames = self._timeline_total_frames_for_tag(tag)
        if total_frames <= 0:
            self._stop_animation_preview()
            return
        in_pos = max(0, min(total_frames - 1, int(self._animation_timeline_in_frame)))
        out_pos = max(in_pos + 1, min(total_frames, int(self._animation_timeline_out_frame)))
        if self._animation_preview_timeline_position < in_pos:
            self._animation_preview_timeline_position = in_pos
        if self._animation_preview_timeline_position >= out_pos:
            if self.animation_loop_check.isChecked():
                self._animation_preview_timeline_position = in_pos
            else:
                self._stop_animation_preview()
                return

        timeline_pos = int(self._animation_preview_timeline_position)
        frame_index, frame_step, frame_start = self._animation_timeline_slot_at_position(tag, timeline_pos)
        if frame_index is None:
            self._set_animation_empty_preview_canvas()
            self._animation_preview_displayed_frame_index = -1
            self.animation_status_label.setText(
                f"{tag.name}: empty frame {timeline_pos + 1}/{total_frames}"
            )
            self._set_animation_playhead(float(timeline_pos))
        else:
            frame = tag.frames[frame_index]
            frame_duration = max(1, int(frame.duration_frames))
            if self._animation_preview_displayed_frame_index != frame_index:
                runtime_key = self._resolve_animation_runtime_key(frame.sprite_key)
                if runtime_key is not None:
                    self._set_animation_preview_frame_by_key(runtime_key, source="advance")
                self._animation_preview_displayed_frame_index = frame_index
            self._animation_preview_frame_index = frame_index
            self._animation_preview_subframe_step = max(0, min(frame_duration - 1, int(frame_step)))
            self._animation_preview_current_duration_frames = frame_duration
            self.animation_status_label.setText(
                f"{tag.name}: frame {frame_index + 1}/{len(tag.frames)} ({self._animation_preview_subframe_step + 1}/{frame_duration}f)"
            )
            self._set_animation_playhead(float(frame_start + self._animation_preview_subframe_step))

        self._animation_preview_timeline_position += 1
        self._animation_preview_timer.start(self._animation_preview_step_interval_ms())

    def _refresh_project_info_banner(self) -> None:
        manifest = self._project_manifest
        paths = self._project_paths
        self.images_panel.set_project_info(
            name=(manifest.project_name if manifest is not None else None),
            mode=(manifest.project_mode if manifest is not None else None),
            root=(paths.root if paths is not None else None),
            dirty=bool(self._autosave_dirty),
            last_saved_text=self._last_project_saved_text,
            last_saved_kind=self._last_project_saved_kind,
            recovery_source_text=self._recovery_source_text,
        )

    def _update_project_action_buttons(self) -> None:
        has_project = (self._project_paths is not None) and (self._project_manifest is not None)
        enabled = not self._is_loading_sprites
        self.action_import_sprites.setEnabled(enabled)
        self.action_new_project.setEnabled(enabled)
        self.action_open_project.setEnabled(enabled)
        if self._recent_projects_menu is not None:
            self._recent_projects_menu.setEnabled(enabled)
        self.action_save_project.setEnabled(enabled and has_project)
        self.action_save_project_as.setEnabled(enabled)

    def _recent_project_manifest_paths(self) -> List[Path]:
        raw = self._settings.value("projects/recent_manifests", "")
        text = str(raw or "").strip()
        if not text:
            return []
        parsed: List[Path] = []
        seen: set[str] = set()
        for token in text.split("|"):
            candidate = Path(token.strip())
            if not candidate:
                continue
            key = candidate.as_posix().lower()
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists() and candidate.is_file() and candidate.name == PROJECT_MANIFEST_NAME:
                parsed.append(candidate)
        if len(parsed) < len([token for token in text.split("|") if token.strip()]):
            self._set_recent_project_manifest_paths(parsed)
        return parsed[:_RECENT_PROJECTS_LIMIT]

    def _set_recent_project_manifest_paths(self, manifests: Sequence[Path]) -> None:
        normalized: List[str] = []
        seen: set[str] = set()
        for path in manifests:
            key = path.as_posix().lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(path.as_posix())
            if len(normalized) >= _RECENT_PROJECTS_LIMIT:
                break
        self._set_pref("projects/recent_manifests", "|".join(normalized))

    def _remember_recent_project(self, manifest_path: Path) -> None:
        if not manifest_path.exists() or manifest_path.name != PROJECT_MANIFEST_NAME:
            return
        recent = self._recent_project_manifest_paths()
        fresh: List[Path] = [manifest_path]
        for existing in recent:
            if existing.resolve() == manifest_path.resolve():
                continue
            fresh.append(existing)
        self._set_recent_project_manifest_paths(fresh)

    def _ordered_sprite_keys(self) -> List[str]:
        keys: List[str] = []
        for row in range(self.images_panel.list_widget.count()):
            item = self.images_panel.list_widget.item(row)
            if item is None:
                continue
            key = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(key, str) and key:
                keys.append(key)
        return keys

    def _is_path_within(self, path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    def _replace_record_path(self, old_path: Path, new_path: Path) -> None:
        old_key = old_path.as_posix()
        new_key = new_path.as_posix()
        if old_key == new_key:
            return
        record = self.sprite_records.pop(old_key, None)
        if record is None:
            return
        self.sprite_records[new_key] = record
        if old_key in self._used_index_cache:
            self._used_index_cache[new_key] = self._used_index_cache.pop(old_key)
        if record.group_id:
            group = self._palette_groups.get(record.group_id)
            if group is not None:
                if old_key in group.member_keys:
                    group.member_keys.remove(old_key)
                group.member_keys.add(new_key)
        list_widget = self.images_panel.list_widget
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item is None:
                continue
            if item.data(Qt.ItemDataRole.UserRole) == old_key:
                item.setData(Qt.ItemDataRole.UserRole, new_key)
                item.setData(_SPRITE_BASE_NAME_ROLE, new_path.name)
                item.setText(new_path.name)
                break
        for tag in self._animation_tags.values():
            for frame in tag.frames:
                if frame.sprite_key == old_key:
                    frame.sprite_key = new_key

    def _project_output_dir_setting(self, paths: ProjectPaths) -> str:
        try:
            return self.output_dir.resolve().relative_to(paths.root).as_posix()
        except ValueError:
            return self.output_dir.resolve().as_posix()

    def _mark_project_dirty(self, reason: str, *, debounce_ms: int = 1800) -> None:
        if not self._autosave_enabled or self._autosave_recovery_mode:
            return
        if self._project_paths is None or self._project_manifest is None:
            return
        self._autosave_dirty = True
        self._refresh_project_info_banner()
        logger.debug("Project marked dirty reason=%s", reason)
        if debounce_ms <= 0:
            self._perform_autosave()
            return
        self._autosave_debounce_timer.start(max(200, int(debounce_ms)))

    def _show_autosave_status(self, message: str, *, is_error: bool = False) -> None:
        now = time.perf_counter()
        min_interval = 5.0 if is_error else 20.0
        if (now - self._autosave_last_status_ts) < min_interval:
            return
        self._autosave_last_status_ts = now
        self.statusBar().showMessage(message, 2500 if not is_error else 4500)

    def _autosave_snapshot_path(self, paths: ProjectPaths) -> Path:
        return paths.backups / "autosave" / "autosave_latest.json"

    def _autosave_snapshot_timestamp_path(self, paths: ProjectPaths) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return paths.backups / "autosave" / f"autosave_{stamp}.json"

    def _perform_autosave(self) -> None:
        if not self._autosave_enabled or self._autosave_recovery_mode:
            return
        if not self._autosave_dirty:
            return
        if self._is_loading_sprites:
            return
        if self._project_paths is None or self._project_manifest is None:
            return
        try:
            snapshot_payload = self._build_project_state_snapshot_payload(self._project_paths, self._project_manifest)
            autosave_dir = self._project_paths.backups / "autosave"
            autosave_dir.mkdir(parents=True, exist_ok=True)
            latest_path = self._autosave_snapshot_path(self._project_paths)
            latest_path.write_text(json.dumps(snapshot_payload, indent=2), encoding="utf-8")
            rolling_path = self._autosave_snapshot_timestamp_path(self._project_paths)
            rolling_path.write_text(json.dumps(snapshot_payload, indent=2), encoding="utf-8")
            self._prune_autosave_snapshots(autosave_dir)
            self._autosave_dirty = False
            self._last_project_saved_text = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
            self._last_project_saved_kind = "Autosaved"
            self._refresh_project_info_banner()
            logger.debug("Autosave snapshot written path=%s", latest_path)
            self._show_autosave_status("Project autosaved")
        except Exception as exc:  # noqa: BLE001
            now = time.perf_counter()
            if now - self._autosave_last_error_ts > 5.0:
                self._autosave_last_error_ts = now
                logger.warning("Autosave failed: %s", exc)
                self._show_autosave_status("Autosave failed (see log)", is_error=True)

    def _prune_autosave_snapshots(self, autosave_dir: Path) -> None:
        snapshots = sorted(autosave_dir.glob("autosave_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for extra in snapshots[self._autosave_roll_limit :]:
            try:
                extra.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                continue

    def _build_project_state_snapshot_payload(self, paths: ProjectPaths, manifest: ProjectManifest) -> Dict[str, Any]:
        entries: List[ProjectSpriteEntry] = []
        metadata_payload: Dict[str, Dict[str, Any]] = {}
        runtime_to_source_key: Dict[str, str] = {}
        for key in self._ordered_sprite_keys():
            record = self.sprite_records.get(key)
            if record is None:
                continue
            source_path = record.path
            if manifest.project_mode == "managed" and not self._is_path_within(source_path, paths.sources_sprites):
                source_path = self._project_service.import_managed_sprite(paths, source_path)
                self._replace_record_path(record.path, source_path)
                record.path = source_path
            if manifest.project_mode == "managed":
                relative_source = source_path.resolve().relative_to(paths.root).as_posix()
            else:
                relative_source = source_path.resolve().as_posix()
            runtime_to_source_key[key] = relative_source
            source_hash = ""
            try:
                source_hash = self._project_service.hash_file(source_path)
            except Exception:  # noqa: BLE001
                source_hash = ""
            entries.append(
                ProjectSpriteEntry(
                    sprite_id=relative_source,
                    source_path=relative_source,
                    load_mode=record.load_mode,
                    source_hash=source_hash,
                )
            )
            metadata_payload[relative_source] = self._sprite_metadata_from_record(record)

        manifest_payload = {
            "schema_version": int(manifest.schema_version),
            "project_name": str(manifest.project_name),
            "project_mode": str(manifest.project_mode),
            "created_at": str(manifest.created_at),
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "source_index": [entry.to_dict() for entry in entries],
            "settings": {
                **dict(manifest.settings),
                "load_mode": self.images_panel.selected_load_mode(),
                "output_dir": self._project_output_dir_setting(paths),
                "group_name_map": self._group_name_map_for_manifest_settings(),
                "animation_tags": self._animation_tags_for_manifest_settings(runtime_to_source_key),
            },
        }
        return {
            "autosave_schema": 1,
            "saved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "project_root": paths.root.as_posix(),
            "manifest": manifest_payload,
            "sprite_metadata": metadata_payload,
        }

    def _group_name_map_for_manifest_settings(self) -> Dict[str, str]:
        payload: Dict[str, str] = {}
        for group in self._palette_groups.values():
            display_name = str(group.display_name or "").strip()
            if not display_name:
                continue
            if display_name == group.group_id:
                continue
            payload[group.signature] = display_name
        return payload

    def _apply_group_display_names_from_manifest_settings(self) -> None:
        manifest = self._project_manifest
        if manifest is None:
            return
        raw_map = manifest.settings.get("group_name_map", {})
        if not isinstance(raw_map, dict):
            return
        changed = 0
        for group in self._palette_groups.values():
            mapped = raw_map.get(group.signature)
            if mapped is None:
                continue
            text = str(mapped).strip()
            if not text:
                continue
            if group.display_name == text:
                continue
            group.display_name = text
            changed += 1
        if changed:
            logger.debug("Applied group display names from manifest count=%s", changed)

    def _load_latest_autosave_payload(self, paths: ProjectPaths) -> Dict[str, Any] | None:
        latest_path = self._autosave_snapshot_path(paths)
        if not latest_path.exists():
            return None
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read autosave payload: %s", exc)
            return None
        return payload if isinstance(payload, dict) else None

    def _parse_iso_utc(self, raw: str) -> datetime | None:
        text = str(raw).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    def _maybe_apply_autosave_recovery(
        self,
        paths: ProjectPaths,
        manifest: ProjectManifest,
        sprite_meta: Dict[str, Dict[str, Any]],
    ) -> tuple[ProjectManifest, Dict[str, Dict[str, Any]]]:
        payload = self._load_latest_autosave_payload(paths)
        if payload is None:
            return manifest, sprite_meta
        manifest_payload = payload.get("manifest")
        autosave_meta = payload.get("sprite_metadata")
        if not isinstance(manifest_payload, dict) or not isinstance(autosave_meta, dict):
            return manifest, sprite_meta

        autosave_dt = self._parse_iso_utc(payload.get("saved_at", ""))
        manifest_dt = self._parse_iso_utc(manifest.updated_at)
        if autosave_dt is None:
            return manifest, sprite_meta
        if manifest_dt is not None and autosave_dt <= manifest_dt:
            return manifest, sprite_meta

        autosave_sources = manifest_payload.get("source_index", []) if isinstance(manifest_payload, dict) else []
        autosave_count = len(autosave_sources) if isinstance(autosave_sources, list) else 0
        current_count = len(manifest.sprites)
        autosave_text = autosave_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        manifest_text = (
            manifest_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
            if manifest_dt is not None
            else "unknown"
        )

        reply = QMessageBox.question(
            self,
            "Autosave Recovery",
            (
                "A newer autosave snapshot was found for this project.\n\n"
                f"Autosave: {autosave_text} (sprites: {autosave_count})\n"
                f"Project file: {manifest_text} (sprites: {current_count})\n\n"
                "Restore autosave state now?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return manifest, sprite_meta

        try:
            recovered_manifest = ProjectManifest.from_dict(manifest_payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Autosave manifest recovery failed: %s", exc)
            return manifest, sprite_meta

        self._recovery_source_text = f"Autosave snapshot ({autosave_text})"

        normalized_meta: Dict[str, Dict[str, Any]] = {}
        for key, value in autosave_meta.items():
            if isinstance(key, str) and isinstance(value, dict):
                normalized_meta[key] = value
        self._autosave_recovery_mode = True
        return recovered_manifest, normalized_meta

    def _sprite_metadata_from_record(self, record: SpriteRecord) -> Dict[str, Any]:
        local_overrides: Dict[str, Dict[str, Any]] = {}
        for idx, payload in record.local_overrides.items():
            color, alpha = payload
            local_overrides[str(int(idx))] = {
                "color": [int(color[0]), int(color[1]), int(color[2])],
                "alpha": int(alpha),
            }
        return {
            "offset_x": int(record.offset_x),
            "offset_y": int(record.offset_y),
            "canvas_width": int(record.canvas_width),
            "canvas_height": int(record.canvas_height),
            "canvas_override_enabled": bool(record.canvas_override_enabled),
            "local_overrides": local_overrides,
        }

    def _apply_project_metadata_to_loaded_records(self) -> None:
        if not self._pending_project_sprite_metadata:
            return
        metadata = self._pending_project_sprite_metadata
        self._pending_project_sprite_metadata = None
        if self._project_paths is None:
            return
        applied = 0
        for key, record in self.sprite_records.items():
            try:
                rel = record.path.resolve().relative_to(self._project_paths.root).as_posix()
            except ValueError:
                rel = record.path.resolve().as_posix()
            payload = metadata.get(rel)
            if not isinstance(payload, dict):
                continue
            record.offset_x = int(payload.get("offset_x", record.offset_x))
            record.offset_y = int(payload.get("offset_y", record.offset_y))
            record.canvas_width = int(payload.get("canvas_width", record.canvas_width))
            record.canvas_height = int(payload.get("canvas_height", record.canvas_height))
            record.canvas_override_enabled = bool(payload.get("canvas_override_enabled", record.canvas_override_enabled))
            local_overrides = payload.get("local_overrides", {})
            parsed_overrides: Dict[int, Tuple[ColorTuple, int]] = {}
            if isinstance(local_overrides, dict):
                for idx_text, value in local_overrides.items():
                    if not isinstance(value, dict):
                        continue
                    try:
                        idx = int(idx_text)
                        color_raw = value.get("color", [0, 0, 0])
                        if not isinstance(color_raw, (list, tuple)) or len(color_raw) < 3:
                            continue
                        color: ColorTuple = (
                            max(0, min(255, int(color_raw[0]))),
                            max(0, min(255, int(color_raw[1]))),
                            max(0, min(255, int(color_raw[2]))),
                        )
                        alpha = max(0, min(255, int(value.get("alpha", 255))))
                        parsed_overrides[idx] = (color, alpha)
                    except (TypeError, ValueError):
                        continue
            record.local_overrides = parsed_overrides
            applied += 1
        if applied:
            self._refresh_palette_for_current_selection()
            self._update_canvas_inputs()
            self._update_sprite_offset_controls(self._current_record())
            self._schedule_preview_update()

    def _on_load_mode_changed(self, _index: int) -> None:
        mode = self.images_panel.selected_load_mode()
        self._save_load_mode_setting()
        logger.info("Load mode changed to %s (applies to future imports only)", mode)
        self.statusBar().showMessage(f"Load mode set to {mode} (next imports)", 2500)

    def _on_sprite_browser_controls_changed(self) -> None:
        current_view_mode = self.images_panel.browser_view_mode()
        current_sort_mode = self.images_panel.browser_sort_mode()
        if current_sort_mode != self._last_browser_sort_mode:
            self._cancel_inplace_icon_refresh()
            start_ts = time.perf_counter()
            reordered = self._reorder_loaded_images_list_for_sort()
            if not reordered:
                self._rebuild_loaded_images_list()
            self._last_browser_sort_mode = current_sort_mode
            self._last_browser_view_mode = current_view_mode
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            reason = "sort-reorder" if reordered else "sort-rebuild-fallback"
            message = "Sprite browser rebuild reason=%s sort=%s view=%s count=%s elapsed_ms=%.2f"
            args = (
                reason,
                current_sort_mode,
                current_view_mode,
                self.images_panel.list_widget.count(),
                elapsed_ms,
            )
            if elapsed_ms >= _BROWSER_REBUILD_WARN_MS:
                logger.warning(message, *args)
            else:
                logger.debug(message, *args)
            return
        reason = "view-change" if current_view_mode != self._last_browser_view_mode else "style-change"
        self._last_browser_view_mode = current_view_mode
        self._start_inplace_icon_refresh(
            reason=reason,
            view_mode=current_view_mode,
            sort_mode=current_sort_mode,
        )

    def _reorder_loaded_images_list_for_sort(self) -> bool:
        list_widget = self.images_panel.list_widget
        item_count = list_widget.count()
        if item_count <= 1:
            self._after_loaded_images_list_changed()
            return True
        if item_count != len(self.sprite_records):
            return False

        items_by_key: Dict[str, QListWidgetItem] = {}
        for row in range(item_count):
            item = list_widget.item(row)
            if item is None:
                return False
            key = item.data(Qt.ItemDataRole.UserRole)
            if not key:
                return False
            key_text = str(key)
            if key_text in items_by_key:
                return False
            items_by_key[key_text] = item

        sorted_keys = self._sorted_sprite_keys_for_browser(list(self.sprite_records.keys()))
        if len(sorted_keys) != item_count:
            return False
        if any(key not in items_by_key for key in sorted_keys):
            return False

        current_item = list_widget.currentItem()
        current_key = (
            str(current_item.data(Qt.ItemDataRole.UserRole))
            if current_item is not None and current_item.data(Qt.ItemDataRole.UserRole)
            else None
        )
        selected_keys: set[str] = set()
        for item in list_widget.selectedItems():
            key = item.data(Qt.ItemDataRole.UserRole)
            if key:
                selected_keys.add(str(key))

        updates_enabled = list_widget.updatesEnabled()
        list_widget.setUpdatesEnabled(False)
        blocked = list_widget.blockSignals(True)
        try:
            while list_widget.count():
                list_widget.takeItem(list_widget.count() - 1)

            for key in sorted_keys:
                list_widget.addItem(items_by_key[key])

            selected_fallback_item: QListWidgetItem | None = None
            for row in range(list_widget.count()):
                item = list_widget.item(row)
                if item is None:
                    continue
                key = item.data(Qt.ItemDataRole.UserRole)
                if key and str(key) in selected_keys:
                    item.setSelected(True)
                if current_key and key and str(key) == current_key:
                    selected_fallback_item = item

            if selected_fallback_item is not None:
                list_widget.setCurrentItem(selected_fallback_item)
            elif list_widget.count() and list_widget.currentItem() is None:
                list_widget.setCurrentRow(0)
        finally:
            list_widget.blockSignals(blocked)
            list_widget.setUpdatesEnabled(updates_enabled)
            list_widget.viewport().update()

        self._after_loaded_images_list_changed()
        return True

    def _start_inplace_icon_refresh(self, *, reason: str, view_mode: str, sort_mode: str) -> None:
        self._cancel_inplace_icon_refresh()
        self._inplace_refresh_active = True
        self._inplace_refresh_next_row = 0
        self._inplace_refresh_total = self.images_panel.list_widget.count()
        self._inplace_refresh_refreshed = 0
        self._inplace_refresh_start_ts = time.perf_counter()
        self._inplace_refresh_reason = reason
        self._inplace_refresh_view_mode = view_mode
        self._inplace_refresh_sort_mode = sort_mode
        self._continue_inplace_icon_refresh()

    def _cancel_inplace_icon_refresh(self) -> None:
        if self._inplace_refresh_timer.isActive():
            self._inplace_refresh_timer.stop()
        self._inplace_refresh_active = False
        self._inplace_refresh_next_row = 0
        self._inplace_refresh_total = 0
        self._inplace_refresh_refreshed = 0

    def _continue_inplace_icon_refresh(self) -> None:
        if not self._inplace_refresh_active:
            return
        list_widget = self.images_panel.list_widget
        total = list_widget.count()
        if total != self._inplace_refresh_total:
            self._inplace_refresh_total = total
            self._inplace_refresh_next_row = 0
            self._inplace_refresh_refreshed = 0
            self._inplace_refresh_start_ts = time.perf_counter()

        if total >= 1500:
            chunk_size = _BROWSER_INPLACE_CHUNK_LARGE
        elif total >= 500:
            chunk_size = _BROWSER_INPLACE_CHUNK_MEDIUM
        else:
            chunk_size = _BROWSER_INPLACE_CHUNK_SMALL

        start_row = self._inplace_refresh_next_row
        end_row = min(total, start_row + chunk_size)
        updates_enabled = list_widget.updatesEnabled()
        list_widget.setUpdatesEnabled(False)
        try:
            for row in range(start_row, end_row):
                item = list_widget.item(row)
                if item is None:
                    continue
                key = item.data(Qt.ItemDataRole.UserRole)
                if not key:
                    continue
                record = self.sprite_records.get(str(key))
                if record is None:
                    continue
                icon_signature = self._sprite_icon_signature(record)
                if item.data(_SPRITE_ICON_SIG_ROLE) != icon_signature:
                    icon, _cache_hit = self._get_cached_sprite_icon(record, icon_signature)
                    item.setIcon(icon)
                    item.setData(_SPRITE_ICON_SIG_ROLE, icon_signature)
                    self._inplace_refresh_refreshed += 1
        finally:
            list_widget.setUpdatesEnabled(updates_enabled)
            list_widget.viewport().update()

        self._inplace_refresh_next_row = end_row
        if self._inplace_refresh_next_row < total:
            self._inplace_refresh_timer.start(0)
            return

        self._refresh_sprite_group_visuals()
        elapsed_ms = (time.perf_counter() - self._inplace_refresh_start_ts) * 1000.0
        refreshed = self._inplace_refresh_refreshed
        unchanged = max(0, total - refreshed)
        hit_rate = (unchanged / total * 100.0) if total else 100.0
        message = (
            "Sprite browser in-place refresh reason=%s view=%s sort=%s refreshed=%s unchanged=%s total=%s hit_rate=%.1f%% elapsed_ms=%.2f"
        )
        args = (
            self._inplace_refresh_reason,
            self._inplace_refresh_view_mode,
            self._inplace_refresh_sort_mode,
            refreshed,
            unchanged,
            total,
            hit_rate,
            elapsed_ms,
        )
        if elapsed_ms >= _BROWSER_INPLACE_WARN_MS:
            logger.warning(message, *args)
        else:
            logger.debug(message, *args)
        self._cancel_inplace_icon_refresh()

    def _sprite_browser_zoom_by(self, direction: int) -> None:
        slider = self.images_panel.zoom_slider
        step = max(1, slider.singleStep())
        slider.setValue(slider.value() + (step * int(direction)))

    def _sprite_browser_zoom_reset(self) -> None:
        self.images_panel.zoom_slider.setValue(64)

    def _sorted_sprite_keys_for_browser(self, keys: Sequence[str]) -> List[str]:
        sort_mode = self.images_panel.browser_sort_mode()
        if sort_mode == "name":
            return sorted(keys, key=lambda key: (Path(key).name.lower(), key.lower()))
        if sort_mode == "path":
            return sorted(keys, key=lambda key: key.lower())
        if sort_mode == "group":
            return sorted(
                keys,
                key=lambda key: (
                    1 if not (self.sprite_records.get(key).group_id if self.sprite_records.get(key) else None) else 0,
                    str((self.sprite_records.get(key).group_id if self.sprite_records.get(key) else "") or "").lower(),
                    Path(key).name.lower(),
                    key.lower(),
                ),
            )
        return list(keys)

    def _build_loaded_sprite_icon(self, record: SpriteRecord) -> QIcon:
        icon_size = self.images_panel.list_widget.iconSize()
        marker_mode = self.images_panel.browser_group_marker_mode()
        thumbnail_mode = self.images_panel.browser_view_mode() == "thumbnails"
        marker_thickness = self.images_panel.browser_group_square_thickness()
        marker_padding = self.images_panel.browser_group_square_padding()
        marker_fill_alpha = self.images_panel.browser_group_square_fill_alpha()
        icon_width = max(16, int(icon_size.width()))
        icon_height = max(16, int(icon_size.height()))
        inner_width = max(8, icon_width - (2 * max(0, marker_padding + marker_thickness)))
        inner_height = max(8, icon_height - (2 * max(0, marker_padding + marker_thickness)))
        scaled_pixmap = record.pixmap.scaled(
            QSize(inner_width, inner_height),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        icon_canvas = QPixmap(icon_width, icon_height)
        icon_canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(icon_canvas)
        marker_rect = None
        marker_color = None
        if thumbnail_mode and marker_mode in ("square", "both") and record.group_id:
            group = self._palette_groups.get(record.group_id)
            if group is not None:
                marker_color = QColor(group.color[0], group.color[1], group.color[2])
                inset = max(0, marker_padding)
                rect_w = max(1, icon_width - (2 * inset))
                rect_h = max(1, icon_height - (2 * inset))
                marker_rect = QRect(inset, inset, rect_w, rect_h)
                fill_color = QColor(marker_color)
                fill_color.setAlpha(marker_fill_alpha)
                painter.fillRect(marker_rect, fill_color)
        painter.drawPixmap(
            (icon_width - scaled_pixmap.width()) // 2,
            (icon_height - scaled_pixmap.height()) // 2,
            scaled_pixmap,
        )
        if marker_rect is not None and marker_color is not None:
            painter.setPen(QPen(marker_color, marker_thickness))
            border_rect = QRect(
                marker_rect.left(),
                marker_rect.top(),
                max(1, marker_rect.width() - 1),
                max(1, marker_rect.height() - 1),
            )
            painter.drawRect(border_rect)
        painter.end()
        return QIcon(icon_canvas)

    def _sprite_icon_signature(self, record: SpriteRecord) -> tuple[Any, ...]:
        icon_size = self.images_panel.list_widget.iconSize()
        marker_mode = self.images_panel.browser_group_marker_mode()
        marker_thickness = self.images_panel.browser_group_square_thickness()
        marker_padding = self.images_panel.browser_group_square_padding()
        marker_fill_alpha = self.images_panel.browser_group_square_fill_alpha()
        group_color: tuple[int, int, int] | None = None
        if record.group_id:
            group = self._palette_groups.get(record.group_id)
            if group is not None:
                group_color = (int(group.color[0]), int(group.color[1]), int(group.color[2]))
        return (
            self.images_panel.browser_view_mode(),
            int(icon_size.width()),
            int(icon_size.height()),
            marker_mode,
            int(marker_thickness),
            int(marker_padding),
            int(marker_fill_alpha),
            str(record.group_id or ""),
            group_color,
            int(record.pixmap.cacheKey()),
        )

    def _get_cached_sprite_icon(self, record: SpriteRecord, icon_signature: tuple[Any, ...]) -> tuple[QIcon, bool]:
        cached_icon = self._sprite_icon_cache.get(icon_signature)
        if cached_icon is not None:
            self._sprite_icon_cache.move_to_end(icon_signature)
            return cached_icon, True
        generated_icon = self._build_loaded_sprite_icon(record)
        self._sprite_icon_cache[icon_signature] = generated_icon
        if len(self._sprite_icon_cache) > _SPRITE_ICON_CACHE_LIMIT:
            self._sprite_icon_cache.popitem(last=False)
        return generated_icon, False

    def _create_loaded_sprite_item(self, key: str, record: SpriteRecord) -> tuple[QListWidgetItem, bool]:
        path = record.path
        item = QListWidgetItem(path.name)
        icon_signature = self._sprite_icon_signature(record)
        icon, cache_hit = self._get_cached_sprite_icon(record, icon_signature)
        item.setIcon(icon)
        item.setData(_SPRITE_ICON_SIG_ROLE, icon_signature)
        item.setData(Qt.ItemDataRole.UserRole, key)
        item.setData(_SPRITE_BASE_NAME_ROLE, path.name)
        item.setToolTip(path.as_posix())
        return item, cache_hit

    def _rebuild_loaded_images_list(self) -> None:
        list_widget = self.images_panel.list_widget
        cache_hits = 0
        cache_misses = 0
        current_item = list_widget.currentItem()
        current_key = (
            str(current_item.data(Qt.ItemDataRole.UserRole))
            if current_item is not None and current_item.data(Qt.ItemDataRole.UserRole)
            else None
        )
        selected_keys: set[str] = set()
        for item in list_widget.selectedItems():
            key = item.data(Qt.ItemDataRole.UserRole)
            if key:
                selected_keys.add(str(key))

        updates_enabled = list_widget.updatesEnabled()
        list_widget.setUpdatesEnabled(False)
        blocked = list_widget.blockSignals(True)
        try:
            list_widget.clear()
            for key in self._sorted_sprite_keys_for_browser(list(self.sprite_records.keys())):
                record = self.sprite_records.get(key)
                if record is None:
                    continue
                item, cache_hit = self._create_loaded_sprite_item(key, record)
                list_widget.addItem(item)
                if cache_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1

            selected_fallback_item: QListWidgetItem | None = None
            for row in range(list_widget.count()):
                item = list_widget.item(row)
                if item is None:
                    continue
                key = item.data(Qt.ItemDataRole.UserRole)
                if key and str(key) in selected_keys:
                    item.setSelected(True)
                if current_key and key and str(key) == current_key:
                    selected_fallback_item = item

            if selected_fallback_item is not None:
                list_widget.setCurrentItem(selected_fallback_item)
            elif list_widget.count() and list_widget.currentItem() is None:
                list_widget.setCurrentRow(0)
        finally:
            list_widget.blockSignals(blocked)
            list_widget.setUpdatesEnabled(updates_enabled)
            list_widget.viewport().update()

        self._after_loaded_images_list_changed()
        logger.debug(
            "Sprite list rebuild icon-cache hits=%s misses=%s size=%s",
            cache_hits,
            cache_misses,
            len(self._sprite_icon_cache),
        )

    def _after_loaded_images_list_changed(self) -> None:
        self.images_panel.apply_browser_search_filter()
        self._refresh_sprite_group_visuals()
        self._update_loaded_count()
        self._update_export_buttons()
        self._refresh_palette_for_current_selection()

    def _load_images(self, paths: Sequence[Path]) -> None:
        prepared_paths = self._prepare_paths_for_load(paths)
        queue_candidates = [path for path in prepared_paths if path.as_posix() not in self.sprite_records]
        if not queue_candidates:
            self.statusBar().showMessage("No new images loaded", 3000)
            return

        if self._load_timer.isActive():
            for path in queue_candidates:
                self._load_queue.append(path)
            self._load_total_count += len(queue_candidates)
            self.statusBar().showMessage(
                f"Queued {len(queue_candidates)} more image(s) ({self._load_added_count}/{self._load_total_count} loaded)",
                3000,
            )
            logger.debug("Extended active load queue added=%s total=%s", len(queue_candidates), self._load_total_count)
            return

        list_widget = self.images_panel.list_widget
        self._load_had_selection = list_widget.currentRow() >= 0
        self._load_starting_count = list_widget.count()
        current_item = list_widget.currentItem()
        self._load_selected_key_before = (
            str(current_item.data(Qt.ItemDataRole.UserRole))
            if current_item is not None and current_item.data(Qt.ItemDataRole.UserRole)
            else None
        )
        self._load_mode_in_progress = self.images_panel.selected_load_mode()
        self._load_added_count = 0
        self._load_failed_count = 0
        self._load_total_count = len(queue_candidates)
        self._load_queue.clear()
        for path in queue_candidates:
            self._load_queue.append(path)

        logger.debug("Load images queued mode=%s count=%s", self._load_mode_in_progress, self._load_total_count)
        self._set_loading_state(True)
        self.statusBar().showMessage(
            f"Loading {self._load_total_count} image(s) in background-safe batches...",
            3000,
        )
        self._load_timer.start()

    def _prepare_paths_for_load(self, paths: Sequence[Path]) -> List[Path]:
        normalized: List[Path] = []
        seen: set[str] = set()
        managed_project_active = (
            self._project_paths is not None
            and self._project_manifest is not None
            and self._project_manifest.project_mode == "managed"
        )
        copied_count = 0

        for raw_path in paths:
            path = Path(raw_path)
            candidate = path
            if managed_project_active and self._project_paths is not None:
                if not self._is_path_within(path, self._project_paths.sources_sprites):
                    try:
                        imported = self._project_service.import_managed_sprite(self._project_paths, path)
                        candidate = imported
                        copied_count += 1
                    except Exception as exc:  # noqa: BLE001
                        QMessageBox.warning(
                            self,
                            "Import Sprite",
                            f"Failed to copy {path.name} into project sources:\n{exc}",
                        )
                        continue
            key = candidate.as_posix()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(candidate)

        if copied_count:
            self.statusBar().showMessage(
                f"Copied {copied_count} sprite(s) into project sources",
                3500,
            )
        return normalized

    def _process_load_queue_tick(self) -> None:
        if not self._load_queue:
            self._load_timer.stop()
            self._finalize_load_batch()
            return

        load_mode = self._load_mode_in_progress or self.images_panel.selected_load_mode()
        chunk_size = 6 if load_mode == "detect" else 16
        list_widget = self.images_panel.list_widget
        processed_this_tick = 0
        blocked = list_widget.blockSignals(True)

        try:
            while processed_this_tick < chunk_size and self._load_queue:
                path = self._load_queue.popleft()
                processed_this_tick += 1
                item_mode = self._load_mode_overrides.pop(path.as_posix(), None)
                effective_mode = item_mode or load_mode
                record = self._create_record(path, load_mode=effective_mode)
                if record is None:
                    self._load_failed_count += 1
                    continue
                key = path.as_posix()
                self.sprite_records[key] = record
                self._assign_record_group(key, record)
                self._merge_record_palette(record)
                item, _cache_hit = self._create_loaded_sprite_item(key, record)
                list_widget.addItem(item)
                self._load_added_count += 1
        finally:
            list_widget.blockSignals(blocked)

        done = self._load_total_count - len(self._load_queue)
        self.statusBar().showMessage(
            f"Loading sprites... {done}/{self._load_total_count} (added {self._load_added_count})",
            1000,
        )

    def _finalize_load_batch(self) -> None:
        list_widget = self.images_panel.list_widget
        added = self._load_added_count
        failed = self._load_failed_count

        if added:
            self._invalidate_used_index_cache("load-images")
            self._rebuild_loaded_images_list()

        if added and self._pending_project_sprite_metadata:
            self._apply_project_metadata_to_loaded_records()

        if added:
            self._apply_group_display_names_from_manifest_settings()

        self._update_export_buttons()
        self._update_loaded_count()
        self._refresh_group_overview()

        if added:
            message = f"Loaded {added} image(s)"
            if failed:
                message += f" ({failed} failed)"
            self.statusBar().showMessage(message, 5000)
            self._reset_history()
            self._mark_project_dirty("load-images")
        else:
            self.statusBar().showMessage("No new images loaded", 3000)

        logger.debug(
            "Load batch complete total=%s added=%s failed=%s mode=%s",
            self._load_total_count,
            added,
            failed,
            self._load_mode_in_progress,
        )

        self._load_mode_in_progress = None
        self._load_mode_overrides.clear()
        self._load_total_count = 0
        self._load_added_count = 0
        self._load_failed_count = 0
        self._load_selected_key_before = None
        self._set_loading_state(False)

    def _set_loading_state(self, loading: bool) -> None:
        if self._is_loading_sprites == loading:
            return
        self._is_loading_sprites = loading
        self.images_panel.list_widget.setEnabled(not loading)
        self.images_panel.clear_button.setEnabled(not loading)
        self.images_panel.display_mode_combo.setEnabled(not loading)
        self.images_panel.sort_mode_combo.setEnabled(not loading)
        self.images_panel.zoom_slider.setEnabled(not loading)
        self._update_project_action_buttons()
        self.palette_list.setEnabled(not loading)
        self.merge_button.setEnabled(not loading)
        self.set_local_override_btn.setEnabled(not loading)
        self.clear_local_override_btn.setEnabled(not loading)
        self.shift_steps_spin.setEnabled(not loading)
        self.shift_left_button.setEnabled(not loading)
        self.shift_right_button.setEnabled(not loading)
        if loading:
            self.preview_panel.set_drag_mode(False)
            self._active_offset_drag_mode = "none"
        self._update_sprite_offset_controls(self._current_record())
        self._update_export_buttons()
        logger.debug("Loading state updated loading=%s", loading)

    def _clear_detect_batch_jobs(self, reason: str) -> None:
        if self._detect_batch_timer.isActive():
            self._detect_batch_timer.stop()
        if self._detect_batch_queue:
            logger.debug(
                "Cleared detect batch jobs reason=%s pending=%s done=%s total=%s",
                reason,
                len(self._detect_batch_queue),
                self._detect_batch_done,
                self._detect_batch_total,
            )
        self._detect_batch_queue.clear()
        self._detect_batch_total = 0
        self._detect_batch_done = 0

    def _enqueue_detect_slot_batch_updates(
        self,
        records: Sequence[SpriteRecord],
        changed_slots: Sequence[int],
        new_color_by_slot: Dict[int, ColorTuple],
        new_alpha_by_slot: Dict[int, int],
        *,
        reason: str,
        anchor_key: str,
    ) -> None:
        batch_slots = [int(slot) for slot in changed_slots]
        batch_colors = dict(new_color_by_slot)
        batch_alphas = {int(slot): int(alpha) for slot, alpha in new_alpha_by_slot.items()}
        for record in records:
            key = record.path.as_posix()
            if key == anchor_key:
                continue
            self._detect_batch_queue.append(
                (
                    "slot",
                    key,
                    {
                        "slots": batch_slots,
                        "colors": batch_colors,
                        "alphas": batch_alphas,
                    },
                )
            )
        queued = len(self._detect_batch_queue)
        if queued == 0:
            return
        self._detect_batch_total = queued
        self._detect_batch_done = 0
        self._detect_batch_timer.start()
        self.statusBar().showMessage(f"Applying detect batch updates... 0/{self._detect_batch_total}", 1000)
        logger.debug("Queued detect slot batch reason=%s jobs=%s", reason, queued)

    def _enqueue_detect_remap_batch_updates(
        self,
        records: Sequence[SpriteRecord],
        pixel_remap: Dict[int, int],
        *,
        reason: str,
        anchor_key: str,
    ) -> None:
        batch_remap = {int(src): int(dst) for src, dst in pixel_remap.items()}
        for record in records:
            key = record.path.as_posix()
            if key == anchor_key:
                continue
            self._detect_batch_queue.append(("remap", key, {"remap": batch_remap}))
        queued = len(self._detect_batch_queue)
        if queued == 0:
            return
        self._detect_batch_total = queued
        self._detect_batch_done = 0
        self._detect_batch_timer.start()
        self.statusBar().showMessage(f"Applying detect batch updates... 0/{self._detect_batch_total}", 1000)
        logger.debug("Queued detect remap batch reason=%s jobs=%s", reason, queued)

    def _process_detect_batch_tick(self) -> None:
        if not self._detect_batch_queue:
            self._detect_batch_timer.stop()
            self.statusBar().showMessage("Detect batch updates complete", 1200)
            return

        jobs_this_tick = 6
        while jobs_this_tick > 0 and self._detect_batch_queue:
            job_type, path_key, payload = self._detect_batch_queue.popleft()
            record = self.sprite_records.get(path_key)
            if record is None:
                self._detect_batch_done += 1
                jobs_this_tick -= 1
                continue
            if job_type == "slot":
                self._apply_detect_slot_deltas_to_record(
                    record,
                    payload["slots"],
                    payload["colors"],
                    payload["alphas"],
                )
            elif job_type == "remap":
                self._apply_detect_remap_to_record(record, payload["remap"])
            self._detect_batch_done += 1
            jobs_this_tick -= 1

        self.statusBar().showMessage(
            f"Applying detect batch updates... {self._detect_batch_done}/{self._detect_batch_total}",
            1000,
        )

    def _merge_record_palette(self, record: SpriteRecord) -> None:
        """Merge a sprite's palette into the unified palette and remap sprite to use unified palette."""
        preserve_indexed = record.load_mode == "preserve"
        record.slot_bindings = {}
        palette = record.palette
        if not palette:
            return
        
        if preserve_indexed:
            for idx in range(len(palette)):
                record.slot_bindings[idx] = idx
            logger.debug(
                "Preserve mode load record=%s palette_len=%s transparency=%s",
                record.path.name,
                len(palette),
                record.indexed_image.info.get("transparency"),
            )
            return

        detect_group = self._palette_groups.get(record.group_id) if record.group_id else None
        if detect_group is not None and detect_group.mode == "detect":
            self._load_detect_group_palette_state(detect_group, "merge-record-start")
        else:
            self.palette_colors = []
            self.palette_alphas = []
            self._palette_slot_ids = []
            self._slot_color_lookup = {}
        
        if not self.palette_colors:
            self._initialize_palette_from_record(record)
            if detect_group is not None and detect_group.mode == "detect":
                self._store_detect_group_palette_state(detect_group, "initialize-group")
            return
        
        start_len = len(self.palette_colors)
        usage = defaultdict(int)
        source_index_to_slot_id: Dict[int, int] = {}
        
        # First pass: ensure all sprite colors exist in unified palette
        for idx, color in enumerate(palette):
            slot_id = self._claim_slot_for_color(color, usage)
            source_index_to_slot_id[idx] = slot_id
            usage[color] += 1
        
        # Sync to update _palette_slot_ids with any new colors
        if len(self.palette_colors) != start_len:
            self._sync_palette_model()
        
        # Now convert sprite to use unified palette directly.
        # Build deterministic remapping from source palette index -> unified palette index.
        slot_to_unified_index = {slot_id: idx for idx, slot_id in enumerate(self._palette_slot_ids)}
        remap: Dict[int, int] = {}
        for source_idx, slot_id in source_index_to_slot_id.items():
            remap[source_idx] = slot_to_unified_index.get(slot_id, 0)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"=== Converting {record.path.name} to unified palette")
            logger.debug(f"  Source palette entries: {len(palette)}")
            logger.debug(f"  Unified palette: {self.palette_colors[:8]}")
            logger.debug(f"  source->slot sample: {dict(list(source_index_to_slot_id.items())[:8])}")
            logger.debug(f"  slot->unified sample: {dict(list(slot_to_unified_index.items())[:8])}")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  Remap: {dict(list(remap.items())[:8])}")
        
        # Apply remapping to pixels
        pixels = list(record.indexed_image.getdata())
        new_pixels = [remap.get(p, 0) for p in pixels]
        
        # Create new image with unified palette
        new_image = Image.new("P", record.indexed_image.size)
        new_image.putdata(new_pixels)
        
        # Apply unified palette
        flat_palette = []
        for color in self.palette_colors:
            flat_palette.extend(color)
        while len(flat_palette) < 768:
            flat_palette.append(0)
        new_image.putpalette(flat_palette[:768])
        
        record.indexed_image = new_image
        
        # Set slot bindings - sprite now uses unified palette directly
        # Each slot_id maps to the unified palette index where that color now lives
        # After remapping, sprite uses unified indices directly, so it's identity for all slots
        record.slot_bindings = {}
        for idx, slot_id in enumerate(self._palette_slot_ids):
            record.slot_bindings[slot_id] = idx
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  Final slot_bindings (first 8): {dict(list(record.slot_bindings.items())[:8])}")
            logger.debug(f"  _palette_slot_ids (first 8): {self._palette_slot_ids[:8]}")
            # Show what unified indices the sprite is actually using
            unique_pixels = set(new_pixels)
            logger.debug(f"  Sprite now uses unified indices: {sorted(unique_pixels)[:16]}")
            logger.debug(f"  Conversion complete")
        if detect_group is not None and detect_group.mode == "detect":
            self._store_detect_group_palette_state(detect_group, "merge-record-complete")

    def _initialize_palette_from_record(self, record: SpriteRecord) -> None:
        limited = list(record.palette[:256])
        slot_ids = list(range(len(limited)))
        record.slot_bindings = {slot_id: idx for slot_id, idx in zip(slot_ids, range(len(slot_ids)))}
        self.palette_colors = limited
        self.palette_alphas = [255] * len(limited)
        self._palette_slot_ids = slot_ids
        self._next_slot_id = len(slot_ids)
        self._sync_palette_model()

    def _claim_slot_for_color(self, color: ColorTuple, usage: DefaultDict[ColorTuple, int]) -> int:
        slots = self._slot_color_lookup.get(color)
        count = usage[color]
        if slots:
            return slots[count] if count < len(slots) else slots[-1]
        fallback_slot = self._find_existing_slot_for_color(color)
        if fallback_slot is not None:
            logger.debug("Palette lookup repaired color=%s slot=%s", color, fallback_slot)
            self._slot_color_lookup[color] = [fallback_slot]
            return fallback_slot
        if len(self.palette_colors) < 256:
            slot_id = self._append_palette_slot(color)
            self._slot_color_lookup[color] = [slot_id]
            return slot_id
        slot_id = self._palette_slot_ids[-1] if self._palette_slot_ids else 0
        logger.warning(
            "Palette full; reusing slot_id=%s for color=%s duplicate=%s",
            slot_id,
            color,
            count + 1,
        )
        return slot_id

    def _find_existing_slot_for_color(self, color: ColorTuple) -> int | None:
        if not self.palette_colors:
            return None
        for idx, existing in enumerate(self.palette_colors):
            if existing != color:
                continue
            if idx < len(self._palette_slot_ids):
                return self._palette_slot_ids[idx]
            return idx
        return None


    def _append_palette_slot(self, color: ColorTuple) -> int:
        if len(self.palette_colors) >= 256:
            logger.warning("Palette full; cannot add new slot for color %s", color)
            return self._palette_slot_ids[-1] if self._palette_slot_ids else 0
        slot_id = self._next_slot_id
        self._next_slot_id += 1
        self.palette_colors.append(color)
        self.palette_alphas.append(255)
        self._palette_slot_ids.append(slot_id)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Appended palette slot slot_id=%s color=%s total_colors=%s",
                slot_id,
                color,
                len(self.palette_colors),
            )
        return slot_id

    def _sync_palette_model(self) -> None:
        if not self.palette_colors:
            self.palette_alphas = []
            self._palette_slot_ids = []
            self._slot_color_lookup = {}
            self.palette_list.set_colors([], emit_signal=False)
            self._refresh_main_palette_usage_badges()
            self._update_fill_preview()
            return
        if len(self._palette_slot_ids) != len(self.palette_colors):
            self._palette_slot_ids = list(range(len(self.palette_colors)))
        if len(self.palette_alphas) != len(self.palette_colors):
            self.palette_alphas = [255] * len(self.palette_colors)

        # Model will display 256 slots with colors at their slot positions
        self.palette_list.set_colors(
            self.palette_colors,
            slots=self._palette_slot_ids,
            alphas=self.palette_alphas,
            emit_signal=False,
        )
        self._refresh_main_palette_usage_badges()
        self._rebuild_slot_color_lookup()
        self._log_palette_stats("sync")
        self._update_fill_preview()

    def _snapshot_detect_palette(self, reason: str) -> None:
        self._detect_palette_colors = list(self.palette_colors)
        self._detect_palette_alphas = list(self.palette_alphas)
        self._detect_palette_slot_ids = list(self._palette_slot_ids)
        logger.debug(
            "Snapshot detect palette reason=%s colors=%s",
            reason,
            len(self._detect_palette_colors),
        )

    def _restore_detect_palette(self, reason: str) -> None:
        self.palette_colors = list(self._detect_palette_colors)
        self.palette_alphas = list(self._detect_palette_alphas)
        self._palette_slot_ids = list(self._detect_palette_slot_ids)
        logger.debug(
            "Restore detect palette reason=%s colors=%s",
            reason,
            len(self.palette_colors),
        )
        self._sync_palette_model()

    def _refresh_palette_for_current_selection(self) -> None:
        record = self._current_record()
        if record is None:
            self.palette_list.set_colors([], emit_signal=False)
            self.palette_list.set_local_rows([])
            return
        if record.load_mode == "preserve":
            self._sync_palette_from_current_record_preserve()
            self._sync_local_override_visuals()
            return
        self._sync_palette_from_current_record_detect_group()
        self._sync_local_override_visuals()

    def _capture_palette_selection_rows(self) -> Tuple[List[int], int | None]:
        selected_rows = sorted({index.row() for index in self.palette_list.selectedIndexes() if 0 <= index.row() < 256})
        current_index = self.palette_list.currentIndex()
        current_row = current_index.row() if current_index.isValid() else None
        return selected_rows, current_row

    def _restore_palette_selection_rows(self, rows: Sequence[int], current_row: int | None) -> None:
        selection = self.palette_list.selectionModel()
        if selection is None:
            return
        selection.clearSelection()
        for row in rows:
            if 0 <= row < 256:
                idx = self.palette_list.model_obj.index(row)
                selection.select(idx, QItemSelectionModel.SelectionFlag.Select)
        if current_row is not None and 0 <= current_row < 256:
            self.palette_list.setCurrentIndex(self.palette_list.model_obj.index(current_row))
        logger.debug(
            "Restored palette selection rows=%s current=%s",
            list(rows)[:12],
            current_row,
        )

    def _sync_local_override_visuals(self) -> None:
        record = self._current_record()
        if record is None:
            self.palette_list.set_local_rows([])
            return
        local_rows = sorted(record.local_overrides.keys())
        self.palette_list.set_local_rows(local_rows)

    def _store_detect_group_palette_state(self, group: PaletteGroup, reason: str) -> None:
        if group.mode != "detect":
            return
        group.detect_palette_colors = list(self.palette_colors)
        group.detect_palette_alphas = list(self.palette_alphas)
        group.detect_palette_slot_ids = list(self._palette_slot_ids)
        logger.debug(
            "Stored detect group palette group=%s reason=%s colors=%s",
            group.group_id,
            reason,
            len(group.detect_palette_colors),
        )

    def _load_detect_group_palette_state(self, group: PaletteGroup, reason: str) -> None:
        if group.mode != "detect":
            return
        self.palette_colors = list(group.detect_palette_colors)
        self.palette_alphas = list(group.detect_palette_alphas)
        self._palette_slot_ids = list(group.detect_palette_slot_ids)
        self._rebuild_slot_color_lookup()
        logger.debug(
            "Loaded detect group palette group=%s reason=%s colors=%s",
            group.group_id,
            reason,
            len(self.palette_colors),
        )

    def _sync_palette_from_current_record_detect_group(self) -> None:
        record = self._current_record()
        if record is None or record.load_mode != "detect":
            self.palette_list.set_colors([], emit_signal=False)
            self._update_fill_preview()
            return
        group = self._palette_groups.get(record.group_id) if record.group_id else None
        if (
            group is not None
            and group.mode == "detect"
            and group.detect_palette_colors
            and len(group.detect_palette_slot_ids) == len(group.detect_palette_colors)
        ):
            self._load_detect_group_palette_state(group, "sync-current-selection")
            if len(self.palette_alphas) < len(self.palette_colors):
                self.palette_alphas.extend([255] * (len(self.palette_colors) - len(self.palette_alphas)))
            elif len(self.palette_alphas) > len(self.palette_colors):
                self.palette_alphas = self.palette_alphas[: len(self.palette_colors)]
            logger.debug(
                "Sync detect palette from cached group state record=%s group=%s colors=%s",
                record.path.name,
                record.group_id,
                len(self.palette_colors),
            )
            self._sync_palette_model()
            return
        palette_info = extract_palette(record.indexed_image, include_unused=False)
        self.palette_colors = list(palette_info.colors[:256])
        self.palette_alphas = self._extract_palette_alphas_from_image(record.indexed_image, len(self.palette_colors))
        self._palette_slot_ids = list(range(len(self.palette_colors)))
        logger.debug(
            "Sync detect palette from record=%s group=%s colors=%s raw_palette_entries=%s",
            record.path.name,
            record.group_id,
            len(self.palette_colors),
            (len(record.indexed_image.getpalette() or []) // 3),
        )
        self._sync_palette_model()
        if group is not None and group.mode == "detect":
            self._store_detect_group_palette_state(group, "sync-current-selection-fallback")

    def _record_group_key(self, record: SpriteRecord) -> str:
        if record.load_mode == "detect":
            signature = "|".join(f"{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in record.palette[:256])
            digest = hashlib.sha1(signature.encode("utf-8", errors="ignore")).hexdigest()[:16]
            logger.debug(
                "Detect group signature record=%s colors=%s digest=%s",
                record.path.name,
                len(record.palette),
                digest,
            )
            return f"detect:v2:{digest}:len:{len(record.palette)}"
        palette_info = extract_palette(record.indexed_image, include_unused=True)
        colors = list(palette_info.colors[:256])
        alphas = self._extract_palette_alphas_from_image(record.indexed_image, min(256, len(colors)))
        used_indices = sorted(set(int(px) for px in record.indexed_image.getdata()))
        signature_parts: List[str] = []
        for idx in range(len(colors)):
            color = colors[idx]
            alpha = alphas[idx] if idx < len(alphas) else 255
            signature_parts.append(f"{idx}:{color[0]:02x}{color[1]:02x}{color[2]:02x}{alpha:02x}")
        signature_blob = "|".join(signature_parts).encode("utf-8", errors="ignore")
        digest = hashlib.sha1(signature_blob).hexdigest()[:16]
        logger.debug(
            "Preserve group signature record=%s used=%s entries=%s digest=%s sample=%s",
            record.path.name,
            len(used_indices),
            len(colors),
            digest,
            [signature_parts[idx] for idx in used_indices[:6] if idx < len(signature_parts)],
        )
        return f"preserve:v3:{digest}:entries:{len(colors)}"

    def _assign_record_group(self, key: str, record: SpriteRecord) -> None:
        group_key = self._record_group_key(record)
        group_id = self._group_key_to_id.get(group_key)
        if group_id is None:
            group = self._create_palette_group(mode=record.load_mode, signature=group_key)
            group_id = group.group_id
            self._palette_groups[group_id] = group
            self._group_key_to_id[group_key] = group_id
            logger.debug("Created palette group id=%s mode=%s signature=%s", group_id, record.load_mode, group_key)
        record.group_id = group_id
        self._palette_groups[group_id].member_keys.add(key)
        logger.debug(
            "Assigned sprite to group path=%s group=%s members=%s",
            record.path.name,
            group_id,
            len(self._palette_groups[group_id].member_keys),
        )

    def _pick_group_color(self, seed: int) -> ColorTuple:
        return _GROUP_COLOR_PRESETS[seed % len(_GROUP_COLOR_PRESETS)]

    def _default_group_display_name(self, mode: Literal["detect", "preserve"], ordinal: int) -> str:
        prefix = "Detected" if mode == "detect" else "Preserved"
        return f"{prefix} Group {ordinal:02d}"

    def _group_display_name(self, group: PaletteGroup) -> str:
        text = str(group.display_name or "").strip()
        return text if text else group.group_id

    def _create_palette_group(self, *, mode: Literal["detect", "preserve"], signature: str) -> PaletteGroup:
        ordinal = self._next_group_id
        group_id = f"G{ordinal:04d}"
        color = self._pick_group_color(ordinal)
        self._next_group_id += 1
        width = self.canvas_width_spin.value() if hasattr(self, "canvas_width_spin") else 304
        height = self.canvas_height_spin.value() if hasattr(self, "canvas_height_spin") else 224
        return PaletteGroup(
            group_id=group_id,
            mode=mode,
            signature=signature,
            display_name=self._default_group_display_name(mode, ordinal),
            color=color,
            canvas_width=width,
            canvas_height=height,
        )

    def _refresh_group_overview(self) -> None:
        group_list = self.images_panel.group_list
        selected = group_list.currentItem().data(_GROUP_ID_ROLE) if group_list.currentItem() else None
        group_list.blockSignals(True)
        group_list.clear()
        for group_id in sorted(self._palette_groups.keys()):
            group = self._palette_groups[group_id]
            display_name = self._group_display_name(group)
            label = f"{display_name} [{group.mode}] ({len(group.member_keys)})"
            item = QListWidgetItem(label)
            item.setData(_GROUP_ID_ROLE, group.group_id)
            item.setBackground(QColor(group.color[0], group.color[1], group.color[2], 96))
            item.setToolTip(
                f"Name: {display_name}\nID: {group.group_id}\nMembers: {len(group.member_keys)}\nSignature: {group.signature}"
            )
            group_list.addItem(item)
            if selected == group.group_id:
                group_list.setCurrentItem(item)
        group_list.blockSignals(False)
        if self._hover_group_id and self._hover_group_id not in self._palette_groups:
            self._hover_group_id = None
        needs_rebuild_for_markers = (
            self.images_panel.browser_view_mode() == "thumbnails"
            and self.images_panel.browser_group_marker_mode() in ("square", "both")
        )
        if self.images_panel.browser_sort_mode() == "group" or needs_rebuild_for_markers:
            self._rebuild_loaded_images_list()
        else:
            self._refresh_sprite_group_visuals()
        self._update_sprite_offset_controls(self._current_record())
        self._update_canvas_inputs()
        logger.debug("Refreshed group overview groups=%s", group_list.count())

    def _refresh_sprite_group_visuals(self) -> None:
        list_widget = self.images_panel.list_widget
        thumbnail_mode = self.images_panel.browser_view_mode() == "thumbnails"
        marker_mode = self.images_panel.browser_group_marker_mode()
        hovered_group_id = self._hover_group_id
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item is None:
                continue
            key = item.data(Qt.ItemDataRole.UserRole)
            base_name = item.data(_SPRITE_BASE_NAME_ROLE) or item.text()
            record = self.sprite_records.get(key)
            in_hover_group = bool(
                hovered_group_id
                and record is not None
                and record.group_id
                and record.group_id == hovered_group_id
            )
            out_of_hover_group = bool(hovered_group_id and not in_hover_group)
            if record is None or not record.group_id:
                item.setText(str(base_name))
                if hovered_group_id:
                    item.setBackground(QColor(0, 0, 0, 55))
                else:
                    item.setBackground(QColor(0, 0, 0, 0))
                item.setToolTip(str(base_name))
                item.setForeground(QColor(150, 150, 150) if out_of_hover_group else QColor(230, 230, 230))
                if thumbnail_mode:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
                continue
            group = self._palette_groups.get(record.group_id)
            if group is None:
                item.setText(str(base_name))
                if hovered_group_id:
                    item.setBackground(QColor(0, 0, 0, 55))
                else:
                    item.setBackground(QColor(0, 0, 0, 0))
                item.setToolTip(str(base_name))
                item.setForeground(QColor(150, 150, 150) if out_of_hover_group else QColor(230, 230, 230))
                if thumbnail_mode:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
                continue
            group_name = self._group_display_name(group)
            if thumbnail_mode:
                item.setText(str(base_name))
                if in_hover_group:
                    item.setBackground(QColor(group.color[0], group.color[1], group.color[2], 75))
                elif hovered_group_id:
                    item.setBackground(QColor(0, 0, 0, 55))
                else:
                    item.setBackground(QColor(0, 0, 0, 0))
                item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
                if out_of_hover_group:
                    item.setForeground(QColor(145, 145, 145))
                else:
                    if marker_mode in ("text", "both"):
                        item.setForeground(QColor(group.color[0], group.color[1], group.color[2]))
                    else:
                        item.setForeground(QColor(230, 230, 230))
            else:
                item.setText(f"[{group_name}] {base_name}")
                if in_hover_group:
                    item.setBackground(QColor(group.color[0], group.color[1], group.color[2], 140))
                elif hovered_group_id:
                    item.setBackground(QColor(group.color[0], group.color[1], group.color[2], 30))
                else:
                    item.setBackground(QColor(group.color[0], group.color[1], group.color[2], 70))
                item.setForeground(QColor(155, 155, 155) if out_of_hover_group else QColor(230, 230, 230))
            item.setToolTip(f"{base_name}\nGroup: {group_name} ({group.mode})\nID: {group.group_id}")
        self._highlight_sprites_for_palette_index(self._current_selected_palette_index())

    def _invalidate_used_index_cache(self, reason: str) -> None:
        self._used_index_cache = {}
        logger.debug("Used-index cache invalidated reason=%s", reason)

    def _record_used_indices(self, record: SpriteRecord) -> set[int]:
        key = str(record.path)
        cached = self._used_index_cache.get(key)
        if cached is not None:
            return cached
        used = set(int(px) for px in record.indexed_image.getdata())
        self._used_index_cache[key] = used
        return used

    def _merge_scope_records(self, scope: Literal["global", "group", "local"]) -> List[SpriteRecord]:
        if scope == "global":
            return list(self.sprite_records.values())
        anchor = self._current_record()
        if anchor is None:
            return []
        if scope == "local":
            return [anchor]
        if anchor.group_id:
            return list(self._iter_group_records(anchor))
        return [anchor]

    def _usage_counts_for_records(self, records: Sequence[SpriteRecord]) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for record in records:
            for idx in self._record_used_indices(record):
                counts[idx] = counts.get(idx, 0) + 1
        logger.debug("Usage counts computed scope_records=%s non_zero=%s", len(records), len(counts))
        return counts

    def _analyze_merge_impact_for_records(
        self,
        source_rows: Sequence[int],
        destination_row: int,
        records: Sequence[SpriteRecord],
    ) -> Dict[str, Any]:
        source_set = set(int(row) for row in source_rows)
        affected_sprites = 0
        risky_sprites = 0
        affected_keys: List[str] = []
        risky_keys: List[str] = []
        source_hits_by_key: Dict[str, int] = {}
        for record in records:
            used_indices = self._record_used_indices(record)
            key = str(record.path)
            source_hits = sum(1 for idx in source_set if idx in used_indices)
            source_hits_by_key[key] = source_hits
            if source_hits > 0:
                affected_sprites += 1
                affected_keys.append(key)
            destination_present = destination_row in used_indices
            risk = source_hits >= 2 or (destination_present and source_hits >= 1)
            if risk:
                risky_sprites += 1
                risky_keys.append(key)
        payload = {
            "source_rows": list(source_rows),
            "destination_row": int(destination_row),
            "total_sprites": len(records),
            "affected_sprites": affected_sprites,
            "risky_sprites": risky_sprites,
            "affected_keys": affected_keys,
            "risky_keys": risky_keys,
            "source_hits_by_key": source_hits_by_key,
        }
        logger.debug("Merge impact scoped %s", payload)
        return payload

    def _compute_destination_risk_levels(
        self,
        source_rows: Sequence[int],
        records: Sequence[SpriteRecord],
    ) -> Dict[int, str]:
        source_set = {int(row) for row in source_rows}
        if not source_set:
            return {}
        candidate_risk: Dict[int, str] = {}
        for candidate in range(len(self.palette_colors)):
            if candidate in source_set:
                continue
            affected = 0
            risky = 0
            for record in records:
                used = self._record_used_indices(record)
                source_hits = sum(1 for src in source_set if src in used)
                if source_hits == 0:
                    continue
                affected += 1
                if source_hits >= 2 or candidate in used:
                    risky += 1
            if affected == 0:
                continue
            if risky == 0:
                candidate_risk[candidate] = "safe"
            else:
                candidate_risk[candidate] = "risky"
        logger.debug(
            "Merge destination risk sources=%s records=%s candidates=%s",
            sorted(source_set),
            len(records),
            len(candidate_risk),
        )
        return candidate_risk

    @staticmethod
    def _compose_checkerboard_rgba(base_rgba: Image.Image) -> Image.Image:
        width, height = base_rgba.size
        checker = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        tile = 8
        light = (200, 200, 200, 255)
        dark = (150, 150, 150, 255)
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                checker.paste(
                    light if ((x // tile) + (y // tile)) % 2 == 0 else dark,
                    (x, y, min(x + tile, width), min(y + tile, height)),
                )
        checker.alpha_composite(base_rgba)
        return checker

    def _build_merge_preview_pixmap(
        self,
        source_rows: Sequence[int] | None,
        destination_row: int | None,
        *,
        scope: Literal["global", "group", "local"],
        record: SpriteRecord | None = None,
        highlight_index: int | None = None,
        blink_highlight: bool = False,
        blink_phase: float = 0.0,
    ) -> QPixmap | None:
        target_record = record or self._current_record()
        if target_record is None:
            return None
        preview_img = target_record.indexed_image.copy()
        merged_index_data: List[int] | None = None
        if destination_row is not None:
            source_set = {int(row) for row in (source_rows or []) if int(row) != int(destination_row)}
            if source_set:
                pixels = list(preview_img.getdata())
                merged_index_data = [destination_row if px in source_set else px for px in pixels]
                preview_img.putdata(merged_index_data)
        rgba = preview_img.convert("RGBA")

        if blink_highlight and highlight_index is not None:
            index_data = merged_index_data if merged_index_data is not None else list(preview_img.getdata())
            rgba_data = list(rgba.getdata())
            import math
            fade = (math.sin(blink_phase) + 1) / 2
            alpha_min = int(getattr(self, "_overlay_alpha_min", 77))
            alpha_max = int(getattr(self, "_overlay_alpha_max", 255))
            alpha_range = max(0, alpha_max - alpha_min)
            animated_alpha = int(alpha_min + alpha_range * fade)
            overlay_rgb = tuple(getattr(self, "_overlay_color", (255, 255, 255)))
            overlay_r, overlay_g, overlay_b = int(overlay_rgb[0]), int(overlay_rgb[1]), int(overlay_rgb[2])
            blend = max(0.0, min(1.0, animated_alpha / 255.0))
            highlighted: List[Tuple[int, int, int, int]] = []
            for pixel_idx, (r, g, b, a) in enumerate(rgba_data):
                if pixel_idx < len(index_data) and index_data[pixel_idx] == highlight_index and a > 0:
                    nr = min(255, int(r * (1.0 - blend) + overlay_r * blend))
                    ng = min(255, int(g * (1.0 - blend) + overlay_g * blend))
                    nb = min(255, int(b * (1.0 - blend) + overlay_b * blend))
                    highlighted.append((nr, ng, nb, a))
                else:
                    highlighted.append((r, g, b, a))
            rgba.putdata(highlighted)

        composed = self._compose_checkerboard_rgba(rgba)
        qimage = ImageQt(composed)
        pixmap = QPixmap.fromImage(qimage)
        logger.debug(
            "Merge preview sprite=%s scope=%s source_rows=%s destination=%s hover_index=%s blink=%s phase=%.2f size=%sx%s",
            target_record.path.name,
            scope,
            sorted({int(row) for row in (source_rows or [])}),
            destination_row,
            highlight_index,
            blink_highlight,
            blink_phase,
            pixmap.width(),
            pixmap.height(),
        )
        return pixmap

    def _current_selected_palette_index(self) -> int | None:
        current = self.palette_list.currentIndex()
        if current.isValid():
            return int(current.row())
        selected = self.palette_list.selectedIndexes()
        if selected:
            return int(selected[0].row())
        return None

    def _highlight_sprites_for_palette_index(self, palette_index: int | None) -> None:
        list_widget = self.images_panel.list_widget
        thumbnail_mode = self.images_panel.browser_view_mode() == "thumbnails"
        hovered_group_id = self._hover_group_id
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item is None:
                continue
            key = item.data(Qt.ItemDataRole.UserRole)
            if not key:
                continue
            record = self.sprite_records.get(str(key))
            if record is None:
                continue
            base_name = item.data(_SPRITE_BASE_NAME_ROLE) or record.path.name
            base_tooltip = item.toolTip().split("\nUses index:")[0]
            base_bg = QColor(0, 0, 0, 0)
            if record.group_id:
                group = self._palette_groups.get(record.group_id)
                if group is not None:
                    in_hover_group = bool(hovered_group_id and record.group_id == hovered_group_id)
                    if thumbnail_mode:
                        if in_hover_group:
                            base_bg = QColor(group.color[0], group.color[1], group.color[2], 75)
                        elif hovered_group_id:
                            base_bg = QColor(0, 0, 0, 55)
                        else:
                            base_bg = QColor(0, 0, 0, 0)
                    else:
                        if in_hover_group:
                            base_bg = QColor(group.color[0], group.color[1], group.color[2], 140)
                        elif hovered_group_id:
                            base_bg = QColor(group.color[0], group.color[1], group.color[2], 30)
                        else:
                            base_bg = QColor(group.color[0], group.color[1], group.color[2], 70)
                    group_name = self._group_display_name(group)
                    if "\nGroup:" not in base_tooltip:
                        base_tooltip = f"{base_name}\nGroup: {group_name} ({group.mode})\nID: {group.group_id}"
            elif hovered_group_id:
                base_bg = QColor(0, 0, 0, 55)

            if palette_index is None:
                item.setBackground(base_bg)
                item.setToolTip(base_tooltip)
                continue

            used = self._record_used_indices(record)
            if palette_index in used:
                item.setBackground(QColor(70, 180, 255, 120))
                item.setToolTip(f"{base_tooltip}\nUses index: {palette_index}")
            else:
                item.setBackground(base_bg)
                item.setToolTip(base_tooltip)

    def _open_merge_operation_dialog(self) -> None:
        if not self.palette_colors:
            QMessageBox.information(self, "Merge", "No palette colors available for merge.")
            return
        dialog = MergeOperationDialog(self)
        self._merge_dialog = dialog
        try:
            dialog.exec()
        finally:
            if self._merge_dialog is dialog:
                self._merge_dialog = None

    def _selected_group_id(self) -> str | None:
        item = self.images_panel.group_list.currentItem()
        if item is None:
            return None
        group_id = item.data(_GROUP_ID_ROLE)
        return str(group_id) if group_id else None

    def _set_hover_group_id(self, group_id: str | None) -> None:
        normalized = str(group_id).strip() if group_id else None
        if normalized == "":
            normalized = None
        if self._hover_group_id == normalized:
            return
        self._hover_group_id = normalized
        self._refresh_sprite_group_visuals()

    def _on_group_item_entered(self, item: QListWidgetItem) -> None:
        group_id = item.data(_GROUP_ID_ROLE)
        self._set_hover_group_id(str(group_id) if group_id else None)

    def _on_group_item_double_clicked(self, _item: QListWidgetItem) -> None:
        self._rename_selected_group()

    def _rename_selected_group(self) -> None:
        group_id = self._selected_group_id()
        if not group_id:
            QMessageBox.information(self, "Rename Group", "Select a group first.")
            return
        group = self._palette_groups.get(group_id)
        if group is None:
            return
        current_name = self._group_display_name(group)
        value, accepted = QInputDialog.getText(self, "Rename Group", "Group name:", text=current_name)
        if not accepted:
            return
        proposed = str(value).strip()
        if not proposed:
            QMessageBox.warning(self, "Rename Group", "Group name cannot be empty.")
            return
        for other_id, other in self._palette_groups.items():
            if other_id == group_id:
                continue
            if self._group_display_name(other).strip().lower() == proposed.lower():
                QMessageBox.warning(self, "Rename Group", "A group with that name already exists.")
                return
        group.display_name = proposed
        self._refresh_group_overview()
        self._mark_project_dirty("rename-group")
        self.statusBar().showMessage(f"Renamed group to {proposed}", 2500)

    def _selected_sprite_keys(self) -> List[str]:
        keys: List[str] = []
        for item in self.images_panel.list_widget.selectedItems():
            key = item.data(Qt.ItemDataRole.UserRole)
            if key:
                keys.append(str(key))
        return keys

    def _assign_record_to_group(self, key: str, record: SpriteRecord, target_group: PaletteGroup) -> bool:
        if record.group_id == target_group.group_id:
            return False
        if target_group.mode != record.load_mode:
            logger.debug(
                "Assign rejected mode mismatch path=%s sprite_mode=%s target_mode=%s",
                record.path.name,
                record.load_mode,
                target_group.mode,
            )
            return False
        self._remove_record_from_group(key)
        record.group_id = target_group.group_id
        target_group.member_keys.add(key)
        logger.debug(
            "Assigned sprite manually path=%s group=%s mode=%s",
            record.path.name,
            target_group.group_id,
            record.load_mode,
        )
        if record.load_mode == "detect":
            palette_info = extract_palette(record.indexed_image, include_unused=True)
            record.slot_bindings = {idx: idx for idx in range(min(256, len(palette_info.colors)))}
            logger.debug(
                "Detect manual assign keep-baseline path=%s group=%s palette_len=%s",
                record.path.name,
                target_group.group_id,
                len(palette_info.colors),
            )
        return True

    def _create_group_from_selected_sprites(self) -> None:
        keys = self._selected_sprite_keys()
        if not keys:
            self.statusBar().showMessage("Select sprites to create a group", 2500)
            return
        first = self.sprite_records.get(keys[0])
        if first is None:
            return
        group = self._create_palette_group(mode=first.load_mode, signature=f"manual:{first.load_mode}:{self._next_group_id}")
        self._palette_groups[group.group_id] = group
        changed = 0
        for key in keys:
            record = self.sprite_records.get(key)
            if record is None or record.load_mode != group.mode:
                continue
            if self._assign_record_to_group(key, record, group):
                changed += 1
        self._refresh_group_overview()
        self._refresh_palette_for_current_selection()
        self._schedule_preview_update()
        self.statusBar().showMessage(
            f"Created group {self._group_display_name(group)} with {changed} sprite(s)",
            3000,
        )
        logger.debug("Created manual group id=%s mode=%s assigned=%s", group.group_id, group.mode, changed)

    def _assign_selected_sprites_to_selected_group(self) -> None:
        group_id = self._selected_group_id()
        if not group_id:
            self.statusBar().showMessage("Select a target group first", 2500)
            return
        target = self._palette_groups.get(group_id)
        if target is None:
            return
        keys = self._selected_sprite_keys()
        if not keys:
            self.statusBar().showMessage("Select sprites to assign", 2500)
            return
        changed = 0
        skipped = 0
        for key in keys:
            record = self.sprite_records.get(key)
            if record is None:
                continue
            if self._assign_record_to_group(key, record, target):
                changed += 1
            else:
                skipped += 1
        self._refresh_group_overview()
        self._refresh_palette_for_current_selection()
        self._schedule_preview_update()
        self.statusBar().showMessage(
            f"Assigned {changed} sprite(s) to {group_id}" + (f" ({skipped} skipped)" if skipped else ""),
            3500,
        )
        logger.debug("Assign selected sprites target=%s changed=%s skipped=%s", group_id, changed, skipped)

    def _set_selected_group_color(self) -> None:
        group_id = self._selected_group_id()
        if not group_id:
            self.statusBar().showMessage("Select a group to recolor", 2500)
            return
        group = self._palette_groups.get(group_id)
        if group is None:
            return
        initial = QColor(group.color[0], group.color[1], group.color[2])
        chosen = QColorDialog.getColor(initial, self, f"Group Color {group.group_id}")
        if not chosen.isValid():
            return
        group.color = (chosen.red(), chosen.green(), chosen.blue())
        self._refresh_group_overview()
        logger.debug("Updated group color group=%s color=%s", group.group_id, group.color)

    def _detach_selected_sprites_to_individual_groups(self) -> None:
        keys = self._selected_sprite_keys()
        if not keys:
            self.statusBar().showMessage("Select sprites to detach", 2500)
            return
        detached = 0
        for key in keys:
            record = self.sprite_records.get(key)
            if record is None:
                continue
            group = self._create_palette_group(
                mode=record.load_mode,
                signature=f"manual-detach:{record.load_mode}:{record.path.name}:{self._next_group_id}",
            )
            self._palette_groups[group.group_id] = group
            if self._assign_record_to_group(key, record, group):
                detached += 1
        self._refresh_group_overview()
        self._refresh_palette_for_current_selection()
        self._schedule_preview_update()
        self.statusBar().showMessage(f"Detached {detached} sprite(s)", 3000)
        logger.debug("Detached sprites count=%s selected=%s", detached, len(keys))

    def _auto_assign_selected_sprites_to_signature_groups(self) -> None:
        keys = self._selected_sprite_keys()
        if not keys:
            self.statusBar().showMessage("Select sprites to auto-assign", 2500)
            return
        reassigned = 0
        for key in keys:
            record = self.sprite_records.get(key)
            if record is None:
                continue
            self._remove_record_from_group(key)
            self._assign_record_group(key, record)
            reassigned += 1
        self._refresh_group_overview()
        self._refresh_palette_for_current_selection()
        self._schedule_preview_update()
        self.statusBar().showMessage(f"Auto-assigned {reassigned} sprite(s)", 3000)
        logger.debug("Auto-assigned sprites by signature count=%s", reassigned)

    def _selected_palette_rows(self) -> List[int]:
        rows = sorted({index.row() for index in self.palette_list.selectedIndexes()})
        return [row for row in rows if 0 <= row < 256]

    def _resolve_group_slot_value(self, anchor: SpriteRecord, slot: int) -> Tuple[ColorTuple, int] | None:
        for member in self._iter_group_records(anchor):
            if member is anchor:
                continue
            palette = member.indexed_image.getpalette() or []
            base = slot * 3
            if base + 2 >= len(palette):
                continue
            color: ColorTuple = (palette[base], palette[base + 1], palette[base + 2])
            alphas = self._extract_palette_alphas_from_image(member.indexed_image, 256)
            alpha = alphas[slot] if slot < len(alphas) else 255
            return color, alpha
        return None

    def _set_selected_indices_local_override(self) -> None:
        record = self._current_record()
        if record is None:
            self.statusBar().showMessage("Select a sprite first", 2500)
            return
        rows = self._selected_palette_rows()
        if not rows:
            self.statusBar().showMessage("Select palette indices first", 2500)
            return
        model = self.palette_list.model_obj
        changed = 0
        selected_rows_before, current_row_before = self._capture_palette_selection_rows()
        for row in rows:
            color = model.colors[row]
            if color is None:
                continue
            alpha = model.alphas[row] if row < len(model.alphas) else 255
            self._set_local_override(record, row, color, alpha)
            changed += 1
        self._refresh_palette_for_current_selection()
        self._restore_palette_selection_rows(selected_rows_before, current_row_before)
        self._schedule_preview_update()
        self.statusBar().showMessage(f"Set local override for {changed} index(es)", 3000)
        logger.debug(
            "Set local overrides sprite=%s group=%s rows=%s changed=%s",
            record.path.name,
            record.group_id,
            rows,
            changed,
        )

    def _clear_selected_indices_local_override(self) -> None:
        record = self._current_record()
        if record is None:
            self.statusBar().showMessage("Select a sprite first", 2500)
            return
        rows = self._selected_palette_rows()
        if not rows:
            self.statusBar().showMessage("Select palette indices first", 2500)
            return
        cleared = 0
        selected_rows_before, current_row_before = self._capture_palette_selection_rows()
        for row in rows:
            if row not in record.local_overrides:
                continue
            group_value = self._resolve_group_slot_value(record, row)
            self._clear_local_override(record, row)
            if group_value is not None:
                color, alpha = group_value
                self._apply_detect_slot_deltas_to_record(record, [row], {row: color}, {row: alpha})
            cleared += 1
        self._refresh_palette_for_current_selection()
        self._restore_palette_selection_rows(selected_rows_before, current_row_before)
        self._schedule_preview_update()
        self.statusBar().showMessage(f"Cleared local override for {cleared} index(es)", 3000)
        logger.debug(
            "Cleared local overrides sprite=%s group=%s rows=%s cleared=%s",
            record.path.name,
            record.group_id,
            rows,
            cleared,
        )

    def _remove_record_from_group(self, key: str) -> None:
        record = self.sprite_records.get(key)
        if record is None or not record.group_id:
            return
        group_id = record.group_id
        group = self._palette_groups.get(group_id)
        if group is None:
            record.group_id = None
            return
        group.member_keys.discard(key)
        record.group_id = None
        if group.member_keys:
            logger.debug("Removed sprite from group id=%s remaining=%s", group_id, len(group.member_keys))
            return
        self._palette_groups.pop(group_id, None)
        stale_keys = [k for k, gid in self._group_key_to_id.items() if gid == group_id]
        for stale in stale_keys:
            self._group_key_to_id.pop(stale, None)
        logger.debug("Removed empty palette group id=%s", group_id)

    def _iter_group_records(self, anchor: SpriteRecord) -> List[SpriteRecord]:
        if not anchor.group_id:
            return [anchor]
        group = self._palette_groups.get(anchor.group_id)
        if group is None:
            return [anchor]
        records: List[SpriteRecord] = []
        for key in group.member_keys:
            member = self.sprite_records.get(key)
            if member is not None:
                records.append(member)
        if not records:
            return [anchor]
        return records

    def _apply_palette_to_record(
        self,
        record: SpriteRecord,
        colors: List[ColorTuple],
        alphas: List[int],
    ) -> None:
        palette_data: List[int] = []
        for idx in range(min(256, len(colors))):
            color = colors[idx]
            alpha = alphas[idx] if idx < len(alphas) else 255
            if idx in record.local_overrides:
                color, alpha = record.local_overrides[idx]
            palette_data.extend([int(color[0]), int(color[1]), int(color[2])])
        while len(palette_data) < 768:
            palette_data.append(0)
        record.indexed_image.putpalette(palette_data[:768])

        alpha_data = [255] * 256
        for idx in range(min(256, len(colors))):
            alpha_data[idx] = max(0, min(255, int(alphas[idx] if idx < len(alphas) else 255)))
        for idx, (_color, alpha) in record.local_overrides.items():
            if 0 <= idx < 256:
                alpha_data[idx] = max(0, min(255, int(alpha)))
        if any(alpha < 255 for alpha in alpha_data):
            record.indexed_image.info["transparency"] = bytes(alpha_data)
        elif "transparency" in record.indexed_image.info:
            del record.indexed_image.info["transparency"]

    def _set_local_override(self, record: SpriteRecord, index: int, color: ColorTuple, alpha: int) -> None:
        if index < 0 or index > 255:
            return
        alpha_clamped = max(0, min(255, int(alpha)))
        record.local_overrides[index] = (color, alpha_clamped)
        logger.debug(
            "Set local override sprite=%s group=%s index=%s color=%s alpha=%s total_overrides=%s",
            record.path.name,
            record.group_id,
            index,
            color,
            alpha_clamped,
            len(record.local_overrides),
        )

    def _clear_local_override(self, record: SpriteRecord, index: int) -> None:
        if index in record.local_overrides:
            del record.local_overrides[index]
            logger.debug(
                "Cleared local override sprite=%s group=%s index=%s remaining=%s",
                record.path.name,
                record.group_id,
                index,
                len(record.local_overrides),
            )

    def _apply_overrides_to_indexed_image(
        self,
        image: Image.Image,
        overrides: Dict[int, Tuple[ColorTuple, int]],
    ) -> None:
        if not overrides:
            return
        palette_data = list(image.getpalette() or [])
        if len(palette_data) < 768:
            palette_data.extend([0] * (768 - len(palette_data)))
        alpha_data = self._extract_palette_alphas_from_image(image, 256)
        for index, (color, alpha) in overrides.items():
            if index < 0 or index > 255:
                continue
            base = index * 3
            palette_data[base] = int(color[0])
            palette_data[base + 1] = int(color[1])
            palette_data[base + 2] = int(color[2])
            alpha_data[index] = max(0, min(255, int(alpha)))
        image.putpalette(palette_data[:768])
        if any(a < 255 for a in alpha_data):
            image.info["transparency"] = bytes(alpha_data)
        elif "transparency" in image.info:
            del image.info["transparency"]
        logger.debug("Applied local overrides count=%s", len(overrides))

    def _current_palette_by_slot(self) -> Tuple[Dict[int, ColorTuple], Dict[int, int]]:
        color_by_slot: Dict[int, ColorTuple] = {}
        alpha_by_slot: Dict[int, int] = {}
        for idx, slot_id in enumerate(self._palette_slot_ids):
            if idx >= len(self.palette_colors):
                continue
            color_by_slot[slot_id] = self.palette_colors[idx]
            alpha_by_slot[slot_id] = self.palette_alphas[idx] if idx < len(self.palette_alphas) else 255
        return color_by_slot, alpha_by_slot

    def _apply_detect_slot_deltas_to_record(
        self,
        record: SpriteRecord,
        changed_slots: List[int],
        new_color_by_slot: Dict[int, ColorTuple],
        new_alpha_by_slot: Dict[int, int],
    ) -> None:
        if not changed_slots:
            return
        palette_data = list(record.indexed_image.getpalette() or [])
        if len(palette_data) < 768:
            palette_data.extend([0] * (768 - len(palette_data)))
        alpha_data = self._extract_palette_alphas_from_image(record.indexed_image, 256)

        for slot in changed_slots:
            if slot < 0 or slot > 255:
                continue
            color = new_color_by_slot.get(slot)
            if color is None:
                continue
            alpha = new_alpha_by_slot.get(slot, 255)
            base = slot * 3
            palette_data[base] = int(color[0])
            palette_data[base + 1] = int(color[1])
            palette_data[base + 2] = int(color[2])
            alpha_data[slot] = max(0, min(255, int(alpha)))

        record.indexed_image.putpalette(palette_data[:768])
        if any(a < 255 for a in alpha_data):
            record.indexed_image.info["transparency"] = bytes(alpha_data)
        elif "transparency" in record.indexed_image.info:
            del record.indexed_image.info["transparency"]
        if record.local_overrides:
            self._apply_overrides_to_indexed_image(record.indexed_image, record.local_overrides)

    def _apply_detect_remap_to_record(self, record: SpriteRecord, pixel_remap: Dict[int, int]) -> None:
        if not pixel_remap:
            return
        img = record.indexed_image
        pixels = list(img.getdata())
        img.putdata([pixel_remap.get(px, px) for px in pixels])

        palette_data = list(img.getpalette() or [])
        if len(palette_data) < 768:
            palette_data.extend([0] * (768 - len(palette_data)))
        alpha_data = self._extract_palette_alphas_from_image(img, 256)

        inverse_remap = {new_idx: old_idx for old_idx, new_idx in pixel_remap.items()}
        new_palette_data = list(palette_data)
        new_alpha_data = list(alpha_data)
        for new_idx in range(256):
            old_idx = inverse_remap.get(new_idx, new_idx)
            if old_idx < 0 or old_idx > 255:
                continue
            old_base = old_idx * 3
            new_base = new_idx * 3
            new_palette_data[new_base] = palette_data[old_base]
            new_palette_data[new_base + 1] = palette_data[old_base + 1]
            new_palette_data[new_base + 2] = palette_data[old_base + 2]
            new_alpha_data[new_idx] = alpha_data[old_idx] if old_idx < len(alpha_data) else 255

        img.putpalette(new_palette_data[:768])
        if any(a < 255 for a in new_alpha_data):
            img.info["transparency"] = bytes(new_alpha_data)
        elif "transparency" in img.info:
            del img.info["transparency"]
        if record.local_overrides:
            self._apply_overrides_to_indexed_image(img, record.local_overrides)

    def _rebuild_slot_color_lookup(self) -> None:
        lookup: Dict[ColorTuple, List[int]] = {}
        for slot_id, color in zip(self._palette_slot_ids, self.palette_colors):
            if color is not None:  # Skip None values
                lookup.setdefault(color, []).append(slot_id)
        self._slot_color_lookup = lookup
        self._slot_color_lookup = lookup

    def _log_palette_stats(self, reason: str) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        first_seen: Dict[ColorTuple, int] = {}
        duplicates: List[Tuple[int, int, ColorTuple]] = []
        for idx, color in enumerate(self.palette_colors):
            prior = first_seen.get(color)
            if prior is None:
                first_seen[color] = idx
            else:
                duplicates.append((prior, idx, color))
        if duplicates:
            sample = duplicates[:5]
            logger.debug(
                "Palette stats reason=%s total=%s unique=%s duplicate_count=%s sample=%s",
                reason,
                len(self.palette_colors),
                len(first_seen),
                len(duplicates),
                sample,
            )
        else:
            logger.debug(
                "Palette stats reason=%s total=%s unique=%s duplicate_count=0",
                reason,
                len(self.palette_colors),
                len(first_seen),
            )

    def _rebuild_palette_from_records(self) -> None:
        records = [record for record in self.sprite_records.values() if record.load_mode == "detect"]
        for group in self._palette_groups.values():
            if group.mode == "detect":
                group.detect_palette_colors = []
                group.detect_palette_alphas = []
                group.detect_palette_slot_ids = []
        self.palette_colors = []
        self.palette_alphas = []
        self._palette_slot_ids = []
        self._slot_color_lookup = {}
        self._next_slot_id = 0
        for record in records:
            record.slot_bindings = {}
            self._merge_record_palette(record)
        self._refresh_palette_for_current_selection()

    def _extract_palette_alphas_from_image(self, image: Image.Image, count: int) -> List[int]:
        alphas = [255] * max(0, count)
        transparency = image.info.get("transparency")
        if isinstance(transparency, int):
            if 0 <= transparency < len(alphas):
                alphas[transparency] = 0
        elif isinstance(transparency, (bytes, bytearray, list, tuple)):
            for idx in range(min(len(alphas), len(transparency))):
                try:
                    alphas[idx] = max(0, min(255, int(transparency[idx])))
                except (TypeError, ValueError):
                    alphas[idx] = 255
        return alphas

    def _sync_palette_from_current_record_preserve(self) -> None:
        record = self._current_record()
        if record is None:
            self.palette_colors = []
            self.palette_alphas = []
            self._palette_slot_ids = []
            self.palette_list.set_colors([], emit_signal=False)
            self._update_fill_preview()
            return
        palette_info = extract_palette(record.indexed_image, include_unused=True)
        self.palette_colors = list(palette_info.colors[:256])
        self.palette_alphas = self._extract_palette_alphas_from_image(record.indexed_image, len(self.palette_colors))
        self._palette_slot_ids = list(range(len(self.palette_colors)))
        self._rebuild_slot_color_lookup()
        self.palette_list.set_colors(
            self.palette_colors,
            slots=self._palette_slot_ids,
            alphas=self.palette_alphas,
            emit_signal=False,
        )
        self._update_fill_preview()

    def _remove_selected_images(self) -> None:
        list_widget = self.images_panel.list_widget
        selected = list_widget.selectedItems()
        if not selected:
            return
        for item in selected:
            key = item.data(Qt.ItemDataRole.UserRole)
            row = list_widget.row(item)
            list_widget.takeItem(row)
            if key in self.sprite_records:
                self._remove_record_from_group(key)
                del self.sprite_records[key]
        self._rebuild_loaded_images_list()
        self._rebuild_palette_from_records()
        self._refresh_palette_for_current_selection()
        self._refresh_group_overview()
        self._clear_preview_cache()
        self._update_export_buttons()
        self._update_loaded_count()
        self._reset_history()
        self._schedule_preview_update()
        self._mark_project_dirty("remove-images")

    def _create_record(self, path: Path, load_mode: Literal["detect", "preserve"] = "detect") -> SpriteRecord | None:
        try:
            with Image.open(path) as img:
                preview = img.copy()
                preview.thumbnail((256, 256))
                qimage = ImageQt(preview)
                pixmap = QPixmap.fromImage(qimage)
                indexed_image, colors = self._prepare_indexed_sprite_image(img, load_mode=load_mode)
        except PaletteError as exc:
            logger.warning("Skipping %s: %s", path, exc)
            self.statusBar().showMessage(f"{path.name}: {exc}", 5000)
            return None
        except OSError:
            self.statusBar().showMessage(f"Failed to load {path.name}", 5000)
            return None
        record = SpriteRecord(
            path=path,
            pixmap=pixmap,
            palette=colors,
            slot_bindings={},
            indexed_image=indexed_image,
            load_mode=load_mode,
            canvas_width=self.canvas_width_spin.value() if hasattr(self, "canvas_width_spin") else 304,
            canvas_height=self.canvas_height_spin.value() if hasattr(self, "canvas_height_spin") else 224,
        )
        logger.debug(
            "Created record path=%s load_mode=%s source_mode=%s palette_len=%s",
            path.name,
            load_mode,
            record.indexed_image.mode,
            len(record.palette),
        )
        return record

    def _prepare_indexed_sprite_image(self, image: Image.Image, load_mode: Literal["detect", "preserve"] = "detect") -> tuple[Image.Image, List[ColorTuple]]:
        working = image.copy()
        preserve_indexed = load_mode == "preserve"
        if preserve_indexed and working.mode != "P":
            raise PaletteError("Preserve Indexes mode requires indexed PNG images (mode 'P').")
        if working.mode == "P":
            palette = extract_palette(working, include_unused=preserve_indexed)
            if preserve_indexed:
                # Preserve mode: keep the indexed image as-is with its original palette
                return working.copy(), palette.colors[:256]
            return working.copy(), palette.colors[:256]

        rgba = working.convert("RGBA")
        ordered_entries, overflow = self._collect_unique_colors(rgba, max_colors=256)
        if not overflow and ordered_entries:
            logger.debug("Building exact indexed image unique_entries=%s", len(ordered_entries))
            indexed, palette_colors = self._build_exact_indexed_image(rgba, ordered_entries)
            return indexed, palette_colors

        limit = self._estimate_color_budget(rgba)
        logger.debug(
            "Quantizing %s image for palette extraction (limit=%s)",
            working.mode,
            limit,
        )
        quantized = quantize_image(rgba, max_colors=limit, dither=False)
        palette = extract_palette(quantized)
        return quantized.copy(), palette.colors[:256]

    def _estimate_color_budget(self, image: Image.Image) -> int:
        rgba = image.convert("RGBA")
        samples = rgba.getcolors(maxcolors=1_000_000)
        if not samples:
            return 256
        unique = {color[:3] for _count, color in samples}
        budget = min(max(1, len(unique)), 256)
        logger.debug("Estimated color budget unique=%s (limit=%s)", len(unique), budget)
        return budget

    def _collect_unique_colors(self, image: Image.Image, max_colors: int) -> tuple[List[tuple[int, int, int, int]], bool]:
        seen: Dict[tuple[int, int, int, int], int] = {}
        order: List[tuple[int, int, int, int]] = []
        for pixel in image.getdata():
            rgba = (
                int(pixel[0]),
                int(pixel[1]),
                int(pixel[2]),
                int(pixel[3]) if len(pixel) > 3 else 255,
            )
            if rgba in seen:
                continue
            seen[rgba] = 1
            order.append(rgba)
            if len(order) > max_colors:
                return order, True
        return order, False

    def _build_exact_indexed_image(
        self,
        rgba: Image.Image,
        palette_entries: List[tuple[int, int, int, int]],
    ) -> tuple[Image.Image, List[ColorTuple]]:
        color_to_index: Dict[tuple[int, int, int, int], int] = {
            entry: idx for idx, entry in enumerate(palette_entries)
        }
        data = [
            color_to_index[(
                int(pixel[0]),
                int(pixel[1]),
                int(pixel[2]),
                int(pixel[3]) if len(pixel) > 3 else 255,
            )]
            for pixel in rgba.getdata()
        ]
        indexed = Image.new("P", rgba.size)
        indexed.putdata(data)
        palette_colors = [(entry[0], entry[1], entry[2]) for entry in palette_entries]
        palette_alphas = [entry[3] for entry in palette_entries]
        flat_palette: List[int] = []
        for color in palette_colors:
            flat_palette.extend(color)
        if len(flat_palette) < 768:
            flat_palette.extend([0] * (768 - len(flat_palette)))
        indexed.putpalette(flat_palette)
        if any(alpha < 255 for alpha in palette_alphas):
            padded_alphas = palette_alphas[:256] + [255] * max(0, 256 - len(palette_alphas))
            indexed.info["transparency"] = bytes(padded_alphas[:256])
        return indexed, palette_colors

    def _on_selection_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if self._is_loading_sprites:
            return
        self._update_export_buttons()
        self._update_animation_tag_controls()
        record = self._current_record()
        if record is None:
            if self._is_animation_assist_view_active() and not self.animation_play_btn.isChecked():
                self._update_sprite_offset_controls(None)
                self._refresh_palette_for_current_selection()
                return
            self.preview_panel.set_pixmap(None, reset_zoom=True)
            self._clear_preview_cache()
            self._update_sprite_offset_controls(None)
            self._refresh_palette_for_current_selection()
            return
        self.statusBar().showMessage(f"Previewing {record.path.name}", 2000)
        # Update per-sprite offset controls
        self._update_sprite_offset_controls(record)
        self._refresh_palette_for_current_selection()
        if self._is_animation_assist_view_active() and not self.animation_play_btn.isChecked():
            if self._preview_animation_follow_selection:
                self._refresh_animation_assist_preview_frame()
            return
        # Don't reset zoom when switching images - retain user's zoom level
        self._reset_zoom_next = False
        self._schedule_preview_update()
    
    def _update_sprite_offset_controls(self, record: SpriteRecord | None) -> None:
        """Update per-sprite offset spinboxes to match current sprite."""
        scope = self.offset_scope_combo.currentData() if hasattr(self, "offset_scope_combo") else "combined"
        show_global = scope in ("combined", "global")
        show_group = scope in ("combined", "group")
        show_local = scope in ("combined", "local")
        if hasattr(self, "global_offset_widget"):
            self.global_offset_widget.setVisible(show_global)
        if hasattr(self, "group_offset_widget"):
            self.group_offset_widget.setVisible(show_group)
        if hasattr(self, "local_offset_widget"):
            self.local_offset_widget.setVisible(show_local)

        if record is None:
            self.sprite_offset_x_spin.setEnabled(False)
            self.sprite_offset_y_spin.setEnabled(False)
            self.group_offset_x_spin.setEnabled(False)
            self.group_offset_y_spin.setEnabled(False)
            self.global_offset_x_spin.setEnabled(False)
            self.global_offset_y_spin.setEnabled(False)
            self.global_drag_mode_btn.setEnabled(False)
            self.group_drag_mode_btn.setEnabled(False)
            self.individual_drag_mode_btn.setEnabled(False)
            self.global_drag_mode_btn.setChecked(False)
            self.group_drag_mode_btn.setChecked(False)
            self.individual_drag_mode_btn.setChecked(False)
            self._active_offset_drag_mode = "none"
            self.preview_panel.set_drag_mode(False)
        else:
            self.sprite_offset_x_spin.setEnabled(True)
            self.sprite_offset_y_spin.setEnabled(True)
            group = self._palette_groups.get(record.group_id) if record.group_id else None
            group_enabled = group is not None
            self.group_offset_x_spin.setEnabled(group_enabled)
            self.group_offset_y_spin.setEnabled(group_enabled)
            self.global_offset_x_spin.setEnabled(True)
            self.global_offset_y_spin.setEnabled(True)
            self.global_drag_mode_btn.setEnabled(show_global)
            self.group_drag_mode_btn.setEnabled(show_group and group_enabled)
            self.individual_drag_mode_btn.setEnabled(show_local)
            
            # Block signals to prevent triggering preview update
            self.sprite_offset_x_spin.blockSignals(True)
            self.sprite_offset_y_spin.blockSignals(True)
            self.group_offset_x_spin.blockSignals(True)
            self.group_offset_y_spin.blockSignals(True)
            self.sprite_offset_x_spin.setValue(record.offset_x)
            self.sprite_offset_y_spin.setValue(record.offset_y)
            if group is not None:
                self.group_offset_x_spin.setValue(group.offset_x)
                self.group_offset_y_spin.setValue(group.offset_y)
            else:
                self.group_offset_x_spin.setValue(0)
                self.group_offset_y_spin.setValue(0)
            self.sprite_offset_x_spin.blockSignals(False)
            self.sprite_offset_y_spin.blockSignals(False)
            self.group_offset_x_spin.blockSignals(False)
            self.group_offset_y_spin.blockSignals(False)
        self._update_canvas_inputs()

    def _on_palette_changed(self, colors: List[ColorTuple]) -> None:
        if self._is_loading_sprites:
            return
        self._clear_detect_batch_jobs("new-palette-change")
        anchor = self._current_record()
        if anchor is not None and anchor.load_mode == "preserve":
            self._on_palette_changed_preserve(colors)
            return
        if anchor is None:
            return

        old_color_by_slot, old_alpha_by_slot = self._current_palette_by_slot()

        # colors now only contains non-None values from the model
        # Extract slot positions and colors from model's 256-slot array
        model = self.palette_list.model_obj
        model_colors = model.colors
        model_alphas = model.alphas
        
        # Build mapping from old palette position to new position
        old_palette_colors = list(self.palette_colors)
        old_slot_ids = list(self._palette_slot_ids)
        
        self.palette_colors = []
        self.palette_alphas = []
        self._palette_slot_ids = []
        for slot_id in range(256):
            color = model_colors[slot_id]
            if color is not None:
                self.palette_colors.append(color)
                self.palette_alphas.append(model_alphas[slot_id])
                self._palette_slot_ids.append(slot_id)
        
        # If palette was reordered/swapped, remap sprite pixels to maintain visual appearance
        reason = self.palette_list._last_change_reason
        if reason in ("reorder", "swap", "shift"):
            # Build pixel remap: old_index -> new_index
            pixel_remap: Dict[int, int] = {}
            if reason == "shift" and self._pending_palette_index_remap is not None:
                pixel_remap = dict(self._pending_palette_index_remap)
                logger.debug("Using pending shift remap entries=%s", len(pixel_remap))
            elif self.palette_list.model_obj.last_index_remap is not None:
                pixel_remap = dict(self.palette_list.model_obj.last_index_remap)
                logger.debug("Using model index remap reason=%s entries=%s", reason, len(pixel_remap))
            else:
                for old_idx, old_color in enumerate(old_palette_colors):
                    # Find where this color is in the new palette
                    try:
                        new_idx = self.palette_colors.index(old_color)
                        pixel_remap[old_idx] = new_idx
                    except ValueError:
                        # Color was removed? shouldn't happen during reorder/swap
                        pixel_remap[old_idx] = old_idx
            
            # Apply remap to all sprites if any indices changed
            if any(old_idx != new_idx for old_idx, new_idx in pixel_remap.items()):
                logger.debug(f"Remapping pixels due to palette {reason}: {pixel_remap}")
                targets = [member for member in self._iter_group_records(anchor) if member.load_mode == "detect"]
                anchor_key = anchor.path.as_posix()
                self._apply_detect_remap_to_record(anchor, pixel_remap)
                self._enqueue_detect_remap_batch_updates(
                    targets,
                    pixel_remap,
                    reason=reason,
                    anchor_key=anchor_key,
                )
                self._invalidate_used_index_cache(f"palette-remap-detect:{reason}")

                logger.debug(
                    "Detect remap propagated group=%s reason=%s targets=%s entries=%s",
                    anchor.group_id,
                    reason,
                    len(targets),
                    len(pixel_remap),
                )
                
                self._invalidate_preview_cache(reset_zoom=False, reason=f"detect-remap:{reason}")
            if reason == "shift":
                self._pending_palette_index_remap = None
        else:
            new_color_by_slot, new_alpha_by_slot = self._current_palette_by_slot()
            changed_slots = sorted(
                slot
                for slot in set(old_color_by_slot.keys()) | set(new_color_by_slot.keys())
                if old_color_by_slot.get(slot) != new_color_by_slot.get(slot)
                or old_alpha_by_slot.get(slot, 255) != new_alpha_by_slot.get(slot, 255)
            )
            if changed_slots:
                local_only_slots = sorted([slot for slot in changed_slots if slot in anchor.local_overrides])
                group_slots = [slot for slot in changed_slots if slot not in anchor.local_overrides]

                for slot in local_only_slots:
                    color = new_color_by_slot.get(slot)
                    if color is None:
                        continue
                    alpha = new_alpha_by_slot.get(slot, 255)
                    self._set_local_override(anchor, slot, color, alpha)

                targets = [member for member in self._iter_group_records(anchor) if member.load_mode == "detect"]
                anchor_key = anchor.path.as_posix()
                if group_slots:
                    self._apply_detect_slot_deltas_to_record(
                        anchor,
                        group_slots,
                        new_color_by_slot,
                        new_alpha_by_slot,
                    )
                    self._enqueue_detect_slot_batch_updates(
                        targets,
                        group_slots,
                        new_color_by_slot,
                        new_alpha_by_slot,
                        reason=reason,
                        anchor_key=anchor_key,
                    )
                if local_only_slots:
                    self._apply_detect_slot_deltas_to_record(
                        anchor,
                        local_only_slots,
                        new_color_by_slot,
                        new_alpha_by_slot,
                    )
                logger.debug(
                    "Detect slot deltas propagated group=%s reason=%s targets=%s group_slots=%s local_only_slots=%s sample=%s",
                    anchor.group_id,
                    reason,
                    len(targets),
                    len(group_slots),
                    len(local_only_slots),
                    changed_slots[:12],
                )
                self._invalidate_preview_cache(reset_zoom=False, reason=f"detect-slot-deltas:{reason}")
        
        self._rebuild_slot_color_lookup()
        if anchor.group_id:
            group = self._palette_groups.get(anchor.group_id)
            if group is not None and group.mode == "detect":
                self._store_detect_group_palette_state(group, f"palette-changed:{reason}")
        delay = 75 if reason == "reorder" else 0
        logger.debug(
            "Palette changed reason=%s delay=%sms colors=%s alpha_sample=%s",
            reason,
            delay,
            len(self.palette_colors),
            self.palette_alphas[:8],
        )
        self._update_fill_preview()
        self._schedule_preview_update(delay_ms=delay)
        if reason != "refresh":
            self._record_history(f"palette:{reason}", force=True)

    def _on_palette_changed_preserve(self, colors: List[ColorTuple]) -> None:
        anchor = self._current_record()
        if anchor is None:
            return

        model = self.palette_list.model_obj
        model_colors = model.colors
        model_alphas = model.alphas
        old_palette = extract_palette(anchor.indexed_image, include_unused=True)
        old_colors = list(old_palette.colors)
        old_alphas = self._extract_palette_alphas_from_image(anchor.indexed_image, len(old_colors))

        self.palette_colors = []
        self.palette_alphas = []
        self._palette_slot_ids = []
        for slot_id in range(256):
            color = model_colors[slot_id]
            if color is not None:
                self.palette_colors.append(color)
                self.palette_alphas.append(model_alphas[slot_id])
                self._palette_slot_ids.append(slot_id)

        reason = self.palette_list._last_change_reason
        pixel_remap: Dict[int, int] = {}
        if reason in ("reorder", "swap", "shift"):
            if reason == "shift" and self._pending_palette_index_remap is not None:
                pixel_remap = dict(self._pending_palette_index_remap)
                logger.debug("Using pending preserve shift remap entries=%s", len(pixel_remap))
            elif self.palette_list.model_obj.last_index_remap is not None:
                pixel_remap = dict(self.palette_list.model_obj.last_index_remap)
                logger.debug("Using model preserve remap reason=%s entries=%s", reason, len(pixel_remap))
            else:
                for old_idx, old_color in enumerate(old_colors):
                    try:
                        pixel_remap[old_idx] = self.palette_colors.index(old_color)
                    except ValueError:
                        pixel_remap[old_idx] = old_idx
            if reason == "shift":
                self._pending_palette_index_remap = None

        targets = [r for r in self._iter_group_records(anchor) if r.load_mode == "preserve"]
        remap_changed = any(old_idx != new_idx for old_idx, new_idx in pixel_remap.items())
        if remap_changed:
            for target in targets:
                pixels = list(target.indexed_image.getdata())
                target.indexed_image.putdata([pixel_remap.get(px, px) for px in pixels])
                self._apply_palette_to_record(target, self.palette_colors, self.palette_alphas)
                target.palette = list(self.palette_colors)
            self._invalidate_used_index_cache(f"palette-remap-preserve:{reason}")
        else:
            old_color_by_slot = {idx: color for idx, color in enumerate(old_colors)}
            old_alpha_by_slot = {idx: (old_alphas[idx] if idx < len(old_alphas) else 255) for idx in range(len(old_colors))}
            new_color_by_slot = {slot_id: color for slot_id, color in zip(self._palette_slot_ids, self.palette_colors)}
            new_alpha_by_slot = {
                slot_id: (self.palette_alphas[idx] if idx < len(self.palette_alphas) else 255)
                for idx, slot_id in enumerate(self._palette_slot_ids)
            }
            changed_slots = sorted(
                slot
                for slot in set(old_color_by_slot.keys()) | set(new_color_by_slot.keys())
                if old_color_by_slot.get(slot) != new_color_by_slot.get(slot)
                or old_alpha_by_slot.get(slot, 255) != new_alpha_by_slot.get(slot, 255)
            )
            local_only_slots = sorted([slot for slot in changed_slots if slot in anchor.local_overrides])
            group_slots = [slot for slot in changed_slots if slot not in anchor.local_overrides]

            for slot in local_only_slots:
                color = new_color_by_slot.get(slot)
                if color is None:
                    continue
                alpha = new_alpha_by_slot.get(slot, 255)
                self._set_local_override(anchor, slot, color, alpha)

            if group_slots:
                for target in targets:
                    self._apply_detect_slot_deltas_to_record(target, group_slots, new_color_by_slot, new_alpha_by_slot)
                    target.palette = list(extract_palette(target.indexed_image, include_unused=True).colors[:256])
            if local_only_slots:
                self._apply_detect_slot_deltas_to_record(anchor, local_only_slots, new_color_by_slot, new_alpha_by_slot)
                anchor.palette = list(extract_palette(anchor.indexed_image, include_unused=True).colors[:256])

            logger.debug(
                "Preserve slot deltas propagated group=%s reason=%s targets=%s group_slots=%s local_only_slots=%s sample=%s",
                anchor.group_id,
                reason,
                len(targets),
                len(group_slots),
                len(local_only_slots),
                changed_slots[:12],
            )

        self._rebuild_slot_color_lookup()
        logger.debug(
            "Preserve palette changed anchor=%s group=%s reason=%s colors=%s alpha_sample=%s targets=%s remap_changed=%s",
            anchor.path.name,
            anchor.group_id,
            reason,
            len(self.palette_colors),
            self.palette_alphas[:8],
            len(targets),
            remap_changed,
        )
        self._invalidate_preview_cache(reset_zoom=False, reason=f"preserve-palette-changed:{reason}")
        self._update_fill_preview()
        self._schedule_preview_update()
        if reason != "refresh":
            self._record_history(f"palette:{reason}:preserve", force=True)

    def _schedule_preview_update(self, delay_ms: int = 0) -> None:
        self._preview_request_serial += 1
        effective_delay = 12 if delay_ms <= 0 else delay_ms
        if self._preview_timer.isActive():
            remaining = self._preview_timer.remainingTime()
            if remaining >= 0 and remaining <= effective_delay:
                return
            self._preview_timer.stop()
        self._preview_timer.start(effective_delay)

    def _handle_preview_timer_timeout(self) -> None:
        if self._is_animation_assist_view_active():
            if self.animation_play_btn.isChecked():
                # Playback timer owns assist frame cadence while playing.
                return
            else:
                self._refresh_animation_assist_preview_frame()
            return
        self._render_preview()

    def _render_preview(self) -> None:
        record = self._current_record()
        if record is None:
            self.preview_panel.set_pixmap(None, reset_zoom=True)
            self._clear_preview_cache()
            return
        options = self._build_process_options(
            record,
            output_dir=record.path.parent / "_preview",
            preserve_palette=True,
        )
        self._log_process_options("preview", options, record)
        source = record.indexed_image.copy()
        if record.local_overrides:
            self._apply_overrides_to_indexed_image(source, record.local_overrides)
        request = {
            "id": self._preview_request_serial,
            "record_key": record.path.as_posix(),
            "record_name": record.path.name,
            "source": source,
            "options": options,
            "started_at": time.perf_counter(),
        }
        self._preview_pending_request = request
        self._kick_preview_worker()

    def _kick_preview_worker(self) -> None:
        if self._preview_future is not None and not self._preview_future.done():
            return
        if self._preview_pending_request is None:
            return
        request = self._preview_pending_request
        self._preview_pending_request = None
        self._preview_active_request = request
        self._preview_future = self._preview_executor.submit(
            _process_preview_request,
            request["source"],
            request["options"],
        )

        def _notify_preview_done(_future: Future[tuple[Image.Image, PaletteInfo]]) -> None:
            try:
                self.previewFutureDone.emit()
            except RuntimeError:
                return

        self._preview_future.add_done_callback(_notify_preview_done)

    def _drain_preview_result(self) -> None:
        future = self._preview_future
        if future is None:
            if self._preview_pending_request is not None:
                self._kick_preview_worker()
            return
        if not future.done():
            return

        self._preview_future = None
        request = self._preview_active_request
        self._preview_active_request = None

        try:
            processed, palette = future.result()
        except (OSError, ValueError) as exc:
            if request and request.get("id") == self._preview_request_serial:
                self.statusBar().showMessage(f"Preview failed: {exc}", 3000)
                record = self._current_record()
                if record is not None:
                    self.preview_panel.set_pixmap(record.pixmap, reset_zoom=False)
                self._clear_preview_cache()
        else:
            is_latest = bool(request and request.get("id") == self._preview_request_serial)
            current = self._current_record()
            is_current_record = bool(
                request
                and current is not None
                and request.get("record_key") == current.path.as_posix()
            )
            if request is not None:
                elapsed_ms = (time.perf_counter() - float(request.get("started_at", time.perf_counter()))) * 1000.0
                logger.debug(
                    "Preview worker result request_id=%s latest=%s current=%s elapsed_ms=%.2f",
                    request.get("id"),
                    is_latest,
                    is_current_record,
                    elapsed_ms,
                )
            if (
                is_latest
                and is_current_record
                and not self.animation_play_btn.isChecked()
                and not self._is_animation_assist_view_active()
            ):
                self._last_processed_indexed = processed.copy()
                self._last_index_data = list(self._last_processed_indexed.getdata())
                self._overlay_region_by_index = {}
                self._overlay_region_cache_shape = None
                self._overlay_region_cache_token = None
                self._last_preview_rgba = processed.convert("RGBA")
                self._preview_base_pixmap = None
                self._last_palette_info = palette
                self._palette_index_lookup = self._build_palette_lookup(palette)
                self._log_palette_debug_info(palette)
                self._update_preview_pixmap()

        if self._preview_pending_request is not None:
            self._kick_preview_worker()

    def _current_record(self) -> SpriteRecord | None:
        current_item = self.images_panel.list_widget.currentItem()
        if current_item is None:
            return None
        path_key = current_item.data(Qt.ItemDataRole.UserRole)
        return self.sprite_records.get(path_key)

    def _current_drag_target_record(self) -> SpriteRecord | None:
        if self._offset_drag_live and self._offset_drag_target_key:
            locked_record = self.sprite_records.get(self._offset_drag_target_key)
            if locked_record is not None:
                return locked_record
        if self._is_animation_assist_view_active() or self.animation_play_btn.isChecked():
            visible_key = self._animation_preview_visible_sprite_key
            if visible_key:
                visible_record = self.sprite_records.get(visible_key)
                if visible_record is not None:
                    return visible_record
        return self._current_record()

    def _build_process_options(
        self,
        record: SpriteRecord,
        *,
        output_dir: Path,
        preserve_palette: bool = False,
        write_act: bool = False,
        debug_context: str = "general",
    ) -> ProcessOptions:
        preserve_mode = record.load_mode == "preserve"
        palette = None
        fill_mode = self._selected_fill_mode()
        fill_index = self.fill_index_spin.value()
        transparent_index = fill_index if fill_mode == "transparent" else None
        canvas_size = self._resolve_canvas_size(record)
        preserve_palette_order = True
        slot_order = self._palette_slot_ids or list(range(len(self.palette_colors)))
        slot_map = None
        if logger.isEnabledFor(logging.DEBUG) and self._should_log_process_options_debug(record, debug_context):
            if not (debug_context == "animation-preview" and slot_map is None):
                if slot_map is not None:
                    sample = slot_map[:8]
                    missing = sum(1 for value in slot_map if value < 0)
                else:
                    sample = []
                    missing = 0
                logger.debug(
                    "slot_map record=%s mode=%s context=%s len=%s missing=%s sample=%s",
                    record.path.name,
                    "preserve" if preserve_mode else "detect",
                    debug_context,
                    len(slot_map) if slot_map is not None else 0,
                    missing,
                    sample,
                )
            if slot_map is not None and debug_context != "animation-preview":
                sprite_palette_data = record.indexed_image.getpalette()
                if sprite_palette_data:
                    sprite_colors = []
                    for i in range(min(8, 256)):
                        idx = i * 3
                        if idx + 2 < len(sprite_palette_data):
                            sprite_colors.append((sprite_palette_data[idx], sprite_palette_data[idx+1], sprite_palette_data[idx+2]))
                    logger.debug(f"  sprite palette first 8: {sprite_colors}")
                    logger.debug(f"  unified palette first 8: {self.palette_colors[:8]}")
                    for i in range(min(8, len(slot_map))):
                        src_idx = slot_map[i]
                        if src_idx >= 0 and src_idx < len(sprite_colors):
                            logger.debug(f"  output[{i}] = source[{src_idx}] = {sprite_colors[src_idx]} (should be {self.palette_colors[i]})")
        total_offset_x, total_offset_y = self._resolve_offsets(record)
        
        return ProcessOptions(
            input_path=record.path,
            output_dir=output_dir,
            canvas_size=canvas_size,
            fill_index=fill_index,
            fill_mode=fill_mode,
            transparent_index=transparent_index,
            target_palette=palette,
            preserve_palette_order=preserve_palette_order,
            slot_map=slot_map,
            write_act=write_act,
            offset_x=total_offset_x,
            offset_y=total_offset_y,
        )

    def _should_log_process_options_debug(self, record: SpriteRecord, context: str) -> bool:
        if context != "animation-preview":
            return True
        now = time.perf_counter()
        key = f"{context}:{record.path.as_posix()}"
        next_allowed = self._process_options_debug_next_log_at.get(key, 0.0)
        if now < next_allowed:
            return False
        self._process_options_debug_next_log_at[key] = now + 1.0
        return True

    def _resolve_canvas_size(self, record: SpriteRecord) -> tuple[int, int] | None:
        if self.output_size_mode.currentData() != "custom":
            return None
        scope = self.canvas_scope_combo.currentData() if hasattr(self, "canvas_scope_combo") else "global"
        group = self._palette_groups.get(record.group_id) if record.group_id else None
        if scope == "combined":
            if record.canvas_override_enabled:
                return (max(1, record.canvas_width), max(1, record.canvas_height))
            if group is not None and group.canvas_override_enabled:
                return (max(1, group.canvas_width), max(1, group.canvas_height))
            return (
                max(1, self.canvas_width_spin.value()),
                max(1, self.canvas_height_spin.value()),
            )
        if scope == "group":
            if group is not None:
                return (max(1, group.canvas_width), max(1, group.canvas_height))
        elif scope == "local":
            return (max(1, record.canvas_width), max(1, record.canvas_height))
        return (
            max(1, self.canvas_width_spin.value()),
            max(1, self.canvas_height_spin.value()),
        )

    def _resolve_offsets(self, record: SpriteRecord) -> tuple[int, int]:
        scope = self.offset_scope_combo.currentData() if hasattr(self, "offset_scope_combo") else "combined"
        group = self._palette_groups.get(record.group_id) if record.group_id else None
        if scope == "combined":
            group_x = group.offset_x if group is not None else 0
            group_y = group.offset_y if group is not None else 0
            return (
                self.global_offset_x_spin.value() + group_x + record.offset_x,
                self.global_offset_y_spin.value() + group_y + record.offset_y,
            )
        if scope == "group":
            if group is not None:
                return group.offset_x, group.offset_y
        elif scope == "local":
            return record.offset_x, record.offset_y
        return self.global_offset_x_spin.value(), self.global_offset_y_spin.value()

    def _process_record_image(
        self, record: SpriteRecord, options: ProcessOptions
    ) -> tuple[Image.Image, PaletteInfo]:
        source = record.indexed_image.copy()
        if record.local_overrides:
            self._apply_overrides_to_indexed_image(source, record.local_overrides)
        return process_image_object(source, options)

    def _save_processed_image(
        self,
        record: SpriteRecord,
        processed: Image.Image,
        palette: PaletteInfo,
        options: ProcessOptions,
    ) -> tuple[Path, Path | None]:
        output_dir = options.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (record.path.stem + ".png")
        processed.save(output_path)
        act_path: Path | None = None
        if options.write_act:
            act_path = output_path.with_suffix(".act")
            write_act(act_path, palette)
            logger.debug("Wrote sprite ACT %s -> %s", record.path.name, act_path)
        return output_path, act_path

    def _handle_filter_mode_change(self, index: int) -> None:
        mode = Qt.TransformationMode.FastTransformation if index == 0 else Qt.TransformationMode.SmoothTransformation
        self.preview_panel.set_scaling_mode(mode)

    def _choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select output directory", str(self.output_dir))
        if not directory:
            return
        self.output_dir = Path(directory)
        self.output_dir_label.setText(str(self.output_dir))
        self._update_export_buttons()

    def _export_selected(self) -> None:
        record = self._current_record()
        if record is None:
            self.statusBar().showMessage("Select an image first", 3000)
            return
        self._export_records([record])

    def _export_all(self) -> None:
        records = list(self.sprite_records.values())
        if not records:
            self.statusBar().showMessage("No images loaded", 3000)
            return
        self._export_records(records)

    def _export_records(self, records: List[SpriteRecord]) -> None:
        if not records:
            return
        successes = 0
        failures: List[str] = []
        for record in records:
            options = self._build_process_options(record, output_dir=self.output_dir, write_act=False)
            self._log_process_options("export", options, record)
            try:
                processed, palette = self._process_record_image(record, options)
                output_path, _ = self._save_processed_image(record, processed, palette, options)
                successes += 1
                logger.debug("Exported %s -> %s", record.path.name, output_path)
            except Exception as exc:  # noqa: BLE001 - surfaced via dialog
                failures.append(f"{record.path.name}: {exc}")
        palette_path = None
        if successes:
            palette_path = self._write_global_palette_file(self.output_dir)
        if successes:
            message = f"Exported {successes} image(s) to {self.output_dir}"
            if palette_path:
                message += f" (palette: {palette_path.name})"
            self.statusBar().showMessage(message, 5000)
        if failures:
            QMessageBox.warning(self, "Export errors", "\n".join(failures))

    def _write_global_palette_file(self, output_dir: Path) -> Path | None:
        if not self.palette_colors:
            return None
        act_path = output_dir / "palette.act"
        palette = PaletteInfo(colors=list(self.palette_colors[:256]), transparent_index=None)
        write_act(act_path, palette)
        logger.debug("Wrote palette ACT -> %s", act_path)
        return act_path

    def _export_palette_only(self) -> None:
        """Export only the palette as a .act file without exporting any images."""
        if not self.palette_colors:
            QMessageBox.information(self, "Export Palette", "No palette colors to export.")
            return
        
        palette_path = self._write_global_palette_file(self.output_dir)
        if palette_path:
            QMessageBox.information(
                self,
                "Palette Exported",
                f"Palette exported to:\n{palette_path}"
            )
            self.statusBar().showMessage(f"Palette exported: {palette_path.name}", 3000)

    def _open_floating_palette_window(self) -> None:
        self._floating_palette_windows = [w for w in self._floating_palette_windows if w is not None and w.isVisible()]
        used_indices = {
            int(getattr(window, "_window_index", 0))
            for window in self._floating_palette_windows
            if int(getattr(window, "_window_index", 0)) > 0 and window.isVisible()
        }
        next_index = 1
        while next_index in used_indices:
            next_index += 1
        self._floating_palette_window_counter = next_index
        window = FloatingPaletteWindow(next_index, self)
        self._floating_palette_windows.append(window)

        def _cleanup_closed_window() -> None:
            self._floating_palette_windows = [w for w in self._floating_palette_windows if w is not window]
            logger.debug("Floating palette closed remaining=%s", len(self._floating_palette_windows))

        window.destroyed.connect(_cleanup_closed_window)
        window.show()
        logger.debug(
            "Opened floating palette window index=%s active_windows=%s",
            next_index,
            len(self._floating_palette_windows),
        )

    def _apply_main_palette_layout_options(self) -> None:
        if not hasattr(self, "palette_list"):
            return
        if self._main_palette_layout_in_progress:
            logger.debug("Main palette layout skipped (in progress)")
            return

        self._main_palette_layout_in_progress = True
        try:
            gap = max(-8, int(self._main_palette_gap))
            cols = max(1, int(self._main_palette_columns))
            effective_gap = max(0, gap)
            tightness = max(0, -gap)
            force_columns = bool(self._main_palette_force_columns)
            available_width = self.palette_list.viewport().width()
            signature = (
                force_columns,
                cols,
                int(self._main_palette_zoom),
                gap,
                bool(self._main_palette_show_indices),
                bool(self._main_palette_show_grid),
                int(available_width),
            )
            if signature == self._last_main_palette_layout_signature:
                return
            self._last_main_palette_layout_signature = signature

            if force_columns:
                available_width = max(120, available_width)
                cell = _compute_forced_cell_size(available_width, cols, effective_gap)
                self.palette_list.setSpacing(effective_gap)
                self.palette_list.set_cell_size(cell)
                realized_cols = self.palette_list.measure_first_row_columns()
                if realized_cols and realized_cols != cols:
                    attempts = 0
                    while realized_cols < cols and cell > 16 and attempts < 40:
                        cell -= 1
                        self.palette_list.set_cell_size(cell)
                        realized_cols = self.palette_list.measure_first_row_columns()
                        attempts += 1
                    while realized_cols > cols and cell < 96 and attempts < 80:
                        cell += 1
                        self.palette_list.set_cell_size(cell)
                        new_realized = self.palette_list.measure_first_row_columns()
                        attempts += 1
                        if new_realized < cols:
                            cell -= 1
                            self.palette_list.set_cell_size(cell)
                            realized_cols = self.palette_list.measure_first_row_columns()
                            break
                        realized_cols = new_realized
                    logger.debug(
                        "Main palette exact-fit correction target_cols=%s realized_cols=%s cell=%s attempts=%s",
                        cols,
                        realized_cols,
                        cell,
                        attempts,
                    )
                self._main_palette_zoom = cell
            else:
                available_width = self.palette_list.viewport().width()
                cell = max(16, min(96, int(self._main_palette_zoom)))
            self.palette_list.setSpacing(effective_gap)
            self.palette_list.set_cell_size(cell)
            self.palette_list.set_swatch_inset(max(1, 6 - tightness))
            self.palette_list.set_show_index_labels(bool(self._main_palette_show_indices))
            self.palette_list.set_show_grid_lines(bool(self._main_palette_show_grid))
            self._refresh_main_palette_usage_badges()
            if force_columns:
                self.palette_list.setMinimumWidth(260)
            else:
                width_hint = cols * (cell + effective_gap) + 30
                self.palette_list.setMinimumWidth(max(260, width_hint))
            final_realized_cols = self.palette_list.measure_first_row_columns()
            logger.debug(
                "Main palette layout force_cols=%s cols=%s realized_cols=%s cell=%s gap=%s effective_gap=%s tightness=%s viewport_w=%s show_idx=%s grid=%s",
                force_columns,
                cols,
                final_realized_cols,
                cell,
                gap,
                effective_gap,
                tightness,
                available_width,
                self._main_palette_show_indices,
                self._main_palette_show_grid,
            )
        finally:
            self._main_palette_layout_in_progress = False

    def _open_main_palette_view_settings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Palette View Settings")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)

        row = QHBoxLayout()
        row.addWidget(QLabel("Columns"))
        cols_spin = QSpinBox()
        cols_spin.setRange(1, 32)
        cols_spin.setValue(self._main_palette_columns)
        cols_spin.setFixedWidth(58)
        row.addWidget(cols_spin)
        row.addWidget(QLabel("Zoom"))
        zoom_spin = QSpinBox()
        zoom_spin.setRange(16, 96)
        zoom_spin.setValue(self._main_palette_zoom)
        zoom_spin.setFixedWidth(58)
        row.addWidget(zoom_spin)
        row.addWidget(QLabel("Gap"))
        gap_spin = QSpinBox()
        gap_spin.setRange(-8, 20)
        gap_spin.setValue(self._main_palette_gap)
        gap_spin.setFixedWidth(58)
        row.addWidget(gap_spin)
        layout.addLayout(row)

        toggles = QHBoxLayout()
        show_indices = QCheckBox("Indices")
        show_indices.setChecked(self._main_palette_show_indices)
        toggles.addWidget(show_indices)
        show_grid = QCheckBox("Grid")
        show_grid.setChecked(self._main_palette_show_grid)
        toggles.addWidget(show_grid)
        show_usage = QCheckBox("Usage badge")
        show_usage.setChecked(self._main_palette_show_usage_badge)
        toggles.addWidget(show_usage)
        force_columns_check = QCheckBox("Force palette view columns")
        force_columns_check.setChecked(self._main_palette_force_columns)
        toggles.addWidget(force_columns_check)
        toggles.addStretch(1)
        layout.addLayout(toggles)

        def apply_settings() -> None:
            self._main_palette_columns = cols_spin.value()
            self._main_palette_force_columns = force_columns_check.isChecked()
            self._main_palette_zoom = zoom_spin.value()
            self._main_palette_gap = gap_spin.value()
            self._main_palette_show_indices = show_indices.isChecked()
            self._main_palette_show_grid = show_grid.isChecked()
            self._main_palette_show_usage_badge = show_usage.isChecked()
            zoom_spin.setEnabled(not self._main_palette_force_columns)
            self._set_pref("palette/main_columns", int(self._main_palette_columns))
            self._set_pref("palette/main_force_columns", bool(self._main_palette_force_columns))
            self._set_pref("palette/main_zoom", int(self._main_palette_zoom))
            self._set_pref("palette/main_gap", int(self._main_palette_gap))
            self._set_pref("palette/main_show_indices", bool(self._main_palette_show_indices))
            self._set_pref("palette/main_show_grid", bool(self._main_palette_show_grid))
            self._set_pref("palette/main_show_usage_badge", bool(self._main_palette_show_usage_badge))
            self._apply_main_palette_layout_options()
            if self._main_palette_force_columns:
                zoom_spin.blockSignals(True)
                zoom_spin.setValue(int(self._main_palette_zoom))
                zoom_spin.blockSignals(False)

        cols_spin.valueChanged.connect(lambda _v: apply_settings())
        zoom_spin.valueChanged.connect(lambda _v: apply_settings())
        gap_spin.valueChanged.connect(lambda _v: apply_settings())
        show_indices.toggled.connect(lambda _v: apply_settings())
        show_grid.toggled.connect(lambda _v: apply_settings())
        show_usage.toggled.connect(lambda _v: apply_settings())
        force_columns_check.toggled.connect(lambda _v: apply_settings())

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        apply_settings()
        dialog.exec()

    def _update_export_buttons(self) -> None:
        has_records = bool(self.sprite_records)
        has_palette = bool(self.palette_colors)
        current = self._current_record()
        locked = self._is_loading_sprites
        self.export_selected_btn.setEnabled((current is not None) and not locked)
        self.export_all_btn.setEnabled(has_records and not locked)
        self.export_palette_btn.setEnabled(has_palette and not locked)
        self.output_dir_button.setEnabled(not locked)

    def _update_loaded_count(self) -> None:
        if hasattr(self, "images_panel"):
            self.images_panel.set_loaded_count(len(self.sprite_records))

    def _install_shortcuts(self) -> None:
        self._apply_configured_shortcuts()

    def _apply_configured_shortcuts(self) -> None:
        existing_runtime: Dict[str, List[QShortcut]] = getattr(self, "_runtime_binding_shortcuts", {}) or {}
        for shortcuts in existing_runtime.values():
            for shortcut in shortcuts:
                shortcut.setParent(None)
        self._runtime_binding_shortcuts: Dict[str, List[QShortcut]] = {}

        def _install_shortcut_set(
            binding_key: str,
            default_shortcut: str,
            callback: Callable[[], None],
            *,
            skip_first: bool = False,
        ) -> None:
            created: List[QShortcut] = []
            bindings = self._get_key_bindings(binding_key, default_shortcut)
            if skip_first and bindings:
                bindings = bindings[1:]
            for binding in bindings:
                sequence = str(binding.get("shortcut", "")).strip()
                if not sequence:
                    continue
                is_global = bool(binding.get("global", False))
                context = Qt.ShortcutContext.ApplicationShortcut if is_global else Qt.ShortcutContext.WindowShortcut
                shortcut = QShortcut(QKeySequence(sequence), self)
                shortcut.setContext(context)
                shortcut.activated.connect(callback)
                created.append(shortcut)
            self._runtime_binding_shortcuts[binding_key] = created

        for binding_key, action, default_shortcut in self._action_shortcut_registry:
            bindings = self._get_key_bindings(binding_key, default_shortcut)
            if bindings:
                primary = bindings[0]
                primary_sequence = str(primary.get("shortcut", "")).strip()
                action.setShortcut(QKeySequence(primary_sequence))
                action.setShortcutContext(
                    Qt.ShortcutContext.ApplicationShortcut
                    if bool(primary.get("global", False))
                    else Qt.ShortcutContext.WindowShortcut
                )
            else:
                action.setShortcuts([])
                action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            _install_shortcut_set(binding_key, default_shortcut, action.trigger, skip_first=True)

        undo_default = QKeySequence(QKeySequence.StandardKey.Undo).toString() or "Ctrl+Z"
        redo_default = QKeySequence(QKeySequence.StandardKey.Redo).toString() or "Ctrl+Y"
        _install_shortcut_set("edit/undo", undo_default, self._undo_history)
        _install_shortcut_set("edit/redo", redo_default, self._redo_history)
        _install_shortcut_set(
            "browser/sprites_zoom_in",
            "Ctrl+=",
            lambda: self._sprite_browser_zoom_by(1),
        )
        _install_shortcut_set(
            "browser/sprites_zoom_out",
            "Ctrl+-",
            lambda: self._sprite_browser_zoom_by(-1),
        )
        _install_shortcut_set(
            "browser/sprites_zoom_reset",
            "Ctrl+0",
            self._sprite_browser_zoom_reset,
        )

    def _setup_history_manager(self) -> None:
        manager = HistoryManager()
        manager.register_field("palette", self._history_capture_palette, self._history_apply_palette)
        manager.register_field("fill", self._history_capture_fill, self._history_apply_fill)
        manager.register_field("canvas", self._history_capture_canvas, self._history_apply_canvas)
        manager.register_field("offsets", self._history_capture_offsets, self._history_apply_offsets)
        manager.register_field("animation_tags", self._history_capture_animation_tags, self._history_apply_animation_tags)
        manager.register_field("animation_timeline_range", self._history_capture_animation_timeline_range, self._history_apply_animation_timeline_range)
        self._history_manager = manager

    def _history_capture_animation_tags(self) -> Dict[str, Any]:
        tags_payload = [
            AnimationTag(
                tag_id=tag.tag_id,
                name=tag.name,
                state_label=tag.state_label,
                frames=[
                    AnimationFrameEntry(
                        sprite_key=str(frame.sprite_key),
                        duration_frames=max(1, int(frame.duration_frames)),
                        gap_before_frames=max(0, int(getattr(frame, "gap_before_frames", 0))),
                        notes=str(frame.notes or ""),
                    )
                    for frame in tag.frames
                ],
                notes=tag.notes,
            ).to_dict()
            for tag in sorted(self._animation_tags.values(), key=lambda entry: entry.tag_id)
        ]
        return {"tags": tags_payload}

    def _history_apply_animation_tags(self, payload: Dict[str, Any]) -> None:
        raw_tags = payload.get("tags", [])
        restored: Dict[str, AnimationTag] = {}
        if isinstance(raw_tags, list):
            for item in raw_tags:
                if not isinstance(item, dict):
                    continue
                try:
                    parsed = AnimationTag.from_dict(item)
                except Exception:  # noqa: BLE001
                    continue
                restored[parsed.tag_id] = parsed
        self._animation_tags = restored
        if self._animation_preview_tag_id is not None and self._animation_preview_tag_id not in self._animation_tags:
            self._stop_animation_preview()
        if hasattr(self, "animation_tag_list"):
            self._refresh_animation_tag_list()
        if not self.animation_play_btn.isChecked():
            self._set_animation_playhead_from_selected_frame()

    def _history_capture_animation_timeline_range(self) -> Dict[str, int]:
        return {
            "in_frame": int(self._animation_timeline_in_frame),
            "out_frame": int(self._animation_timeline_out_frame),
        }

    def _history_apply_animation_timeline_range(self, payload: Dict[str, Any]) -> None:
        in_frame = max(0, int(payload.get("in_frame", self._animation_timeline_in_frame)))
        out_frame = max(in_frame + 1, int(payload.get("out_frame", self._animation_timeline_out_frame)))
        self._animation_timeline_in_frame = in_frame
        self._animation_timeline_out_frame = out_frame
        self._sync_animation_timeline_range_controls()
        self._save_animation_timeline_ui_settings()
        if self.animation_play_btn.isChecked():
            pos = max(
                self._animation_timeline_in_frame,
                min(self._animation_timeline_out_frame - 1, int(self._animation_preview_timeline_position)),
            )
            self._seek_animation_playhead(float(pos), from_user_scrub=False)
        else:
            self._set_animation_playhead_from_selected_frame()
        self._refresh_animation_frame_list()

    def _history_capture_palette(self) -> Dict[str, Any]:
        slot_ids = self._palette_slot_ids or list(range(len(self.palette_colors)))
        # Capture sprite pixel data and slot_bindings for proper undo of merge operations
        sprite_data = {}
        for path_str, record in self.sprite_records.items():
            img_copy = record.indexed_image.copy()
            sprite_data[path_str] = {
                "pixels": list(img_copy.getdata()),
                "palette": img_copy.getpalette()[:768] if img_copy.getpalette() else None,
                "slot_bindings": dict(record.slot_bindings),
            }
        group_detect_data = {}
        for group_id, group in self._palette_groups.items():
            group_detect_data[group_id] = {
                "colors": list(group.detect_palette_colors),
                "alphas": list(group.detect_palette_alphas),
                "slots": list(group.detect_palette_slot_ids),
            }
        return {
            "colors": list(self.palette_colors),
            "alphas": list(self.palette_alphas),
            "slots": list(slot_ids),
            "sprites": sprite_data,
            "groups": group_detect_data,
            "detect_cache": {
                "colors": list(self._detect_palette_colors),
                "alphas": list(self._detect_palette_alphas),
                "slots": list(self._detect_palette_slot_ids),
            },
        }

    def _history_apply_palette(self, payload: Dict[str, Any]) -> None:
        colors = list(payload.get("colors", []))
        alphas = list(payload.get("alphas", []))
        slots = list(payload.get("slots", []))
        if not slots or len(slots) != len(colors):
            slots = list(range(len(colors)))
        if len(alphas) != len(colors):
            alphas = [255] * len(colors)
        
        # Restore sprite pixel data and slot_bindings
        sprite_data = payload.get("sprites", {})
        for path_str, data in sprite_data.items():
            record = self.sprite_records.get(path_str)
            if record and record.indexed_image:
                pixels = data.get("pixels")
                palette = data.get("palette")
                slot_bindings = data.get("slot_bindings")
                
                if pixels:
                    record.indexed_image.putdata(pixels)
                if palette:
                    record.indexed_image.putpalette(palette)
                if slot_bindings:
                    record.slot_bindings = dict(slot_bindings)

        for group_id, group_payload in payload.get("groups", {}).items():
            group = self._palette_groups.get(group_id)
            if group is None:
                continue
            group.detect_palette_colors = list(group_payload.get("colors", []))
            group.detect_palette_alphas = list(group_payload.get("alphas", []))
            group.detect_palette_slot_ids = list(group_payload.get("slots", []))

        detect_cache = payload.get("detect_cache", {})
        self._detect_palette_colors = list(detect_cache.get("colors", []))
        self._detect_palette_alphas = list(detect_cache.get("alphas", []))
        self._detect_palette_slot_ids = list(detect_cache.get("slots", []))
        
        self.palette_list.set_colors(colors, slots=slots, alphas=alphas, emit_signal=False)
        self.palette_colors = colors
        self.palette_alphas = alphas
        self._palette_slot_ids = slots
        self._rebuild_slot_color_lookup()
        self._update_fill_preview()

    def _history_capture_fill(self) -> Dict[str, Any]:
        mode = self.fill_mode_combo.currentData() or "palette"
        return {
            "mode": str(mode),
            "index": self.fill_index_spin.value(),
        }

    def _history_apply_fill(self, payload: Dict[str, Any]) -> None:
        mode = str(payload.get("mode", "palette"))
        index = int(payload.get("index", 0))
        self._set_combo_data(self.fill_mode_combo, mode)
        self._set_spin_value(self.fill_index_spin, index)
        self._update_fill_preview()

    def _history_capture_canvas(self) -> Dict[str, Any]:
        mode = self.output_size_mode.currentData() or "native"
        scope = self.canvas_scope_combo.currentData() if hasattr(self, "canvas_scope_combo") else "global"
        group_canvas: Dict[str, Dict[str, Any]] = {}
        for group_id, group in self._palette_groups.items():
            group_canvas[group_id] = {
                "width": int(group.canvas_width),
                "height": int(group.canvas_height),
                "override": bool(group.canvas_override_enabled),
            }
        local_canvas: Dict[str, Dict[str, Any]] = {}
        for path_key, record in self.sprite_records.items():
            local_canvas[path_key] = {
                "width": int(record.canvas_width),
                "height": int(record.canvas_height),
                "override": bool(record.canvas_override_enabled),
            }
        logger.debug(
            "History capture canvas mode=%s scope=%s global=%sx%s groups=%s locals=%s",
            mode,
            scope,
            self.canvas_width_spin.value(),
            self.canvas_height_spin.value(),
            len(group_canvas),
            len(local_canvas),
        )
        return {
            "mode": str(mode),
            "scope": str(scope),
            "width": self.canvas_width_spin.value(),
            "height": self.canvas_height_spin.value(),
            "group_canvas": group_canvas,
            "local_canvas": local_canvas,
        }

    def _history_apply_canvas(self, payload: Dict[str, Any]) -> None:
        mode = str(payload.get("mode", "native"))
        scope = str(payload.get("scope", "global"))
        width = int(payload.get("width", self.canvas_width_spin.minimum()))
        height = int(payload.get("height", self.canvas_height_spin.minimum()))
        group_canvas = payload.get("group_canvas", {}) or {}
        local_canvas = payload.get("local_canvas", {}) or {}
        self._set_combo_data(self.output_size_mode, mode)
        if hasattr(self, "canvas_scope_combo"):
            self._set_combo_data(self.canvas_scope_combo, scope)
        self._set_spin_value(self.canvas_width_spin, width)
        self._set_spin_value(self.canvas_height_spin, height)
        for group_id, group_payload in group_canvas.items():
            group = self._palette_groups.get(group_id)
            if group is None:
                continue
            group.canvas_width = int(group_payload.get("width", group.canvas_width))
            group.canvas_height = int(group_payload.get("height", group.canvas_height))
            group.canvas_override_enabled = bool(group_payload.get("override", group.canvas_override_enabled))
        for path_key, record_payload in local_canvas.items():
            record = self.sprite_records.get(path_key)
            if record is None:
                continue
            record.canvas_width = int(record_payload.get("width", record.canvas_width))
            record.canvas_height = int(record_payload.get("height", record.canvas_height))
            record.canvas_override_enabled = bool(record_payload.get("override", record.canvas_override_enabled))
        self._update_canvas_inputs()
        logger.debug(
            "History apply canvas mode=%s scope=%s global=%sx%s groups=%s locals=%s",
            mode,
            scope,
            width,
            height,
            len(group_canvas),
            len(local_canvas),
        )

    def _history_capture_offsets(self) -> Dict[str, Any]:
        scope = self.offset_scope_combo.currentData() if hasattr(self, "offset_scope_combo") else "combined"
        group_offsets: Dict[str, Dict[str, int]] = {}
        for group_id, group in self._palette_groups.items():
            group_offsets[group_id] = {"x": int(group.offset_x), "y": int(group.offset_y)}
        local_offsets: Dict[str, Dict[str, int]] = {}
        for path_key, record in self.sprite_records.items():
            local_offsets[path_key] = {"x": int(record.offset_x), "y": int(record.offset_y)}
        logger.debug(
            "History capture offsets scope=%s global=(%s,%s) groups=%s locals=%s",
            scope,
            self.global_offset_x_spin.value(),
            self.global_offset_y_spin.value(),
            len(group_offsets),
            len(local_offsets),
        )
        return {
            "scope": str(scope),
            "global_x": self.global_offset_x_spin.value(),
            "global_y": self.global_offset_y_spin.value(),
            "group_offsets": group_offsets,
            "local_offsets": local_offsets,
        }

    def _history_apply_offsets(self, payload: Dict[str, Any]) -> None:
        scope = str(payload.get("scope", "combined"))
        global_x = int(payload.get("global_x", 0))
        global_y = int(payload.get("global_y", 0))
        self._set_spin_value(self.global_offset_x_spin, global_x)
        self._set_spin_value(self.global_offset_y_spin, global_y)
        group_offsets = payload.get("group_offsets", {}) or {}
        for group_id, values in group_offsets.items():
            group = self._palette_groups.get(group_id)
            if group is None:
                continue
            group.offset_x = int(values.get("x", group.offset_x))
            group.offset_y = int(values.get("y", group.offset_y))
        local_offsets = payload.get("local_offsets", {}) or {}
        for path_key, values in local_offsets.items():
            record = self.sprite_records.get(path_key)
            if record is None:
                continue
            record.offset_x = int(values.get("x", record.offset_x))
            record.offset_y = int(values.get("y", record.offset_y))
        if hasattr(self, "offset_scope_combo"):
            self._set_combo_data(self.offset_scope_combo, scope)
        self._update_sprite_offset_controls(self._current_record())
        logger.debug(
            "History apply offsets scope=%s global=(%s,%s) groups=%s locals=%s",
            scope,
            global_x,
            global_y,
            len(group_offsets),
            len(local_offsets),
        )

    def _reset_history(self) -> None:
        self._clear_animation_timeline_range_history_pending()
        if self._history_manager is not None:
            self._history_manager.reset()

    def _record_history(
        self,
        label: str,
        *,
        force: bool = False,
        include_fields: Sequence[str] | None = None,
        mark_dirty: bool = True,
    ) -> None:
        if self._history_manager is not None:
            started = time.perf_counter()
            recorded = self._history_manager.record(
                label,
                force=force,
                include_fields=include_fields,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.debug(
                "Record history label=%s force=%s fields=%s recorded=%s elapsed_ms=%.2f",
                label,
                force,
                list(include_fields) if include_fields is not None else "all",
                recorded,
                elapsed_ms,
            )
            if recorded and mark_dirty:
                self._mark_project_dirty(f"history:{label}")

    def _set_combo_data(self, combo: QComboBox, data: str) -> None:
        blocked = combo.blockSignals(True)
        index = combo.findData(data)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(blocked)

    def _set_spin_value(self, spinner: QSpinBox, value: int) -> None:
        blocked = spinner.blockSignals(True)
        spinner.setValue(value)
        spinner.blockSignals(blocked)

    def _undo_history(self) -> None:
        self._commit_animation_timeline_range_history()
        if self._history_manager and self._history_manager.undo():
            self._sync_palette_model()
            self._sync_local_override_visuals()
            self._update_selected_color_label()
            self._invalidate_preview_cache(reset_zoom=False, reason="history-undo")
            undone_label = self._history_manager.last_undo_label or "change"
            logger.debug("History undo applied; refreshed palette + preview state undone=%s", undone_label)
            self.statusBar().showMessage(f"Undid {undone_label}", 2000)
            self._schedule_preview_update()
            self._update_animation_tag_controls()

    def _redo_history(self) -> None:
        self._commit_animation_timeline_range_history()
        if self._history_manager and self._history_manager.redo():
            self._sync_palette_model()
            self._sync_local_override_visuals()
            self._update_selected_color_label()
            self._invalidate_preview_cache(reset_zoom=False, reason="history-redo")
            redone_label = self._history_manager.last_redo_label or "change"
            logger.debug("History redo applied; refreshed palette + preview state redone=%s", redone_label)
            self.statusBar().showMessage(f"Redid {redone_label}", 2000)
            self._schedule_preview_update()
            self._update_animation_tag_controls()

    def _handle_size_mode_change(self, _index: int) -> None:
        self._update_canvas_inputs()
        self._schedule_preview_update()
        self._record_history("size-mode")

    def _handle_canvas_scope_change(self, _index: int) -> None:
        self._update_canvas_inputs()
        self._schedule_preview_update()
        self._record_history("canvas-scope")

    def _handle_canvas_dimension_change(self, _value: int) -> None:
        self._schedule_preview_update()
        self._record_history("canvas-dimension")

    def _handle_group_canvas_dimension_change(self, _value: int) -> None:
        record = self._current_record()
        group = self._palette_groups.get(record.group_id) if record and record.group_id else None
        if group is None:
            return
        group.canvas_width = self.group_canvas_width_spin.value()
        group.canvas_height = self.group_canvas_height_spin.value()
        group.canvas_override_enabled = True
        self._schedule_preview_update()
        logger.debug(
            "Updated group canvas group=%s size=%sx%s override=%s",
            group.group_id,
            group.canvas_width,
            group.canvas_height,
            group.canvas_override_enabled,
        )
        self._record_history("canvas-group-dimension")

    def _handle_local_canvas_dimension_change(self, _value: int) -> None:
        record = self._current_record()
        if record is None:
            return
        record.canvas_width = self.local_canvas_width_spin.value()
        record.canvas_height = self.local_canvas_height_spin.value()
        record.canvas_override_enabled = True
        self._schedule_preview_update()
        logger.debug(
            "Updated local canvas record=%s size=%sx%s override=%s",
            record.path.name,
            record.canvas_width,
            record.canvas_height,
            record.canvas_override_enabled,
        )
        self._record_history("canvas-local-dimension")

    def _handle_offset_scope_change(self, _index: int) -> None:
        self._update_sprite_offset_controls(self._current_record())
        self._schedule_preview_update()
        self._record_history("offset-scope", include_fields=["offsets"])

    def _handle_global_offset_change(self, _value: int) -> None:
        self._schedule_preview_update()
        self._record_history("offset-global", include_fields=["offsets"])
        logger.debug(
            "Updated global offset -> (%s, %s)",
            self.global_offset_x_spin.value(),
            self.global_offset_y_spin.value(),
        )

    def _handle_fill_mode_change(self, _index: int) -> None:
        self._schedule_preview_update()
        self._record_history("fill-mode")

    def _handle_fill_index_change(self, _value: int) -> None:
        self._update_fill_preview()
        self._schedule_preview_update()
        self._record_history("fill-index")

    def _handle_sprite_offset_change(self, _value: int) -> None:
        """Handle per-sprite offset changes."""
        record = self._current_record()
        if record is None:
            return
        
        record.offset_x = self.sprite_offset_x_spin.value()
        record.offset_y = self.sprite_offset_y_spin.value()
        self._schedule_preview_update()
        self._record_history("offset-local", include_fields=["offsets"])
        logger.debug(f"Updated sprite offset: {record.path.name} -> ({record.offset_x}, {record.offset_y})")

    def _handle_group_offset_change(self, _value: int) -> None:
        record = self._current_record()
        group = self._palette_groups.get(record.group_id) if record and record.group_id else None
        if group is None:
            return
        group.offset_x = self.group_offset_x_spin.value()
        group.offset_y = self.group_offset_y_spin.value()
        self._schedule_preview_update()
        self._record_history("offset-group", include_fields=["offsets"])
        logger.debug(
            "Updated group offset group=%s -> (%s, %s)",
            group.group_id,
            group.offset_x,
            group.offset_y,
        )
    
    def _handle_drag_mode_button_click(self, mode: Literal["global", "group", "individual"]) -> None:
        """Deterministic on/off click behavior for drag mode buttons."""
        button_map = {
            "global": self.global_drag_mode_btn,
            "group": self.group_drag_mode_btn,
            "individual": self.individual_drag_mode_btn,
        }
        target_button = button_map[mode]

        if self._active_offset_drag_mode == mode:
            for button in button_map.values():
                blocked = button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(blocked)
            self._active_offset_drag_mode = "none"
            self._offset_drag_target_key = None
            self._offset_drag_group_id = None
            self.preview_panel.set_drag_mode(False)
            logger.debug("Offset drag mode disabled via click mode=%s", mode)
            return

        for key, button in button_map.items():
            blocked = button.blockSignals(True)
            button.setChecked(key == mode)
            button.blockSignals(blocked)

        self._active_offset_drag_mode = mode
        self._offset_drag_target_key = None
        self._offset_drag_group_id = None
        self.preview_panel.set_drag_mode(True)
        logger.debug("Offset drag mode enabled via click mode=%s", mode)

    def _handle_drag_started(self) -> None:
        self._offset_drag_live = True
        record = self._current_drag_target_record()
        self._offset_drag_target_key = None
        self._offset_drag_group_id = None
        if record is not None:
            for key, candidate in self.sprite_records.items():
                if candidate is record:
                    self._offset_drag_target_key = key
                    break
            if self._active_offset_drag_mode == "group" and record.group_id:
                self._offset_drag_group_id = record.group_id
        logger.debug(
            "Drag started mode=%s locked_key=%s locked_group=%s",
            self._active_offset_drag_mode,
            self._offset_drag_target_key,
            self._offset_drag_group_id,
        )

    def _handle_drag_finished(self, total_dx: int, total_dy: int) -> None:
        was_live = self._offset_drag_live
        self._offset_drag_live = False
        locked_key = self._offset_drag_target_key
        locked_group = self._offset_drag_group_id
        self._offset_drag_target_key = None
        self._offset_drag_group_id = None
        if not was_live:
            return
        if total_dx == 0 and total_dy == 0:
            return
        mode = self._active_offset_drag_mode
        label = {
            "global": "offset-global",
            "group": "offset-group",
            "individual": "offset-local",
        }.get(mode)
        if label is not None:
            self._record_history(label, include_fields=["offsets"])
        if self._is_animation_assist_view_active() and not self.animation_play_btn.isChecked():
            self._refresh_animation_assist_preview_frame()
        else:
            self._schedule_preview_update(delay_ms=0)
        logger.debug(
            "Drag finished mode=%s total_delta=(%s,%s) history_label=%s locked_key=%s locked_group=%s",
            mode,
            total_dx,
            total_dy,
            label,
            locked_key,
            locked_group,
        )
    
    def _handle_drag_offset_changed(self, dx: int, dy: int) -> None:
        """Handle offset changes from viewport dragging."""
        started = time.perf_counter()
        if self._is_loading_sprites:
            return
        if self._active_offset_drag_mode == "none":
            return

        if self._active_offset_drag_mode == "global":
            new_x = self.global_offset_x_spin.value() + dx
            new_y = self.global_offset_y_spin.value() + dy
            bx = self.global_offset_x_spin.blockSignals(True)
            by = self.global_offset_y_spin.blockSignals(True)
            self.global_offset_x_spin.setValue(new_x)
            self.global_offset_y_spin.setValue(new_y)
            self.global_offset_x_spin.blockSignals(bx)
            self.global_offset_y_spin.blockSignals(by)
            self._schedule_preview_update(delay_ms=16 if self._offset_drag_live else 0)
            if not self._offset_drag_live:
                self._record_history("offset-global", include_fields=["offsets"])
            logger.debug("Drag offset global -> (%s, %s) delta=(%s,%s) history=offset-global", new_x, new_y, dx, dy)
            logger.debug("Drag offset commit mode=global elapsed_ms=%.2f", (time.perf_counter() - started) * 1000.0)
            return

        if self._active_offset_drag_mode == "group":
            group: PaletteGroup | None = None
            if self._offset_drag_live and self._offset_drag_group_id:
                group = self._palette_groups.get(self._offset_drag_group_id)
            if group is None:
                record = self._current_drag_target_record()
                group = self._palette_groups.get(record.group_id) if record and record.group_id else None
            if group is None:
                return
            new_x = group.offset_x + dx
            new_y = group.offset_y + dy
            group.offset_x = new_x
            group.offset_y = new_y
            bx = self.group_offset_x_spin.blockSignals(True)
            by = self.group_offset_y_spin.blockSignals(True)
            self.group_offset_x_spin.setValue(new_x)
            self.group_offset_y_spin.setValue(new_y)
            self.group_offset_x_spin.blockSignals(bx)
            self.group_offset_y_spin.blockSignals(by)
            self._schedule_preview_update(delay_ms=16 if self._offset_drag_live else 0)
            if not self._offset_drag_live:
                self._record_history("offset-group", include_fields=["offsets"])
            logger.debug("Drag offset group=%s -> (%s, %s) delta=(%s,%s) history=offset-group", group.group_id, new_x, new_y, dx, dy)
            logger.debug("Drag offset commit mode=group elapsed_ms=%.2f", (time.perf_counter() - started) * 1000.0)
            return

        record = self._current_drag_target_record()
        if record is None:
            return
        
        # Update per-sprite offset
        record.offset_x += dx
        record.offset_y += dy
        
        # Update spin boxes (will trigger preview update)
        self.sprite_offset_x_spin.blockSignals(True)
        self.sprite_offset_y_spin.blockSignals(True)
        self.sprite_offset_x_spin.setValue(record.offset_x)
        self.sprite_offset_y_spin.setValue(record.offset_y)
        self.sprite_offset_x_spin.blockSignals(False)
        self.sprite_offset_y_spin.blockSignals(False)
        
        self._schedule_preview_update(delay_ms=16 if self._offset_drag_live else 0)
        if not self._offset_drag_live:
            self._record_history("offset-local", include_fields=["offsets"])
        logger.debug(
            "Drag offset individual record=%s -> (%s, %s) delta=(%s,%s) history=offset-local",
            record.path.name,
            record.offset_x,
            record.offset_y,
            dx,
            dy,
        )
        logger.debug("Drag offset commit mode=individual elapsed_ms=%.2f", (time.perf_counter() - started) * 1000.0)

    def _update_canvas_inputs(self) -> None:
        if not hasattr(self, "canvas_width_spin"):
            return
        mode = self.output_size_mode.currentData()
        if not hasattr(self, "canvas_scope_combo") or not hasattr(self, "group_canvas_width_spin") or not hasattr(self, "local_canvas_width_spin"):
            enabled = mode == "custom"
            self.canvas_width_spin.setEnabled(enabled)
            self.canvas_height_spin.setEnabled(enabled)
            return
        is_custom = mode == "custom"
        scope = self.canvas_scope_combo.currentData() if hasattr(self, "canvas_scope_combo") else "global"
        record = self._current_record()
        group = self._palette_groups.get(record.group_id) if record and record.group_id else None
        show_global = scope in ("combined", "global")
        show_group = scope in ("combined", "group")
        show_local = scope in ("combined", "local")
        if hasattr(self, "canvas_global_widget"):
            self.canvas_global_widget.setVisible(show_global)
            if hasattr(self, "_size_layout"):
                label = self._size_layout.labelForField(self.canvas_global_widget)
                if label is not None:
                    label.setVisible(show_global)
        if hasattr(self, "canvas_group_widget"):
            self.canvas_group_widget.setVisible(show_group)
            if hasattr(self, "_size_layout"):
                label = self._size_layout.labelForField(self.canvas_group_widget)
                if label is not None:
                    label.setVisible(show_group)
        if hasattr(self, "canvas_local_widget"):
            self.canvas_local_widget.setVisible(show_local)
            if hasattr(self, "_size_layout"):
                label = self._size_layout.labelForField(self.canvas_local_widget)
                if label is not None:
                    label.setVisible(show_local)

        self.canvas_width_spin.setEnabled(is_custom and show_global)
        self.canvas_height_spin.setEnabled(is_custom and show_global)
        self.group_canvas_width_spin.setEnabled(is_custom and show_group and group is not None)
        self.group_canvas_height_spin.setEnabled(is_custom and show_group and group is not None)
        self.local_canvas_width_spin.setEnabled(is_custom and show_local and record is not None)
        self.local_canvas_height_spin.setEnabled(is_custom and show_local and record is not None)

        if record is not None:
            self.local_canvas_width_spin.blockSignals(True)
            self.local_canvas_height_spin.blockSignals(True)
            self.local_canvas_width_spin.setValue(record.canvas_width)
            self.local_canvas_height_spin.setValue(record.canvas_height)
            self.local_canvas_width_spin.blockSignals(False)
            self.local_canvas_height_spin.blockSignals(False)
        if group is not None:
            self.group_canvas_width_spin.blockSignals(True)
            self.group_canvas_height_spin.blockSignals(True)
            self.group_canvas_width_spin.setValue(group.canvas_width)
            self.group_canvas_height_spin.setValue(group.canvas_height)
            self.group_canvas_width_spin.blockSignals(False)
            self.group_canvas_height_spin.blockSignals(False)

    def _update_fill_preview(self) -> None:
        if not hasattr(self, "fill_preview_label"):
            return
        index = self.fill_index_spin.value()
        color = self.palette_colors[index] if 0 <= index < len(self.palette_colors) else None
        if color:
            self.fill_preview_label.setText(f"{index}: #{color[0]:02X}{color[1]:02X}{color[2]:02X}")
        else:
            self.fill_preview_label.setText(f"{index}: —")
        self._update_background_indices_button_text()

    def _refresh_main_palette_usage_badges(self) -> None:
        if not hasattr(self, "palette_list"):
            return
        if self._main_palette_show_usage_badge:
            counts = self._usage_counts_for_records(list(self.sprite_records.values()))
            self.palette_list.set_usage_counts(counts, show_badge=True)
        else:
            self.palette_list.set_usage_counts({}, show_badge=False)

    def _update_background_indices_button_text(self) -> None:
        if not hasattr(self, "background_indices_btn"):
            return
        count = len(self._preview_background_indices)
        if count == 0:
            self.background_indices_btn.setText("Background Indexes...")
        else:
            self.background_indices_btn.setText(f"Background Indexes ({count})")

    def _open_background_index_selector(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Background Indexes")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)

        hint = QLabel("Select palette indexes used as background in preview options.")
        layout.addWidget(hint)

        palette_view = PaletteListView(dialog)
        palette_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        palette_view.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        palette_view.setDragEnabled(False)
        palette_view.setAcceptDrops(False)
        palette_view.setSpacing(max(0, self._main_palette_gap))
        palette_view.set_cell_size(max(16, min(96, self._main_palette_zoom)))
        palette_view.set_show_index_labels(bool(self._main_palette_show_indices))
        palette_view.set_show_grid_lines(bool(self._main_palette_show_grid))
        palette_view.set_usage_counts({}, show_badge=bool(self._main_palette_show_usage_badge))
        palette_view.set_colors(
            self.palette_colors,
            slots=self._palette_slot_ids,
            alphas=self.palette_alphas,
            emit_signal=False,
        )
        layout.addWidget(palette_view, 1)

        selection = palette_view.selectionModel()
        if selection is not None:
            for row in sorted(self._preview_background_indices):
                if 0 <= row < 256:
                    index = palette_view.model_obj.index(row)
                    if index.isValid():
                        selection.select(index, QItemSelectionModel.SelectionFlag.Select)

        action_row = QHBoxLayout()
        select_none_btn = QPushButton("Clear")
        select_fill_btn = QPushButton("Use Fill Index")
        set_selected_btn = QPushButton("Set Selected")
        clear_selected_btn = QPushButton("Clear Selected")
        action_row.addWidget(select_none_btn)
        action_row.addWidget(select_fill_btn)
        action_row.addWidget(set_selected_btn)
        action_row.addWidget(clear_selected_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        def clear_checks() -> None:
            palette_view.clearSelection()

        def apply_fill_index() -> None:
            fill_index = self.fill_index_spin.value()
            if not (0 <= fill_index < 256):
                return
            index = palette_view.model_obj.index(fill_index)
            if not index.isValid() or palette_view.selectionModel() is None:
                return
            palette_view.clearSelection()
            palette_view.selectionModel().select(index, QItemSelectionModel.SelectionFlag.Select)
            palette_view.setCurrentIndex(index)

        def set_selected() -> None:
            selected_rows = {
                idx.row()
                for idx in palette_view.selectedIndexes()
                if idx.isValid() and 0 <= idx.row() < 256
            }
            if not selected_rows:
                return
            self._preview_background_indices |= selected_rows
            if palette_view.selectionModel() is not None:
                palette_view.clearSelection()
                for row in sorted(self._preview_background_indices):
                    index = palette_view.model_obj.index(row)
                    if index.isValid():
                        palette_view.selectionModel().select(index, QItemSelectionModel.SelectionFlag.Select)

        def clear_selected() -> None:
            selected_rows = {
                idx.row()
                for idx in palette_view.selectedIndexes()
                if idx.isValid() and 0 <= idx.row() < 256
            }
            if not selected_rows:
                return
            self._preview_background_indices -= selected_rows
            if palette_view.selectionModel() is not None:
                palette_view.clearSelection()
                for row in sorted(self._preview_background_indices):
                    index = palette_view.model_obj.index(row)
                    if index.isValid():
                        palette_view.selectionModel().select(index, QItemSelectionModel.SelectionFlag.Select)

        select_none_btn.clicked.connect(clear_checks)
        select_fill_btn.clicked.connect(apply_fill_index)
        set_selected_btn.clicked.connect(set_selected)
        clear_selected_btn.clicked.connect(clear_selected)

        buttons = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        buttons.addStretch(1)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected = {
            index.row()
            for index in palette_view.selectedIndexes()
            if index.isValid() and 0 <= index.row() <= 255
        }
        self._preview_background_indices = {int(value) for value in selected}
        self._update_background_indices_button_text()
        self._save_preview_ui_settings()
        self._preview_base_pixmap = None
        self._invalidate_animation_preview_pixmap_cache(refresh_current_frame=True)
        self._update_preview_pixmap()

    def _resolve_drag_backdrop_rgba(self) -> tuple[int, int, int, int] | None:
        fill_mode = self._selected_fill_mode()
        if fill_mode == "transparent":
            return None
        fill_index = self.fill_index_spin.value()
        if self._preview_bg_transparent_enabled and fill_index in self._preview_background_indices:
            return None
        if fill_index < 0 or fill_index >= len(self.palette_colors):
            return None
        color = self.palette_colors[fill_index]
        alpha = self.palette_alphas[fill_index] if 0 <= fill_index < len(self.palette_alphas) else 255
        alpha = max(0, min(255, int(alpha)))
        if alpha < 255:
            return None
        return (int(color[0]), int(color[1]), int(color[2]), alpha)

    def _selected_fill_mode(self) -> Literal["palette", "transparent"]:
        mode = self.fill_mode_combo.currentData()
        return "transparent" if mode == "transparent" else "palette"

    def _export_canvas_size(self) -> tuple[int, int] | None:
        if self.output_size_mode.currentData() != "custom":
            return None
        return (
            max(1, self.canvas_width_spin.value()),
            max(1, self.canvas_height_spin.value()),
        )

    def _log_process_options(self, label: str, options: ProcessOptions, record: SpriteRecord | None = None) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        palette_preview = []
        if options.target_palette:
            preview_len = min(4, len(options.target_palette))
            palette_preview = [options.target_palette[i] for i in range(preview_len)]
        logger.debug(
            "%s options path=%s canvas=%s fill_mode=%s fill_index=%s preserve=%s palette_len=%s palette_sample=%s",
            label,
            record.path if record else None,
            options.canvas_size,
            options.fill_mode,
            options.fill_index,
            options.preserve_palette_order,
            len(options.target_palette) if options.target_palette else 0,
            palette_preview,
        )

    def _invalidate_preview_cache(self, reset_zoom: bool, reason: str = "unspecified") -> None:
        self._preview_request_serial += 1
        self._last_processed_indexed = None
        self._last_index_data = None
        self._overlay_region_by_index = {}
        self._overlay_region_cache_shape = None
        self._overlay_region_cache_token = None
        self._overlay_region_multi_cache.clear()
        self._last_preview_rgba = None
        self._preview_base_pixmap = None
        self._last_preview_render_signature = None
        self._last_palette_info = None
        self._palette_index_lookup = {}
        self._animation_preview_pixmap_cache.clear()
        self._animation_preview_index_cache.clear()
        self._animation_assist_layer_cache.clear()
        if reset_zoom:
            self._reset_zoom_next = True
        logger.debug("Preview cache invalidated reset_zoom=%s reason=%s", reset_zoom, reason)

    def _handle_preview_panning_changed(self, active: bool) -> None:
        self._preview_pan_active = bool(active)
        if self._preview_pan_active:
            self._last_pan_overlay_refresh_ts = 0.0
        if not self._preview_pan_active and self._preview_highlight_targets():
            self._request_preview_pixmap_update(0)

    def _request_preview_pixmap_update(self, delay_ms: int = 8) -> None:
        delay = max(0, int(delay_ms))
        if self._preview_pixmap_timer.isActive():
            remaining = self._preview_pixmap_timer.remainingTime()
            if remaining >= 0 and remaining <= delay:
                return
            self._preview_pixmap_timer.stop()
        self._preview_pixmap_timer.start(delay)

    def _flush_preview_pixmap_update(self) -> None:
        if self._is_animation_assist_view_active():
            if self.animation_play_btn.isChecked():
                # During playback, frame advancement owns redraw cadence.
                # Palette/highlight interactions use explicit one-shot refreshes.
                return
            else:
                self._refresh_animation_assist_preview_frame()
            return
        self._update_preview_pixmap()

    def _refresh_assist_visible_frame(self) -> bool:
        if not self._is_animation_assist_view_active() or not self.animation_play_btn.isChecked():
            return False
        if self._assist_interaction_refresh_pending:
            return True
        self._assist_interaction_refresh_pending = True
        interval_ms = max(1, int(self._animation_preview_step_interval_ms())) if self.animation_play_btn.isChecked() else 1
        self._assist_interaction_refresh_timer.start(interval_ms)
        return True

    def _flush_assist_interaction_refresh(self) -> None:
        self._assist_interaction_refresh_pending = False
        if not self._is_animation_assist_view_active() or not self.animation_play_btn.isChecked():
            return
        key = self._animation_preview_visible_sprite_key
        if key:
            self._set_animation_preview_frame_by_key(key, source="interaction")
            return
        logger.debug("Assist visible-frame refresh requested while playing but visible key missing")
        self._refresh_animation_assist_preview_frame()

    def _pan_pulse_refresh_interval(self) -> float:
        interval = self._pan_overlay_refresh_interval_s
        if self._last_processed_indexed is None:
            return interval

        width, height = self._last_processed_indexed.size
        zoom = max(0.1, float(self.preview_panel.current_zoom())) if hasattr(self, "preview_panel") else 1.0
        effective_area = int(width * zoom) * int(height * zoom)
        if effective_area <= 256 * 256:
            interval = 0.02
        elif effective_area <= 512 * 512:
            interval = 0.033
        elif effective_area <= 1024 * 1024:
            interval = 0.05
        elif effective_area <= 2048 * 2048:
            interval = 0.066
        else:
            interval = 0.1

        if zoom >= 8.0:
            interval = max(interval, 0.12)
        elif zoom >= 6.0:
            interval = max(interval, 0.085)

        if self._overlay_compose_ms_ema > 0.0:
            # Keep pulse alive but bound redraw pressure from real measured compose cost.
            # Rough target: refresh no faster than ~2x measured compose duration.
            interval = max(interval, min(0.2, (self._overlay_compose_ms_ema * 2.0) / 1000.0))
        return interval

    def _overlay_region_for_index(
        self,
        target_index: int,
        width: int,
        height: int,
        index_data: Sequence[int],
        *,
        source_token: Any = None,
    ) -> QRegion:
        if target_index < 0 or width <= 0 or height <= 0:
            return QRegion()

        if source_token is not None:
            multi_key = (source_token, int(target_index), int(width), int(height))
            cached_multi = self._overlay_region_multi_cache.get(multi_key)
            if cached_multi is not None:
                self._overlay_region_multi_cache.move_to_end(multi_key)
                return cached_multi

        shape = (int(width), int(height))
        if self._overlay_region_cache_shape != shape or self._overlay_region_cache_token != source_token:
            self._overlay_region_by_index = {}
            self._overlay_region_cache_shape = shape
            self._overlay_region_cache_token = source_token

        cached = self._overlay_region_by_index.get(target_index)
        if cached is not None:
            return cached

        total = min(len(index_data), width * height)
        region = QRegion()
        for y in range(height):
            row_start = y * width
            if row_start >= total:
                break
            row_end = min(total, row_start + width)
            x = row_start
            while x < row_end:
                if index_data[x] != target_index:
                    x += 1
                    continue
                run_start = x
                x += 1
                while x < row_end and index_data[x] == target_index:
                    x += 1
                run_len = x - run_start
                if run_len > 0:
                    region = region.united(QRegion(run_start - row_start, y, run_len, 1))

        self._overlay_region_by_index[target_index] = region
        if source_token is not None:
            self._overlay_region_multi_cache[(source_token, int(target_index), int(width), int(height))] = region
            if len(self._overlay_region_multi_cache) > 4096:
                self._overlay_region_multi_cache.popitem(last=False)
        return region

    def _clear_preview_cache(self) -> None:
        self._invalidate_preview_cache(reset_zoom=True, reason="full-clear")

    def _on_palette_selection_changed(self, *_args) -> None:
        # Start/stop fade animation based on selection
        selected_indexes = self.palette_list.selectedIndexes()
        logger.debug(
            "Palette selection changed selected=%s selected_highlight=%s hover_highlight=%s",
            len(selected_indexes),
            self.highlight_checkbox.isChecked(),
            self.hover_highlight_checkbox.isChecked(),
        )
        self._update_preview_highlight_animation_state()
        self._last_overlay_alpha_signature = None
        if not self._refresh_assist_visible_frame():
            self._request_preview_pixmap_update(0)
        self._update_selected_color_label()
        self._update_merge_button_state()
        self._highlight_sprites_for_palette_index(self._current_selected_palette_index())

    def _on_highlight_checkbox_toggled(self, checked: bool) -> None:
        self._set_pref("preview/highlight_enabled", bool(checked))
        self._update_preview_highlight_animation_state()
        self._last_overlay_alpha_signature = None
        self._save_preview_ui_settings()
        if not self._refresh_assist_visible_frame():
            self._request_preview_pixmap_update(0)

    def _on_hover_highlight_checkbox_toggled(self, checked: bool) -> None:
        self._set_pref("preview/highlight_hover_enabled", bool(checked))
        self._update_preview_highlight_animation_state()
        self._last_overlay_alpha_signature = None
        self._save_preview_ui_settings()
        if not self._refresh_assist_visible_frame():
            self._request_preview_pixmap_update(0)

    def _set_preview_hover_palette_row(self, row: int | None) -> None:
        if self._preview_hover_palette_row == row:
            return
        self._preview_hover_palette_row = row
        self._update_preview_highlight_animation_state()
        self._last_overlay_alpha_signature = None
        if not self._refresh_assist_visible_frame():
            self._request_preview_pixmap_update(0)

    def _update_preview_highlight_animation_state(self) -> None:
        if self._preview_highlight_targets():
            if not self._highlight_animation_timer.isActive():
                self._selected_highlight_animation_phase = 0.0
                self._hover_highlight_animation_phase = 0.0
                self._highlight_animation_phase = 0.0
                self._last_overlay_alpha_signature = None
                self._overlay_timer_interval_ms = self._target_overlay_timer_interval_ms()
                self._highlight_animation_timer.setInterval(self._overlay_timer_interval_ms)
                self._highlight_animation_timer.start()
        else:
            self._highlight_animation_timer.stop()
            self._last_overlay_alpha_signature = None

    def _overlay_alpha_signature(self) -> tuple[Any, ...]:
        import math

        targets = self._preview_highlight_targets()
        if not targets:
            return ()

        signature: List[tuple[str, int, int]] = []
        quantum = max(1, int(self._overlay_alpha_quantum))
        for mode, index, _color in targets:
            if mode == "hover":
                alpha_min = self._hover_overlay_alpha_min
                alpha_max = self._hover_overlay_alpha_max
                phase = self._hover_highlight_animation_phase
            else:
                alpha_min = self._selected_overlay_alpha_min
                alpha_max = self._selected_overlay_alpha_max
                phase = self._selected_highlight_animation_phase

            fade = (math.sin(phase) + 1) / 2
            alpha_range = max(0, int(alpha_max) - int(alpha_min))
            animated_alpha = int(int(alpha_min) + alpha_range * fade)
            animated_alpha = int(round(animated_alpha / quantum) * quantum)
            animated_alpha = max(0, min(255, animated_alpha))
            signature.append((str(mode), int(index), int(animated_alpha)))

        return tuple(signature)

    def _target_overlay_timer_interval_ms(self) -> int:
        base_ms = 30
        if self._overlay_compose_ms_ema > 0.0:
            base_ms = max(base_ms, int(self._overlay_compose_ms_ema * 1.35))

        zoom = max(0.1, float(self.preview_panel.current_zoom())) if hasattr(self, "preview_panel") else 1.0
        if zoom >= 8.0:
            base_ms = max(base_ms, 80)
        elif zoom >= 6.0:
            base_ms = max(base_ms, 66)
        elif zoom >= 4.0:
            base_ms = max(base_ms, 50)

        if self._preview_pan_active and zoom >= 8.0:
            base_ms = max(base_ms, 90)
        elif self._preview_pan_active and zoom >= 6.0:
            base_ms = max(base_ms, 75)
        elif self._preview_pan_active and zoom >= 4.0:
            base_ms = max(base_ms, 60)

        if self._overlay_show_both and zoom >= 6.0:
            base_ms = max(base_ms, 85)

        if self._preview_pan_active:
            base_ms = max(base_ms, int(self._pan_pulse_refresh_interval() * 1000.0))

        return max(20, min(120, int(base_ms)))
    
    def _update_highlight_animation(self) -> None:
        """Update smooth fade animation for highlight."""
        import math
        self._selected_highlight_animation_phase += self._selected_animation_speed
        if self._selected_highlight_animation_phase > 2 * math.pi:
            self._selected_highlight_animation_phase -= 2 * math.pi
        self._hover_highlight_animation_phase += self._hover_animation_speed
        if self._hover_highlight_animation_phase > 2 * math.pi:
            self._hover_highlight_animation_phase -= 2 * math.pi
        self._highlight_animation_phase = self._selected_highlight_animation_phase

        new_interval = self._target_overlay_timer_interval_ms()
        if new_interval != self._overlay_timer_interval_ms:
            self._overlay_timer_interval_ms = new_interval
            self._highlight_animation_timer.setInterval(new_interval)

        alpha_signature = self._overlay_alpha_signature()
        if alpha_signature == self._last_overlay_alpha_signature:
            return
        self._last_overlay_alpha_signature = alpha_signature

        if self._is_animation_assist_view_active() and self.animation_play_btn.isChecked():
            # Avoid fighting playback timer with extra overlay pulse renders.
            return

        if self._preview_pan_active:
            now = time.perf_counter()
            if now - self._last_pan_overlay_refresh_ts >= self._pan_pulse_refresh_interval():
                self._last_pan_overlay_refresh_ts = now
                self._request_preview_pixmap_update(0)
            return
        self._request_preview_pixmap_update(0)

    def _preview_highlight_targets(self) -> List[tuple[Literal["hover", "selected"], int, ColorTuple]]:
        model = self.palette_list.model_obj

        hover_target: tuple[Literal["hover", "selected"], int, ColorTuple] | None = None
        if self.hover_highlight_checkbox.isChecked() and self._preview_hover_palette_row is not None:
            hover_row = int(self._preview_hover_palette_row)
            if 0 <= hover_row < 256:
                hover_color = model.colors[hover_row]
                if hover_color is not None:
                    actual_hover = self.palette_colors[hover_row] if hover_row < len(self.palette_colors) else hover_color
                    hover_target = ("hover", hover_row, actual_hover)

        selected_target: tuple[Literal["hover", "selected"], int, ColorTuple] | None = None
        if self.highlight_checkbox.isChecked():
            index = self.palette_list.currentIndex()
            if index.isValid() and 0 <= index.row() < 256:
                slot_index = index.row()
                color = model.colors[slot_index]
                if color is not None:
                    actual_color = self.palette_colors[slot_index] if slot_index < len(self.palette_colors) else color
                    selected_target = ("selected", slot_index, actual_color)

        if self._overlay_show_both:
            targets: List[tuple[Literal["hover", "selected"], int, ColorTuple]] = []
            if hover_target is not None:
                targets.append(hover_target)
            if selected_target is not None and (
                hover_target is None or selected_target[1] != hover_target[1]
            ):
                targets.append(selected_target)
            return targets

        if hover_target is not None:
            return [hover_target]
        if selected_target is not None:
            return [selected_target]
        return []

    def _build_overlay_layers_from_index_data(
        self,
        width: int,
        height: int,
        index_data: Sequence[int],
        *,
        include_dragging: bool = False,
        source_token: Any = None,
    ) -> tuple[List[tuple[int, QRegion, QColor]], tuple[tuple[int, int, int, int, int], ...]]:
        targets: List[tuple[Literal["hover", "selected"], int, ColorTuple]] = []
        if include_dragging or not self.preview_panel.is_dragging():
            targets = self._preview_highlight_targets()
        if not targets:
            return [], ()

        import math

        overlay_layers: List[tuple[int, QRegion, QColor]] = []
        layer_signature: List[tuple[int, int, int, int, int]] = []
        for mode, index, _color in targets:
            if mode == "hover":
                overlay_rgb = self._hover_overlay_color
                alpha_min = self._hover_overlay_alpha_min
                alpha_max = self._hover_overlay_alpha_max
                phase = self._hover_highlight_animation_phase
            else:
                overlay_rgb = self._selected_overlay_color
                alpha_min = self._selected_overlay_alpha_min
                alpha_max = self._selected_overlay_alpha_max
                phase = self._selected_highlight_animation_phase

            fade = (math.sin(phase) + 1) / 2
            alpha_range = max(0, int(alpha_max) - int(alpha_min))
            animated_alpha = int(int(alpha_min) + alpha_range * fade)
            quantum = max(1, int(self._overlay_alpha_quantum))
            animated_alpha = int(round(animated_alpha / quantum) * quantum)
            animated_alpha = max(0, min(255, animated_alpha))

            overlay_qcolor = QColor(int(overlay_rgb[0]), int(overlay_rgb[1]), int(overlay_rgb[2]), animated_alpha)
            region = self._overlay_region_for_index(
                index,
                int(width),
                int(height),
                index_data,
                source_token=source_token,
            )
            if region.isEmpty():
                continue
            layer_key = int(index) if mode == "selected" else int(1000 + index)
            overlay_layers.append((layer_key, region, overlay_qcolor))
            layer_signature.append((layer_key, int(overlay_rgb[0]), int(overlay_rgb[1]), int(overlay_rgb[2]), int(animated_alpha)))

        return overlay_layers, tuple(layer_signature)

    def _animation_assist_overlay_layers_for_key(
        self,
        key: str,
    ) -> tuple[List[tuple[int, QRegion, QColor]], QSize, tuple[Any, ...]]:
        record = self.sprite_records.get(key)
        if record is None:
            return [], QSize(0, 0), ()
        cache_key = self._animation_preview_cache_key(record)
        cached = self._animation_preview_index_cache.get(cache_key)
        if cached is None:
            _ = self._get_animation_preview_pixmap(record)
            cached = self._animation_preview_index_cache.get(cache_key)
        if cached is None:
            return [], QSize(0, 0), ()
        width, height, index_bytes = cached
        self._animation_preview_index_cache.move_to_end(cache_key)
        layers, signature = self._build_overlay_layers_from_index_data(
            int(width),
            int(height),
            index_bytes,
            include_dragging=False,
            source_token=("assist", cache_key),
        )
        return layers, QSize(int(width), int(height)), (key, signature)
    
    
    def _open_overlay_settings(self) -> None:
        """Open overlay settings dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Overlay Settings")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)

        def build_overlay_group(
            title: str,
            color_attr: str,
            min_attr: str,
            max_attr: str,
            speed_attr: str,
        ) -> QGroupBox:
            group = QGroupBox(title)
            group_layout = QVBoxLayout(group)

            color_layout = QHBoxLayout()
            color_layout.addWidget(QLabel("Overlay Color:"))
            color_button = QPushButton()
            color_button.setFixedSize(60, 30)

            def _refresh_color_button() -> None:
                current_color = getattr(self, color_attr)
                color_button.setStyleSheet(
                    f"background-color: rgb({current_color[0]}, {current_color[1]}, {current_color[2]}); border: 1px solid #888;"
                )

            def _choose_color() -> None:
                current_color = getattr(self, color_attr)
                result = QColorDialog.getColor(QColor(*current_color), dialog, f"Choose {title} Color")
                if not result.isValid():
                    return
                setattr(self, color_attr, (result.red(), result.green(), result.blue()))
                if color_attr == "_selected_overlay_color":
                    self._overlay_color = self._selected_overlay_color
                _refresh_color_button()
                self._save_preview_ui_settings()
                self._update_preview_pixmap()

            _refresh_color_button()
            color_button.clicked.connect(_choose_color)
            color_layout.addWidget(color_button)
            color_layout.addStretch(1)
            group_layout.addLayout(color_layout)

            min_layout = QHBoxLayout()
            min_layout.addWidget(QLabel("Min Opacity:"))
            min_slider = QSlider(Qt.Orientation.Horizontal)
            min_slider.setRange(0, 255)
            min_slider.setValue(int(getattr(self, min_attr)))
            min_label = QLabel(f"{int(getattr(self, min_attr) / 2.55)}%")

            def _update_min(value: int) -> None:
                max_value = int(getattr(self, max_attr))
                if value > max_value:
                    value = max_value
                    min_slider.setValue(value)
                setattr(self, min_attr, int(value))
                min_label.setText(f"{int(value / 2.55)}%")
                if min_attr == "_selected_overlay_alpha_min":
                    self._overlay_alpha_min = self._selected_overlay_alpha_min
                self._save_preview_ui_settings()
                self._update_preview_pixmap()

            min_slider.valueChanged.connect(_update_min)
            min_layout.addWidget(min_slider)
            min_layout.addWidget(min_label)
            group_layout.addLayout(min_layout)

            max_layout = QHBoxLayout()
            max_layout.addWidget(QLabel("Max Opacity:"))
            max_slider = QSlider(Qt.Orientation.Horizontal)
            max_slider.setRange(0, 255)
            max_slider.setValue(int(getattr(self, max_attr)))
            max_label = QLabel(f"{int(getattr(self, max_attr) / 2.55)}%")

            def _update_max(value: int) -> None:
                min_value = int(getattr(self, min_attr))
                if value < min_value:
                    value = min_value
                    max_slider.setValue(value)
                setattr(self, max_attr, int(value))
                max_label.setText(f"{int(value / 2.55)}%")
                if max_attr == "_selected_overlay_alpha_max":
                    self._overlay_alpha_max = self._selected_overlay_alpha_max
                self._save_preview_ui_settings()
                self._update_preview_pixmap()

            max_slider.valueChanged.connect(_update_max)
            max_layout.addWidget(max_slider)
            max_layout.addWidget(max_label)
            group_layout.addLayout(max_layout)

            speed_layout = QHBoxLayout()
            speed_layout.addWidget(QLabel("Animation Speed:"))
            speed_slider = QSlider(Qt.Orientation.Horizontal)
            speed_slider.setRange(1, 50)
            speed_slider.setValue(int(float(getattr(self, speed_attr)) * 100))
            speed_label = QLabel(f"{float(getattr(self, speed_attr)):.2f}x")

            def _update_speed(value: int) -> None:
                speed = max(0.01, min(0.50, value / 100.0))
                setattr(self, speed_attr, float(speed))
                if speed_attr == "_selected_animation_speed":
                    self._animation_speed = self._selected_animation_speed
                speed_label.setText(f"{speed:.2f}x")
                self._save_preview_ui_settings()

            speed_slider.valueChanged.connect(_update_speed)
            speed_layout.addWidget(speed_slider)
            speed_layout.addWidget(speed_label)
            group_layout.addLayout(speed_layout)

            return group

        layout.addWidget(
            build_overlay_group(
                "Selected Overlay",
                "_selected_overlay_color",
                "_selected_overlay_alpha_min",
                "_selected_overlay_alpha_max",
                "_selected_animation_speed",
            )
        )
        layout.addWidget(
            build_overlay_group(
                "Hover Overlay",
                "_hover_overlay_color",
                "_hover_overlay_alpha_min",
                "_hover_overlay_alpha_max",
                "_hover_animation_speed",
            )
        )

        highlight_row = QHBoxLayout()
        selected_toggle = QCheckBox("Highlight selected")
        selected_toggle.setChecked(self.highlight_checkbox.isChecked())
        hover_toggle = QCheckBox("Highlight hover")
        hover_toggle.setChecked(self.hover_highlight_checkbox.isChecked())

        def update_selected_toggle(checked: bool) -> None:
            self.highlight_checkbox.setChecked(bool(checked))

        def update_hover_toggle(checked: bool) -> None:
            self.hover_highlight_checkbox.setChecked(bool(checked))

        selected_toggle.toggled.connect(update_selected_toggle)
        hover_toggle.toggled.connect(update_hover_toggle)
        highlight_row.addWidget(selected_toggle)
        highlight_row.addWidget(hover_toggle)
        highlight_row.addStretch(1)
        layout.addLayout(highlight_row)

        show_both_toggle = QCheckBox("Show both overlays simultaneously")
        show_both_toggle.setChecked(bool(self._overlay_show_both))

        highlight_hint = QLabel()

        def update_show_both(checked: bool) -> None:
            self._overlay_show_both = bool(checked)
            if checked:
                highlight_hint.setText("Both enabled overlays are rendered at once (hover + selected).")
            else:
                highlight_hint.setText("Rule mode: hover highlight first; if not hovering, selected highlight is used.")
            self._save_preview_ui_settings()
            self._update_preview_highlight_animation_state()
            self._update_preview_pixmap()

        show_both_toggle.toggled.connect(update_show_both)
        layout.addWidget(show_both_toggle)

        update_show_both(bool(self._overlay_show_both))

        layout.addWidget(highlight_hint)

        bg_toggle = QCheckBox("Transparent BG indexes in preview")
        bg_toggle.setChecked(self._preview_bg_transparent_enabled)

        def update_bg_transparency(checked: bool) -> None:
            self._preview_bg_transparent_enabled = bool(checked)
            self._save_preview_ui_settings()
            self._preview_base_pixmap = None
            self._invalidate_animation_preview_pixmap_cache(refresh_current_frame=True)
            self._update_preview_pixmap()

        bg_toggle.toggled.connect(update_bg_transparency)
        layout.addWidget(bg_toggle)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def _choose_overlay_color(self) -> None:
        """Open color picker for overlay color."""
        current = QColor(*self._overlay_color)
        result = QColorDialog.getColor(current, self, "Choose Overlay Color")
        if result.isValid():
            self._overlay_color = (result.red(), result.green(), result.blue())
            self.overlay_color_button.setStyleSheet(
                f"background-color: rgb({result.red()}, {result.green()}, {result.blue()}); border: 1px solid #888;"
            )
            self._update_preview_pixmap()
    
    def _update_overlay_alpha_min(self, value: int) -> None:
        """Update minimum overlay alpha from slider."""
        # Ensure min doesn't exceed max
        if value > self._overlay_alpha_max:
            value = self._overlay_alpha_max
            self.overlay_alpha_min_slider.setValue(value)
        self._overlay_alpha_min = value
        percent = int((value / 255.0) * 100)
        self.overlay_alpha_min_label.setText(f"{percent}%")
        logger.debug(f"Overlay alpha min changed: {value} ({percent}%)")
        self._update_preview_pixmap()
    
    def _update_overlay_alpha_max(self, value: int) -> None:
        """Update maximum overlay alpha from slider."""
        # Ensure max doesn't go below minself._animation_speed  # Use configurable speed
        if value < self._overlay_alpha_min:
            value = self._overlay_alpha_min
            self.overlay_alpha_max_slider.setValue(value)
        self._overlay_alpha_max = value
        percent = int((value / 255.0) * 100)
        self.overlay_alpha_max_label.setText(f"{percent}%")
        logger.debug(f"Overlay alpha max changed: {value} ({percent}%)")
        self._update_preview_pixmap()
    
    def _update_selected_color_label(self) -> None:
        """Update the hex color display label based on selected palette color."""
        selected_indexes = self.palette_list.selectedIndexes()
        if not selected_indexes:
            self.selected_color_label.setText("No color selected")
            return
        
        if len(selected_indexes) == 1:
            index = selected_indexes[0]
            slot_index = index.row()
            color = self.palette_list.model_obj.colors[slot_index]
            alpha = self.palette_list.model_obj.alphas[slot_index]
            record = self._current_record()
            local_flag = ""
            if record is not None and slot_index in record.local_overrides:
                local_flag = " [LOCAL]"
            if color is None:
                self.selected_color_label.setText(f"Index {slot_index}: Empty slot{local_flag}")
            else:
                hex_color = "#{:02X}{:02X}{:02X}".format(*color)
                rgb_text = "RGB({}, {}, {})".format(*color)
                alpha_text = f" Alpha: {alpha}" if alpha < 255 else ""
                self.selected_color_label.setText(f"Index {slot_index}: {hex_color} - {rgb_text}{alpha_text}{local_flag}")
        else:
            self.selected_color_label.setText(f"{len(selected_indexes)} colors selected")
    
    def _update_merge_button_state(self) -> None:
        """Enable merge tool button whenever a palette exists."""
        enabled = len(self.palette_colors) > 0
        self.merge_button.setEnabled(enabled)
        self.merge_button.setText("Merge Mode")

    def _resolve_merge_source_destination(self) -> Tuple[List[int], int] | None:
        selected_rows = sorted({idx.row() for idx in self.palette_list.selectedIndexes()})
        if len(selected_rows) < 2:
            return None
        current = self.palette_list.currentIndex()
        destination_row = current.row() if current.isValid() else selected_rows[0]
        if destination_row not in selected_rows:
            destination_row = selected_rows[0]
        source_rows = [row for row in selected_rows if row != destination_row]
        if not source_rows:
            return None
        return source_rows, destination_row

    def _analyze_merge_impact(self, source_rows: Sequence[int], destination_row: int) -> Dict[str, Any]:
        return self._analyze_merge_impact_for_records(
            source_rows,
            destination_row,
            list(self.sprite_records.values()),
        )

    def _apply_source_destination_merge(
        self,
        source_rows: Sequence[int],
        destination_row: int,
        *,
        scope: Literal["global", "group", "local"] = "global",
        source_after_mode: str = "compact_shift",
        source_after_color: ColorTuple = (255, 0, 255),
    ) -> bool:
        source_rows_sorted = sorted({int(row) for row in source_rows if int(row) != int(destination_row)})
        if not source_rows_sorted:
            logger.debug("Merge apply skipped: empty source rows")
            return False
        if not (0 <= destination_row < len(self.palette_colors)):
            logger.debug("Merge apply skipped: invalid destination row=%s", destination_row)
            return False

        records = self._merge_scope_records(scope)
        if not records:
            logger.debug("Merge apply skipped: no records for scope=%s", scope)
            return False

        impact = self._analyze_merge_impact_for_records(source_rows_sorted, destination_row, records)
        mode = source_after_mode
        if mode == "compact_shift" and scope != "global":
            logger.debug("Merge mode compact_shift downgraded to preserve_colors for scope=%s", scope)
            mode = "preserve_colors"
        compact = mode == "compact_shift"
        logger.info(
            "Merge apply start scope=%s mode=%s sources=%s destination=%s affected=%s risky=%s",
            scope,
            mode,
            source_rows_sorted,
            destination_row,
            impact["affected_sprites"],
            impact["risky_sprites"],
        )

        if compact:
            pixel_remap: Dict[int, int] = {}
            for old_idx in range(len(self.palette_colors)):
                shift = sum(1 for row in source_rows_sorted if row < old_idx)
                pixel_remap[old_idx] = old_idx - shift
            for source_row in source_rows_sorted:
                pixel_remap[source_row] = pixel_remap[destination_row]

            for record in self.sprite_records.values():
                pixels = list(record.indexed_image.getdata())
                record.indexed_image.putdata([pixel_remap.get(px, px) for px in pixels])

            for source_row in reversed(source_rows_sorted):
                self.palette_colors.pop(source_row)
                self.palette_alphas.pop(source_row)
                self._palette_slot_ids.pop(source_row)

            self._palette_slot_ids = list(range(len(self.palette_colors)))
            if len(self.palette_alphas) != len(self.palette_colors):
                self.palette_alphas = [255] * len(self.palette_colors)

            new_pil_palette: List[int] = []
            for color in self.palette_colors:
                new_pil_palette.extend(color)
            while len(new_pil_palette) < 768:
                new_pil_palette.append(0)

            for record in self.sprite_records.values():
                record.indexed_image.putpalette(new_pil_palette)
                record.slot_bindings = {slot_id: idx for idx, slot_id in enumerate(self._palette_slot_ids)}

            self._sync_palette_model()
        else:
            pixel_remap = {source_row: destination_row for source_row in source_rows_sorted}
            for record in records:
                pixels = list(record.indexed_image.getdata())
                record.indexed_image.putdata([pixel_remap.get(px, px) for px in pixels])

            if mode == "recolor_sources":
                for source_row in source_rows_sorted:
                    if 0 <= source_row < len(self.palette_colors):
                        self.palette_colors[source_row] = source_after_color
                new_pil_palette: List[int] = []
                for color in self.palette_colors:
                    new_pil_palette.extend(color)
                while len(new_pil_palette) < 768:
                    new_pil_palette.append(0)
                for record in self.sprite_records.values():
                    record.indexed_image.putpalette(new_pil_palette)
                self._sync_palette_model()

        self._invalidate_used_index_cache(f"merge:{scope}:{mode}")
        self._last_processed_indexed = None
        self._last_index_data = None
        self._last_palette_info = None
        self._last_preview_rgba = None
        self._update_selected_color_label()
        self._schedule_preview_update()
        self._record_history(f"merge:{scope}", force=True)
        self.statusBar().showMessage(
            f"Merged {len(source_rows_sorted)} Source index(es) → Destination #{destination_row} | scope={scope} mode={mode} affected={impact['affected_sprites']} risk={impact['risky_sprites']}",
            4000,
        )
        return True

    @staticmethod
    def _shift_block_list(values: List[Any], start: int, end: int, target_start: int) -> List[Any]:
        block = values[start : end + 1]
        remaining = values[:start] + values[end + 1 :]
        insert_at = target_start
        return remaining[:insert_at] + block + remaining[insert_at:]

    def _compute_shift_remap(self, start: int, end: int, target_start: int) -> Dict[int, int]:
        tokens: List[int | None] = [None] * 256
        for old_idx, slot_id in enumerate(self._palette_slot_ids):
            if 0 <= slot_id < 256:
                tokens[slot_id] = old_idx
        shifted_tokens = self._shift_block_list(tokens, start, end, target_start)
        remap: Dict[int, int] = {}
        for new_idx, token in enumerate([t for t in shifted_tokens if t is not None]):
            remap[int(token)] = new_idx
        logger.debug(
            "compute_shift_remap start=%s end=%s target_start=%s remap_sample=%s",
            start,
            end,
            target_start,
            dict(list(remap.items())[:10]),
        )
        return remap

    def _shift_selected_indices(self, direction: Literal["left", "right"]) -> None:
        selected_rows = sorted({idx.row() for idx in self.palette_list.selectedIndexes()})
        if not selected_rows:
            QMessageBox.information(self, "Shift Indexes", "Select a contiguous block of indexes first.")
            return
        if selected_rows[-1] - selected_rows[0] + 1 != len(selected_rows):
            QMessageBox.warning(self, "Shift Indexes", "Selection must be contiguous.")
            return

        start = selected_rows[0]
        end = selected_rows[-1]
        block_len = end - start + 1
        steps = self.shift_steps_spin.value()
        if direction == "left":
            target_start = max(0, start - steps)
        else:
            target_start = min(256 - block_len, start + steps)

        if target_start == start:
            self.statusBar().showMessage("Shift reached boundary; no change", 2000)
            return

        remap = self._compute_shift_remap(start, end, target_start)
        self._pending_palette_index_remap = remap
        moved = self.palette_list.model_obj.shift_block(start, end, target_start)
        if not moved:
            self._pending_palette_index_remap = None
            return

        new_rows = list(range(target_start, target_start + block_len))
        selection = self.palette_list.selectionModel()
        if selection:
            first_index = self.palette_list.model_obj.index(new_rows[0])
            selection.clearSelection()
            for row in new_rows:
                row_index = self.palette_list.model_obj.index(row)
                selection.select(row_index, QItemSelectionModel.SelectionFlag.Select)
            selection.setCurrentIndex(
                first_index,
                QItemSelectionModel.SelectionFlag.NoUpdate,
            )
            self.palette_list.scrollTo(first_index)

        selected_after = sorted({idx.row() for idx in self.palette_list.selectedIndexes()})
        logger.debug(
            "shift_selected_indices selection_after count=%s rows=%s",
            len(selected_after),
            selected_after[:20],
        )

        logger.debug(
            "shift_selected_indices direction=%s start=%s end=%s target_start=%s steps=%s includes_empty=%s",
            direction,
            start,
            end,
            target_start,
            steps,
            any(self.palette_list.model_obj.colors[row] is None for row in selected_rows),
        )
        self.statusBar().showMessage(
            f"Shifted indexes {start}-{end} {'left' if direction == 'left' else 'right'} by {steps}",
            2500,
        )
    
    def _merge_selected_colors(self) -> None:
        """Merge selected Source palette indexes into a single Destination index."""
        selection = self._resolve_merge_source_destination()
        if selection is None:
            QMessageBox.warning(self, "Merge Error", "Select at least 2 indexes including one Destination.")
            return

        source_rows, destination_row = selection
        destination_color = self.palette_colors[destination_row]
        impact = self._analyze_merge_impact(source_rows, destination_row)

        # Ask user for confirmation
        reply = QMessageBox.question(
            self,
            "Merge Colors",
            (
                f"Merge {len(source_rows)} Source index(es) into Destination #{destination_row} RGB{destination_color}?\n"
                f"Affected sprites: {impact['affected_sprites']}/{impact['total_sprites']}\n"
                f"Detail-loss risk sprites: {impact['risky_sprites']}\n"
                f"All Source instances will be replaced with Destination #{destination_row}."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        self._apply_source_destination_merge(
            source_rows,
            destination_row,
            scope="global",
            source_after_mode="compact_shift",
        )
    
    def _auto_merge_colors(self) -> None:
        """Auto-merge similar colors based on tolerance."""
        tolerance = self.tolerance_spin.value()
        if not self.palette_colors:
            QMessageBox.information(self, "Auto-Merge", "No colors in palette to merge.")
            return
        
        tolerance_sq = tolerance * tolerance * 3  # RGB distance squared
        
        # Build list of merge operations (source_idx -> target_idx)
        merge_map: Dict[int, int] = {}
        processed = set()
        
        for i in range(len(self.palette_colors)):
            if i in processed:
                continue
            
            color_i = self.palette_colors[i]
            
            for j in range(i + 1, len(self.palette_colors)):
                if j in processed:
                    continue
                
                color_j = self.palette_colors[j]
                distance_sq = self._color_distance_sq(color_i, color_j)
                
                if distance_sq <= tolerance_sq:
                    merge_map[j] = i
                    processed.add(j)
        
        if not merge_map:
            QMessageBox.information(
                self,
                "Auto-Merge",
                f"No similar colors found within tolerance {tolerance}."
            )
            return
        
        reply = QMessageBox.question(
            self,
            "Auto-Merge Confirmation",
            f"Found {len(merge_map)} colors to merge. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        logger.info(f"=== AUTO-MERGE: tolerance={tolerance}, merging {len(merge_map)} colors")
        
        # Build pixel remap: unified_index -> new_unified_index
        sorted_merges = sorted(merge_map.keys(), reverse=True)
        pixel_remap: Dict[int, int] = {}
        
        for old_idx in range(len(self.palette_colors)):
            # Count how many indices below old_idx will be removed
            shift = sum(1 for r in sorted_merges if r < old_idx)
            new_idx = old_idx - shift
            pixel_remap[old_idx] = new_idx
        
        # For merged colors, remap to their target's new position
        for source_idx, target_idx in merge_map.items():
            pixel_remap[source_idx] = pixel_remap[target_idx]
            logger.debug(f"  Merge index {source_idx} -> {pixel_remap[target_idx]}")
        
        # Apply pixel remap to all sprites
        for record in self.sprite_records.values():
            img = record.indexed_image
            pixels = list(img.getdata())
            remapped_pixels = [pixel_remap.get(px, px) for px in pixels]
            img.putdata(remapped_pixels)
            logger.debug(f"  Remapped pixels in {record.path.name}")
        self._invalidate_used_index_cache("auto-merge-remap")
        
        # Remove merged colors from unified palette
        for source_idx in sorted_merges:
            self.palette_colors.pop(source_idx)
            self.palette_alphas.pop(source_idx)
            self._palette_slot_ids.pop(source_idx)
        
        # After merge, palette indices are sequential 0, 1, 2, ...
        # Reset slot IDs to match the new sequential indices
        self._palette_slot_ids = list(range(len(self.palette_colors)))
        if len(self.palette_alphas) != len(self.palette_colors):
            self.palette_alphas = [255] * len(self.palette_colors)
        logger.debug(f"  Reset slot IDs to sequential: {self._palette_slot_ids[:20]}")
        
        # Update the PIL palette in each sprite's indexed_image to match the new unified palette
        new_pil_palette = []
        for color in self.palette_colors:
            new_pil_palette.extend(color)
        # Pad to 256 entries
        while len(new_pil_palette) < 768:
            new_pil_palette.append(0)
        
        for record in self.sprite_records.values():
            record.indexed_image.putpalette(new_pil_palette)
            logger.debug(f"  Updated PIL palette in {record.path.name}")
        
        # Rebuild slot_bindings for all sprites after palette change
        for record in self.sprite_records.values():
            record.slot_bindings = {}
            for idx, slot_id in enumerate(self._palette_slot_ids):
                record.slot_bindings[slot_id] = idx
        
        # Sync model
        self._sync_palette_model()
        
        # Clear all cached preview data since pixels were remapped and palette changed
        self._last_processed_indexed = None
        self._last_index_data = None
        self._last_palette_info = None
        self._last_preview_rgba = None
        
        self.statusBar().showMessage(f"Auto-merged {len(merge_map)} similar colors", 3000)
        self._record_history("auto-merge")
        self._schedule_preview_update()

    def _show_sprites_using_index_dialog(self, index: int) -> None:
        """Show a dialog with a list of sprites that use the specified palette index."""
        if index < 0 or index >= len(self.palette_colors):
            QMessageBox.information(self, "Index Info", f"Index {index} is not in the current palette.")
            return
        
        color = self.palette_colors[index]
        
        # Find all sprites that use this index
        sprites_using_index: List[str] = []
        for path_str, record in self.sprite_records.items():
            img = record.indexed_image
            pixels = list(img.getdata())
            if index in pixels:
                sprites_using_index.append(record.path.name)
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Sprites Using Index {index}")
        dialog.setModal(True)
        dialog.resize(400, 300)
        
        layout = QVBoxLayout(dialog)
        
        # Color info label
        hex_color = "#{:02X}{:02X}{:02X}".format(*color)
        info_label = QLabel(f"Index {index}: {hex_color} RGB{color}")
        info_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(info_label)
        
        # Sprite count label
        count_label = QLabel(f"Found in {len(sprites_using_index)} sprite(s):")
        layout.addWidget(count_label)
        
        # List of sprites
        if sprites_using_index:
            sprite_list = QListWidget()
            for sprite_name in sorted(sprites_using_index):
                sprite_list.addItem(sprite_name)
            layout.addWidget(sprite_list)
        else:
            no_sprites_label = QLabel("No sprites use this index.")
            no_sprites_label.setStyleSheet("color: #888; font-style: italic;")
            layout.addWidget(no_sprites_label)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()

    def _selected_highlight_target(self) -> tuple[int, ColorTuple] | None:
        model = self.palette_list.model_obj
        if self.hover_highlight_checkbox.isChecked() and self._preview_hover_palette_row is not None:
            hover_row = int(self._preview_hover_palette_row)
            if 0 <= hover_row < 256:
                hover_color = model.colors[hover_row]
                if hover_color is not None:
                    actual_hover = self.palette_colors[hover_row] if hover_row < len(self.palette_colors) else hover_color
                    return hover_row, actual_hover

        if self.highlight_checkbox.isChecked():
            index = self.palette_list.currentIndex()
            if index.isValid() and 0 <= index.row() < 256:
                slot_index = index.row()
                color = model.colors[slot_index]
                if color is not None:
                    actual_color = self.palette_colors[slot_index] if slot_index < len(self.palette_colors) else color
                    return slot_index, actual_color

        return None

    @staticmethod
    def _color_distance_sq(a: ColorTuple, b: ColorTuple) -> float:
        return float((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _build_palette_lookup(self, palette: PaletteInfo) -> Dict[ColorTuple, List[int]]:
        lookup: Dict[ColorTuple, List[int]] = {}
        for idx, color in enumerate(palette.colors):
            lookup.setdefault(color, []).append(idx)
        return lookup

    def _highlight_rgba(self, color: ColorTuple) -> tuple[int, int, int, int]:
        lighten = lambda channel: min(255, int(channel * 0.35 + 255 * 0.65))
        return (lighten(color[0]), lighten(color[1]), lighten(color[2]), 200)

    def _apply_highlight_overlay(self, base: Image.Image, target_index: int, color: ColorTuple, alpha: int | None = None) -> Image.Image:
        if self._last_processed_indexed is None:
            return base
        index_data = self._last_index_data
        if index_data is None:
            index_data = list(self._last_processed_indexed.getdata())
            self._last_index_data = index_data
        mask = Image.new("L", base.size, 0)
        mask.putdata([255 if idx == target_index else 0 for idx in index_data])
        if not mask.getbbox():
            return base
        # Use custom overlay color and alpha
        if alpha is None:
            alpha = self._overlay_alpha
        overlay_rgba = (*self._overlay_color, alpha)
        overlay = Image.new("RGBA", base.size, overlay_rgba)
        overlay.putalpha(mask)
        return Image.alpha_composite(base, overlay)

    def _log_palette_debug_info(self, processed_palette: PaletteInfo) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        limit = min(len(self.palette_colors), processed_palette.size, 16)
        for row in range(limit):
            source = self.palette_colors[row]
            processed = processed_palette.colors[row]
            if source != processed:
                logger.debug(
                    "Palette mismatch row=%s source=%s processed=%s",
                    row,
                    source,
                    processed,
                )

    def _build_checkerboard_preview_rgba(self) -> Image.Image | None:
        if self._last_preview_rgba is None:
            return None
        index_data = self._last_index_data
        if index_data is None and self._last_processed_indexed is not None:
            index_data = list(self._last_processed_indexed.getdata())
            self._last_index_data = index_data
        return self._apply_preview_background_transparency_to_rgba(self._last_preview_rgba, index_data)

    def _apply_preview_background_transparency_to_rgba(
        self,
        rgba_image: Image.Image,
        index_data: Sequence[int] | None,
    ) -> Image.Image:
        preview = rgba_image.copy()
        if not self._preview_bg_transparent_enabled or not self._preview_background_indices:
            return preview
        if index_data is None:
            return preview

        bg_indices = self._preview_background_indices
        data = list(preview.getdata())
        remapped: List[tuple[int, int, int, int]] = []
        changed = False
        for px_idx, (r, g, b, a) in enumerate(data):
            idx = int(index_data[px_idx]) if px_idx < len(index_data) else -1
            if idx in bg_indices and a != 0:
                remapped.append((r, g, b, 0))
                changed = True
            else:
                remapped.append((r, g, b, a))
        if changed:
            preview.putdata(remapped)
        return preview

    def _ensure_preview_base_pixmap(self) -> QPixmap | None:
        if self._preview_base_pixmap is not None:
            return self._preview_base_pixmap
        sprite_rgba = self._build_checkerboard_preview_rgba()
        if sprite_rgba is None:
            return None
        qimage = ImageQt(sprite_rgba)
        self._preview_base_pixmap = QPixmap.fromImage(qimage)
        return self._preview_base_pixmap

    def _update_preview_pixmap(self, *_args) -> None:
        if self._is_animation_assist_view_active():
            return
        self._animation_preview_visible_sprite_key = None
        frame_started = time.perf_counter()
        self.preview_panel.set_drag_backdrop_rgba(self._resolve_drag_backdrop_rgba())
        self.preview_panel.set_static_pixmap(None)
        if self._last_preview_rgba is None:
            self.preview_panel.set_overlay_layers([], QSize(0, 0))
            record = self._current_record()
            if record is None:
                self.preview_panel.set_pixmap(None, reset_zoom=self._reset_zoom_next)
            else:
                self.preview_panel.set_pixmap(record.pixmap, reset_zoom=self._reset_zoom_next)
            self._reset_zoom_next = False
            return
        
        # Compose checkerboard + alpha visualization for viewport preview once and reuse.
        base_pixmap = self._ensure_preview_base_pixmap()
        if base_pixmap is None:
            return
        base_cache_key = int(base_pixmap.cacheKey())

        # Build overlay layers; rendering is done in PreviewPane on cached scaled pixels.
        overlay_layers: List[tuple[int, QRegion, QColor]] = []
        layer_signature: tuple[tuple[int, int, int, int, int], ...] = ()
        if self._last_processed_indexed is not None:
            index_data = self._last_index_data
            if index_data is None:
                index_data = list(self._last_processed_indexed.getdata())
                self._last_index_data = index_data
            overlay_layers, layer_signature = self._build_overlay_layers_from_index_data(
                base_pixmap.width(),
                base_pixmap.height(),
                index_data,
                include_dragging=False,
                source_token=("sprite", base_cache_key),
            )

        render_signature: tuple[Any, ...] = (
            base_cache_key,
            bool(self._reset_zoom_next),
            layer_signature,
        )
        if self._last_preview_render_signature == render_signature:
            return
        self._last_preview_render_signature = render_signature

        self.preview_panel.set_overlay_layers(overlay_layers, base_pixmap.size())
        self.preview_panel.set_pixmap(base_pixmap, reset_zoom=self._reset_zoom_next)
        self._reset_zoom_next = False

        elapsed_ms = (time.perf_counter() - frame_started) * 1000.0
        if self._overlay_compose_samples <= 0:
            self._overlay_compose_ms_ema = elapsed_ms
        else:
            self._overlay_compose_ms_ema = (self._overlay_compose_ms_ema * 0.85) + (elapsed_ms * 0.15)
        self._overlay_compose_samples += 1
        if (
            logger.isEnabledFor(logging.DEBUG)
            and self._overlay_compose_samples % max(1, int(self._overlay_perf_log_every)) == 0
        ):
            logger.debug(
                "Preview overlay perf samples=%s last_ms=%.2f ema_ms=%.2f pan=%s zoom=%.2f interval=%.3f",
                self._overlay_compose_samples,
                elapsed_ms,
                self._overlay_compose_ms_ema,
                self._preview_pan_active,
                float(self.preview_panel.current_zoom()) if hasattr(self, "preview_panel") else 1.0,
                self._pan_pulse_refresh_interval(),
            )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if not self._confirm_close_with_unsaved_changes():
            event.ignore()
            return
        self._is_app_closing = True
        self._save_persistent_ui_state()
        self._perform_autosave()
        if self._preview_timer.isActive():
            self._preview_timer.stop()
        if self._preview_pixmap_timer.isActive():
            self._preview_pixmap_timer.stop()
        if self._detect_batch_timer.isActive():
            self._detect_batch_timer.stop()
        if self._load_timer.isActive():
            self._load_timer.stop()
        if self._autosave_debounce_timer.isActive():
            self._autosave_debounce_timer.stop()
        if self._autosave_periodic_timer.isActive():
            self._autosave_periodic_timer.stop()
        self._preview_pending_request = None
        self._preview_active_request = None
        self._detect_batch_queue.clear()
        self._preview_executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)

    def _confirm_close_with_unsaved_changes(self) -> bool:
        if self._is_loading_sprites:
            reply_loading = QMessageBox.question(
                self,
                "Close",
                "Sprites are currently loading. Close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply_loading != QMessageBox.StandardButton.Yes:
                return False

        if not self._autosave_dirty:
            return True
        if self._project_paths is None or self._project_manifest is None:
            return True

        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "There are unsaved project changes. Save before closing?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Discard:
            return True

        saved = self._write_project_state(self._project_paths, self._project_manifest, context="Save Project")
        return bool(saved)

    def eventFilter(self, source, event):  # type: ignore[override]
        if hasattr(self, "palette_list") and source is self.palette_list.viewport():
            if event.type() in (QEvent.Type.MouseMove, QEvent.Type.HoverMove):
                point = None
                if isinstance(event, QMouseEvent):
                    point = event.pos()
                elif hasattr(event, "position"):
                    point = event.position().toPoint()
                if point is None:
                    return super().eventFilter(source, event)
                hover_index = self.palette_list.indexAt(point)
                self._set_preview_hover_palette_row(int(hover_index.row()) if hover_index.isValid() else None)
            if event.type() == QEvent.Type.Resize and self._main_palette_force_columns:
                self._main_palette_layout_timer.start(24)
            if event.type() in (QEvent.Type.Leave, QEvent.Type.HoverLeave):
                self._set_preview_hover_palette_row(None)
        if source is self.images_panel.group_list.viewport():
            if event.type() in (QEvent.Type.Leave, QEvent.Type.HoverLeave):
                self._set_hover_group_id(None)
        if source is self.images_panel.list_widget:
            if event.type() == QEvent.Type.KeyPress:
                if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                    self._remove_selected_images()
                    return True
                # Handle Ctrl+A to select all sprites
                if event.key() == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self.images_panel.list_widget.selectAll()
                    return True
        if hasattr(self, "animation_frame_list") and source is self.animation_frame_list.viewport():
            if event.type() == QEvent.Type.MouseButtonPress and isinstance(event, QMouseEvent):
                if event.button() == Qt.MouseButton.LeftButton:
                    row = self._animation_frame_row_at_point(event.pos())
                    if row is not None:
                        tag = self._selected_animation_tag()
                        if tag is None or row >= len(tag.frames):
                            return True
                        edge_hit = self._animation_frame_edge_hit(row, event.pos())
                        self._animation_manual_drag_active = True
                        self._animation_manual_drag_session_seq += 1
                        self._animation_manual_drag_session_id = self._animation_manual_drag_session_seq
                        self._animation_manual_drag_source_row = row
                        self._animation_manual_drag_target_row = row
                        self._animation_manual_drag_start_x = int(event.pos().x())
                        self._animation_manual_drag_changed = False
                        self._animation_manual_drag_gap_target_timeline_position = None
                        current_rows = self._selected_animation_frame_rows()
                        if edge_hit is None and row not in current_rows:
                            selection_model = self.animation_frame_list.selectionModel()
                            if selection_model is not None:
                                selection_model.clearSelection()
                                row_item = self._timeline_row_for_frame_index(row)
                                if row_item is not None:
                                    idx = self.animation_frame_list.model().index(row_item, 0)
                                    selection_model.select(idx, QItemSelectionModel.SelectionFlag.Select)
                            current_row = self._timeline_row_for_frame_index(row)
                            if current_row is not None:
                                self.animation_frame_list.setCurrentRow(current_row)
                            current_rows = [row]
                        self._animation_manual_drag_selected_rows = current_rows
                        self._animation_manual_drag_resize_edge = "none"
                        self._animation_manual_drag_resize_left_row = row
                        self._animation_manual_drag_resize_right_row = -1
                        self._animation_manual_drag_resize_left_duration = max(1, int(tag.frames[row].duration_frames))
                        self._animation_manual_drag_resize_right_duration = self._animation_frame_gap_before(tag.frames[row])
                        logger.debug(
                            "Timeline press drag_id=%s row=%s edge=%s selected_rows=%s start_x=%s context=%s",
                            self._animation_manual_drag_session_id,
                            row,
                            edge_hit,
                            self._animation_manual_drag_selected_rows,
                            self._animation_manual_drag_start_x,
                            self._animation_debug_point_context(event.pos()),
                        )
                        self._log_animation_timeline_snapshot(f"press-before-drag#{self._animation_manual_drag_session_id}")
                        if edge_hit == "left":
                            self._animation_manual_drag_mode = "resize"
                            self._animation_manual_drag_resize_edge = "left"
                            self._record_history("animation-tags-before-duration", include_fields=["animation_tags"])
                            self.animation_frame_list.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
                            item = self._timeline_item_for_frame_index(row)
                            if item is not None:
                                rect = self.animation_frame_list.visualItemRect(item)
                                self._show_animation_timeline_marker(rect.left(), color="#f5a742")
                        elif edge_hit == "right":
                            self._animation_manual_drag_mode = "resize"
                            self._animation_manual_drag_resize_edge = "right"
                            self._animation_manual_drag_resize_right_duration = 0
                            self._record_history("animation-tags-before-duration", include_fields=["animation_tags"])
                            self.animation_frame_list.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
                            item = self._timeline_item_for_frame_index(row)
                            if item is not None:
                                rect = self.animation_frame_list.visualItemRect(item)
                                self._show_animation_timeline_marker(rect.right() + 1, color="#f5a742")
                        else:
                            self._animation_manual_drag_mode = "move"
                            insert_index, marker_x = self._animation_drop_insert_index_from_point(event.pos())
                            self._animation_manual_drag_target_row = insert_index
                            self._show_animation_timeline_marker(marker_x, color="#7fb6ff")
                            self.animation_frame_list.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                            logger.debug(
                                "Timeline move-start drag_id=%s source_row=%s insert_index=%s",
                                self._animation_manual_drag_session_id,
                                row,
                                insert_index,
                            )
                        return True
            if event.type() == QEvent.Type.MouseMove and isinstance(event, QMouseEvent):
                if self._animation_manual_drag_active:
                    if self._animation_manual_drag_mode == "resize":
                        row = self._animation_manual_drag_resize_left_row
                        edge = self._animation_manual_drag_resize_edge
                        unit_px = max(1, self._animation_timeline_duration_unit_px())
                        delta_px = int(event.pos().x()) - int(self._animation_manual_drag_start_x)
                        delta_frames = int(round(float(delta_px) / float(unit_px)))
                        logger.debug(
                            "Timeline resize-move row=%s edge=%s delta_px=%s delta_frames=%s",
                            row,
                            edge,
                            delta_px,
                            delta_frames,
                        )
                        changed = self._apply_animation_manual_resize_delta(row, edge, delta_frames)
                        if changed:
                            self._animation_manual_drag_changed = True
                            self._refresh_animation_frame_list()
                            row_item = self._timeline_row_for_frame_index(row)
                            if row_item is not None:
                                self.animation_frame_list.setCurrentRow(row_item)
                            if not self.animation_play_btn.isChecked():
                                self._set_animation_playhead_from_selected_frame()
                        item = self._timeline_item_for_frame_index(row) if row >= 0 else None
                        if item is not None:
                            rect = self.animation_frame_list.visualItemRect(item)
                            marker_x = rect.left() if edge == "left" else (rect.right() + 1)
                            self._show_animation_timeline_marker(marker_x, color="#f5a742")
                    else:
                        insert_index, marker_x = self._animation_drop_insert_index_from_point(event.pos())
                        self._animation_manual_drag_target_row = insert_index
                        gap_target_timeline_pos = self._animation_gap_target_timeline_position_at_point(event.pos())
                        logger.debug(
                            "Timeline move-hover drag_id=%s source_row=%s insert_index=%s gap_pos=%s x=%s context=%s",
                            self._animation_manual_drag_session_id,
                            self._animation_manual_drag_source_row,
                            insert_index,
                            gap_target_timeline_pos,
                            int(event.pos().x()),
                            self._animation_debug_point_context(event.pos()),
                        )
                        tag = self._selected_animation_tag()
                        frame_count = len(tag.frames) if tag is not None else 0
                        if frame_count > 0:
                            target_frame = max(0, min(frame_count - 1, insert_index if insert_index < frame_count else (frame_count - 1)))
                            item_row = self._timeline_row_for_frame_index(target_frame)
                            if item_row is not None:
                                self.animation_frame_list.setCurrentRow(item_row)
                        gap_slot_rect = self._animation_gap_slot_rect_at_point(event.pos())
                        if gap_slot_rect is not None:
                            if gap_target_timeline_pos is not None:
                                self._animation_manual_drag_gap_target_timeline_position = int(gap_target_timeline_pos)
                            self._show_animation_timeline_drop_zone(gap_slot_rect)
                        else:
                            self._hide_animation_timeline_drop_zone()
                        self._show_animation_timeline_marker(marker_x, color="#7fb6ff")
                    return True
                else:
                    row = self._animation_frame_row_at_point(event.pos())
                    if row is not None:
                        edge_hit = self._animation_frame_edge_hit(row, event.pos())
                        if edge_hit is not None:
                            self.animation_frame_list.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
                            item = self._timeline_item_for_frame_index(row)
                            if item is not None:
                                rect = self.animation_frame_list.visualItemRect(item)
                                marker_x = rect.left() if edge_hit == "left" else (rect.right() + 1)
                                self._show_animation_timeline_marker(marker_x, color="#5f5f5f")
                        else:
                            self.animation_frame_list.viewport().unsetCursor()
                            self._hide_animation_timeline_marker()
                            self._hide_animation_timeline_drop_zone()
                    else:
                        self.animation_frame_list.viewport().unsetCursor()
                        self._hide_animation_timeline_marker()
                        self._hide_animation_timeline_drop_zone()
            if event.type() == QEvent.Type.MouseButtonRelease and isinstance(event, QMouseEvent):
                if self._animation_manual_drag_active and event.button() == Qt.MouseButton.LeftButton:
                    self._animation_manual_drag_active = False
                    self.animation_frame_list.viewport().unsetCursor()
                    self._hide_animation_timeline_marker()
                    self._hide_animation_timeline_drop_zone()
                    mode = self._animation_manual_drag_mode
                    source_row = self._animation_manual_drag_source_row
                    target_row = self._animation_manual_drag_target_row
                    selected_rows = list(self._animation_manual_drag_selected_rows)
                    drag_delta_px = int(event.pos().x()) - int(self._animation_manual_drag_start_x)
                    gap_target_index = self._animation_gap_target_frame_index_at_point(event.pos())
                    release_gap_target_timeline_pos = self._animation_gap_target_timeline_position_at_point(event.pos())
                    fallback_gap_target_timeline_pos: int | None = None
                    if mode == "move" and release_gap_target_timeline_pos is None and abs(drag_delta_px) >= 2:
                        fallback_gap_target_timeline_pos = self._animation_timeline_position_from_point(event.pos())
                    gap_target_timeline_pos = (
                        release_gap_target_timeline_pos
                        if release_gap_target_timeline_pos is not None
                        else (
                            self._animation_manual_drag_gap_target_timeline_position
                            if self._animation_manual_drag_gap_target_timeline_position is not None
                            else fallback_gap_target_timeline_pos
                        )
                    )
                    drag_id = self._animation_manual_drag_session_id
                    logger.debug(
                        "Timeline release drag_id=%s mode=%s source_row=%s target=%s gap_target=%s gap_pos_release=%s gap_pos_fallback=%s gap_pos_effective=%s selected_rows=%s delta_px=%s release_context=%s",
                        drag_id,
                        mode,
                        source_row,
                        target_row,
                        gap_target_index,
                        release_gap_target_timeline_pos,
                        fallback_gap_target_timeline_pos,
                        gap_target_timeline_pos,
                        selected_rows,
                        drag_delta_px,
                        self._animation_debug_point_context(event.pos()),
                    )
                    self._log_animation_timeline_snapshot(f"release-before-apply#{drag_id}")
                    self._animation_manual_drag_source_row = -1
                    self._animation_manual_drag_target_row = -1
                    self._animation_manual_drag_selected_rows = []
                    self._animation_manual_drag_gap_target_timeline_position = None
                    self._animation_manual_drag_mode = "none"
                    self._animation_manual_drag_resize_edge = "none"
                    self._animation_manual_drag_session_id = 0
                    if mode == "resize":
                        if self._animation_manual_drag_changed:
                            self._record_history("animation-tags-duration", include_fields=["animation_tags"], force=True)
                            self._mark_project_dirty("animation-tags-duration")
                            self._log_animation_timeline_snapshot(f"resize-after-release#{drag_id}")
                        self._animation_manual_drag_resize_left_row = -1
                        self._animation_manual_drag_resize_right_row = -1
                        self._animation_manual_drag_resize_left_duration = 1
                        self._animation_manual_drag_resize_right_duration = 1
                        self._animation_manual_drag_changed = False
                    else:
                        rows = sorted({int(row) for row in selected_rows if row >= 0})
                        no_reorder = False
                        if rows:
                            first = rows[0]
                            last = rows[-1]
                            no_reorder = target_row >= first and target_row <= (last + 1)
                        if no_reorder and gap_target_timeline_pos is None:
                            self._apply_animation_manual_gap_drag(source_row, drag_delta_px, rows)
                        else:
                            self._apply_animation_manual_reorder(
                                source_row,
                                target_row,
                                selected_rows,
                                gap_target_timeline_position=gap_target_timeline_pos,
                            )
                    return True
            if event.type() in (QEvent.Type.Leave, QEvent.Type.HoverLeave):
                if not self._animation_manual_drag_active:
                    self.animation_frame_list.viewport().unsetCursor()
                    self._hide_animation_timeline_marker()
                    self._hide_animation_timeline_drop_zone()
                    self._animation_manual_drag_gap_target_timeline_position = None
        return super().eventFilter(source, event)

    # Drag-and-drop image loading support
    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        paths: List[Path] = []
        for url in event.mimeData().urls():
            local = Path(url.toLocalFile())
            if not local.exists():
                continue
            if local.is_dir():
                options = ScanOptions(roots=[local], recursive=True)
                paths.extend(iter_image_files(options))
            elif local.is_file() and is_supported_image(local):
                paths.append(local)
        if paths:
            self._load_images(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = SpriteToolsWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
