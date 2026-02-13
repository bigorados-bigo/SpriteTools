from __future__ import annotations

import json
import hashlib
import re
import sys
from collections import defaultdict
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
from PySide6.QtGui import QColor, QIcon, QKeySequence, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent, QShortcut, QBitmap, QImage
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMainWindow,
    QMessageBox,
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
    QVBoxLayout,
    QWidget,
)

from sprite_tools.file_scanner import ScanOptions, iter_image_files, is_supported_image
from sprite_tools.palette_ops import PaletteError, PaletteInfo, extract_palette, read_act_palette
from sprite_tools.quantization import quantize_image
from sprite_tools.processing import ProcessOptions, process_image_object, process_sprite, write_act

ColorTuple = Tuple[int, int, int]

logger = logging.getLogger(__name__)
DEBUG_LOG_PATH: Path | None = None
_EXCEPTION_HOOK_INSTALLED = False
_PALETTE_CLIPBOARD_MIME = "application/x-spritetools-palette+json"
_SPRITE_BASE_NAME_ROLE = Qt.ItemDataRole.UserRole + 1
_GROUP_ID_ROLE = Qt.ItemDataRole.UserRole + 2
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
    logger.debug(
        "extract_palette_alphas count=%s transparency_type=%s sample=%s",
        len(alphas),
        type(transparency).__name__ if transparency is not None else None,
        alphas[:8],
    )
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
class HistoryEntry:
    label: str
    state: Dict[str, Any]


@dataclass
class HistoryField:
    capture: Callable[[], Any]
    apply: Callable[[Any], None]


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

    def record(self, label: str, *, force: bool = False) -> bool:
        if not self._ready or self._restoring:
            logger.debug(
                "History record skipped label=%s ready=%s restoring=%s force=%s",
                label,
                self._ready,
                self._restoring,
                force,
            )
            return False
        snapshot = self._capture_state()
        if not force and self._history and snapshot == self._history[self._index].state:
            logger.debug("History record dedup label=%s index=%s entries=%s", label, self._index, len(self._history))
            return False
        if self._index < len(self._history) - 1:
            self._history = self._history[: self._index + 1]
            logger.debug("History record truncated future branch label=%s new_len=%s", label, len(self._history))
        self._history.append(HistoryEntry(label=label, state=snapshot))
        self._index += 1
        logger.debug(
            "History record added label=%s force=%s index=%s entries=%s",
            label,
            force,
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

    def _capture_state(self) -> Dict[str, Any]:
        return {name: field.capture() for name, field in self._fields.items()}

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
        logger.debug(f"Model.set_colors: reason={reason}, num_colors={len(colors)}, slots={slots[:20] if slots else None}")
        if len(colors) >= 14:
            logger.debug(f"  colors[13] = {colors[13]}")
        
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
        
        logger.debug(f"  After set: self.colors[13] = {self.colors[13]}")
        self.last_index_remap = None
        
        non_empty = sum(1 for c in self.colors if c is not None)
        logger.debug(
            "Model set_colors count=%s/%s reason=%s emit=%s slots=%s",
            non_empty,
            256,
            reason,
            emit_signal,
            slots is not None,
        )
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
        logger.debug(
            "PaletteListView set_colors count=%s emit=%s slots=%s",
            len(colors),
            emit_signal,
            slots is not None,
        )
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
        logger.debug("PaletteListView local rows updated count=%s sample=%s", len(self._delegate.local_rows), sorted(self._delegate.local_rows)[:12])
        self.viewport().update()

    def set_usage_counts(self, counts: Dict[int, int], *, show_badge: bool = True) -> None:
        normalized: Dict[int, int] = {}
        for row, count in counts.items():
            row_int = int(row)
            if 0 <= row_int < 256:
                normalized[row_int] = max(0, int(count))
        self._delegate.usage_counts = normalized
        self._delegate.show_usage_badge = bool(show_badge)
        logger.debug(
            "PaletteListView usage counts updated rows=%s show_badge=%s sample=%s",
            len(normalized),
            show_badge,
            dict(list(normalized.items())[:12]),
        )
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


class PreviewPane(QWidget):
    zoomChanged = Signal(float)
    drag_offset_changed = Signal(int, int)  # (dx, dy) in pixels

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.title = QLabel("Preview")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.viewport().installEventFilter(self)
        self.image_label.installEventFilter(self)
        self._source_pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._min_zoom = 0.25
        self._max_zoom = 8.0
        self._transform_mode = Qt.TransformationMode.FastTransformation
        self._pan_active = False
        self._pan_last_pos = QPointF(0, 0)
        self._drag_mode = False
        self._drag_active = False
        self._drag_start_pos = QPointF(0, 0)
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
        logger.debug(f"PreviewPane.set_pixmap called: pixmap={'None' if pixmap is None else f'{pixmap.width()}x{pixmap.height()}'}, reset_zoom={reset_zoom}")
        if pixmap is None:
            self.image_label.clear()
            self.image_label.setText("Select an image to preview")
            self._source_pixmap = None
            if reset_zoom:
                self._zoom = 1.0
            self.image_label.adjustSize()
            return
        self._source_pixmap = pixmap
        if reset_zoom:
            self._zoom = 1.0
        self.image_label.setText("")
        self._apply_scaled_pixmap()
        logger.debug(f"PreviewPane.set_pixmap completed")

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        self._apply_scaled_pixmap()
        super().resizeEvent(event)

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
        return super().eventFilter(obj, event)
    
    def _start_drag(self, event: QMouseEvent) -> None:
        """Start dragging sprite for repositioning."""
        self._drag_active = True
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
                self.drag_offset_changed.emit(dx, dy)
                self._drag_start_pos = current_pos
    
    def _end_drag(self) -> None:
        """End dragging sprite."""
        self._drag_active = False

    def wheelEvent(self, event):  # type: ignore[override]
        if not isinstance(event, QWheelEvent) or not self._handle_wheel_zoom(event):
            super().wheelEvent(event)

    def _apply_scaled_pixmap(self) -> None:
        if not self._source_pixmap:
            return
        target_width = max(1, int(self._source_pixmap.width() * self._zoom))
        target_height = max(1, int(self._source_pixmap.height() * self._zoom))
        scaled = self._source_pixmap.scaled(
            QSize(target_width, target_height),
            Qt.AspectRatioMode.KeepAspectRatio,
            self._transform_mode,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())

    def set_scaling_mode(self, mode: Qt.TransformationMode) -> None:
        if self._transform_mode == mode:
            return
        self._transform_mode = mode
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
        self._pan_last_pos = event.globalPosition()
        self.scroll_area.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)

    def _pan(self, event: QMouseEvent) -> None:
        if not self._pan_active:
            return
        delta = event.globalPosition() - self._pan_last_pos
        self._pan_last_pos = event.globalPosition()
        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()
        hbar.setValue(int(max(hbar.minimum(), min(hbar.maximum(), hbar.value() - delta.x()))))
        vbar.setValue(int(max(vbar.minimum(), min(vbar.maximum(), vbar.value() - delta.y()))))

    def _end_pan(self) -> None:
        if not self._pan_active:
            return
        self._pan_active = False
        self.scroll_area.viewport().unsetCursor()


class LoadedImagesPanel(QWidget):
    def __init__(self, on_selection_change, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        mode_row = QHBoxLayout()
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
        
        self.load_button = QPushButton("Load Images")
        self.clear_button = QPushButton("Clear All")
        self.count_label = QLabel("No files loaded")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        group_label = QLabel("Groups")
        group_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.group_list = QListWidget()
        self.group_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.group_list.setToolTip("Loaded sprite groups")

        group_actions = QHBoxLayout()
        self.new_group_button = QPushButton("New Group")
        self.assign_group_button = QPushButton("Assign")
        self.group_color_button = QPushButton("Color")
        self.detach_group_button = QPushButton("Detach")
        self.auto_group_button = QPushButton("Auto")
        self.new_group_button.setToolTip("Create a new group from selected sprites")
        self.assign_group_button.setToolTip("Assign selected sprites to selected group")
        self.group_color_button.setToolTip("Customize selected group color")
        self.detach_group_button.setToolTip("Detach selected sprites into individual groups")
        self.auto_group_button.setToolTip("Auto-assign selected sprites by signature")
        group_actions.addWidget(self.new_group_button)
        group_actions.addWidget(self.assign_group_button)
        group_actions.addWidget(self.group_color_button)
        group_actions.addWidget(self.detach_group_button)
        group_actions.addWidget(self.auto_group_button)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.load_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.count_label)
        layout.addWidget(group_label)
        layout.addWidget(self.group_list, 1)
        layout.addLayout(group_actions)
        layout.addWidget(self.list_widget, 1)
        self.list_widget.currentItemChanged.connect(on_selection_change)

    def set_loaded_count(self, count: int) -> None:
        if count <= 0:
            self.count_label.setText("No files loaded")
        else:
            self.count_label.setText(f"{count} file(s) loaded")

    def selected_load_mode(self) -> Literal["detect", "preserve"]:
        data = self.load_mode_combo.currentData()
        return "preserve" if data == "preserve" else "detect"


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
        finally:
            for control, blocked in zip(controls, blocked_states):
                control.blockSignals(blocked)
            self._is_loading_settings = False
        logger.debug(
            "Floating palette settings loaded window=%s cols=%s zoom=%s gap=%s force=%s idx=%s grid=%s",
            self._window_index,
            self.columns_spin.value(),
            self.zoom_spin.value(),
            self.spacing_spin.value(),
            self.force_columns_check.isChecked(),
            self.show_indices_check.isChecked(),
            self.grid_lines_check.isChecked(),
        )

    def _save_view_settings(self) -> None:
        self._settings.setValue(self._settings_key("columns"), int(self.columns_spin.value()))
        self._settings.setValue(self._settings_key("zoom"), int(self.zoom_spin.value()))
        self._settings.setValue(self._settings_key("gap"), int(self.spacing_spin.value()))
        self._settings.setValue(self._settings_key("force_columns"), bool(self.force_columns_check.isChecked()))
        self._settings.setValue(self._settings_key("show_indices"), bool(self.show_indices_check.isChecked()))
        self._settings.setValue(self._settings_key("show_grid"), bool(self.grid_lines_check.isChecked()))

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
            "merge.apply": (self._owner._get_key_binding("merge.apply", "A"), self._on_apply_shortcut),
        }
        for action, (sequence, callback) in binding_map.items():
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._binding_shortcuts.append(shortcut)
            logger.debug("Merge dialog binding action=%s sequence=%s", action, sequence)

    def _on_apply_shortcut(self) -> None:
        if self.apply_button.isEnabled():
            self._apply_merge()

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


class SpriteToolsWindow(QMainWindow):
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
        self._group_key_to_id: Dict[str, str] = {}
        self._next_group_id = 1
        self._slot_color_lookup: Dict[ColorTuple, List[int]] = {}
        self._next_slot_id = 0
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._render_preview)
        self._main_palette_layout_timer = QTimer(self)
        self._main_palette_layout_timer.setSingleShot(True)
        self._main_palette_layout_timer.timeout.connect(self._apply_main_palette_layout_options)
        
        # Timer for smooth fade animation on selected color in preview
        self._highlight_animation_timer = QTimer(self)
        self._highlight_animation_timer.setInterval(30)  # ~33 FPS for smooth animation
        self._highlight_animation_timer.timeout.connect(self._update_highlight_animation)
        self._highlight_animation_phase = 0.0
        
        # Customizable overlay settings
        self._overlay_color = (255, 255, 255)  # Default white for maximum visibility
        self._overlay_alpha_min = 77  # Minimum opacity (30%)
        self._overlay_alpha_max = 255  # Maximum opacity (100%)
        self._animation_speed = 0.15  # Animation speed increment per frame
        self._active_offset_drag_mode: Literal["none", "global", "group", "individual"] = "none"
        self._floating_palette_windows: List[FloatingPaletteWindow] = []
        self._floating_palette_window_counter = 0
        self._main_palette_columns = 8
        self._main_palette_force_columns = True
        self._main_palette_zoom = 42
        self._main_palette_gap = 6
        self._main_palette_show_indices = True
        self._main_palette_show_grid = False
        self._main_palette_layout_in_progress = False
        self._last_main_palette_layout_signature: tuple[Any, ...] | None = None
        self._merge_dialog: MergeOperationDialog | None = None
        self._used_index_cache: Dict[str, set[int]] = {}
        self._settings = QSettings("SpriteTools", "SpriteTools")
        
        self._history_manager: HistoryManager | None = None
        self._pending_palette_index_remap: Dict[int, int] | None = None
        self.setAcceptDrops(True)

        splitter = QSplitter()

        self.images_panel = LoadedImagesPanel(self._on_selection_changed)
        self.images_panel.list_widget.installEventFilter(self)
        self.images_panel.load_button.clicked.connect(self._prompt_and_load)
        self.images_panel.clear_button.clicked.connect(self._clear_all_sprites)
        self.images_panel.load_mode_combo.currentIndexChanged.connect(self._on_load_mode_changed)
        self.images_panel.new_group_button.clicked.connect(self._create_group_from_selected_sprites)
        self.images_panel.assign_group_button.clicked.connect(self._assign_selected_sprites_to_selected_group)
        self.images_panel.group_color_button.clicked.connect(self._set_selected_group_color)
        self.images_panel.detach_group_button.clicked.connect(self._detach_selected_sprites_to_individual_groups)
        self.images_panel.auto_group_button.clicked.connect(self._auto_assign_selected_sprites_to_signature_groups)

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
        
        self._update_canvas_inputs()
        self._update_fill_preview()

        self.output_dir = Path.cwd() / "sprite_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_processed_indexed: Image.Image | None = None
        self._last_index_data: List[int] | None = None
        self._last_preview_rgba: Image.Image | None = None
        self._last_palette_info: PaletteInfo | None = None
        self._reset_zoom_next = True
        self._palette_index_lookup: Dict[ColorTuple, List[int]] = {}

        self.preview_container = QWidget()
        preview_layout = QVBoxLayout(self.preview_container)
        preview_layout.setContentsMargins(6, 6, 6, 6)
        preview_layout.setSpacing(6)
        self.preview_panel = PreviewPane()
        self.preview_panel.drag_offset_changed.connect(self._handle_drag_offset_changed)
        preview_layout.addWidget(self.preview_panel, 1)

        # Single controls row: highlight checkbox, overlay settings button, and scaling
        controls_row = QHBoxLayout()
        self.highlight_checkbox = QCheckBox("Highlight selected color")
        self.highlight_checkbox.setChecked(True)
        self.highlight_checkbox.toggled.connect(self._on_highlight_checkbox_toggled)
        controls_row.addWidget(self.highlight_checkbox)
        
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

        splitter.addWidget(self.images_panel)
        splitter.addWidget(self.palette_panel)
        splitter.addWidget(self.preview_container)
        splitter.setSizes([250, 500, 450])

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(splitter)
        self.setCentralWidget(container)
        self.statusBar().showMessage("Ready")
        if DEBUG_LOG_PATH:
            self.statusBar().showMessage(f"Debug log: {DEBUG_LOG_PATH}", 5000)
        self._update_export_buttons()
        self._update_loaded_count()
        self._setup_history_manager()
        self._install_shortcuts()
        self._load_ui_settings()
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

    def _get_key_binding(self, action: str, default: str) -> str:
        value = self._settings.value(f"bindings/{action}", default)
        text = str(value).strip()
        return text if text else default

    def _save_preview_ui_settings(self) -> None:
        self._set_pref("preview/highlight_enabled", bool(self.highlight_checkbox.isChecked()))
        self._set_pref("preview/overlay_color", f"{self._overlay_color[0]},{self._overlay_color[1]},{self._overlay_color[2]}")
        self._set_pref("preview/overlay_alpha_min", int(self._overlay_alpha_min))
        self._set_pref("preview/overlay_alpha_max", int(self._overlay_alpha_max))
        self._set_pref("preview/animation_speed", float(self._animation_speed))

    def _load_ui_settings(self) -> None:
        self._main_palette_columns = max(1, min(32, self._get_pref_int("palette/main_columns", self._main_palette_columns)))
        self._main_palette_force_columns = self._get_pref_bool("palette/main_force_columns", self._main_palette_force_columns)
        self._main_palette_zoom = max(16, min(96, self._get_pref_int("palette/main_zoom", self._main_palette_zoom)))
        self._main_palette_gap = max(-8, min(20, self._get_pref_int("palette/main_gap", self._main_palette_gap)))
        self._main_palette_show_indices = self._get_pref_bool("palette/main_show_indices", self._main_palette_show_indices)
        self._main_palette_show_grid = self._get_pref_bool("palette/main_show_grid", self._main_palette_show_grid)
        self._apply_main_palette_layout_options()

        color_raw = str(self._settings.value("preview/overlay_color", "255,255,255"))
        parts = [part.strip() for part in color_raw.split(",")]
        if len(parts) == 3:
            try:
                parsed = tuple(max(0, min(255, int(part))) for part in parts)
                if len(parsed) == 3:
                    self._overlay_color = (parsed[0], parsed[1], parsed[2])
            except ValueError:
                pass
        self._overlay_alpha_min = max(0, min(255, self._get_pref_int("preview/overlay_alpha_min", self._overlay_alpha_min)))
        self._overlay_alpha_max = max(self._overlay_alpha_min, min(255, self._get_pref_int("preview/overlay_alpha_max", self._overlay_alpha_max)))
        self._animation_speed = max(0.01, min(0.50, self._get_pref_float("preview/animation_speed", self._animation_speed)))

        highlight_enabled = self._get_pref_bool("preview/highlight_enabled", self.highlight_checkbox.isChecked())
        self.highlight_checkbox.setChecked(highlight_enabled)
        logger.debug(
            "UI settings loaded main_cols=%s main_force=%s main_zoom=%s main_gap=%s highlight=%s overlay_color=%s alpha_min=%s alpha_max=%s speed=%.2f",
            self._main_palette_columns,
            self._main_palette_force_columns,
            self._main_palette_zoom,
            self._main_palette_gap,
            highlight_enabled,
            self._overlay_color,
            self._overlay_alpha_min,
            self._overlay_alpha_max,
            self._animation_speed,
        )

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
        
        self.sprite_records.clear()
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
        self.status_label.setText("")
        self._update_export_buttons()
        self._reset_history()
        logger.info("Cleared all sprites")

    def _prompt_and_load(self) -> None:
        file_dialog = QFileDialog(self, "Select sprite images")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.bmp *.gif *.jpg *.jpeg)")
        if not file_dialog.exec():
            return
        paths = [Path(p) for p in file_dialog.selectedFiles()]
        self._load_images(paths)

    def _on_load_mode_changed(self, _index: int) -> None:
        mode = self.images_panel.selected_load_mode()
        logger.info("Load mode changed to %s (applies to future imports only)", mode)
        self.statusBar().showMessage(f"Load mode set to {mode} (next imports)", 2500)

    def _load_images(self, paths: Sequence[Path]) -> None:
        added = 0
        list_widget = self.images_panel.list_widget
        had_selection = list_widget.currentRow() >= 0
        starting_count = list_widget.count()
        load_mode = self.images_panel.selected_load_mode()
        logger.debug("Load images mode=%s count=%s", load_mode, len(paths))
        for path in paths:
            if path.as_posix() in self.sprite_records:
                continue
            record = self._create_record(path, load_mode=load_mode)
            if record is None:
                continue
            key = path.as_posix()
            self.sprite_records[key] = record
            self._assign_record_group(key, record)
            self._merge_record_palette(record)
            item = QListWidgetItem(path.name)
            icon_pixmap = record.pixmap.scaled(
                QSize(48, 48), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            item.setIcon(QIcon(icon_pixmap))
            item.setData(Qt.ItemDataRole.UserRole, path.as_posix())
            item.setData(_SPRITE_BASE_NAME_ROLE, path.name)
            list_widget.addItem(item)
            added += 1
        if added:
            self._invalidate_used_index_cache("load-images")
            self.statusBar().showMessage(f"Loaded {added} image(s)", 3000)
            if not had_selection and list_widget.count():
                first_new_row = max(0, starting_count)
                list_widget.setCurrentRow(first_new_row)
            self._refresh_palette_for_current_selection()
        else:
            self.statusBar().showMessage("No new images loaded", 3000)
        self._update_export_buttons()
        self._update_loaded_count()
        self._refresh_group_overview()
        if added:
            self._reset_history()

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
            self._update_fill_preview()
            return
        if len(self._palette_slot_ids) != len(self.palette_colors):
            self._palette_slot_ids = list(range(len(self.palette_colors)))
        if len(self.palette_alphas) != len(self.palette_colors):
            self.palette_alphas = [255] * len(self.palette_colors)
        
        logger.debug(f"_sync_palette_model: num_colors={len(self.palette_colors)}, num_slots={len(self._palette_slot_ids)}")
        if len(self.palette_colors) >= 14:
            logger.debug(f"  self.palette_colors[13] = {self.palette_colors[13]}")
        
        # Model will display 256 slots with colors at their slot positions
        self.palette_list.set_colors(
            self.palette_colors,
            slots=self._palette_slot_ids,
            alphas=self.palette_alphas,
            emit_signal=False,
        )
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
        logger.debug(
            "Synced local override visuals record=%s group=%s local_count=%s",
            record.path.name,
            record.group_id,
            len(local_rows),
        )

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

    def _create_palette_group(self, *, mode: Literal["detect", "preserve"], signature: str) -> PaletteGroup:
        group_id = f"G{self._next_group_id:04d}"
        color = self._pick_group_color(self._next_group_id)
        self._next_group_id += 1
        width = self.canvas_width_spin.value() if hasattr(self, "canvas_width_spin") else 304
        height = self.canvas_height_spin.value() if hasattr(self, "canvas_height_spin") else 224
        return PaletteGroup(
            group_id=group_id,
            mode=mode,
            signature=signature,
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
            label = f"{group.group_id} [{group.mode}] ({len(group.member_keys)})"
            item = QListWidgetItem(label)
            item.setData(_GROUP_ID_ROLE, group.group_id)
            item.setBackground(QColor(group.color[0], group.color[1], group.color[2], 96))
            item.setToolTip(f"{group.signature}\nMembers: {len(group.member_keys)}")
            group_list.addItem(item)
            if selected == group.group_id:
                group_list.setCurrentItem(item)
        group_list.blockSignals(False)
        self._refresh_sprite_group_visuals()
        self._update_sprite_offset_controls(self._current_record())
        self._update_canvas_inputs()
        logger.debug("Refreshed group overview groups=%s", group_list.count())

    def _refresh_sprite_group_visuals(self) -> None:
        list_widget = self.images_panel.list_widget
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item is None:
                continue
            key = item.data(Qt.ItemDataRole.UserRole)
            base_name = item.data(_SPRITE_BASE_NAME_ROLE) or item.text()
            record = self.sprite_records.get(key)
            if record is None or not record.group_id:
                item.setText(str(base_name))
                item.setBackground(QColor(0, 0, 0, 0))
                item.setToolTip(str(base_name))
                continue
            group = self._palette_groups.get(record.group_id)
            if group is None:
                item.setText(str(base_name))
                item.setBackground(QColor(0, 0, 0, 0))
                item.setToolTip(str(base_name))
                continue
            item.setText(f"[{group.group_id}] {base_name}")
            item.setBackground(QColor(group.color[0], group.color[1], group.color[2], 70))
            item.setToolTip(f"{base_name}\nGroup: {group.group_id} ({group.mode})")
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
                    base_bg = QColor(group.color[0], group.color[1], group.color[2], 70)

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
        dialog.exec()

    def _selected_group_id(self) -> str | None:
        item = self.images_panel.group_list.currentItem()
        if item is None:
            return None
        group_id = item.data(_GROUP_ID_ROLE)
        return str(group_id) if group_id else None

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
        self.statusBar().showMessage(f"Created group {group.group_id} with {changed} sprite(s)", 3000)
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
        logger.debug(
            "Extracted palette alpha count=%s transparency_type=%s sample=%s",
            len(alphas),
            type(transparency).__name__ if transparency is not None else None,
            alphas[:8],
        )
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
        logger.debug(
            "Synced preserve palette record=%s colors=%s raw_palette_entries=%s alpha_sample=%s",
            record.path.name,
            len(self.palette_colors),
            (len(record.indexed_image.getpalette() or []) // 3),
            self.palette_alphas[:8],
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
        self._rebuild_palette_from_records()
        self._refresh_palette_for_current_selection()
        self._refresh_group_overview()
        self._clear_preview_cache()
        self._update_export_buttons()
        self._update_loaded_count()
        self._reset_history()
        self._schedule_preview_update()

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
                logger.debug(
                    "Preserve Indexed PNG mode: keeping original palette colors=%s raw_palette_entries=%s transparency=%s",
                    len(palette.colors),
                    (len(working.getpalette() or []) // 3),
                    working.info.get("transparency"),
                )
                return working.copy(), palette.colors[:256]
            return working.copy(), palette.colors[:256]

        rgba = working.convert("RGBA")
        ordered_colors, overflow = self._collect_unique_colors(rgba, max_colors=256)
        if not overflow and ordered_colors:
            logger.debug("Building exact indexed image unique_colors=%s", len(ordered_colors))
            indexed, palette_colors = self._build_exact_indexed_image(rgba, ordered_colors)
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

    def _collect_unique_colors(self, image: Image.Image, max_colors: int) -> tuple[List[ColorTuple], bool]:
        seen: Dict[ColorTuple, int] = {}
        order: List[ColorTuple] = []
        for pixel in image.getdata():
            rgb = pixel[:3]
            if rgb in seen:
                continue
            seen[rgb] = 1
            order.append(rgb)
            if len(order) > max_colors:
                return order, True
        return order, False

    def _build_exact_indexed_image(
        self,
        rgba: Image.Image,
        palette_colors: List[ColorTuple],
    ) -> tuple[Image.Image, List[ColorTuple]]:
        color_to_index: Dict[ColorTuple, int] = {color: idx for idx, color in enumerate(palette_colors)}
        data = [color_to_index[pixel[:3]] for pixel in rgba.getdata()]
        indexed = Image.new("P", rgba.size)
        indexed.putdata(data)
        flat_palette: List[int] = []
        for color in palette_colors:
            flat_palette.extend(color)
        if len(flat_palette) < 768:
            flat_palette.extend([0] * (768 - len(flat_palette)))
        indexed.putpalette(flat_palette)
        return indexed, palette_colors

    def _on_selection_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        self._update_export_buttons()
        record = self._current_record()
        if record is None:
            self.preview_panel.set_pixmap(None, reset_zoom=True)
            self._clear_preview_cache()
            self._update_sprite_offset_controls(None)
            self._refresh_palette_for_current_selection()
            return
        self.statusBar().showMessage(f"Previewing {record.path.name}", 2000)
        # Update per-sprite offset controls
        self._update_sprite_offset_controls(record)
        self._refresh_palette_for_current_selection()
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
                for record in targets:
                    self._apply_detect_remap_to_record(record, pixel_remap)
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
                if group_slots:
                    for record in targets:
                        self._apply_detect_slot_deltas_to_record(
                            record,
                            group_slots,
                            new_color_by_slot,
                            new_alpha_by_slot,
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
        if delay_ms <= 0:
            if self._preview_timer.isActive():
                self._preview_timer.stop()
            self._render_preview()
        else:
            self._preview_timer.start(delay_ms)

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
        try:
            processed, palette = self._process_record_image(record, options)
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage(f"Preview failed: {exc}", 3000)
            self.preview_panel.set_pixmap(record.pixmap, reset_zoom=True)
            self._clear_preview_cache()
            return
        self._last_processed_indexed = processed.copy()
        self._last_index_data = list(self._last_processed_indexed.getdata())
        self._last_preview_rgba = processed.convert("RGBA")
        self._last_palette_info = palette
        self._palette_index_lookup = self._build_palette_lookup(palette)
        self._log_palette_debug_info(palette)
        self._update_preview_pixmap()

    def _current_record(self) -> SpriteRecord | None:
        current_item = self.images_panel.list_widget.currentItem()
        if current_item is None:
            return None
        path_key = current_item.data(Qt.ItemDataRole.UserRole)
        return self.sprite_records.get(path_key)

    def _build_process_options(
        self,
        record: SpriteRecord,
        *,
        output_dir: Path,
        preserve_palette: bool = False,
        write_act: bool = False,
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
        if logger.isEnabledFor(logging.DEBUG):
            sample = slot_map[:8] if slot_map is not None else []
            missing = sum(1 for value in slot_map if value < 0) if slot_map is not None else 0
            logger.debug(
                "slot_map record=%s mode=%s len=%s missing=%s sample=%s",
                record.path.name,
                "preserve" if preserve_mode else "detect",
                len(slot_map) if slot_map is not None else 0,
                missing,
                sample,
            )
            # Debug: show what colors the sprite actually has
            sprite_palette_data = record.indexed_image.getpalette()
            if sprite_palette_data:
                sprite_colors = []
                for i in range(min(8, 256)):
                    idx = i * 3
                    if idx + 2 < len(sprite_palette_data):
                        sprite_colors.append((sprite_palette_data[idx], sprite_palette_data[idx+1], sprite_palette_data[idx+2]))
                logger.debug(f"  sprite palette first 8: {sprite_colors}")
                logger.debug(f"  unified palette first 8: {self.palette_colors[:8]}")
                # Show mapping
                for i in range(min(8, len(slot_map) if slot_map is not None else 0)):
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
            zoom_spin.setEnabled(not self._main_palette_force_columns)
            self._set_pref("palette/main_columns", int(self._main_palette_columns))
            self._set_pref("palette/main_force_columns", bool(self._main_palette_force_columns))
            self._set_pref("palette/main_zoom", int(self._main_palette_zoom))
            self._set_pref("palette/main_gap", int(self._main_palette_gap))
            self._set_pref("palette/main_show_indices", bool(self._main_palette_show_indices))
            self._set_pref("palette/main_show_grid", bool(self._main_palette_show_grid))
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
        self.export_selected_btn.setEnabled(current is not None)
        self.export_all_btn.setEnabled(has_records)
        self.export_palette_btn.setEnabled(has_palette)
        self.output_dir_button.setEnabled(True)

    def _update_loaded_count(self) -> None:
        if hasattr(self, "images_panel"):
            self.images_panel.set_loaded_count(len(self.sprite_records))

    def _install_shortcuts(self) -> None:
        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._undo_shortcut.activated.connect(self._undo_history)
        self._redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        self._redo_shortcut.activated.connect(self._redo_history)

    def _setup_history_manager(self) -> None:
        manager = HistoryManager()
        manager.register_field("palette", self._history_capture_palette, self._history_apply_palette)
        manager.register_field("fill", self._history_capture_fill, self._history_apply_fill)
        manager.register_field("canvas", self._history_capture_canvas, self._history_apply_canvas)
        manager.register_field("offsets", self._history_capture_offsets, self._history_apply_offsets)
        self._history_manager = manager

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
        if self._history_manager is not None:
            self._history_manager.reset()

    def _record_history(self, label: str, *, force: bool = False) -> None:
        if self._history_manager is not None:
            recorded = self._history_manager.record(label, force=force)
            logger.debug("Record history label=%s force=%s recorded=%s", label, force, recorded)

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
        if self._history_manager and self._history_manager.undo():
            self._sync_palette_model()
            self._sync_local_override_visuals()
            self._update_selected_color_label()
            self._invalidate_preview_cache(reset_zoom=False, reason="history-undo")
            undone_label = self._history_manager.last_undo_label or "change"
            logger.debug("History undo applied; refreshed palette + preview state undone=%s", undone_label)
            self.statusBar().showMessage(f"Undid {undone_label}", 2000)
            self._schedule_preview_update()

    def _redo_history(self) -> None:
        if self._history_manager and self._history_manager.redo():
            self._sync_palette_model()
            self._sync_local_override_visuals()
            self._update_selected_color_label()
            self._invalidate_preview_cache(reset_zoom=False, reason="history-redo")
            redone_label = self._history_manager.last_redo_label or "change"
            logger.debug("History redo applied; refreshed palette + preview state redone=%s", redone_label)
            self.statusBar().showMessage(f"Redid {redone_label}", 2000)
            self._schedule_preview_update()

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
        self._record_history("offset-scope")

    def _handle_global_offset_change(self, _value: int) -> None:
        self._schedule_preview_update()
        self._record_history("offset-global")
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
        self._record_history("offset-local")
        logger.debug(f"Updated sprite offset: {record.path.name} -> ({record.offset_x}, {record.offset_y})")

    def _handle_group_offset_change(self, _value: int) -> None:
        record = self._current_record()
        group = self._palette_groups.get(record.group_id) if record and record.group_id else None
        if group is None:
            return
        group.offset_x = self.group_offset_x_spin.value()
        group.offset_y = self.group_offset_y_spin.value()
        self._schedule_preview_update()
        self._record_history("offset-group")
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
            self.preview_panel.set_drag_mode(False)
            logger.debug("Offset drag mode disabled via click mode=%s", mode)
            return

        for key, button in button_map.items():
            blocked = button.blockSignals(True)
            button.setChecked(key == mode)
            button.blockSignals(blocked)

        self._active_offset_drag_mode = mode
        self.preview_panel.set_drag_mode(True)
        logger.debug("Offset drag mode enabled via click mode=%s", mode)
    
    def _handle_drag_offset_changed(self, dx: int, dy: int) -> None:
        """Handle offset changes from viewport dragging."""
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
            self._handle_global_offset_change(0)
            logger.debug("Drag offset global -> (%s, %s) delta=(%s,%s) history=offset-global", new_x, new_y, dx, dy)
            return

        if self._active_offset_drag_mode == "group":
            record = self._current_record()
            group = self._palette_groups.get(record.group_id) if record and record.group_id else None
            if group is None:
                return
            new_x = group.offset_x + dx
            new_y = group.offset_y + dy
            bx = self.group_offset_x_spin.blockSignals(True)
            by = self.group_offset_y_spin.blockSignals(True)
            self.group_offset_x_spin.setValue(new_x)
            self.group_offset_y_spin.setValue(new_y)
            self.group_offset_x_spin.blockSignals(bx)
            self.group_offset_y_spin.blockSignals(by)
            self._handle_group_offset_change(0)
            logger.debug("Drag offset group=%s -> (%s, %s) delta=(%s,%s) history=offset-group", group.group_id, new_x, new_y, dx, dy)
            return

        record = self._current_record()
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
        
        self._schedule_preview_update()
        self._record_history("offset-local")
        logger.debug(
            "Drag offset individual record=%s -> (%s, %s) delta=(%s,%s) history=offset-local",
            record.path.name,
            record.offset_x,
            record.offset_y,
            dx,
            dy,
        )

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
        self._last_processed_indexed = None
        self._last_index_data = None
        self._last_preview_rgba = None
        self._last_palette_info = None
        self._palette_index_lookup = {}
        if reset_zoom:
            self._reset_zoom_next = True
        logger.debug("Preview cache invalidated reset_zoom=%s reason=%s", reset_zoom, reason)

    def _clear_preview_cache(self) -> None:
        self._invalidate_preview_cache(reset_zoom=True, reason="full-clear")

    def _on_palette_selection_changed(self, *_args) -> None:
        # Start/stop fade animation based on selection
        selected_indexes = self.palette_list.selectedIndexes()
        logger.debug(f"Palette selection changed: {len(selected_indexes)} selected, highlight_enabled={self.highlight_checkbox.isChecked()}")
        if selected_indexes and self.highlight_checkbox.isChecked():
            if not self._highlight_animation_timer.isActive():
                self._highlight_animation_phase = 0.0
                self._highlight_animation_timer.start()
                logger.debug("Started animation timer")
        else:
            self._highlight_animation_timer.stop()
            logger.debug("Stopped animation timer")
        
        self._update_preview_pixmap()
        self._update_selected_color_label()
        self._update_merge_button_state()
        self._highlight_sprites_for_palette_index(self._current_selected_palette_index())

    def _on_highlight_checkbox_toggled(self, checked: bool) -> None:
        self._set_pref("preview/highlight_enabled", bool(checked))
        if not checked:
            self._highlight_animation_timer.stop()
        else:
            if self.palette_list.selectedIndexes() and not self._highlight_animation_timer.isActive():
                self._highlight_animation_phase = 0.0
                self._highlight_animation_timer.start()
        self._update_preview_pixmap()
    
    def _update_highlight_animation(self) -> None:
        """Update smooth fade animation for highlight."""
        import math
        self._highlight_animation_phase += self._animation_speed  # Use configurable speed
        if self._highlight_animation_phase > 2 * math.pi:
            self._highlight_animation_phase -= 2 * math.pi
        logger.debug(f"Animation phase: {self._highlight_animation_phase:.2f}")
        self._update_preview_pixmap()
        # Force palette view to repaint so swatches show animation
        self.palette_list.viewport().update()
    
    
    def _open_overlay_settings(self) -> None:
        """Open overlay settings dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Overlay Settings")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        
        # Color picker
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Overlay Color:"))
        color_button = QPushButton()
        color_button.setFixedSize(60, 30)
        color_button.setStyleSheet(f"background-color: rgb({self._overlay_color[0]}, {self._overlay_color[1]}, {self._overlay_color[2]}); border: 1px solid #888;")
        
        def choose_color():
            current = QColor(*self._overlay_color)
            result = QColorDialog.getColor(current, dialog, "Choose Overlay Color")
            if result.isValid():
                self._overlay_color = (result.red(), result.green(), result.blue())
                color_button.setStyleSheet(f"background-color: rgb({result.red()}, {result.green()}, {result.blue()}); border: 1px solid #888;")
                self._save_preview_ui_settings()
                self._update_preview_pixmap()
        
        color_button.clicked.connect(choose_color)
        color_layout.addWidget(color_button)
        color_layout.addStretch()
        layout.addLayout(color_layout)
        
        # Min opacity
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min Opacity:"))
        min_slider = QSlider(Qt.Orientation.Horizontal)
        min_slider.setRange(0, 255)
        min_slider.setValue(self._overlay_alpha_min)
        min_label = QLabel(f"{int(self._overlay_alpha_min / 2.55)}%")
        
        def update_min(value):
            if value > self._overlay_alpha_max:
                value = self._overlay_alpha_max
                min_slider.setValue(value)
            self._overlay_alpha_min = value
            min_label.setText(f"{int(value / 2.55)}%")
            self._save_preview_ui_settings()
            self._update_preview_pixmap()
        
        min_slider.valueChanged.connect(update_min)
        min_layout.addWidget(min_slider)
        min_layout.addWidget(min_label)
        layout.addLayout(min_layout)
        
        # Max opacity
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max Opacity:"))
        max_slider = QSlider(Qt.Orientation.Horizontal)
        max_slider.setRange(0, 255)
        max_slider.setValue(self._overlay_alpha_max)
        max_label = QLabel(f"{int(self._overlay_alpha_max / 2.55)}%")
        
        def update_max(value):
            if value < self._overlay_alpha_min:
                value = self._overlay_alpha_min
                max_slider.setValue(value)
            self._overlay_alpha_max = value
            max_label.setText(f"{int(value / 2.55)}%")
            self._save_preview_ui_settings()
            self._update_preview_pixmap()
        
        max_slider.valueChanged.connect(update_max)
        max_layout.addWidget(max_slider)
        max_layout.addWidget(max_label)
        layout.addLayout(max_layout)
        
        # Animation speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Animation Speed:"))
        speed_slider = QSlider(Qt.Orientation.Horizontal)
        speed_slider.setRange(1, 50)  # 0.01 to 0.50 (much wider range)
        speed_slider.setValue(int(self._animation_speed * 100))
        speed_label = QLabel(f"{self._animation_speed:.2f}x")
        
        def update_speed(value):
            self._animation_speed = value / 100.0
            speed_label.setText(f"{self._animation_speed:.2f}x")
            logger.debug(f"Animation speed changed: {self._animation_speed:.2f}")
            self._save_preview_ui_settings()
        
        speed_slider.valueChanged.connect(update_speed)
        speed_layout.addWidget(speed_slider)
        speed_layout.addWidget(speed_label)
        layout.addLayout(speed_layout)
        
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
        self.merge_button.setText("Merge Sources → Destination")

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
        if not self.highlight_checkbox.isChecked():
            return None
        model = self.palette_list.model_obj
        index = self.palette_list.currentIndex()
        if not index.isValid() or not (0 <= index.row() < 256):
            return None
        
        slot_index = index.row()
        color = model.colors[slot_index]
        
        # If empty slot, nothing to highlight
        if color is None:
            return None
        
        # Use the unified palette directly instead of _last_palette_info
        # This ensures we always use the current palette state, even right after a merge
        if slot_index < len(self.palette_colors):
            actual_color = self.palette_colors[slot_index]
            logger.debug(
                "Highlight slot_index=%s color=%s actual_palette_color=%s",
                slot_index,
                color,
                actual_color,
            )
            return slot_index, actual_color
        
        # Slot index is beyond current palette, nothing to highlight
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
        base = self._last_preview_rgba.copy()
        if self._last_processed_indexed is None:
            return base

        index_data = self._last_index_data
        if index_data is None:
            index_data = list(self._last_processed_indexed.getdata())
            self._last_index_data = index_data

        palette_has_alpha = any(alpha < 255 for alpha in self.palette_alphas)
        base_data = list(base.getdata())
        has_transparent_pixels = False

        if palette_has_alpha:
            adjusted_data: List[tuple[int, int, int, int]] = []
            for px_idx, (r, g, b, a) in enumerate(base_data):
                palette_index = index_data[px_idx] if px_idx < len(index_data) else -1
                palette_alpha = self.palette_alphas[palette_index] if 0 <= palette_index < len(self.palette_alphas) else 255
                adjusted_alpha = (a * palette_alpha) // 255
                if adjusted_alpha < 255:
                    has_transparent_pixels = True
                adjusted_data.append((r, g, b, adjusted_alpha))
            base.putdata(adjusted_data)
            base_data = adjusted_data
        else:
            has_transparent_pixels = any(a < 255 for _r, _g, _b, a in base_data)

        if not has_transparent_pixels:
            return base

        width, height = base.size
        checker = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        tile = 8
        light = (200, 200, 200, 255)
        dark = (150, 150, 150, 255)
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                use_light = ((x // tile) + (y // tile)) % 2 == 0
                checker.paste(
                    light if use_light else dark,
                    (x, y, min(x + tile, width), min(y + tile, height)),
                )
        checker.alpha_composite(base)
        logger.debug(
            "Preview checkerboard composed size=%sx%s palette_has_alpha=%s",
            width,
            height,
            palette_has_alpha,
        )
        return checker

    def _update_preview_pixmap(self, *_args) -> None:
        if self._last_preview_rgba is None:
            record = self._current_record()
            if record is None:
                self.preview_panel.set_pixmap(None, reset_zoom=self._reset_zoom_next)
            else:
                self.preview_panel.set_pixmap(record.pixmap, reset_zoom=self._reset_zoom_next)
            self._reset_zoom_next = False
            return
        
        # Compose checkerboard + alpha visualization for viewport preview.
        composed = self._build_checkerboard_preview_rgba()
        if composed is None:
            return

        # Convert base RGBA image to QPixmap
        qimage = ImageQt(composed)
        base_pixmap = QPixmap.fromImage(qimage)
        
        # Apply overlay using QPainter if needed
        target = self._selected_highlight_target()
        if target is not None:
            index, color = target
            import math
            fade = (math.sin(self._highlight_animation_phase) + 1) / 2  # 0 to 1
            # Interpolate between min and max alpha
            alpha_range = self._overlay_alpha_max - self._overlay_alpha_min
            animated_alpha = int(self._overlay_alpha_min + alpha_range * fade)
            logger.debug(f"Applying QPainter overlay: alpha_min={self._overlay_alpha_min}, alpha_max={self._overlay_alpha_max}, fade={fade:.2f}, animated_alpha={animated_alpha}, color={self._overlay_color}")
            
            # Create mask from indexed image
            if self._last_processed_indexed is not None:
                index_data = self._last_index_data
                if index_data is None:
                    index_data = list(self._last_processed_indexed.getdata())
                    self._last_index_data = index_data
                
                # Create a new pixmap to draw on
                result_pixmap = QPixmap(base_pixmap)
                
                # Use QPainter to draw overlay
                painter = QPainter(result_pixmap)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                
                # Create overlay brush
                overlay_color = QColor(*self._overlay_color, animated_alpha)
                painter.setBrush(overlay_color)
                painter.setPen(Qt.PenStyle.NoPen)
                
                # Draw overlay on matching pixels
                width = composed.width
                height = composed.height
                for y in range(height):
                    for x in range(width):
                        pixel_index = y * width + x
                        if pixel_index < len(index_data) and index_data[pixel_index] == index:
                            painter.drawRect(x, y, 1, 1)
                
                painter.end()
                base_pixmap = result_pixmap
                logger.debug(f"QPainter overlay completed")
        
        self.preview_panel.set_pixmap(base_pixmap, reset_zoom=self._reset_zoom_next)
        self._reset_zoom_next = False

    def eventFilter(self, source, event):  # type: ignore[override]
        if hasattr(self, "palette_list") and source is self.palette_list.viewport():
            if event.type() == QEvent.Type.Resize and self._main_palette_force_columns:
                self._main_palette_layout_timer.start(24)
        if source is self.images_panel.list_widget:
            if event.type() == QEvent.Type.KeyPress:
                if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                    self._remove_selected_images()
                    return True
                # Handle Ctrl+A to select all sprites
                if event.key() == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self.images_panel.list_widget.selectAll()
                    return True
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
