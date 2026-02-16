"""High-level indexed PNG processing pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

from PIL import Image

from .palette_ops import (
    ColorTuple,
    PaletteError,
    PaletteInfo,
    ensure_indexed,
    extract_palette,
)
from .quantization import quantize_image


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProcessOptions:
    input_path: Path
    output_dir: Path
    canvas_size: int | Tuple[int, int] | None = 256
    bg_mode: str = "auto"  # auto|manual|sample|alpha
    bg_color: ColorTuple = (255, 0, 255)
    fill_index: int = 0
    fill_mode: Literal["palette", "transparent"] = "palette"
    transparent_index: int | None = None
    max_colors: int = 256
    dither: bool = True
    target_palette: Sequence[ColorTuple] | None = None
    preserve_palette_order: bool = False
    slot_map: Sequence[int] | None = None
    write_act: bool = True
    offset_x: int = 0  # Horizontal offset in pixels
    offset_y: int = 0  # Vertical offset in pixels


@dataclass(slots=True)
class ProcessResult:
    input_path: Path
    output_path: Path
    act_path: Path | None
    palette: PaletteInfo


def hex_to_rgb(value: str) -> ColorTuple:
    value = value.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        raise ValueError("Expected hex RGB in the form RRGGBB")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (r, g, b)


def _flatten_palette(colors: Sequence[ColorTuple]) -> List[int]:
    limited = list(colors[:256])
    if len(limited) < 256:
        limited.extend([(0, 0, 0)] * (256 - len(limited)))
    flat: List[int] = []
    for color in limited:
        flat.extend(color)
    return flat


def _apply_palette_override(image: Image.Image, colors: Sequence[ColorTuple]) -> Image.Image:
    working = image.copy()
    flat = _flatten_palette(colors)
    working.putpalette(flat)
    return working


def _build_slot_remap(slot_map: Sequence[int] | None, palette_size: int) -> Dict[int, int]:
    if not slot_map:
        return {}
    remap: Dict[int, int] = {}
    for dest, src in enumerate(slot_map):
        if src < 0 or src >= palette_size:
            continue
        remap[src] = dest
    return remap


def _apply_index_remap(image: Image.Image, mapping: Dict[int, int]) -> None:
    if not mapping:
        return
    lut = list(range(256))
    for src, dst in mapping.items():
        if 0 <= src < 256:
            lut[src] = max(0, min(255, int(dst)))
    remapped = image.point(lut)
    image.paste(remapped)


def _prepare_indexed_image(image: Image.Image, options: ProcessOptions) -> Image.Image:
    if (
        image.mode == "P"
        and options.target_palette is not None
        and options.preserve_palette_order
    ):
        logger.debug(
            "Applying palette override without quantization size=%s colors=%s",
            image.size,
            len(options.target_palette),
        )
        ensure_indexed(image)
        return _apply_palette_override(image, options.target_palette)
    should_quantize_with_target = (
        options.target_palette is not None and not options.preserve_palette_order
    )
    if image.mode != "P" or should_quantize_with_target:
        logger.debug(
            "Quantizing image mode=%s target_palette=%s max_colors=%s dither=%s",
            image.mode,
            bool(options.target_palette) and not options.preserve_palette_order,
            options.max_colors,
            options.dither,
        )
        quantized = quantize_image(
            image,
            palette=(list(options.target_palette) if should_quantize_with_target else None),
            max_colors=options.max_colors,
            dither=options.dither,
        )
        return quantized
    ensure_indexed(image)
    return image.copy()


def load_indexed(path: Path, options: ProcessOptions) -> Image.Image:
    with Image.open(path) as img:
        return _prepare_indexed_image(img, options)


def _border_indices(img: Image.Image) -> Dict[int, int]:
    width, height = img.size
    pixels = img.load()
    freq: Dict[int, int] = {}
    for x in range(width):
        idx_top = pixels[x, 0]
        idx_bottom = pixels[x, height - 1]
        freq[idx_top] = freq.get(idx_top, 0) + 1
        freq[idx_bottom] = freq.get(idx_bottom, 0) + 1
    for y in range(height):
        idx_left = pixels[0, y]
        idx_right = pixels[width - 1, y]
        freq[idx_left] = freq.get(idx_left, 0) + 1
        freq[idx_right] = freq.get(idx_right, 0) + 1
    return freq


def _majority_index(freq: Dict[int, int]) -> int:
    return max(freq.items(), key=lambda kv: kv[1])[0] if freq else 0


def _find_matching_color_index(
    freq: Dict[int, int], palette: PaletteInfo, target: ColorTuple
) -> int | None:
    best_index = None
    best_count = -1
    for idx, count in freq.items():
        if idx < palette.size and palette.colors[idx] == target and count > best_count:
            best_index = idx
            best_count = count
    return best_index


def detect_background_index(
    image: Image.Image, palette: PaletteInfo, mode: str, chosen_color: ColorTuple
) -> Tuple[int, ColorTuple]:
    if mode == "alpha":
        return 0, palette.colors[0]
    freq = _border_indices(image)
    if mode in {"manual", "sample"}:
        match = _find_matching_color_index(freq, palette, chosen_color)
        if match is not None:
            return match, chosen_color
    idx = _majority_index(freq)
    color = palette.colors[idx] if idx < palette.size else palette.colors[0]
    return idx, color


def build_index_map(
    palette: PaletteInfo, bg_index: int, bg_color: ColorTuple
) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if bg_index != 0:
        for i in range(palette.size):
            if i == bg_index:
                mapping[i] = 0
            elif i < bg_index:
                mapping[i] = i + 1
            else:
                mapping[i] = i
    for i, color in enumerate(palette.colors):
        if i != bg_index and color == bg_color:
            mapping[i] = 0
    return mapping


def apply_index_map(image: Image.Image, mapping: Dict[int, int]) -> None:
    if not mapping:
        return
    lut = list(range(256))
    for src, dst in mapping.items():
        if 0 <= src < 256:
            lut[src] = max(0, min(255, int(dst)))
    remapped = image.point(lut)
    image.paste(remapped)


def rotate_palette(image: Image.Image, palette: PaletteInfo, bg_index: int, bg_color: ColorTuple) -> None:
    colors = palette.colors.copy()
    if bg_index != 0 and bg_index < len(colors):
        colors = [colors[bg_index]] + colors[:bg_index] + colors[bg_index + 1 :]
    if colors:
        colors[0] = bg_color
    flat: List[int] = []
    for color in colors:
        flat.extend(color)
    image.putpalette(flat)


def set_transparency(image: Image.Image, palette: PaletteInfo, transparent_index: int | None) -> None:
    palette_size = max(256, palette.size)
    alpha_table: List[int] | None = None
    existing = image.info.get("transparency")

    if isinstance(existing, int):
        alpha_table = [255] * palette_size
        if 0 <= existing < palette_size:
            alpha_table[existing] = 0
    elif isinstance(existing, (bytes, bytearray, list, tuple)):
        alpha_table = [255] * palette_size
        for index in range(min(palette_size, len(existing))):
            try:
                alpha_table[index] = max(0, min(255, int(existing[index])))
            except (TypeError, ValueError):
                alpha_table[index] = 255

    if transparent_index is not None:
        if alpha_table is None:
            alpha_table = [255] * palette_size
        if 0 <= transparent_index < palette_size:
            alpha_table[transparent_index] = 0
        image.info["transparency"] = bytes(alpha_table)
        return

    if alpha_table is not None and any(alpha < 255 for alpha in alpha_table):
        image.info["transparency"] = bytes(alpha_table)
        return

    if palette.transparent_index is not None:
        image.info["transparency"] = palette.transparent_index
    elif "transparency" in image.info:
        del image.info["transparency"]


def _normalize_canvas_size(
    requested: int | Tuple[int, int],
) -> Tuple[int, int]:
    if isinstance(requested, tuple):
        target_width = max(1, requested[0])
        target_height = max(1, requested[1])
    else:
        target_width = target_height = max(1, requested)
    return target_width, target_height


def _align_dimension(source: int, target: int) -> Tuple[int, int, int, int]:
    if target >= source:
        src_start = 0
        copy = source
        dst_start = (target - source) // 2
    else:
        src_start = max(0, (source - target) // 2)
        copy = target
        dst_start = 0
    src_end = src_start + copy
    return src_start, src_end, dst_start, copy


def make_canvas(
    image: Image.Image, size: int | Tuple[int, int], fill_index: int, offset_x: int = 0, offset_y: int = 0
) -> Image.Image:
    width, height = image.size
    target_width, target_height = _normalize_canvas_size(size)
    if width == target_width and height == target_height and offset_x == 0 and offset_y == 0:
        return image
    canvas = Image.new("P", (target_width, target_height))
    canvas.putpalette(image.getpalette())
    if "transparency" in image.info:
        canvas.info["transparency"] = image.info["transparency"]
    canvas.paste(fill_index, (0, 0, target_width, target_height))

    # Apply offset to sprite placement first, then compute crop via intersection.
    # This preserves all source pixels that can become visible due to offset changes.
    place_x = (target_width - width) // 2 + offset_x
    place_y = (target_height - height) // 2 + offset_y

    src_x0 = max(0, -place_x)
    src_y0 = max(0, -place_y)
    dst_x0 = max(0, place_x)
    dst_y0 = max(0, place_y)

    copy_w = min(width - src_x0, target_width - dst_x0)
    copy_h = min(height - src_y0, target_height - dst_y0)

    logger.debug(
        "make_canvas src=%sx%s dst=%sx%s offset=(%s,%s) place=(%s,%s) src0=(%s,%s) dst0=(%s,%s) copy=%sx%s",
        width,
        height,
        target_width,
        target_height,
        offset_x,
        offset_y,
        place_x,
        place_y,
        src_x0,
        src_y0,
        dst_x0,
        dst_y0,
        copy_w,
        copy_h,
    )

    if copy_w > 0 and copy_h > 0:
        region = image.crop((src_x0, src_y0, src_x0 + copy_w, src_y0 + copy_h))
        canvas.paste(region, (dst_x0, dst_y0))
    else:
        logger.debug("make_canvas no-overlap after offset; returning fill-only canvas")
    
    return canvas


def write_act(path: Path, palette: PaletteInfo) -> None:
    colors = palette.colors[:256]
    padded = colors + [(0, 0, 0)] * (256 - len(colors))
    with path.open("wb") as fh:
        for r, g, b in padded:
            fh.write(bytes((r, g, b)))


def transform_indexed_image(
    image: Image.Image, options: ProcessOptions
) -> Tuple[Image.Image, PaletteInfo]:
    working = image.copy()
    palette = extract_palette(working)
    transparent_index = options.transparent_index
    if options.fill_mode == "transparent":
        transparent_index = options.fill_index
    if options.preserve_palette_order:
        logger.debug(
            "Preserving palette order canvas_size=%s fill_index=%s fill_mode=%s",
            options.canvas_size,
            options.fill_index,
            options.fill_mode,
        )
        remap = _build_slot_remap(options.slot_map, palette.size)
        if remap:
            logger.debug("Applying slot remap entries=%s", len(remap))
            _apply_index_remap(working, remap)
        if options.target_palette:
            working = _apply_palette_override(working, options.target_palette)
        set_transparency(working, palette, transparent_index)
        canvas_target = options.canvas_size if options.canvas_size is not None else working.size
        should_recanvas = (
            options.canvas_size is not None
            or options.offset_x != 0
            or options.offset_y != 0
        )
        canvas = (
            make_canvas(working, canvas_target, options.fill_index, options.offset_x, options.offset_y)
            if should_recanvas
            else working
        )
        updated_palette = extract_palette(canvas)
        return canvas, updated_palette
    bg_index, bg_color = detect_background_index(
        working, palette, options.bg_mode, options.bg_color
    )
    logger.debug(
        "Rebuilding palette bg_index=%s fill_index=%s fill_mode=%s canvas_size=%s",
        bg_index,
        options.fill_index,
        options.fill_mode,
        options.canvas_size,
    )
    mapping = build_index_map(palette, bg_index, bg_color)
    apply_index_map(working, mapping)
    rotate_palette(working, palette, bg_index, bg_color)
    set_transparency(working, palette, transparent_index)
    canvas_target = options.canvas_size if options.canvas_size is not None else working.size
    should_recanvas = (
        options.canvas_size is not None
        or options.offset_x != 0
        or options.offset_y != 0
    )
    canvas = (
        make_canvas(working, canvas_target, options.fill_index, options.offset_x, options.offset_y)
        if should_recanvas
        else working
    )
    updated_palette = extract_palette(canvas)
    return canvas, updated_palette


def process_image_object(
    image: Image.Image, options: ProcessOptions
) -> Tuple[Image.Image, PaletteInfo]:
    indexed = _prepare_indexed_image(image, options)
    return transform_indexed_image(indexed, options)


def process_sprite(options: ProcessOptions) -> ProcessResult:
    src_image = load_indexed(options.input_path, options)
    canvas, updated_palette = transform_indexed_image(src_image, options)

    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (options.input_path.stem + ".png")
    canvas.save(output_path)

    act_path = output_path.with_suffix(".act")
    if options.write_act:
        write_act(act_path, updated_palette)
    else:
        act_path = None

    return ProcessResult(
        input_path=options.input_path,
        output_path=output_path,
        act_path=act_path,
        palette=updated_palette,
    )
