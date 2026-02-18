from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import List, Literal, Sequence, Tuple
import math

import numpy as np
from PIL import Image
import logging


logger = logging.getLogger(__name__)


@dataclass
class PivotAssistFrame:
    key: str
    label: str
    width: int
    height: int
    centroid_x: float
    centroid_y: float
    anchor_x: float
    anchor_y: float
    raw_dx: float
    raw_dy: float
    phase_dx: float
    phase_dy: float
    phase_confidence: float
    blend_weight: float
    smooth_dx: float
    smooth_dy: float
    suggested_dx: int
    suggested_dy: int
    confidence: float


@dataclass
class _FrameAnalysis:
    centroid_x: float
    centroid_y: float
    opaque_count: int
    width: int
    height: int
    bbox_left: int
    bbox_top: int
    bbox_right: int
    bbox_bottom: int


@dataclass
class PivotAssistResult:
    frames: List[PivotAssistFrame]
    reference_centroid: Tuple[float, float]


def _analyze_alpha_shape(image: Image.Image, alpha_threshold: int = 8) -> _FrameAnalysis:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    width, height = alpha.size
    total = 0
    sum_x = 0.0
    sum_y = 0.0
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    for idx, value in enumerate(alpha.getdata()):
        if int(value) <= alpha_threshold:
            continue
        x = idx % width
        y = idx // width
        total += 1
        sum_x += x
        sum_y += y
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
    if total <= 0:
        center_x = width * 0.5
        center_y = height * 0.5
        result = _FrameAnalysis(
            centroid_x=center_x,
            centroid_y=center_y,
            opaque_count=0,
            width=width,
            height=height,
            bbox_left=max(0, int(round(center_x - 1))),
            bbox_top=max(0, int(round(center_y - 1))),
            bbox_right=min(width - 1, int(round(center_x + 1))),
            bbox_bottom=min(height - 1, int(round(center_y + 1))),
        )
        logger.debug(
            "pivot.alpha empty size=%sx%s threshold=%s",
            width,
            height,
            alpha_threshold,
        )
        return result

    result = _FrameAnalysis(
        centroid_x=(sum_x / float(total)),
        centroid_y=(sum_y / float(total)),
        opaque_count=total,
        width=width,
        height=height,
        bbox_left=min_x,
        bbox_top=min_y,
        bbox_right=max_x,
        bbox_bottom=max_y,
    )
    logger.debug(
        "pivot.alpha ok size=%sx%s opaque=%s centroid=(%.3f,%.3f) bbox=(%s,%s)-(%s,%s)",
        width,
        height,
        total,
        result.centroid_x,
        result.centroid_y,
        result.bbox_left,
        result.bbox_top,
        result.bbox_right,
        result.bbox_bottom,
    )
    return result


def _extract_transparent_indices(image: Image.Image) -> set[int]:
    value = image.info.get("transparency")
    if isinstance(value, int):
        if 0 <= value <= 255:
            out = {int(value)}
            logger.debug("pivot.transparency int count=%s", len(out))
            return out
        logger.debug("pivot.transparency int out-of-range")
        return set()
    if isinstance(value, (bytes, bytearray, list, tuple)):
        out: set[int] = set()
        limit = min(256, len(value))
        for index in range(limit):
            try:
                if int(value[index]) <= 0:
                    out.add(index)
            except Exception:  # noqa: BLE001
                continue
        logger.debug("pivot.transparency table count=%s", len(out))
        return out
    logger.debug("pivot.transparency none")
    return set()


def _major_border_index(index_pixels: Sequence[int], width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        return 0
    freq: dict[int, int] = {}
    for x in range(width):
        top = int(index_pixels[x])
        bottom = int(index_pixels[(height - 1) * width + x])
        freq[top] = freq.get(top, 0) + 1
        freq[bottom] = freq.get(bottom, 0) + 1
    for y in range(height):
        left = int(index_pixels[(y * width)])
        right = int(index_pixels[(y * width) + (width - 1)])
        freq[left] = freq.get(left, 0) + 1
        freq[right] = freq.get(right, 0) + 1
    if not freq:
        return 0
    winner = max(freq.items(), key=lambda item: item[1])[0]
    logger.debug("pivot.border.major winner=%s bins=%s size=%sx%s", winner, len(freq), width, height)
    return winner


def _border_index_histogram(index_pixels: Sequence[int], width: int, height: int) -> dict[int, int]:
    freq: dict[int, int] = {}
    if width <= 0 or height <= 0:
        return freq
    for x in range(width):
        top = int(index_pixels[x])
        bottom = int(index_pixels[(height - 1) * width + x])
        freq[top] = freq.get(top, 0) + 1
        freq[bottom] = freq.get(bottom, 0) + 1
    for y in range(height):
        left = int(index_pixels[(y * width)])
        right = int(index_pixels[(y * width) + (width - 1)])
        freq[left] = freq.get(left, 0) + 1
        freq[right] = freq.get(right, 0) + 1
    return freq


def _border_background_indices(
    index_pixels: Sequence[int],
    width: int,
    height: int,
    *,
    coverage: float = 0.88,
    max_indices: int = 8,
) -> set[int]:
    freq = _border_index_histogram(index_pixels, width, height)
    if not freq:
        return {0}
    ordered = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    total = float(sum(freq.values()))
    if total <= 0:
        return {ordered[0][0]}
    selected: set[int] = set()
    cumulative = 0.0
    for index, count in ordered:
        selected.add(int(index))
        cumulative += float(count)
        if len(selected) >= int(max_indices):
            break
        if (cumulative / total) >= float(coverage):
            break
    logger.debug(
        "pivot.border.background selected=%s coverage=%.2f max=%s",
        sorted(selected),
        coverage,
        max_indices,
    )
    return selected


def _analyze_index_shape(
    image: Image.Image,
    *,
    mask_mode: Literal["auto", "alpha", "border", "transparency"],
    alpha_threshold: int,
) -> _FrameAnalysis:
    width, height = image.size
    if width <= 0 or height <= 0:
        return _FrameAnalysis(
            centroid_x=0.0,
            centroid_y=0.0,
            opaque_count=0,
            width=width,
            height=height,
            bbox_left=0,
            bbox_top=0,
            bbox_right=0,
            bbox_bottom=0,
        )

    index_pixels = [int(value) for value in image.getdata()]
    transparent_indices = _extract_transparent_indices(image)
    border_index = _major_border_index(index_pixels, width, height)
    border_background = _border_background_indices(index_pixels, width, height)

    use_mode = mask_mode
    if use_mode == "auto":
        if transparent_indices:
            use_mode = "transparency"
        else:
            use_mode = "border"
    logger.debug(
        "pivot.index start mode=%s effective=%s size=%sx%s trans_count=%s border_major=%s",
        mask_mode,
        use_mode,
        width,
        height,
        len(transparent_indices),
        border_index,
    )

    if use_mode == "alpha":
        return _analyze_alpha_shape(image, alpha_threshold=alpha_threshold)

    total = 0
    sum_x = 0.0
    sum_y = 0.0
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1

    for idx, value in enumerate(index_pixels):
        x = idx % width
        y = idx // width
        if use_mode == "transparency":
            is_foreground = value not in transparent_indices
        else:
            is_foreground = value not in border_background
        if not is_foreground:
            continue
        total += 1
        sum_x += x
        sum_y += y
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y

    if total <= 0:
        logger.debug("pivot.index no foreground, fallback alpha")
        return _analyze_alpha_shape(image, alpha_threshold=alpha_threshold)

    total_pixels = max(1, width * height)
    foreground_ratio = float(total) / float(total_pixels)
    # If border-based segmentation swallowed almost everything or almost nothing,
    # fallback to alpha-based analysis as a safer default.
    if use_mode == "border" and (foreground_ratio < 0.005 or foreground_ratio > 0.96):
        logger.debug("pivot.index border ratio=%.5f fallback alpha", foreground_ratio)
        return _analyze_alpha_shape(image, alpha_threshold=alpha_threshold)

    result = _FrameAnalysis(
        centroid_x=(sum_x / float(total)),
        centroid_y=(sum_y / float(total)),
        opaque_count=total,
        width=width,
        height=height,
        bbox_left=min_x,
        bbox_top=min_y,
        bbox_right=max_x,
        bbox_bottom=max_y,
    )
    logger.debug(
        "pivot.index ok effective=%s foreground=%s ratio=%.5f centroid=(%.3f,%.3f)",
        use_mode,
        total,
        foreground_ratio,
        result.centroid_x,
        result.centroid_y,
    )
    return result


def _foreground_mask_from_record(
    record: object,
    *,
    mask_mode: Literal["auto", "alpha", "border", "transparency"],
    alpha_threshold: int,
) -> np.ndarray:
    indexed = getattr(record, "indexed_image", None)
    if indexed is None:
        logger.debug("pivot.mask record missing indexed image")
        return np.zeros((1, 1), dtype=np.float32)

    if indexed.mode != "P":
        rgba = indexed.convert("RGBA")
        alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
        mask = (alpha > int(alpha_threshold)).astype(np.float32)
        logger.debug("pivot.mask non-P alpha ratio=%.5f", float(np.mean(mask > 0.5)))
        return mask

    width, height = indexed.size
    if width <= 0 or height <= 0:
        logger.debug("pivot.mask empty dimensions")
        return np.zeros((1, 1), dtype=np.float32)

    index_pixels = np.array(indexed.getdata(), dtype=np.int32).reshape((height, width))
    transparent_indices = _extract_transparent_indices(indexed)

    effective_mode = mask_mode
    if effective_mode == "auto":
        effective_mode = "transparency" if transparent_indices else "border"

    if effective_mode == "alpha":
        rgba = indexed.convert("RGBA")
        alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
        mask = (alpha > int(alpha_threshold)).astype(np.float32)
        logger.debug("pivot.mask alpha ratio=%.5f", float(np.mean(mask > 0.5)))
        return mask

    if effective_mode == "transparency":
        if not transparent_indices:
            border_background = _border_background_indices(index_pixels.ravel().tolist(), width, height)
            mask = np.ones((height, width), dtype=np.float32)
            for index in border_background:
                mask[index_pixels == int(index)] = 0.0
            foreground_ratio = float(np.mean(mask > 0.5))
            if foreground_ratio < 0.005 or foreground_ratio > 0.96:
                rgba = indexed.convert("RGBA")
                alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
                mask = (alpha > int(alpha_threshold)).astype(np.float32)
                logger.debug(
                    "pivot.mask transparency(no table) border ratio=%.5f -> alpha ratio=%.5f",
                    foreground_ratio,
                    float(np.mean(mask > 0.5)),
                )
                return mask
            logger.debug(
                "pivot.mask transparency(no table) using border ratio=%.5f bg_bins=%s",
                foreground_ratio,
                len(border_background),
            )
            return mask
        mask = np.ones((height, width), dtype=np.float32)
        for index in transparent_indices:
            mask[index_pixels == int(index)] = 0.0
        logger.debug("pivot.mask transparency ratio=%.5f indices=%s", float(np.mean(mask > 0.5)), len(transparent_indices))
        return mask

    border_background = _border_background_indices(index_pixels.ravel().tolist(), width, height)
    mask = np.ones((height, width), dtype=np.float32)
    for index in border_background:
        mask[index_pixels == int(index)] = 0.0
    foreground_ratio = float(np.mean(mask > 0.5))
    if foreground_ratio < 0.005 or foreground_ratio > 0.96:
        rgba = indexed.convert("RGBA")
        alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
        mask = (alpha > int(alpha_threshold)).astype(np.float32)
        logger.debug("pivot.mask border fallback ratio=%.5f -> alpha ratio=%.5f", foreground_ratio, float(np.mean(mask > 0.5)))
        return mask
    logger.debug("pivot.mask border ratio=%.5f bg_bins=%s", foreground_ratio, len(border_background))
    return mask


def _center_pad_to_common(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ha, wa = int(a.shape[0]), int(a.shape[1])
    hb, wb = int(b.shape[0]), int(b.shape[1])
    target_h = max(ha, hb)
    target_w = max(wa, wb)
    out_a = np.zeros((target_h, target_w), dtype=np.float32)
    out_b = np.zeros((target_h, target_w), dtype=np.float32)

    ay = max(0, (target_h - ha) // 2)
    ax = max(0, (target_w - wa) // 2)
    by = max(0, (target_h - hb) // 2)
    bx = max(0, (target_w - wb) // 2)

    out_a[ay : ay + ha, ax : ax + wa] = a
    out_b[by : by + hb, bx : bx + wb] = b
    return out_a, out_b


def _phase_correlation_shift(reference_mask: np.ndarray, frame_mask: np.ndarray) -> tuple[float, float, float]:
    if reference_mask.size <= 0 or frame_mask.size <= 0:
        logger.debug("pivot.phase empty masks")
        return (0.0, 0.0, 0.0)
    ref, frm = _center_pad_to_common(reference_mask.astype(np.float32), frame_mask.astype(np.float32))
    if ref.shape[0] < 2 or ref.shape[1] < 2:
        logger.debug("pivot.phase tiny shape=%s", ref.shape)
        return (0.0, 0.0, 0.0)

    wy = np.hanning(ref.shape[0]).astype(np.float32)
    wx = np.hanning(ref.shape[1]).astype(np.float32)
    window = np.outer(wy, wx)

    refw = ref * window
    frmw = frm * window
    ref_energy = float(np.std(refw))
    frm_energy = float(np.std(frmw))
    if ref_energy <= 1e-6 or frm_energy <= 1e-6:
        logger.debug(
            "pivot.phase low-energy ref_std=%.9f frm_std=%.9f",
            ref_energy,
            frm_energy,
        )
        return (0.0, 0.0, 0.0)

    fa = np.fft.fft2(refw)
    fb = np.fft.fft2(frmw)
    cross = fa * np.conj(fb)
    denom = np.abs(cross)
    cross /= np.maximum(denom, 1e-9)
    corr = np.fft.ifft2(cross)
    mag = np.abs(corr)

    peak_index = np.unravel_index(int(np.argmax(mag)), mag.shape)
    peak_y = int(peak_index[0])
    peak_x = int(peak_index[1])

    def _parabolic_offset(left: float, center: float, right: float) -> float:
        denom = (left - (2.0 * center) + right)
        if abs(denom) <= 1e-9:
            return 0.0
        value = 0.5 * (left - right) / denom
        return max(-0.5, min(0.5, float(value)))

    sub_x = 0.0
    sub_y = 0.0
    if 1 <= peak_x < (mag.shape[1] - 1):
        sub_x = _parabolic_offset(
            float(mag[peak_y, peak_x - 1]),
            float(mag[peak_y, peak_x]),
            float(mag[peak_y, peak_x + 1]),
        )
    if 1 <= peak_y < (mag.shape[0] - 1):
        sub_y = _parabolic_offset(
            float(mag[peak_y - 1, peak_x]),
            float(mag[peak_y, peak_x]),
            float(mag[peak_y + 1, peak_x]),
        )
    h, w = mag.shape
    if peak_y > (h // 2):
        peak_y -= h
    if peak_x > (w // 2):
        peak_x -= w
    shift_x = float(peak_x) + sub_x
    shift_y = float(peak_y) + sub_y

    peak_value = float(np.max(mag))
    mean_value = float(np.mean(mag))
    std_value = float(np.std(mag))
    if std_value <= 1e-9 or peak_value <= 1e-9:
        confidence = 0.0
    else:
        z = (peak_value - mean_value) / std_value
        flat = mag.ravel()
        if flat.size >= 2:
            top2 = np.partition(flat, -2)[-2:]
            second_value = float(min(top2[0], top2[1]))
        else:
            second_value = 0.0
        prominence = max(0.0, (peak_value - second_value) / max(peak_value, 1e-9))
        y0 = max(0, int(round(peak_index[0])) - 2)
        y1 = min(mag.shape[0], int(round(peak_index[0])) + 3)
        x0 = max(0, int(round(peak_index[1])) - 2)
        x1 = min(mag.shape[1], int(round(peak_index[1])) + 3)
        response = float(np.sum(mag[y0:y1, x0:x1]))
        response = max(0.0, min(1.0, response))
        confidence = max(0.0, min(1.0, (z / 10.0) * 0.45 + prominence * 0.25 + response * 0.30))

    # Shift required for frame->reference alignment in pixel units.
    logger.debug(
        "pivot.phase shift=(%.3f,%.3f) conf=%.3f peak=%.6f mean=%.6f std=%.6f",
        float(shift_x),
        float(shift_y),
        confidence,
        peak_value,
        mean_value,
        std_value,
    )
    return (float(shift_x), float(shift_y), confidence)


def _one_euro_filter(
    values: Sequence[float],
    *,
    min_cutoff: float,
    beta: float,
    d_cutoff: float = 1.0,
) -> List[float]:
    if not values:
        return []

    dt = 1.0

    def _alpha(cutoff: float) -> float:
        cutoff = max(1e-4, float(cutoff))
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + (tau / dt))

    filtered: List[float] = [float(values[0])]
    prev_raw = float(values[0])
    prev_dx_hat = 0.0

    for value in values[1:]:
        raw = float(value)
        dx = (raw - prev_raw) / dt
        a_d = _alpha(d_cutoff)
        dx_hat = (a_d * dx) + ((1.0 - a_d) * prev_dx_hat)
        cutoff = float(min_cutoff) + (float(beta) * abs(dx_hat))
        a = _alpha(cutoff)
        x_hat = (a * raw) + ((1.0 - a) * filtered[-1])
        filtered.append(x_hat)
        prev_raw = raw
        prev_dx_hat = dx_hat
    return filtered


def _analyze_record(
    record: object,
    *,
    alpha_threshold: int,
    mask_mode: Literal["auto", "alpha", "border", "transparency"],
) -> tuple[str, str, _FrameAnalysis]:
    indexed = getattr(record, "indexed_image", None)
    path = getattr(record, "path", None)
    key = path.as_posix() if isinstance(path, Path) else str(path or "")
    label = path.name if isinstance(path, Path) else key
    if indexed is None:
        return (
            key,
            label,
            _FrameAnalysis(
                centroid_x=0.0,
                centroid_y=0.0,
                opaque_count=0,
                width=0,
                height=0,
                bbox_left=0,
                bbox_top=0,
                bbox_right=0,
                bbox_bottom=0,
            ),
        )

    if indexed.mode == "P":
        analyzed = _analyze_index_shape(
            indexed,
            mask_mode=mask_mode,
            alpha_threshold=alpha_threshold,
        )
    else:
        analyzed = _analyze_alpha_shape(indexed, alpha_threshold=alpha_threshold)
    return key, label, analyzed


def _anchor_for_mode(
    frame: _FrameAnalysis,
    mode: Literal["hybrid", "centroid", "foot", "bbox_center"],
) -> tuple[float, float]:
    if frame.opaque_count <= 0:
        return (frame.centroid_x, frame.centroid_y)
    bbox_center_x = (float(frame.bbox_left) + float(frame.bbox_right)) * 0.5
    bbox_center_y = (float(frame.bbox_top) + float(frame.bbox_bottom)) * 0.5
    bbox_bottom_y = float(frame.bbox_bottom)
    if mode == "centroid":
        return (frame.centroid_x, frame.centroid_y)
    if mode == "foot":
        return (bbox_center_x, bbox_bottom_y)
    if mode == "bbox_center":
        return (bbox_center_x, bbox_center_y)
    anchor_x = (frame.centroid_x * 0.72) + (bbox_center_x * 0.28)
    anchor_y = (frame.centroid_y * 0.64) + (bbox_bottom_y * 0.36)
    return (anchor_x, anchor_y)


def _reference_anchor_for_mode(
    anchors: Sequence[tuple[float, float]],
    mode: Literal["first", "median"],
) -> tuple[float, float]:
    if not anchors:
        return (0.0, 0.0)
    if mode == "first":
        return (anchors[0][0], anchors[0][1])
    xs = [anchor[0] for anchor in anchors]
    ys = [anchor[1] for anchor in anchors]
    return (float(median(xs)), float(median(ys)))


def _smooth_with_jump_clamp(raw: Sequence[float], smoothing: int, max_jump: int) -> List[float]:
    if not raw:
        return []
    if smoothing <= 0:
        return list(raw)
    alpha = max(0.0, min(0.95, (float(smoothing) / 100.0) * 0.85))
    out: List[float] = [float(raw[0])]
    clamp = max(0.0, float(max_jump))
    for index in range(1, len(raw)):
        predicted = (out[index - 1] * alpha) + (float(raw[index]) * (1.0 - alpha))
        if clamp > 0.0:
            delta = predicted - out[index - 1]
            if delta > clamp:
                predicted = out[index - 1] + clamp
            elif delta < -clamp:
                predicted = out[index - 1] - clamp
        out.append(predicted)
    return out


def _median3(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    if len(values) <= 2:
        return [float(v) for v in values]
    out: List[float] = []
    for idx in range(len(values)):
        left = max(0, idx - 1)
        right = min(len(values) - 1, idx + 1)
        window = [float(values[pos]) for pos in range(left, right + 1)]
        out.append(float(median(window)))
    return out


def _anchor_phase_cap(mode: str) -> float:
    if mode == "foot":
        return 0.78
    if mode == "hybrid":
        return 0.86
    return 0.90


def _blend_trust(weight: float, phase_confidence: float) -> float:
    return max(0.0, min(1.0, (weight * 0.75) + (phase_confidence * 0.25)))


def _record_local_offset(record: object) -> tuple[float, float]:
    try:
        ox = float(getattr(record, "offset_x", 0))
    except Exception:  # noqa: BLE001
        ox = 0.0
    try:
        oy = float(getattr(record, "offset_y", 0))
    except Exception:  # noqa: BLE001
        oy = 0.0
    return (ox, oy)


def build_pivot_assist_result(
    records: Sequence[object],
    *,
    smoothing: int = 35,
    max_jump: int = 8,
    alpha_threshold: int = 8,
    anchor_mode: Literal["hybrid", "centroid", "foot", "bbox_center"] = "hybrid",
    reference_mode: Literal["first", "median", "index", "records_median"] = "first",
    reference_index: int | None = None,
    reference_records: Sequence[object] | None = None,
    response_gain: int = 100,
    mask_mode: Literal["auto", "alpha", "border", "transparency"] = "auto",
    solver_mode: Literal["hybrid", "anchor", "phase"] = "hybrid",
) -> PivotAssistResult:
    logger.debug(
        "pivot.build start count=%s smoothing=%s max_jump=%s anchor=%s reference=%s ref_index=%s gain=%s mask=%s solver=%s",
        len(records),
        smoothing,
        max_jump,
        anchor_mode,
        reference_mode,
        reference_index,
        response_gain,
        mask_mode,
        solver_mode,
    )
    if not records:
        logger.debug("pivot.build empty records")
        return PivotAssistResult(frames=[], reference_centroid=(0.0, 0.0))

    analyzed_frames: List[_FrameAnalysis] = []
    labels: List[tuple[str, str]] = []
    record_offsets: List[tuple[float, float]] = []
    for record in records:
        key, label, analyzed = _analyze_record(
            record,
            alpha_threshold=alpha_threshold,
            mask_mode=mask_mode,
        )
        analyzed_frames.append(analyzed)
        labels.append((key, label))
        record_offsets.append(_record_local_offset(record))

    anchors_local: List[tuple[float, float]] = [_anchor_for_mode(frame, anchor_mode) for frame in analyzed_frames]
    anchors: List[tuple[float, float]] = [
        (anchor[0] + offset[0], anchor[1] + offset[1])
        for anchor, offset in zip(anchors_local, record_offsets)
    ]

    reference_offsets: List[tuple[float, float]] = []
    external_anchors_local: List[tuple[float, float]] = []
    external_anchors_world: List[tuple[float, float]] = []
    if reference_mode == "records_median":
        for record in (reference_records or []):
            _key, _label, analyzed = _analyze_record(
                record,
                alpha_threshold=alpha_threshold,
                mask_mode=mask_mode,
            )
            anchor_local = _anchor_for_mode(analyzed, anchor_mode)
            offset = _record_local_offset(record)
            external_anchors_local.append(anchor_local)
            external_anchors_world.append((anchor_local[0] + offset[0], anchor_local[1] + offset[1]))
            reference_offsets.append(offset)

    if reference_mode == "index":
        idx = 0 if reference_index is None else max(0, min(len(anchors) - 1, int(reference_index)))
        reference_x, reference_y = anchors[idx] if anchors else (0.0, 0.0)
        if record_offsets:
            reference_offsets = [record_offsets[idx]]
    elif reference_mode == "records_median":
        if external_anchors_world:
            reference_x, reference_y = _reference_anchor_for_mode(external_anchors_world, "median")
        else:
            reference_x, reference_y = _reference_anchor_for_mode(anchors, "median")
            if record_offsets:
                xs = [offset[0] for offset in record_offsets]
                ys = [offset[1] for offset in record_offsets]
                reference_offsets = [(float(median(xs)), float(median(ys)))]
    elif reference_mode == "median":
        reference_x, reference_y = _reference_anchor_for_mode(anchors, reference_mode)
        if record_offsets:
            xs = [offset[0] for offset in record_offsets]
            ys = [offset[1] for offset in record_offsets]
            reference_offsets = [(float(median(xs)), float(median(ys)))]
    else:
        reference_x, reference_y = _reference_anchor_for_mode(anchors, reference_mode)
        if record_offsets:
            reference_offsets = [record_offsets[0]]
    logger.debug("pivot.build reference=(%.3f,%.3f)", reference_x, reference_y)

    raw_dx = [reference_x - anchor[0] for anchor in anchors]
    raw_dy = [reference_y - anchor[1] for anchor in anchors]
    gain = max(10, min(300, int(response_gain))) / 100.0

    phase_dx: List[float] = [0.0 for _ in anchors]
    phase_dy: List[float] = [0.0 for _ in anchors]
    phase_conf: List[float] = [0.0 for _ in anchors]
    masks: List[np.ndarray] = [
        _foreground_mask_from_record(
            record,
            mask_mode=mask_mode,
            alpha_threshold=alpha_threshold,
        )
        for record in records
    ]

    ref_masks: List[np.ndarray] = []
    if solver_mode in {"hybrid", "phase"}:
        ref_offsets_for_masks: List[tuple[float, float]] = []
        if reference_mode == "records_median" and reference_records:
            ref_masks = [
                _foreground_mask_from_record(
                    record,
                    mask_mode=mask_mode,
                    alpha_threshold=alpha_threshold,
                )
                for record in reference_records
            ]
            ref_offsets_for_masks = list(reference_offsets) if reference_offsets else [(0.0, 0.0) for _ in ref_masks]
        elif reference_mode == "index" and masks:
            idx = 0 if reference_index is None else max(0, min(len(masks) - 1, int(reference_index)))
            ref_masks = [masks[idx]]
            ref_offsets_for_masks = [record_offsets[idx]] if record_offsets else [(0.0, 0.0)]
        elif reference_mode == "median" and masks:
            if anchors:
                distances = [abs(anchor[0] - reference_x) + abs(anchor[1] - reference_y) for anchor in anchors]
                median_idx = int(np.argmin(np.array(distances, dtype=np.float32)))
                ref_masks = [masks[median_idx]]
                ref_offsets_for_masks = [record_offsets[median_idx]] if record_offsets else [(0.0, 0.0)]
        elif masks:
            ref_masks = [masks[0]]
            ref_offsets_for_masks = [record_offsets[0]] if record_offsets else [(0.0, 0.0)]

        if not ref_offsets_for_masks and ref_masks:
            ref_offsets_for_masks = [(0.0, 0.0) for _ in ref_masks]

        for idx, frame_mask in enumerate(masks):
            if not ref_masks:
                break
            dx_samples: List[float] = []
            dy_samples: List[float] = []
            conf_samples: List[float] = []
            frame_offset = record_offsets[idx] if idx < len(record_offsets) else (0.0, 0.0)
            for ref_mask, ref_offset in zip(ref_masks, ref_offsets_for_masks):
                dx_fwd, dy_fwd, conf_fwd = _phase_correlation_shift(ref_mask, frame_mask)
                dx_bwd, dy_bwd, conf_bwd = _phase_correlation_shift(frame_mask, ref_mask)

                consistency_error = abs(dx_fwd + dx_bwd) + abs(dy_fwd + dy_bwd)
                consistency = max(0.0, min(1.0, 1.0 - (consistency_error / 2.5)))

                dx = (dx_fwd - dx_bwd) * 0.5
                dy = (dy_fwd - dy_bwd) * 0.5
                conf = min(conf_fwd, conf_bwd) * consistency

                dx_samples.append(dx + (float(ref_offset[0]) - float(frame_offset[0])))
                dy_samples.append(dy + (float(ref_offset[1]) - float(frame_offset[1])))
                conf_samples.append(conf)
            if dx_samples:
                phase_dx[idx] = float(median(dx_samples))
                phase_dy[idx] = float(median(dy_samples))
                phase_conf[idx] = float(max(conf_samples)) if conf_samples else 0.0

    blended_dx: List[float] = []
    blended_dy: List[float] = []
    blend_weights: List[float] = []
    for idx in range(len(raw_dx)):
        if solver_mode == "anchor":
            weight = 0.0
        elif solver_mode == "phase":
            weight = 1.0
        else:
            # Hybrid: adaptively fuse anchor and phase using confidence + agreement,
            # with anchor-mode-specific caps so anchor choice is always meaningful.
            phase_strength = abs(phase_dx[idx]) + abs(phase_dy[idx])
            anchor_strength = abs(raw_dx[idx]) + abs(raw_dy[idx])
            disagreement = abs(phase_dx[idx] - raw_dx[idx]) + abs(phase_dy[idx] - raw_dy[idx])

            conf_norm = max(0.0, min(1.0, (phase_conf[idx] - 0.30) / 0.65))
            base_weight = 0.20 + (0.66 * conf_norm)
            penalty = max(0.0, min(1.0, (disagreement - 1.25) / 6.0))
            weight = (base_weight * (1.0 - (0.55 * penalty))) + ((1.0 - penalty) * 0.08)

            if phase_strength < 0.9:
                weight *= 0.55
            if anchor_strength < 0.8 and phase_strength > 1.6 and phase_conf[idx] > 0.70:
                weight += 0.10

            phase_cap = _anchor_phase_cap(anchor_mode)
            if phase_strength > (anchor_strength + 3.0) and phase_conf[idx] > 0.92:
                phase_cap = min(0.95, phase_cap + 0.04)
            weight = max(0.10, min(phase_cap, weight))
        blend_weights.append(weight)
        blended_dx.append((raw_dx[idx] * (1.0 - weight)) + (phase_dx[idx] * weight))
        blended_dy.append((raw_dy[idx] * (1.0 - weight)) + (phase_dy[idx] * weight))
        logger.debug(
            "pivot.blend idx=%s raw=(%.3f,%.3f) phase=(%.3f,%.3f|c=%.3f) w=%.3f disagree=%.3f blend=(%.3f,%.3f)",
            idx,
            raw_dx[idx],
            raw_dy[idx],
            phase_dx[idx],
            phase_dy[idx],
            phase_conf[idx],
            weight,
            abs(phase_dx[idx] - raw_dx[idx]) + abs(phase_dy[idx] - raw_dy[idx]),
            blended_dx[idx],
            blended_dy[idx],
        )

    # Jump-aware vertical relief: if a frame appears airborne, avoid forcing it
    # back down to the ground reference too aggressively.
    world_bottoms: List[float] = []
    for analyzed, offset in zip(analyzed_frames, record_offsets):
        world_bottoms.append(float(analyzed.bbox_bottom) + float(offset[1]))
    ground_y = float(np.percentile(np.array(world_bottoms, dtype=np.float32), 70.0)) if world_bottoms else 0.0
    for idx in range(len(blended_dy)):
        bottom_y = world_bottoms[idx] if idx < len(world_bottoms) else ground_y
        airborne_px = max(0.0, ground_y - bottom_y)
        if airborne_px < 4.0:
            continue
        if blended_dy[idx] <= 0.0:
            continue
        relief = max(0.0, min(0.78, airborne_px / 24.0))
        before = blended_dy[idx]
        blended_dy[idx] = before * (1.0 - (relief * 0.75))
        logger.debug(
            "pivot.jump_relief idx=%s bottom=%.3f ground=%.3f airborne=%.3f relief=%.3f dy_before=%.3f dy_after=%.3f",
            idx,
            bottom_y,
            ground_y,
            airborne_px,
            relief,
            before,
            blended_dy[idx],
        )

    gain_blended_dx = [value * gain for value in blended_dx]
    gain_blended_dy = [value * gain for value in blended_dy]
    effective_jump = 0
    if max_jump > 0:
        effective_jump = max(1, int(round(float(max_jump) * max(0.5, gain))))
    smoothed_dx = _smooth_with_jump_clamp(gain_blended_dx, smoothing=smoothing, max_jump=effective_jump)
    smoothed_dy = _smooth_with_jump_clamp(gain_blended_dy, smoothing=smoothing, max_jump=effective_jump)

    euro_min_cutoff = max(0.08, 2.25 - (float(smoothing) / 36.0))
    euro_beta = max(0.02, min(1.10, 0.06 + (float(effective_jump) / 40.0) + (max(0.0, gain - 1.0) * 0.25)))
    smoothed_dx = _one_euro_filter(smoothed_dx, min_cutoff=euro_min_cutoff, beta=euro_beta)
    smoothed_dy = _one_euro_filter(smoothed_dy, min_cutoff=euro_min_cutoff, beta=euro_beta)
    logger.debug(
        "pivot.one_euro min_cutoff=%.3f beta=%.3f",
        euro_min_cutoff,
        euro_beta,
    )

    # Adaptive catch-up: when a frame is still far from its gain-applied target,
    # allow an extra step so obvious outliers (e.g. one frame far to the right)
    # can be corrected in a single recompute.
    catch_threshold = max(6.0, float(effective_jump) * 1.4) if effective_jump > 0 else 6.0
    max_catch_step = max(10.0, float(effective_jump) * 3.0) if effective_jump > 0 else 12.0
    for idx in range(1, len(smoothed_dx)):
        target_dx = gain_blended_dx[idx]
        target_dy = gain_blended_dy[idx]
        residual_dx = target_dx - smoothed_dx[idx]
        residual_dy = target_dy - smoothed_dy[idx]
        if abs(residual_dx) <= catch_threshold and abs(residual_dy) <= catch_threshold:
            continue
        trust = _blend_trust(blend_weights[idx], phase_conf[idx])
        catch_ratio = 0.35 + (0.55 * trust)
        step_dx = max(-max_catch_step, min(max_catch_step, residual_dx * catch_ratio))
        step_dy = max(-max_catch_step, min(max_catch_step, residual_dy * catch_ratio))
        before_dx = smoothed_dx[idx]
        before_dy = smoothed_dy[idx]
        smoothed_dx[idx] = before_dx + step_dx
        smoothed_dy[idx] = before_dy + step_dy
        logger.debug(
            "pivot.catchup idx=%s target=(%.3f,%.3f) before=(%.3f,%.3f) residual=(%.3f,%.3f) trust=%.3f ratio=%.3f step=(%.3f,%.3f) after=(%.3f,%.3f)",
            idx,
            target_dx,
            target_dy,
            before_dx,
            before_dy,
            residual_dx,
            residual_dy,
            trust,
            catch_ratio,
            step_dx,
            step_dy,
            smoothed_dx[idx],
            smoothed_dy[idx],
        )

    # Terminal outlier snap: if a frame still has notable residual after catch-up,
    # nudge it further toward the direct target without destabilizing the whole run.
    for idx in range(len(smoothed_dx)):
        target_dx = gain_blended_dx[idx]
        target_dy = gain_blended_dy[idx]
        residual_dx = target_dx - smoothed_dx[idx]
        residual_dy = target_dy - smoothed_dy[idx]
        target_mag = abs(target_dx) + abs(target_dy)
        residual_mag = abs(residual_dx) + abs(residual_dy)
        trust = _blend_trust(blend_weights[idx], phase_conf[idx])
        if target_mag < 14.0 or residual_mag < 2.0 or trust < 0.75:
            continue
        snap_step_dx = max(-6.0, min(6.0, residual_dx * (0.45 + (0.30 * trust))))
        snap_step_dy = max(-6.0, min(6.0, residual_dy * (0.45 + (0.30 * trust))))
        before_dx = smoothed_dx[idx]
        before_dy = smoothed_dy[idx]
        smoothed_dx[idx] = before_dx + snap_step_dx
        smoothed_dy[idx] = before_dy + snap_step_dy
        logger.debug(
            "pivot.snap idx=%s target=(%.3f,%.3f) before=(%.3f,%.3f) residual=(%.3f,%.3f) trust=%.3f step=(%.3f,%.3f) after=(%.3f,%.3f)",
            idx,
            target_dx,
            target_dy,
            before_dx,
            before_dy,
            residual_dx,
            residual_dy,
            trust,
            snap_step_dx,
            snap_step_dy,
            smoothed_dx[idx],
            smoothed_dy[idx],
        )

    # Jitter kill pass: median stabilize and dead-band lock tiny oscillations.
    median_dx = _median3(smoothed_dx)
    median_dy = _median3(smoothed_dy)
    for idx in range(len(smoothed_dx)):
        trust = _blend_trust(blend_weights[idx], phase_conf[idx])
        stabilize = 0.30 + (0.35 * trust)
        smoothed_dx[idx] = (smoothed_dx[idx] * (1.0 - stabilize)) + (median_dx[idx] * stabilize)
        smoothed_dy[idx] = (smoothed_dy[idx] * (1.0 - stabilize)) + (median_dy[idx] * stabilize)

    for idx in range(1, len(smoothed_dx)):
        delta_x = smoothed_dx[idx] - smoothed_dx[idx - 1]
        delta_y = smoothed_dy[idx] - smoothed_dy[idx - 1]
        motion_hint = abs(phase_dx[idx]) + abs(phase_dy[idx])
        if abs(delta_x) < 0.85 and motion_hint < 2.2:
            smoothed_dx[idx] = smoothed_dx[idx - 1]
        if abs(delta_y) < 0.85 and motion_hint < 2.2:
            smoothed_dy[idx] = smoothed_dy[idx - 1]

    final_dx = smoothed_dx
    final_dy = smoothed_dy
    logger.debug(
        "pivot.gain gain=%.3f max_jump=%s effective_jump=%s",
        gain,
        max_jump,
        effective_jump,
    )

    frames: List[PivotAssistFrame] = []
    for index, ((key, label), analyzed) in enumerate(zip(labels, analyzed_frames)):
        cx = analyzed.centroid_x
        cy = analyzed.centroid_y
        opaque_count = analyzed.opaque_count
        width = analyzed.width
        height = analyzed.height
        total_px = max(1, width * height)
        density = min(1.0, float(opaque_count) / float(total_px))
        bbox_w = max(1, (analyzed.bbox_right - analyzed.bbox_left + 1))
        bbox_h = max(1, (analyzed.bbox_bottom - analyzed.bbox_top + 1))
        occupancy = min(1.0, float(opaque_count) / float(max(1, bbox_w * bbox_h)))
        smooth_error = abs(gain_blended_dx[index] - smoothed_dx[index]) + abs(gain_blended_dy[index] - smoothed_dy[index])
        confidence = max(
            0.0,
            min(
                1.0,
                (density * 2.6)
                + (occupancy * 0.45)
                + max(0.0, 0.75 - (smooth_error * 0.05)),
            ),
        )
        confidence = max(confidence, phase_conf[index] * 0.75)
        anchor_x, anchor_y = anchors[index]
        offset_x, offset_y = record_offsets[index] if index < len(record_offsets) else (0.0, 0.0)
        frames.append(
            PivotAssistFrame(
                key=key,
                label=label,
                width=width,
                height=height,
                centroid_x=float(cx),
                centroid_y=float(cy),
                anchor_x=float(anchor_x),
                anchor_y=float(anchor_y),
                raw_dx=float(raw_dx[index]),
                raw_dy=float(raw_dy[index]),
                phase_dx=float(phase_dx[index]),
                phase_dy=float(phase_dy[index]),
                phase_confidence=float(phase_conf[index]),
                blend_weight=float(blend_weights[index]),
                smooth_dx=float(smoothed_dx[index]),
                smooth_dy=float(smoothed_dy[index]),
                suggested_dx=int(round(final_dx[index])),
                suggested_dy=int(round(final_dy[index])),
                confidence=confidence,
            )
        )
        logger.debug(
            "pivot.frame idx=%s label=%s anchor=(%.3f,%.3f) offset=(%.3f,%.3f) raw=(%.3f,%.3f) phase=(%.3f,%.3f|%.3f) smooth=(%.3f,%.3f) final=(%s,%s) conf=%.3f",
            index,
            label,
            anchor_x,
            anchor_y,
            offset_x,
            offset_y,
            raw_dx[index],
            raw_dy[index],
            phase_dx[index],
            phase_dy[index],
            phase_conf[index],
            smoothed_dx[index],
            smoothed_dy[index],
            int(round(final_dx[index])),
            int(round(final_dy[index])),
            confidence,
        )

    logger.debug("pivot.build done frames=%s", len(frames))
    return PivotAssistResult(frames=frames, reference_centroid=(float(reference_x), float(reference_y)))
