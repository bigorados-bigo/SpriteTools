"""Command-line interface for SpriteTools batch operations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

from .file_scanner import ScanOptions, iter_image_files
from .palette_ops import PaletteError, read_act_palette
from .processing import ProcessOptions, hex_to_rgb, process_sprite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch indexed PNG utilities")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input files or folders")
    parser.add_argument("--size", type=int, default=256, help="Output canvas size (square)")
    parser.add_argument(
        "--bg-mode",
        choices=("auto", "manual", "sample", "alpha"),
        default="auto",
        help="Background detection strategy",
    )
    parser.add_argument(
        "--bg-color",
        default="#ff00ff",
        help="Background RGB hex when using manual/sample modes",
    )
    parser.add_argument(
        "--trans0",
        action="store_true",
        help="Force index 0 to be transparent",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Descend into subfolders"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination folder (defaults to <input>/out)",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        default=256,
        help="Maximum colors to keep when quantizing non-indexed sprites",
    )
    parser.add_argument(
        "--no-dither",
        action="store_true",
        help="Disable Floyd-Steinberg dithering during quantization",
    )
    parser.add_argument(
        "--palette-act",
        type=Path,
        default=None,
        help="Optional ACT palette to enforce slot colors/order",
    )
    return parser


def _expand_inputs(inputs: Iterable[Path], recursive: bool) -> List[Path]:
    files: List[Path] = []
    for path in inputs:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            options = ScanOptions(roots=[path], recursive=recursive)
            files.extend(iter_image_files(options))
        else:
            raise FileNotFoundError(path)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        bg_color = hex_to_rgb(args.bg_color)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        input_files = _expand_inputs(args.inputs, args.recursive)
    except FileNotFoundError as exc:
        parser.error(f"Input path not found: {exc}")

    if not input_files:
        parser.error("No image files found")

    palette_override = None
    if args.palette_act:
        try:
            palette_override = read_act_palette(args.palette_act)
        except (OSError, PaletteError) as exc:
            parser.error(f"Failed to read palette: {exc}")

    successes = 0
    failures = 0
    for file_path in input_files:
        out_dir = args.out or (file_path.parent / "out")
        options = ProcessOptions(
            input_path=file_path,
            output_dir=out_dir,
            canvas_size=args.size,
            bg_mode=args.bg_mode,
            bg_color=bg_color,
            fill_index=0,
            fill_mode="transparent" if args.trans0 else "palette",
            transparent_index=0 if args.trans0 else None,
            max_colors=args.max_colors,
            dither=not args.no_dither,
            target_palette=palette_override,
            preserve_palette_order=bool(palette_override),
        )
        try:
            result = process_sprite(options)
        except (PaletteError, OSError) as exc:
            failures += 1
            print(f"[FAIL] {file_path}: {exc}")
            continue
        successes += 1
        relative = result.output_path.relative_to(out_dir)
        print(f"[OK] {file_path.name} -> {out_dir} ({relative})")

    print(f"Completed {successes} file(s), {failures} failure(s).")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
