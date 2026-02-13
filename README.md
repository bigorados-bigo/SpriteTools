# SpriteTools (prototype)

Utilities for batch-editing indexed PNG sprites with palette-aware rules inspired by the bundled Aseprite script.

## Preview

![SpriteTools UI preview](docs/preview.png)

> Place your screenshot file at `docs/preview.png` so GitHub renders this preview image.

## Features

- Automatically quantizes RGB/RGBA sprites down to indexed mode (up to 256 colors) before palette operations.
- Detects likely background palette index (auto/manual/sample/alpha modes) via border sampling.
- Reorders palette entries so the background color lands in slot 0 while preserving relative order.
- Optionally forces palette index 0 to be transparent.
- Pads/crops sprites onto a square canvas and exports both PNG and Adobe ACT palette files.
- Simple CLI that works on individual files or entire folders, with optional ACT palette enforcement.
- PySide6 GUI prototype with a loaded-images list, detected-colors pane that supports drag & drop reorder plus color editing (double-click), and a live preview pane that reflects palette tweaks immediately.
- Drag-and-drop support: drop individual images or entire folders onto the window to import sprites quickly.
- Custom canvas sizing (including smart cropping when the canvas is smaller than the sprite).
- Ctrl+Z / Ctrl+Y undo and redo for palette edits, slot reordering, fill modes, and canvas settings.

## Requirements

- Python 3.11+
- Dependencies from `requirements.txt` (activate the provided virtual environment or install manually)

```pwsh
python -m pip install -r requirements.txt
```

## Command-line usage

```pwsh
python -m sprite_tools.cli input_folder --recursive --size 256 --bg-mode auto --trans0 --max-colors 128
```

Arguments of note:

- `inputs`: one or more PNG files or directories.
- `--size`: target square canvas (default 256).
- `--bg-mode`: `auto`, `manual`, `sample`, or `alpha`.
- `--bg-color`: hex color used when mode is `manual`/`sample` (default `#ff00ff`).
- `--trans0`: force palette index 0 to be transparent.
- `--out`: custom output directory (defaults to `<input parent>/out`).
- `--max-colors`: cap colors retained during quantization (default 256).
- `--no-dither`: disable Floyd–Steinberg dithering when quantizing.
- `--palette-act`: supply an ACT palette to enforce identical palette slots across every sprite.

Each processed sprite is saved to `{output}/{original_name}.png` alongside a `{original_name}.act` palette dump.

## Current testing workflow

- **CLI smoke test** (verifies quantization and palette export):
	```pwsh
	python -m sprite_tools.cli path\to\sprites --size 128 --bg-mode auto
	```
	Check that `out/` contains updated PNG + ACT pairs and the console prints `[OK]` lines.
- **GUI prototype** (launches PySide6 workspace with loaded-images pane, palette grid, and preview panel):
	```pwsh
	python -m sprite_tools.ui
	```
	Drag files/folders directly onto the window or use **Load Images**. Click entries to see thumbnails/preview, drag swatches to reorder the palette, and double-click a swatch to recolor it. The preview pane will re-render using the current palette mapping.

## Building a shareable GUI bundle

Install the runtime and build dependencies (PyInstaller is listed in `requirements-dev.txt`):

```pwsh
python -m pip install -r requirements-dev.txt
```

Then generate a self-contained EXE (no companion folders or DLL hunts required):

```pwsh
python -m PyInstaller --noconfirm --clean SpriteTools.spec
```

The command drops `dist/SpriteTools.exe`, which already embeds Python, Qt, and Shiboken. Share the single file directly or wrap it in a zip if you prefer:

```pwsh
Compress-Archive -Path dist/SpriteTools.exe -DestinationPath SpriteTools-win64.zip -Force
```

PyInstaller still prints warnings about optional Qt SQL/Designer helpers (OCI, LIBPQ, etc.); SpriteTools does not load those plugins, so the warnings can be ignored.

## Next steps

- Add a diagnostics pane that shows palette index usage across all loaded sprites.
- Provide preset canvas sizes plus per-sprite overrides.
- Optional dithering controls inside the GUI preview.

## GitHub upload and release

Repository remote:

- `https://github.com/bigorados-bigo/SpriteTools`

After staging/committing locally, push `main` and publish a release tag:

```pwsh
git push -u origin main
git tag -a v0.2.0 -m "SpriteTools v0.2.0"
git push origin v0.2.0
```

Then create a GitHub Release from tag `v0.2.0` (or create it with GitHub CLI if installed).
