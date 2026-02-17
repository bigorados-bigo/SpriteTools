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
- Project-system MVP in GUI: create/open/save folder-based `.spto` projects (legacy `.spritetools` still supported) with managed sprite import into `sources/sprites` and per-sprite metadata persistence.
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
	Use the classic **File** menu for **Import Sprites... / New Project... / Open Project... / Open Recent / Save / Save Project As...**. (`Save Project As...` snapshots whatever is currently open into a named project folder.) `Open Recent` is a dropdown submenu and shows a placeholder when no history exists. The loaded-images pane keeps **Clear Sprites** plus a read-only project info strip (name/mode/state/folder) so active context is always visible while maximizing workspace area. Click entries to see thumbnails/preview, drag swatches to reorder the palette, and double-click a swatch to recolor it. The preview pane will re-render using the current palette mapping.

## Project Conventions

- Default project folder suffix is `.spto`.
- Legacy project folder suffix `.spritetools` remains supported for opening existing projects.
- Project manifest file is always `project.json` at project root.
- Managed projects copy imported sprites into `sources/sprites` so the project is portable.
- Editable state is persisted in `metadata/sprites.json`; exports should default to `exports/renders`.
- On project open, missing sources can be relinked by scanning a user-selected folder for matching filenames.
- Projects autosave into `backups/autosave/`; if a newer autosave snapshot exists on open, SpriteTools prompts to restore it.
- Autosave reports lightweight status in the status bar (throttled to avoid UI noise).
- Project info tooltip shows last save/autosave UTC timestamp for quick recency checks.
- If autosave recovery is used on open, project info tooltip includes a recovery-source tag for session clarity.
- `Save` is enabled only when an active project is open; `Save Project As...` remains available for snapshotting current workspace state.
- On window close with unsaved project changes, SpriteTools prompts with `Save / Discard / Cancel`.
- File menu actions ship with conventional defaults (`Ctrl+N`, `Ctrl+O`, `Ctrl+S`, `Ctrl+Shift+S`, `Ctrl+Q`) and remain fully settings-backed via `bindings/*`.
- Use `Edit > Keyboard Shortcuts...` (`Ctrl+Alt+K`) to remap commands with live key capture, conflict checks, and reset options.
- Merge Mode actions are also bindable (apply, source/destination tagging, clear actions, scope switching, view settings, close).
- Keyboard shortcuts support JSON profile export/import from the same dialog.
- Closing the shortcuts dialog with unsaved changes prompts `Save / Discard / Cancel`.

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

## Release QA Checklist (Shortcuts + Merge)

- Open `Edit > Keyboard Shortcuts...`, change one shortcut, save, reopen app, verify persistence.
- Trigger a conflict by assigning a used shortcut and verify conflict blocking message.
- Export shortcuts to JSON, import JSON, verify bindings are applied.
- Close shortcuts dialog with unsaved changes and verify `Save / Discard / Cancel` prompt.
- Open Merge Mode and verify configured bindings for apply/tag/clear/scope/view/close all execute correctly.
- Reopen Merge Mode and verify no stale dialog state or shortcut regressions.

## GitHub upload and release

Repository remote:

- `https://github.com/bigorados-bigo/SpriteTools`

After staging/committing locally, push `main` and publish a release tag:

```pwsh
git push -u origin main
git tag -a vX.Y.Z -m "SpriteTools vX.Y.Z"
git push origin vX.Y.Z
```

Release automation is now tag-driven via GitHub Actions:

- `.github/workflows/ci.yml` validates pushes and pull requests to `main`.
- `.github/workflows/release.yml` builds and publishes release assets when a `v*` tag is pushed.

Published release assets:

- `SpriteTools-vX.Y.Z.exe`
- `SpriteTools-win64.zip`

Optional manual fallback (if automation is unavailable):

```pwsh
python -m PyInstaller --noconfirm --clean SpriteTools.spec
Copy-Item dist/SpriteTools.exe dist/SpriteTools-vX.Y.Z.exe -Force
Compress-Archive -Path dist/SpriteTools.exe -DestinationPath dist/SpriteTools-win64.zip -Force
gh release create vX.Y.Z dist/SpriteTools-vX.Y.Z.exe dist/SpriteTools-win64.zip --title "SpriteTools vX.Y.Z"
```
