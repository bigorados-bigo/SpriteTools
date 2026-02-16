# SpriteTools Release Notes

## v0.3.0 (Main)

### Project System

- Added project-system MVP with create/open/save flows and folder-based manifests (`project.json`).
- Added `Save As` support to duplicate/relocate projects.
- Added `Save As` fallback behavior: when no project is active, user is prompted for a project folder name.
- Updated default project folder suffix to `.spto` (legacy `.spritetools` remains supported).
- Managed projects now copy imported sprites into `sources/sprites` so projects do not rely on original source paths.
- Added autosave snapshots in `backups/autosave` with open-time recovery prompt when autosave is newer than manifest state.

### Planning Updates

- Added long-term roadmap milestone for a full keybindings window (edit/reset/conflict detection/profile portability).

### UX

- Added `Edit > Keyboard Shortcuts...` dialog for in-app shortcut editing.
- Added organized two-column keybindings list (`Action`, `Shortcut`) with live key-capture assignment.
- Added conflict checks and reset options (per-action and reset-all) in keybindings dialog.
- Shortcut updates now apply immediately to menu actions plus Undo/Redo and persist via `QSettings` (`bindings/*`).
- Added extensive Merge Mode bindable actions (apply, source/destination tagging, clear actions, scope switching, view settings, close).
- Added shortcuts profile export/import (`.json`) with validation and conflict blocking on import.
- Added unsaved-change indicator and close confirmation (`Save / Discard / Cancel`) in the keybindings dialog.

### Fixes

- Fixed project open/reset crash caused by stale `status_label` access.
- Added and passed manual QA checklist for shortcuts editor + merge-mode shortcut flows (persistence, conflict handling, import/export, unsaved-close prompt, merge dialog bindings).

### Notes

- This release supersedes `v0.2.0` for current `main`.

## v0.2.0 (Main)

### Highlights

- Added full merge workflow with explicit Source/Destination semantics and a dedicated merge dialog.
- Added scope-aware merge application (`Global`, `Group`, `Local`) with impact/risk visibility.
- Added merge preview improvements including sprite picker, hover highlight, and role coloring.
- Added palette usage/risk overlays in swatch view for safer merge decisions.

### Palette View + Layout

- Added shared view settings across main palette, merge palette, and floating palette windows:
  - Columns
  - Zoom
  - Gap
  - Force Columns
  - Show Indices
  - Show Grid
- Added robust forced-column fitting logic with measured realized-columns correction.
- Added resize/debounce/re-entrancy protections to reduce layout churn.
- Added per-floating-window persisted settings and first-show deferred layout apply.

### Interaction and UX

- Added deterministic click behavior for drag mode buttons:
  - click active mode again to turn it off
  - click another mode to switch mode cleanly
- Updated preview drag/pan behavior to avoid accidental mode conflicts.
- Added clear action and drag/drop loading improvements for floating palette windows.

### Notes

- This release supersedes `v0.1.0` for current `main`.

## v0.1.0 (Main)

### Highlights

- Added tri-scope canvas sizing and sprite offset controls:
  - Global
  - Per Group
  - Local
- Added combined preview behavior with scope-aware resolution and override precedence.
- Decoupled drag target controls from preview scope to avoid hidden coupling.
- Improved undo/redo coverage and determinism for palette, canvas, and offset operations.
- Added deeper debug logging around history record/apply, palette operations, and UI state restoration.
- Reworked palette swatch rendering/selection alignment using canonical delegate geometry.

### Quality and Diagnostics

- History snapshots now restore more complete editing context.
- Undo/redo now reports operation labels in logs/status for easier verification.
- Additional synchronization and preview invalidation fixes after history restore.

### Notes

- If you see a missing preview image on GitHub, add the screenshot at `docs/preview.png`.
- PyInstaller build warnings for optional Qt plugins may still appear and are expected for unused components.
