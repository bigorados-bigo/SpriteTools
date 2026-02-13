# SpriteTools Release Notes

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
