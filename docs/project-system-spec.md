# SpriteTools Project System Spec (Non-Destructive)

## Status
v0.2 (partially implemented)

Implemented in current build:
- Folder-based project manifests (`project.json`) with create/open/save.
- Managed-mode source copy into `sources/sprites` on import/load.
- Per-sprite metadata persistence (`metadata/sprites.json`) for offsets/canvas/local overrides.
- `Save As` workflow for project duplication/relocation.

Project folder suffix:
- Default: `.spto`
- Legacy compatibility: `.spritetools`

## Objective
Define a robust project model so SpriteTools can manage large workflows safely without destructive edits to original sprite assets.

---

## Design Goals

1. Non-destructive by default.
2. Reproducible renders from project state.
3. Fast reopen/load for large sprite sets.
4. Explicit separation of:
   - source assets,
   - edit metadata,
   - temporary caches,
   - final exports.
5. Backups/recovery that do not clutter source folders.

---

## Project Modes

## A) Managed Project (recommended default)
- SpriteTools imports/copies selected source sprites into the project source folder.
- Best for portability and archival.

## B) Linked Project (advanced)
- Project references external source files by absolute/relative paths.
- Faster setup, but depends on external file stability.

Both modes can share the same metadata format with a source resolver layer.

---

## Proposed Project Folder Layout

MyProject.spto/
- project.json                 # project manifest (versioned)
- metadata/
  - sprites.json               # per-sprite metadata (offsets, tags, pivots, notes)
  - animations.json            # animation tags/timelines/durations
  - palette-state.json         # palette + remap state
  - ui-state.json              # optional workspace UI state (layout, view prefs, optional keybinding overrides)
- sources/
  - sprites/                   # imported originals (managed mode)
- cache/
  - thumbs/                    # thumbnails and preview caches
  - analysis/                  # optional pivot/onion/helper precomputes
- backups/
  - snapshots/                 # optional point-in-time project snapshots
  - autosave/                  # rolling autosave files
- exports/
  - renders/                   # final output PNG/ACT (or configured path)

Notes:
- `cache/` should be safe to delete and rebuild.
- `sources/` is immutable after import unless explicit replace workflow is used.

---

## Non-Destructive Data Model

Edits should be represented as metadata transforms, not pixel overwrites.

Per-sprite editable state examples:
- offset_x / offset_y
- canvas override
- palette remap bindings
- animation tag membership
- pivot metadata (future)
- notes/flags

Render path should apply transforms in deterministic order at export/preview time.

---

## Manifest (project.json) Outline

- schema_version
- project_name
- created_at / updated_at
- project_mode (`managed` | `linked`)
- source_index:
  - sprite_id
  - relative_path (managed) or linked path (linked)
  - hash/fingerprint
  - dimensions
- settings:
  - default export settings
  - palette behavior
  - preview behavior
- pointers:
  - metadata files
  - cache policy

Versioning rules:
- Always include `schema_version`.
- Add forward migration hooks for future versions.

---

## Backup and Recovery Strategy

## Autosave
- Trigger on meaningful edit events and timed intervals.
- Keep rolling N autosaves (e.g. 10–20).
- Store in `backups/autosave/`.

## Snapshot
- Manual “Create Snapshot” command.
- Optional pre-major-operation snapshot.
- Store in `backups/snapshots/`.

## Restore
- “Restore from Autosave/Snapshot” dialog with timestamp + brief diff summary.

---

## Source Integrity Rules

On project open:
- Verify source files exist.
- Verify hash/fingerprint (or size+mtime fallback for speed mode).
- Mark entries as:
  - OK,
  - Missing,
  - Changed.

Provide resolution tools:
- relink file,
- relink folder,
- accept external change and refresh cache.

---

## Export Model

- Keep final renders separate from source and metadata.
- Default to project-local `exports/renders/`.
- Allow user override to external output folder.
- Export should be deterministic from project state + source files.

---

## Performance Considerations

- Do not load full pixel data for all sprites at project open.
  - Build lazy loading + thumbnail-first index.
- Cache derived preview assets aggressively under `cache/`.
- Invalidate cache selectively on metadata/source changes.
- Keep heavy analyses async (worker thread).

---

## MVP Implementation Scope (Phase A)

1. `project.json` schema + loader/saver.
2. Create/open/save project commands.
3. Managed mode import into `sources/sprites`.
4. Metadata persistence for current existing editable state.
5. Export target separation (`exports/renders`).
6. Basic autosave + recovery prompt on crash/dirty reopen.

Out of MVP:
- full snapshot UX,
- linked mode integrity tooling depth,
- advanced migration UI.

---

## Open Decisions

1. Hash policy:
   - full content hash vs fast fingerprint + optional deep verify.
2. Autosave cadence for very large projects.
3. Linked mode portability rules across machines.
4. Keybinding scope model:
  - global-only vs optional per-project override layer.

---

## Recommendation

Ship folder-based managed projects first.

Reason:
- easy to inspect/debug,
- simplest recovery story,
- least risky for large sprite workflows,
- cleanly supports future advanced features (animation tags, onion skin, pivot data).
