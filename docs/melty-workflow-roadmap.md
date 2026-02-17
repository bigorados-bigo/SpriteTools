# SpriteTools Melty Workflow Roadmap (Living Plan)

## Purpose

This document defines a long-term product plan for SpriteTools focused on high-volume Melty Blood sprite modding workflows, while preserving current project structure and existing stable features.

This is a living plan and should be updated as workflow details evolve.

---

## Workflow Context (Current Understanding)

- Creator works with very large sprite sets (example given: ~2700 sprites in a character workflow).
- Sprites are organized so index ranges map to expected animation/state usage in Hantei.
- Practical goal is fast replacement/alignment of many frames while preserving animation consistency and pivot quality.

### Product Direction

SpriteTools should evolve from a palette/canvas utility into a lightweight sprite workflow assistant for:

1. frame grouping/tagging,
2. sequence preview and visual alignment,
3. pivot/offset assistance,
4. batch-safe operations across large collections,
5. non-destructive project-based workflows.

---

## Product Principles

1. Keep current core stable.
   - Do not break existing palette/edit/export pipeline.
2. Add advanced features as opt-in layers.
   - New capabilities should be additive, not disruptive.
3. Preserve responsiveness on large datasets.
   - UI must stay smooth with thousands of sprites.
4. Be explicit and reversible.
   - Every major edit path should support undo/history when feasible.
5. Keep data portable.
   - Tagging and metadata should be exportable/importable.
6. Keep source assets immutable by default.
  - Original sprites should never be overwritten during normal editing.

---

## Planned Capability Areas

## 0) Non-Destructive Project System (Foundational)

### Goal
Introduce a first-class project format/folder workflow where SpriteTools edits are represented as metadata and derived outputs, while preserving original source sprites.

### Core Features
- Create/open/save SpriteTools projects.
- Project folder structure with clear separation:
  - source imports,
  - metadata/state,
  - derived caches,
  - final rendered exports.
- Non-destructive edit model:
  - palette/offset/tag/onion/pivot changes are persisted as project metadata,
  - source files remain untouched.
- Recovery/safety features:
  - autosave,
  - optional project snapshots,
  - integrity checks for missing/changed source files.

### Why Foundational
This unlocks safer large-scale workflows, easier iteration, and better reproducibility for all later advanced features.

---

## 1) Animation Tagging and Sequence Preview

### Goal
Allow users to assign sprite frames to named animation groups and preview playback directly in editor.

### Core Features
- Create animation tags (name + optional category/state).
- Assign selected sprites/indices to an animation tag.
- Define ordered frame list and per-frame duration.
- Preview playback controls:
  - play/pause,
  - FPS override,
  - loop,
  - step frame forward/back.

### Suggested Data Model
- AnimationTag
  - id
  - name
  - optional state label
  - list of FrameEntry
- FrameEntry
  - sprite_key (or index reference)
  - duration_frames
  - optional notes

### UX Placement (Recommended)
- Phase 1: integrate in main view with collapsible panel.
- Phase 2: optional dedicated timeline/animation workspace if needed.

---

## 2) Onion Skin System

### Goal
Make alignment and frame-to-frame continuity much easier during offset/pivot adjustments.

### Core Features
- Configurable onion skin frame window:
  - frames before (N),
  - frames after (N).
- Independent appearance controls:
  - color tint for previous frames,
  - color tint for next frames,
  - opacity strength,
  - optional blend mode style presets.
- Toggle onion skin on/off quickly.

### Performance Requirements
- Onion skin should use cached transformed layers.
- Redraw only affected regions when frame changes.
- Cap redraw cadence adaptively under heavy zoom/pan.

---

## 3) Pivot and Centering Toolkit

### Goal
Improve frame alignment beyond manual offset-only workflow.

### Phase 1 (Practical)
- Better manual alignment aids:
  - center guides/crosshair,
  - baseline/anchor guides,
  - ghost reference frame lock.
- Multi-frame nudge operations:
  - apply current offset delta to selected range/tag.

### Phase 2 (Semi-Automatic)
- Auto-suggest pivot/offset using image-analysis heuristics:
  - silhouette centroid,
  - bounding-box anchor matching,
  - edge/contour overlap scoring against reference frame.
- Accept/reject workflow for suggested offsets.

### Phase 3 (Interop)
- Research and support pivot data interoperability with MUGEN-related exports where technically valid.
- Include import/export mapping options for pivot metadata.

---

## 4) Batch Workflow and Organization for Large Projects

### Goal
Make large collection workflows predictable and low-friction.

### Core Features
- Batch assignment tools:
  - assign animation tag to selected range,
  - batch frame duration edits,
  - batch offset apply by tag/group.
- Filtering and navigation helpers:
  - by tag,
  - by index range,
  - by completion status.
- Validation checks:
  - missing frames in tag sequence,
  - duplicate references,
  - suspicious offset jumps.

---

## 5) Keybindings and Shortcut Customization

### Goal
Provide a full keybindings management window so users can customize program shortcuts safely and persistently.

### Core Features
- Keybindings window with searchable action list.
- Edit shortcut per action with immediate conflict detection.
- Reset options:
  - reset one action,
  - reset all actions to defaults.
- Import/export keybinding profiles (JSON) for portability.
- Profile-safe persistence through settings + project-aware overrides only where needed.

### UX Requirements
- Conventional desktop flow:
  - File menu entry (e.g., Preferences/Keyboard Shortcuts),
  - explicit Save/Apply/Cancel behavior.
- Clear conflict states:
  - conflicting action shown,
  - one-click resolve or cancel.

### Technical Constraints
- Build on existing shortcut registration paths (do not fork input handling).
- Keep startup lightweight:
  - load resolved bindings once,
  - avoid per-frame lookup overhead.
- Maintain backward compatibility with default bindings if user config is missing/invalid.
- Shortcut policy for ongoing development:
  - every new keyboard shortcut and modifier-driven interaction should be represented by a bindable action entry,
  - gesture affordances (such as sprite-browser zoom controls) should map to the same bindable zoom actions whenever practical.

---

## 6) Workspace Panels + Explorer-Grade Sprite Browser

### Goal
Make sprite and group workflows feel like a modern file explorer: fast browsing, flexible organization, and adaptable panel layout (floating first, dockable where viable).

### Core Features (Browser UX)
- Multiple sprite browser display modes:
  - details/list,
  - thumbnails grid,
  - compact tiles/strip.
- Adjustable thumbnail zoom with smooth scaling and lazy thumbnail loading.
- Organization controls:
  - sort by name/index/group/modified,
  - group by folder/tag/group id,
  - optional pin/favorites filter for active working sets.
- Navigation affordances:
  - keyboard-first selection navigation,
  - breadcrumb-like context label,
  - quick jump/search.

### Panel Layout Strategy

#### Phase 1 (Lower Risk): Floating Panel Variants
- Add optional floating versions for sprite list and group panel.
- Keep current main-window splitter as canonical layout.
- Preserve shared selection/model state between docked and floating presentations.

#### Phase 2 (Higher Depth): Dockable Workspace
- Evaluate migration of major panes to `QDockWidget`-based layout.
- Support save/restore of custom workspace layouts.
- Add reset-to-default-layout command.

### Technical Viability Notes
- Floating panels are highly viable in current architecture and can ship incrementally.
- Full dockable workspace is viable in Qt, but is a deeper refactor because current UI relies on tightly-coupled splitter layouts and cross-panel interaction assumptions.
- Recommendation: ship floating + explorer-browser improvements first, then perform dockable migration only after interaction/state synchronization is stabilized.

---

## Roadmap Phases

## Phase A — Foundation (Low Risk)

- Add project container + metadata persistence layer (project-side JSON).
- Add source/import tracking and non-destructive edit state persistence.
- Add animation tag CRUD and assignment UI.
- Add basic sequence preview player.
- Keep all existing operations unchanged.
- Establish shortcut action registry for all user-invokable commands (foundation for keybindings editor).

Exit criteria:
- User can save/reopen project state without modifying source assets.
- User can define animations and preview loops with durations.

## Phase B — Visual Alignment Upgrade

- Add onion skin controls and rendering.
- Add alignment guides and reference frame tools.
- Add batch offset apply by tag.
- Ship keybindings window MVP:
  - list/edit/reset actions,
  - conflict detection,
  - persisted user keymap.
- Ship sprite browser UX baseline:
  - list + thumbnail modes,
  - zoom slider,
  - basic sort/group options.
- Add optional floating sprite/group panels (non-dockable first pass).

Exit criteria:
- User can align animation frames significantly faster than manual single-frame edits.

## Phase C — Smart Pivot Assistance

- Implement offset suggestion heuristics.
- Add confidence/preview/accept workflow.
- Add optional bulk-suggest mode for selected tag.

Exit criteria:
- User can semi-automatically align a sequence, then fine-tune manually.

## Phase D — Interop + Power Workflow

- Add pivot metadata import/export pathways (including MUGEN-oriented mapping if validated).
- Add project validation tooling and reporting.
- Add optional dedicated timeline workspace if main view becomes crowded.
- Add keybinding profile import/export and optional multiple named keymap profiles.
- Evaluate and optionally ship full dockable workspace layout (`QDockWidget`) with layout preset save/restore.

Exit criteria:
- End-to-end animation prep is manageable for large projects with minimal manual repetition.

---

## Architectural Guidance (Do Not Break Core)

- Keep processing pipeline modular:
  - palette/canvas/export pipeline remains isolated from animation metadata features.
- Add metadata service separate from image processing:
  - avoid coupling frame tags to raw image mutation logic.
- Rendering strategy:
  - continue cached base layers + incremental overlays,
  - avoid full-scene recomposition on every playback tick.
- Asynchronous tasks:
  - heavy analysis (pivot suggestion, bulk validation) runs in worker threads,
  - UI updates via event-driven completion.

---

## Initial Technical Backlog (Suggested)

1. Define project schema file (versioned).
2. Add project create/open/save and migration scaffolding.
3. Add source import/index tracking and integrity checks.
4. Animation Tag panel (CRUD + assignment).
5. Playback engine (durations + loop + stepping).
6. Onion skin renderer with configurable before/after counts.
7. Guide overlays and reference-lock frame.
8. Batch offset tools by selection/tag.
9. Pivot auto-suggest prototype.
10. Validation report panel.
11. Shortcut action registry + metadata model (category/label/default binding).
12. Keybindings window (edit/reset/conflict resolution).
13. Keybinding profile import/export support.

---

## Open Questions (For Next Planning Pass)

1. Should animation tags be index-range based, file-key based, or support both?
2. Do you want one global timeline per project or multiple named timelines?
3. For onion skin, should previous/next frame sets have independent opacity curves?
4. For auto-pivot, what is preferred anchor priority:
   - feet/base,
   - torso center,
   - custom marker?
5. Should projects copy sources into project folder, reference external sources, or support both modes?
6. How aggressive should autosave/snapshot retention be for large projects?
7. Should keybindings be global-only, or allow optional per-project override layer?

---

## Near-Term Execution Recommendation

Start with Phase A + minimal Phase B subset:

- Animation tags,
- duration-aware preview,
- basic onion skin,
- simple guides,

then iterate with real workflow feedback before deeper automation.
