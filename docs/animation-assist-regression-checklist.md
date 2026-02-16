# Animation Assist Regression Checklist

Use this checklist after preview/assist rendering changes.

## Setup

1. Launch with debug logging enabled.
2. Load a project with:
   - an animation tag with at least 6 frames,
   - visible palette indices for highlight testing,
   - non-zero offsets on at least 2 frames.
3. Enable onion skin and highlight overlay.

## Test Matrix

Mark each item `PASS` / `FAIL`.

### 1) Drag Target Correctness `PASS`

- In `animation_assist`, start playback and move to frame N via timeline/playhead.
- Drag offset in `group` mode.
- Confirm the *visible* frame moves, not a stale selected sprite-list row.

Expected:

- Drag applies to the visible assist frame.
- No frame mismatch during playback or after manual seek.

  Verified after drag-target/group session lock fix.

### 2) Onion Layer Separation `PASS`

- Keep onion enabled with previous/next frames visible.
- Drag current frame around.

Expected:

- Onion remains static.
- Only current frame layer moves.

Confirmed stable after group-drag fix.

### 3) Zoom + Drag Persistence  `PASS`

- Drag frame to a clear non-zero offset.
- Change zoom level (in/out), pan, then return zoom.

Expected:

- Offset is preserved.
- No snap-back to stale/cached transform.

### 4) Highlight + Playback Stability `PASS`

- Keep highlight enabled while playing animation.
- Scrub timeline, select different highlight indices, continue playback.

Expected:

- No flicker, no baked/stuck overlay.
- Highlight follows current frame correctly.

### 5) Mode Switch Safety `PASS`

- Switch `sprite_edit` -> `animation_assist` -> `sprite_edit` repeatedly.
- Do this while onion/highlight are enabled and playback has run at least once.

Expected:

- No stale overlay residue on switch.
- No corrupted frame composition.

### 6) Playback in Sprite Mode `PASS`

- Start playback in sprite mode.
- Toggle onion/highlight controls where applicable.

Expected:

- Playback remains stable.
- No broken preview or stuck visuals.

Expected behavior for assist playback workflow.

## Log Health Checks

After the run, inspect `sprite_tools_debug.log`.

Expected:

- No `Traceback`, `Exception`, `ERROR`, `CRITICAL`.
- No recurrence of previously-removed high-frequency palette spam signatures.

## Sign-off

- Overall result: `PASS`
- Build verified with: `python -m py_compile sprite_tools/ui/main.py`
- Notes:
  - Verified on 2026-02-16 after group drag target/session lock and group offset persistence fix.
