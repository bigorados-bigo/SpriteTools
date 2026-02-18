# Pivot Assist Modes and Tuning (2026-02-17)

This document explains how Pivot Assist currently works, which knobs are available, and how to extend it with stronger algorithms.

## Current Pipeline

For each frame:

1. Analyze alpha shape.
2. Build an anchor point from the chosen anchor mode.
3. Compute raw offset delta against a reference anchor.
4. Smooth + jump clamp across sequence order.
5. Apply response gain.
6. Return final integer offsets.

## Runtime Modes

### Anchor Mode

- `Hybrid` (default)
  - Blends centroid and lower silhouette anchor.
  - Good general default for fighters and mixed motion.
- `Centroid`
  - Uses alpha centroid only.
  - Good when silhouettes stay compact and symmetric.
- `Foot`
  - Uses bbox center X + bbox bottom Y.
  - Good when “ground contact” consistency matters.
- `BBox Center`
  - Uses bounding-box center.
  - Good when centroid is unstable due to sparse alpha artifacts.

### Reference Mode

- `First Frame`
  - Align all frames to frame 1 anchor.
  - Best for deterministic, author-driven reference starts.
- `Median`
  - Align to median anchor of sequence.
  - Best when first frame is not representative.

## Knobs

- `Smoothing` (0-100)
  - Higher = stronger temporal smoothing.
  - Lower = more reactive frame-to-frame movement.
- `Max Jump`
  - Clamp per-frame suggestion jump.
  - Prevents spikes from noisy silhouettes.
- `Gain` (10%-300%)
  - Scales final response after smoothing.
  - Use >100% when output is too weak, <100% when too aggressive.

## Diagnostics

Pivot Assist now emits a per-run debug report with:

- selected settings,
- reference anchor,
- per-frame anchor,
- raw dx/dy,
- smoothed dx/dy,
- final integer suggestion,
- confidence.

Use `Copy Debug Report` in the dialog and paste into issue/debug notes.

## Why “no movement” can happen

Common causes:

- `Smoothing` too high + low `Max Jump` flattening end-of-sequence offsets.
- Anchor mode not matching the motion pattern (for example centroid on lopsided effects).
- Sequence anchored to a poor first frame (`First Frame` mode).
- Gain too low for the displacement scale.

Fast tuning order:

1. Switch `Reference` to `Median`.
2. Try `Foot` or `Hybrid` anchor.
3. Reduce smoothing.
4. Increase max jump.
5. Raise gain to 120-170%.

## External Algorithm Integration Options

These are viable “bring in stronger math” paths.

### 1) Template matching (OpenCV)

- Method: `cv2.matchTemplate` against reference frame or local windows.
- Pros: simple, robust for rigid sprite regions.
- Cons: sensitive to palette/noise unless preprocessed.

### 2) Phase correlation (OpenCV)

- Method: FFT-based translational shift estimation.
- Pros: strong for global translational alignment.
- Cons: assumes mostly rigid translation; less robust on large shape changes.

### 3) Optical flow (OpenCV)

- Method: dense/sparse flow to estimate dominant motion.
- Pros: can handle deformations better.
- Cons: heavier and needs careful filtering for sprite transparency edges.

### 4) Shape registration (contour/IoU scoring)

- Method: optimize dx/dy by maximizing overlap/edge agreement.
- Pros: sprite-specific and interpretable.
- Cons: can be slower without pruning.

## Proposed Engine Modes (next)

Add a top-level `Solver` mode:

- `Anchor (fast)` — current implementation.
- `Phase Correlation` — optional OpenCV backend.
- `Template Match` — optional OpenCV backend.
- `Hybrid Consensus` — weighted combine of 2+ solvers with confidence gating.

## Safe Implementation Plan

1. Keep current anchor solver as always-available fallback.
2. Add optional solver interface and per-solver confidence score.
3. Integrate one external solver behind feature flag.
4. Emit solver-wise debug data in same report format.
5. Keep all modes deterministic and sequence-order aware.