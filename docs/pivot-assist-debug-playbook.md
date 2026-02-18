# Pivot Assist Debug Playbook

Use this when Pivot Assist appears to do nothing or over/under-corrects.

## 1) Capture baseline report

In Pivot Assist:

1. Press `Recompute`.
2. Press `Copy Debug Report`.
3. Save/paste the report.

Key signals:

- `raw=(0,0)` across all frames means the solver sees identical anchors (likely foreground mask issue).
- `raw != 0` but `smooth ~ 0` means smoothing/jump clamp is flattening movement.
- `smooth != 0` but `final` too small means gain too low.

## 2) Recommended knob order

1. `Mask`: switch from `Auto` to `Border FG`.
2. `Anchor`: try `Foot` then `Hybrid`.
3. `Reference`: try `Current Frame` or `Median`.
4. Lower `Smoothing`.
5. Increase `Max Jump`.
6. Increase `Gain` (120-200%).

## 3) Custom source references

If first frame is not a good anchor:

1. Set `Reference` to `Picked Sources Median`.
2. Press `Pick Sources...`.
3. Use search + thumbnails to select one or more source frames.
4. Recompute and compare output.

## 4) Typical failure patterns

### Pattern A: All zeros

- Cause: foreground extraction sees near-identical shape each frame.
- Fix: `Mask=Border FG`, `Anchor=Foot`, `Reference=Current Frame`.

### Pattern B: Last frame still not catching up

- Cause: heavy smoothing + small max jump.
- Fix: reduce smoothing (20-45), max jump 8-24, gain 130-180%.

### Pattern C: Jittery offsets

- Cause: anchor too reactive and low smoothing.
- Fix: `Anchor=Hybrid`, smoothing 35-65, max jump 6-14.

## 5) Future algorithm expansion

Planned advanced solvers:

- phase correlation,
- template matching,
- hybrid consensus with confidence gating.

Current solver remains deterministic and fully tunable while these are added.