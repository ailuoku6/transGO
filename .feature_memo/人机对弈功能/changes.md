# Changes

## 2026-10-25

- Changed: Added local Python HTTP service, checkpoint inventory/default selection, human-vs-engine game service, responsive SVG board UI, CLI `play` command, SGF/analyze/game controls and README instructions. Added service tests.
- Changed: Engine default search budget is used through an explicit `模型默认` selector, so checkpoints with different search settings work without UI assumptions. Fixed white-side undo history, corrected game coordinate turn label and resignation text, and adjusted short-viewport board sizing.
- Verification: `.venv/bin/python -m pytest -q` => 151 passed, 5 skipped; Node JS syntax and Python `py_compile` pass; `pip wheel . --no-deps --no-build-isolation` succeeds and wheel contains the three static assets. Browser on localhost loaded 88 weights with `runs/zero-9x9/best.pt` default (step 250000); opening game and a human move produced a legal engine reply in 0.60 s using 23/200 visits. HTTP GET for assets and `/api/state` succeeded.
- Known issues: In-app browser tab opened at desktop size; mobile layout was not manually exercised. Port 8765 was occupied, so verification server is on 8766.
- Next: None for this implementation.
