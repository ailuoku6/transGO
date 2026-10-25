# Handoff

Resume from `index.md`.

Current state:
- `transformer-go play` launches a localhost human-vs-engine browser app.
- Default is newest `runs/**/*.pt/best.pt` by mtime; current default is `runs/zero-9x9/best.pt`, step 250000.
- Full tests, wheel asset check, browser game flow, and actual 200-simulation engine reply passed.

Most relevant files:
- `transformer_go/play.py`: game and checkpoint service.
- `transformer_go/web.py`: localhost HTTP server and API.
- `transformer_go/static/`: page, styling, browser logic.
- `tests/test_services.py`: current behavior coverage.

Need next:
- None for the requested scope. The verification server is running at `http://127.0.0.1:8766`.

Watch out:
- Preserve existing dirty changes in `.feature_memo/transformer围棋引擎/`.
- The server must remain loopback-only; mobile UI still merits manual visual QA.
