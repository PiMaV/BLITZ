# Agent handoff — BLITZ SIGSEGV on Log after failed Network Connect

## Context

WETTER suite. **BLITZ** is the viewer. **Do not translate** tool brands
(WETTER / WOLKE / DAMPF / KEIM / BLITZ). Chat with the human in **German**;
repo artifacts in English.

Related work (separate): EVT Sidecar under `../EVT/` feeds BLITZ via the WOLKE
Socket.IO + HTTP `.npy` contract. This crash was observed **without** the
sidecar running.

## Bug

- CLI: `fish: Job 1, './BLITZ' terminated by signal SIGSEGV`
- Repro: Connect to dead Network endpoint → open Options **Log** tab → segfault
- No Python `blitz_crash.log` entry (native)

## Suspected root cause

`WebDataLoader` / `_WebSocket.listen` logs from a **background QThread**.
`LoggingTextEdit.log` used to edit `QTextEdit` off the GUI thread.

## Already changed (check git / working tree)

- `blitz/tools.py` — queued log bridge + one-liner marshal
- `blitz/data/web.py` — longer download timeout, fresh download QThread per push,
  failed download no longer tears down the socket via `image_received(None)`
- `blitz/settings.py` — `web/download_timeout` (120)

## Your job

1. Confirm the Log/Connect SIGSEGV is gone with the tools.py fix (manual repro above).
2. If still crashing: audit all `log()` / widget updates from non-GUI threads
   (loaders, camera, web). Prefer signals/`QTimer.singleShot` / `QueuedConnection`.
3. Do **not** start BLITZ for the human unless they ask; they run the GUI themselves.
4. Do not bloat Flatpak with EVT/Metavision — Network contract only.

## Useful paths

- `blitz/tools.py` — `LoggingTextEdit`, `log()`
- `blitz/data/web.py` — Network client
- `BUGS.md` — this bug entry
- `TODO.md` — roadmap
