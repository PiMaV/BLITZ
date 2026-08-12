# Known bugs / UX rough edges

Working notes for things that feel wrong in daily use — **not** the public changelog
(that lives on [GitHub Releases](https://github.com/PiMaV/BLITZ/releases)).

Prefer short repros here or as GitHub Issues. Prioritized roadmap: [`TODO.md`](TODO.md).

---

## SIGSEGV: Network Connect fail → open Log tab (2026-08-12)

**Symptom:** `./BLITZ` / `uv run blitz` dies with
`terminated by signal SIGSEGV (Adressbereichsfehler)`.
No Python traceback in `blitz_crash.log` (native crash).

**Repro (sidecar not required):**
1. Start BLITZ alone (EVT Sidecar / WOLKE **not** running).
2. Network tab → Connect to e.g. `http://127.0.0.1:5055` / token `evt` (or any dead host).
3. Connect fails (expected).
4. Open Options → **Log** tab → often **SIGSEGV**.

**Likely cause:** `blitz.data.web._WebSocket.listen` runs on a `QThread` and calls
`blitz.tools.log()`, which used to mutate `LoggingTextEdit` (and the status
one-liner) **from the worker thread**. Opening the Log tab paints that widget on
the GUI thread → classic Qt thread-affinity crash.

**Fix landed (same day):** `LoggingTextEdit` marshals appends via
`QueuedConnection`; status one-liner updates via `QTimer.singleShot(0, …)`.
See `blitz/tools.py`.

**Still verify manually** after pull/rebuild. If SIGSEGV remains, next suspects:
other `log()` / widget touches from loader/camera threads; pyqtgraph dock paint
during tab switch.

**Agent handoff:** [`docs/agent_handoff_sigsegv_log.md`](docs/agent_handoff_sigsegv_log.md)
