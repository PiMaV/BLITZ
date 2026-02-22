# Markdown State Overview (for Cleanup)

Reference for the next MD reorganisation step. All paths relative to repo root.

---

## Root-Level (5 files)

| File | Purpose |
|------|---------|
| `README.md` | Main project readme. Links: docs/walkthrough.md, docs/Tabs_explained.md, docker/README.md |
| `TODO.md` | Project TODO list. References: docs/TIMELINE_AGGREGATION.md, docs/SOURCES_AND_VARIANTS.md, docs/SETTINGS.md, docs/OPTIMIZATION.md |
| `CONTRIBUTORS.md` | List of contributors |
| `pyinstaller_command.md` | Short CLI snippets for uv + PyInstaller (dev/build notes) |

---

## docker/ (1 file)

| File | Purpose |
|------|---------|
| `docker/README.md` | Docker setup and usage (moved from root DOCKER.md) |

---

## docs/ (18 files)

### Architecture & Design

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | Code structure, data/UI layers. Links: SOURCES_AND_VARIANTS.md, TIMELINE_AGGREGATION.md |
| `SOURCES_AND_VARIANTS.md` | Build variants (Standard/Full), Loader/Converter/Handler. Links: ARCHITECTURE.md, LOADING.md, LIVE_AND_MOCK.md |

### Build & CI

| File | Purpose |
|------|---------|
| `build_workflow.md` | GitHub Actions build triggers, artifacts, local build/ and dist/ |

### Features & UX

| File | Purpose |
|------|---------|
| `walkthrough.md` | Quick Start – basic usage with screenshots |
| `Tabs_explained.md` | Core functionalities – all tab options explained |
| `LOADING.md` | Load flow, dialogs, session defaults |
| `TIMELINE_AGGREGATION.md` | Timeline panel, Aggregate mode |
| `Extraction_Envelopes.md` | H/V extraction plot envelopes |
| `LIVE_AND_MOCK.md` | Live sources (Cam Mock, Webcam), ring-buffer |
| `SETTINGS.md` | Settings and project storage strategy (discussed, deferred) |

### Performance & Benchmarks

| File | Purpose |
|------|---------|
| `OPTIMIZATION.md` | Performance analysis, bottlenecks. Links: MULTICORE_AND_PROGRESS.md, NUMBA_CANDIDATES.md, AUTOGRAD_POTENTIAL.md |
| `MULTICORE_AND_PROGRESS.md` | Multicore loading benchmarks, progress bar |
| `NUMBA_CANDIDATES.md` | Numba optimization candidates |
| `AUTOGRAD_POTENTIAL.md` | Lightweight autograd concept |
| `BENCHMARK_LOGIC.md` | boot_bench thresholds, parallel vs sequential |

### Internal / Technical

| File | Purpose |
|------|---------|
| `UNRAVEL_AND_STORES.md` | unravel(), Raw/Result store semantics |
| `MONITORING_AND_SENSORS.md` | CPU/RAM/disk monitoring, psutil |

---

## Link Graph (inbound references)

```
README.md        → walkthrough.md, Tabs_explained.md, docker/README.md
TODO.md          → TIMELINE_AGGREGATION, SOURCES_AND_VARIANTS, SETTINGS, OPTIMIZATION
ARCHITECTURE.md  → SOURCES_AND_VARIANTS, TIMELINE_AGGREGATION
SOURCES_AND_VARIANTS → ARCHITECTURE, LOADING, LIVE_AND_MOCK
OPTIMIZATION.md  → MULTICORE_AND_PROGRESS, NUMBA_CANDIDATES, AUTOGRAD_POTENTIAL
```

---

## Suggested Groupings (for next step)

- **docs/build/** – build_workflow.md, pyinstaller_command.md (from root)
- **docs/architecture/** – ARCHITECTURE.md, SOURCES_AND_VARIANTS.md
- **docs/features/** – walkthrough, Tabs_explained, LOADING, TIMELINE_AGGREGATION, Extraction_Envelopes, LIVE_AND_MOCK, SETTINGS
- **docs/optimization/** – OPTIMIZATION, MULTICORE_AND_PROGRESS, NUMBA_CANDIDATES, AUTOGRAD_POTENTIAL, BENCHMARK_LOGIC
- **docs/internal/** – UNRAVEL_AND_STORES, MONITORING_AND_SENSORS

Root: README.md, TODO.md, CONTRIBUTORS.md remain at top level.
