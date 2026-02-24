# Documentation Index

Overview of all documentation files in this repository.

## Start Here

| File | Purpose |
|------|---------|
| [`README.md`](../README.md) | Main project overview, installation, and quick links. |
| [`docs/walkthrough.md`](walkthrough.md) | Quick Start Guide with screenshots. |
| [`docs/Tabs_explained.md`](Tabs_explained.md) | **Features & Tabs:** Comprehensive user guide for all UI elements. |
| [`docs/MISSING_FEATURES.md`](MISSING_FEATURES.md) | **Missing/Hidden:** Overview of features not present in this build (e.g. crop widget, OMERO). |

## Architecture & Design

| File | Purpose |
|------|---------|
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | High-level code structure, data/UI separation. |
| [`docs/SOURCES_AND_VARIANTS.md`](SOURCES_AND_VARIANTS.md) | Standard vs. Full build explanation, Loader/Handler architecture. |
| [`docs/LOADING.md`](LOADING.md) | Detailed flow of data loading, dialogs, and session defaults. |
| [`docs/UNRAVEL_AND_STORES.md`](UNRAVEL_AND_STORES.md) | Technical details on internal data storage (Raw/Result stores). |

## Feature Deep Dives

| File | Purpose |
|------|---------|
| [`docs/TIMELINE_AGGREGATION.md`](TIMELINE_AGGREGATION.md) | Details on the Timeline panel and aggregation modes. |
| [`docs/Extraction_Envelopes.md`](Extraction_Envelopes.md) | Explanation of the H/V extraction plot envelopes. |
| [`docs/LIVE_AND_MOCK.md`](LIVE_AND_MOCK.md) | Technical details on Live Sources (Ring Buffer, Camera, Lissajous). |
| [`docs/SETTINGS.md`](SETTINGS.md) | Discussion on settings and project storage strategy. |
| [`docs/MONITORING_AND_SENSORS.md`](MONITORING_AND_SENSORS.md) | CPU/RAM/Disk monitoring via `psutil`. |

## Performance & Optimization

| File | Purpose |
|------|---------|
| [`docs/OPTIMIZATION.md`](OPTIMIZATION.md) | **Main Report:** Current bottlenecks, Numba usage, video loading strategies. |
| [`docs/BENCHMARK_LOGIC.md`](BENCHMARK_LOGIC.md) | Logic behind benchmarking and parallel vs sequential decisions. |
| [`docs/MULTICORE_AND_PROGRESS.md`](MULTICORE_AND_PROGRESS.md) | Multicore loading benchmarks and progress bar implementation. |
| [`docs/NUMBA_CANDIDATES.md`](NUMBA_CANDIDATES.md) | List of functions identified for JIT acceleration. |

## Concepts & Future Work

| File | Purpose |
|------|---------|
| [`docs/AUTOGRAD_POTENTIAL.md`](AUTOGRAD_POTENTIAL.md) | Concept for a lightweight autograd engine for parameter tuning. |
| [`TODO.md`](../TODO.md) | Active development tasks and known issues. |

## Build & CI

| File | Purpose |
|------|---------|
| [`docs/build_workflow.md`](build_workflow.md) | **Release checklist**, GitHub Actions workflow, how to trigger a build. |
| [`docs/pyinstaller_command.md`](pyinstaller_command.md) | Developer notes for building the executable. |
| [`docker/README.md`](../docker/README.md) | Docker deployment instructions. |
