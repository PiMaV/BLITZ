# Documentation Index

Overview of all documentation files in this repository.

## Start Here

| File | Purpose |
|------|---------|
| [`README.md`](../README.md) | Main project overview, installation, and quick links. |
| [`docs/walkthrough.md`](walkthrough.md) | Quick Start Guide with screenshots. |
| [`docs/tabs_explained.md`](tabs_explained.md) | **Features & Tabs:** Comprehensive user guide for all UI elements. |
| [`docs/missing_features.md`](missing_features.md) | **Missing/Hidden:** Overview of features not present in this build (e.g. crop widget, OMERO). |

## Architecture & Design

| File | Purpose |
|------|---------|
| [`docs/architecture.md`](architecture.md) | High-level code structure, data/UI separation. |
| [`docs/sources_and_variants.md`](sources_and_variants.md) | Standard vs. Full build explanation, Loader/Handler architecture. |
| [`docs/loading.md`](loading.md) | Detailed flow of data loading, dialogs, and session defaults. |
| [`docs/unravel_and_stores.md`](unravel_and_stores.md) | Technical details on internal data storage (Raw/Result stores). |

## Feature Deep Dives

| File | Purpose |
|------|---------|
| [`docs/timeline_aggregation.md`](timeline_aggregation.md) | Details on the Timeline panel and aggregation modes. |
| [`docs/extraction_envelopes.md`](extraction_envelopes.md) | Explanation of the H/V extraction plot envelopes. |
| [`docs/live_and_mock.md`](live_and_mock.md) | Technical details on Live Sources (Ring Buffer, Camera, Lissajous). |
| [`docs/settings.md`](settings.md) | Discussion on settings and project storage strategy. |
| [`docs/monitoring_and_sensors.md`](monitoring_and_sensors.md) | CPU/RAM/Disk monitoring via `psutil`. |

## Performance & Optimization

| File | Purpose |
|------|---------|
| [`docs/optimization.md`](optimization.md) | **Main Report:** Current bottlenecks, Numba usage, video loading strategies. |
| [`docs/benchmark_logic.md`](benchmark_logic.md) | Logic behind benchmarking and parallel vs sequential decisions. |
| [`docs/multicore_and_progress.md`](multicore_and_progress.md) | Multicore loading benchmarks and progress bar implementation. |
| [`docs/numba_candidates.md`](numba_candidates.md) | List of functions identified for JIT acceleration. |

## Concepts & Future Work

| File | Purpose |
|------|---------|
| [`docs/autograd_potential.md`](autograd_potential.md) | Concept for a lightweight autograd engine for parameter tuning. |
| [`TODO.md`](../TODO.md) | Active development tasks and known issues. |

## Build & CI

| File | Purpose |
|------|---------|
| [`docs/build_workflow.md`](build_workflow.md) | **Release checklist**, GitHub Actions workflow, how to trigger a build. |
| [`docs/pyinstaller_command.md`](pyinstaller_command.md) | Developer notes for building the executable. |
| [`docs/smoke_test.md`](smoke_test.md) | **10-Minute Smoke Test:** Manual checklist for release verification. |
| [`docker/README.md`](../docker/README.md) | Docker deployment instructions. |
