# Documentation Verification Report

This report outlines the changes made to the documentation to verify that no information was lost during the overhaul.

## 1. Newly Created Files

| File | Purpose | Source of Content |
|------|---------|-------------------|
| `docs/missing_features.md` | Documents hidden, removed, or planned features. | User request ("what is lost along the way"); content extracted from code analysis (Crop Widget visibility, Save/Load removal). |
| `docs/verification_report.md` | This file. | Generated to track changes. |

## 2. Rewritten / Expanded Files

| File | Changes | Verification |
|------|---------|--------------|
| `docs/tabs_explained.md` | **Rewritten.** Renamed to "Features & Tabs Explained". Now covers *all* UI tabs (File, View, Ops, Tools, RoSEE, PCA, Bench, Stream). | **Old Content:** File, View, Time, Tools, RoSEE sections. <br> **New Content:** All old sections kept and expanded. "Time" split into "Timeline Panel" and "Ops Tab". Live/Webcam moved to "Stream Tab". |
| `docs/md_state.md` | **Rewritten.** Now serves as the main Documentation Index. | Previous content was a temporary organization list; new content indexes the final structure. |

## 3. Translated Files (German -> English)

| File | Status | Notes |
|------|--------|-------|
| `docs/sources_and_variants.md` | **Translated & Updated.** | Clarified Standard vs. Full build. Mock/Webcam implementation details summarized (details remain in `live_and_mock.md`). |
| `docs/timeline_aggregation.md` | **Translated.** | Technical details about `ImageData.image_timeline` preserved. |
| `docs/TODO.md` | **Translated & Cleaned.** | Converted mixed language TODOs into a structured English Roadmap. No items dropped. |

## 4. Summarized Files

| File | Changes | Notes |
|------|---------|-------|
| `docs/optimization.md` | **Updated.** Removed raw German debugging logs. | **Preserved:** Conclusions from the logs (e.g., Single-core video loading is better). <br> **Restored:** Insight about "Qt Signal Bottleneck" added back in English. <br> **Added:** Current status of Numba and Threaded Reductions. |

## 5. Unchanged (Context)

| File | Status |
|------|--------|
| `docs/live_and_mock.md` | **Unchanged.** Contains detailed technical specs for Live View. |
| `docs/autograd_potential.md` | **Unchanged.** Concept document. |

## Summary

The documentation is now fully in English. No feature descriptions were lost. Deprecated/Hidden features are now explicitly documented in `missing_features.md`. Technical implementation details (profiling logs) were summarized into architectural insights.
