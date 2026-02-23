# Optimization Report

## Current Performance Analysis

BLITZ is designed for high-performance in-memory analysis. The current implementation uses `multiprocessing` to parallelize image loading and Numba for JIT-compiled critical paths.

### Key Strengths
*   **Parallel Loading**: Uses `multiprocessing.Pool` to load and resize images concurrently (for image sequences).
*   **Memory Guard**: Automatically downsamples (subsets) data if the estimated size exceeds the user-defined RAM limit.
*   **JIT Compilation**: Uses `numba` to accelerate pixel-wise operations and sliding window calculations.
*   **Fast Operations**: Uses `numpy` for vectorized operations (normalization, reduction).

## Implemented Optimizations

### 1. Numba Acceleration (`blitz/data/optimized.py`)
To overcome Python's loop overhead and NumPy's memory overhead for complex chains of operations, we use Numba to JIT-compile "hot paths":

*   **Fused Pipeline (`apply_pipeline_fused`):**  Combines Subtraction and Division operations into a single kernel.
    *   *Benefit:* Avoids creating intermediate arrays for each step, significantly reducing memory bandwidth and usage.
    *   *Parallelism:* Automatically parallelized over the image height.
    *   *FastMath:* Enabled for additional speedups.
*   **Sliding Window (`sliding_mean_numba`):**
    *   *Benefit:* O(N) complexity independent of window size (vs O(N*W) for naive implementation).
    *   *Mechanism:* Updates the sum incrementally as the window slides, rather than recomputing.
*   **Reduction Kernels:** Custom parallel implementations for Mean, Max, Min, Std.

### 2. Threaded Reduction Operations
For large arrays (>10 MB), reduction operations (Mean, Std, Min, Max, Median) utilize `ThreadPoolExecutor`.
*   *Benefit:* ~3.4x speedup on 4 cores.
*   *Mechanism:* Numpy releases the GIL for these operations, allowing true parallelism. The array is split along the spatial height axis.

### 3. Background Subtraction
*   **Optimization:** Switched to `float32` (halves memory usage vs `float64`) and in-place operations (`-=`, `/=`).
*   **Result:** ~4x speedup and ~3x memory reduction.

### 4. Video Loading Strategy
*   **Single-Core Default:** Multicore loading for video was benchmarked and found to be slower due to I/O contention and codec state overhead.
*   **Seek vs Grab:** The loader intelligently chooses between `grab()` (sequential) and `set(CAP_PROP_POS_FRAMES)` (random access) based on the sampling step size.

## Profiling Insights: Qt Signal Bottleneck

When loading large datasets, profiling reveals that the post-load delay is dominated by Qt signal processing rather than raw data loading.

*   **Observation:** The `setImage()` call triggers a cascade of signals within PyQtGraph (`sigRegionChanged`, histogram updates, timeline updates).
*   **Quantified Impact:** In extreme cases (e.g., 40k images), signal processing can account for ~80% of the perceived "load time" (e.g., 20s of signal processing vs 4s of disk I/O).
*   **Mitigation:**
    *   **Lazy Updates:** Extraction plots (`ExtractionPlot`) check `isVisible()` before redrawing. If hidden (dock collapsed), computation is skipped.
    *   **Blocked Signals:** During batch updates like `reset_options`, signals are explicitly blocked (`blockSignals(True)`) to prevent redundant re-renders.

## Identified Bottlenecks & Recommendations

### 1. Memory Spikes during Loading
**Issue:** The `_load_folder` and `_load_video` methods collect loaded images in a list before stacking.
*   **Impact:** Temporarily requires 2x memory.
*   **Recommendation:** Pre-allocate the final numpy array.

### 2. Large `.npy` File Loading
**Issue:** `np.load(path)` loads the entire array into memory.
*   **Recommendation:** Use `np.load(path, mmap_mode='r')` for files larger than RAM.

## Future Concepts

### Lightweight Autograd
See `docs/MISSING_FEATURES.md` and `docs/AUTOGRAD_POTENTIAL.md` for a concept on using a minimal autograd engine for parameter tuning.

### Video to Float32 on Disk
**Idea:** Decode compressed video once to a raw float32 memory-mapped file on disk.
*   *Pros:* Instant random access, no RAM limit.
*   *Cons:* Large disk usage, initial decode time.
