# Numba Optimization Candidates

This document lists functions and code blocks identified as candidates for optimization using Numba. Numba allows JIT compilation of Python functions to machine code, often achieving C-like performance, especially for loop-heavy code or complex array operations that are inefficient in pure Python or require excessive temporary memory allocations in Numpy.

## 1. `ImageData._apply_ops_pipeline` (Fusion of Operations)

*   **File:** `blitz/data/image.py`
*   **Current Implementation:** Iterates over pipeline steps (subtract, divide). For each step, it creates intermediate arrays (e.g., `ref_scaled`, `denom`), converts the entire dataset to `float64`, and performs element-wise arithmetic.
*   **Bottleneck:**
    *   **Memory Bandwidth & Allocation:** Creates multiple full-size copies of the dataset. For "massive 4D matrices", this leads to high memory usage and cache thrashing.
    *   **Redundant Passes:** Iterates over the memory multiple times (once per operation).
*   **Numba Strategy:** Implement a JIT-compiled kernel that applies the entire pipeline (subtraction, division) in a single pass per pixel (kernel fusion). This avoids intermediate arrays and keeps the data in cache.
    *   *Input:* Image array, pipeline parameters (references, amounts).
    *   *Output:* Processed array (or in-place modification).

## 2. `ExtractionPlot._compute_dataset_envelope`

*   **File:** `blitz/layout/widgets.py`
*   **Current Implementation:** Iterates over `range(n_frames)` in Python. Inside the loop, it extracts a slice from each frame, appends it to a list, stacks them into a new Numpy array `(T, slice_width, ...)`, and then calls `np.min/max`.
*   **Bottleneck:**
    *   **Python Overhead:** Loop runs `T` times.
    *   **Memory Allocation:** `np.stack` creates a new array containing the sliced data for the entire timeline. While smaller than the full dataset, it's still an allocation that can be avoided.
    *   **Double Pass:** Stacking writes to memory; `min/max` reads it back.
*   **Numba Strategy:** A JIT function that iterates over frames and computes `min` and `max` accumulators on the fly for the specified slice coordinates.
    *   *Input:* 4D Image array, slice parameters (x/y, width, vertical/horizontal).
    *   *Output:* `min_curve`, `max_curve` (1D arrays of length T).

## 3. `sliding_mean_normalization` (Custom Kernels)

*   **File:** `blitz/data/tools.py`
*   **Current Implementation:** Optimized using `np.cumsum` and chunked processing. This is fast for simple uniform sliding windows.
*   **Future Potential:** If non-uniform kernels (Gaussian, weighted moving average) or more complex temporal filters (e.g., median filter, which `np.cumsum` cannot do) are needed, `np.cumsum` doesn't work.
*   **Numba Strategy:** A general-purpose `sliding_window_filter` JIT function. It can implement any per-window logic (mean, median, custom weights) without the O(N*Window) overhead of Python loops, and without the O(N) memory overhead of full `cumsum` arrays (it can process in-place or with small buffers).

## 4. `ImageData` Masking and Geometry

*   **File:** `blitz/data/image.py`
*   **Current Implementation:** Uses boolean indexing `image[mask]`. This flattens the result or creates a copy.
*   **Optimization:** If complex geometric operations (e.g., arbitrary polygon ROI extraction that preserves shape) are needed, Numba can iterate over the bounding box and apply the point-in-polygon test efficiently without generating a full boolean mask array.

## Implementation Plan

1.  **Add Dependency:** Add `numba` to `pyproject.toml` (if not present).
2.  **Create Module:** Create `blitz/data/optimized.py` to house JIT functions.
3.  **Refactor:** Replace the Python implementations with calls to the JIT functions.
4.  **Fallback:** Ensure pure-Python fallbacks exist if Numba compilation fails or is disabled (though Numba works on CPU).
