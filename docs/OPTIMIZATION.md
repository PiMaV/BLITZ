# Optimization Report

## Current Performance Analysis

BLITZ is designed for high-performance in-memory analysis. The current implementation uses `multiprocessing` to parallelize image loading and processing, which is effective for CPU-bound tasks like JPEG decoding.

### Key Strengths
*   **Parallel Loading**: Uses `multiprocessing.Pool` to load and resize images concurrently.
*   **Memory Guard**: Automatically downsamples (subsets) data if the estimated size exceeds the user-defined RAM limit (`adjust_ratio_for_memory`).
*   **Fast Operations**: Uses `numpy` for vectorized operations (normalization, reduction).

## Identified Bottlenecks & Recommendations

### 1. Memory Spikes during Loading
**Issue:** The `_load_folder` and `_load_video` methods collect loaded images in a Python list (`matrices` or `frames`) and then call `np.stack()`.
*   **Impact:** This temporarily requires 2x the memory of the final dataset (1x for the list of arrays, 1x for the contiguous stacked array), potentially causing Out-Of-Memory (OOM) errors on machines with limited RAM relative to the dataset size.
*   **Recommendation:** Pre-allocate the final numpy array using `np.empty` or `np.zeros` once the shape of the first image is known and the total count is determined. Fill this array directly or via shared memory in multiprocessing.

### 2. Large `.npy` File Loading
**Issue:** `np.load(path)` loads the entire array into memory.
*   **Impact:** If a source `.npy` file is larger than available RAM, the application will crash immediately.
*   **Recommendation:** Use `np.load(path, mmap_mode='r')`. This maps the file to memory without loading it all at once. Slicing and processing can then happen on chunks of the data, allowing for processing of datasets larger than RAM (if coupled with chunked processing) or at least avoiding the initial load spike.

### 3. Video Loading Efficiency
**Issue:** `_load_video` uses `video.grab()` to skip frames.
*   **Impact:** While faster than `read()`, `grab()` still incurs some overhead.
*   **Recommendation:** For significant subsampling, seeking (`cap.set(cv2.CAP_PROP_POS_FRAMES, ...)`) might be faster, though it depends on the video codec's keyframe structure.

### 4. Pure Python/Numpy vs. Compiled Extensions
**Issue:** Some complex per-pixel operations might be slow in pure Numpy if they involve many temporary array allocations.
*   **Recommendation:** Since Numba usage was removed, consider bringing it back for specific "hot paths" (like custom normalization filters) if profiling indicates a CPU bottleneck.

## Action Plan
1.  **Immediate:** Implement pre-allocation in `DataLoader` to avoid memory spikes.
2.  **Short-term:** Investigate `mmap` for `.npy` files.
3.  **Long-term:** Re-evaluate Numba or Cython for specific processing filters.
