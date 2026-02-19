import time
import numpy as np
import psutil
import os
import sys

# Ensure blitz is importable
sys.path.append(os.getcwd())

try:
    from blitz.data.image import ImageData, MetaData
except ImportError:
    print("Could not import blitz. Make sure you are in the repo root.")
    sys.exit(1)

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def benchmark():
    # Use 300x1000x1000 to be safe on smaller RAM
    # A 1000x1000x1000 dataset would effectively triple the time and memory.
    T, H, W = 300, 1000, 1000
    print(f"--- Benchmark: Background Subtraction ---")
    print(f"Dataset Shape: ({T}, {H}, {W}) (uint8)")

    print(f"Allocating input...")
    image_uint8 = np.random.randint(0, 255, size=(T, H, W, 1), dtype=np.uint8)
    meta = [MetaData(file_name="test", file_size_MB=0, size=(H, W), dtype=np.uint8, bit_depth=8, color_model="grayscale") for _ in range(T)]
    data = ImageData(image_uint8, meta)

    ref_img = np.random.randint(0, 255, size=(1, H, W, 1), dtype=np.uint8)
    ref_meta = [MetaData(file_name="ref", file_size_MB=0, size=(H, W), dtype=np.uint8, bit_depth=8, color_model="grayscale")]
    ref_data = ImageData(ref_img, ref_meta)

    config = {
        "subtract": {
            "source": "file",
            "reference": ref_data,
            "amount": 1.0
        }
    }
    data.set_ops_pipeline(config)

    print_memory_usage()
    print("Running subtraction (ImageData.image property)...")

    start_time = time.time()
    result = data.image
    end_time = time.time()

    duration = end_time - start_time
    print(f"Time: {duration:.4f} seconds")
    print_memory_usage()
    print(f"Output dtype: {result.dtype}")

    if result.dtype == np.float32:
        print("SUCCESS: Optimized float32 path used.")
    else:
        print(f"WARNING: Output is {result.dtype}, optimization might not be active.")

if __name__ == "__main__":
    benchmark()
