"""
Standalone script: DICOM -> .npy converter for BLITZ.
Knowledge store only - not imported by BLITZ. Run with: python scripts/dicom_converter.py ...

Dependencies (install manually): pip install pydicom numpy
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
try:
    import pydicom
except ImportError:
    print("Error: 'pydicom' is required for this script. Please install it via 'pip install pydicom'.")
    sys.exit(1)

def is_dicom(path: Path) -> bool:
    try:
        pydicom.dcmread(path, stop_before_pixels=True)
        return True
    except Exception:
        return False

def load_dicom_series(input_dir: Path) -> np.ndarray:
    """
    Loads a series of DICOM files from a directory, sorts them by InstanceNumber,
    and returns a 3D numpy array (Time, Height, Width).
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"{input_path} is not a directory.")

    files = [f for f in input_path.iterdir() if f.is_file()]
    dicom_files = []

    print(f"Scanning {len(files)} files in {input_path}...")

    for f in files:
        # Simple check first
        if f.suffix.lower() in ('.dcm', '.dicom') or is_dicom(f):
            try:
                ds = pydicom.dcmread(f)
                if hasattr(ds, "pixel_array") and hasattr(ds, "InstanceNumber"):
                    # Extract pixel array and ensure it's a numpy array
                    arr = ds.pixel_array
                    instance_num = int(ds.InstanceNumber)
                    dicom_files.append((instance_num, arr))
            except Exception as e:
                print(f"Skipping {f.name}: {e}")

    if not dicom_files:
        raise ValueError("No valid DICOM files found in the directory.")

    # Sort by InstanceNumber to ensure correct time/z-axis order
    dicom_files.sort(key=lambda x: x[0])

    print(f"Found {len(dicom_files)} valid DICOM files. Stacking...")

    # Stack images into a (Time, Height, Width) array
    images = [img for _, img in dicom_files]
    stack = np.stack(images)

    return stack

def main():
    parser = argparse.ArgumentParser(
        description="Convert DICOM series to .npy for BLITZ. "
                    "Requires 'pydicom' to be installed."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing DICOM files")
    parser.add_argument("output_file", type=str, help="Output .npy file path")

    args = parser.parse_args()

    try:
        stack = load_dicom_series(args.input_dir)
        print(f"Resulting Shape: {stack.shape}, Dtype: {stack.dtype}")

        output_path = args.output_file
        if not output_path.endswith('.npy'):
            output_path += '.npy'

        np.save(output_path, stack)
        print(f"Successfully saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
