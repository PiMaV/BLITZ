# Features & Tabs Explained

A comprehensive guide to BLITZ's tabs and core features.

## Table of Contents

- [File Tab](#file-tab)
- [View Tab](#view-tab)
- [Ops Tab](#ops-tab)
- [Tools Tab](#tools-tab)
- [RoSEE Tab](#rosee-tab)
- [PCA Tab](#pca-tab)
- [Bench Tab](#bench-tab)
- [Stream Tab](#stream-tab)
- [Timeline Panel](#timeline-panel)

---

## File Tab

Start here to load your data.

### Loading Data
- **Buttons:** `Load File` and `Load Folder` open standard system dialogs.
- **Drag & Drop:** You can drag files or folders directly into the viewer area.

### Import Settings
These settings control how data is processed *during* loading.
- **Show load options dialog:** If checked, a dialog appears for every load operation (video/folder), allowing you to tweak parameters. If unchecked, the last used settings are applied automatically.
- **8 bit:** Converts high-bit-depth images (12/16-bit) to 8-bit to save RAM.
- **Grayscale:** Converts color images to grayscale (luminance) to save RAM and speed up processing.
- **Size ratio:** Downscales images (e.g., 0.5 = 50% width/height). Useful for massive datasets.
- **Subset ratio:** Loads only a fraction of frames (e.g., 0.1 = every 10th frame).
- **Max. RAM:** Limits the memory buffer for loading.

### Hidden Features
- **Crop:** A destructive crop widget exists in the code but is currently hidden to prevent accidental data loss. Use the *Subset* or *Size ratio* for downsampling instead.

---

## View Tab

Controls for visualization and non-destructive image manipulation.

### View
- **Flip x / Flip y:** Mirrors the image.
- **Rotate 90°:** Rotates the image 90 degrees clockwise.

### Display Mask
Define regions to exclude from analysis and display.
- **Show:** Toggles the mask overlay.
- **Apply:** Activates the mask (pixels outside the mask are ignored in calculations).
- **Load binary image:** Loads a black/white image to use as a mask.
- **Reset:** Clears the mask.

### Crosshair
- **Show:** Displays a crosshair cursor.
- **Show Markings:** Indicates the crosshair position on the H/V extraction plots.
- **Line width (H/V):** Sets the thickness of the lines used for the H/V extraction plots (averaging over N pixels).

### Extraction Plots
These settings control the side panels (H Plot / V Plot).
- **Min/Max per image:** Shows the global min/max values of the current frame in the plots.
- **Envelope per crosshair:** Shows the min/max range of the crosshair line over time.
- **Envelope per position (dataset):** Shows the min/max range of the entire dataset at the crosshair position.

### Timeline Plot
Controls the bottom chart.
- **ROI:** Toggles the Region of Interest functionality.
- **Type:** Choose between Rectangular or Polygon ROI.
- **Update on drop:** If checked, the plot updates only when you release the mouse (better performance).

---

## Ops Tab

The core image processing pipeline. Operations are applied in real-time.

### Subtract & Divide
Apply arithmetic operations to the image stream.
- **Source:**
  - *Off:* No operation.
  - *Range (Aggregate):* Uses the aggregated result from the Timeline Panel (e.g., Mean of frames 0-100).
  - *File:* Uses an external loaded reference image (e.g., Dark Frame, Flat Field).
  - *Sliding range:* Uses a moving window average (see below).
- **Amount:** Slider (0-100%) to blend the operation intensity.

### Sliding Window
Advanced temporal filtering (e.g., for motion detection or background removal).
- **Range method:** Algorithm for the window (Mean, Max, Min, Median, Std).
- **Window:** Number of frames in the moving buffer.
- **Lag:** Delay between the current frame and the window.
- **Apply to full:** If checked, applies the sliding window to the entire dataset (reducing frame count).

### Timeline Crop
- **Apply Crop:** Destructively crops the dataset in time (RAM is freed).
- **Undo Crop:** Restores the full dataset (only works if data wasn't purged).

---

## Tools Tab

Measurement and analysis tools.

### Measure Tool
- **Show:** Activates a draggable ROI for measurement.
- **Display in au:** Converts pixel units to arbitrary units (e.g., mm).
- **Pixels / in au:** Calibration factor (pixels per unit).
- **Stats:** Displays Area, Circularity, and Bounding Box dimensions of the measured region.

---

## RoSEE Tab

**RoSEE** (Robust and Simple Event Extraction) is a specialized algorithm for detecting and analyzing events in the image data.

- **Show RoSEE:** Activates the overlay.
- **Use local extrema:** Refines detection to local peaks.
- **Smoothing:** Applies spatial smoothing before detection.
- **Plots (H/V):** Toggles auxiliary plots for RoSEE analysis.
- **Normalize:** Normalizes values for better visualization.
- **Isocurves:** Draws contour lines at specific intensity levels.

---

## PCA Tab

Principal Component Analysis (SVD) for dimensionality reduction and pattern extraction.

### Calculate
- **Target Comp:** Number of components to calculate.
- **Exact (Slow):** Uses full SVD (accurate). Uncheck for Randomized SVD (fast approximation).
- **Calculate PCA:** Starts the computation (runs in background).

### View
- **Reconstruction:** Shows the image reconstructed from selected components.
- **Components:** Shows the raw Eigenimages.
- **Include mean:** Adds the average image to the reconstruction.

### Results
- **Variance Plot:** Scree plot showing the variance explained by each component.
- **Table:** Detailed variance statistics.

---

## Bench Tab

Performance monitoring and optimization info.

- **Show CPU load:** Displays a sparkline graph of CPU usage.
- **Status Labels:** Shows state of raw matrix, result matrix, cache, and Numba acceleration status.
- **Sparklines:** Real-time graphs of memory and processing load.

---

## Stream Tab

Live data sources.

### Simulated Live
Generates synthetic data for testing without hardware.
- **Generate:** Starts a Lissajous or Lightning simulation.
- **Uses:** Good for testing the pipeline performance and ring-buffer logic.

### Webcam
Connects to a USB camera (via OpenCV).
- **Settings:** Exposure, Gain, Brightness, Contrast.
- **Buffer:** Configurable ring-buffer size (frames or seconds).

### Network
Connects to a remote data server (e.g., WOLKE).
- **Address / Token:** Connection credentials.
- **Connect:** Establishes a WebSocket connection for remote control/data loading.

---

## Timeline Panel

Located at the bottom, this panel controls navigation and aggregation.

### Frame Tab
Single-frame navigation mode.
- **Idx:** Current frame index.
- **Curve:** Aggregation method for the ROI plot (Mean/Median).
- **Upper/lower band:** Shows min/max envelope in the plot.

### Aggregate Tab
Multi-frame aggregation mode (reduces time dimension).
- **Reduce:** Method to collapse frames (Mean, Max, Min, Std).
- **Start / End:** Range of frames to aggregate.
- **Win const.:** Keeps the window size fixed when moving start/end.
- **Full Range:** Resets range to the entire dataset.
- **Update on drag:** Updates the aggregation live while dragging the range slider (resource intensive).
