# Live sources and connection to BLITZ

This document describes how **Cam Mock** and **Webcam** (real camera) connect to the BLITZ viewer, and the **ring-buffer semantics** shared by both.

---

## 1. Ring-buffer semantics (same for Mock and Real Camera)

- **Index 0** = oldest frame in the current timeline = **ring start**.
- **Last index (n_images - 1)** = newest frame = **ring end** (“last point”).

**During live stream:** The viewer is always moved to the **last frame** when new data arrives, so you see the live head (ring end).

**On Stop:** Both sources send **one final snapshot** of the ring at stop time. The viewer stays on the last frame. The frozen timeline is then:
- **Index 0** = ring start (oldest frame at stop),
- **Last index** = ring end (newest frame at stop).

You can scroll back in the timeline to inspect older frames; the event “Stop” is represented by this frozen ring.

---

## 2. Who connects to whom (handshake)

There is **no separate “Connect” button**. The link is established when you open the live source and start it:

1. **BLITZ (main window)** has the **viewer** (central image/timeline widget).
2. You open **Cam Mock** or **Webcam** from the main window. The main window creates the dialog and registers a callback so that incoming frames are passed to the viewer (and the current index is set to the last frame).
3. When you press **Play** (Mock) or **Start** (Real Camera), the stream starts and the viewer is updated with ring snapshots; the viewer stays on the **last frame**.
4. When you press **Stop**, the source sends **one final snapshot** (the ring at stop time), the viewer is set to the last frame, then the stream is torn down. The frozen image in BLITZ is exactly the ring at stop.

So: **“Connected”** = dialog is open and stream is running. After **Stop**, the last state is the ring at stop; you can scroll in the timeline (0 = start, last = end).

---

## 3. Cam Mock (simulated camera)

- **Purpose:** Simulated live source (Lissajous, later e.g. Blitz). No real device.
- **Architecture:** **Pull model.** Worker fills a ring buffer in the handler; a **QTimer** in the mock (e.g. every 35 ms) calls `handler.get_snapshot()` and passes the result to the viewer callback.
- **Parameters:** FPS (1–120), exposure, resolution, buffer (MB), variant, grayscale. See `blitz/data/live.py`, `blitz/layout/winamp_mock.py`.
- **Display cap:** To avoid UI freeze at high FPS/resolution, `get_snapshot(max_display_mb=50)` returns at most the last N frames that fit in 50 MB. The ring buffer itself keeps all frames; only the portion sent to the viewer is capped. **On Stop**, the mock requests a final snapshot with a high cap (`max_display_mb=999`) so the frozen state is (up to) the full ring.
- **Callback (main):** `set_image(img)` + `setCurrentIndex(img.n_images - 1)` so the viewer always shows the last frame.

---

## 4. Webcam (real camera)

- **Purpose:** Live stream from USB webcam via `cv2.VideoCapture`. Buffer configurable; FPS measured from actual capture.
- **Architecture:** **Push model.** Worker thread reads frames, keeps a ring buffer with per-frame timestamps. **send_live_only=True:** emit only the newest frame each tick; ring still kept for final snapshot on Stop. Main window throttles display updates (fixed 10 FPS) and sets current index to last frame.
- **UI:** Start/Stop **toggle**, **Close** button. **Device:** 0 or 1. **Resolution:** 640x480–1920x1080. **FPS:** Display shows measured capture rate (e.g. `~12.2 fps (capture)`); init `10 fps (fixed)` until first measurement. Capture FPS is used to assign buffer length to the correct time span. **Buffer:** Frames or Seconds mode; time span and RAM estimated. During stream, measured time span and capture FPS shown live.
- **Display throttle:** Main window applies camera frames at fixed 10 FPS (`_CAMERA_APPLY_MS = 100`). Configurable display FPS planned later.
- **Capture rate:** The worker does **not** sleep between reads; reads as fast as camera delivers. Per-frame timestamps (`time.perf_counter()`); actual time span (first to last frame) emitted via `buffer_time_span_sec`. Capture FPS = frames / measured_sec. Typical: ~10-12 FPS. This value is kept after Stop for correct buffer-time mapping.
- **Parameters:** device_id (0/1), width, height, fps (last measured value reused when available), buffer_size (1–10000), grayscale, exposure, gain, brightness, contrast, auto_exposure, send_live_only. See `blitz/data/live_camera.py`, `blitz/layout/dialogs.py` (`RealCameraDialog`).
- **On Stop:** Worker emits **one final** `frame_ready` with the full ring. Main window `_camera_stop_cleanup` applies any pending frame (final buffer) before disconnecting. Dialog disconnects only **after** `wait_stopped()`.
- **Code:** `_CameraWorker` stores `(frame, timestamp)` per buffer entry; emits `buffer_time_span_sec` when len>=2. `send_live_only` -> emit `[buffer[-1]]` each tick; on exit emit full buffer. `RealCameraDialog`: tracks `_last_capture_fps`; uses it for buffer time calculations and FPS label.

---

## 5. Summary table

| Aspect              | Cam Mock                    | Webcam                 |
|---------------------|-----------------------------|------------------------------|
| Model               | Pull (timer → get_snapshot) | Push (frame_ready)           |
| Ring in             | Handler                     | Worker                       |
| Live vs timeline    | —                           | Combo: Live (last frame) or Timeline (full ring) |
| Display cap         | Yes (e.g. 50 MB)           | Live = 1 frame; Timeline = full buffer |
| Final snapshot      | get_snapshot(999 MB) on stop| Worker emits final buffer on exit |
| Viewer on last frame| Callback sets index         | Callback sets index          |

---

## 6. Optional: double buffer

A **double buffer** (swap two buffers so the producer writes to one while the observer reads from the other) can reduce lock contention. For the mock, the main performance gain at high FPS/resolution comes from the **display cap**; double buffering can be added later if needed.
