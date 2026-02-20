# Live sources and connection to BLITZ

This document describes how **Cam Mock** and **Echte Kamera** (real camera) connect to the BLITZ viewer, and the **ring-buffer semantics** shared by both.

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
2. You open **Cam Mock** or **Echte Kamera** from the main window. The main window creates the dialog and registers a callback so that incoming frames are passed to the viewer (and the current index is set to the last frame).
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

## 4. Echte Kamera (real camera)

- **Purpose:** Live stream from USB webcam via `cv2.VideoCapture`. Exposure, gain, FPS, buffer configurable.
- **Architecture:** **Push model.** Worker thread reads frames, keeps a ring buffer, and on each frame emits `frame_ready(ImageData)` with the current buffer (full ring). Main window throttles application to the viewer (e.g. 20 Hz) and sets current index to last frame.
- **Parameters:** Device ID, exposure, gain, brightness, contrast, auto exposure, FPS, buffer (frames), grayscale. See `blitz/data/live_camera.py`, `blitz/layout/dialogs.py` (`RealCameraDialog`).
- **On Stop:** The worker, when exiting the loop, emits **one final** `frame_ready` with the current ring buffer so the viewer gets the ring at stop. The dialog disconnects `frame_ready` only **after** `wait_stopped()`, so this final frame is delivered. Callback (main) already does `set_image(img)` + `setCurrentIndex(img.n_images - 1)`.
- **Code:** `_CameraWorker.run()` emits a final `frame_ready` with `self._buffer` before releasing the capture and emitting `stopped`. `RealCameraDialog._stop()` calls `handler.stop()`, then `wait_stopped()`, then disconnects `frame_ready` and clears the handler.

---

## 5. Summary table

| Aspect              | Cam Mock                    | Echte Kamera                 |
|---------------------|-----------------------------|------------------------------|
| Model               | Pull (timer → get_snapshot) | Push (frame_ready)           |
| Ring in             | Handler                     | Worker                       |
| Display cap         | Yes (e.g. 50 MB)            | No (full buffer each emit)   |
| Final snapshot      | get_snapshot(999 MB) on stop| Worker emits final buffer on exit |
| Viewer on last frame| Callback sets index         | Callback sets index          |

---

## 6. Optional: double buffer

A **double buffer** (swap two buffers so the producer writes to one while the observer reads from the other) can reduce lock contention. For the mock, the main performance gain at high FPS/resolution comes from the **display cap**; double buffering can be added later if needed.
