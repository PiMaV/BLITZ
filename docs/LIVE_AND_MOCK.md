# Live sources and connection to BLITZ

## Who connects to whom (Handshake)

There is **no separate "Connect" button**. The link is set when you open the live source and start it:

1. **BLITZ (main window)** has the **viewer** – the central image/timeline widget.
2. You open **Cam Mock** (or **Echte Kamera**) from the main window. The main window creates the dialog and does:
   - `mock.set_frame_callback(self.ui.image_viewer.set_image)`
   So the viewer’s `set_image` becomes the callback of the mock.
3. When you press **Play** in the mock:
   - The **producer** (worker thread) starts and writes frames into a **ring buffer** (in the handler).
   - A **QTimer** in the mock (e.g. every 35 ms) runs: pull snapshot from the handler → call the callback with it → viewer shows it.
4. So: **"Connected" = mock is open and playing.** While it plays, the mock pushes snapshots to the viewer and the viewer stays on the **last frame** (ring end). When you **Stop**, the mock sends one final snapshot (full ring, high cap) and the viewer stays on the last frame; the frozen timeline is then **index 0 = ring start, last index = ring end**. You can scroll back to see older frames. Later you can start again or open a file.

Summary:

- **Producer:** Mock (or real camera) worker → fills ring buffer at target FPS.
- **Consumer:** BLITZ viewer. It does **not** pull by itself; the **mock’s timer** pulls from the handler and then calls `viewer.set_image(snapshot)`.
- So the mock is both: it **owns** the ring buffer and the worker, and it **drives** the updates to BLITZ by pulling on a timer and passing the result to the viewer.

## Ring buffer and display cap

- The **ring buffer** can be large (e.g. 84 frames at 1920×1080 for 120 FPS). It is used for a rolling timeline and (in future) e.g. “save last N seconds”.
- For **display**, we do **not** send the whole buffer to the viewer at high res/high FPS: that would copy hundreds of MB every 35 ms and freeze the UI. So the handler’s `get_snapshot()` returns at most a **display cap** (e.g. last N frames that fit in a max display size in MB). The ring buffer still keeps all frames internally; only the portion passed to the viewer is limited.

## Double buffer (optional)

A **double buffer** (swap two buffers so the producer writes to one while the observer reads from the other) can reduce lock contention: the observer only needs a quick swap, then can build the snapshot without holding the lock. The main performance gain for high FPS/high res is still the **display cap** (fewer frames sent to the UI).
