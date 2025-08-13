# SentinelCam

Real-time video streaming + YOLO object detection + lightweight tracking, with zoom/pan, replay buffer, snapshots, and recording. Built for webcams or RTSP streams.

---

## Quick start
```bash
git clone <your-repo-url> SentinelCam
cd SentinelCam

# (Optional) create venv
# python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Put your weights at ./models/yolo11n.pt (or change model_name in config.py)
python main.py
```
> 💡 Tip: If your camera is RTSP, set `rtsp_stream = "rtsp://user:pass@ip:port/..."` in `config.py`.

---

## Features
- **Real-time stream** from webcam (`0`) or RTSP URL  
- **YOLO detection (async load)**: app shows live video while model loads, then auto-switches to detection  
- **Lightweight tracking**: center-point + stationary window + timeout  
- **Zoom / Pan** with mouse (throttled inference while panning for smooth UX)  
- **Replay buffer**: rewind recent seconds to review events  
- **Snapshots**: save frames to `./snapshots`  
- **Recording (.mp4)**: start/stop and save to `./recordings`  
- **Grayscale (high contrast)** mode for readability  
- **On-screen overlays**: FPS, clock, replay status, REC indicator  

---

## Controls
**Keyboard**
- `q` — quit  
- `space` — clear frame queue (quick skip)  
- `m` — toggle **grayscale** mode  
- `s` — **save snapshot** to `./snapshots`  
- `Backspace` — toggle **replay** mode  
- `r` — start/stop **recording** to `./recordings`  

**Mouse**
- Left drag — draw rectangle (visual aid)  
- Mouse wheel — zoom in/out  
- Right drag — pan (when zoomed)  

---

## Project structure
```
SentinelCam/
├─ main.py                # main loop + key bindings + window rendering
├─ config.py              # all app settings (model, stream, thresholds, IO paths)
├─ utils.py               # helpers + AppState (global runtime state)
├─ camera.py              # frame grabbing thread + replay buffer
├─ detection.py           # Detector class (YOLO async load + frame processing)
├─ tracking.py            # BoundingBox dataclass + iteration helpers
├─ ui_controls.py         # mouse handlers + fullscreen/reset helpers
├─ recorder.py            # start/stop/write recording (.mp4)
├─ db/
│  └─ coco.names          # class labels (COCO)
├─ models/                # place model weights here (e.g., yolo11n.pt)
├─ snapshots/             # saved snapshots
└─ recordings/            # saved videos
```

---

## Configuration
Edit `config.py`:
- `model_name`: weight file in `./models/<model_name>.pt` (e.g., `"yolo11n"`)  
- `rtsp_stream`: `0` for webcam or RTSP URL string  
- `opsize`: output window size (w, h)  
- `min_confidence`, `class_confidence`: global and per-class thresholds  
- `visible_classes`: subset of classes to detect  
- `yolo_skip_frames`: inference every N frames (higher = faster, lower = more accurate)  
- `snapshot_directory`, `recording_directory`: IO paths  
- `point_timeout`, `stationary_val`: tracking sensitivity  
- `zoom_max_factor`, `double_click_threshold_ms`: UI/zoom behavior  
- `labels_path`: path to `coco.names`  

---

## Requirements
- Python 3.9+  
- OpenCV, Ultralytics YOLO, PyTorch, scikit-learn, numpy  

Example:
```bash
pip install opencv-python ultralytics torch torchvision torchaudio scikit-learn numpy
```
> For CUDA acceleration, install a Torch build that matches your GPU/driver.

---

## Troubleshooting
- **Black window / no frames**: check `rtsp_stream` value and camera permissions  
- **Model not found**: ensure `models/<model_name>.pt` exists or update `config.py`  
- **Recording file won’t open**: try a different codec (`fourcc_str='avc1'`) or verify frame size matches the written frames  
- **High CPU usage**: increase `yolo_skip_frames`, limit `visible_classes`, reduce `opsize`  

---

## License
MIT (adjust as needed)  
