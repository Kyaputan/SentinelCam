# Model & stream
model_name: str = "yolo11n"       # weights file in ./models/<model_name>.pt
rtsp_stream = 0                   # 0 for default webcam, or RTSP/HTTP URL string

# Display
opsize = (640, 480)               # output window size (w, h)
topmost_window = True

# Queues & performance
buffer = 512
min_confidence = 0.15
min_size = 20
yolo_skip_frames = 3              # inference every N frames (reduced when "military mode" on)

# Object / class thresholds (overrides min_confidence by class)
class_confidence = {
    "truck": 0.35,
    "car": 0.15,
    "boat": 0.85,
    "bus": 0.5,
    "aeroplane": 0.85,
    "frisbee": 0.85,
    "pottedplant": 0.55,
    "train": 0.85,
    "chair": 0.5,
    "parking meter": 0.9,
    "fire hydrant": 0.65,
    "traffic light": 0.65,
    "backpack": 0.65,
    "bicycle": 0.55,
    "bench": 0.75,
    "zebra": 0.90,
    "tvmonitor": 0.80,
}

# Visible COCO classes to detect (subset)
visible_classes = [
    "person",
    "car",
    "motorbike",
    "bicycle",
    "truck",
    "traffic light",
    "stop sign",
    "bench",
    "bird",
    "cat",
    "dog",
    "backpack",
    "suitcase",
    "handbag",
]

# IO
snapshot_directory = "snapshots"
recording_directory = "recordings"

# Behaviour
point_timeout = 2500
stationary_val = 16
idle_reset = 3000
padding = 6
zoom_max_factor = 6.0
double_click_threshold_ms = 300
replay_buffer_max_size = 300      # ~10 seconds @30fps

# Files
labels_path = "./db/target.txt"     # will be loaded by utils.load_labels()

# Window name (derived from stream in main if left None)
window_name = None
