
# -*- coding: utf-8 -*-
import os, time, cv2, math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List

_font = cv2.FONT_HERSHEY_SIMPLEX

def millis() -> int:
    return round(time.perf_counter() * 1000)

def timestamp() -> int:
    return int(time.time())

def format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        m = seconds // 60
        s = seconds % 60
        return f"{m}m" if s == 0 else f"{m}m{s}s"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if m == 0 and s == 0: return f"{h}h"
        if s == 0: return f"{h}h{m}m"
        return f"{h}h{m}m{s}s"

def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=8):
    def draw_dashed_line(img, pt1, pt2, color, thickness, dash_length):
        dist = math.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1])
        dashes = max(1, int(dist / dash_length))
        for i in range(dashes):
            start = (
                int(pt1[0] + (pt2[0]-pt1[0]) * i / dashes),
                int(pt1[1] + (pt2[1]-pt1[1]) * i / dashes),
            )
            end = (
                int(pt1[0] + (pt2[0]-pt1[0]) * (i+0.5) / dashes),
                int(pt1[1] + (pt2[1]-pt1[1]) * (i+0.5) / dashes),
            )
            cv2.line(img, start, end, color, thickness)
    draw_dashed_line(img, pt1, (pt2[0], pt1[1]), color, thickness, dash_length)
    draw_dashed_line(img, (pt2[0], pt1[1]), pt2, color, thickness, dash_length)
    draw_dashed_line(img, pt2, (pt1[0], pt2[1]), color, thickness, dash_length)
    draw_dashed_line(img, (pt1[0], pt2[1]), pt1, color, thickness, dash_length)
    return img

def generate_color_shades(num_classes: int) -> np.ndarray:
    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    base_colors = [np.array([0,200,0]), np.array([0,165,255]), np.array([0,200,255]), np.array([0,0,255])]
    for i in range(num_classes):
        base = base_colors[i % len(base_colors)]
        shade_factor = (i // len(base_colors)) / (num_classes // len(base_colors) + 1)
        shade = base*(1-shade_factor) + np.array([128,128,128])*shade_factor
        colors[i] = shade.astype(np.uint8)
    return colors

def load_labels(path: str) -> Tuple[list, dict]:
    labels = open(path, "r", encoding="utf-8").read().strip().split("\n")
    idx_by_name = {name: i for i, name in enumerate(labels)}
    return labels, idx_by_name

def transform_to_stream(xmin, ymin, xmax, ymax, pad, streamsize, opsize):
    x_scale = streamsize[0]/opsize[0]
    y_scale = streamsize[1]/opsize[1]
    return (
        int(xmin*x_scale) - pad,
        int(ymin*y_scale) - pad,
        int(xmax*x_scale) + pad,
        int(ymax*y_scale) + pad,
    )

def ensure_dir_cwd(path: str):
    os.makedirs(path, exist_ok=True)

def ensure_dir(path: str):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.join(project_root, path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

@dataclass
class AppState:
    # window
    window_name: str = "0"
    window_aspect_ratio: float = 4/3
    original_window_size: Tuple[int,int] = (640,480)
    fullscreen: bool = False
    topmost: bool = True

    # stream sizes
    streamsize: Tuple[int,int] = (0,0)
    opsize: Tuple[int,int] = (640,480)
    hdstream: bool = False

    # queues / fps
    frames: int = 0
    prev_frames: int = 0
    last_frame_ms: int = 0
    fps: int = 0

    # flags
    loop: bool = True
    recording: bool = False
    military_mode: bool = False
    basic_stream_mode: bool = True
    replay_mode: bool = False

    # zoom / pan
    zoom_factor: float = 1.0
    pan_x: int = 0
    pan_y: int = 0
    zoom_mode_active: bool = False
    zoom_pan_active: bool = False
    zoom_pan_pause_time_ms: int = 0

    # draw rect
    drawing: bool = False
    dragging: bool = False
    draw_start_x: int = 0
    draw_start_y: int = 0
    draw_end_x: int = 0
    draw_end_y: int = 0
    drag_start_x: int = 0
    drag_start_y: int = 0

    # tracking stats
    object_count_hist: list = field(default_factory=list)
    old_count: int = 0
    obj_break_ms: int = 0
    obj_idle_ms: int = 0

    # replay
    replay_buffer: list = field(default_factory=list)
    replay_index: int = 0
    replay_last_flash_time_ms: int = 0

    # cache
    cached_yolo_results: any = None
    yolo_frame_count: int = 0

    recording: bool = False
    recording_writer: any = None
    recording_fps: int = 20
    recording_size: tuple = (640, 480)
    recording_path: str = ""