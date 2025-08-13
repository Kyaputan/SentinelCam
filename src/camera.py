
# -*- coding: utf-8 -*-
import cv2, time, queue
from typing import Optional
from utils import timestamp
from config import buffer
from config import replay_buffer_max_size

def make_capture(rtsp_stream):
    cap = cv2.VideoCapture(rtsp_stream)
    if rtsp_stream == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def start_stream_loop(state, cap, q: queue.Queue):
    """Grab frames and push to queue. Handles reconnects."""
    if not cap.isOpened():
        return
    ret, _ = cap.read()
    last_fskip = timestamp()
    while state.loop:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame, restarting video...")
            cap.release()
            cap = cv2.VideoCapture(state.window_name)
            continue
        # replay buffer
        if not state.replay_mode:
            add_to_replay_buffer(state, frame)
        # simple idle-based skip (mirrors original behavior)
        if (state.obj_idle_ms > 0) and (state.obj_idle_ms >= 3000) and (timestamp() - last_fskip >= 30):
            last_fskip = timestamp()
            with q.mutex:
                q.queue.clear()
            state.obj_idle_ms = 0
            print("Frame skip")
        else:
            try:
                q.put(frame, timeout=0.01)
            except queue.Full:
                pass

def add_to_replay_buffer(state, frame):
    if len(state.replay_buffer) >= replay_buffer_max_size:
        state.replay_buffer.pop(0)
    state.replay_buffer.append(frame.copy())
