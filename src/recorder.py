# recorder.py
# -*- coding: utf-8 -*-
import cv2, time
from utils import ensure_dir
from config import recording_directory

def start_recording(state, frame_size, fps=20, fourcc_str='mp4v'):

    ensure_dir(recording_directory)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"../{recording_directory}/recording_{ts}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try another codec (e.g., 'avc1') or check OpenCV build.")
    state.recording_writer = writer
    state.recording = True
    state.recording_path = path
    state.recording_fps = fps
    state.recording_size = frame_size
    print(f"Started recording: {path}")

def stop_recording(state):
    if getattr(state, 'recording_writer', None) is not None:
        state.recording_writer.release()
        state.recording_writer = None
    if getattr(state, 'recording', False):
        print(f"Stopped recording: {getattr(state, 'recording_path', '')}")
    state.recording = False

def write_frame(state, frame):
    if getattr(state, 'recording_writer', None) is not None and getattr(state, 'recording', False):
        state.recording_writer.write(frame)
