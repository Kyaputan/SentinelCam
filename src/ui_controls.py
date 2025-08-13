
# -*- coding: utf-8 -*-
import cv2
from utils import millis

def toggle_fullscreen(state):
    if state.fullscreen:
        cv2.setWindowProperty(state.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(state.window_name, state.original_window_size[0], state.original_window_size[1])
        state.fullscreen = False
        print("Exited fullscreen")
    else:
        cv2.setWindowProperty(state.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        state.fullscreen = True
        print("Entered fullscreen")

def reset_window_to_stream_resolution(state):
    cv2.resizeWindow(state.window_name, state.opsize[0], state.opsize[1])
    state.original_window_size = state.opsize
    print(f"Reset window to {state.opsize[0]}x{state.opsize[1]}")

def mouse_callback(state, cfg, detector):
    def _cb(event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            state.dragging = True
            state.zoom_pan_active = True
            state.zoom_pan_pause_time_ms = millis()
            detector.cached_yolo_results = None
            state.drag_start_x, state.drag_start_y = x, y

        if event == cv2.EVENT_RBUTTONUP:
            state.dragging = False
            state.zoom_pan_active = False
            detector.cached_yolo_results = None

        if event == cv2.EVENT_LBUTTONUP:
            state.drawing = False
            state.draw_start_x = state.draw_start_y = state.draw_end_x = state.draw_end_y = 0

        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            state.draw_start_x, state.draw_start_y = x, y
            state.draw_end_x = state.draw_end_y = 0

        if event == cv2.EVENT_MOUSEWHEEL:
            state.zoom_pan_active = True
            state.zoom_pan_pause_time_ms = millis()
            detector.cached_yolo_results = None
            old_zoom = state.zoom_factor
            if flags > 0:
                state.zoom_factor = min(cfg['zoom_max_factor'], state.zoom_factor * 1.1)
            else:
                state.zoom_factor = max(1.0, state.zoom_factor / 1.1)
            if (old_zoom == 1.0) and (state.zoom_factor > 1.0) and (not state.zoom_mode_active):
                detector.stored_bounding_boxes = detector.bounding_boxes.copy()
                state.zoom_mode_active = True
                print(f"Entering zoom mode - stored {len(detector.stored_bounding_boxes)} objects")
            elif (state.zoom_factor == 1.0) and state.zoom_mode_active:
                state.zoom_mode_active = False
                detector.bounding_boxes = detector.stored_bounding_boxes.copy()
                detector.stored_bounding_boxes = []
                print(f"Exiting zoom mode - restored {len(detector.bounding_boxes)} objects")

        if event == cv2.EVENT_MOUSEMOVE:
            if state.drawing:
                state.draw_end_x, state.draw_end_y = x, y
            if state.dragging:
                dx, dy = x - state.drag_start_x, y - state.drag_start_y
                state.pan_x -= int(dx * state.zoom_factor)
                state.pan_y -= int(dy * state.zoom_factor)
                state.drag_start_x, state.drag_start_y = x, y
    return _cb
