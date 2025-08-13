
# -*- coding: utf-8 -*-
import cv2, queue, time, torch
from config import (
    model_name, rtsp_stream, opsize, buffer, yolo_skip_frames, snapshot_directory,
    recording_directory, point_timeout, stationary_val, padding, double_click_threshold_ms,
    replay_buffer_max_size, labels_path, window_name, class_confidence
)
from utils import AppState, load_labels, ensure_dir, millis
from camera import make_capture, start_stream_loop
from detection import Detector
from ui_controls import toggle_fullscreen, reset_window_to_stream_resolution, mouse_callback
from recorder import start_recording, stop_recording, write_frame

def main():
    # Setup directories
    ensure_dir(snapshot_directory)
    ensure_dir(recording_directory)

    # Init state
    labels, labels_index = load_labels(labels_path)
    state = AppState()
    state.opsize = opsize
    state.window_name = str(rtsp_stream) if window_name is None else window_name

    # Capture
    cap = make_capture(rtsp_stream)
    state.streamsize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    state.hdstream = (state.streamsize[0] > opsize[0] or state.streamsize[1] > opsize[1])
    if state.streamsize[0] > 0 and state.streamsize[1] > 0:
        state.window_aspect_ratio = state.streamsize[0]/state.streamsize[1]
        state.original_window_size = state.streamsize
    else:
        state.window_aspect_ratio = opsize[0]/opsize[1]
        state.original_window_size = opsize

    # Window
    cv2.namedWindow(state.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(state.window_name, opsize[0], opsize[1])
    try:
        cv2.setWindowProperty(state.window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    # Detector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = Detector(labels, labels_index, device, opsize, state.streamsize)
    detector.load_model_async()

    # Mouse
    cv2.setMouseCallback(state.window_name, mouse_callback(state, {
        'zoom_max_factor': 6.0
    }, detector))

    # Queue + Stream thread
    q = queue.Queue(maxsize=buffer)
    import threading
    t = threading.Thread(target=start_stream_loop, args=(state, cap, q), daemon=True)
    t.start()

    # Main loop
    last_click_ms = 0
    recording = None
    print("Starting main loop...")
    while state.loop:
        frame = None
        # replay mode
        if state.replay_mode and len(state.replay_buffer)>0:
            if state.replay_index < len(state.replay_buffer):
                frame = state.replay_buffer[state.replay_index]
                state.replay_index += 1
            else:
                state.replay_mode = False
                state.replay_index = 0
                print("Replay completed")
                continue
        elif not q.empty():
            try:
                frame = q.get_nowait()
            except Exception:
                frame = None

        if frame is None:
            time.sleep(0.01)
            continue

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  
            with q.mutex: q.queue.clear()

        if key == ord('q'): state.loop = False
        elif key == 8:   
            if not state.replay_mode and len(state.replay_buffer)>0:
                state.replay_mode = True
                state.replay_index = 0
                with q.mutex: q.queue.clear()
                print(f"Starting replay of {len(state.replay_buffer)} frames")
            elif state.replay_mode:
                state.replay_mode = False
                with q.mutex: q.queue.clear()
                print("Exiting replay mode")

        elif key == ord('r'):
            # กด r: toggle บันทึก
            if not state.recording:
                # เริ่มบันทึก ใช้ขนาดภาพหลังประมวลผล (opsize) และ fps จากกล้อง
                try:
                    start_recording(state, frame_size=opsize, fps=24, fourcc_str='mp4v')
                except Exception as e:
                    print(f"Cannot start recording: {e}")
            else:
                stop_recording(state)
            with q.mutex: q.queue.clear()
            
            
            
        elif key == ord('s'):
            # snapshot - save current frame
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = f"../{snapshot_directory}/snapshot_{ts}.jpg"
            cv2.imwrite(path, frame)
            print(f"Snapshot saved: {path}")
            with q.mutex: q.queue.clear()
        elif key == ord('f'):
            reset_window_to_stream_resolution(state)
            with q.mutex: q.queue.clear()
        elif key == ord('m'):
            state.military_mode = not state.military_mode
            print(f"Military mode: {'ON' if state.military_mode else 'OFF'}")
            with q.mutex: q.queue.clear()

        # process
        if state.replay_mode:
            img = frame.copy()
            if state.hdstream:
                img = cv2.resize(img, opsize, interpolation=cv2.INTER_LINEAR)
            if state.military_mode:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            if detector.yolo_loading_complete:
                state.basic_stream_mode = False
            if state.basic_stream_mode:
                img = detector.process_basic(frame, state.military_mode, opsize)
            else:
                img = detector.process(frame, state, {
                    'yolo_skip_frames': yolo_skip_frames if not state.military_mode else 1,
                    'stationary_val': stationary_val,
                    'point_timeout': point_timeout,
                })

        write_frame(state, img)        # recording indicator
        if state.recording:
            cv2.putText(img, "REC", (16, img.shape[0]-38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(img, "REC", (16, img.shape[0]-38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        state.frames += 1
        if millis() - state.last_frame_ms >= 1000:
            state.fps = (state.frames - state.prev_frames)
            state.prev_frames = state.frames
            state.last_frame_ms = millis()

        # footer
        lag_ms = 0  # simplified; can compute around process()
        footer = f"{'REPLAY MODE - LAG' if state.replay_mode else 'FPS'}: {state.fps}"
        cv2.putText(img, footer, (16, img.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
        cv2.putText(img, footer, (16, img.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0) if not state.replay_mode else (255,128,0), 1)

        # draw selection
        if state.drawing and state.draw_start_x>0 and state.draw_end_x>0:
            cv2.rectangle(img, (state.draw_start_x, state.draw_start_y), (state.draw_end_x, state.draw_end_y), (0,255,0), 2)

        # clock
        if not state.replay_mode:
            clock = time.strftime("%H:%M:%S")
            (tw, th), _ = cv2.getTextSize(clock, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(img, clock, (img.shape[1]-tw-10, img.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
            cv2.putText(img, clock, (img.shape[1]-tw-10, img.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        cv2.imshow(state.window_name, img)

    print("closing cv window..")
    cv2.destroyAllWindows()
    print("terminating..")

if __name__ == "__main__":
    main()
