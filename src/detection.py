
# -*- coding: utf-8 -*-
import cv2, math, threading, copy
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
import torch
import os

from config import (
    model_name, min_confidence, min_size, yolo_skip_frames, visible_classes,
    class_confidence, padding, zoom_max_factor
)
from utils import generate_color_shades, transform_to_stream, millis, format_duration
from tracking import BoundingBox, reset_iteration

class Detector:
    def __init__(self, labels, labels_index, device, opsize, streamsize):
        self.labels = labels
        self.labels_index = labels_index
        self.colors = generate_color_shades(len(labels))
        self.device = device
        self.model = None
        self.classlist_idx = [labels.index(x) for x in visible_classes if x in labels]
        self.bounding_boxes = []
        self.obj_score_template = labels
        self.stored_bounding_boxes = []
        self.yolo_loaded = False
        self.yolo_loading_complete = False
        self.opsize = opsize
        self.streamsize = streamsize

        self.yolo_frame_count = 0
        self.cached_yolo_results = None

    def load_model_async(self):
        t = threading.Thread(target=self._load_model, daemon=True)
        t.start()

    def _load_model(self):
        print("Loading YOLO model...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "models", f"{model_name}.pt")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.yolo_loaded = True
        self.yolo_loading_complete = True
        print("YOLO model loaded.")

    def _center(self, xmin, ymin, xmax, ymax):
        return ((xmin + xmax)//2, (ymin + ymax)//2)

    def _size1(self, x1, y1, x2, y2):
        return abs(x1 - y2)

    def process_basic(self, frame, military_mode, opsize):
        img = cv2.resize(frame, opsize, interpolation=cv2.INTER_LINEAR) if (frame.shape[1]>opsize[0] or frame.shape[0]>opsize[1]) else frame.copy()
        if military_mode:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        msg = "Loading YOLO model..." if not self.yolo_loading_complete else "YOLO model ready! Switching automatically..."
        cv2.putText(img, msg, (16,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv2.putText(img, msg, (16,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0) if self.yolo_loading_complete else (0,255,255), 1)
        return img

    def process(self, frame, state, cfg):
        # resample for HD -> opsize
        img = self._resample(frame, state)
        if state.military_mode:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Zoom/pan throttling
        now_ms = millis()
        if state.zoom_pan_active and not state.dragging and (now_ms - state.zoom_pan_pause_time_ms > 500):
            state.zoom_pan_active = False
            self.cached_yolo_results = None

        # pick results
        if state.zoom_pan_active or state.dragging:
            results = self.cached_yolo_results
        else:
            if self.yolo_frame_count % max(1, cfg['yolo_skip_frames']) == 0:
                img_tensor = (
                    torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(self.device).float()/255.0
                )
                img_tensor = img_tensor.permute(2,0,1).unsqueeze(0)
                with torch.no_grad():
                    results = self.model(
                        img_tensor, verbose=False, iou=0.45, half=False,
                        max_det=32, conf=min_confidence, classes=self.classlist_idx
                    )
                self.cached_yolo_results = results
            else:
                results = self.cached_yolo_results
            self.yolo_frame_count += 1

        # iterate results
        obj_score = [0 for _ in range(len(self.obj_score_template))]
        points = []
        now = millis()
        boxes = [box for r in results for box in getattr(r, 'boxes', [])] if results else []
        # Only reset iteration when not in zoom mode
        if not state.zoom_mode_active:
            reset_iteration(self.bounding_boxes)

        rawcrowd = []
        for i, box in enumerate(boxes):
            class_id = int(box.cls)
            class_name = self.labels[class_id]
            conf = float(box.conf)

            if class_name in class_confidence and conf <= class_confidence[class_name]:
                continue
            if conf <= min_confidence:
                continue

            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            width, height = (xmax-xmin), (ymax-ymin)
            if (xmin==0 or ymin==0 or xmax==0 or ymax==0 or xmax==img.shape[1] or ymax==img.shape[0]):
                continue
            if class_name=="car" and ((width>height and (width/height)>=2) or (width<min_size or height<min_size)):
                continue

            # if zoomed, draw overlay and skip tracking
            if state.zoom_factor > 1.0:
                color = self.colors[class_id].tolist()
                alpha = 0.35
                overlay = img[ymin:ymax+1, xmin:xmax+1].copy()
                cv2.rectangle(overlay, (0,0), (xmax-xmin, ymax-ymin), color, thickness=-1)
                cv2.addWeighted(overlay, alpha, img[ymin:ymax+1, xmin:xmax+1], 1-alpha, 0, img[ymin:ymax+1, xmin:xmax+1])
                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color, 1)
                continue

            obj_score[class_id] += 1
            cx, cy = self._center(xmin, ymin, xmax, ymax)
            if class_name == "person":
                rawcrowd.append((cx, cy))

            # tracking
            obj = self._get_object((cx, cy), class_name, cfg['stationary_val'], cfg['point_timeout'])
            if obj:
                points.append((cx, cy))
                obj.see()
                # draw
                idle = format_duration(obj.idle)
                cv2.circle(img, (cx, cy), 1, (0,0,255), 2)
                cv2.putText(img, idle, (obj.x, obj.y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 2)
                cv2.putText(img, idle, (obj.x, obj.y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
            else:
                # new object
                cv2.circle(img, (cx, cy), 1, (255,255,0), 2)
                cv2.putText(img, f"{class_name} {round(conf,6)}", (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 2)
                cv2.putText(img, f"{class_name} {round(conf,6)}", (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
                qxmin, qymin, qxmax, qymax = transform_to_stream(xmin, ymin, xmax, ymax, padding, self.streamsize, self.opsize)
                snap = frame[qymin:qymax, qxmin:qxmax]
                nr = sum(1 for _ in self.bounding_boxes) + 1
                b = BoundingBox(nr=nr, name=class_name, x=cx, y=cy, size=self._size1(xmin,ymin,xmax,ymax),
                                image=snap, created=millis(), timestamp=millis(), desc="")
                b.init_stationary(cfg['stationary_val'])
                self.bounding_boxes.append(b)

        # crowd rectangles (visual only)
        if state.zoom_factor <= 1.0:
            clusters = self._get_clusters(np.array(rawcrowd), 50, 2)
            for _, (min_x, min_y, max_x, max_y, count) in clusters.items():
                cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0,255,0), 1)
                cv2.putText(img, f"{count} people", (min_x, min_y-18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 2)
                cv2.putText(img, f"{count} people", (min_x, min_y-18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

            # ping stale objects
            now_ms = millis()
            for obj in list(self.bounding_boxes):
                if (not obj.checkin) and (obj.detections >= 3) and (obj.idle > 1) and (obj.idle < 8):
                    if (now_ms - obj.seen) > 2500:
                        continue
                    cv2.circle(img, (obj.x, obj.y), 1, (0,255,255), 2)

        return img

    def _resample(self, photo, state):
        # if stream is larger than opsize, crop+resize with zoom/pan
        if state.hdstream:
            zoomed_w = int(self.streamsize[0] / state.zoom_factor)
            zoomed_h = int(self.streamsize[1] / state.zoom_factor)
            cx = self.streamsize[0]//2 + state.pan_x
            cy = self.streamsize[1]//2 + state.pan_y
            sx = max(0, min(self.streamsize[0]-zoomed_w, cx - zoomed_w//2))
            sy = max(0, min(self.streamsize[1]-zoomed_h, cy - zoomed_h//2))
            zoomed = photo[sy:sy+zoomed_h, sx:sx+zoomed_w]
            return cv2.resize(zoomed, self.opsize, interpolation=cv2.INTER_LINEAR_EXACT)
        return photo.copy()

    def _get_clusters(self, detected_points, eps=30, min_samples=2):
        if not isinstance(detected_points, np.ndarray) or detected_points.size == 0:
            return {}
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(detected_points)
        labels = db.labels_
        clusters = {}
        for label in set(labels):
            if label == -1: continue
            cluster_points = detected_points[labels == label]
            if cluster_points.size == 0: continue
            min_x, min_y = np.min(cluster_points, axis=0)
            max_x, max_y = np.max(cluster_points, axis=0)
            clusters[label] = (int(min_x), int(min_y), int(max_x), int(max_y), len(cluster_points))
        return clusters

    def _get_object(self, point, cname, stationary_val, point_timeout):
        x, y = point
        now_ms = millis()
        for i in range(len(self.bounding_boxes)-1, -1, -1):
            box = self.bounding_boxes[i]
            if cname != box.name:
                continue
            if box.contains(x, y, now_ms, point_timeout):
                box.checkin = True
                box.timestamp = now_ms
                idle_ms = now_ms - box.created
                box.idle = (idle_ms//1000) if idle_ms >= 1000 else 0
                box.x, box.y = x, y
                box.detections += 1
                box.disappeared_cycles = 0
                box.init_stationary(stationary_val)
                return box
            if box.disappeared_cycles >= 3:
                del self.bounding_boxes[i]
        return None
