
# -*- coding: utf-8 -*-
import base64, cv2
from typing import List, Tuple
from dataclasses import dataclass
from utils import millis, format_duration

@dataclass
class BoundingBox:
    nr: int
    name: str
    x: int
    y: int
    size: int
    image: any
    created: int
    timestamp: int
    checkin: bool = True
    detections: int = 0
    idle: int = 0
    desc: str = ""
    state: int = 0
    seen: int = 0
    disappeared_cycles: int = 0
    min_x: int = 0
    max_x: int = 0
    min_y: int = 0
    max_y: int = 0

    def see(self):
        self.seen = millis()

    def ping(self):
        self.timestamp = millis()
        idle = self.timestamp - self.created
        self.idle = idle//1000 if idle >= 1000 else 0
        return self.idle

    def export_b64(self) -> str:
        _, buffer = cv2.imencode(".png", self.image)
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def init_stationary(self, stationary_val: int):
        self.min_x = self.x - stationary_val
        self.max_x = self.x + stationary_val
        self.min_y = self.y - stationary_val
        self.max_y = self.y + stationary_val

    def contains(self, x, y, current_ms: int, point_timeout_ms: int) -> bool:
        return (not self.checkin) and (self.min_x <= x <= self.max_x) and \
               (self.min_y <= y <= self.max_y) and (current_ms - self.seen < point_timeout_ms)

def reset_iteration(boxes: List[BoundingBox]):
    for b in boxes:
        if b.checkin:
            b.disappeared_cycles = 0
        else:
            b.disappeared_cycles += 1
        b.checkin = False
