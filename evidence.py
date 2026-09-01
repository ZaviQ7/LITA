"""Temporal evidence utilities used by LITA.

This module has no ML dependencies, which makes the critical decision logic easy
to unit test independently of PyTorch/OpenCV.
"""

from collections import deque
from dataclasses import dataclass, field
import time


def box_iou(a, b):
    if not a or not b:
        return 0.0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


@dataclass
class WeaponTrack:
    key: object
    box: list
    label: str
    last_seen: float
    observations: deque = field(default_factory=deque)


class WeaponEvidence:
    """Turns noisy frame-level weapon detections into stable event evidence."""

    def __init__(
        self,
        window_seconds=1.0,
        min_hits=3,
        min_mean_conf=0.42,
        immediate_conf=0.90,
        iou_threshold=0.25,
        stale_seconds=2.0,
    ):
        self.window_seconds = window_seconds
        self.min_hits = min_hits
        self.min_mean_conf = min_mean_conf
        self.immediate_conf = immediate_conf
        self.iou_threshold = iou_threshold
        self.stale_seconds = stale_seconds
        self.tracks = {}
        self._next_pseudo = -1

    def _cleanup(self, now):
        stale = [k for k, v in self.tracks.items() if now - v.last_seen > self.stale_seconds]
        for key in stale:
            del self.tracks[key]

    def _find_spatial_track(self, box, label):
        best_key = None
        best_iou = self.iou_threshold
        for key, track in self.tracks.items():
            if track.label != label:
                continue
            score = box_iou(box, track.box)
            if score > best_iou:
                best_iou = score
                best_key = key
        return best_key

    def update(self, detection, now=None):
        now = time.time() if now is None else now
        self._cleanup(now)

        box = detection["box"]
        label = detection["class"]
        confidence = float(detection["confidence"])
        track_id = detection.get("track_id")

        key = ("id", track_id, label) if track_id is not None else None
        if key is None or key not in self.tracks:
            spatial = self._find_spatial_track(box, label)
            if spatial is not None:
                key = spatial
            else:
                key = ("pseudo", self._next_pseudo, label)
                self._next_pseudo -= 1
                self.tracks[key] = WeaponTrack(key=key, box=box, label=label, last_seen=now)

        track = self.tracks[key]
        track.box = box
        track.last_seen = now
        track.observations.append((now, confidence))
        while track.observations and now - track.observations[0][0] > self.window_seconds:
            track.observations.popleft()

        confidences = [c for _, c in track.observations]
        hits = len(confidences)
        mean_conf = sum(confidences) / hits
        max_conf = max(confidences)
        confirmed = max_conf >= self.immediate_conf or (
            hits >= self.min_hits and mean_conf >= self.min_mean_conf
        )

        return {
            "weapon_state": "CONFIRMED" if confirmed else "SUSPECTED",
            "evidence_hits": hits,
            "evidence_mean_conf": mean_conf,
            "evidence_max_conf": max_conf,
            "evidence_track": str(key),
        }
