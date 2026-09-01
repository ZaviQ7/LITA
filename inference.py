import os
import re
import time

import cv2
import numpy as np
import torch
from deepface import DeepFace
from ultralytics import YOLO

import config
from evidence import WeaponEvidence, box_iou


class ThreatInference:
    def __init__(self):
        self.device = config.YOLO_DEVICE
        self.half = config.YOLO_HALF
        if "cuda" in str(self.device).lower() and not torch.cuda.is_available():
            print("[ThreatInference] CUDA requested but unavailable; using CPU.")
            self.device = "cpu"
            self.half = False

        print(f"[ThreatInference] Loading base detector: {config.YOLO_MODEL_NAME}")
        self.yolo = YOLO(config.YOLO_MODEL_NAME)

        self.weapon_yolo = None
        if os.path.isfile(config.WEAPON_MODEL_PATH):
            print(f"[ThreatInference] Loading dedicated weapon model: {config.WEAPON_MODEL_PATH}")
            self.weapon_yolo = YOLO(config.WEAPON_MODEL_PATH)
        else:
            print(
                "[ThreatInference] Dedicated weapon weights not found; "
                "using COCO knife/scissors fallback only."
            )

        self.weapon_evidence = WeaponEvidence(
            window_seconds=config.WEAPON_WINDOW_SECONDS,
            min_hits=config.WEAPON_MIN_HITS,
            min_mean_conf=config.WEAPON_MIN_MEAN_CONF,
            immediate_conf=config.WEAPON_IMMEDIATE_CONF,
            iou_threshold=config.WEAPON_TRACK_IOU_THRESHOLD,
            stale_seconds=config.WEAPON_TRACK_STALE_SECONDS,
        )

        self.whitelist_dir = config.WHITELIST_DIR
        os.makedirs(self.whitelist_dir, exist_ok=True)
        self.whitelist_cache = {}
        self.face_track_cache = {}
        self._pseudo_track_counter = -1
        self.load_whitelist()

    @staticmethod
    def _normalize_name(filename):
        stem = os.path.splitext(filename)[0]
        stem = re.sub(r"[_\s]?\d+$", "", stem)
        return stem.replace("_", " ").title()

    @staticmethod
    def _cosine_distance(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _face_quality(face):
        if face is None or face.size == 0:
            return 0.0
        h, w = face.shape[:2]
        if min(h, w) < config.FACE_MIN_SIZE:
            return 0.0
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(gray.mean())
        sharp_score = min(sharpness / 250.0, 1.0)
        light_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        return 0.7 * sharp_score + 0.3 * light_score

    def _represent(self, image, enforce_detection=True):
        last_error = None
        for backend in config.FACE_DETECTOR_BACKENDS:
            try:
                reps = DeepFace.represent(
                    img_path=image,
                    model_name=config.FACE_MODEL,
                    detector_backend=backend,
                    enforce_detection=enforce_detection,
                    align=True,
                )
                if reps:
                    return reps, backend
            except Exception as exc:
                last_error = exc
        if last_error and enforce_detection:
            return [], None
        return [], None

    def load_whitelist(self):
        valid = (".jpg", ".jpeg", ".png")
        files = sorted(f for f in os.listdir(self.whitelist_dir) if f.lower().endswith(valid))
        if not files:
            print("[ThreatInference] Whitelist is empty.")
            return

        for filename in files:
            name = self._normalize_name(filename)
            path = os.path.join(self.whitelist_dir, filename)
            reps, backend = self._represent(path, enforce_detection=True)
            if not reps:
                print(f"[ThreatInference] Skipping unusable whitelist image: {filename}")
                continue
            for rep in reps:
                embedding = np.asarray(rep["embedding"], dtype=np.float32)
                self.whitelist_cache.setdefault(name, []).append(embedding)
            print(f"[ThreatInference] Enrolled {name} via {backend}: {filename}")

        total = sum(len(v) for v in self.whitelist_cache.values())
        print(f"[ThreatInference] Whitelist: {len(self.whitelist_cache)} identities / {total} embeddings.")

    def _face_track_key(self, track_id, box):
        if track_id is not None:
            return ("id", int(track_id))
        best = None
        best_iou = 0.40
        for key, cached in self.face_track_cache.items():
            score = box_iou(box, cached.get("box"))
            if score > best_iou:
                best_iou = score
                best = key
        if best is not None:
            return best
        key = ("pseudo", self._pseudo_track_counter)
        self._pseudo_track_counter -= 1
        return key

    def _cleanup_faces(self, now):
        stale = [
            k
            for k, v in self.face_track_cache.items()
            if now - v["last_seen"] > config.FACE_TRACK_STALE_SECONDS
        ]
        for key in stale:
            del self.face_track_cache[key]

    def verify_face(self, person_crop, track_id=None, box=None):
        now = time.time()
        self._cleanup_faces(now)
        key = self._face_track_key(track_id, box)

        cached = self.face_track_cache.get(key)
        if cached:
            ttl = (
                config.FACE_CACHE_TTL_KNOWN
                if cached["identity_state"] == "KNOWN"
                else config.FACE_CACHE_TTL_UNKNOWN
            )
            cached["last_seen"] = now
            cached["box"] = box
            if now - cached["last_verified"] < ttl:
                return cached.copy()

        if not self.whitelist_cache:
            result = {
                "identity_state": "UNVERIFIED",
                "identity": None,
                "distance": None,
                "face_quality": 0.0,
                "reason": "empty_whitelist",
            }
            return self._cache_face(key, box, now, result)

        reps, backend = self._represent(person_crop, enforce_detection=True)
        if not reps:
            result = {
                "identity_state": "UNVERIFIED",
                "identity": None,
                "distance": None,
                "face_quality": 0.0,
                "reason": "no_reliable_face",
            }
            return self._cache_face(key, box, now, result)

        rep = max(reps, key=lambda r: r.get("facial_area", {}).get("w", 0) * r.get("facial_area", {}).get("h", 0))
        area = rep.get("facial_area", {})
        x, y, w, h = (area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0))
        face_crop = person_crop[max(0, y): max(0, y) + max(0, h), max(0, x): max(0, x) + max(0, w)]
        quality = self._face_quality(face_crop)
        if quality < config.FACE_MIN_QUALITY:
            result = {
                "identity_state": "UNVERIFIED",
                "identity": None,
                "distance": None,
                "face_quality": quality,
                "reason": "low_face_quality",
            }
            return self._cache_face(key, box, now, result)

        query = np.asarray(rep["embedding"], dtype=np.float32)
        candidates = []
        for name, embeddings in self.whitelist_cache.items():
            distances = sorted(self._cosine_distance(query, emb) for emb in embeddings)
            top = distances[: max(1, min(config.FACE_MATCH_TOP_K, len(distances)))]
            candidates.append((sum(top) / len(top), name, top))

        candidates.sort(key=lambda item: item[0])
        best_distance, best_name, best_distances = candidates[0]
        threshold = config.FACE_MODELS[config.FACE_MODEL]["threshold"]
        votes = sum(distance <= threshold for distance in best_distances)
        required_votes = min(config.FACE_MATCH_MIN_VOTES, len(best_distances))
        is_known = best_distance <= threshold and votes >= required_votes

        result = {
            "identity_state": "KNOWN" if is_known else "UNKNOWN",
            "identity": best_name if is_known else None,
            "candidate": best_name,
            "distance": best_distance,
            "face_quality": quality,
            "reason": "matched" if is_known else "no_whitelist_match",
            "face_backend": backend,
        }
        return self._cache_face(key, box, now, result)

    def _cache_face(self, key, box, now, result):
        cached = dict(result)
        cached.update({"box": box, "last_seen": now, "last_verified": now})
        self.face_track_cache[key] = cached
        return dict(result)

    def _run_base_detector(self, frame):
        kwargs = dict(
            device=self.device,
            half=self.half,
            verbose=False,
            imgsz=config.YOLO_IMGSZ,
            conf=config.YOLO_CONF,
            classes=config.YOLO_CLASS_IDS,
        )
        try:
            return self.yolo.track(
                frame,
                tracker=config.YOLO_TRACKER,
                persist=True,
                **kwargs,
            )
        except Exception:
            return self.yolo(frame, **kwargs)

    @staticmethod
    def _track_id(box):
        if getattr(box, "id", None) is not None:
            return int(box.id[0].item())
        return None

    def _weapon_name(self, raw_name):
        normalized = str(raw_name).lower().strip().replace("-", " ")
        return config.WEAPON_CLASS_ALIASES.get(normalized, normalized.replace(" ", "_"))

    def _weapon_detection(self, box, names):
        cls_id = int(box.cls[0].item())
        raw_name = names[cls_id]
        class_name = self._weapon_name(raw_name)
        conf = float(box.conf[0].item())
        coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
        det = {
            "box": coords,
            "class": class_name,
            "confidence": conf,
            "label": class_name.upper(),
            "track_id": self._track_id(box),
            "kind": "weapon",
        }
        det.update(self.weapon_evidence.update(det))
        det["label"] = f"{det['weapon_state']}: {class_name.upper()}"
        return det

    def _run_weapon_model(self, frame):
        if self.weapon_yolo is None:
            return []
        try:
            results = self.weapon_yolo.track(
                frame,
                device=self.device,
                half=self.half,
                verbose=False,
                imgsz=config.WEAPON_MODEL_IMGSZ,
                conf=config.WEAPON_MODEL_CONF,
                tracker=config.YOLO_TRACKER,
                persist=True,
            )
        except Exception:
            results = self.weapon_yolo(
                frame,
                device=self.device,
                half=self.half,
                verbose=False,
                imgsz=config.WEAPON_MODEL_IMGSZ,
                conf=config.WEAPON_MODEL_CONF,
            )
        if not results:
            return []
        return [self._weapon_detection(box, self.weapon_yolo.names) for box in results[0].boxes]

    def run_inference(self, frame):
        detections = []
        results = self._run_base_detector(frame)
        if results:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                class_name = self.yolo.names[cls_id]
                conf = float(box.conf[0].item())
                if conf < config.YOLO_CLASS_CONF.get(class_name, config.YOLO_CONF):
                    continue

                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = coords
                track_id = self._track_id(box)

                if class_name == "person":
                    h, w = frame.shape[:2]
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(w, x2), min(h, y2)
                    identity = {
                        "identity_state": "UNVERIFIED",
                        "identity": None,
                        "distance": None,
                    }
                    if x2c > x1c and y2c > y1c:
                        identity = self.verify_face(
                            frame[y1c:y2c, x1c:x2c],
                            track_id=track_id,
                            box=coords,
                        )
                    state = identity["identity_state"]
                    label = f"KNOWN: {identity['identity']}" if state == "KNOWN" else state
                    detections.append(
                        {
                            "box": coords,
                            "class": "person",
                            "kind": "person",
                            "confidence": conf,
                            "track_id": track_id,
                            "label": label,
                            **identity,
                        }
                    )
                elif self.weapon_yolo is None:
                    # COCO fallback only. A dedicated model supersedes these detections.
                    detections.append(self._weapon_detection(box, self.yolo.names))

        detections.extend(self._run_weapon_model(frame))
        return detections
