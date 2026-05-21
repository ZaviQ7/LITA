import os
import re
import time
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import config

class ThreatInference:
    def __init__(self):
        print(f"[ThreatInference] Loading YOLO model: {config.YOLO_MODEL_NAME}...")
        self.yolo = YOLO(config.YOLO_MODEL_NAME)
        self.device = config.YOLO_DEVICE
        self.half = config.YOLO_HALF
        print(f"[ThreatInference] YOLO initialized on {self.device} with half={self.half}")

        self.whitelist_dir = config.WHITELIST_DIR
        os.makedirs(self.whitelist_dir, exist_ok=True)
        self.write_whitelist_readme()

        self.whitelist_cache = {}
        self.load_whitelist()

        self._face_cache_result = None
        self._face_cache_time = 0

    def write_whitelist_readme(self):
        readme_path = os.path.join(self.whitelist_dir, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write("# LITA Whitelist Portraits Directory\n\n"
                        "Drop your authenticated `.jpg` (or `.png`) portrait profiles in this directory.\n"
                        "LITA will pre-compute and cache facial embeddings for these files at startup.\n\n"
                        "Naming convention:\n"
                        "Use the person's name as the filename (e.g., `john_doe.jpg` or `jane_doe.jpg`).\n"
                        "LITA will display this name as the 'KNOWN' label on the HUD.\n")
            print(f"[ThreatInference] Created whitelist README at {readme_path}")

    @staticmethod
    def _normalize_name(filename):
        stem = os.path.splitext(filename)[0]
        stem = re.sub(r'[_\s]?\d+$', '', stem)
        return stem.replace("_", " ").title()

    def load_whitelist(self):
        print(f"[ThreatInference] Initializing Whitelist Database from '{self.whitelist_dir}'...")
        valid_extensions = (".jpg", ".jpeg", ".png")
        files = [f for f in os.listdir(self.whitelist_dir) if f.lower().endswith(valid_extensions)]

        if not files:
            print("[ThreatInference] Whitelist database is currently empty. Drop .jpg portraits to verify known individuals.")
            return

        for file in files:
            name = self._normalize_name(file)
            path = os.path.join(self.whitelist_dir, file)
            try:
                representations = DeepFace.represent(
                    img_path=path,
                    model_name=config.FACE_MODEL,
                    enforce_detection=True,
                    detector_backend="ssd"
                )
                if representations:
                    embedding = np.array(representations[0]["embedding"])
                    if name not in self.whitelist_cache:
                        self.whitelist_cache[name] = []
                    self.whitelist_cache[name].append(embedding)
                    count = len(self.whitelist_cache[name])
                    print(f"[ThreatInference] Cached whitelist profile: '{name}' (embedding #{count} from {file})")
            except Exception as e:
                print(f"[ThreatInference] WARNING: Failed to compute embedding for {file}: {e}")

        total_embeddings = sum(len(v) for v in self.whitelist_cache.values())
        print(f"[ThreatInference] Whitelist load completed. {len(self.whitelist_cache)} identities, {total_embeddings} total embeddings.")

    def _extract_face_embedding(self, person_crop):
        h, w = person_crop.shape[:2]
        if h <= 0 or w <= 0:
            return None

        for ratio in config.FACE_CROP_RATIOS:
            face_region = person_crop[0:int(h * ratio), 0:w]
            try:
                representations = DeepFace.represent(
                    img_path=face_region,
                    model_name=config.FACE_MODEL,
                    enforce_detection=True,
                    detector_backend="ssd"
                )
                if representations:
                    return np.array(representations[0]["embedding"])
            except Exception:
                continue

        return None

    def verify_face(self, person_crop):
        if not self.whitelist_cache:
            return False, "UNKNOWN", 1.0

        now = time.time()
        if (self._face_cache_result
                and self._face_cache_result[0]
                and (now - self._face_cache_time) < config.FACE_CACHE_TTL):
            return self._face_cache_result

        crop_emb = self._extract_face_embedding(person_crop)
        if crop_emb is None:
            return False, "UNKNOWN", 1.0

        best_match = "UNKNOWN"
        min_distance = 1.0
        norm_crop = np.linalg.norm(crop_emb)
        if norm_crop == 0:
            return False, "UNKNOWN", 1.0

        for name, embeddings_list in self.whitelist_cache.items():
            for whitelist_emb in embeddings_list:
                norm_white = np.linalg.norm(whitelist_emb)
                if norm_white > 0:
                    similarity = np.dot(crop_emb, whitelist_emb) / (norm_crop * norm_white)
                    distance = 1.0 - similarity
                    if distance < min_distance:
                        min_distance = distance
                        best_match = name

        threshold = config.FACE_MODELS[config.FACE_MODEL]["threshold"]
        if min_distance <= threshold:
            result = (True, best_match, min_distance)
            self._face_cache_result = result
            self._face_cache_time = now
            return result
        else:
            return False, "UNKNOWN", min_distance

    def run_inference(self, frame):
        results = self.yolo(
            frame, device=self.device, half=self.half, verbose=False,
            imgsz=config.YOLO_IMGSZ, conf=config.YOLO_CONF,
            classes=config.YOLO_CLASS_IDS
        )
        detections = []

        if not results:
            return detections

        result = results[0]
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = self.yolo.names[cls_id]
            conf = float(box.conf[0].item())

            min_conf = config.YOLO_CLASS_CONF.get(class_name, config.YOLO_CONF)
            if conf < min_conf:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            detection = {
                "box": [x1, y1, x2, y2],
                "class": class_name,
                "confidence": conf,
                "label": class_name.upper()
            }

            if class_name == "person":
                h, w = frame.shape[:2]
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, y2)
                
                if x2_c > x1_c and y2_c > y1_c:
                    person_crop = frame[y1_c:y2_c, x1_c:x2_c]
                    is_known, label, distance = self.verify_face(person_crop)
                    
                    if is_known:
                        detection["label"] = f"KNOWN: {label}"
                        detection["is_known"] = True
                    else:
                        detection["label"] = "UNKNOWN"
                        detection["is_known"] = False
                else:
                    detection["label"] = "UNKNOWN"
                    detection["is_known"] = False
            else:
                detection["label"] = "WEAPON"

            detections.append(detection)

        return detections
