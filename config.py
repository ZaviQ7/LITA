"""Central configuration for LITA V2.

The defaults remain usable with the stock COCO model. If a specialized weapon
model is placed at WEAPON_MODEL_PATH, LITA automatically uses it for weapon
detection while keeping the stock detector for people.
"""

WEBCAM_SOURCE = 0

# Face recognition
WHITELIST_DIR = "./whitelist"
FACE_MODEL = "ArcFace"
FACE_DETECTOR_BACKENDS = ("retinaface", "opencv")
FACE_DISTANCE_METRIC = "cosine"

# Starting points only. Calibrate with benchmark.py for your camera and users.
FACE_MODELS = {
    "Facenet512": {"model_name": "Facenet512", "threshold": 0.30},
    "ArcFace": {"model_name": "ArcFace", "threshold": 0.32},
}
FACE_CACHE_TTL_KNOWN = 2.0
FACE_CACHE_TTL_UNKNOWN = 0.8
FACE_TRACK_STALE_SECONDS = 8.0
FACE_MIN_SIZE = 48
FACE_MIN_QUALITY = 0.18
FACE_MATCH_TOP_K = 3
FACE_MATCH_MIN_VOTES = 2

# Base detector (people + COCO weapon fallback)
YOLO_MODEL_NAME = "yolov8m.pt"
YOLO_DEVICE = "cuda:0"
YOLO_HALF = True
YOLO_IMGSZ = 640
YOLO_CONF = 0.20
YOLO_TRACKER = "botsort.yaml"

YOLO_CLASS_IDS = [0, 43, 76]  # person, knife, scissors in COCO
YOLO_CLASS_CONF = {
    "person": 0.45,
    "knife": 0.28,
    "scissors": 0.35,
}

# Optional dedicated weapon model
WEAPON_MODEL_PATH = "./models/lita_weapon.pt"
WEAPON_MODEL_IMGSZ = 960
WEAPON_MODEL_CONF = 0.18
WEAPON_CLASS_ALIASES = {
    "gun": "handgun",
    "pistol": "handgun",
    "revolver": "handgun",
    "handgun": "handgun",
    "rifle": "long_gun",
    "shotgun": "long_gun",
    "long gun": "long_gun",
    "long_gun": "long_gun",
    "knife": "knife",
    "weapon": "other_weapon",
    "other weapon": "other_weapon",
    "other_weapon": "other_weapon",
}

# Temporal weapon evidence
WEAPON_WINDOW_SECONDS = 1.0
WEAPON_MIN_HITS = 3
WEAPON_MIN_MEAN_CONF = 0.42
WEAPON_IMMEDIATE_CONF = 0.90
WEAPON_TRACK_IOU_THRESHOLD = 0.25
WEAPON_TRACK_STALE_SECONDS = 2.0

# Threat scoring
THREAT_WEIGHT_UNKNOWN = 25
THREAT_WEIGHT_UNVERIFIED = 0
THREAT_WEIGHT_WEAPON_SUSPECTED = 20
THREAT_WEIGHT_WEAPON_CONFIRMED = 75
THREAT_SCORE_MAX = 100

STATUS_SAFE_LIMIT = 24
STATUS_ELEVATED_LIMIT = 59
