WEBCAM_SOURCE = 0

WHITELIST_DIR = "./whitelist"
FACE_MODEL = "Facenet512"

FACE_MODELS = {
    "Facenet512": {
        "model_name": "Facenet512",
        "threshold": 0.68
    },
    "ArcFace": {
        "model_name": "ArcFace",
        "threshold": 0.68
    }
}

FACE_CROP_RATIOS = [0.45, 0.55, 0.75, 0.35]
FACE_CACHE_TTL = 1.5

YOLO_MODEL_NAME = "yolov8m.pt"
YOLO_DEVICE = "cuda:0"
YOLO_HALF = True
YOLO_IMGSZ = 640
YOLO_CONF = 0.20

FACE_RECOGNITION_COOLDOWN = 1.5


YOLO_CLASS_IDS = [0, 43, 76]
YOLO_CLASS_CONF = {
    "person": 0.45,
    "knife": 0.20,
    "scissors": 0.20,
}

YOLO_CLASSES = ["person", "knife", "scissors"]

THREAT_WEIGHT_UNKNOWN = 30
THREAT_WEIGHT_WEAPON = 60
THREAT_SCORE_MAX = 100

STATUS_SAFE_LIMIT = 29
STATUS_ELEVATED_LIMIT = 59

