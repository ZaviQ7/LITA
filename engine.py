import cv2
import time
import numpy as np
import config

class ThreatHUD:
    def __init__(self):
        pass

    def calculate_score(self, detections):
        score = 0
        unknown_detected = False
        weapon_detected = False

        for det in detections:
            cls = det["class"]
            if cls == "person" and not det.get("is_known", False):
                unknown_detected = True
            elif cls in ("knife", "scissors"):
                weapon_detected = True

        if unknown_detected:
            score += config.THREAT_WEIGHT_UNKNOWN
        if weapon_detected:
            score += config.THREAT_WEIGHT_WEAPON

        score = min(score, config.THREAT_SCORE_MAX)

        if score <= config.STATUS_SAFE_LIMIT:
            status = "SAFE"
            color = (0, 255, 0)
        elif score <= config.STATUS_ELEVATED_LIMIT:
            status = "ELEVATED"
            color = (255, 255, 255)
        else:
            status = "CRITICAL"
            color = (0, 0, 255)

        return score, status, color

    def draw_hud(self, frame, detections, fps_val):
        h, w = frame.shape[:2]
        score, status, status_color = self.calculate_score(detections)

        cv2.putText(frame, f"LITA V1.0 | THREAT LEVEL: {status} ({score}%)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls = det["class"]
            label = det["label"]
            conf = det["confidence"]

            if cls == "person":
                is_known = det.get("is_known", False)
                box_color = (0, 255, 0) if is_known else (255, 255, 255)
                text_color = (0, 0, 0)
            else:
                box_color = (0, 0, 255)
                text_color = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            tag_text = f"{label} ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            label_y = max(y1, 20)
            cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 10, label_y), box_color, -1)
            cv2.putText(frame, tag_text, (x1 + 5, label_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        return frame
