import cv2
import config


class ThreatHUD:
    def calculate_score(self, detections):
        unknown = any(
            d.get("kind") == "person" and d.get("identity_state") == "UNKNOWN"
            for d in detections
        )
        unverified = any(
            d.get("kind") == "person" and d.get("identity_state") == "UNVERIFIED"
            for d in detections
        )
        confirmed_weapon = any(
            d.get("kind") == "weapon" and d.get("weapon_state") == "CONFIRMED"
            for d in detections
        )
        suspected_weapon = any(
            d.get("kind") == "weapon" and d.get("weapon_state") == "SUSPECTED"
            for d in detections
        )

        score = 0
        if unknown:
            score += config.THREAT_WEIGHT_UNKNOWN
        if unverified:
            score += config.THREAT_WEIGHT_UNVERIFIED
        if confirmed_weapon:
            score += config.THREAT_WEIGHT_WEAPON_CONFIRMED
        elif suspected_weapon:
            score += config.THREAT_WEIGHT_WEAPON_SUSPECTED

        score = min(score, config.THREAT_SCORE_MAX)
        if score <= config.STATUS_SAFE_LIMIT:
            return score, "SAFE", (0, 255, 0)
        if score <= config.STATUS_ELEVATED_LIMIT:
            return score, "ELEVATED", (255, 255, 255)
        return score, "CRITICAL", (0, 0, 255)

    def draw_hud(self, frame, detections, fps_val):
        score, status, status_color = self.calculate_score(detections)
        cv2.putText(
            frame,
            f"LITA V2 | {status} ({score}) | {fps_val:.1f} FPS",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            status_color,
            2,
            cv2.LINE_AA,
        )

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            conf = det["confidence"]

            if det.get("kind") == "person":
                state = det.get("identity_state", "UNVERIFIED")
                if state == "KNOWN":
                    box_color = (0, 255, 0)
                elif state == "UNKNOWN":
                    box_color = (0, 165, 255)
                else:
                    box_color = (220, 220, 220)
            else:
                box_color = (0, 0, 255) if det.get("weapon_state") == "CONFIRMED" else (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            tag = f"{label} ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            label_y = max(y1, 20)
            cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 10, label_y), box_color, -1)
            cv2.putText(
                frame,
                tag,
                (x1 + 5, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0) if det.get("kind") == "person" else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return frame
