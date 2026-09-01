import unittest

from evidence import WeaponEvidence


class WeaponEvidenceTests(unittest.TestCase):
    def detector(self):
        return WeaponEvidence(
            window_seconds=1.0,
            min_hits=3,
            min_mean_conf=0.42,
            immediate_conf=0.90,
            iou_threshold=0.25,
            stale_seconds=2.0,
        )

    def test_single_weak_detection_is_not_confirmed(self):
        ev = self.detector()
        result = ev.update(
            {"box": [0, 0, 100, 100], "class": "handgun", "confidence": 0.30},
            now=0.0,
        )
        self.assertEqual(result["weapon_state"], "SUSPECTED")

    def test_repeated_consistent_detections_confirm(self):
        ev = self.detector()
        states = []
        for i, conf in enumerate((0.51, 0.60, 0.55)):
            result = ev.update(
                {"box": [10, 10, 110, 110], "class": "handgun", "confidence": conf},
                now=i * 0.1,
            )
            states.append(result["weapon_state"])
        self.assertEqual(states[-1], "CONFIRMED")

    def test_high_confidence_can_confirm_immediately(self):
        ev = self.detector()
        result = ev.update(
            {"box": [0, 0, 100, 100], "class": "knife", "confidence": 0.95},
            now=0.0,
        )
        self.assertEqual(result["weapon_state"], "CONFIRMED")

    def test_different_locations_do_not_share_evidence(self):
        ev = self.detector()
        for i in range(2):
            ev.update(
                {"box": [0, 0, 100, 100], "class": "handgun", "confidence": 0.70},
                now=i * 0.1,
            )
        result = ev.update(
            {"box": [400, 400, 500, 500], "class": "handgun", "confidence": 0.70},
            now=0.3,
        )
        self.assertEqual(result["weapon_state"], "SUSPECTED")


if __name__ == "__main__":
    unittest.main()
