"""Scenario-level benchmark for LITA.

Manifest format:
{
  "samples": [
    {
      "path": "benchmarks/images/no_weapon_001.jpg",
      "expect_weapon": false,
      "expect_person": true
    }
  ]
}

This intentionally measures event-level behavior (weapon/no weapon, person/no person)
rather than pretending a single mock image is an accuracy benchmark.
"""

import argparse
import json
from pathlib import Path
import time

import cv2

from inference import ThreatInference


def safe_div(a, b):
    return a / b if b else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/manifest.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing {manifest_path}. Copy benchmarks/manifest.example.json and add labeled samples."
        )

    data = json.loads(manifest_path.read_text())
    detector = ThreatInference()

    tp = fp = tn = fn = 0
    latencies = []
    for sample in data.get("samples", []):
        image = cv2.imread(sample["path"])
        if image is None:
            print(f"SKIP unreadable: {sample['path']}")
            continue

        # Temporal logic requires repeated observations, so evaluate each still over
        # several pseudo-frames. Videos should be sampled into multiple manifest rows.
        detections = []
        start = time.perf_counter()
        for _ in range(max(3, __import__("config").WEAPON_MIN_HITS)):
            detections = detector.run_inference(image)
        latencies.append((time.perf_counter() - start) * 1000)

        predicted = any(
            d.get("kind") == "weapon" and d.get("weapon_state") == "CONFIRMED"
            for d in detections
        )
        expected = bool(sample.get("expect_weapon", False))
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
        else:
            tn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    print("\n========== LITA V2 BENCHMARK ==========")
    print(f"Samples:     {tp + fp + tn + fn}")
    print(f"Precision:   {precision:.3f}")
    print(f"Recall:      {recall:.3f}")
    print(f"F1:          {f1:.3f}")
    print(f"TP/FP/TN/FN: {tp}/{fp}/{tn}/{fn}")
    if latencies:
        ordered = sorted(latencies)
        print(f"Scenario latency p50: {ordered[len(ordered)//2]:.1f} ms")
    print("=======================================\n")


if __name__ == "__main__":
    main()
