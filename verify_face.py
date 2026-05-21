import os
import cv2
import shutil
from inference import ThreatInference

def main():
    print("==================================================")
    print("      LITA FACIAL VERIFICATION INTEGRATION TEST   ")
    print("==================================================")

    whitelist_dir = "./whitelist"
    if os.path.exists(whitelist_dir):
        for f in os.listdir(whitelist_dir):
            if f.endswith((".jpg", ".png", ".jpeg")):
                os.remove(os.path.join(whitelist_dir, f))
    os.makedirs(whitelist_dir, exist_ok=True)

    image_path = "mock_camera.png"
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[verify_face] ERROR: Could not load {image_path}")
        return

    print("[verify_face] Phase 1: Running initial inference with empty whitelist...")
    detector_initial = ThreatInference()
    detections = detector_initial.run_inference(frame)
    
    person_det = None
    for det in detections:
        if det["class"] == "person":
            person_det = det
            break
            
    if person_det is None:
        print("[verify_face] ERROR: No person detected in initial run. Cannot verify.")
        return

    print(f"[verify_face] Initial Detection Result: {person_det['label']} (Expected: UNKNOWN)")
    assert person_det["label"] == "UNKNOWN", "Person should be UNKNOWN on initial run"

    print("[verify_face] Phase 2: Cropping person's face/upper body to simulate whitelist portrait...")
    x1, y1, x2, y2 = person_det["box"]
    
    box_h = y2 - y1
    face_crop = frame[y1:y1 + int(box_h * 0.4), x1:x2]
    
    portrait_path = os.path.join(whitelist_dir, "jane.jpg")
    cv2.imwrite(portrait_path, face_crop)
    print(f"[verify_face] Saved mock portrait to: {portrait_path}")

    print("\n[verify_face] Phase 3: Re-initializing ThreatInference with whitelisted portrait...")
    detector_whitelist = ThreatInference()
    
    print("[verify_face] Running inference on mock frame with active whitelist...")
    detections_new = detector_whitelist.run_inference(frame)
    
    person_det_new = None
    for det in detections_new:
        if det["class"] == "person":
            person_det_new = det
            break
            
    if person_det_new is None:
        print("[verify_face] ERROR: Person not found in second run.")
        return

    print(f"[verify_face] Second Detection Result: {person_det_new['label']} (Expected: KNOWN: Jane)")
    
    if "KNOWN: Jane" in person_det_new["label"]:
        print("\n==================================================")
        print(" SUCCESS: LITA FACIAL VERIFICATION WORKING FLAWLESSLY!")
        print(f" Detected label changed from UNKNOWN to: {person_det_new['label']}")
        print("==================================================")
    else:
        print("\n==================================================")
        print(" FAILURE: Face verification did not match.")
        print(f" Label returned: {person_det_new['label']}")
        print("==================================================")

if __name__ == "__main__":
    main()
