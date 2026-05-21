import time
import torch
from inference import ThreatInference
import cv2

def main():
    print("==================================================")
    print("      LITA YOLO & CUDA ACCELERATION VERIFIER      ")
    print("==================================================")

    # Check torch CUDA status
    cuda_available = torch.cuda.is_available()
    print(f"[verify_yolo] PyTorch CUDA available: {cuda_available}")
    if cuda_available:
        print(f"[verify_yolo] CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"[verify_yolo] Current Device Index: {torch.cuda.current_device()}")

    # Initialize ThreatInference (loads YOLOv8m.pt on cuda:0 with FP16)
    print("[verify_yolo] Initializing ThreatInference class...")
    detector = ThreatInference()

    # Load mock camera image
    image_path = "mock_camera.png"
    print(f"[verify_yolo] Loading mock frame from: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[verify_yolo] ERROR: Could not load {image_path}")
        return

    # Warmup frame
    print("[verify_yolo] Running GPU warmup inference...")
    detector.run_inference(frame)
    
    # Benchmarking
    iterations = 20
    total_time = 0.0
    detections = []
    
    print(f"[verify_yolo] Running {iterations} benchmarking iterations...")
    for i in range(iterations):
        t0 = time.perf_counter()
        detections = detector.run_inference(frame)
        dt = time.perf_counter() - t0
        total_time += dt
        
    avg_latency_ms = (total_time / iterations) * 1000.0
    print("\n================ BENCHMARK RESULTS ================")
    print(f"Average Inference Latency: {avg_latency_ms:.2f} ms")
    print(f"CUDA Active: {detector.device == 'cuda:0'}")
    print(f"FP16 Math (Half Precision): {detector.half}")
    print(f"Detections Found on Mock Frame:")
    for det in detections:
        print(f" - Class: {det['class']}, Confidence: {det['confidence']:.2f}, Box: {det['box']}, Label: {det['label']}")
    print("==================================================")

    # Success assertion
    if avg_latency_ms < 25.0:  # Allow up to 25ms to account for various GPU series, but target <15ms
        print("[verify_yolo] SUCCESS: Inference latency meets speed threshold!")
    else:
        print("[verify_yolo] WARNING: Latency exceeds optimal speed target.")

if __name__ == "__main__":
    main()
