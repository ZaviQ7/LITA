import time
from stream import VideoStreamThread
import cv2

def main():
    print("[verify_stream] Starting stream thread test with source: mock_camera.png...")
    # Initialize thread
    thread = VideoStreamThread(src="mock_camera.png")
    thread.start()
    
    # Wait a bit for initialization
    time.sleep(2.0)
    
    frames_received = 0
    start_time = time.time()
    test_duration = 5.0  # test for 5 seconds
    
    print(f"[verify_stream] Reading frames for {test_duration} seconds...")
    
    while time.time() - start_time < test_duration:
        frame = thread.get_frame()
        if frame is not None:
            frames_received += 1
            # Perform a fast check on dimensions
            h, w, c = frame.shape
            if frames_received % 30 == 0:
                print(f"[verify_stream] Received frame {frames_received}: resolution {w}x{h}, channels {c}")
        time.sleep(0.01)  # sleep briefly to yield control
        
    actual_duration = time.time() - start_time
    fps = frames_received / actual_duration
    print(f"[verify_stream] Test complete!")
    print(f"[verify_stream] Received {frames_received} frames over {actual_duration:.2f} seconds.")
    print(f"[verify_stream] Average frame rate retrieved: {fps:.2f} FPS")
    
    print("[verify_stream] Stopping stream thread...")
    thread.stop()
    print("[verify_stream] Thread stopped successfully.")

if __name__ == "__main__":
    main()
