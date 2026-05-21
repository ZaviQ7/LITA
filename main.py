import time
import cv2
import threading
import queue
import config
from stream import VideoStreamThread
from inference import ThreatInference
from engine import ThreatHUD

def inference_worker(input_queue, output_queue, stop_event):
    print("[InferenceWorker] Thread started. Loading models...")
    detector = ThreatInference()
    print("[InferenceWorker] ML Models loaded and initialized.")

    while not stop_event.is_set():
        try:
            frame = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        detections = detector.run_inference(frame)

        if output_queue.full():
            try:
                output_queue.get_nowait()
            except queue.Empty:
                pass
        
        output_queue.put(detections)

    print("[InferenceWorker] Thread shut down.")

def main():
    print("==================================================")
    print("      LOCAL INTELLIGENT THREAT ASSESSMENT (LITA)   ")
    print("==================================================")

    input_q = queue.Queue(maxsize=1)
    output_q = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    stream_thread = VideoStreamThread(src=config.WEBCAM_SOURCE)
    stream_thread.start()

    inf_thread = threading.Thread(
        target=inference_worker,
        args=(input_q, output_q, stop_event),
        daemon=True
    )
    inf_thread.start()

    hud = ThreatHUD()

    print("[Main] Initializing high-speed display loop...")
    time.sleep(1.0)

    prev_time = time.time()
    detections = []
    fps_val = 30.0

    cv2.namedWindow("LITA Stream HUD Overlay", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = stream_thread.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            if input_q.empty():
                input_q.put(frame.copy())

            try:
                detections = output_q.get_nowait()
            except queue.Empty:
                pass

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_val = 0.9 * fps_val + 0.1 * (1.0 / dt)

            annotated_frame = hud.draw_hud(frame, detections, fps_val)

            cv2.imshow("LITA Stream HUD Overlay", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    except KeyboardInterrupt:
        print("[Main] User interrupt caught.")
    finally:
        print("[Main] Stopping LITA and releasing resource handles...")
        stop_event.set()
        stream_thread.stop()
        cv2.destroyAllWindows()
        print("[Main] Graceful shutdown completed. Systems OFFLINE.")

if __name__ == "__main__":
    main()
