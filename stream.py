import queue
import threading
import time
import cv2


class VideoStreamThread(threading.Thread):
    """
    VideoStreamThread continuously polls the camera stream (IP URL or local index)
    in a background thread. It ensures the queue contains only the freshest frames,
    dropping older frames to avoid lag and thread choking.
    """
    def __init__(self, src=0):
        super().__init__()
        self.src = src
        self.is_file = False
        self.mock_frame = None
        
        import os
        if isinstance(self.src, str) and os.path.isfile(self.src):
            self.is_file = True
            self.mock_frame = cv2.imread(self.src)
            print(f"[VideoStreamThread] Initialized in mock file mode with path: {self.src}")
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(self.src)
            
        self.queue = queue.Queue(maxsize=2)
        self.running = False
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        print(f"[VideoStreamThread] Stream thread started for source: {self.src}")

        while self.running:
            if self.is_file:
                if self.mock_frame is None:
                    time.sleep(0.1)
                    continue
                frame = self.mock_frame.copy()
                time.sleep(0.033)  # Sleep ~33ms to simulate 30 FPS stream
            else:
                # Grab the next frame (fast operation)
                grabbed = self.cap.grab()
                if not grabbed:
                    print("[VideoStreamThread] Stream disconnected or failed to grab frame. Reconnecting...")
                    with self.lock:
                        self.cap.release()
                        time.sleep(1.0)
                        self.cap = cv2.VideoCapture(self.src)
                    continue

                # Retrieve and decode the frame (more expensive operation, only done for freshest)
                ret, frame = self.cap.retrieve()
                if not ret or frame is None:
                    continue

            # Push the freshest frame into the queue.
            # If the queue is full, drop the oldest frame to prevent thread-choking and lag.
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.queue.put(frame)

        if self.cap is not None:
            self.cap.release()
        print("[VideoStreamThread] Stream thread terminated.")

    def get_frame(self):
        """
        Retrieve the latest frame from the queue. Non-blocking.
        Returns None if no frame is available.
        """
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """
        Stop the thread and release the camera capture resources.
        """
        self.running = False
        if self.is_alive():
            self.join(timeout=2.0)
