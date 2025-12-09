import threading
import time
import numpy as np
import zmq
from typing import Optional
from numpy.typing import NDArray

class ZMQCapture:
    # --- Standard Stream Profiles ---
    COLOR_PROFILE = {
        "port": 5556,
        "width": 1280,
        "height": 720,
        "channels": 3,
        "dtype_str": "uint8",
        "fps": 30
    }
    
    DEPTH_PROFILE = {
        "port": 5555,
        "width": 480,
        "height": 270,
        "channels": 1,
        "dtype_str": "uint16",
        "fps": 30
    }
    # --------------------------------

    def __init__(
        self, 
        host: str, 
        port: int, 
        width: int, 
        height: int, 
        channels: int, 
        dtype: np.dtype,
        connect_timeout: float = 10.0
    ):
        self.url = f"tcp://{host}:{port}"
        self.width = width
        self.height = height
        self.channels = channels
        self.dtype = dtype
        self.connect_timeout = connect_timeout
        
        self.expected_bytes = width * height * channels * dtype.itemsize

        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.latest_frame: Optional[NDArray] = None
        self.frame_lock = threading.Lock()

    def connect(self):
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.setsockopt(zmq.CONFLATE, 1)
            
            print(f"Connecting to {self.url}...")
            self.socket.connect(self.url)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            
            self.running = True
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
            
            self._wait_for_first_frame()
            
        except Exception as e:
            self.disconnect()
            raise ConnectionError(f"Failed to connect to ZMQ stream at {self.url}: {e}")

    def _wait_for_first_frame(self):
        start_t = time.time()
        while time.time() - start_t < self.connect_timeout:
            if self.read() is not None:
                print(f"✅ Connected to {self.url}")
                return
            time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for data from {self.url}")

    def _worker(self):
        while self.running and self.context:
            try:
                raw_bytes = self.socket.recv()

                if len(raw_bytes) != self.expected_bytes:
                    print(f"Warning: Dropped frame. Got {len(raw_bytes)} bytes, expected {self.expected_bytes}")
                    continue

                frame = np.frombuffer(raw_bytes, dtype=self.dtype)
                frame = frame.reshape((self.height, self.width, self.channels))

                if self.channels == 1 and frame.ndim == 2:
                    frame = frame[..., None]

                with self.frame_lock:
                    self.latest_frame = frame

            except zmq.ContextTerminated:
                break
            except Exception as e:
                if self.running:
                    print(f"ZMQ Error: {e}")

    def read(self) -> Optional[NDArray]:
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def disconnect(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if self.socket:
            self.socket.close(linger=0)
            self.socket = None
        
        if self.context:
            self.context.term()
            self.context = None