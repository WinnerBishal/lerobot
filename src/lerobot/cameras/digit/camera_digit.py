import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
from numpy.typing import NDArray
from digit_interface import Digit, DigitHandler

from lerobot.cameras.camera import Camera
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.digit.configuration_digit import DigitCameraConfig

logger = logging.getLogger(__name__)

class DigitCamera(Camera):
    def __init__(self, config: DigitCameraConfig):
        super().__init__(config)
        self.config = config
        self.serial_number = config.serial_number
        
        self.digit: Digit | None = None
        
        # Threading resources for async reads
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

    @property
    def is_connected(self) -> bool:
        return self.digit is not None

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            return
            
        try:
            self.digit = Digit(self.serial_number)
            self.digit.connect()
            
            # Apply configuration parameters
            if hasattr(self.digit, "set_intensity"):
                self.digit.set_intensity(self.config.intensity)
            if hasattr(self.digit, "set_resolution"):
                target_res = self.config.resolution.upper()
                if target_res in Digit.STREAMS:
                    self.digit.set_resolution(Digit.STREAMS[target_res])
                else:
                    logger.warning(f"Unknown resolution {target_res}, using default.")
                
            self._start_read_thread()

            if warmup:
                start_time = time.time()
                # Wait for the first valid frame
                while time.time() - start_time < 2.0:
                    try:
                        self.async_read(timeout_ms=2000)
                        break
                    except TimeoutError:
                        time.sleep(0.1)
                
                with self.frame_lock:
                    if self.latest_frame is None:
                        raise ConnectionError(f"{self} failed to capture frames during warmup.")
                        
            logger.info(f"{self} connected.")
            
        except Exception as e:
            self.disconnect()
            raise ConnectionError(f"Failed to open DIGIT sensor {self.serial_number}: {e}")

    def _start_read_thread(self) -> None:
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"digit_{self.serial_number}_read_loop")
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.1)

    def _read_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.digit.get_frame()
                if frame is not None:
                    if self.config.color_mode == ColorMode.RGB:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                    capture_time = time.perf_counter()
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.latest_timestamp = capture_time
                    self.new_frame_event.set()
                else:
                    time.sleep(0.001)
            except Exception as e:
                logger.warning(f"Error reading from DIGIT {self.serial_number}: {e}")
                time.sleep(0.1)
                
        # Safely close the USB connection when the thread is stopped
        if self.digit is not None:
            self.digit.disconnect()
            self.digit = None

    def read(self) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        self.new_frame_event.clear()
        return self.async_read(timeout_ms=5000)

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from DIGIT {self.serial_number}.")
            
        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()
            
        return frame
        
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame is too old: {age_ms:.1f} ms.")

        return frame

    def disconnect(self) -> None:
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join(timeout=3.0)
            self.thread = None
            
        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        found_cameras_info = []
        try:
            digits = DigitHandler.list_digits()
            for d in digits:
                found_cameras_info.append({
                    "name": f"DIGIT Sensor {d['serial']}",
                    "type": "digit",
                    "id": d['serial']
                })
        except Exception as e:
            logger.error(f"Failed to list DIGIT devices: {e}")
            
        return found_cameras_info
        
    def __str__(self) -> str:
        return f"DigitCamera({self.serial_number})"