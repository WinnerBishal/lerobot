import numpy as np
import cv2
from typing import Optional

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode
from .configuration_network_camera import NetworkCameraConfig
from .zmq_capture import ZMQCapture

class NetworkCamera(Camera):
    def __init__(self, config: NetworkCameraConfig):
        # 1. Initialize base class (stores self.fps, self.width, self.height)
        super().__init__(config)
        
        self.config = config
        self.backend: Optional[ZMQCapture] = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            return

        # 2. Use self.width/height directly (guaranteed by config validation)
        self.backend = ZMQCapture(
            host=self.config.host,
            port=self.config.port,
            width=self.width, 
            height=self.height,
            channels=self.config.channels,
            dtype=self.config.numpy_dtype,
            connect_timeout=self.config.connect_timeout
        )
        
        self.backend.connect()
        self._is_connected = True

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        if not self.is_connected or not self.backend:
            raise ConnectionError("NetworkCamera is not connected.")

        frame = self.backend.read()

        if frame is None:
            # Return safe black frame if drop occurs
            return np.zeros(
                (self.height, self.width, self.config.channels), 
                dtype=self.config.numpy_dtype
            )

        # Handle Color Conversion (assuming stream is RGB)
        if self.config.channels == 3 and color_mode == ColorMode.BGR:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame

    def async_read(self, timeout_ms: float = 0) -> np.ndarray:
        return self.read()

    def disconnect(self) -> None:
        if self.backend:
            self.backend.disconnect()
        self._is_connected = False