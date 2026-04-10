import numpy as np
import cv2
from typing import Optional, List, Dict, Any

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
    
    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        """
        Network cameras cannot be automatically discovered. 
        Returns an empty list to satisfy the Camera interface.
        """
        return []

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
            return np.zeros(
                (self.height, self.width, self.config.channels), 
                dtype=self.config.numpy_dtype
            )

        # --- FIX: Handle Depth Map Normalization and Colorization ---
        if self.config.mode == "depth":
            # 1. Ensure it's a 2D array for processing
            if frame.ndim == 3:
                frame = frame[:, :, 0]
                
            # 2. Normalize raw depth (e.g., 0-10000mm) to 0-255 uint8
            # We use a 4000mm (4 meter) cutoff for better visual contrast
            frame_norm = np.clip(frame, 0, 4000) / 4000.0 * 255
            frame_uint8 = frame_norm.astype(np.uint8)
            
            # 3. Apply a colormap (Jet) to create a 3-channel RGB image
            # This fixes both the "float range" error and the "1 vs 3 channels" requirement
            frame = cv2.applyColorMap(frame_uint8, cv2.COLORMAP_JET)
            
            # 4. Optional: Convert to RGB (OpenCV defaults to BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Handle Color Conversion for standard RGB streams
        elif self.config.channels == 3 and color_mode == ColorMode.BGR:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame

    def async_read(self, timeout_ms: float = 0) -> np.ndarray:
        return self.read()

    def disconnect(self) -> None:
        if self.backend:
            self.backend.disconnect()
        self._is_connected = False