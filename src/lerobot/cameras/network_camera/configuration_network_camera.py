from dataclasses import dataclass
from typing import Optional
import numpy as np

from lerobot.cameras.configs import CameraConfig
from .zmq_capture import ZMQCapture

@CameraConfig.register_subclass("networkcam")
@dataclass
class NetworkCameraConfig(CameraConfig):
    """
    Configuration for ZMQ Camera Streams.
    
    Args:
        mode (str): 'color' or 'depth'. Automatically sets resolution, port, and format.
                    Set to 'custom' (default) to specify everything manually.
        host (str): IP address of the sender (default: localhost).
        port (int): ZMQ port (overrides mode default).
        ...
    """
    host: str = "localhost"
    port: Optional[int] = None
    
    # Master switch for defaults
    mode: str = "custom"  # "color", "depth", "custom"
    
    # Stream specifics (Optional, filled by mode defaults)
    channels: Optional[int] = None
    dtype_str: Optional[str] = None
    
    connect_timeout: float = 10.0

    def __post_init__(self):
        # 1. Select Profile based on mode
        defaults = {}
        if self.mode == "color":
            defaults = ZMQCapture.COLOR_PROFILE
        elif self.mode == "depth":
            defaults = ZMQCapture.DEPTH_PROFILE

        # 2. Apply Defaults (User explicit values take precedence)
        if self.port is None:
            self.port = defaults.get("port", 5556)
            
        if self.width is None:
            self.width = defaults.get("width")
            
        if self.height is None:
            self.height = defaults.get("height")
            
        if self.fps is None:
            self.fps = defaults.get("fps", 30)
            
        if self.channels is None:
            self.channels = defaults.get("channels", 3)
            
        if self.dtype_str is None:
            self.dtype_str = defaults.get("dtype_str", "uint8")

        # 3. Validation
        if self.width is None or self.height is None:
            raise ValueError(
                "NetworkCameraConfig: Resolution (width/height) must be set via 'mode' or explicitly."
            )
        
        if self.channels not in [1, 3]:
            raise ValueError(f"channels must be 1 or 3, got {self.channels}")

        try:
            np.dtype(self.dtype_str)
        except TypeError:
            raise ValueError(f"Invalid dtype_str: {self.dtype_str}")

    @property
    def numpy_dtype(self):
        return np.dtype(self.dtype_str)