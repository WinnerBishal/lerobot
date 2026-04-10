from dataclasses import dataclass
from lerobot.cameras.configs import CameraConfig, ColorMode

@CameraConfig.register_subclass("digit")
@dataclass
class DigitCameraConfig(CameraConfig):
    """Configuration class for Meta AI DIGIT tactile sensors.
    
    Attributes:
        serial_number: The unique serial number of the DIGIT sensor. Available: D21358, D21357
        fps: Requested frames per second. 
        resolution: "VGA" (640x480) or "QVGA" (320x240).
        intensity: Internal RGB LED intensity (0-15).
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
    """
    serial_number: str = "D21357"
    fps: int = 30
    width: int = 240
    height: int = 320
    resolution: str = "QVGA"
    intensity: int = 15
    color_mode: ColorMode = ColorMode.RGB