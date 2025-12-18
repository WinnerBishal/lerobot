from dataclasses import dataclass
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("xbox")
@dataclass
class XboxTeleoperatorConfig(TeleoperatorConfig):
    # Joystick ID (0 for first controller, 1 for second, etc.)
    joystick_id: int = 0
    
    # Speed limits (matches the units expected by your robot driver)
    max_lin_vel: float = 0.2  # meters/sec
    max_ang_vel: float = 0.5  # radians/sec
    grip_speed: float = 1.0   # gripper units/sec
    
    # Deadzone to prevent drift
    deadzone: float = 0.1
    
    # Directory to store specific xbox calibrations if needed (optional)
    calibration_dir: Path | None = None