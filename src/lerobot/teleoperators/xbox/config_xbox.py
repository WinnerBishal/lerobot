from dataclasses import dataclass
from pathlib import Path
from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("xbox")
@dataclass
class XboxTeleoperatorConfig(TeleoperatorConfig):
    joystick_id: int = 0
    max_lin_vel: float = 0.2 
    max_ang_vel: float = 0.5
    grip_speed: float = 1.0
    deadzone: float = 0.1
    calibration_dir: Path | None = None