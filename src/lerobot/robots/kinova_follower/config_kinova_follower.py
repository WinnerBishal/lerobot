from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path

from lerobot.robots.config import RobotConfig
from lerobot.cameras.camera import CameraConfig

@RobotConfig.register_subclass("kinova_follower")
@dataclass
class KinovaFollowerConfig(RobotConfig):
    # Custom Driver parameters
    ip: str = "192.168.1.10"
    username: str = "admin"
    password: str = "admin"

    cameras: Dict[str, CameraConfig] = field(default_factory=dict)
    
    # REMOVED: features dict. (Defined dynamically in the class to support aggregation)
    
    calibration_dir: Path = Path(".cache/calibration/kinova_follower")