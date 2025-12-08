from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path

from ..config import RobotConfig

@dataclass
class KinovaFollowerConfig(RobotConfig):

    # standarad field required by LeRobot
    type: str = "kinova_follower"

    # Custom Driver prameters for Kinova Gen3 API
    ip: str = "192.168.1.10"
    username: str = "admin"
    password: str = "admin"

    # Features Definitions
    features: Dict[str, Dict] = field(
        default_factory=lambda:{
            "observation.state": {
                "dtype": "float32",
                # 7 joints + gripper
                "shape": [8],
                "names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper_pos"],
            },
            # Since we are using xbox controller, we define actions as velocities on cartesian space
            "action": {
                "dtype": "float32",
                # twist (6) + gripper velocity
                "shape": [7],
                "names": ["vx", "vy", "vz", "wx", "wy", "wz", "gripper_vel"],
            }
        }
    )
    calibration_dir: Path = Path(".cache/calibration/kinova_follower")