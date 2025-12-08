import torch
import numpy as np
from functools import cached_property
from typing import Any

# Import the Base Class
from lerobot.robots.robot import Robot
from .config_kinova_follower import KinovaFollowerConfig
from .kinova_utilities import ExecuteRobotAction

class KinovaFollower(Robot):
    # 1. Define Config Class and Name (Required by Base Class)
    config_class = KinovaFollowerConfig
    name = "kinova_follower"

    def __init__(self, config: KinovaFollowerConfig):
        # Initialize Parent
        super().__init__(config)
        self.config = config
        
        # Instantiate Driver
        self.arm = ExecuteRobotAction()
        self._connected = False

    # 2. Implement Properties
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True 

    @cached_property
    def observation_features(self) -> dict:
        return {"observation.state": self.config.features["observation.state"]}

    @cached_property
    def action_features(self) -> dict:
        return {"action": self.config.features["action"]}

    # 3. Implement Methods
    def connect(self, calibrate: bool = True):
        print(f"Connecting to Kinova at {self.config.ip}...")
        self.arm.connect_to_robot()
        self._connected = True
        
        if calibrate:
            self.calibrate()
            
        print("Kinova Connected.")

    def calibrate(self):
        pass

    def configure(self):
        pass

    def get_observation(self) -> dict[str, Any]:
        """
        Must return a dictionary matching observation_features.
        """
        if not self.is_connected:
            raise ConnectionError("Kinova is not connected.")

        # Read from utility class
        current_joints = self.arm.currentJointAngles # List[float]
        current_gripper = self.arm.currentGripperPosition # float
        
        obs_dict = {}
        feature_names = self.config.features["observation.state"]["names"]
        
        # Combine data
        all_values = list(current_joints) + [current_gripper]

        
        state_tensor = torch.tensor(all_values, dtype=torch.float32)
        return {"observation.state": state_tensor}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Input is a dictionary matching action_features.
        """
        if not self.is_connected:
            raise ConnectionError("Kinova is not connected.")
        
        action_obj = action
        if isinstance(action, dict):
            if "action" in action:
                action_obj = action["action"]
            else:
                # fallback to first value
                action_obj = next(iter(action.values()))

        try:
            import torch

            if hasattr(torch, "is_tensor") and torch.is_tensor(action_obj):
                action_np = action_obj.detach().cpu().numpy()
            elif isinstance(action_obj, (list, tuple)):
                action_np = np.asarray(action_obj, dtype=float)
            elif isinstance(action_obj, np.ndarray):
                action_np = action_obj
            else:
                action_np = np.asarray(action_obj)
        except Exception:
            if isinstance(action_obj, (list, tuple)):
                action_np = np.asarray(action_obj, dtype=float)
            elif isinstance(action_obj, np.ndarray):
                action_np = action_obj
            else:
                action_np = np.asarray(action_obj)

        # Send to Driver
        self.arm.act_twist(action_np)

        # Return what we actually sent
        return {"action": action_obj}

    def disconnect(self):
        print("Disconnecting Kinova...")
        try:
            if hasattr(self.arm, "disconnect_from_robot"):
                self.arm.disconnect_from_robot()
        except Exception:
            pass
        self._connected = False