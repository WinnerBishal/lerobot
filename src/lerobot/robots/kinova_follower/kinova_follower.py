import torch
import numpy as np
from functools import cached_property
from typing import Any

from lerobot.robots.robot import Robot
from lerobot.cameras.utils import make_cameras_from_configs  # <--- IMPORT THIS
from .config_kinova_follower import KinovaFollowerConfig
from .kinova_utilities import ExecuteRobotAction

class KinovaFollower(Robot):
    config_class = KinovaFollowerConfig
    name = "kinova_follower"

    def __init__(self, config: KinovaFollowerConfig):
        super().__init__(config)
        self.config = config
        
        self.arm = ExecuteRobotAction()
        
        # 1. Initialize Cameras explicitly
        self.cameras = make_cameras_from_configs(config.cameras)
        
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True 

    @cached_property
    def observation_features(self) -> dict:
        features = {"observation.state": self.config.features["observation.state"]}
        
        # Add camera features to the dictionary
        for name, camera in self.cameras.items():
            features[f"observation.images.{name}"] = {
                "dtype": "uint8",
                "shape": (camera.height, camera.width, 3),
                "names": ["height", "width", "channel"],
            }
        return features

    @cached_property
    def action_features(self) -> dict:
        return {"action": self.config.features["action"]}

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            return

        print(f"Connecting to Kinova at {self.config.ip}...")
        self.arm.connect_to_robot(
            ip=self.config.ip,
            username=self.config.username,
            password=self.config.password
        )
        
        # 2. Connect all cameras
        for name, camera in self.cameras.items():
            print(f"Connecting to camera: {name}")
            camera.connect()
        
        self._connected = True
        
        if calibrate:
            self.calibrate()
            
        print("Kinova System Connected.")

    def capture_images(self) -> dict[str, Any]:
        """Reads frames from all connected cameras."""
        images = {}
        for name, camera in self.cameras.items():
            # Use async_read for better performance in the loop
            image = camera.async_read()
            images[f"observation.images.{name}"] = image
        return images

    def calibrate(self):
        pass

    def configure(self):
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError("Kinova is not connected.")

        current_joints = self.arm.currentJointAngles
        current_gripper = self.arm.currentGripperPosition
        all_values = list(current_joints) + [current_gripper]
        
        observation = {
            "observation.state": torch.tensor(all_values, dtype=torch.float32)
        }

        # 3. Capture Images (Now self.cameras exists)
        if self.cameras:
            images = self.capture_images() 
            observation.update(images)

        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError("Kinova is not connected.")
        
        action_obj = action
        if isinstance(action, dict):
            if "action" in action:
                action_obj = action["action"]
            else:
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
            action_np = np.asarray(action_obj)

        self.arm.act_twist(action_np)
        return {"action": action_obj}

    def disconnect(self):
        print("Disconnecting Kinova System...")
        
        # 4. Disconnect cameras
        for name, camera in self.cameras.items():
            try:
                camera.disconnect()
            except Exception as e:
                print(f"Error disconnecting camera {name}: {e}")

        try:
            if hasattr(self.arm, "disconnect_from_robot"):
                self.arm.disconnect_from_robot()
        except Exception:
            pass
        self._connected = False