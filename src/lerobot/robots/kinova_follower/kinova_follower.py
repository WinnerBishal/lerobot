import torch
import numpy as np
from functools import cached_property
from typing import Any
import copy

from lerobot.robots.robot import Robot
from lerobot.cameras.utils import make_cameras_from_configs
from .config_kinova_follower import KinovaFollowerConfig
from .kinova_utilities import ExecuteRobotAction

class KinovaFollower(Robot):
    config_class = KinovaFollowerConfig
    name = "kinova_follower"

    def __init__(self, config: KinovaFollowerConfig):
        super().__init__(config)
        self.config = config
        
        self.arm = ExecuteRobotAction()
        
        # 1. Initialize Cameras
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # 2. Sync Config Features
        # Ensure config matches the Tuple format required for video recording.
        # This allows LeRobot to detect these as 'video' features.
        if "observation.images" not in self.config.features:
             for name, camera in self.cameras.items():
                self.config.features[f"observation.images.{name}"] = (camera.height, camera.width, 3)
        
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True 

    @cached_property
    def observation_features(self) -> dict:
        """
        Defines the structure of the observation dictionary.
        """
        # 1. Flatten state features
        state_names = self.config.features["observation.state"]["names"]
        features = {name: float for name in state_names}
        
        # 2. Define images as TUPLES (H, W, C) for video recording
        # This format signals to LeRobot that these are video streams.
        for name, camera in self.cameras.items():
            features[f"observation.images.{name}"] = (camera.height, camera.width, 3)
            
        return features

    @cached_property
    def action_features(self) -> dict:
        """
        Defines the structure of the action dictionary.
        We return a single vector shape to match the Xbox controller output.
        """
        return {
            "action": {
                "dtype": "float32",
                "shape": (7,),
            }
        }

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            return

        print(f"Connecting to Kinova at {self.config.ip}...")
        self.arm.connect_to_robot(
            ip=self.config.ip,
            username=self.config.username,
            password=self.config.password
        )
        
        for name, camera in self.cameras.items():
            print(f"Connecting to camera: {name}")
            camera.connect()
        
        self._connected = True
        
        if calibrate:
            self.calibrate()
            
        print("Kinova System Connected.")

    def capture_images(self) -> dict[str, Any]:
        """
        Reads frames from all connected cameras.
        """
        images = {}
        for name, camera in self.cameras.items():
            img = camera.async_read()
            
            # Safe conversion to uint8 to prevent dark videos
            # This handles cases where drivers might return float [0,1]
            if img is not None:
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                
                if isinstance(img, np.ndarray):
                    if np.issubdtype(img.dtype, np.floating):
                        # Scale float [0, 1] to [0, 255] if needed
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    elif img.dtype != np.uint8:
                        img = img.astype(np.uint8)
            
            # FIX: Return ONLY the short key (e.g., 'wrist_image').
            # The dataset recorder automatically maps 'wrist_image' -> 'observation.images.wrist_image'.
            # Rerun will now see only this one active window per camera.
            images[name] = img
            
        return images

    def calibrate(self):
        pass

    def configure(self):
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError("Kinova is not connected.")

        # 1. Get Robot State
        current_joints = self.arm.currentJointAngles
        current_gripper = self.arm.currentGripperPosition
        all_values = list(current_joints) + [current_gripper]
        
        # 2. Map values to flattened keys
        state_names = self.config.features["observation.state"]["names"]
        observation = {name: val for name, val in zip(state_names, all_values)}

        # 3. Add images
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