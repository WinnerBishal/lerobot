import torch
import numpy as np
from typing import Any

from lerobot.robots.robot import Robot
from lerobot.cameras.utils import make_cameras_from_configs
from .config_kinova_follower import KinovaFollowerConfig
from .kinova_utilities_v0 import ExecuteRobotAction

class KinovaFollower(Robot):
    config_class = KinovaFollowerConfig
    name = "kinova_follower"

    def __init__(self, config: KinovaFollowerConfig):
        self.cameras = make_cameras_from_configs(config.cameras)
        super().__init__(config)
        self.config = config
        self.arm = ExecuteRobotAction()
        self._connected = False

        self.last_action_np = None
        self.smoothing_alpha = 0.3

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True 

    @property
    def observation_features(self) -> dict:
        features = {}
        # 1. Define Atomic State Features (8 Total)
        # Explicit definition ensures names like 'j1', 'gripper_pos' are saved in info.json
        state_names = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper_pos"]
        for name in state_names:
            features[name] = float
            
        # 2. Define Cameras
        for name, camera in self.cameras.items():
            features[name] = (camera.height, camera.width, 3)
        return features

    # @property
    # def action_features(self) -> dict:
    #     features = {}
    #     # 1. Define Atomic Action Features (7 Total)
    #     action_names = ["vx", "vy", "vz", "wx", "wy", "wz", "gripper_vel"]
    #     for name in action_names:
    #         features[name] = float
    #     return features
    
    # Uncomment below to use policy trained on joint actions
    @property
    def action_features(self) -> dict:
        features = {}
        # 1. Define Atomic Action Features (7 Total)
        action_names = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper_val"]  
        for name in action_names:
            features[name] = float
        return features
    
    def connect(self, calibrate: bool = True):
        if self.is_connected: return
        print(f"Connecting to Kinova at {self.config.ip}...")
        self.arm.connect_to_robot(ip=self.config.ip, username=self.config.username, password=self.config.password)
        for name, camera in self.cameras.items():
            print(f"Connecting to camera: {name}")
            camera.connect()
        self._connected = True
        if calibrate: self.calibrate()
        print("Kinova System Connected.")

    def capture_images(self) -> dict[str, Any]:
        images = {}
        for name, camera in self.cameras.items():
            images[name] = camera.async_read()
        return images

    def calibrate(self): pass
    def configure(self): pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected: raise ConnectionError("Kinova is not connected.")
        
        # Fetch data
        current_joints = self.arm.currentJointAngles
        current_gripper = self.arm.currentGripperPosition
        
        # Combine into list (8 values)
        all_values = list(current_joints) + [current_gripper]
        
        # Map to Atomic Keys
        state_names = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper_pos"]
        observation = {name: val for name, val in zip(state_names, all_values)}
        
        if self.cameras:
            observation.update(self.capture_images())
        return observation

    # def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    #     if not self.is_connected: raise ConnectionError("Kinova is not connected.")
        
    #     # Extract values efficiently
    #     target_names = ["vx", "vy", "vz", "wx", "wy", "wz", "gripper_vel"]
        
    #     # Handle dictionary input
    #     if all(k in action for k in target_names):
    #         vals = [action[k] for k in target_names]
    #     elif "action" in action:
    #         # Handle vector input (from policy)
    #         vals = action["action"]
    #         if hasattr(vals, "tolist"): vals = vals.tolist()
    #     else:
    #         # Fallback
    #         vals = list(action.values())

    #     # OPTIMIZATION: Convert directly to Python list of floats.
    #     # This removes the PyTorch overhead that caused the lag.
    #     if hasattr(vals, "detach"): vals = vals.detach().cpu()
    #     vals_np = np.array(vals, dtype=np.float32).flatten()
    #     vals_list = vals_np.tolist()

    #     # Send to driver
    #     self.arm.act_twist(vals_list)
    #     return action
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected: raise ConnectionError("Kinova is not connected.")
        
        # Target names now match Joint Mode
        target_names = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper_pos"]
        
        if all(k in action for k in target_names):
            vals = [action[k] for k in target_names]
        elif "action" in action:
            vals = action["action"]
            if hasattr(vals, "tolist"): vals = vals.tolist()
        else:
            vals = list(action.values())

        if hasattr(vals, "detach"): vals = vals.detach().cpu()
        current_action_np = np.array(vals, dtype=np.float32).flatten()

        # 2. APPLY SMOOTHING (EMA)
        if self.last_action_np is None:
            smoothed_action = current_action_np
        else:
            # Formula: Smoothed = (Alpha * New) + ((1-Alpha) * Old)
            # Alpha 0.3 means we only accept 30% of the new command per frame
            smoothed_action = (self.smoothing_alpha * current_action_np) + \
                              ((1 - self.smoothing_alpha) * self.last_action_np)
        
        self.last_action_np = smoothed_action
        
        # 3. Send to Robot
        self.arm.act_joints(smoothed_action.tolist())
        
        return action

    def disconnect(self):
        for c in self.cameras.values(): c.disconnect()
        if hasattr(self.arm, "disconnect_from_robot"):
            self.arm.disconnect_from_robot()
        self._connected = False