import time
import numpy as np
import torch
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_xbox import XboxTeleoperatorConfig
from .xbox_utilities import RobotJoystick

class XboxTeleoperator(Teleoperator):
    config_class = XboxTeleoperatorConfig
    name = "xbox"

    def __init__(self, config: XboxTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        
        # FIX: Pass the config values to the joystick driver!
        # Otherwise it defaults to deadzone=0.9 and max_vel=1.5
        self.controller = RobotJoystick(
            joystick_id=config.joystick_id,
            max_lin_vel=config.max_lin_vel,
            max_ang_vel=config.max_ang_vel,
            grip_speed=config.grip_speed,
            deadzone=config.deadzone
        )
        
        self._connected = False
        self.last_time = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def action_features(self) -> dict:
        features = {}
        # Define 7 Atomic Features
        for name in ["vx", "vy", "vz", "wx", "wy", "wz", "gripper_vel"]:
            features[name] = float
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected: return
        print(f"Connecting to Xbox Controller (ID: {self.config.joystick_id})...")
        self._connected = True
        self.last_time = time.perf_counter()
        print("Xbox Controller Connected.")

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError("Xbox controller is not connected.")

        current_time = time.perf_counter()
        dt = current_time - self.last_time
        self.last_time = current_time
        if dt > 0.2: dt = 0.05

        # get_command returns [dx, dy, dz, droll, dpitch, dyaw, gripper]
        raw_action = self.controller.get_command(dt)
        
        # Return floats
        names = ["vx", "vy", "vz", "wx", "wy", "wz", "gripper_vel"]
        return {name: float(val) for name, val in zip(names, raw_action)}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def calibrate(self) -> None: pass
    def configure(self) -> None: pass

    def disconnect(self) -> None:
        import pygame
        pygame.quit()
        self._connected = False