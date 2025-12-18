import time
import numpy as np
import torch
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_xbox import XboxTeleoperatorConfig
from .xbox_utilities import RobotJoystick  #  existing driver script

class XboxTeleoperator(Teleoperator):
    config_class = XboxTeleoperatorConfig
    name = "xbox"

    def __init__(self, config: XboxTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.controller = None
        self._connected = False
        self.last_time = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        # Xbox controllers are self-centering, so we consider them always calibrated
        return True

    @property
    def action_features(self) -> dict:
        """
        Returns the feature definition for the dataset.
        Matches the 7-DOF structure (6D pose + gripper) used by KinovaFollower.
        """
        return {
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": ["vx", "vy", "vz", "wx", "wy", "wz", "gripper_pos"],
            }
        }

    @property
    def feedback_features(self) -> dict:
        # No haptic feedback implemented for this controller
        return {}

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return

        print(f"Connecting to Xbox Controller (ID: {self.config.joystick_id})...")
        try:
            self.controller = RobotJoystick(
                joystick_id=self.config.joystick_id,
                max_lin_vel=self.config.max_lin_vel,
                max_ang_vel=self.config.max_ang_vel,
                grip_speed=self.config.grip_speed,
                deadzone=self.config.deadzone
            )
            self._connected = True
            self.last_time = time.perf_counter()
            print("Xbox Controller Connected.")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Xbox controller: {e}")

    def get_action(self) -> dict[str, Any]:
        """
        Reads controller state, calculates dt, and returns displacement command.
        """
        if not self.is_connected:
            raise ConnectionError("Xbox controller is not connected.")

        # Calculate time delta for smooth velocity integration
        current_time = time.perf_counter()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Safety: clamp dt to prevent jumps if the loop hangs
        if dt > 0.2: 
            dt = 0.05

        # Get command from driver [dx, dy, dz, droll, dpitch, dyaw, gripper_pos]
        # Your RobotJoystick calculates displacements based on the speeds provided
        raw_action = self.controller.get_command(dt)

        # Convert to Torch Tensor as expected by LeRobot dataset tools
        action_tensor = torch.from_numpy(np.array(raw_action, dtype=np.float32))
        
        return {"action": action_tensor}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # Optional: Implement rumble here if needed
        pass

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if self.controller:
            try:
                # Assuming standard pygame cleanup
                import pygame
                pygame.quit()
            except ImportError:
                pass
        self._connected = False