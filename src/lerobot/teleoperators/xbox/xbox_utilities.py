"""
Joystick Controller Module (Standard Safe Architecture)
-----------------------------------------------------
Implements Mode Switching and Deadman Safety protocols.

Safety:
-------
- LB (Button 4) : DEADMAN SWITCH. Must hold to move.
- Triggers      : Analog Gripper Control (Decoupled from arm motion).

Modes (Toggled by Button 0/A):
------------------------------
MODE 1: TRANSLATION
- Left Stick    : X (Surge) / Y (Sway)
- Right Stick Y : Z (Heave)

MODE 2: ROTATION
- Left Stick    : Pitch / Yaw
- Right Stick Y : Roll

Logic Notes:
------------
- Gripper output is a POSITION (0.0 to 1.0) integrated from Analog Triggers.
- Returns: [dx, dy, dz, droll, dpitch, dyaw, gripper_pos]
"""

import pygame
import numpy as np

class RobotJoystick:
    def __init__(self, joystick_id=0, max_lin_vel=1.50, max_ang_vel=3.0, grip_speed=1.0, 
                 deadzone=0.9, smooth_factor=0.8, initial_gripper_pos=1.0):
        
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.grip_speed = grip_speed # Max speed at full trigger pull
        self.deadzone = deadzone
        self.alpha = smooth_factor
        
        # --- INPUT MAPPING ---
        self.AXIS_LX, self.AXIS_LY = 0, 1
        self.AXIS_RX, self.AXIS_RY = 2, 3
        self.AXIS_RT, self.AXIS_LT = 4, 5
        
        # Standard Gamepad Buttons (Logitech/Xbox style)
        self.BTN_MODE = 0     # Button A (Cross) - Toggle Mode
        self.BTN_DEADMAN = 6  # LB (L1) - Hold to Enable
        
        # Internal State
        self.prev_axes = np.zeros(6)
        self.gripper_pos = max(0.0, min(1.0, initial_gripper_pos))
        self.dgrip_pos = 0.0
        
        # Control Mode: 0 = Translation, 1 = Rotation
        self.control_mode = 0 
        
        # Debounce for Mode Switch
        self.prev_btn_mode = False

        # Init
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= joystick_id:
            raise ConnectionError(f"No joystick found at ID {joystick_id}")
        self.joy = pygame.joystick.Joystick(joystick_id)
        self.joy.init()
        
        print(f"Joystick Initialized: {self.joy.get_name()}")
        print("HOLD LB (Button 4) TO MOVE.")
        print("PRESS A (Button 0) TO SWITCH MODES.")

    def _apply_deadzone(self, value):
        return 0.0 if abs(value) < self.deadzone else value

    def _normalize_trigger(self, raw_val):
        # Maps [-1, 1] to [0, 1]
        return (1.0 - raw_val) / 2.0

    def _smooth_input(self, current_axes):
        smoothed = (self.alpha * current_axes) + ((1.0 - self.alpha) * self.prev_axes)
        self.prev_axes = smoothed
        return smoothed

    def get_command(self, dt):
        pygame.event.pump()

        # --- 1. SAFETY CHECK (DEADMAN) ---
        deadman_active = self.joy.get_button(self.BTN_DEADMAN)
        
        # If Deadman is NOT held, we can still process non-motion logic 
        # (like state internal updates), but OUTPUT velocity must be zero.
        # However, for Mode Switching, we allow it even if deadman is off (usability).
        
        # --- 2. MODE SWITCHING ---
        curr_btn_mode = self.joy.get_button(self.BTN_MODE)
        if curr_btn_mode and not self.prev_btn_mode:
            self.control_mode = 1 - self.control_mode # Toggle 0 <-> 1
            mode_name = "ROTATION" if self.control_mode == 1 else "TRANSLATION"
            print(f"*** SWITCHED MODE: {mode_name} ***")
        self.prev_btn_mode = curr_btn_mode

        # --- 3. READ AXES ---
        raw_axes = np.array([
            self.joy.get_axis(self.AXIS_LX), self.joy.get_axis(self.AXIS_LY),
            self.joy.get_axis(self.AXIS_RX), self.joy.get_axis(self.AXIS_RY),
            self.joy.get_axis(self.AXIS_RT), self.joy.get_axis(self.AXIS_LT)
        ])
        s_axes = self._smooth_input(raw_axes)

        # Initialize Outputs
        dx, dy, dz = 0.0, 0.0, 0.0
        droll, dpitch, dyaw = 0.0, 0.0, 0.0

        # --- 4. APPLY MAPPING BASED ON MODE ---
        if deadman_active:
            # Shared Mapping: Left Stick is primary
            # Inverted Y on stick inputs for intuitive feel (Up = Forward/Pitch Down)
            val_lx = self._apply_deadzone(s_axes[self.AXIS_LX])
            val_ly = self._apply_deadzone(-s_axes[self.AXIS_LY]) 
            val_rx = self._apply_deadzone(s_axes[self.AXIS_RX])
            val_ry = self._apply_deadzone(-s_axes[self.AXIS_RY])

            if self.control_mode == 0: # TRANSLATION MODE
                # Left Stick -> X, Y
                dx = val_ly * self.max_lin_vel * dt
                dy = -val_lx * self.max_lin_vel * dt # Inverted X for correct "Left/Right" feel
                
                # Right Stick Y -> Z (Height)
                dz = val_ry * self.max_lin_vel * dt
                
            else: # ROTATION MODE
                # Left Stick -> Pitch, Yaw
                dpitch = val_ly * self.max_ang_vel * dt
                dyaw = -val_lx * self.max_ang_vel * dt
                
                # Right Stick Y -> Roll
                droll = -val_ry * self.max_ang_vel * dt # Roll is often mapped to X, but Y works if 1 DOF

            # --- 5. GRIPPER CONTROL (Always Active if Deadman Held) ---
            # Triggers are normalized [0.0 to 1.0]
            # RT opens, LT closes
            val_rt = self._normalize_trigger(s_axes[self.AXIS_RT]) # Open force
            val_lt = self._normalize_trigger(s_axes[self.AXIS_LT]) # Close force
            
            # Net Velocity = (Open - Close) * Speed
            grip_velocity = (val_rt - val_lt) * self.grip_speed
            # print(f"Gripper Velocity Command: {val_lt:.3f}, {val_rt:.3f}, {grip_velocity:.3f}")
            self.dgrip_pos = grip_velocity * dt
            # print(f"Gripper Position: {self.dgrip_pos:.3f}, dt: {dt:.3f}")
        else:
            # Deadman released: Everything stops.
            # We do NOT reset gripper_pos (it holds its grip).
            pass

        return [dx, dy, dz, droll, dpitch, dyaw, self.dgrip_pos]
    
    def close(self):
        pygame.quit()