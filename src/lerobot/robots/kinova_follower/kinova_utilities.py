#!/usr/bin/env python3

import argparse
import threading
import time
import numpy as np

from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2, Base_pb2, BaseCyclic_pb2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient


class DeviceConnection:
    TCP_PORT = 10000
    UDP_PORT = 10001

    @staticmethod
    def createTcpConnection(args):
        return DeviceConnection(args.ip, port=DeviceConnection.TCP_PORT, credentials=(args.username, args.password))

    def __init__(self, ipAddress, port=TCP_PORT, credentials=("", "")):
        self.ipAddress = ipAddress
        self.port = port
        self.credentials = credentials
        self.sessionManager = None
        self.transport = TCPTransport() if port == DeviceConnection.TCP_PORT else UDPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    def __enter__(self):
        self.transport.connect(self.ipAddress, self.port)
        if self.credentials[0] != "":
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000
            session_info.connection_inactivity_timeout = 2000
            self.sessionManager = SessionManager(self.router)
            self.sessionManager.CreateSession(session_info)
        return self.router

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sessionManager is not None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self.sessionManager.CloseSession(router_options)
        self.transport.disconnect()


class ExecuteRobotAction:
    # Feedback polling rate (Hz) — decoupled from the command loop.
    FEEDBACK_HZ = 100

    def __init__(self):
        self.n_joints = 7
        self._feedback_lock = threading.Lock()
        self._currentJointAngles = np.zeros(self.n_joints)
        self._currentGripperPosition = 0.0
        self.isConnected = False
        self.last_gripper_val = -1.0
        self.last_gripper_pos = -1.0

        # --- WATCHDOG ---
        self.watchdog_running = False
        self.watchdog_thread = None
        self.last_command_time = time.time()
        self.TIMEOUT_SEC = 0.3

        # --- BACKGROUND FEEDBACK THREAD ---
        self._feedback_running = False
        self._feedback_thread = None

    # ------------------------------------------------------------------ #
    # Public properties — read joint state under lock                     #
    # ------------------------------------------------------------------ #

    @property
    def currentJointAngles(self) -> np.ndarray:
        with self._feedback_lock:
            return self._currentJointAngles.copy()

    @property
    def currentGripperPosition(self) -> float:
        with self._feedback_lock:
            return self._currentGripperPosition

    # ------------------------------------------------------------------ #
    # Background threads                                                   #
    # ------------------------------------------------------------------ #

    def _feedback_loop(self):
        """Poll RefreshFeedback() at FEEDBACK_HZ, decoupled from command sends."""
        dt = 1.0 / self.FEEDBACK_HZ
        while self._feedback_running:
            t0 = time.time()
            try:
                feedback = self.baseCyclic.RefreshFeedback()
                joints = np.array([np.deg2rad(a.position) for a in feedback.actuators])
                gripper = feedback.interconnect.gripper_feedback.motor[0].position
                with self._feedback_lock:
                    self._currentJointAngles = joints
                    self._currentGripperPosition = gripper
            except Exception:
                pass
            elapsed = time.time() - t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _watchdog_loop(self):
        while self.watchdog_running:
            time_diff = time.time() - self.last_command_time
            if self.isConnected and time_diff > self.TIMEOUT_SEC:
                if abs(self.last_gripper_val) > 0.01 or time_diff < (self.TIMEOUT_SEC + 0.2):
                    self.stop_all_movement()
            time.sleep(0.1)

    # ------------------------------------------------------------------ #
    # Connection                                                           #
    # ------------------------------------------------------------------ #

    def connect_to_robot(self, ip="192.168.1.10", username="admin", password="admin"):
        try:
            class ConnectionArgs:
                def __init__(self, ip, u, p):
                    self.ip = ip; self.username = u; self.password = p

            connectionArgs = ConnectionArgs(ip, username, password)
            self.connection = DeviceConnection.createTcpConnection(connectionArgs)
            self.router = self.connection.__enter__()
            self.base = BaseClient(self.router)
            self.baseCyclic = BaseCyclicClient(self.router)

            # TCP connection — for commands (BaseClient)
            self.connection = DeviceConnection.createTcpConnection(connectionArgs)
            self.router = self.connection.__enter__()
            self.base = BaseClient(self.router)

            self.isConnected = True
            print(f"\n Connected to Robot at {ip} Successfully \n")

            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            self.base.SetServoingMode(base_servo_mode)

            # Initial feedback read to populate state
            self._update_feedback_once()

            # Start background feedback thread
            self._feedback_running = True
            self._feedback_thread = threading.Thread(target=self._feedback_loop, daemon=True, name="KinovaFeedback")
            self._feedback_thread.start()
            print("Background feedback thread started (100 Hz).")

            # Start watchdog
            self.last_command_time = time.time()
            self.watchdog_running = True
            self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="KinovaWatchdog")
            self.watchdog_thread.start()
            print("Safety Watchdog Started.")

        except Exception as e:
            self.isConnected = False
            print(f"ERROR: {e}")
            raise e

    def _update_feedback_once(self):
        """One-shot feedback read used only during connect."""
        try:
            feedback = self.baseCyclic.RefreshFeedback()
            with self._feedback_lock:
                self._currentJointAngles = np.array([np.deg2rad(a.position) for a in feedback.actuators])
                self._currentGripperPosition = feedback.interconnect.gripper_feedback.motor[0].position
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Commands — fire-and-return, no blocking feedback read               #
    # ------------------------------------------------------------------ #

    def move_gripper_velocity(self, speed):
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = speed
        self.base.SendGripperCommand(gripper_command)
    
    def move_gripper_position(self, position: float):
        """Send a direct position command (0.0 = fully open, 1.0 = fully closed)."""
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = float(max(0.0, min(0.8, position)))
        self.base.SendGripperCommand(gripper_command)

    def stop_all_movement(self):
        if not self.isConnected: return
        try:
            joint_speeds = Base_pb2.JointSpeeds()
            for i in range(self.n_joints):
                js = joint_speeds.joint_speeds.add()
                js.joint_identifier = i
                js.value = 0.0
            self.base.SendJointSpeedsCommand(joint_speeds)

            twist = Base_pb2.TwistCommand()
            twist.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            self.base.SendTwistCommand(twist)

            gripper_cmd = Base_pb2.GripperCommand()
            gripper_cmd.mode = Base_pb2.GRIPPER_SPEED
            finger = gripper_cmd.gripper.finger.add()
            finger.finger_identifier = 1
            finger.value = 0.0
            self.base.SendGripperCommand(gripper_cmd)

            self.last_gripper_val = 0.0
        except Exception:
            pass

    def act_twist(self, action, dt=0.05):
        """Cartesian velocity control. Returns immediately — no feedback read."""
        if not self.isConnected: return False

        self.last_command_time = time.time()

        command = Base_pb2.TwistCommand()
        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        twist = command.twist

        twist.linear_x = action[0] / dt
        twist.linear_y = action[1] / dt
        twist.linear_z = action[2] / dt
        twist.angular_x = np.rad2deg(action[3] / dt)
        twist.angular_y = np.rad2deg(action[4] / dt)
        twist.angular_z = np.rad2deg(action[5] / dt)

        self.base.SendTwistCommand(command)

        current_val = action[6]
        cmd_vel = current_val / dt
        MAX_GRIP_SPEED = 0.4
        cmd_vel = max(min(cmd_vel, MAX_GRIP_SPEED), -MAX_GRIP_SPEED)

        if abs(cmd_vel) > 0.05 or abs(self.last_gripper_val) > 0.05:
            self.move_gripper_velocity(cmd_vel)
            self.last_gripper_val = cmd_vel

        return True

    def act_joints(self, action, dt=0.05):
        """Joint speed control. Returns immediately — no blocking feedback read."""
        if not self.isConnected: return False

        self.last_command_time = time.time()

        KP = 6.0
        MAX_VEL_DEG = 20.0

        target_joints_rad = np.array(action[:7])
        current_joints_rad = self.currentJointAngles  # reads from background thread via property

        error = target_joints_rad - current_joints_rad
        error = (error + np.pi) % (2 * np.pi) - np.pi

        vel_rad = error * KP

        joint_speeds = Base_pb2.JointSpeeds()
        for i, vel in enumerate(vel_rad):
            js = joint_speeds.joint_speeds.add()
            js.joint_identifier = i
            vel_deg = np.rad2deg(vel)
            vel_deg = max(min(vel_deg, MAX_VEL_DEG), -MAX_VEL_DEG)
            js.value = vel_deg

        self.base.SendJointSpeedsCommand(joint_speeds)

        # target_grip = action[7]
        # current_grip = self.currentGripperPosition
        # if target_grip <= 1.0 and current_grip > 1.0: target_grip *= 100.0
        # grip_err = target_grip - current_grip
        # grip_vel = grip_err * (KP * 2.0)
        # grip_vel = max(min(grip_vel, 0.5), -0.5)

        # if abs(grip_vel) > 0.05 or abs(self.last_gripper_val) > 0.05:
        #     self.move_gripper_velocity(grip_vel)
        #     self.last_gripper_val = grip_vel
        kortex_pos = 1.0 - float(action[7])
        kortex_pos = max(0.0, min(1.0, kortex_pos))
        # Deadband: only replan when target changed by >1% (prevents step motion)
        if abs(kortex_pos - self.last_gripper_pos) > 0.01:
            self.move_gripper_position(kortex_pos)
            self.last_gripper_pos = kortex_pos
            self.last_gripper_val = 1.0 

        return True

    def get_current_state(self):
        return list(self.currentJointAngles) + [self.currentGripperPosition]

    # ------------------------------------------------------------------ #
    # Disconnect                                                           #
    # ------------------------------------------------------------------ #

    def disconnect_from_robot(self):
        # Stop feedback thread
        self._feedback_running = False
        if self._feedback_thread:
            self._feedback_thread.join(timeout=1.0)

        # Stop watchdog
        self.watchdog_running = False
        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=1.0)

        self.stop_all_movement()

        if self.connection:
            self.connection.__exit__(None, None, None)
            
        self.isConnected = False
        print("Robot disconnected and stopped.")
