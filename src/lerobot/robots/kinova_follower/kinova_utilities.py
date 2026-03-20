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

    def __init__(self, ipAddress, port=TCP_PORT, credentials=("","")):
        self.ipAddress = ipAddress
        self.port = port
        self.credentials = credentials
        self.sessionManager = None
        self.transport = TCPTransport() if port == DeviceConnection.TCP_PORT else UDPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    def __enter__(self):
        self.transport.connect(self.ipAddress, self.port)
        if (self.credentials[0] != ""):
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
    def __init__(self):
        self.n_joints = 7
        self.currentJointAngles = np.zeros(self.n_joints)
        self.currentGripperPosition = 0.0
        self.isConnected = False
        self.last_gripper_val = -1.0
        
        # --- WATCHDOG VARIABLES ---
        self.watchdog_running = False
        self.watchdog_thread = None
        self.last_command_time = time.time()
        self.TIMEOUT_SEC = 0.3  # Stop if no command for 300ms
    
    def _watchdog_loop(self):
        """
        Background thread that stops the robot if the main script hangs
        or is busy (e.g., encoding video).
        """
        while self.watchdog_running:
            # Check time since last command
            time_diff = time.time() - self.last_command_time
            
            if self.isConnected and time_diff > self.TIMEOUT_SEC:
                # We haven't received a command in a while -> EMERGENCY STOP
                # (We check if we already stopped to avoid spamming logs)
                if abs(self.last_gripper_val) > 0.01 or time_diff < (self.TIMEOUT_SEC + 0.2):
                    # print(f"Watchdog triggered! (No command for {time_diff:.3f}s)")
                    self.stop_all_movement()
                    # Reset timer slightly to prevent 100% CPU spam, 
                    # but keep it expired so we stop again next loop if needed.
                    # We don't fully reset last_command_time because we want to stay in "stop mode".
            
            time.sleep(0.1)

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

            self.isConnected = True
            print(f"\n Connected to Robot at {ip} Successfully \n")
            
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            self.base.SetServoingMode(base_servo_mode)

            self._update_feedback()

            # --- START WATCHDOG ---
            self.last_command_time = time.time()
            self.watchdog_running = True
            self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
            self.watchdog_thread.start()
            print("Safety Watchdog Started.")

        except Exception as e:
            self.isConnected = False
            print(f"ERROR: {e}")
            raise e

    def _update_feedback(self):
        try:
            feedback = self.baseCyclic.RefreshFeedback()
            # Convert Degrees -> Radians for LeRobot
            self.currentJointAngles = np.array([np.deg2rad(a.position) for a in feedback.actuators])
            self.currentGripperPosition = feedback.interconnect.gripper_feedback.motor[0].position
            return feedback
        except Exception:
            return None

    def get_current_state(self):
        self._update_feedback()
        return list(self.currentJointAngles) + [self.currentGripperPosition]
    
    def move_gripper_velocity(self, speed):
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = speed
        self.base.SendGripperCommand(gripper_command)

    def stop_all_movement(self):
        """Sends zero-velocity commands to everything."""
        if not self.isConnected: return
        try:
            # Stop Joints
            joint_speeds = Base_pb2.JointSpeeds()
            for i in range(self.n_joints):
                js = joint_speeds.joint_speeds.add()
                js.joint_identifier = i
                js.value = 0.0
            self.base.SendJointSpeedsCommand(joint_speeds)

            # Stop Twist
            twist = Base_pb2.TwistCommand()
            twist.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            self.base.SendTwistCommand(twist)

            # Stop Gripper
            gripper_cmd = Base_pb2.GripperCommand()
            gripper_cmd.mode = Base_pb2.GRIPPER_SPEED
            finger = gripper_cmd.gripper.finger.add()
            finger.finger_identifier = 1
            finger.value = 0.0
            self.base.SendGripperCommand(gripper_cmd)
            
            # Reset internal states
            self.last_gripper_val = 0.0
            
        except Exception as e:
            pass # Suppress errors during shutdown

    def act_twist(self, action, dt=0.05):
        """Mode 1: Cartesian Velocity Control"""
        if not self.isConnected: return False
        
        # Reset Watchdog Timer
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

        self._update_feedback()
        return True

    def act_joints(self, action, dt=0.05):
        """Mode 2: Joint Position Control"""
        if not self.isConnected: return False

        # Reset Watchdog Timer
        self.last_command_time = time.time()

        KP = 6.0 
        MAX_VEL_DEG = 5.0 

        target_joints_rad = np.array(action[:7])
        current_joints_rad = self.currentJointAngles 
        
        # 1. Error with Wrap-Around Fix
        error = target_joints_rad - current_joints_rad
        error = (error + np.pi) % (2 * np.pi) - np.pi
        
        # 2. Velocity
        vel_rad = error * KP
        
        joint_speeds = Base_pb2.JointSpeeds()
        for i, vel in enumerate(vel_rad):
            js = joint_speeds.joint_speeds.add()
            js.joint_identifier = i
            vel_deg = np.rad2deg(vel)
            vel_deg = max(min(vel_deg, MAX_VEL_DEG), -MAX_VEL_DEG)
            js.value = vel_deg
            # js.duration = 0 # Removed for your API version

        self.base.SendJointSpeedsCommand(joint_speeds)

        # Gripper
        target_grip = action[7] 
        current_grip = self.currentGripperPosition 
        if target_grip <= 1.0 and current_grip > 1.0: target_grip *= 100.0
        grip_err = target_grip - current_grip
        grip_vel = grip_err * (KP * 2.0)
        grip_vel = max(min(grip_vel, 0.5), -0.5)

        if abs(grip_vel) > 0.05 or abs(self.last_gripper_val) > 0.05:
            self.move_gripper_velocity(grip_vel)
            self.last_gripper_val = grip_vel

        self._update_feedback()
        return True

    def disconnect_from_robot(self):
        # Stop Watchdog
        self.watchdog_running = False
        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=1.0)

        # Final Stop
        self.stop_all_movement()
        
        if self.connection:
            self.connection.__exit__(None, None, None)
        self.isConnected = False
        print("Robot disconnected and stopped.")