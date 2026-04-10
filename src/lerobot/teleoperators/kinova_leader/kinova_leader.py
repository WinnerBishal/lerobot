"""
KinovaLeader — LeRobot Teleoperator wrapping a 7-DOF ST3215 serial servo arm.

Architecture
------------
  Main thread (get_action, 50 Hz recommended):
      Acquires serial_lock → reads all 7 motor angles → posts Motor 4 angle to
      SharedState → applies offsets / directions → returns joint radians dict.

  JointGuardian thread (200 Hz):
      Reads Motor 4 angle from SharedState (NO serial reads).
      Issues write-only serial packets to enforce:
        • Motor 3 — rigid torque hold at calibrated neutral.
        • Motor 4 — soft floor: torque ON when angle ≤ floor, OFF above floor+hyst.
      Acquires serial_lock before each write burst to avoid bus contention.
"""

import logging
import threading
import time
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_kinova_leader import KinovaLeaderConfig
from .controller_class import LeaderArmController

logger = logging.getLogger(__name__)


class _SharedState:
    """Lightweight, lock-protected container for the Motor 4 angle."""

    def __init__(self):
        self._lock = threading.Lock()
        self._m4_angle: float | None = None

    def set_m4_angle(self, angle: float | None) -> None:
        with self._lock:
            self._m4_angle = angle

    def get_m4_angle(self) -> float | None:
        with self._lock:
            return self._m4_angle


class _JointGuardian(threading.Thread):
    """
    Enforces leader arm joint constraints at ~200 Hz using write-only packets.

    Motor 3 — re-asserts torque + goal position every cycle.
    Motor 4 — soft-floor state machine:
               FREE       → FLOOR_HOLD  when angle ≤ floor
               FLOOR_HOLD → FREE        when angle >  floor + hysteresis
    """

    def __init__(
        self,
        leader: LeaderArmController,
        serial_lock: threading.Lock,
        shared: _SharedState,
        motor3_id: int,
        motor3_hold_deg: float,
        motor4_id: int,
        motor4_floor_deg: float,
        motor4_hysteresis: float,
        guardian_hz: int,
    ):
        super().__init__(daemon=True, name="KinovaLeaderGuardian")
        self.leader = leader
        self.lock = serial_lock
        self.shared = shared
        self.motor3_id = motor3_id
        self.motor3_hold_deg = motor3_hold_deg
        self.motor4_id = motor4_id
        self.motor4_floor_deg = motor4_floor_deg
        self.motor4_hysteresis = motor4_hysteresis
        self._dt = 1.0 / guardian_hz
        self._stop_event = threading.Event()
        self._m4_state = "FREE"  # "FREE" | "FLOOR_HOLD"

    def stop(self) -> None:
        self._stop_event.set()

    def _enforce_motor3(self) -> None:
        self.leader.set_torque(self.motor3_id, True)
        self.leader.write_goal_position(self.motor3_id, self.motor3_hold_deg)

    def _enforce_motor4(self) -> None:
        angle = self.shared.get_m4_angle()
        if angle is None:
            return

        if self._m4_state == "FREE":
            if angle <= self.motor4_floor_deg:
                self.leader.set_torque(self.motor4_id, True)
                self.leader.write_goal_position(self.motor4_id, self.motor4_floor_deg)
                self._m4_state = "FLOOR_HOLD"
        elif self._m4_state == "FLOOR_HOLD":
            if angle > self.motor4_floor_deg + self.motor4_hysteresis:
                self.leader.set_torque(self.motor4_id, False)
                self._m4_state = "FREE"
            else:
                self.leader.write_goal_position(self.motor4_id, self.motor4_floor_deg)

    def run(self) -> None:
        logger.debug("[KinovaLeaderGuardian] Started.")
        while not self._stop_event.is_set():
            t0 = time.time()
            with self.lock:
                self._enforce_motor3()
                self._enforce_motor4()
            elapsed = time.time() - t0
            remaining = self._dt - elapsed
            if remaining > 0:
                time.sleep(remaining)
        logger.debug("[KinovaLeaderGuardian] Stopped.")


class KinovaLeader(Teleoperator):
    """
    LeRobot teleoperator wrapping a 7-DOF ST3215 serial servo leader arm.

    Reads joint angles from 7 servo motors over serial, applies per-motor
    calibration offsets and direction flips, and returns normalized joint
    positions in radians suitable for the KinovaFollower robot.

    Joint constraints (Motor 3 lock, Motor 4 soft floor) are enforced by a
    background JointGuardian daemon thread.

    Usage
    -----
    >>> config = KinovaLeaderConfig(port="/dev/ttyACM0")
    >>> teleop = KinovaLeader(config)
    >>> teleop.connect()
    >>> action = teleop.get_action()   # {j1: float, ..., j7: float, gripper_val: float}
    >>> teleop.disconnect()
    """

    config_class = KinovaLeaderConfig
    name = "kinova_leader"

    def __init__(self, config: KinovaLeaderConfig):
        super().__init__(config)
        self.config = config

        self._motor_ids = config.motor_ids
        self._offsets = np.array(config.offsets, dtype=np.float64)
        self._directions = np.array(config.directions, dtype=np.float64)

        self._m3_idx = self._motor_ids.index(3)
        self._m4_idx = self._motor_ids.index(4)

        self._leader: LeaderArmController | None = None
        self._guardian: _JointGuardian | None = None
        self._serial_lock = threading.Lock()
        self._shared = _SharedState()
        self._connected = False

        # Last valid action — returned if a read cycle yields None readings
        self._last_action: dict[str, float] = {f"j{i}": 0.0 for i in range(1, 8)}
        self._last_action["gripper_val"] = config.default_gripper_val

        self._gripper_motor_id = config.gripper_motor_id
        self._gripper_closed  = config.gripper_closed_deg
        self._gripper_open    = config.gripper_open_deg
        self._gripper_span    = config.gripper_open_deg - config.gripper_closed_deg

    @property
    def action_features(self) -> dict[str, type]:
        """Returns j1–j7 radians + gripper_val, matching KinovaFollower.action_features."""
        features = {f"j{i}": float for i in range(1, 8)}
        features["gripper_val"] = float
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        """Always True — offsets/directions live in config."""
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._leader = LeaderArmController(
            port=self.config.port,
            baudrate=self.config.baudrate,
            motor_ids=self._motor_ids,
        )
        if not self._leader.connect():
            raise ConnectionError(
                f"Failed to open serial port {self.config.port} for KinovaLeader."
            )

        # All motors free-wheel, then lock Motor 3
        self._leader.set_passive_mode()
        self._leader.set_torque(3, True)
        self._leader.write_goal_position(3, self.config.motor3_hold_deg)
        logger.info(f"[KinovaLeader] Motor 3 locked at {self.config.motor3_hold_deg:.2f}°")
        logger.info(
            f"[KinovaLeader] Motor 4 soft floor at {self.config.motor4_floor_deg:.1f}° "
            f"(hysteresis {self.config.motor4_hysteresis:.1f}°)"
        )

        self._guardian = _JointGuardian(
            leader=self._leader,
            serial_lock=self._serial_lock,
            shared=self._shared,
            motor3_id=3,
            motor3_hold_deg=self.config.motor3_hold_deg,
            motor4_id=4,
            motor4_floor_deg=self.config.motor4_floor_deg,
            motor4_hysteresis=self.config.motor4_hysteresis,
            guardian_hz=self.config.guardian_hz,
        )
        self._guardian.start()

        self._connected = True
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        pass  # No-op: calibration data lives in config

    def configure(self) -> None:
        pass  # Motor init is done in connect()

    def get_action(self) -> dict[str, Any]:
        """
        Read all 7 motor angles, apply calibration, return radians dict.
        Falls back to last valid action if any motor read fails.
        Motor 3 raw angle is overridden with motor3_hold_deg before calibration.
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._serial_lock:
            leader_data = self._leader.get_all_angles()
            gripper_raw = self._leader.read_angle(self._gripper_motor_id)     # 8th servo

        # Share Motor 4 angle with guardian for soft-floor enforcement
        self._shared.set_m4_angle(leader_data.get(self._motor_ids[self._m4_idx]))

        # Fallback if any reading is missing
        if not (
            len(leader_data) == len(self._motor_ids)
            and all(v is not None for v in leader_data.values())
        ):
            logger.warning("[KinovaLeader] Incomplete motor readings — returning last action.")
            return dict(self._last_action)

        raw_angles = np.array(
            [leader_data[mid] for mid in self._motor_ids], dtype=np.float64
        )

        # Override Motor 3 so follower joint 3 always holds neutral
        raw_angles[self._m3_idx] = self.config.motor3_hold_deg

        norm_angles = (raw_angles - self._offsets) * self._directions
        target_rad = np.deg2rad(norm_angles)

        # Normalize gripper values
        if gripper_raw is not None:
            gripper_val = (gripper_raw - self._gripper_closed) / self._gripper_span
            gripper_val = float(max(0.0, min(0.8, gripper_val)))
        else:
            gripper_val = self._last_action.get("gripper_val", self.config.default_gripper_val)

        action: dict[str, Any] = {f"j{i + 1}": float(target_rad[i]) for i in range(7)}
        action["gripper_val"] = gripper_val

        self._last_action = dict(action)
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # No feedback mechanism on the leader arm

    def disconnect(self) -> None:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._guardian is not None:
            self._guardian.stop()
            self._guardian.join(timeout=1.0)
            self._guardian = None

        if self._leader is not None:
            with self._serial_lock:
                self._leader.set_torque(3, False)
                self._leader.set_torque(4, False)
                self._leader.set_passive_mode()
            self._leader.close()
            self._leader = None

        self._connected = False
        logger.info(f"{self} disconnected. All motors released to passive mode.")
