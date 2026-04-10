from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("kinova_leader")
@dataclass
class KinovaLeaderConfig(TeleoperatorConfig):
    # Serial port configuration
    port: str = "/dev/ttyACM0"
    baudrate: int = 1000000
    motor_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    gripper_motor_id: int = 8  # virtual servo motor

    # Per-motor calibration in degrees (measured experimentally).
    # norm_angle = (raw_angle - offset) * direction  →  then converted to radians.
    # Index i corresponds to motor_ids[i].
    offsets: list[float] = field(
        default_factory=lambda: [297.51, 13.8, 125.86, 44.38, 203.7, 308.0, 202]
    )
    directions: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, -1, 1])

    # Motor 3 (motor_ids index 2): torque-locked to this angle for the entire session.
    # The follower's joint 3 will always receive its calibrated neutral command.
    motor3_hold_deg: float = 125.86

    # Motor 4 (motor_ids index 3): soft floor — resist movement below this angle.
    motor4_floor_deg: float = 45.0
    motor4_hysteresis: float = 3.0

    # Guardian thread rate (Hz). Runs faster than the main 50 Hz read loop
    # so joint constraints are enforced quickly between angle reads.
    guardian_hz: int = 200

    # Gripper value sent to the follower on every get_action() call.
    # The leader arm has no gripper motor; this constant is appended instead.
    # Range: 0.0 (fully closed) – 100.0 (fully open) in follower units.
    default_gripper_val: float = 1.0

    gripper_closed_deg: float = 22.0  # calibrate before starting
    gripper_open_deg: float = 67.0 