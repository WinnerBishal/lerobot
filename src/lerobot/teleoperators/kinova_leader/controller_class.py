import serial
import time


class LeaderArmController:
    # --- Register Addresses (SMS_STS protocol) ---
    REG_TORQUE_ENABLE = 40
    REG_GOAL_POS_L = 42    # Goal Position Low byte  (2-byte, little-endian)
    REG_PRESENT_POS_L = 56  # Present Position Low byte (2-byte, little-endian)

    def __init__(self, port="/dev/ttyACM0", baudrate=1000000, motor_ids=None):
        self.port = port
        self.baudrate = baudrate
        self.ids = motor_ids if motor_ids else [1, 2, 3, 4, 5, 6, 7]
        self.ser = None

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
            self.ser.reset_input_buffer()
            return True
        except Exception as e:
            print(f"Connection Error: {e}")
            return False

    def _calculate_checksum(self, packet_no_sum):
        return (~sum(packet_no_sum[2:]) & 0xFF)

    def _send_packet(self, id, instruction, parameters):
        length = len(parameters) + 2
        packet = [0xFF, 0xFF, id, length, instruction] + parameters
        packet.append(self._calculate_checksum(packet))
        self.ser.write(bytes(packet))

    def _read_status_packet(self, expected_id, data_len):
        """Synchronises to 0xFF 0xFF header and validates checksum."""
        start_time = time.time()
        while (time.time() - start_time) < 0.04:   # 40 ms timeout
            if self.ser.in_waiting >= 2:
                if self.ser.read(1) == b'\xff' and self.ser.read(1) == b'\xff':
                    remainder = self.ser.read(data_len + 4)
                    if len(remainder) == (data_len + 4):
                        packet = [0xFF, 0xFF] + list(remainder)
                        if packet[2] != expected_id:
                            continue
                        calc_sum = self._calculate_checksum(packet[:-1])
                        if calc_sum == packet[-1]:
                            return packet[5:-1]   # data bytes only
        return None

    def set_torque(self, motor_id: int, enable: bool):
        value = 1 if enable else 0
        self._send_packet(motor_id, 0x03, [self.REG_TORQUE_ENABLE, value])

    def write_goal_position(self, motor_id: int, angle_deg: float):
        angle_deg = max(0.0, min(359.99, angle_deg))
        ticks = int(round(angle_deg * 4096.0 / 360.0)) & 0x0FFF
        low = ticks & 0xFF
        high = (ticks >> 8) & 0xFF
        self._send_packet(motor_id, 0x03, [self.REG_GOAL_POS_L, low, high])

    def read_angle(self, motor_id: int) -> float | None:
        self.ser.reset_input_buffer()
        self._send_packet(motor_id, 0x02, [self.REG_PRESENT_POS_L, 2])
        data = self._read_status_packet(motor_id, 2)
        if data:
            raw_val = data[0] | (data[1] << 8)
            if raw_val & (1 << 15):
                signed_ticks = -(raw_val & 0x7FFF)
            else:
                signed_ticks = raw_val
            angle_deg = (signed_ticks % 4096) * (360.0 / 4096.0)
            return round(angle_deg, 2)
        return None

    def get_all_angles(self) -> dict:
        return {mid: self.read_angle(mid) for mid in self.ids}

    def set_passive_mode(self):
        print("Disabling Torque on all motors (Passive Mode)…")
        for mid in self.ids:
            self.set_torque(mid, False)
            time.sleep(0.005)

    def close(self):
        if self.ser:
            self.ser.close()
