from __future__ import annotations

import serial


class MiniFeetechDriver:
    """Minimal STS3215 serial driver used by leader calibration/recording."""

    REG_PRESENT_POSITION = 56
    REG_TORQUE_ENABLE = 40

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 1_000_000):
        self.ser = serial.Serial(port, baudrate, timeout=0.05)

    @staticmethod
    def _make_packet(motor_id: int, instruction: int, parameters: list[int]) -> bytearray:
        length = len(parameters) + 2
        packet = [0xFF, 0xFF, motor_id, length, instruction, *parameters]
        packet.append((~(sum(packet[2:]) & 0xFF)) & 0xFF)
        return bytearray(packet)

    def _write_only(self, motor_id: int, instruction: int, parameters: list[int]) -> None:
        self.ser.write(self._make_packet(motor_id, instruction, parameters))

    def _write_and_read(
        self,
        motor_id: int,
        instruction: int,
        parameters: list[int],
        response_bytes: int,
    ) -> bytes:
        self.ser.reset_input_buffer()
        self.ser.write(self._make_packet(motor_id, instruction, parameters))
        return self.ser.read(response_bytes)

    def set_torque(self, motor_id: int, enable: bool) -> None:
        self._write_only(
            motor_id, 0x03, [self.REG_TORQUE_ENABLE, 1 if enable else 0]
        )

    def get_position(self, motor_id: int) -> int | None:
        response = self._write_and_read(
            motor_id, 0x02, [self.REG_PRESENT_POSITION, 2], response_bytes=8
        )
        if len(response) < 8:
            return None
        if response[0:2] != b"\xff\xff" or response[2] != motor_id:
            return None
        return ((response[6] << 8) | response[5]) & 0x0FFF

    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()

    def __enter__(self) -> "MiniFeetechDriver":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
