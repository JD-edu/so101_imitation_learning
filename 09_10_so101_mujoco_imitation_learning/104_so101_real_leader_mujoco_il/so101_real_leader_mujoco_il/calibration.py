"""Hardware-leader calibration consumed directly by the MuJoCo pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import mujoco

from . import JOINT_NAMES
from .hardware import MiniFeetechDriver
from .runtime import DEFAULT_CALIBRATION, DEFAULT_XML


HALF_TURN = 2047


class LeaderCalibrator:
    """Measure leader endpoints without rewriting persistent motor registers."""

    def __init__(self, xml_path: str | Path, port: str, baudrate: int = 1_000_000):
        self.xml_path = Path(xml_path).expanduser().resolve()
        self.driver = MiniFeetechDriver(port=port, baudrate=baudrate)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self._validate_model()
        print(f"MuJoCo model: {self.xml_path}")

    def _validate_model(self) -> None:
        missing = [
            name
            for name in JOINT_NAMES
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0
        ]
        if missing:
            raise ValueError(f"MuJoCo model is missing joints: {missing}")

    def _read_position(self, motor_id: int, label: str) -> int:
        for _ in range(5):
            value = self.driver.get_position(motor_id)
            if value is not None:
                return int(value)
            time.sleep(0.05)
        raise RuntimeError(f"motor {motor_id}: failed to read {label} position")

    def calibrate_joint(self, joint_name: str, motor_id: int) -> dict[str, int]:
        print(f"\n>>> {joint_name} (motor ID {motor_id})")
        self.driver.set_torque(motor_id, False)

        input("  리더를 MuJoCo 0 rad 자세로 옮긴 후 ENTER: ")
        center_raw = self._read_position(motor_id, "center")
        input("  리더를 MuJoCo 최소 각도 자세로 옮긴 후 ENTER: ")
        raw_at_mujoco_min = self._read_position(motor_id, "minimum")
        input("  리더를 MuJoCo 최대 각도 자세로 옮긴 후 ENTER: ")
        raw_at_mujoco_max = self._read_position(motor_id, "maximum")

        if raw_at_mujoco_min == raw_at_mujoco_max:
            raise ValueError(f"{joint_name}: minimum and maximum encoder values are equal")
        range_low = min(raw_at_mujoco_min, raw_at_mujoco_max)
        range_high = max(raw_at_mujoco_min, raw_at_mujoco_max)
        if not range_low < center_raw < range_high:
            raise ValueError(
                f"{joint_name}: center encoder value {center_raw} is not inside "
                f"the measured range {range_low}..{range_high}; rerun calibration"
            )

        result = {
            "id": motor_id,
            "drive_mode": int(raw_at_mujoco_min > raw_at_mujoco_max),
            "center_raw": center_raw,
            "homing_offset": center_raw - HALF_TURN,
            "range_min": range_low,
            "range_max": range_high,
        }
        direction = "reversed" if result["drive_mode"] else "normal"
        print(
            f"  center={center_raw}, range={range_low}..{range_high}, "
            f"direction={direction}"
        )
        return result

    def run(self) -> dict[str, dict[str, int]]:
        results = {}
        try:
            for motor_id, joint_name in enumerate(JOINT_NAMES, start=1):
                results[joint_name] = self.calibrate_joint(joint_name, motor_id)
        finally:
            self.driver.close()
        return results


def save_calibration(calibration: dict, output_path: str | Path) -> Path | None:
    """Atomically save calibration and retain a timestamped previous version."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if output.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output.with_name(f"{output.stem}.backup_{timestamp}{output.suffix}")
        shutil.copy2(output, backup)

    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(calibration, stream, indent=4, ensure_ascii=False)
            stream.write("\n")
            temporary_name = stream.name
        os.replace(temporary_name, output)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return backup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a real SO-101 leader for the MuJoCo ACT pipeline"
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    calibration = LeaderCalibrator(args.xml, args.port, args.baudrate).run()
    backup = save_calibration(calibration, args.output)
    print(f"\nCalibration saved: {args.output.expanduser().resolve()}")
    if backup is not None:
        print(f"Previous calibration backup: {backup}")
    print("The collect command will use this file by default.")


if __name__ == "__main__":
    main()
