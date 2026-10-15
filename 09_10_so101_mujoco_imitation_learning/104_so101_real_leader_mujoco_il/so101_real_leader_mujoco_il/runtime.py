from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from . import JOINT_NAMES


ASSET_DIR = Path(__file__).resolve().with_name("assets")
DEFAULT_XML = ASSET_DIR / "lift_cube_calibration.xml"
DEFAULT_CALIBRATION = ASSET_DIR / "full_arm_calibration_leader.json"


def raw_to_joint_target(raw: float, calibration: dict[str, Any], limits: np.ndarray) -> float:
    """Map a calibrated leader encoder value to a MuJoCo joint position."""
    low = float(calibration["range_min"])
    high = float(calibration["range_max"])
    if high <= low:
        raise ValueError(f"invalid calibration range: {low}..{high}")
    bounded_raw = float(np.clip(raw, low, high))
    reversed_direction = bool(calibration.get("drive_mode", 0))

    # New calibrations include an observed encoder position for MuJoCo 0 rad.
    # Use a piecewise map so zero remains exact even when the physical travel
    # is asymmetric. Older JSON files remain compatible via linear mapping.
    if "center_raw" in calibration:
        center = float(calibration["center_raw"])
        if reversed_direction:
            bounded_raw = low + high - bounded_raw
            center = low + high - center
        if not low < center < high:
            raise ValueError(f"center_raw {calibration['center_raw']} is outside calibration range")
        if bounded_raw <= center:
            ratio = (bounded_raw - low) / (center - low)
            return float(limits[0] + ratio * (0.0 - limits[0]))
        ratio = (bounded_raw - center) / (high - center)
        return float(ratio * limits[1])

    ratio = (bounded_raw - low) / (high - low)
    if reversed_direction:
        ratio = 1.0 - ratio
    return float(limits[0] + ratio * (limits[1] - limits[0]))


class MujocoFollower:
    """MuJoCo follower with explicit joint, DoF, and actuator addresses."""

    def __init__(self, xml_path: str | Path = DEFAULT_XML, seed: int = 0):
        self.xml_path = Path(xml_path).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.joint_ids = self._ids(mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES)
        self.qpos_adr = self.model.jnt_qposadr[self.joint_ids]
        self.dof_adr = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = self._ids(mujoco.mjtObj.mjOBJ_ACTUATOR, JOINT_NAMES)
        self.joint_limits = self.model.jnt_range[self.joint_ids].copy()

        self.block_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        if self.block_joint < 0:
            raise RuntimeError("MuJoCo model has no 'cube_joint'")
        self.block_qpos_adr = int(self.model.jnt_qposadr[self.block_joint])
        self.reset()

    def _ids(self, object_type: mujoco.mjtObj, names: tuple[str, ...]) -> np.ndarray:
        ids = np.asarray(
            [mujoco.mj_name2id(self.model, object_type, name) for name in names],
            dtype=np.int32,
        )
        if np.any(ids < 0):
            missing = [name for name, idx in zip(names, ids) if idx < 0]
            raise RuntimeError(f"MuJoCo model is missing: {missing}")
        return ids

    @property
    def state(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].astype(np.float32).copy()

    def apply_target(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (len(JOINT_NAMES),):
            raise ValueError(f"expected six joint targets, got {target.shape}")
        if not np.isfinite(target).all():
            raise ValueError(f"joint target contains NaN or Inf: {target}")
        # The XML uses position actuators. ctrl therefore is a joint target,
        # not an externally computed PD torque.
        low = self.model.actuator_ctrlrange[self.actuator_ids, 0]
        high = self.model.actuator_ctrlrange[self.actuator_ids, 1]
        self.data.ctrl[self.actuator_ids] = np.clip(target, low, high)

    def step_for(self, seconds: float) -> None:
        end_time = self.data.time + seconds
        while self.data.time + 1e-12 < end_time:
            mujoco.mj_step(self.model, self.data)

    def reset(self, randomize_block: bool = True) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if randomize_block:
            qadr = self.block_qpos_adr
            self.data.qpos[qadr : qadr + 3] = [
                self.rng.uniform(0.40, 0.50),
                self.rng.uniform(-0.12, 0.12),
                0.03,
            ]
            self.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.qpos_adr]
        mujoco.mj_forward(self.model, self.data)


def load_leader_calibration(path: str | Path = DEFAULT_CALIBRATION) -> dict[str, Any]:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        calibration = json.load(stream)
    missing = [name for name in JOINT_NAMES if name not in calibration]
    if missing:
        raise ValueError(f"leader calibration is missing joints: {missing}")
    required = {"id", "range_min", "range_max"}
    for name in JOINT_NAMES:
        missing_fields = required.difference(calibration[name])
        if missing_fields:
            raise ValueError(
                f"leader calibration for {name} is missing fields: "
                f"{sorted(missing_fields)}"
            )
        if float(calibration[name]["range_max"]) <= float(
            calibration[name]["range_min"]
        ):
            raise ValueError(f"leader calibration for {name} has an invalid range")
        if "center_raw" in calibration[name] and not (
            float(calibration[name]["range_min"])
            < float(calibration[name]["center_raw"])
            < float(calibration[name]["range_max"])
        ):
            raise ValueError(f"leader calibration for {name} has center_raw outside its range")
    return calibration
