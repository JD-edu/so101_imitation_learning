from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from . import JOINT_NAMES


@dataclass(frozen=True)
class Workspace:
    block_x: tuple[float, float] = (0.28, 0.32)
    block_y: tuple[float, float] = (0.10, 0.14)
    target_xy: tuple[float, float] = (0.20, -0.16)


class SO101PickPlaceEnv:
    """Small MuJoCo environment with explicit indices and no hardware dependency."""

    def __init__(self, seed: int = 0, render_size: tuple[int, int] = (224, 224)):
        xml = Path(__file__).with_name("assets") / "pick_place.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.workspace = Workspace()
        self.width, self.height = render_size
        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        )
        if np.any(self.joint_ids < 0):
            raise RuntimeError("SO-101 joint names do not match the model")
        self.qpos_adr = self.model.jnt_qposadr[self.joint_ids]
        self.dof_adr = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES]
        )
        self.tcp_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "graspframe")
        self.gripper_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )
        self.pad_geoms = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
             for name in ("static_finger_pad", "moving_finger_pad")]
        )
        self.block_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "red_block_geom"
        )
        self.block_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "red_block"
        )
        # Replace point-like fingertip proxies with thin rubber pad surfaces.
        self.model.geom_size[self.pad_geoms] = [0.0025, 0.015, 0.018]
        self.block_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint"
        )
        self.block_qadr = int(self.model.jnt_qposadr[self.block_joint])
        self.block_dadr = int(self.model.jnt_dofadr[self.block_joint])
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview"
        )
        self.renderer: Optional[mujoco.Renderer] = None
        self.grip_local_offset: Optional[np.ndarray] = None
        # Define the IK point at the midpoint of the physical pads when their
        # gap matches the 30 mm block. Store it in the rigid gripper frame.
        reference = mujoco.MjData(self.model)
        reference.qpos[self.qpos_adr] = [0.0, -0.45, 0.85, 0.45, 0.0, 0.25]
        mujoco.mj_forward(self.model, reference)
        midpoint = reference.geom_xpos[self.pad_geoms].mean(axis=0)
        rotation = reference.xmat[self.gripper_body].reshape(3, 3)
        self.pinch_local = rotation.T @ (midpoint - reference.xpos[self.gripper_body])
        pad_axis = reference.geom_xpos[self.pad_geoms[1]] - reference.geom_xpos[self.pad_geoms[0]]
        self.pinch_axis_local = rotation.T @ (pad_axis / np.linalg.norm(pad_axis))
        self.reset()

    @property
    def joint_limits(self) -> np.ndarray:
        return self.model.jnt_range[self.joint_ids].copy()

    @property
    def state(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].astype(np.float32).copy()

    @property
    def tcp_position(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site].copy()

    @property
    def pinch_position(self) -> np.ndarray:
        rotation = self.data.xmat[self.gripper_body].reshape(3, 3)
        return self.data.xpos[self.gripper_body] + rotation @ self.pinch_local

    @property
    def block_position(self) -> np.ndarray:
        return self.data.qpos[self.block_qadr : self.block_qadr + 3].copy()

    def reset(self, block_xy: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        home = np.array([0.0, -0.45, 0.85, 0.45, 0.0, 1.35])
        self.data.qpos[self.qpos_adr] = np.clip(
            home, self.joint_limits[:, 0], self.joint_limits[:, 1]
        )
        if block_xy is None:
            block_xy = np.array(
                [
                    self.rng.uniform(*self.workspace.block_x),
                    self.rng.uniform(*self.workspace.block_y),
                ]
            )
        self.data.qpos[self.block_qadr : self.block_qadr + 3] = [*block_xy, 0.015]
        self.data.qpos[self.block_qadr + 3 : self.block_qadr + 7] = [1, 0, 0, 0]
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.qpos_adr]
        self.grip_local_offset = None
        mujoco.mj_forward(self.model, self.data)
        return self.observation(render=False)

    def finger_contacts(self) -> tuple[bool, bool, float]:
        """Return bilateral pad contact and summed physical normal force."""
        touching = [False, False]
        force = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            for side, pad in enumerate(self.pad_geoms):
                if pair == {int(pad), self.block_geom}:
                    touching[side] = True
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, index, wrench)
                    force += max(0.0, float(wrench[0]))
        return touching[0], touching[1], force

    def step(self, action: np.ndarray, substeps: int = 17) -> dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,):
            raise ValueError(f"expected six joint targets, got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError(f"action contains NaN or Inf: {action}")
        self.data.ctrl[self.actuator_ids] = np.clip(
            action, self.model.actuator_ctrlrange[self.actuator_ids, 0],
            self.model.actuator_ctrlrange[self.actuator_ids, 1]
        )
        if action[5] > 0.8:
            self.grip_local_offset = None
        for _ in range(substeps):
            self.data.xfrc_applied[self.block_body] = 0.0
            if self.grip_local_offset is not None:
                rotation = self.data.xmat[self.gripper_body].reshape(3, 3)
                desired = self.data.xpos[self.gripper_body] + rotation @ self.grip_local_offset
                position_error = desired - self.data.xpos[self.block_body]
                relative_velocity = (
                    self.data.cvel[self.block_body, 3:6]
                    - self.data.cvel[self.gripper_body, 3:6]
                )
                force = 180.0 * position_error - 4.0 * relative_velocity
                norm = np.linalg.norm(force)
                if norm > 3.0:
                    force *= 3.0 / norm
                self.data.xfrc_applied[self.block_body, :3] = force
            mujoco.mj_step(self.model, self.data)
            if self.grip_local_offset is None and action[5] < 0.3:
                left, right, _ = self.finger_contacts()
                if left and right:
                    rotation = self.data.xmat[self.gripper_body].reshape(3, 3)
                    self.grip_local_offset = rotation.T @ (
                        self.data.xpos[self.block_body] - self.data.xpos[self.gripper_body]
                    )
        return self.observation(render=False)

    def observation(self, render: bool = True) -> dict[str, np.ndarray]:
        obs = {"state": self.state, "block_position": self.block_position.astype(np.float32)}
        if render:
            obs["image"] = self.render()
        return obs

    def render(self) -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.renderer.update_scene(self.data, camera=self.camera_id)
        return self.renderer.render().copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
