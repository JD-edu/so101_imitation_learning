from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import SO101PickPlaceEnv
from .ik import DampedLeastSquaresIK


@dataclass
class ExpertFrame:
    state: np.ndarray
    action: np.ndarray
    image: np.ndarray


class IKPickPlaceExpert:
    def __init__(self, env: SO101PickPlaceEnv, fps: int = 30):
        self.env = env
        self.fps = fps
        self.ik = DampedLeastSquaresIK(env)
        self.last_episode_metrics: dict[str, float | int | bool] = {}

    @staticmethod
    def _smooth_segment(start: np.ndarray, end: np.ndarray, steps: int) -> np.ndarray:
        t = np.linspace(0.0, 1.0, steps + 1)[1:]
        blend = t * t * (3.0 - 2.0 * t)
        return start[None] + blend[:, None] * (end - start)[None]

    def _pose_for(self, xyz: np.ndarray, gripper: float, seed: np.ndarray) -> np.ndarray:
        pose = self.ik.solve(xyz, seed=seed)
        pose[5] = gripper
        return pose

    def generate(
        self, block_xy: np.ndarray | None = None, capture: bool = True,
        callback=None, image_provider=None,
    ) -> list[ExpertFrame]:
        self.env.reset(block_xy)
        block = self.env.block_position.copy()
        target = np.array([*self.env.workspace.target_xy, 0.015])
        open_q, closed_q = 1.20, 0.18
        q = self.env.state
        keyframes: list[tuple[str, np.ndarray, int]] = []
        for label, xyz, grip, steps in (
            ("approach", block + [0, 0, 0.075], open_q, 30),
            ("descend", block + [0, 0, 0.002], open_q, 24),
            ("close", block + [0, 0, 0.002], closed_q, 24),
            ("lift", block + [0, 0, 0.075], closed_q, 25),
            ("transfer", target + [0, 0, 0.075], closed_q, 35),
            ("lower", target + [0, 0, 0.002], closed_q, 26),
            ("open", target + [0, 0, 0.002], open_q, 20),
            ("retreat", target + [0, 0, 0.085], open_q, 24),
        ):
            goal = self._pose_for(np.asarray(xyz), grip, q)
            keyframes.append((label, goal, steps))
            q = goal

        frames: list[ExpertFrame] = []
        bilateral_frames = 0
        max_block_height = float(self.env.block_position[2])
        q = self.env.state
        for label, goal, steps in keyframes:
            trajectory = self._smooth_segment(q, goal, steps)
            for i, action in enumerate(trajectory):
                state = self.env.state
                image = (
                    image_provider() if image_provider is not None
                    else self.env.render() if capture
                    else np.empty((0, 0, 3), np.uint8)
                )
                frames.append(ExpertFrame(state, action.astype(np.float32), image))
                self.env.step(action)
                left, right, _ = self.env.finger_contacts()
                bilateral_frames += int(left and right)
                max_block_height = max(max_block_height, float(self.env.block_position[2]))
                if callback is not None:
                    callback()
            q = goal
        # Let the released block settle before success evaluation.
        for _ in range(30):
            self.env.step(q, substeps=4)
        error = np.linalg.norm(self.env.block_position[:2] - target[:2])
        self.last_episode_metrics = {
            "bilateral_contact_frames": bilateral_frames,
            "max_block_height": max_block_height,
            "final_target_error": float(error),
            "uses_weld": False,
        }
        if bilateral_frames < 3:
            raise RuntimeError("expert failed: no bilateral physical finger contact")
        if max_block_height < 0.055:
            raise RuntimeError(
                f"expert failed: block was not physically lifted (max z={max_block_height:.3f} m)"
            )
        if error > 0.045:
            raise RuntimeError(f"expert failed to place block (xy error={error:.3f} m)")
        return frames
