from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .env import SO101PickPlaceEnv


class DampedLeastSquaresIK:
    def __init__(self, env: SO101PickPlaceEnv, damping: float = 2e-3):
        self.env = env
        self.damping = damping
        self.arm_dofs = env.dof_adr[:5]
        self.data = mujoco.MjData(env.model)

    def solve(
        self,
        target: np.ndarray,
        seed: np.ndarray | None = None,
        max_iterations: int = 180,
        tolerance: float = 1.5e-3,
    ) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64)
        home = np.array([0.0, -0.45, 0.85, 0.45, 0.0])
        limits = self.env.joint_limits[:5]
        lower, upper = limits[:, 0] + 0.02, limits[:, 1] - 0.02

        def residual(q: np.ndarray) -> np.ndarray:
            self.data.qpos[self.env.qpos_adr[:5]] = q
            self.data.qpos[self.env.qpos_adr[5]] = 0.25
            mujoco.mj_forward(self.env.model, self.data)
            rotation = self.data.xmat[self.env.gripper_body].reshape(3, 3)
            position = self.data.xpos[self.env.gripper_body] + rotation @ self.env.pinch_local
            pinch_axis = rotation @ self.env.pinch_axis_local
            # Align the pad closing axis with world X so a square block is
            # pinched face-to-face rather than across its corners.
            return np.r_[position - target, 0.10 * pinch_axis[1], 0.10 * pinch_axis[2]]

        pan_hint = -np.arctan2(target[1], target[0])
        candidates = [
            seed[:5] if seed is not None else home,
            home,
            np.array([pan_hint, -0.6, 0.8, 0.5, 0.0]),
        ]
        results = [
            least_squares(
                residual, np.clip(initial, lower, upper), bounds=(lower, upper),
                max_nfev=max(300, max_iterations * 2),
                xtol=1e-9, ftol=1e-9, gtol=1e-9,
            )
            for initial in candidates
        ]
        result = min(results, key=lambda item: np.linalg.norm(residual(item.x)))
        error = residual(result.x)
        if np.linalg.norm(error[:3]) > 0.020 or np.linalg.norm(error[3:]) > 0.020:
            raise RuntimeError(
                f"IK did not converge for target {target.round(3)} "
                f"(position error={np.linalg.norm(error[:3]):.3f} m)"
            )
        pose = self.env.state.copy()
        pose[:5] = result.x
        return pose.astype(np.float32)
