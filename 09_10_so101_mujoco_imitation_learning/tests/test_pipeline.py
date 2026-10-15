import numpy as np

from so101_mujoco_ik.env import SO101PickPlaceEnv
from so101_mujoco_ik.expert import IKPickPlaceExpert
from so101_real_leader_mujoco_il.runtime import MujocoFollower, raw_to_joint_target


def test_model_indices_and_limits():
    env = SO101PickPlaceEnv(seed=1)
    assert env.model.nu == 6
    assert env.state.shape == (6,)
    assert np.all(env.state >= env.joint_limits[:, 0])
    assert np.all(env.state <= env.joint_limits[:, 1])
    env.close()


def test_expert_places_randomized_block():
    env = SO101PickPlaceEnv(seed=11)
    expert = IKPickPlaceExpert(env)
    frames = expert.generate(capture=False)
    assert len(frames) == 208
    assert all(frame.state.shape == (6,) and frame.action.shape == (6,) for frame in frames)
    error = np.linalg.norm(env.block_position[:2] - env.workspace.target_xy)
    assert error < 0.045
    assert env.grip_local_offset is None
    assert expert.last_episode_metrics["bilateral_contact_frames"] >= 3
    assert expert.last_episode_metrics["max_block_height"] >= 0.055
    assert expert.last_episode_metrics["uses_weld"] is False
    # The UI demo replays these actions in a GLFW window afterwards.  Keeping
    # trajectory generation renderer-free avoids terminating GLFW before that
    # replay window is created.
    assert env.renderer is None
    env.close()


def test_randomized_ik_regression():
    for seed in range(10):
        env = SO101PickPlaceEnv(seed=seed)
        IKPickPlaceExpert(env).generate(capture=False)
        assert np.linalg.norm(env.block_position[:2] - env.workspace.target_xy) < 0.045
        env.close()


def test_real_leader_follower_uses_explicit_addresses_and_position_targets():
    follower = MujocoFollower(seed=3)
    assert follower.state.shape == (6,)
    assert np.all(follower.qpos_adr >= 0)
    assert np.all(follower.dof_adr >= 0)
    target = follower.joint_limits.mean(axis=1)
    follower.apply_target(target)
    np.testing.assert_allclose(
        follower.data.ctrl[follower.actuator_ids], target, rtol=0, atol=1e-6
    )


def test_leader_calibration_maps_to_joint_limits_and_drive_mode():
    limits = np.array([-2.0, 2.0])
    calibration = {"range_min": 100, "range_max": 900, "drive_mode": 0}
    assert raw_to_joint_target(100, calibration, limits) == -2.0
    assert raw_to_joint_target(900, calibration, limits) == 2.0
    calibration["drive_mode"] = 1
    assert raw_to_joint_target(100, calibration, limits) == 2.0


def test_new_leader_calibration_maps_measured_center_to_zero():
    limits = np.array([-1.0, 2.0])
    calibration = {
        "range_min": 100,
        "range_max": 900,
        "center_raw": 300,
        "drive_mode": 0,
    }
    assert raw_to_joint_target(100, calibration, limits) == -1.0
    assert raw_to_joint_target(300, calibration, limits) == 0.0
    assert raw_to_joint_target(900, calibration, limits) == 2.0

    calibration.update(center_raw=700, drive_mode=1)
    assert raw_to_joint_target(900, calibration, limits) == -1.0
    assert raw_to_joint_target(700, calibration, limits) == 0.0
    assert raw_to_joint_target(100, calibration, limits) == 2.0
