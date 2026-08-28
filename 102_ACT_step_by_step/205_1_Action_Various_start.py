"""
205_Action_Various_start.py의 학습/평가 흐름을 그대로 사용하면서,
생성된 합성 데이터의 시작점 분포와 샘플 궤적을 Matplotlib로 표시한다.

원본 파일은 수정하지 않는다.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ORIGINAL_PATH = Path(__file__).with_name("205_Action_Various_start.py")
PLOT_PATH = Path(__file__).with_name("205_1_synthetic_data_distribution.png")
NUM_TRAJECTORIES_TO_PLOT = 10
END_EFFECTOR_SITE = "graspframe"


def load_original_module():
    """숫자로 시작하는 원본 파일을 파이썬 모듈로 불러온다."""
    spec = spec_from_file_location("action_various_start_original", ORIGINAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"원본 파일을 불러올 수 없습니다: {ORIGINAL_PATH}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def states_to_end_effector_xyz(mj_model, states_np):
    """관절 상태 궤적을 MuJoCo FK로 엔드이펙터 XYZ 궤적으로 변환한다."""
    site_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE
    )
    if site_id == -1:
        raise ValueError(
            f"MuJoCo 모델에서 엔드이펙터 site '{END_EFFECTOR_SITE}'를 찾을 수 없습니다."
        )

    fk_data = mujoco.MjData(mj_model)
    num_episodes, num_steps, _ = states_np.shape
    ee_xyz = np.empty((num_episodes, num_steps, 3), dtype=np.float64)

    for episode_idx in range(num_episodes):
        for step_idx in range(num_steps):
            mujoco.mj_resetData(mj_model, fk_data)
            fk_data.qpos[: states_np.shape[2]] = states_np[episode_idx, step_idx]
            mujoco.mj_forward(mj_model, fk_data)
            ee_xyz[episode_idx, step_idx] = fk_data.site_xpos[site_id]

    return ee_xyz


def plot_synthetic_data(mj_model, states):
    """
    states: [episode, step, joint] 형태의 Tensor

    관절 상태에 MuJoCo FK를 적용하여 graspframe의 XYZ를 구하고,
    전체 시작점 분포와 샘플 10개 궤적 및 끝점을 한 그래프에 표시한다.
    """
    states_np = states.detach().cpu().numpy()
    ee_trajectories = states_to_end_effector_xyz(mj_model, states_np)
    start_points = ee_trajectories[:, 0, :]

    num_episodes = states_np.shape[0]
    sample_count = min(NUM_TRAJECTORIES_TO_PLOT, num_episodes)
    sampled_indices = np.linspace(
        0, num_episodes - 1, sample_count, dtype=int
    )

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 100개 전체 에피소드의 시작점 분포
    ax.scatter(
        start_points[:, 0],
        start_points[:, 1],
        start_points[:, 2],
        s=28,
        color="tab:blue",
        alpha=0.35,
        label=f"all start points ({num_episodes})",
    )

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, sample_count))
    for color, episode_idx in zip(colors, sampled_indices):
        trajectory = ee_trajectories[episode_idx]

        # 선택된 에피소드의 전체 궤적
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label=f"episode {episode_idx + 1}",
        )

        # 각 궤적의 시작점과 끝점
        ax.scatter(
            trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
            marker="o", s=65, color=color, edgecolor="black", linewidth=0.8,
            zorder=4,
        )
        ax.scatter(
            trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
            marker="X", s=100, color=color, edgecolor="black", linewidth=0.8,
            zorder=5,
        )

    # 마커 의미를 범례에 한 번만 추가
    ax.scatter([], [], [], marker="o", s=65, color="white", edgecolor="black",
               label="sampled start")
    ax.scatter([], [], [], marker="X", s=100, color="white", edgecolor="black",
               label="sampled end")

    ax.set_title(
        f"End-Effector ({END_EFFECTOR_SITE}) FK Trajectories",
        fontsize=14,
        pad=18,
    )
    ax.set_xlabel("End-effector X (m)", labelpad=10)
    ax.set_ylabel("End-effector Y (m)", labelpad=10)
    ax.set_zlabel("End-effector Z (m)", labelpad=10)
    ax.view_init(elev=25, azim=45)
    ax.grid(True, alpha=0.3)

    # 세 축의 실제 길이 비율을 맞춰 궤적이 시각적으로 왜곡되지 않게 한다.
    xyz_range = np.ptp(ee_trajectories.reshape(-1, 3), axis=0)
    ax.set_box_aspect(np.maximum(xyz_range, 1e-6))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160, bbox_inches="tight")

    print(f"[*] 시작점 분포 및 샘플 궤적 그래프 저장 완료: {PLOT_PATH}")
    print(f"[*] 표시한 에피소드 번호: {(sampled_indices + 1).tolist()}")
    plt.show()


def main():
    original = load_original_module()
    original_generator = original.generate_multistart_episodes

    def generate_and_plot(model, data):
        states, actions = original_generator(model, data)
        plot_synthetic_data(model, states)
        return states, actions

    # 원본 main의 데이터 생성 지점에 시각화만 끼워 넣는다.
    original.generate_multistart_episodes = generate_and_plot
    original.main()


if __name__ == "__main__":
    main()
