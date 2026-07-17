"""
아이디어 A — Waypoint 예측 과정 단계별 재생
=============================================
- 저장된 act_policy.pt 로드
- 추론(1회) → 20개 waypoint chunk 획득
- matplotlib 애니메이션으로 단계별 재생
  · 연한 점선: 전체 예측 경로 미리 표시 (한 번의 추론 결과)
  · 검정 원: 그리퍼가 waypoint를 순서대로 이동
  · 파란 점: 이미 지나온 궤적
  · 오른쪽 패널: 현재 스텝의 3축(x/y/z) 위치 & 목표까지 거리 바 차트
- 교육 포인트 텍스트 하단에 표시
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE    = 20
WORKSPACE_MIN = np.array([0.0, -0.3, 0.0])
WORKSPACE_MAX = np.array([0.5,  0.3, 0.4])
MODEL_PATH    = "./act_policy.pt"
SAVE_GIF      = "./viz_A_waypoint_playback.gif"
SAVE_PNG      = "./viz_A_final_frame.png"

# ─────────────────────────────────────────
# 모델 정의 & 로드
# ─────────────────────────────────────────
class ActionChunkingPolicy(nn.Module):
    def __init__(self, obs_dim=6, chunk_dim=CHUNK_SIZE*3,
                 hidden=256, n_layers=3, dropout=0.1):
        super().__init__()
        layers, in_dim = [], obs_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden),
                       nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers.append(nn.Linear(hidden, chunk_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict(self, start, target):
        obs = torch.tensor(np.concatenate([start, target]),
                           dtype=torch.float32).unsqueeze(0).to(DEVICE)
        out = self(obs).squeeze(0).cpu().numpy()
        return out.reshape(CHUNK_SIZE, 3)


def load_model(path):
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model = ActionChunkingPolicy(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(DEVICE).eval()


# ─────────────────────────────────────────
# 궤적 생성 (GT 비교용)
# ─────────────────────────────────────────
def generate_trajectory(start, target, n_steps=CHUNK_SIZE, curvature=0.15):
    t       = np.linspace(0, 1, n_steps).reshape(-1, 1)
    linear  = start + t * (target - start)
    d       = target - start
    perp    = np.cross(d, [0,0,1]) if abs(d[2]) < 1e-6 else np.cross(d, [1,0,0])
    norm    = np.linalg.norm(perp)
    perp    = perp / norm if norm > 1e-6 else np.array([0,0,1])
    offset  = curvature * 4 * t * (1-t) * perp.reshape(1,3)
    return linear + offset


# ─────────────────────────────────────────
# 메인 시각화
# ─────────────────────────────────────────
def main():
    # ── 1. 모델 로드 & 추론 ──────────────────────────
    print("모델 로드 중...")
    model = load_model(MODEL_PATH)

    rng    = np.random.default_rng(42)
    start  = rng.uniform(WORKSPACE_MIN, WORKSPACE_MAX)
    target = rng.uniform(WORKSPACE_MIN, WORKSPACE_MAX)

    # ★ 핵심: 한 번의 추론으로 chunk 전체 획득
    print("추론(1회) 실행 → action chunk 20개 waypoint 획득")
    chunk  = model.predict(start, target)          # shape: (20, 3)
    gt     = generate_trajectory(start, target)    # shape: (20, 3)

    rmse_per_step = np.sqrt(np.sum((chunk - gt)**2, axis=1))  # (20,)
    dist_to_goal  = np.linalg.norm(chunk - target, axis=1)    # (20,)

    print(f"  시작점  : {start.round(3)}")
    print(f"  목표점  : {target.round(3)}")
    print(f"  전체 RMSE: {np.sqrt(np.mean((chunk-gt)**2)):.4f} m")

    # ── 2. 그림 레이아웃 ────────────────────────────
    fig = plt.figure(figsize=(13, 7), facecolor="#0d0d0d")
    fig.suptitle(
        "ACT — Waypoint Chunk 단계별 재생  |  추론 횟수: 1회  →  실행 스텝: 20개",
        color="white", fontsize=13, fontweight="bold", y=0.97
    )

    gs  = gridspec.GridSpec(2, 2,
                            left=0.05, right=0.97,
                            top=0.91, bottom=0.10,
                            wspace=0.30, hspace=0.45)

    # 왼쪽 큰 패널: 3D 궤적
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax3d.set_facecolor("#111111")
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333333")

    ax3d.set_xlim(WORKSPACE_MIN[0], WORKSPACE_MAX[0])
    ax3d.set_ylim(WORKSPACE_MIN[1], WORKSPACE_MAX[1])
    ax3d.set_zlim(WORKSPACE_MIN[2], WORKSPACE_MAX[2])
    ax3d.set_xlabel("x (m)", color="#aaaaaa", labelpad=4)
    ax3d.set_ylabel("y (m)", color="#aaaaaa", labelpad=4)
    ax3d.set_zlabel("z (m)", color="#aaaaaa", labelpad=4)
    ax3d.tick_params(colors="#666666", labelsize=7)
    ax3d.set_title("3D 작업공간", color="white", fontsize=10, pad=6)

    # GT 참조 경로 (연한 흰 점선)
    ax3d.plot(gt[:,0], gt[:,1], gt[:,2],
              '--', color="#444444", linewidth=1.2, label="GT (참조)")

    # 예측 전체 경로 미리 표시 — 연한 청록색 점선
    ax3d.plot(chunk[:,0], chunk[:,1], chunk[:,2],
              ':', color="#00ffcc", linewidth=1.0, alpha=0.35,
              label="Predicted chunk (전체 미리보기)")

    # 시작점 · 목표점
    ax3d.scatter(*start,  color="#44ff44", s=90,  marker="s",  zorder=5, label="start")
    ax3d.scatter(*target, color="#ff4444", s=140, marker="*",  zorder=5, label="target")

    ax3d.legend(loc="upper left", fontsize=7,
                facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")

    # 오른쪽 위: 목표까지 거리 곡선
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_dist.set_facecolor("#111111")
    ax_dist.plot(range(CHUNK_SIZE), dist_to_goal,
                 color="#888888", linewidth=1.0, alpha=0.4)
    ax_dist.set_xlim(-0.5, CHUNK_SIZE - 0.5)
    ax_dist.set_ylim(0, dist_to_goal.max() * 1.15)
    ax_dist.set_xlabel("Waypoint 인덱스", color="#aaaaaa", fontsize=9)
    ax_dist.set_ylabel("목표까지 거리 (m)", color="#aaaaaa", fontsize=9)
    ax_dist.set_title("목표까지 남은 거리", color="white", fontsize=10)
    ax_dist.tick_params(colors="#666666")
    ax_dist.spines[:].set_edgecolor("#333333")

    # 오른쪽 아래: 스텝별 RMSE
    ax_err = fig.add_subplot(gs[1, 1])
    ax_err.set_facecolor("#111111")
    ax_err.bar(range(CHUNK_SIZE), rmse_per_step,
               color="#ff6666", alpha=0.3, width=0.7)
    ax_err.set_xlim(-0.5, CHUNK_SIZE - 0.5)
    ax_err.set_xlabel("Waypoint 인덱스", color="#aaaaaa", fontsize=9)
    ax_err.set_ylabel("RMSE (m)", color="#aaaaaa", fontsize=9)
    ax_err.set_title("스텝별 예측 오차 (vs GT)", color="white", fontsize=10)
    ax_err.tick_params(colors="#666666")
    ax_err.spines[:].set_edgecolor("#333333")

    # 교육 포인트 텍스트
    edu_text = (
        "💡 Action Chunking 핵심: 추론 1회 → waypoint 20개를 한꺼번에 예측  |  "
        "매 스텝 추론 대비 20× 효율  |  chunk 끝으로 갈수록 오차가 누적됨"
    )
    fig.text(0.5, 0.02, edu_text,
             ha="center", va="bottom", fontsize=8.5,
             color="#aaaaaa",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#1a1a1a", edgecolor="#444444"))

    # ── 3. 애니메이션 요소 초기화 ─────────────────────
    # 지나온 궤적 선
    traj_line, = ax3d.plot([], [], [], '-',
                           color="#00ccff", linewidth=2.0, alpha=0.9,
                           zorder=8)
    # 그리퍼 현재 위치
    gripper_pt, = ax3d.plot([], [], [], 'o',
                            color="white", markersize=10,
                            markeredgecolor="#00ccff", markeredgewidth=2,
                            zorder=10)
    # 거리 차트: 현재 스텝 강조 선
    vline_dist = ax_dist.axvline(x=-1, color="#00ccff", linewidth=1.5, alpha=0.8)
    # RMSE 차트: 현재 스텝 강조 막대
    highlight_bars = ax_err.bar(range(CHUNK_SIZE), rmse_per_step,
                                color="#00ccff", alpha=0.0, width=0.7)
    # 스텝 정보 텍스트
    step_text = ax3d.text2D(
        0.02, 0.97, "", transform=ax3d.transAxes,
        color="white", fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="#000000aa", edgecolor="#444444")
    )

    # ── 4. 애니메이션 콜백 ─────────────────────────────
    PAUSE_FRAMES = 8   # 앞뒤로 잠시 멈추는 프레임 수
    total_frames = PAUSE_FRAMES + CHUNK_SIZE + PAUSE_FRAMES

    def update(frame):
        # 앞 일시정지
        if frame < PAUSE_FRAMES:
            idx = 0
        # 실행 구간
        elif frame < PAUSE_FRAMES + CHUNK_SIZE:
            idx = frame - PAUSE_FRAMES
        # 뒤 일시정지
        else:
            idx = CHUNK_SIZE - 1

        # 3D 궤적
        xs = chunk[:idx+1, 0]
        ys = chunk[:idx+1, 1]
        zs = chunk[:idx+1, 2]
        traj_line.set_data(xs, ys)
        traj_line.set_3d_properties(zs)

        gripper_pt.set_data([chunk[idx,0]], [chunk[idx,1]])
        gripper_pt.set_3d_properties([chunk[idx,2]])

        # 거리 차트 수직선
        vline_dist.set_xdata([idx, idx])

        # RMSE 차트 강조
        for bi, bar in enumerate(highlight_bars):
            bar.set_alpha(0.85 if bi == idx else 0.0)

        # 스텝 텍스트
        dist_now = dist_to_goal[idx]
        err_now  = rmse_per_step[idx]
        step_text.set_text(
            f"Step {idx+1:2d} / {CHUNK_SIZE}\n"
            f"pos  : ({chunk[idx,0]:.3f}, {chunk[idx,1]:.3f}, {chunk[idx,2]:.3f})\n"
            f"dist : {dist_now:.4f} m\n"
            f"err  : {err_now:.4f} m"
        )
        return traj_line, gripper_pt, vline_dist, step_text

    anim = FuncAnimation(fig, update,
                         frames=total_frames,
                         interval=160, blit=False, repeat=True)

    # ── 5. 저장 ──────────────────────────────────────
    print("GIF 저장 중...")
    anim.save(SAVE_GIF, writer="pillow", fps=7, dpi=120)
    print(f"[saved] {SAVE_GIF}")

    # 마지막 프레임 PNG
    update(PAUSE_FRAMES + CHUNK_SIZE - 1)
    plt.savefig(SAVE_PNG, dpi=150, facecolor=fig.get_facecolor())
    print(f"[saved] {SAVE_PNG}")

    plt.show()


if __name__ == "__main__":
    main()