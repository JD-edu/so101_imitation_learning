"""
아이디어 E — Single-Shot vs Receding Horizon 나란히 비교
=========================================================
왼쪽: Single-Shot  (추론 1회, chunk 20개 전부 실행)
오른쪽: Receding Horizon (k스텝마다 재예측, 앞 k개만 실행)

시각화 요소
  · 3D 궤적 (파란선: 지나온 길, 청록 점선: 현재 chunk 미리보기)
  · 재예측 시점마다 수직 마커(주황) + "Re-plan!" 텍스트
  · 추론 횟수 카운터 (실시간)
  · 오른쪽 패널: 누적 오차(RMSE) 곡선 비교
  · 하단: 교육 포인트 텍스트
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D   # noqa
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE    = 20
K             = 5          # Receding Horizon: k스텝마다 재예측
WORKSPACE_MIN = np.array([0.0, -0.3, 0.0])
WORKSPACE_MAX = np.array([0.5,  0.3, 0.4])
MODEL_PATH    = "./act_policy.pt"
SAVE_GIF      = "./viz_E_comparison.gif"
SAVE_PNG      = "./viz_E_final_frame.png"

# ─────────────────────────────────────────
# 모델
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
    def predict(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        obs = torch.tensor(np.concatenate([start, target]),
                           dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return self(obs).squeeze(0).cpu().numpy().reshape(CHUNK_SIZE, 3)


def load_model(path):
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model = ActionChunkingPolicy(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(DEVICE).eval()


# ─────────────────────────────────────────
# 궤적 생성 (GT)
# ─────────────────────────────────────────
def make_gt(start, target, n=CHUNK_SIZE, curv=0.15):
    t      = np.linspace(0, 1, n).reshape(-1, 1)
    linear = start + t * (target - start)
    d      = target - start
    perp   = np.cross(d, [0,0,1]) if abs(d[2]) < 1e-6 else np.cross(d, [1,0,0])
    nm     = np.linalg.norm(perp)
    perp   = perp/nm if nm > 1e-6 else np.array([0,0,1])
    return linear + curv * 4 * t * (1-t) * perp.reshape(1,3)


# ─────────────────────────────────────────
# 두 방식의 궤적 시뮬레이션
# ─────────────────────────────────────────
def simulate_single_shot(model, start, target):
    """
    추론 1회 → waypoint 20개 전부 확정 → 순서대로 실행
    반환: steps (리스트), n_inferences (int)
    """
    chunk  = model.predict(start, target)   # 1회 추론
    steps  = list(chunk)                    # 20개 전부
    return steps, 1, [chunk] * len(steps)   # (위치목록, 추론횟수, 각 스텝의 active chunk)


def simulate_receding_horizon(model, start, target, k=K):
    """
    k스텝마다 재예측
    반환: steps, replan_indices, n_inferences, active_chunks
    """
    steps          = []
    replan_indices = []      # 재예측이 일어난 스텝 인덱스
    active_chunks  = []      # 각 스텝에서 사용된 chunk
    n_inferences   = 0
    current_pos    = start.copy()
    MAX_STEPS      = CHUNK_SIZE * 4   # 무한루프 방지

    while len(steps) < MAX_STEPS:
        # ── 재예측 ─────────────────────────────────────────────
        replan_indices.append(len(steps))
        chunk        = model.predict(current_pos, target)
        n_inferences += 1

        # ── 앞 k개만 실행 ───────────────────────────────────────
        for i in range(k):
            if len(steps) >= MAX_STEPS:
                break
            wp = chunk[i]
            steps.append(wp.copy())
            active_chunks.append(chunk.copy())
            current_pos = wp.copy()

            # 목표 도달 체크
            if np.linalg.norm(current_pos - target) < 0.03:
                return steps, replan_indices, n_inferences, active_chunks

    return steps, replan_indices, n_inferences, active_chunks


# ─────────────────────────────────────────
# 메인 시각화
# ─────────────────────────────────────────
def main():
    print("모델 로드 중...")
    model = load_model(MODEL_PATH)

    rng    = np.random.default_rng(7)
    start  = rng.uniform(WORKSPACE_MIN + 0.05, WORKSPACE_MAX - 0.05)
    target = rng.uniform(WORKSPACE_MIN + 0.05, WORKSPACE_MAX - 0.05)
    while np.linalg.norm(target - start) < 0.25:
        target = rng.uniform(WORKSPACE_MIN + 0.05, WORKSPACE_MAX - 0.05)

    print(f"  시작점: {start.round(3)}")
    print(f"  목표점: {target.round(3)}")
    gt = make_gt(start, target)

    # ── 시뮬레이션 ──────────────────────────────────────────────
    print("\nSingle-Shot 시뮬레이션...")
    ss_steps, ss_ninf, ss_chunks = simulate_single_shot(model, start, target)
    ss_pos = np.array(ss_steps)           # (20, 3)

    print("Receding Horizon 시뮬레이션...")
    rh_steps, rh_replan, rh_ninf, rh_chunks = simulate_receding_horizon(
        model, start, target, k=K)
    rh_pos = np.array(rh_steps)           # (N, 3)

    print(f"\n  Single-Shot  : 추론 {ss_ninf}회, 총 {len(ss_steps)}스텝")
    print(f"  Receding(k={K}): 추론 {rh_ninf}회, 총 {len(rh_steps)}스텝")

    # GT 대비 누적 RMSE 계산
    def cumulative_rmse(pos_arr, ref_arr):
        """각 스텝까지의 누적 RMSE (ref는 동일 길이로 자르거나 보간)"""
        n    = len(pos_arr)
        rmse = []
        for i in range(1, n+1):
            t_idx = np.linspace(0, len(ref_arr)-1, i).astype(int)
            diff  = pos_arr[:i] - ref_arr[t_idx]
            rmse.append(float(np.sqrt(np.mean(diff**2))))
        return rmse

    ss_rmse = cumulative_rmse(ss_pos, gt)
    rh_rmse = cumulative_rmse(rh_pos, gt)

    # ── 레이아웃 ──────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7.5), facecolor="#0d0d0d")
    fig.suptitle(
        "Single-Shot (추론 1회)  vs  Receding Horizon (k=5스텝마다 재예측)",
        color="white", fontsize=13, fontweight="bold", y=0.97
    )

    gs = gridspec.GridSpec(
        2, 3,
        left=0.04, right=0.98, top=0.91, bottom=0.09,
        wspace=0.28, hspace=0.40,
        width_ratios=[2, 2, 1.4]
    )

    def make_3d_ax(pos, title):
        ax = fig.add_subplot(pos, projection='3d')
        ax.set_facecolor("#111111")
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#2a2a2a")
        ax.set_xlim(WORKSPACE_MIN[0], WORKSPACE_MAX[0])
        ax.set_ylim(WORKSPACE_MIN[1], WORKSPACE_MAX[1])
        ax.set_zlim(WORKSPACE_MIN[2], WORKSPACE_MAX[2])
        ax.set_xlabel("x", color="#666", labelpad=3, fontsize=8)
        ax.set_ylabel("y", color="#666", labelpad=3, fontsize=8)
        ax.set_zlabel("z", color="#666", labelpad=3, fontsize=8)
        ax.tick_params(colors="#444", labelsize=6)
        ax.set_title(title, color="white", fontsize=10, pad=5)
        # GT 참조
        ax.plot(gt[:,0], gt[:,1], gt[:,2],
                '--', color="#333333", lw=1.0, label="GT 참조")
        # 전체 예측 경로 미리보기 (매우 연하게)
        return ax

    ax_ss = make_3d_ax(gs[0, 0], f"Single-Shot  |  추론 횟수: 0")
    ax_rh = make_3d_ax(gs[0, 1], f"Receding Horizon (k={K})  |  추론 횟수: 0")

    # Single-Shot: 전체 chunk 미리보기
    ax_ss.plot(ss_pos[:,0], ss_pos[:,1], ss_pos[:,2],
               ':', color="#00ffcc", lw=0.8, alpha=0.25, label="chunk 미리보기")
    # 시작 · 목표
    for ax in [ax_ss, ax_rh]:
        ax.scatter(*start,  color="#44ff44", s=80,  marker="s", zorder=5)
        ax.scatter(*target, color="#ff4444", s=130, marker="*", zorder=5)
        ax.legend(loc="upper left", fontsize=6,
                  facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")

    # ── 오른쪽 RMSE 패널 ─────────────────────────────────────
    ax_rmse = fig.add_subplot(gs[:, 2])
    ax_rmse.set_facecolor("#111111")
    ax_rmse.set_title("누적 RMSE 비교", color="white", fontsize=10)
    ax_rmse.set_xlabel("실행 스텝", color="#aaa", fontsize=9)
    ax_rmse.set_ylabel("누적 RMSE (m)", color="#aaa", fontsize=9)
    ax_rmse.tick_params(colors="#555")
    ax_rmse.spines[:].set_edgecolor("#333")

    # 완성 곡선 (배경 참조용, 매우 연하게)
    max_step = max(len(ss_rmse), len(rh_rmse))
    ax_rmse.plot(range(1, len(ss_rmse)+1), ss_rmse,
                 color="#00ccff", lw=0.8, alpha=0.15)
    ax_rmse.plot(range(1, len(rh_rmse)+1), rh_rmse,
                 color="#ff9944", lw=0.8, alpha=0.15)
    ax_rmse.set_xlim(0, max_step + 1)
    ax_rmse.set_ylim(0, max(max(ss_rmse), max(rh_rmse)) * 1.15)

    # 실시간 업데이트 선 (초기 빈 선)
    line_ss_rmse, = ax_rmse.plot([], [], '-', color="#00ccff", lw=2.0,
                                  label=f"Single-Shot (추론 {ss_ninf}회)")
    line_rh_rmse, = ax_rmse.plot([], [], '-', color="#ff9944", lw=2.0,
                                  label=f"Receding (추론 {rh_ninf}회)")
    ax_rmse.legend(fontsize=8, facecolor="#1a1a1a",
                   edgecolor="#444", labelcolor="white")

    # 재예측 마커선 (Receding Horizon RMSE 차트)
    for ri in rh_replan[1:]:   # 첫 번째는 시작이므로 제외
        if ri < len(rh_rmse):
            ax_rmse.axvline(x=ri, color="#ff9944", lw=0.6,
                            alpha=0.3, linestyle=":")

    # ── 이동 궤적 요소 ────────────────────────────────────────
    # Single-Shot
    ss_traj, = ax_ss.plot([], [], [], '-',  color="#00ccff", lw=2.2, alpha=0.9)
    ss_grip, = ax_ss.plot([], [], [], 'o',  color="white", markersize=9,
                          markeredgecolor="#00ccff", markeredgewidth=2, zorder=10)
    ss_info  = ax_ss.text2D(0.02, 0.97, "", transform=ax_ss.transAxes,
                            color="white", fontsize=8, va="top",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="#000000bb", edgecolor="#444"))

    # Receding Horizon
    rh_traj,     = ax_rh.plot([], [], [], '-',  color="#ff9944", lw=2.2, alpha=0.9)
    rh_grip,     = ax_rh.plot([], [], [], 'o',  color="white", markersize=9,
                              markeredgecolor="#ff9944", markeredgewidth=2, zorder=10)
    rh_preview,  = ax_rh.plot([], [], [], ':',  color="#ff9944", lw=0.8, alpha=0.3)
    rh_info      = ax_rh.text2D(0.02, 0.97, "", transform=ax_rh.transAxes,
                                color="white", fontsize=8, va="top",
                                bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor="#000000bb", edgecolor="#444"))
    rh_replan_txt = ax_rh.text2D(0.50, 0.90, "", transform=ax_rh.transAxes,
                                 color="#ff9944", fontsize=10, va="top",
                                 ha="center", fontweight="bold")

    # 교육 포인트
    edu = (
        "💡 Receding Horizon: k스텝마다 현재 위치로 재예측 → "
        "오차 누적 방지  |  chunk 경계(주황 점선)마다 새 경로 계획  |  "
        "실제 ACT 기본값 k=5"
    )
    fig.text(0.5, 0.02, edu, ha="center", va="bottom", fontsize=8.5,
             color="#aaaaaa",
             bbox=dict(boxstyle="round,pad=0.35",
                       facecolor="#1a1a1a", edgecolor="#444"))

    # ── 애니메이션 ────────────────────────────────────────────
    PAUSE    = 6
    n_ss     = len(ss_steps)
    n_rh     = len(rh_steps)
    n_common = max(n_ss, n_rh)
    total    = PAUSE + n_common + PAUSE

    def update(frame):
        # 앞뒤 일시정지 처리
        if frame < PAUSE:
            fi = 0
        elif frame < PAUSE + n_common:
            fi = frame - PAUSE
        else:
            fi = n_common - 1

        ss_idx = min(fi, n_ss - 1)
        rh_idx = min(fi, n_rh - 1)

        # ── Single-Shot 3D ──────────────────────────────
        ss_traj.set_data(ss_pos[:ss_idx+1, 0], ss_pos[:ss_idx+1, 1])
        ss_traj.set_3d_properties(ss_pos[:ss_idx+1, 2])
        ss_grip.set_data([ss_pos[ss_idx, 0]], [ss_pos[ss_idx, 1]])
        ss_grip.set_3d_properties([ss_pos[ss_idx, 2]])
        dist_ss = np.linalg.norm(ss_pos[ss_idx] - target)
        ax_ss.set_title(f"Single-Shot  |  추론 횟수: {ss_ninf}회",
                        color="white", fontsize=10, pad=5)
        ss_info.set_text(
            f"Step {ss_idx+1:2d}/{n_ss}\n"
            f"dist: {dist_ss:.4f} m\n"
            f"추론: {ss_ninf}회 (고정)"
        )

        # ── Receding Horizon 3D ─────────────────────────
        rh_cur = rh_pos[rh_idx]
        rh_traj.set_data(rh_pos[:rh_idx+1, 0], rh_pos[:rh_idx+1, 1])
        rh_traj.set_3d_properties(rh_pos[:rh_idx+1, 2])
        rh_grip.set_data([rh_cur[0]], [rh_cur[1]])
        rh_grip.set_3d_properties([rh_cur[2]])

        # 현재 active chunk 미리보기
        cur_chunk = rh_chunks[rh_idx]
        rh_preview.set_data(cur_chunk[:,0], cur_chunk[:,1])
        rh_preview.set_3d_properties(cur_chunk[:,2])

        # 재예측 시점 체크 — 현재 스텝이 replan 직후인지
        is_replan = rh_idx in rh_replan
        infer_count = sum(1 for r in rh_replan if r <= rh_idx)
        dist_rh = np.linalg.norm(rh_cur - target)

        ax_rh.set_title(
            f"Receding Horizon (k={K})  |  추론 횟수: {infer_count}회",
            color="white", fontsize=10, pad=5
        )
        rh_info.set_text(
            f"Step {rh_idx+1:2d}/{n_rh}\n"
            f"dist: {dist_rh:.4f} m\n"
            f"추론: {infer_count}회 (재예측)"
        )
        rh_replan_txt.set_text("🔄 Re-plan!" if is_replan and fi > 0 else "")

        # ── RMSE 그래프 실시간 업데이트 ─────────────────
        line_ss_rmse.set_data(range(1, ss_idx+2), ss_rmse[:ss_idx+1])
        line_rh_rmse.set_data(range(1, rh_idx+2), rh_rmse[:rh_idx+1])

        return (ss_traj, ss_grip, rh_traj, rh_grip,
                rh_preview, rh_replan_txt,
                line_ss_rmse, line_rh_rmse,
                ss_info, rh_info)

    anim = FuncAnimation(fig, update, frames=total,
                         interval=170, blit=False, repeat=True)

    print("\nGIF 저장 중 (잠시 기다려 주세요)...")
    anim.save(SAVE_GIF, writer="pillow", fps=7, dpi=120)
    print(f"[saved] {SAVE_GIF}")

    # 마지막 프레임 PNG
    update(PAUSE + n_common - 1)
    plt.savefig(SAVE_PNG, dpi=150, facecolor=fig.get_facecolor())
    print(f"[saved] {SAVE_PNG}")


if __name__ == "__main__":
    main()