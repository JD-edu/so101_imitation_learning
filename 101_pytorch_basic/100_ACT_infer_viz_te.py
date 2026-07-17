"""
Temporal Ensemble 구현 & 시각화
=================================
세 가지 방식을 나란히 비교:
  Left  : Single-Shot      (추론 1회, chunk 20개 전부 실행)
  Center: Receding Horizon (k=5마다 재예측, 경계 불연속 있음)
  Right : Temporal Ensemble(k=1마다 재예측 + 겹치는 chunk 가중 평균)

Temporal Ensemble 수식
  실행값(t) = Σ_{i=0}^{min(t, W-1)} w_i * chunk_{t-i}[i]
  w_i = exp(-i * λ)  (최신 chunk에 높은 가중치, λ=0.1)
  W   = 창(window) 크기 = 사용할 최근 chunk 수

시각화 요소
  · 3D 궤적 애니메이션
  · 오른쪽 패널 1: 누적 RMSE 3방식 비교 곡선
  · 오른쪽 패널 2: Temporal Ensemble 가중치 막대 (실시간)
  · 하단: 교육 포인트 텍스트
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D   # noqa
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE    = 20
K_RH          = 5        # Receding Horizon: 매 k스텝마다 재예측
K_TE          = 1        # Temporal Ensemble: 매 1스텝마다 재예측 (매 스텝)
TE_WINDOW     = 8        # Temporal Ensemble: 최근 몇 개 chunk를 평균할지
TE_LAMBDA     = 0.15     # 가중치 감쇠 계수 (클수록 최신 chunk 가중치 집중)
WORKSPACE_MIN = np.array([0.0, -0.3, 0.0])
WORKSPACE_MAX = np.array([0.5,  0.3, 0.4])
MODEL_PATH    = "./act_policy.pt"
SAVE_GIF      = "./viz_TE_comparison.gif"
SAVE_PNG      = "./viz_TE_final_frame.png"


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
        obs = torch.tensor(
            np.concatenate([start, target]), dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)
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
    perp   = perp / nm if nm > 1e-6 else np.array([0,0,1])
    return linear + curv * 4 * t * (1-t) * perp.reshape(1,3)


# ─────────────────────────────────────────
# 시뮬레이션 함수들
# ─────────────────────────────────────────
def simulate_single_shot(model, start, target):
    """추론 1회 → chunk 20개 전부 실행"""
    chunk = model.predict(start, target)
    steps = list(chunk)
    return np.array(steps), 1, [chunk]*len(steps)


def simulate_receding_horizon(model, start, target, k=K_RH):
    """k스텝마다 재예측, 앞 k개만 실행"""
    steps, replan_idx, active_chunks = [], [], []
    n_infer, current_pos = 0, start.copy()
    MAX = CHUNK_SIZE * 4

    while len(steps) < MAX:
        replan_idx.append(len(steps))
        chunk   = model.predict(current_pos, target)
        n_infer += 1
        for i in range(k):
            if len(steps) >= MAX: break
            wp = chunk[i]
            steps.append(wp.copy())
            active_chunks.append(chunk.copy())
            current_pos = wp.copy()
            if np.linalg.norm(current_pos - target) < 0.03:
                return np.array(steps), n_infer, replan_idx, active_chunks
    return np.array(steps), n_infer, replan_idx, active_chunks


def simulate_temporal_ensemble(model, start, target,
                                window=TE_WINDOW, lam=TE_LAMBDA):
    """
    Temporal Ensemble:
      - 매 스텝 재예측 (k=1)
      - 최근 window개 chunk를 지수 가중 평균으로 합산
      - 가중치: w_i = exp(-i * lam)  (i=0이 가장 최신)

    반환
    ----
    steps        : (N, 3)  실제 실행된 위치
    n_infer      : 추론 횟수
    chunk_buffer : 각 스텝의 최근 chunk 목록 (시각화용)
    weight_history: 각 스텝의 가중치 벡터 (시각화용)
    """
    steps          = []
    chunk_buffer   = []   # 최근 window개 chunk 저장
    weight_history = []   # 각 스텝에서 사용한 가중치
    n_infer        = 0
    current_pos    = start.copy()
    MAX            = CHUNK_SIZE * 4

    while len(steps) < MAX:
        # ── 매 스텝 재예측 ───────────────────────────────
        new_chunk   = model.predict(current_pos, target)   # (CHUNK_SIZE, 3)
        n_infer    += 1

        # buffer 앞에 삽입 (index 0 = 가장 최신)
        chunk_buffer.insert(0, new_chunk)
        if len(chunk_buffer) > window:
            chunk_buffer.pop()           # 오래된 chunk 제거

        # ── 가중 평균으로 실행값 계산 ────────────────────
        # chunk_buffer[i] 는 i스텝 전에 예측한 chunk
        # chunk_buffer[i][i] 가 현재 스텝에 해당하는 waypoint
        # (i=0: 방금 예측한 chunk의 첫 번째 waypoint)
        n_buf   = len(chunk_buffer)
        raw_w   = np.array([np.exp(-i * lam) for i in range(n_buf)])
        weights = raw_w / raw_w.sum()             # 정규화

        weighted_wp = np.zeros(3)
        for i, (ch, w) in enumerate(zip(chunk_buffer, weights)):
            # i스텝 전에 예측한 chunk에서 i번째 waypoint가 현재 스텝 해당
            if i < len(ch):
                weighted_wp += w * ch[i]

        weight_history.append(weights.copy())

        steps.append(weighted_wp.copy())
        current_pos = weighted_wp.copy()

        if np.linalg.norm(current_pos - target) < 0.03:
            break

    return np.array(steps), n_infer, chunk_buffer, weight_history


# ─────────────────────────────────────────
# 누적 RMSE 계산
# ─────────────────────────────────────────
def cumulative_rmse(pos_arr, ref_arr):
    rmse = []
    for i in range(1, len(pos_arr)+1):
        t_idx = np.linspace(0, len(ref_arr)-1, i).astype(int)
        diff  = pos_arr[:i] - ref_arr[t_idx]
        rmse.append(float(np.sqrt(np.mean(diff**2))))
    return rmse


# ─────────────────────────────────────────
# 메인
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

    # ── 시뮬레이션 ──────────────────────────────────────────
    print("\nSingle-Shot 시뮬레이션...")
    ss_pos, ss_ninf, _            = simulate_single_shot(model, start, target)

    print("Receding Horizon 시뮬레이션...")
    rh_pos, rh_ninf, rh_replan, _ = simulate_receding_horizon(model, start, target)

    print("Temporal Ensemble 시뮬레이션...")
    te_pos, te_ninf, _, te_weights = simulate_temporal_ensemble(model, start, target)

    print(f"\n  Single-Shot      : 추론 {ss_ninf:2d}회, {len(ss_pos):2d}스텝")
    print(f"  Receding(k={K_RH})    : 추론 {rh_ninf:2d}회, {len(rh_pos):2d}스텝")
    print(f"  Temporal Ensemble: 추론 {te_ninf:2d}회, {len(te_pos):2d}스텝")

    ss_rmse = cumulative_rmse(ss_pos, gt)
    rh_rmse = cumulative_rmse(rh_pos, gt)
    te_rmse = cumulative_rmse(te_pos, gt)

    final_ss = ss_rmse[-1]
    final_rh = rh_rmse[-1]
    final_te = te_rmse[-1]
    print(f"\n  최종 누적 RMSE:")
    print(f"    Single-Shot      : {final_ss:.4f} m")
    print(f"    Receding Horizon : {final_rh:.4f} m")
    print(f"    Temporal Ensemble: {final_te:.4f} m")

    # ─────────────────────────────────────────
    # 레이아웃
    # ─────────────────────────────────────────
    fig = plt.figure(figsize=(16, 8.5), facecolor="#0d0d0d")
    fig.suptitle(
        "Single-Shot  vs  Receding Horizon  vs  Temporal Ensemble",
        color="white", fontsize=13, fontweight="bold", y=0.97
    )

    gs = gridspec.GridSpec(
        2, 4,
        left=0.03, right=0.98, top=0.92, bottom=0.09,
        wspace=0.30, hspace=0.42,
        width_ratios=[2, 2, 2, 1.6]
    )

    # ── 3D 축 공통 설정 ────────────────────────────────────
    def make_ax(subplot_spec, title, color):
        ax = fig.add_subplot(subplot_spec, projection='3d')
        ax.set_facecolor("#111111")
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#2a2a2a")
        ax.set_xlim(*WORKSPACE_MIN[[0]], *WORKSPACE_MAX[[0]])
        ax.set_ylim(*WORKSPACE_MIN[[1]], *WORKSPACE_MAX[[1]])
        ax.set_zlim(*WORKSPACE_MIN[[2]], *WORKSPACE_MAX[[2]])
        for axis, label in zip([ax.set_xlabel, ax.set_ylabel, ax.set_zlabel],
                               ["x","y","z"]):
            axis(label, color="#555", labelpad=2, fontsize=7)
        ax.tick_params(colors="#333", labelsize=5)
        ax.set_title(title, color=color, fontsize=9, pad=4, fontweight="bold")
        ax.plot(gt[:,0], gt[:,1], gt[:,2],
                '--', color="#2a2a2a", lw=1.0)
        ax.scatter(*start,  color="#44ff44", s=70,  marker="s", zorder=5)
        ax.scatter(*target, color="#ff4444", s=120, marker="*", zorder=5)
        return ax

    ax_ss = make_ax(gs[:, 0], f"Single-Shot  |  추론 {ss_ninf}회", "#00ccff")
    ax_rh = make_ax(gs[:, 1], f"Receding Horizon (k={K_RH})  |  추론 {rh_ninf}회", "#ff9944")
    ax_te = make_ax(gs[:, 2], f"Temporal Ensemble (W={TE_WINDOW}, λ={TE_LAMBDA})  |  추론 {te_ninf}회", "#cc88ff")

    # Single-Shot chunk 미리보기
    ax_ss.plot(ss_pos[:,0], ss_pos[:,1], ss_pos[:,2],
               ':', color="#00ccff", lw=0.7, alpha=0.2)

    # ── 오른쪽 패널 ─────────────────────────────────────────
    # 위: RMSE 비교
    ax_rmse = fig.add_subplot(gs[0, 3])
    ax_rmse.set_facecolor("#111111")
    ax_rmse.set_title("누적 RMSE 비교", color="white", fontsize=9)
    ax_rmse.set_xlabel("실행 스텝", color="#888", fontsize=8)
    ax_rmse.set_ylabel("RMSE (m)",   color="#888", fontsize=8)
    ax_rmse.tick_params(colors="#555", labelsize=7)
    ax_rmse.spines[:].set_edgecolor("#333")
    max_step = max(len(ss_rmse), len(rh_rmse), len(te_rmse))
    max_val  = max(max(ss_rmse), max(rh_rmse), max(te_rmse))
    ax_rmse.set_xlim(0, max_step+1)
    ax_rmse.set_ylim(0, max_val * 1.15)

    # 배경 완성 곡선 (연하게)
    ax_rmse.plot(range(1, len(ss_rmse)+1), ss_rmse,
                 color="#00ccff", lw=0.6, alpha=0.15)
    ax_rmse.plot(range(1, len(rh_rmse)+1), rh_rmse,
                 color="#ff9944", lw=0.6, alpha=0.15)
    ax_rmse.plot(range(1, len(te_rmse)+1), te_rmse,
                 color="#cc88ff", lw=0.6, alpha=0.15)

    line_ss_r, = ax_rmse.plot([], [], color="#00ccff", lw=1.8,
                               label=f"SS ({ss_ninf}회)")
    line_rh_r, = ax_rmse.plot([], [], color="#ff9944", lw=1.8,
                               label=f"RH ({rh_ninf}회)")
    line_te_r, = ax_rmse.plot([], [], color="#cc88ff", lw=2.2,
                               label=f"TE ({te_ninf}회)")
    ax_rmse.legend(fontsize=7, facecolor="#1a1a1a",
                   edgecolor="#444", labelcolor="white")

    # Receding Horizon 재예측 구분선
    for ri in rh_replan[1:]:
        if ri < len(rh_rmse):
            ax_rmse.axvline(x=ri, color="#ff9944", lw=0.5, alpha=0.25, ls=":")

    # 아래: Temporal Ensemble 가중치 바
    ax_w = fig.add_subplot(gs[1, 3])
    ax_w.set_facecolor("#111111")
    ax_w.set_title(f"TE 가중치 (최근 {TE_WINDOW}개 chunk)", color="white", fontsize=9)
    ax_w.set_xlabel("chunk 인덱스 (0=최신)", color="#888", fontsize=8)
    ax_w.set_ylabel("가중치",                color="#888", fontsize=8)
    ax_w.tick_params(colors="#555", labelsize=7)
    ax_w.spines[:].set_edgecolor("#333")
    ax_w.set_xlim(-0.5, TE_WINDOW - 0.5)
    ax_w.set_ylim(0, 1.05)

    # 이론적 가중치 참조 (배경)
    ref_w_raw = np.array([np.exp(-i * TE_LAMBDA) for i in range(TE_WINDOW)])
    ref_w     = ref_w_raw / ref_w_raw.sum()
    ax_w.bar(range(TE_WINDOW), ref_w, color="#555555", alpha=0.3, width=0.6)

    # 실시간 가중치 막대
    bars_w = ax_w.bar(range(TE_WINDOW),
                      [0]*TE_WINDOW,
                      color="#cc88ff", alpha=0.85, width=0.6)
    w_text = ax_w.text(TE_WINDOW/2, 0.85, "",
                       ha="center", fontsize=8, color="#cc88ff")

    # ── 이동 요소 초기화 ─────────────────────────────────────
    def make_elements(ax, color):
        traj, = ax.plot([], [], [], '-',  color=color, lw=2.2, alpha=0.9)
        grip, = ax.plot([], [], [], 'o',  color="white", markersize=8,
                        markeredgecolor=color, markeredgewidth=2, zorder=10)
        info  = ax.text2D(0.02, 0.97, "", transform=ax.transAxes,
                          color="white", fontsize=7, va="top",
                          bbox=dict(boxstyle="round,pad=0.25",
                                    facecolor="#000000bb", edgecolor="#444"))
        return traj, grip, info

    ss_traj, ss_grip, ss_info = make_elements(ax_ss, "#00ccff")
    rh_traj, rh_grip, rh_info = make_elements(ax_rh, "#ff9944")
    te_traj, te_grip, te_info = make_elements(ax_te, "#cc88ff")

    # Receding Horizon 재예측 표시
    rh_replan_txt = ax_rh.text2D(0.50, 0.88, "", transform=ax_rh.transAxes,
                                  color="#ff9944", fontsize=9, ha="center",
                                  fontweight="bold")
    # Temporal Ensemble 평균 표시
    te_avg_txt = ax_te.text2D(0.50, 0.88, "", transform=ax_te.transAxes,
                               color="#cc88ff", fontsize=9, ha="center",
                               fontweight="bold")

    # 교육 포인트
    edu = (
        "💡 Temporal Ensemble: 매 스텝 재예측 + 최근 chunk들을 지수 가중 평균 "
        f"→ 경계 불연속 제거  |  W={TE_WINDOW}개 chunk 평균  |  λ={TE_LAMBDA} (작을수록 넓게 평균)"
    )
    fig.text(0.5, 0.02, edu, ha="center", va="bottom", fontsize=8.5,
             color="#aaaaaa",
             bbox=dict(boxstyle="round,pad=0.35",
                       facecolor="#1a1a1a", edgecolor="#444"))

    # ─────────────────────────────────────────
    # 애니메이션
    # ─────────────────────────────────────────
    PAUSE    = 6
    n_ss     = len(ss_pos)
    n_rh     = len(rh_pos)
    n_te     = len(te_pos)
    n_common = max(n_ss, n_rh, n_te)
    total    = PAUSE + n_common + PAUSE

    def update(frame):
        fi = max(0, min(frame - PAUSE, n_common - 1))

        si = min(fi, n_ss - 1)
        ri = min(fi, n_rh - 1)
        ti = min(fi, n_te - 1)

        # ── Single-Shot ─────────────────────────────────
        ss_traj.set_data(ss_pos[:si+1,0], ss_pos[:si+1,1])
        ss_traj.set_3d_properties(ss_pos[:si+1,2])
        ss_grip.set_data([ss_pos[si,0]], [ss_pos[si,1]])
        ss_grip.set_3d_properties([ss_pos[si,2]])
        ss_info.set_text(
            f"Step {si+1:2d}/{n_ss}\n"
            f"dist: {np.linalg.norm(ss_pos[si]-target):.3f}m\n"
            f"추론: {ss_ninf}회 고정"
        )

        # ── Receding Horizon ────────────────────────────
        rh_traj.set_data(rh_pos[:ri+1,0], rh_pos[:ri+1,1])
        rh_traj.set_3d_properties(rh_pos[:ri+1,2])
        rh_grip.set_data([rh_pos[ri,0]], [rh_pos[ri,1]])
        rh_grip.set_3d_properties([rh_pos[ri,2]])
        infer_rh = sum(1 for r in rh_replan if r <= ri)
        rh_info.set_text(
            f"Step {ri+1:2d}/{n_rh}\n"
            f"dist: {np.linalg.norm(rh_pos[ri]-target):.3f}m\n"
            f"추론: {infer_rh}회"
        )
        rh_replan_txt.set_text("🔄 Re-plan!" if (ri in rh_replan and fi > 0) else "")

        # ── Temporal Ensemble ───────────────────────────
        te_traj.set_data(te_pos[:ti+1,0], te_pos[:ti+1,1])
        te_traj.set_3d_properties(te_pos[:ti+1,2])
        te_grip.set_data([te_pos[ti,0]], [te_pos[ti,1]])
        te_grip.set_3d_properties([te_pos[ti,2]])
        te_info.set_text(
            f"Step {ti+1:2d}/{n_te}\n"
            f"dist: {np.linalg.norm(te_pos[ti]-target):.3f}m\n"
            f"추론: {ti+1}회"
        )
        te_avg_txt.set_text("⚖ 가중 평균 중" if fi > 0 else "")

        # 가중치 바 업데이트
        if ti < len(te_weights):
            w = te_weights[ti]
            n_w = len(w)
            for bi, bar in enumerate(bars_w):
                bar.set_height(w[bi] if bi < n_w else 0)
            w_text.set_text(f"최신 w₀={w[0]:.3f}")

        # RMSE 실시간 업데이트
        line_ss_r.set_data(range(1, si+2), ss_rmse[:si+1])
        line_rh_r.set_data(range(1, ri+2), rh_rmse[:ri+1])
        line_te_r.set_data(range(1, ti+2), te_rmse[:ti+1])

        return (ss_traj, ss_grip, rh_traj, rh_grip, te_traj, te_grip,
                line_ss_r, line_rh_r, line_te_r,
                rh_replan_txt, te_avg_txt, ss_info, rh_info, te_info,
                *bars_w)

    anim = FuncAnimation(fig, update, frames=total,
                         interval=170, blit=False, repeat=True)

    print("\nGIF 저장 중 (잠시 기다려 주세요)...")
    anim.save(SAVE_GIF, writer="pillow", fps=7, dpi=120)
    print(f"[saved] {SAVE_GIF}")

    update(PAUSE + n_common - 1)
    plt.savefig(SAVE_PNG, dpi=150, facecolor=fig.get_facecolor())
    print(f"[saved] {SAVE_PNG}")

    # ── 최종 수치 요약 ──────────────────────────────────────
    print("\n" + "="*55)
    print("  최종 결과 요약")
    print("="*55)
    print(f"  방식               추론횟수  스텝수  최종RMSE")
    print(f"  Single-Shot        {ss_ninf:5d}회  {n_ss:4d}개  {final_ss:.4f} m")
    print(f"  Receding(k={K_RH})      {rh_ninf:5d}회  {n_rh:4d}개  {final_rh:.4f} m")
    print(f"  Temporal Ensemble  {te_ninf:5d}회  {n_te:4d}개  {final_te:.4f} m")
    print("="*55)


if __name__ == "__main__":
    main()