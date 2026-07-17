"""
ACT Lite — PyTorch MLP 버전 (교육용)
=====================================
- sklearn MLP → PyTorch nn.Module로 완전 교체
- 나중에 Transformer 교체가 쉽도록 Policy 클래스 구조화
- 모델 저장/불러오기 (state_dict .pt 방식)
- 학습 loss 곡선 시각화 포함
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa
import os, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. 디바이스 설정
# ─────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 20
WORKSPACE_MIN = np.array([0.0, -0.3, 0.0])
WORKSPACE_MAX = np.array([0.5,  0.3, 0.4])

print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────
# 1. 궤적 데이터 생성 (동일 로직)
# ─────────────────────────────────────────────
def generate_trajectory(start, target, n_steps=CHUNK_SIZE, curvature=0.15):
    t = np.linspace(0, 1, n_steps).reshape(-1, 1)
    linear = start + t * (target - start)
    direction = target - start
    if abs(direction[2]) < 1e-6:
        perp = np.cross(direction, np.array([0, 0, 1]))
    else:
        perp = np.cross(direction, np.array([1, 0, 0]))
    norm = np.linalg.norm(perp)
    perp = perp / norm if norm > 1e-6 else np.array([0, 0, 1])
    offset = curvature * 4 * t * (1 - t) * perp.reshape(1, 3)
    return linear + offset  # (n_steps, 3)


def generate_dataset(n_samples=2000, seed=42):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_samples):
        s = rng.uniform(WORKSPACE_MIN, WORKSPACE_MAX)
        t = rng.uniform(WORKSPACE_MIN, WORKSPACE_MAX)
        if np.linalg.norm(t - s) < 0.05:
            continue
        traj = generate_trajectory(s, t, CHUNK_SIZE,
                                   curvature=rng.uniform(0.05, 0.25))
        X.append(np.concatenate([s, t]))
        Y.append(traj.flatten())
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# ─────────────────────────────────────────────
# 2. PyTorch 모델 정의
#    ★ 나중에 이 클래스만 Transformer로 바꾸면 ACT 완성!
# ─────────────────────────────────────────────
class ActionChunkingPolicy(nn.Module):
    """
    입력: (batch, obs_dim)   ← 현재위치 + 목표위치 = 6차원
    출력: (batch, chunk_dim) ← CHUNK_SIZE * 3 = 60차원 (action chunk)

    나중에 Transformer 교체 포인트:
        self.net = TransformerEncoder(...)  # ← 이 줄만 바꾸면 됨
    """
    def __init__(self,
                 obs_dim   : int = 6,
                 chunk_dim : int = CHUNK_SIZE * 3,
                 hidden    : int = 256,
                 n_layers  : int = 3,
                 dropout   : float = 0.1):
        super().__init__()

        # ── MLP backbone ──────────────────────────────────────────
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),   # BatchNorm 대신 LayerNorm (Transformer 친화적)
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        layers.append(nn.Linear(hidden, chunk_dim))  # 출력층
        self.net = nn.Sequential(*layers)

        # ── 가중치 초기화 (He init) ───────────────────────────────
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs   : (batch, 6)    — [start_xyz | target_xyz]
        return: (batch, 60)   — action chunk (20 waypoints × 3D)
        """
        return self.net(obs)

    # ── 인퍼런스 헬퍼 (numpy in / numpy out) ─────────────────────
    @torch.no_grad()
    def predict(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        start  : (3,)
        target : (3,)
        return : (CHUNK_SIZE, 3)  — predicted waypoints
        """
        obs = np.concatenate([start, target]).astype(np.float32)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)  # (1, 6)
        chunk = self(obs_t).squeeze(0).cpu().numpy()            # (60,)
        return chunk.reshape(CHUNK_SIZE, 3)


# ─────────────────────────────────────────────
# 3. 학습 루프
# ─────────────────────────────────────────────
def train(model, X, Y,
          epochs=200, batch_size=64, lr=1e-3):
    """
    Returns
    -------
    loss_history : list[float]  학습 epoch별 MSE loss
    """
    # Tensor 변환 → DataLoader
    ds     = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    model.to(DEVICE)
    model.train()
    loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            pred = model(x_batch)           # (batch, 60)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        loss_history.append(epoch_loss)
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}]  "
                  f"MSE Loss = {epoch_loss:.6f}  "
                  f"LR = {scheduler.get_last_lr()[0]:.6f}")

    return loss_history


# ─────────────────────────────────────────────
# 4. 모델 저장 / 불러오기 (.pt state_dict)
# ─────────────────────────────────────────────
def save_model(model, path="act_policy.pt"):
    """
    저장 내용:
      - state_dict : 가중치 (필수)
      - config     : 모델 하이퍼파라미터 (재생성에 필요)
    """
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "obs_dim"   : 6,
            "chunk_dim" : CHUNK_SIZE * 3,
            "hidden"    : 256,
            "n_layers"  : 3,
            "dropout"   : 0.1,
        }
    }, path)
    print(f"[saved] {path}  ({os.path.getsize(path)/1024:.1f} KB)")


def load_model(path="act_policy.pt") -> ActionChunkingPolicy:
    """
    저장된 .pt 파일에서 모델 복원 후 eval 모드로 반환
    """
    ckpt   = torch.load(path, map_location=DEVICE)
    cfg    = ckpt["config"]
    model  = ActionChunkingPolicy(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    print(f"[loaded] {path}")
    return model


# ─────────────────────────────────────────────
# 5. 시각화
# ─────────────────────────────────────────────
def plot_loss_curve(loss_history):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_history, color='steelblue', linewidth=1.8)
    ax.set_xlabel("Epoch");  ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss Curve — Action Chunking MLP (PyTorch)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("./pt_loss.png", dpi=150)
    plt.show()
    print("[saved] pt_loss.png")


def plot_trajectories(model, n_show=4, seed=99):
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(14, 5))

    for i in range(n_show):
        start  = rng.uniform(WORKSPACE_MIN, WORKSPACE_MAX)
        target = rng.uniform(WORKSPACE_MIN, WORKSPACE_MAX)
        gt     = generate_trajectory(start, target, CHUNK_SIZE, 0.15)
        pred   = model.predict(start, target)
        rmse   = float(np.sqrt(np.mean((gt - pred) ** 2)))

        for j, (traj, title) in enumerate([
                (gt,   f"#{i+1} Ground Truth"),
                (pred, f"#{i+1} Predicted\n(RMSE={rmse:.4f} m)")]):
            ax = fig.add_subplot(2, n_show, j * n_show + i + 1, projection='3d')
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    'b-o', markersize=3)
            ax.scatter(*start,  color='green', s=80, marker='s', label='start')
            ax.scatter(*target, color='red',   s=80, marker='*', label='target')
            ax.set_xlim(*WORKSPACE_MIN[[0]], *WORKSPACE_MAX[[0]])
            ax.set_ylim(*WORKSPACE_MIN[[1]], *WORKSPACE_MAX[[1]])
            ax.set_zlim(*WORKSPACE_MIN[[2]], *WORKSPACE_MAX[[2]])
            ax.set_title(title, fontsize=8)
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
            if i == 0 and j == 0:
                ax.legend(fontsize=7)

    plt.suptitle("Action Chunking (PyTorch MLP) — Ground Truth vs Predicted",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("./pt_static.png", dpi=150)
    plt.show()
    print("[saved] pt_static.png")


def animate_gripper(model, seed=7):
    rng    = np.random.default_rng(seed)
    start  = rng.uniform(WORKSPACE_MIN, WORKSPACE_MAX)
    target = rng.uniform(WORKSPACE_MIN, WORKSPACE_MAX)
    traj   = model.predict(start, target)   # (20, 3)

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlim(WORKSPACE_MIN[0], WORKSPACE_MAX[0])
    ax.set_ylim(WORKSPACE_MIN[1], WORKSPACE_MAX[1])
    ax.set_zlim(WORKSPACE_MIN[2], WORKSPACE_MAX[2])
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title("Gripper Animation — PyTorch Action Chunking", fontweight='bold')

    ax.scatter(*target, color='red',   s=150, marker='*', label='target (object)')
    ax.scatter(*start,  color='green', s=100, marker='s', label='start (gripper)')

    line,    = ax.plot([], [], [], 'b-', alpha=0.5, lw=1.8)
    gripper, = ax.plot([], [], [], 'ko', markersize=11, zorder=10)
    ax.legend(loc='upper left', fontsize=9)

    def init():
        line.set_data([], []); line.set_3d_properties([])
        gripper.set_data([], []); gripper.set_3d_properties([])
        return line, gripper

    def update(frame):
        idx = min(frame, CHUNK_SIZE - 1)
        line.set_data(traj[:idx+1, 0], traj[:idx+1, 1])
        line.set_3d_properties(traj[:idx+1, 2])
        gripper.set_data([traj[idx, 0]], [traj[idx, 1]])
        gripper.set_3d_properties([traj[idx, 2]])
        return line, gripper

    anim = FuncAnimation(fig, update, frames=CHUNK_SIZE + 5,
                         init_func=init, blit=False, interval=150, repeat=True)
    anim.save("./pt_animation.gif", writer='pillow', fps=8)
    plt.show()
    print("[saved] pt_animation.gif")


# ─────────────────────────────────────────────
# 6. 메인
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ACT Lite — PyTorch MLP 버전")
    print("=" * 60)

    # 데이터
    print("\n[1/5] 데이터 생성...")
    X, Y = generate_dataset(n_samples=2000)
    X_te, Y_te = generate_dataset(n_samples=300, seed=999)
    print(f"  Train: X={X.shape}, Y={Y.shape}")

    # 모델 생성
    print("\n[2/5] 모델 생성...")
    policy = ActionChunkingPolicy(obs_dim=6, chunk_dim=CHUNK_SIZE*3,
                                  hidden=256, n_layers=3, dropout=0.1)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  파라미터 수: {n_params:,}  ({n_params/1e3:.1f}K)")
    print(policy)

    # 학습
    print("\n[3/5] 학습...")
    history = train(policy, X, Y, epochs=200, batch_size=64, lr=1e-3)

    # 테스트 RMSE
    policy.eval()
    with torch.no_grad():
        X_te_t = torch.from_numpy(X_te).to(DEVICE)
        Y_pred = policy(X_te_t).cpu().numpy()
    rmse = float(np.sqrt(np.mean((Y_te - Y_pred) ** 2)))
    print(f"\n  테스트 RMSE: {rmse:.6f} m  ({rmse*1000:.2f} mm)")

    # 저장 & 불러오기 테스트
    print("\n[4/5] 모델 저장/불러오기...")
    save_model(policy, "./act_policy.pt")
    policy_loaded = load_model("./act_policy.pt")
    print(f"  저장 후 재로드 성공! eval mode: {not policy_loaded.training}")

    # 시각화
    print("\n[5/5] 시각화...")
    plot_loss_curve(history)
    plot_trajectories(policy_loaded)
    animate_gripper(policy_loaded)

    print("\n모두 완료!")