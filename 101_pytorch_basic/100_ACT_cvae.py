"""
ACT Lite – CVAE (Conditional Variational Autoencoder) Full Implementation
=========================================================================
이 코드는 ACT의 핵심 구성 요소인 CVAE를 PyTorch로 구현합니다.

CVAE의 역할:
- 같은 (start, target)에서 다양한 행동(trajectory)을 생성
- 잠재 변수 z ~ N(0, I) 샘플링으로 multimodal behavior 표현
- Training: Encoder가 z를 추론 → Decoder가 복원
- Inference: z를 직접 샘플링 → Decoder가 trajectory 생성

구조:
  Encoder: [obs(6) + chunk(60)] → [256 → 256] → μ(32), log σ²(32)
  Decoder: [obs(6) + z(32)]     → [256 → 256 → 256] → chunk(60)
  Loss: MSE(recon) + β · KL[N(μ,σ²) || N(0,I)]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os

# ─── 하이퍼파라미터 ─────────────────────────────────────────────────
CHUNK_SIZE  = 20      # action chunk 길이 (waypoint 수)
OBS_DIM     = 6       # 입력 차원: start(3) + target(3)
CHUNK_DIM   = 60      # 출력 차원: CHUNK_SIZE * 3
LATENT_DIM  = 32      # 잠재 공간 차원
HIDDEN_DIM  = 256
BETA        = 1.0     # KL divergence 가중치
EPOCHS      = 150
LR          = 1e-3
BATCH_SIZE  = 128
N_DATA      = 3000    # 생성할 데이터 수

WORKSPACE_MIN = np.array([0.0, -0.3, 0.0])
WORKSPACE_MAX = np.array([0.5,  0.3, 0.4])

# ─── 데이터 생성 ────────────────────────────────────────────────────
def generate_trajectory(start, target, n=CHUNK_SIZE, curvature=0.15):
    """
    선형 보간 + 이차 곡선으로 waypoint 시퀀스 생성.
    curvature: 0이면 직선, ±0.3이면 큰 호(arc).
    반환: (n, 3) ndarray
    """
    tv     = np.linspace(0, 1, n).reshape(-1, 1)
    linear = start + tv * (target - start)
    d      = target - start
    perp   = np.cross(d, [0, 0, 1]) if abs(d[2]) < 1e-6 else np.cross(d, [1, 0, 0])
    nm     = np.linalg.norm(perp)
    perp   = perp / nm if nm > 1e-6 else np.array([0, 0, 1])
    return linear + curvature * 4 * tv * (1 - tv) * perp.reshape(1, 3)


def generate_dataset(n_samples=N_DATA, seed=42):
    """
    랜덤 (start, target) 쌍으로 trajectory 데이터셋 생성.
    curvature를 랜덤하게 변화시켜 다양한 경로 표현.
    X: (N, 6)  — obs = [start_xyz, target_xyz]
    Y: (N, 60) — chunk = 20 waypoints × 3D 평탄화
    """
    rng = np.random.default_rng(seed)
    X, Y = [], []
    while len(X) < n_samples:
        s = rng.uniform(WORKSPACE_MIN, WORKSPACE_MAX)
        t = rng.uniform(WORKSPACE_MIN, WORKSPACE_MAX)
        if np.linalg.norm(t - s) < 0.08:   # 너무 가까우면 스킵
            continue
        curv = rng.uniform(-0.3, 0.3)
        traj = generate_trajectory(s, t, CHUNK_SIZE, curvature=curv)
        X.append(np.concatenate([s, t]))
        Y.append(traj.flatten())
    return (np.array(X, dtype=np.float32),
            np.array(Y, dtype=np.float32))


# ─── 모델 정의 ──────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    CVAE Encoder: (obs, chunk) → (μ, log σ²)
    Training에서만 사용. Inference에서는 z ~ N(0,I)로 대체.
    입력: obs(6) + chunk(60) = 66차원
    출력: μ(32), log σ²(32)
    """
    def __init__(self, obs_dim=OBS_DIM, chunk_dim=CHUNK_DIM,
                 latent_dim=LATENT_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + chunk_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),              nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(hidden, latent_dim)
        self.fc_lv  = nn.Linear(hidden, latent_dim)   # log variance

    def forward(self, obs, chunk):
        h = self.net(torch.cat([obs, chunk], dim=-1))
        return self.fc_mu(h), self.fc_lv(h)


class Decoder(nn.Module):
    """
    CVAE Decoder: (obs, z) → chunk
    Training / Inference 모두 사용.
    입력: obs(6) + z(32) = 38차원
    출력: chunk(60) = 20 waypoints × 3
    """
    def __init__(self, obs_dim=OBS_DIM, latent_dim=LATENT_DIM,
                 chunk_dim=CHUNK_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),               nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),               nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, chunk_dim),
        )

    def forward(self, obs, z):
        return self.net(torch.cat([obs, z], dim=-1))


class CVAE(nn.Module):
    """
    Conditional VAE for Action Chunking.

    Training forward:
        obs(6) + chunk(60) → Encoder → μ,logσ² → reparametrize → z
        obs(6) + z(32)     → Decoder → pred_chunk(60)
        Loss = MSE(pred, chunk) + β·KL

    Inference:
        z ~ N(0, I) (Encoder는 쓰지 않음)
        obs(6) + z(32) → Decoder → chunk(60) → reshape(20,3)
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparametrize(self, mu, log_var):
        """
        Reparametrization trick: z = μ + ε·σ,  ε ~ N(0,I)
        역전파가 z를 통해 μ, σ로 흐를 수 있도록 함.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs, chunk):
        """Training 경로"""
        mu, lv = self.encoder(obs, chunk)
        z       = self.reparametrize(mu, lv)
        pred    = self.decoder(obs, z)
        return pred, mu, lv

    @torch.no_grad()
    def sample(self, obs: np.ndarray, n: int = 1) -> np.ndarray:
        """
        Inference: n개의 다양한 trajectory 생성.
        obs: (6,) ndarray
        반환: (n, CHUNK_SIZE, 3) ndarray
        """
        self.eval()
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).expand(n, -1)
        z     = torch.randn(n, LATENT_DIM)
        pred  = self.decoder(obs_t, z)
        return pred.numpy().reshape(n, CHUNK_SIZE, 3)


# ─── 손실 함수 ──────────────────────────────────────────────────────
def cvae_loss(pred, target, mu, log_var, beta=BETA):
    """
    ELBO 손실:
      recon_loss = MSE(pred, target)
      kl_loss    = -0.5 · Σ(1 + logσ² - μ² - σ²)
      total      = recon + β · kl
    β=1: 표준 VAE; β<1: 복원 중시; β>1: 잠재 공간 정규화 강조
    """
    recon = F.mse_loss(pred, target)
    kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + beta * kl, recon, kl


# ─── 학습 ──────────────────────────────────────────────────────────
def train(X, Y, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    model  = CVAE()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {'total': [], 'recon': [], 'kl': []}

    for ep in range(1, epochs + 1):
        model.train()
        tl = rl = kl_ = 0.0
        for obs_b, chunk_b in loader:
            pred, mu, lv   = model(obs_b, chunk_b)
            loss, r, k     = cvae_loss(pred, chunk_b, mu, lv)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); rl += r.item(); kl_ += k.item()
        n = len(loader)
        history['total'].append(tl / n)
        history['recon'].append(rl / n)
        history['kl'].append(kl_ / n)
        sched.step()
        if ep % 30 == 0 or ep == 1:
            print(f"  Epoch [{ep:3d}/{epochs}] "
                  f"Total={tl/n:.5f}  Recon={rl/n:.5f}  KL={kl_/n:.5f}")

    return model, history


# ─── 메인 실행 ─────────────────────────────────────────────────────
if __name__ == '__main__':
    OUTPUT_DIR = './outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) 데이터 생성
    print("Generating dataset...")
    X, Y = generate_dataset(N_DATA, seed=42)
    print(f"  X: {X.shape}, Y: {Y.shape}")

    # 2) 학습
    print("\nTraining CVAE...")
    model, history = train(X, Y)

    # 3) 모델 저장
    ckpt_path = os.path.join(OUTPUT_DIR, 'cvae_policy.pt')
    torch.save({'state_dict': model.state_dict(),
                'history':    history,
                'config': {'obs_dim': OBS_DIM, 'chunk_dim': CHUNK_DIM,
                           'latent_dim': LATENT_DIM, 'hidden': HIDDEN_DIM}},
               ckpt_path)
    print(f"\nSaved: {ckpt_path} ({os.path.getsize(ckpt_path)/1024:.1f} KB)")

    # 4) 모델 로드 예시
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model2 = CVAE()
    model2.load_state_dict(ckpt['state_dict'])
    model2.eval()
    print("Model reload OK")

    # 5) Inference 예시
    START  = np.array([0.10, 0.20, 0.30], dtype=np.float32)
    TARGET = np.array([0.40,-0.15, 0.10], dtype=np.float32)
    obs    = np.concatenate([START, TARGET])

    # 같은 (start, target)에서 여러 번 샘플링 → 다양한 경로
    trajs = model2.sample(obs, n=8)   # (8, 20, 3)
    print(f"\nSampled {len(trajs)} diverse trajectories from same start→target")
    for i, tr in enumerate(trajs):
        rmse = np.sqrt(np.mean((tr - generate_trajectory(START, TARGET))**2))
        print(f"  Traj {i+1}: final_pos={tr[-1]}, RMSE_vs_GT={rmse*100:.1f} cm")

    # 6) 간단한 손실 곡선 플롯
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ep_range = range(1, len(history['total'])+1)
    ax.plot(ep_range, history['total'], color='#f4a261', lw=2, label='Total')
    ax.plot(ep_range, history['recon'], color='#2dc5a2', lw=2, ls='--', label='Recon')
    ax.plot(ep_range, history['kl'],    color='#e06c75', lw=2, ls=':', label='KL')
    ax.set_xlabel('Epoch', color='white'); ax.set_ylabel('Loss', color='white')
    ax.set_title('CVAE Training Loss', color='white')
    ax.legend(); ax.tick_params(colors='white')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cvae_loss.png'), dpi=120, facecolor='#0d1117')
    plt.close()
    print(f"\nPlot saved: {OUTPUT_DIR}/cvae_loss.png")