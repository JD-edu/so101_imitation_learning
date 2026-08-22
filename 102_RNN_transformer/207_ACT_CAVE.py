"""
3_cvae_act_train_eval_glfw.py
SO101 6-DoF CVAE-based Action Chunking Transformer (ACT) Full Pipeline
- CVAE Latent Modeling (Reparameterization, KL Loss)
- Transformer Encoder & Decoder with Action Queries
- Random Start & Random Target Trajectories from ./scene.xml
- Real-time Closed-Loop GLFW Simulation with Temporal Ensembling
"""

import math
import time
import glfw
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mujoco

# 재현성을 위한 시드 설정
torch.manual_seed(42)
np.random.seed(42)

# --- 하이퍼파라미터 정의 ---
DOF = 6
NUM_EPISODES = 100
STEPS_PER_EP = 100
HISTORY = 10         # 슬라이딩 윈도우 크기 (과거 H 프레임)
CHUNK_SIZE = 30      # Action Chunk 크기 (미래 K=30스텝)
LATENT_DIM = 16      # CVAE 잠재 변수 z 차원
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-3
KL_WEIGHT = 10.0     # CVAE KL Divergence 가중치 (Beta-VAE)
MODEL_PATH = "cvae_act_transformer.pt"
XML_PATH = "./scene.xml"

# GLFW 마우스 인터랙션 콜백 변수
button_left = False
button_middle = False
button_right = False
last_x = 0
last_y = 0

def mouse_button(window, button, action, mods):
    global button_left, button_middle, button_right
    button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

def mouse_move(window, xpos, ypos):
    global last_x, last_y, button_left, button_middle, button_right, mj_model, scn, cam
    dx = xpos - last_x
    dy = ypos - last_y
    last_x = xpos
    last_y = ypos

    if not (button_left or button_middle or button_right):
        return

    width, height = glfw.get_window_size(window)
    mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or \
                glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

    if button_right:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM

    mujoco.mjv_moveCamera(mj_model, action, dx / height, dy / height, scn, cam)

def scroll(window, xoffset, yoffset):
    global mj_model, scn, cam
    mujoco.mjv_moveCamera(mj_model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, scn, cam)

# --- 1. 랜덤 시작점 & 랜덤 목표점 P2P 데모 수집 ---
def generate_random_p2p_episodes(model, data):
    print("[*] MuJoCo 환경에서 랜덤 시작점/목표점 100세트 P2P 데모 수집 중...")
    
    base_pos1 = torch.tensor([-0.6, -0.4, 0.5, -0.3, 0.2, 0.01], dtype=torch.float32)
    base_pos2 = torch.tensor([ 0.7,  0.5, -0.4,  0.6, -0.5, 0.035], dtype=torch.float32)
    DEG10_RAD = 10.0 * (math.pi / 180.0)

    all_states, all_actions = [], []
    for ep in range(NUM_EPISODES):
        joint_noise1 = (torch.rand(5) * 2.0 - 1.0) * DEG10_RAD
        gripper_noise1 = (torch.rand(1) * 2.0 - 1.0) * 0.005
        start_pos = base_pos1 + torch.cat([joint_noise1, gripper_noise1])

        joint_noise2 = (torch.rand(5) * 2.0 - 1.0) * DEG10_RAD
        gripper_noise2 = (torch.rand(1) * 2.0 - 1.0) * 0.005
        target_pos = base_pos2 + torch.cat([joint_noise2, gripper_noise2])

        mujoco.mj_resetData(model, data)
        data.qpos[:DOF] = start_pos.numpy()
        data.qvel[:DOF] = 0.0
        mujoco.mj_forward(model, data)

        ep_states, ep_actions = [], []
        for step in range(STEPS_PER_EP):
            tau = step / (STEPS_PER_EP - 1)
            smooth_s = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
            target_ctrl = (1.0 - smooth_s) * start_pos + smooth_s * target_pos

            current_state = data.qpos[:DOF].copy()
            data.ctrl[:DOF] = target_ctrl.numpy()
            mujoco.mj_step(model, data)

            ep_states.append(current_state)
            ep_actions.append(target_ctrl.numpy())

        all_states.append(ep_states)
        all_actions.append(ep_actions)

    return torch.tensor(all_states, dtype=torch.float32), torch.tensor(all_actions, dtype=torch.float32)

# --- 2. Action Chunk Dataset ---
class ActionChunkDataset(Dataset):
    def __init__(self, states, actions, history=HISTORY, chunk_size=CHUNK_SIZE):
        xs, ys = [], []
        num_episodes = states.shape[0]

        for ep in range(num_episodes):
            ep_s, ep_a = states[ep], actions[ep]
            last_t = len(ep_s) - chunk_size - 1
            for t in range(history - 1, last_t + 1):
                xs.append(ep_s[t - history + 1 : t + 1])
                ys.append(ep_a[t + 1 : t + 1 + chunk_size])

        self.x = torch.stack(xs)
        self.y = torch.stack(ys)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# --- 3. CVAE Encoder (학습 시 미래 Chunk와 현재 상태를 인코딩) ---
class CVAEEncoder(nn.Module):
    def __init__(self, dof=DOF, history=HISTORY, chunk_size=CHUNK_SIZE, latent_dim=LATENT_DIM, d_model=D_MODEL):
        super().__init__()
        in_dim = (history + chunk_size) * dof
        self.encoder_net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, state_seq, action_chunk):
        x = torch.cat([state_seq.flatten(start_dim=1), action_chunk.flatten(start_dim=1)], dim=1)
        feat = self.encoder_net(x)
        mu = self.fc_mu(feat)
        logvar = torch.clamp(self.fc_logvar(feat), min=-10.0, max=10.0)
        return mu, logvar

# --- 4. CVAE Transformer ACT 모델 (Encoder-Decoder) ---
class CVAE_ACT(nn.Module):
    def __init__(self, dof=DOF, history=HISTORY, chunk_size=CHUNK_SIZE, latent_dim=LATENT_DIM, 
                 d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS):
        super().__init__()
        self.dof = dof
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        # CVAE Encoder
        self.cvae_encoder = CVAEEncoder(dof, history, chunk_size, latent_dim, d_model)

        # Observation Encoder
        self.state_projector = nn.Linear(dof, d_model)
        self.latent_projector = nn.Linear(latent_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, history + 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Action Decoder (Cross-Attention 기반)
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, dof)

        nn.init.normal_(self.action_queries, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state_seq, action_chunk=None):
        batch_size = state_seq.size(0)

        # 1. 잠재 변수 z 획득
        if self.training and action_chunk is not None:
            mu, logvar = self.cvae_encoder(state_seq, action_chunk)
            z = self.reparameterize(mu, logvar)
        else:
            mu, logvar = None, None
            z = torch.zeros(batch_size, self.latent_dim, device=state_seq.device)

        # 2. Transformer Encoder 입력: [z_token, state_tokens]
        z_token = self.latent_projector(z).unsqueeze(1)
        state_tokens = self.state_projector(state_seq)
        enc_input = torch.cat([z_token, state_tokens], dim=1)
        enc_input = enc_input + self.pos_embedding

        memory = self.transformer_encoder(enc_input)

        # 3. Transformer Decoder에서 Action Chunk 복원
        query = self.action_queries.expand(batch_size, -1, -1)
        dec_out = self.transformer_decoder(tgt=query, memory=memory)
        pred_actions = self.action_head(dec_out)

        return pred_actions, mu, logvar

# --- 5. CVAE 손실 함수 ---
def compute_loss(pred_actions, target_actions, mu, logvar, kl_weight=KL_WEIGHT):
    recon_loss = nn.functional.l1_loss(pred_actions, target_actions)
    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss

# --- 6. GLFW 실시간 시뮬레이션 평가 ---
def run_glfw_eval(mj_model, mj_data, model):
    global cam, scn

    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패")

    window = glfw.create_window(1200, 900, "SO101 Full ACT (CVAE + Transformer) Evaluation", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW 윈도우 생성 실패")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_scroll_callback(window, scroll)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(mj_model, maxgeom=1000)
    con = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    cam.azimuth = 90.0
    cam.elevation = -25.0
    cam.distance = 1.2
    cam.lookat = [0.0, 0.0, 0.2]

    base_pos1 = torch.tensor([-0.6, -0.4, 0.5, -0.3, 0.2, 0.01])
    DEG10_RAD = 10.0 * (math.pi / 180.0)
    EXP_WEIGHT_M = 0.05
    weights = np.exp(-EXP_WEIGHT_M * np.arange(CHUNK_SIZE))

    def reset_eval_robot():
        joint_noise = (torch.rand(5) * 2.0 - 1.0) * DEG10_RAD
        gripper_noise = (torch.rand(1) * 2.0 - 1.0) * 0.005
        test_start = base_pos1 + torch.cat([joint_noise, gripper_noise])

        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:DOF] = test_start.numpy()
        mujoco.mj_forward(mj_model, mj_data)
        return [mj_data.qpos[:DOF].copy() for _ in range(HISTORY)], {}

    state_buffer, all_time_actions = reset_eval_robot()
    current_time_step = 0

    print("\n[*] CVAE-ACT 학습 완료 모델 실시간 시연 중 (창을 닫으면 종료)...")

    while not glfw.window_should_close(window):
        step_start = time.time()

        # 1) 추론: z=0 고정 상태로 미래 K개 청크 예측
        input_tensor = torch.tensor(state_buffer, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_chunk, _, _ = model(input_tensor)
            pred_chunk = pred_chunk.squeeze(0).numpy()

        # 2) Temporal Ensembling 버퍼 누적
        for i in range(CHUNK_SIZE):
            future_t = current_time_step + 1 + i
            if future_t not in all_time_actions:
                all_time_actions[future_t] = []
            all_time_actions[future_t].append((pred_chunk[i], weights[i]))

        # 3) 현재 스텝 가중 평균 제어값 산출
        target_t = current_time_step + 1
        actions_at_t = all_time_actions[target_t]
        weighted_sum = np.zeros(DOF)
        total_weight = 0.0
        for act, w in actions_at_t:
            weighted_sum += act * w
            total_weight += w
        final_action = weighted_sum / total_weight
        del all_time_actions[target_t]

        # 4) 물리 제어 및 버퍼 갱신
        mj_data.ctrl[:DOF] = final_action
        mujoco.mj_step(mj_model, mj_data)

        state_buffer.pop(0)
        state_buffer.append(mj_data.qpos[:DOF].copy())
        current_time_step += 1

        if current_time_step >= STEPS_PER_EP + 20:
            state_buffer, all_time_actions = reset_eval_robot()
            current_time_step = 0

        # 5) GLFW 렌더링
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(mj_model, mj_data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn, con)
        glfw.swap_buffers(window)
        glfw.poll_events()

        time_until_next = mj_model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

    glfw.terminate()

# --- 7. 메인 실행 루프 ---
def main():
    global mj_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 사용 장치: {device}")

    try:
        mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        mj_data = mujoco.MjData(mj_model)
        print(f"[*] MuJoCo XML 로드 성공: {XML_PATH}")
    except Exception as e:
        print(f"[!] XML 로드 실패: {e}")
        return

    states, actions = generate_random_p2p_episodes(mj_model, mj_data)

    train_ds = ActionChunkDataset(states[:80], actions[:80])
    test_ds = ActionChunkDataset(states[80:], actions[80:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[*] 학습 데이터 샘플 수: {len(train_ds)} / 테스트 데이터 샘플 수: {len(test_ds)}")

    model = CVAE_ACT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    print("\n[*] CVAE-ACT Transformer 학습 시작...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

        for seq, chunk in train_loader:
            seq, chunk = seq.to(device), chunk.to(device)
            pred_chunk, mu, logvar = model(seq, chunk)
            loss, recon_l, kl_l = compute_loss(pred_chunk, chunk, mu, logvar, kl_weight=KL_WEIGHT)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * seq.size(0)
            total_recon += recon_l.item() * seq.size(0)
            total_kl += kl_l.item() * seq.size(0)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Total Loss: {total_loss / len(train_ds):.6f} "
                  f"(Recon L1: {total_recon / len(train_ds):.6f}, KL: {total_kl / len(train_ds):.6f})")

    model.eval()
    test_recon = 0.0
    with torch.no_grad():
        for seq, chunk in test_loader:
            seq, chunk = seq.to(device), chunk.to(device)
            pred_chunk, _, _ = model(seq)
            recon_l = nn.functional.l1_loss(pred_chunk, chunk)
            test_recon += recon_l.item() * seq.size(0)

    print(f"\n[최종 Test L1 Reconstruction Loss (z=0)]: {test_recon / len(test_ds):.7f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[*] 모델 가중치 저장 완료: {MODEL_PATH}")

    model.cpu()
    run_glfw_eval(mj_model, mj_data, model)

if __name__ == "__main__":
    main()