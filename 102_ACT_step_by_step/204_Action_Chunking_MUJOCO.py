"""
2_action_chunking_train_eval_glfw.py
SO101 6-DoF Point-to-Point Action Chunking (Chunk Size K=30) Training & GLFW Viewer
"""

import math
import time
import glfw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mujoco
import numpy as np

torch.manual_seed(42)

DOF = 6
NUM_EPISODES = 100
STEPS_PER_EP = 100
HISTORY = 10         # 과거 10개 프레임 슬라이딩 윈도우
CHUNK_SIZE = 30      # 한 번에 일괄 예측할 미래 행동 스텝 수 (K=30)
BATCH_SIZE = 64
EPOCHS = 45
LR = 1e-3
MODEL_PATH = "action_chunking_transformer.pt"

button_left = False
button_middle = False
button_right = False
last_x = 0
last_y = 0


# --- 마우스 인터랙션 콜백 ---
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

# --- 1. 100세트 데모 수집 ---
def generate_point_to_point_episodes(model, data):
    print(f"[*] MuJoCo 환경에서 100세트 P2P 데모 수집 중...")
    base_pos1 = torch.tensor([-0.6, -0.4, 0.5, -0.3, 0.2, 0.01], dtype=torch.float32)
    base_pos2 = torch.tensor([ 0.7,  0.5, -0.4,  0.6, -0.5, 0.035], dtype=torch.float32)

    all_states, all_actions = [], []
    for ep in range(NUM_EPISODES):
        pos1_noise = base_pos1 + (torch.rand(DOF) - 0.5) * 0.08
        pos2_noise = base_pos2 + (torch.rand(DOF) - 0.5) * 0.08

        mujoco.mj_resetData(model, data)
        data.qpos[:DOF] = pos1_noise.numpy()
        data.qvel[:DOF] = 0.0
        mujoco.mj_forward(model, data)

        ep_states, ep_actions = [], []
        for step in range(STEPS_PER_EP):
            tau = step / (STEPS_PER_EP - 1)
            smooth_s = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
            step_noise = (torch.rand(DOF) - 0.5) * 0.005
            target_ctrl = (1.0 - smooth_s) * pos1_noise + smooth_s * pos2_noise + step_noise

            current_state = data.qpos[:DOF].copy()
            data.ctrl[:DOF] = target_ctrl.numpy()
            mujoco.mj_step(model, data)

            ep_states.append(current_state)
            ep_actions.append(target_ctrl.numpy())

        all_states.append(ep_states)
        all_actions.append(ep_actions)

    return torch.tensor(all_states, dtype=torch.float32), torch.tensor(all_actions, dtype=torch.float32)

# --- 2. Action Chunking Dataset ---
class ActionChunkDataset(Dataset):
    def __init__(self, states, actions, history=HISTORY, chunk_size=CHUNK_SIZE):
        xs, ys = [], []
        num_episodes = states.shape[0]

        for ep in range(num_episodes):
            ep_s, ep_a = states[ep], actions[ep]
            # t 시점 기준 미래 chunk_size 개를 모두 가져올 수 있는 범위까지 슬라이딩
            last_t = len(ep_s) - chunk_size - 1
            #print(f"last_t: {last_t}")  # 69
            for t in range(history - 1, last_t + 1):
                # Input: t-history+1 ~ t 관절 상태 [History, DOF]
                # Target Chunk: t+1 ~ t+chunk_size 미래 액션 시퀀스 [CHUNK_SIZE, DOF]
                xs.append(ep_s[t - history + 1 : t + 1])
                ys.append(ep_a[t + 1 : t + 1 + chunk_size])
        #print(f"xs: {np.array(xs).shape}")  #  (7200, 10, 6)
        #print(f"ys: {np.array(ys).shape}")  #  (7200, 6)

        self.x = torch.stack(xs)  # [Total_Samples, HISTORY, DOF]
        self.y = torch.stack(ys)  # [Total_Samples, CHUNK_SIZE, DOF]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# --- 3. Action Chunking Transformer 모델 ---
class ActionChunkTransformer(nn.Module):
    def __init__(self, dof=DOF, history=HISTORY, chunk_size=CHUNK_SIZE, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.dof = dof

        self.state_projector = nn.Linear(dof, d_model)  # [b 10 6]  -> [b 10 64] 
        self.pos_embedding = nn.Parameter(torch.zeros(1, history, d_model))   # [b 10 64] -> [b 10 64]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Context Vector -> 미래 K개 스텝의 6축 각도 묶음(CHUNK_SIZE * DOF) 일괄 예측 Head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, chunk_size * dof)
        )

    def forward(self, state_seq):
        # state_seq: [Batch, History, DOF]
        print(f"seq {state_seq.shape}")  # [64 10 6]
        x = self.state_projector(state_seq) + self.pos_embedding
        print(f"x embedding + pos {x.shape}") # [64 10 6]
        x = self.transformer_encoder(x) 
        print(f"x encoded: {x.shape}") # [64 10 64]
        
        context_last = x[:, -1]  # [Batch, d_model] [64 64]
        print(f"context_last {context_last.shape}")
        pred_flat = self.action_head(context_last)  # [Batch, CHUNK_SIZE * DOF]
        print(f"pred_flat {pred_flat.shape}")  # [b 64 180]
        return pred_flat.view(-1, self.chunk_size, self.dof)  # [Batch, CHUNK_SIZE, DOF]

# --- 4. GLFW 기반 실시간 Action Chunk 제어 평가 ---
def run_glfw_eval(mj_model, mj_data, model):
    global cam, scn

    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패")

    window = glfw.create_window(1200, 900, "SO101 Action Chunking BC - GLFW Eval", None, None)
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

    eval_pos1 = torch.tensor([-0.6, -0.4, 0.5, -0.3, 0.2, 0.01])

    def reset_robot():
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:DOF] = eval_pos1.numpy()
        mujoco.mj_forward(mj_model, mj_data)
        return [mj_data.qpos[:DOF].copy() for _ in range(HISTORY)]

    state_buffer = reset_robot()
    step_cnt = 0
    
    # 생성된 Action Chunk를 저장하고 순차 실행할 실행 큐
    action_plan_queue = []

    print("\n[*] Action Chunking(K=30) 실시간 GLFW 제어 시작...")

    while not glfw.window_should_close(window):
        step_start = time.time()

        # (1) Chunk가 소진되었을 때만 신경망 추론 1회 호출 (K 스텝마다 1번 추론)
        if len(action_plan_queue) == 0:
            input_tensor = torch.tensor(state_buffer, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                # [1, CHUNK_SIZE, DOF] -> [CHUNK_SIZE, DOF]
                pred_chunk = model(input_tensor).squeeze(0).numpy()
            action_plan_queue = list(pred_chunk)

        # (2) 계획된 청크에서 다음 1스텝 꺼내어 실행
        next_action = action_plan_queue.pop(0)
        mj_data.ctrl[:DOF] = next_action
        mujoco.mj_step(mj_model, mj_data)

        # (3) 롤링 버퍼 갱신
        state_buffer.pop(0)
        state_buffer.append(mj_data.qpos[:DOF].copy())

        step_cnt += 1
        if step_cnt >= STEPS_PER_EP + 20:
            state_buffer = reset_robot()
            action_plan_queue.clear()
            step_cnt = 0

        # (4) GLFW 렌더링
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

def main():
    global mj_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mj_model = mujoco.MjModel.from_xml_path('./scene.xml')
    mj_data = mujoco.MjData(mj_model)
    states, actions = generate_point_to_point_episodes(mj_model, mj_data)
    print(f"states: {states.shape}")
    print(f"actions: {actions.shape}")

    train_ds = ActionChunkDataset(states[:80], actions[:80])
    test_ds = ActionChunkDataset(states[80:], actions[80:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[*] 생성된 학습 샘플 수: {len(train_ds)} / 테스트 샘플 수: {len(test_ds)}")

    model = ActionChunkTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("\n[*] Action Chunking Transformer 학습 시작...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for seq, chunk in train_loader:
            #print(f"seq: {seq.shape}")  # [64 10 6]
            #print(f"act: {chunk.shape}")  # [64 30 6]
            seq, chunk = seq.to(device), chunk.to(device)
            pred = model(seq) # [64 30, 6]
            print(f"pred: {pred.shape}")
            loss = criterion(pred, chunk)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seq.size(0)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train MSE: {total_loss / len(train_ds):.6f}")

    # 정량 평가
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for seq, chunk in test_loader:
            seq, chunk = seq.to(device), chunk.to(device)
            pred = model(seq)
            test_loss += criterion(pred, chunk).item() * seq.size(0)
    print(f"\n[최종 Test Chunk MSE Loss]: {test_loss / len(test_ds):.7f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[*] 모델 가중치 저장 완료: {MODEL_PATH}")

    model.cpu()
    run_glfw_eval(mj_model, mj_data, model)

if __name__ == "__main__":
    main()