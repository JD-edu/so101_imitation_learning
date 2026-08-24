"""
1_transformer_encoder_train_eval_glfw.py
SO101 6-DoF Transformer Encoder 1-Step BC Training & GLFW Real-time Evaluation
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
HISTORY = 10         # 슬라이딩 윈도우 크기
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
MODEL_PATH = "transformer_encoder_bc.pt"

# --- 마우스 조작 인터랙션 변수 ---
button_left = False
button_middle = False
button_right = False
last_x = 0
last_y = 0


# --- GLFW 마우스 콜백 함수 ---
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

# --- 100세트 P2P 데모 수집 ---
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

class P2PTrajectoryDataset(Dataset):
    def __init__(self, states, actions, history=HISTORY):
        #print(f"states: {states.shape}")  # [80, 100, 6 ]
        #print(f"actions: {actions.shape}")  # [80, 100, 6 ]
        xs, ys = [], []
        for ep in range(states.shape[0]):  # 80 turn for train data 
            ep_s, ep_a = states[ep], actions[ep]
            #print(f"ep_s {ep_s.shape} ep_s length {len(ep_s)}")  # [100 , 6]
            for t in range(history - 1, len(ep_s) - 1):  # from 9 to 99 
                #print(f"ep_s from {t - history + 1} to {t+1}")
                xs.append(ep_s[t - history + 1 : t + 1])  
                #print(f"ep_a to {t+1}")
                ys.append(ep_a[t + 1])
        
        #print(f"xs: {np.array(xs).shape}")  #  (7200, 10, 6)
        #print(f"ys: {np.array(ys).shape}")  #  (7200, 6)
        self.x = torch.stack(xs)
        self.y = torch.stack(ys)
        #print(f"self.x: {self.x.shape}")
        #print(f"self.y: {self.y.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TransformerEncoderBC(nn.Module):
    def __init__(self, dof=DOF, history=HISTORY, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.state_projector = nn.Linear(dof, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, history, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, dof)

    def forward(self, state_seq):
        x = self.state_projector(state_seq) + self.pos_embedding
        x = self.transformer_encoder(x)
        return self.head(x[:, -1])

# --- GLFW 기반 실시간 Closed-Loop 시뮬레이션 ---
def run_glfw_eval(mj_model, mj_data, model):
    global cam, scn

    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패")

    window = glfw.create_window(1200, 900, "SO101 Transformer Encoder BC - GLFW Eval", None, None)
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

    print("\n[*] GLFW 창에서 로봇 제어 동작이 실시간 렌더링됩니다 (창을 닫거나 ESC로 종료)...")

    while not glfw.window_should_close(window):
        step_start = time.time()

        # (1) 과거 10개 프레임 슬라이딩 윈도우 [1, HISTORY, DOF]
        input_tensor = torch.tensor(state_buffer, dtype=torch.float32).unsqueeze(0)

        # (2) 트랜스포머 인코더로 t+1 목표 각도 예측
        with torch.no_grad():
            pred_action = model(input_tensor).squeeze(0).numpy()

        # (3) 액추에이터 제어 명령 인가 및 물리 전진
        mj_data.ctrl[:DOF] = pred_action
        mujoco.mj_step(mj_model, mj_data)

        # (4) 슬라이딩 윈도우 버퍼 갱신
        state_buffer.pop(0)
        state_buffer.append(mj_data.qpos[:DOF].copy())

        step_cnt += 1
        if step_cnt >= STEPS_PER_EP + 20:
            state_buffer = reset_robot()
            step_cnt = 0

        # (5) GLFW 렌더링
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

    mj_model = mujoco.MjModel.from_xml_path("./scene.xml")
    mj_data = mujoco.MjData(mj_model)
    states, actions = generate_point_to_point_episodes(mj_model, mj_data)

    train_ds = P2PTrajectoryDataset(states[:80], actions[:80])
    test_ds = P2PTrajectoryDataset(states[80:], actions[80:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TransformerEncoderBC().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("\n[*] Transformer Encoder BC 학습 시작...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for seq, act in train_loader:
            #print(f"seq: {seq.shape}")
            #print(f"act: {act.shape}")
            seq, act = seq.to(device), act.to(device)
            pred = model(seq)
            loss = criterion(pred, act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seq.size(0)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train MSE: {total_loss / len(train_ds):.6f}")

    # 테스트 정량 평가
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for seq, act in test_loader:
            seq, act = seq.to(device), act.to(device)
            pred = model(seq)
            test_loss += criterion(pred, act).item() * seq.size(0)
    print(f"\n[Test MSE Loss]: {test_loss / len(test_ds):.7f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[*] 모델 가중치 저장 완료: {MODEL_PATH}")

    # 학습 완료 후 CPU 전환하여 GLFW 실시간 시뮬레이션 평가 실행
    model.cpu()
    run_glfw_eval(mj_model, mj_data, model)

if __name__ == "__main__":
    main()