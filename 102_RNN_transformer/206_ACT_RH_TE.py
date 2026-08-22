"""
2_action_chunking_train_glfw.py
SO101 6-DoF Action Chunking (K=30) Training with Random Start & Random Target Poses
- Dataset: No joint disturbances, Pure Smooth S-Curve Trajectories
- Visualization: GLFW Viewer with ./scene.xml
"""

import math
import time
import glfw
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mujoco

torch.manual_seed(42)

DOF = 6
NUM_EPISODES = 100
STEPS_PER_EP = 100
HISTORY = 10         # 슬라이딩 윈도우 크기
CHUNK_SIZE = 30      # Action Chunk 크기 (미래 K=30스텝 일괄 예측)
BATCH_SIZE = 64
EPOCHS = 45
LR = 1e-3
MODEL_PATH = "action_chunking_transformer.pt"
XML_PATH = "./scene.xml"

# --- 마우스 인터랙션 콜백 변수 ---
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

# --- 1. 랜덤 시작점 & 랜덤 목표점 데모 데이터 수집 (외란 없음) ---
def generate_random_p2p_episodes(model, data):
    print(f"[*] MuJoCo 환경에서 랜덤 시작점/목표점 100세트 P2P 데모 수집 중...")
    
    base_pos1 = torch.tensor([-0.6, -0.4, 0.5, -0.3, 0.2, 0.01], dtype=torch.float32)
    base_pos2 = torch.tensor([ 0.7,  0.5, -0.4,  0.6, -0.5, 0.035], dtype=torch.float32)

    DEG10_RAD = 10.0 * (math.pi / 180.0)  # +-10도 범위 (약 0.1745 rad)

    all_states, all_actions = [], []
    for ep in range(NUM_EPISODES):
        # 1~5축 회전 관절: +-10도 랜덤 / 6축 그리퍼: +-0.005m
        joint_noise1 = (torch.rand(5) * 2.0 - 1.0) * DEG10_RAD
        gripper_noise1 = (torch.rand(1) * 2.0 - 1.0) * 0.005
        start_pos = base_pos1 + torch.cat([joint_noise1, gripper_noise1])

        joint_noise2 = (torch.rand(5) * 2.0 - 1.0) * DEG10_RAD
        gripper_noise2 = (torch.rand(1) * 2.0 - 1.0) * 0.005
        target_pos = base_pos2 + torch.cat([joint_noise2, gripper_noise2])

        # 로봇 상태 초기화
        mujoco.mj_resetData(model, data)
        data.qpos[:DOF] = start_pos.numpy()
        data.qvel[:DOF] = 0.0
        mujoco.mj_forward(model, data)

        ep_states, ep_actions = [], []
        for step in range(STEPS_PER_EP):
            tau = step / (STEPS_PER_EP - 1)
            # Smoothstep (가감속 S-Curve)
            smooth_s = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
            
            # 외란 없이 깔끔한 보간 궤적 제어
            target_ctrl = (1.0 - smooth_s) * start_pos + smooth_s * target_pos

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
            last_t = len(ep_s) - chunk_size - 1
            for t in range(history - 1, last_t + 1):
                # Input: 과거 History 시점의 관절 상태 [History, DOF]
                # Target: 미래 K개 스텝의 목표 액션 [CHUNK_SIZE, DOF]
                xs.append(ep_s[t - history + 1 : t + 1])
                ys.append(ep_a[t + 1 : t + 1 + chunk_size])

        self.x = torch.stack(xs)
        self.y = torch.stack(ys)

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

        self.action_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, chunk_size * dof)
        )

    def forward(self, state_seq):
        x = self.state_projector(state_seq) + self.pos_embedding
        x = self.transformer_encoder(x)
        
        context_last = x[:, -1]
        pred_flat = self.action_head(context_last)
        return pred_flat.view(-1, self.chunk_size, self.dof)

# --- 4. GLFW 실시간 시뮬레이션 평가 (학습 직후 기본 동작 확인) ---
def run_glfw_eval(mj_model, mj_data, model):
    global cam, scn

    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패")

    window = glfw.create_window(1200, 900, "SO101 Action Chunking Training Verification", None, None)
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

    def reset_eval_robot():
        joint_noise = (torch.rand(5) * 2.0 - 1.0) * DEG10_RAD
        gripper_noise = (torch.rand(1) * 2.0 - 1.0) * 0.005
        test_start = base_pos1 + torch.cat([joint_noise, gripper_noise])

        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:DOF] = test_start.numpy()
        mujoco.mj_forward(mj_model, mj_data)
        return [mj_data.qpos[:DOF].copy() for _ in range(HISTORY)]

    state_buffer = reset_eval_robot()
    action_plan_queue = []
    step_cnt = 0

    print("\n[*] 학습 완료 모델 실시간 시연 중 (창을 닫으면 종료)...")

    while not glfw.window_should_close(window):
        step_start = time.time()

        if len(action_plan_queue) == 0:
            input_tensor = torch.tensor(state_buffer, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_chunk = model(input_tensor).squeeze(0).numpy()
            action_plan_queue = list(pred_chunk)

        next_action = action_plan_queue.pop(0)
        mj_data.ctrl[:DOF] = next_action
        mujoco.mj_step(mj_model, mj_data)

        state_buffer.pop(0)
        state_buffer.append(mj_data.qpos[:DOF].copy())

        step_cnt += 1
        if step_cnt >= STEPS_PER_EP + 20:
            state_buffer = reset_eval_robot()
            action_plan_queue.clear()
            step_cnt = 0

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

    # scene.xml 파일 로드
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

    model = ActionChunkTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("\n[*] Action Chunking Transformer 학습 시작...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for seq, chunk in train_loader:
            seq, chunk = seq.to(device), chunk.to(device)
            pred = model(seq)
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