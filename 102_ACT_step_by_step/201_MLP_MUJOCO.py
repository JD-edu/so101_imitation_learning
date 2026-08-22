"""
so101_mlp_bc_glfw.py
SO101 6-DoF Point-to-Point (위치 1 -> 위치 2) MLP Behavior Cloning 교육 코드 (GLFW 렌더링)

특징:
1. 시작 위치(위치 1)와 도착 위치(위치 2) 고정[cite: 3]
2. 100개 세트(에피소드) x 각 100 타임스텝 = 총 10,000개의 6축 관절 데이터 생성[cite: 3]
3. 과거 히스토리 없이 오직 t 시점의 state(6-DoF) -> t+1 시점의 target action(6-DoF) 예측[cite: 1, 4]
4. 학습 후 GLFW 윈도우를 통한 Closed-loop 실시간 시뮬레이션 검증[cite: 3]
"""

import time
import glfw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mujoco

# 재현성을 위한 시드 고정
torch.manual_seed(42)

DOF = 6               # 6개 관절 (Joint 1~5 + Gripper)[cite: 3]
NUM_EPISODES = 100    # 100개의 에피소드(세트)[cite: 3]
STEPS_PER_EP = 100    # 1 세트당 100개 타임스텝 데이터[cite: 3]
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
XML_PATH = "./scene.xml"  # 로봇 씬 파일 경로

# --- 마우스 조작 인터랙션을 위한 전역 변수 ---
button_left = False
button_middle = False
button_right = False
last_x = 0
last_y = 0


# --- GLFW 마우스 이벤트 콜백 함수 ---
def mouse_button(window, button, action, mods):
    global button_left, button_middle, button_right
    button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS


def mouse_move(window, xpos, ypos):
    global last_x, last_y, button_left, button_middle, button_right

    dx = xpos - last_x
    dy = ypos - last_y
    last_x = xpos
    last_y = ypos

    if not (button_left or button_middle or button_right):
        return

    width, height = glfw.get_window_size(window)
    mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or \
                glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

    # 카메라 회전 / 이동 제어
    if button_right:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM

    mujoco.mjv_moveCamera(mj_model, action, dx / height, dy / height, scn, cam)


def scroll(window, xoffset, yoffset):
    mujoco.mjv_moveCamera(mj_model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, scn, cam)


# --- 1. 고정된 위치 1 -> 위치 2 데이터 수집 (100세트 x 100스텝) ---
def generate_p2p_episodes(model, data):
    print(f"[*] MuJoCo 환경에서 100세트(총 {NUM_EPISODES * STEPS_PER_EP} steps) P2P 데모 수집 중...")
    
    # 시작 위치 1 (대기 자세) 및 도착 위치 2 (목표 자세) 고정[cite: 3]
    FIXED_POS1 = torch.tensor([-0.6, -0.4, 0.5, -0.3, 0.2, 0.01], dtype=torch.float32)
    FIXED_POS2 = torch.tensor([ 0.7,  0.5, -0.4,  0.6, -0.5, 0.035], dtype=torch.float32)

    all_states = []
    all_actions = []

    for ep in range(NUM_EPISODES):
        # 시작 위치 고정으로 리셋
        mujoco.mj_resetData(model, data)
        data.qpos[:DOF] = FIXED_POS1.numpy()
        data.qvel[:DOF] = 0.0
        mujoco.mj_forward(model, data)

        ep_states = []
        ep_actions = []

        for step in range(STEPS_PER_EP):
            # 부드러운 S자 가감속 곡선 보간 (Smoothstep: 3s^2 - 2s^3)[cite: 3]
            tau = step / (STEPS_PER_EP - 1)
            smooth_s = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)

            # 다채로운 데이터셋 구성을 위한 미세 제어 노이즈[cite: 3]
            noise = (torch.rand(DOF) - 0.5) * 0.006
            target_ctrl = (1.0 - smooth_s) * FIXED_POS1 + smooth_s * FIXED_POS2 + noise

            current_state = data.qpos[:DOF].copy()
            data.ctrl[:DOF] = target_ctrl.numpy()
            mujoco.mj_step(model, data)

            ep_states.append(current_state)
            ep_actions.append(target_ctrl.numpy())

        # 각 에피소드 내에서 t 시점(state) -> t+1 시점(action) 매핑[cite: 1, 3]
        for t in range(STEPS_PER_EP - 1):
            all_states.append(ep_states[t])
            all_actions.append(ep_actions[t + 1])

    return torch.tensor(all_states, dtype=torch.float32), torch.tensor(all_actions, dtype=torch.float32)


# --- 2. Dataset 정의 ---
class RobotDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index]


# --- 3. MLP 기반 Behavior Cloning 신경망 ---
class BehaviorCloningMLP(nn.Module):
    def __init__(self, dof=DOF):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dof, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dof),
        )

    def forward(self, state):
        # state: [Batch, DOF] -> action: [Batch, DOF]
        return self.net(state)


# --- 4. GLFW 기반 실시간 렌더링 및 Closed-Loop 추론 ---
def run_glfw_simulation(mj_model, mj_data, model):
    global cam, scn

    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패")

    window = glfw.create_window(1200, 900, "SO101 MLP BC Training & Evaluation (GLFW)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW 윈도우 생성 실패")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # 콜백 등록
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_scroll_callback(window, scroll)

    # MuJoCo 렌더링 구조체 초기화
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(mj_model, maxgeom=1000)
    con = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    cam.azimuth = 90.0
    cam.elevation = -25.0
    cam.distance = 1.2
    cam.lookat = [0.0, 0.0, 0.2]

    FIXED_POS1 = torch.tensor([-0.6, -0.4, 0.5, -0.3, 0.2, 0.01])

    def reset_robot():
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:DOF] = FIXED_POS1.numpy()
        mujoco.mj_forward(mj_model, mj_data)

    reset_robot()

    print("\n[*] 학습된 MLP 모델로 GLFW 실시간 시뮬레이션을 시작합니다 (창을 닫거나 ESC로 종료)...")
    step_cnt = 0

    while not glfw.window_should_close(window):
        step_start = time.time()

        # 1) 현재 관절 상태 읽기: [1, 6]
        current_state = torch.tensor(mj_data.qpos[:DOF], dtype=torch.float32).unsqueeze(0)

        # 2) MLP 모델로 t+1 목표 각도 예측[cite: 1, 4]
        with torch.no_grad():
            pred_action = model(current_state).squeeze(0).numpy()

        # 3) 액추에이터 제어 명령 인가 및 물리 1스텝 전진
        mj_data.ctrl[:DOF] = pred_action
        mujoco.mj_step(mj_model, mj_data)

        # 100스텝 후 반복 시연을 위해 위치 1로 리셋
        step_cnt += 1
        if step_cnt >= STEPS_PER_EP + 20:
            reset_robot()
            step_cnt = 0

        # 4) GLFW 화면 렌더링
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)

        mujoco.mjv_updateScene(mj_model, mj_data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn, con)

        glfw.swap_buffers(window)
        glfw.poll_events()

        # 물리 타임스텝 동기화
        time_until_next = mj_model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

    glfw.terminate()


# --- 5. 메인 실행 파이프라인 ---
def main():
    global mj_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 사용 디바이스: {device}")

    # (1) MuJoCo 초기화 및 데모 데이터 수집
    try:
        mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        mj_data = mujoco.MjData(mj_model)
        print(f"[*] MuJoCo XML 로드 성공: {XML_PATH}")
    except Exception as e:
        print(f"[!] XML 로드 실패: {e}")
        return

    states, actions = generate_p2p_episodes(mj_model, mj_data)
    print(f"[*] 총 생성된 (q(t) -> q(t+1)) 데이터 수: {len(states)} 개")

    # (2) Train (80%) / Test (20%) 분할[cite: 3]
    split = int(len(states) * 0.8)
    train_ds = RobotDataset(states[:split], actions[:split])
    test_states = states[split:].to(device)
    test_actions = actions[split:].to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # (3) MLP 모델 학습
    model = BehaviorCloningMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("\n[*] MLP Behavior Cloning 학습 시작...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for state, action in train_loader:
            state = state.to(device)
            action = action.to(device)

            pred = model(state)
            loss = criterion(pred, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * state.size(0)

        if epoch == 1 or epoch % 5 == 0:
            avg_loss = total_loss / len(train_ds)
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train MSE = {avg_loss:.6f}")

    # (4) 오프라인 정량 평가
    model.eval()
    with torch.no_grad():
        pred = model(test_states)
        test_loss = criterion(pred, test_actions).item()
    print(f"\n[Test MSE Loss]: {test_loss:.7f}")

    torch.save(model.state_dict(), "so101_mlp_bc_p2p.pt")
    print("[*] 모델 가중치 저장 완료: so101_mlp_bc_p2p.pt")

    # (5) GLFW 시뮬레이션 실행 (CPU 전환 후 렌더링)
    model.cpu()
    run_glfw_simulation(mj_model, mj_data, model)


if __name__ == "__main__":
    main()