"""
so101_rnn_p2p_glfw.py
SO101 6-DoF Point-to-Point (Pos1 -> Pos2) RNN Behavior Cloning with GLFW Viewer
"""

import math
import time
import glfw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mujoco

# 재현성을 위한 시드 설정
torch.manual_seed(42)

# --- 파라미터 정의 ---
DOF = 6               # 6개 관절
NUM_EPISODES = 100    # 100개 세트
STEPS_PER_EP = 100    # 에피소드당 100 타임스텝
HISTORY = 10          # 과거 10개 프레임 슬라이딩 윈도우
BATCH_SIZE = 64
EPOCHS = 35
LR = 1e-3

# --- 1. SO-101 6-DoF MuJoCo XML ---

# --- 2. 100개 세트의 위치 1 -> 위치 2 데모 데이터 수집 ---
def generate_point_to_point_episodes(model, data):
    print(f"[*] MuJoCo 환경에서 100세트(총 {NUM_EPISODES * STEPS_PER_EP} steps) 데모 데이터 수집 중...")
    base_pos1 = torch.tensor([-0.6, -0.4, 0.5, -0.3, 0.2, 0.01], dtype=torch.float32)
    base_pos2 = torch.tensor([ 0.7,  0.5, -0.4,  0.6, -0.5, 0.035], dtype=torch.float32)

    all_states = []
    all_actions = []

    for ep in range(NUM_EPISODES):
        pos1_noise = base_pos1 + (torch.rand(DOF) - 0.5) * 0.08
        pos2_noise = base_pos2 + (torch.rand(DOF) - 0.5) * 0.08

        mujoco.mj_resetData(model, data)
        data.qpos[:DOF] = pos1_noise.numpy()
        data.qvel[:DOF] = 0.0
        mujoco.mj_forward(model, data)

        ep_states = []
        ep_actions = []

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

# --- 3. 슬라이딩 윈도우 Dataset ---
class P2PTrajectoryDataset(Dataset):
    def __init__(self, states, actions, history=HISTORY):
        xs, ys = [], []
        num_episodes = states.shape[0]
        for ep in range(num_episodes):
            ep_s = states[ep]
            ep_a = actions[ep]
            for t in range(history - 1, len(ep_s) - 1):
                xs.append(ep_s[t - history + 1 : t + 1])
                ys.append(ep_a[t + 1])

        self.x = torch.stack(xs)
        self.y = torch.stack(ys)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# --- 4. 바닐라 RNN BC 모델 ---
class VanillaRNN_BC(nn.Module):
    def __init__(self, dof=DOF, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.projector = nn.Linear(dof, embed_dim)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, dof)

    def forward(self, x):
        embedded = torch.relu(self.projector(x))
        _, hidden = self.rnn(embedded)
        pred_action = self.head(hidden[-1])
        return pred_action

# --- 5. GLFW 기반 실시간 렌더링 함수 ---
def run_glfw_simulation(mj_model, mj_data, model):
    if not glfw.init():
        raise RuntimeError("GLFW 초기화에 실패했습니다.")

    # 윈도우 생성 (너비 1200, 높이 900)
    window = glfw.create_window(1200, 900, "SO101 RNN Behavior Cloning (GLFW)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW 윈도우 생성에 실패했습니다.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # MuJoCo 시각화 구조체 초기화
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(mj_model, maxgeom=1000)
    con = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # 기본 카메라 시점 설정
    cam.azimuth = 90.0
    cam.elevation = -25.0
    cam.distance = 1.2
    cam.lookat = [0.0, 0.0, 0.2]

    eval_pos1 = torch.tensor([-0.6, -0.4, 0.5, -0.3, 0.2, 0.01])
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[:DOF] = eval_pos1.numpy()
    mujoco.mj_forward(mj_model, mj_data)

    state_buffer = [mj_data.qpos[:DOF].copy() for _ in range(HISTORY)]
    step_cnt = 0

    print("\n[*] GLFW 창에서 로봇 제어 궤적이 실시간 렌더링됩니다 (창을 닫거나 ESC를 누르면 종료).")

    while not glfw.window_should_close(window):
        step_start = time.time()

        # 1) 과거 버퍼 -> RNN 추론
        input_tensor = torch.tensor(state_buffer, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_action = model(input_tensor).squeeze(0).numpy()

        # 2) 시뮬레이터 적용
        mj_data.ctrl[:DOF] = pred_action
        mujoco.mj_step(mj_model, mj_data)

        # 3) 버퍼 갱신
        state_buffer.pop(0)
        state_buffer.append(mj_data.qpos[:DOF].copy())

        # 100스텝 후 반복 시연을 위해 위치 1로 리셋
        step_cnt += 1
        if step_cnt >= STEPS_PER_EP + 20:
            mujoco.mj_resetData(mj_model, mj_data)
            mj_data.qpos[:DOF] = eval_pos1.numpy()
            mujoco.mj_forward(mj_model, mj_data)
            state_buffer = [mj_data.qpos[:DOF].copy() for _ in range(HISTORY)]
            step_cnt = 0

        # 4) GLFW 뷰포트 업데이트 및 MuJoCo 씬 렌더링
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)

        mujoco.mjv_updateScene(mj_model, mj_data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn, con)

        glfw.swap_buffers(window)
        glfw.poll_events()

        # 시뮬레이션 타임스텝 동기화
        time_until_next = mj_model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

    # GLFW 종료 정리
    glfw.terminate()

# --- 6. 메인 파이프라인 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 사용 디바이스: {device}")

    mj_model = mujoco.MjModel.from_xml_path("./scene.xml")
    mj_data = mujoco.MjData(mj_model)

    states, actions = generate_point_to_point_episodes(mj_model, mj_data)

    train_ds = P2PTrajectoryDataset(states[:80], actions[:80])
    test_ds = P2PTrajectoryDataset(states[80:], actions[80:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = VanillaRNN_BC().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("\n[*] 바닐라 RNN Behavior Cloning 학습 시작...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for seq, act in train_loader:
            seq, act = seq.to(device), act.to(device)
            pred = model(seq)
            loss = criterion(pred, act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seq.size(0)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train MSE: {total_loss / len(train_ds):.6f}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for seq, act in test_loader:
            seq, act = seq.to(device), act.to(device)
            pred = model(seq)
            test_loss += criterion(pred, act).item() * seq.size(0)
    print(f"\n[Test MSE Loss]: {test_loss / len(test_ds):.7f}")

    torch.save(model.state_dict(), "so101_rnn_bc_p2p.pt")
    print("[*] 모델 가중치 저장 완료: so101_rnn_bc_p2p.pt")

    # 학습 완료 후 CPU 전환 및 GLFW 시뮬레이터 실행
    model.cpu()
    run_glfw_simulation(mj_model, mj_data, model)

if __name__ == "__main__":
    main()