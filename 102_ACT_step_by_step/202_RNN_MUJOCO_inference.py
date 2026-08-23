"""
so101_inference_p2p.py
SO101 6-DoF RNN Behavior Cloning Inference Simulation with GLFW Viewer
"""

import time
import glfw
import torch
import torch.nn as nn
import mujoco

# --- 파라미터 정의 ---
DOF = 6
HISTORY = 10
MODEL_PATH = "so101_rnn_bc_p2p.pt"

# --- 1. SO-101 6-DoF MuJoCo XML ---


# --- 2. 바닐라 RNN BC 모델 클래스 정의 ---
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

def main():
    # 1. 모델 가중치 파일 로드
    model = VanillaRNN_BC()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print(f"[*] 모델 가중치 로드 성공: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[!] 에러: '{MODEL_PATH}' 파일이 없습니다. 먼저 이전 학습 코드를 실행해 모델을 저장하세요.")
        return

    model.eval()

    # 2. MuJoCo 및 GLFW 초기화
    mj_model = mujoco.MjModel.from_xml_path("./scene.xml")
    mj_data = mujoco.MjData(mj_model)

    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패")

    window = glfw.create_window(1200, 900, "SO101 RNN BC Inference (Unseen Start Pose)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW 윈도우 생성 실패")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # 시각화 파라미터 세팅
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(mj_model, maxgeom=1000)
    con = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    cam.azimuth = 90.0
    cam.elevation = -25.0
    cam.distance = 1.2
    cam.lookat = [0.0, 0.0, 0.0]

    # 3. [핵심] 훈련 세트와 의도적으로 다른 새로운 시작 위치 설정
    custom_start_pos = torch.tensor([-0.10, -0.80, 0.15, 0.20, -0.10, 0.000], dtype=torch.float32)

    def reset_to_custom_start():
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:DOF] = custom_start_pos.numpy()
        mujoco.mj_forward(mj_model, mj_data)
        # 초기 10개 프레임 버퍼를 새로운 시작 자세로 채움
        return [mj_data.qpos[:DOF].copy() for _ in range(HISTORY)]

    state_buffer = reset_to_custom_start()
    step_cnt = 0

    print("\n" + "=" * 60)
    print(f"[*] 훈련 시작점: [-0.60, -0.40,  0.50, -0.30,  0.20, 0.010]")
    print(f"[*] 새로운 시작점: {custom_start_pos.numpy().round(3).tolist()} (미학습 자세)")
    print("[*] 실시간 Closed-Loop 추론을 시작합니다...")
    print("=" * 60 + "\n")

    while not glfw.window_should_close(window):
        step_start = time.time()

        # (1) 과거 10개 프레임 텐서화 [1, 10, 6]
        input_seq = torch.tensor(state_buffer, dtype=torch.float32).unsqueeze(0)

        # (2) RNN 모델 순전파 (다음 스텝 목표 관절 각도 예측)
        with torch.no_grad():
            pred_action = model(input_seq).squeeze(0).numpy()

        # (3) 예측된 제어값을 액추에이터에 인가하고 물리 시뮬레이션 1스텝 전진
        mj_data.ctrl[:DOF] = pred_action
        mujoco.mj_step(mj_model, mj_data)

        # (4) 관절 상태 롤링 버퍼 갱신
        state_buffer.pop(0)
        state_buffer.append(mj_data.qpos[:DOF].copy())

        # 120스텝 동안 추론 진행 후, 다시 새로운 시작점으로 리셋 (반복 시연)
        step_cnt += 1
        if step_cnt >= 120:
            state_buffer = reset_to_custom_start()
            step_cnt = 0

        # (5) GLFW 화면 렌더링
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

if __name__ == "__main__":
    main()