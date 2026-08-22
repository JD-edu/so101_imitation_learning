"""
so101_mlp_inference_glfw.py
SO101 6-DoF MLP Behavior Cloning GLFW 기반 추론 코드 (미학습 시작 위치 검증)
"""

import time
import glfw
import torch
import torch.nn as nn
import mujoco

# --- 하이퍼파라미터 ---
DOF = 6
MODEL_PATH = "so101_mlp_bc_p2p.pt"
STEPS_PER_EP = 50
XML_PATH = "./scene.xml"  # 로봇 씬 파일 경로

# --- 마우스 조작을 위한 전역 변수 ---
button_left = False
button_middle = False
button_right = False
last_x = 0
last_y = 0


# --- GLFW 마우스 콜백 함수들 ---
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


# --- 1. MLP 모델 클래스 정의 ---
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
        # state: [1, DOF] -> action: [1, DOF]
        return self.net(state)


def main():
    global mj_model, cam, scn

    # 1. 학습된 모델 가중치 로드
    model = BehaviorCloningMLP(dof=DOF)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print(f"[*] 모델 가중치 로드 성공: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[!] 에러: '{MODEL_PATH}' 파일이 없습니다. 학습 코드를 먼저 실행하세요.")
        return

    model.eval()

    # 2. MuJoCo 모델 및 데이터 로드
    try:
        mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        mj_data = mujoco.MjData(mj_model)
        print(f"[*] MuJoCo XML 로드 성공: {XML_PATH}")
    except Exception as e:
        print(f"[!] XML 로드 실패: {e}")
        return

    # 3. GLFW 초기화 및 윈도우 생성
    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패")

    window = glfw.create_window(1200, 900, "SO101 MLP BC Inference (GLFW Viewer)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW 윈도우 생성 실패")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # 콜백 등록
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_scroll_callback(window, scroll)

    # 4. MuJoCo 렌더링 컨텍스트 및 카메라 설정
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(mj_model, maxgeom=1000)
    con = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    cam.azimuth = 90.0
    cam.elevation = -25.0
    cam.distance = 1.2
    cam.lookat = [0.0, 0.0, 0.2]

    # 5. 훈련 시 사용하지 않은 새로운 시작 위치
    UNSEEN_START_POS = torch.tensor([-0.10, -0.80, 0.15, 0.20, -0.10, 0.000], dtype=torch.float32)

    def reset_to_unseen_start():
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:DOF] = UNSEEN_START_POS.numpy()
        mujoco.mj_forward(mj_model, mj_data)

    reset_to_unseen_start()

    print("\n" + "=" * 60)
    print("[*] 훈련 시작점: [-0.60, -0.40,  0.50, -0.30,  0.20, 0.010]")
    print(f"[*] 추론 시작점: {UNSEEN_START_POS.numpy().round(3).tolist()} (미학습 자세)")
    print("[*] GLFW 기반 MLP Closed-Loop 실시간 추론을 시작합니다 (ESC나 창 닫기로 종료)...")
    print("=" * 60 + "\n")

    step_cnt = 0

    # 6. GLFW 메인 렌더링 & 추론 루프
    while not glfw.window_should_close(window):
        step_start = time.time()

        # (1) 현재 6축 관절 각도 센싱: Shape [1, 6]
        current_state = torch.tensor(mj_data.qpos[:DOF], dtype=torch.float32).unsqueeze(0)

        # (2) MLP 모델로 단일 스텝 예측: q(t) -> q(t+1)
        with torch.no_grad():
            pred_action = model(current_state).squeeze(0).numpy()

        # (3) 액추에이터 제어 명령 인가 및 물리 1스텝 전진
        mj_data.ctrl[:DOF] = pred_action
        mujoco.mj_step(mj_model, mj_data)

        # 120스텝 후 다시 미학습 시작점으로 리셋 (반복 시연)
        step_cnt += 1
        if step_cnt >= STEPS_PER_EP + 20:
            reset_to_unseen_start()
            step_cnt = 0

        # (4) GLFW 화면 렌더링
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)

        mujoco.mjv_updateScene(mj_model, mj_data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn, con)

        glfw.swap_buffers(window)
        glfw.poll_events()

        # (5) 물리 타임스텝 동기화
        time_until_next = mj_model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

    glfw.terminate()


if __name__ == "__main__":
    main()