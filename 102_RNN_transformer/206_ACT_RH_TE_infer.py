"""
2_action_chunking_rh_te_inference_glfw.py
SO101 Action Chunking with Receding Horizon & Temporal Ensembling
- Random Start & Random Target Poses (+-10 deg)
- Small Red Target Marker (Sphere r=0.015m)
- Mouse Left-Click Disturbance Injection
- Receding Horizon (Re-planning every step) + Temporal Ensembling Filter
"""

import math
import time
import glfw
import numpy as np
import torch
import torch.nn as nn
import mujoco

DOF = 6
HISTORY = 10
CHUNK_SIZE = 30
MODEL_PATH = "action_chunking_transformer.pt"
XML_PATH = "./scene.xml"
EXP_WEIGHT_M = 0.05  # Temporal Ensembling 지수 감쇠 계수

# 마우스 인터랙션 콜백 변수
button_left = False
button_middle = False
button_right = False
last_x = 0
last_y = 0
trigger_disturbance = False

def mouse_button(window, button, action, mods):
    global button_left, button_middle, button_right, trigger_disturbance
    button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

    # 마우스 좌클릭 시 외란(Disturbance) 인가
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        trigger_disturbance = True

def mouse_move(window, xpos, ypos):
    global last_x, last_y, button_left, button_middle, button_right, mj_model, scn, cam
    dx = xpos - last_x
    dy = ypos - last_y
    last_x = xpos
    last_y = ypos

    if not (button_right or button_middle):
        return

    width, height = glfw.get_window_size(window)
    mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or \
                glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

    if button_right:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM

    mujoco.mjv_moveCamera(mj_model, action, dx / height, dy / height, scn, cam)

def scroll(window, xoffset, yoffset):
    global mj_model, scn, cam
    mujoco.mjv_moveCamera(mj_model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, scn, cam)

# --- Action Chunking 신경망 정의 ---
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
        pred_flat = self.action_head(x[:, -1])
        return pred_flat.view(-1, self.chunk_size, self.dof)

# 임의의 관절 각도(target_qpos)에 따른 End-Effector 3D 월드 좌표 계산 (FK)
def get_target_3d_position(mj_model, target_qpos):
    temp_data = mujoco.MjData(mj_model)
    temp_data.qpos[:DOF] = target_qpos
    mujoco.mj_forward(mj_model, temp_data)

    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    if site_id != -1:
        return temp_data.site_xpos[site_id].copy()
    return temp_data.xpos[-1].copy()

def main():
    global mj_model, cam, scn, trigger_disturbance

    # 1. 모델 가중치 로드
    model = ActionChunkTransformer()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print(f"[*] 모델 로드 성공: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[!] 에러: '{MODEL_PATH}' 가중치 파일이 없습니다. 학습 코드를 먼저 실행하세요.")
        return

    model.eval()

    # 2. scene.xml 파일 로드
    try:
        mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        mj_data = mujoco.MjData(mj_model)
        print(f"[*] MuJoCo XML 로드 성공: {XML_PATH}")
    except Exception as e:
        print(f"[!] XML 로드 실패: {e}")
        return

    # 3. GLFW 초기화
    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패")

    window = glfw.create_window(1200, 900, "SO101 ACT (RH + TE + Disturbance)", None, None)
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

    base_pos1 = torch.tensor([-0.6, -0.4, 0.5, -0.3, 0.2, 0.01], dtype=torch.float32)
    base_pos2 = torch.tensor([ 0.7,  0.5, -0.4,  0.6, -0.5, 0.035], dtype=torch.float32)
    DEG10_RAD = 10.0 * (math.pi / 180.0)  # +-10도 범위

    # Temporal Ensembling 지수 가중치 (w_i = exp(-m * i))
    weights = np.exp(-EXP_WEIGHT_M * np.arange(CHUNK_SIZE))

    def reset_episode():
        # 학습 범위와 동일한 +-10도 랜덤 시작점/목표점 생성
        joint_noise1 = (torch.rand(5) * 2.0 - 1.0) * DEG10_RAD
        gripper_noise1 = (torch.rand(1) * 2.0 - 1.0) * 0.005
        pos1 = base_pos1 + torch.cat([joint_noise1, gripper_noise1])

        joint_noise2 = (torch.rand(5) * 2.0 - 1.0) * DEG10_RAD
        gripper_noise2 = (torch.rand(1) * 2.0 - 1.0) * 0.005
        pos2 = base_pos2 + torch.cat([joint_noise2, gripper_noise2])

        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:DOF] = pos1.numpy()
        mujoco.mj_forward(mj_model, mj_data)

        state_buf = [mj_data.qpos[:DOF].copy() for _ in range(HISTORY)]
        target_pos_3d = get_target_3d_position(mj_model, pos2.numpy())

        # Temporal Ensembling 중첩 버퍼 초기화
        all_time_actions = {}
        return state_buf, target_pos_3d, all_time_actions

    state_buffer, target_marker_pos, all_time_actions = reset_episode()
    current_time_step = 0

    print("\n" + "=" * 70)
    print("[*] Receding Horizon (매 타임스텝 재추론) + Temporal Ensembling 가동 중")
    print("[*] 마우스 좌클릭: 외란(Impulse) 토크 주입 -> 스스로 안정적 복원 확인")
    print("[*] 작은 붉은 구체(r=0.015m): 현재 에피소드의 랜덤 도달 목표 지점")
    print("=" * 70 + "\n")

    while not glfw.window_should_close(window):
        step_start = time.time()

        # 1) 외란 주입 처리 (마우스 좌클릭 시 관절 속도에 급격한 토크 인가)
        if trigger_disturbance:
            disturbance_torque = (np.random.rand(DOF) - 0.5) * 8.0
            mj_data.qvel[:DOF] += disturbance_torque
            trigger_disturbance = False
            print(f"[!] 외란 주입됨: delta_qvel = {disturbance_torque.round(2)}")

        # 2) [Receding Horizon]: 매 스텝 최신 버퍼 기반 미래 K=30스텝 재예측
        input_tensor = torch.tensor(state_buffer, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_chunk = model(input_tensor).squeeze(0).numpy()  # [CHUNK_SIZE, DOF]

        # 3) [Temporal Ensembling]: 겹치는 미래 시점들에 가중치와 함께 누적
        for i in range(CHUNK_SIZE):
            future_t = current_time_step + 1 + i
            if future_t not in all_time_actions:
                all_time_actions[future_t] = []
            all_time_actions[future_t].append((pred_chunk[i], weights[i]))

        # 4) 현재 스텝에 도달한 예측들의 지수 가중 평균 산출
        target_t = current_time_step + 1
        actions_at_t = all_time_actions[target_t]

        weighted_sum = np.zeros(DOF)
        total_weight = 0.0
        for act, w in actions_at_t:
            weighted_sum += act * w
            total_weight += w
        final_action = weighted_sum / total_weight

        del all_time_actions[target_t]

        # 5) 액추에이터 제어 및 롤링 버퍼 갱신
        mj_data.ctrl[:DOF] = final_action
        mujoco.mj_step(mj_model, mj_data)

        state_buffer.pop(0)
        state_buffer.append(mj_data.qpos[:DOF].copy())
        current_time_step += 1

        # 120스텝 실행 후 새로운 랜덤 에피소드로 리셋
        if current_time_step >= 120:
            state_buffer, target_marker_pos, all_time_actions = reset_episode()
            current_time_step = 0

        # 6) GLFW 화면 렌더링 & 작은 목표 마커 구체 추가
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)

        mujoco.mjv_updateScene(mj_model, mj_data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)

        # 씬에 작은 크기(반지름 0.015m)의 붉은 반투명 구체 마커 렌더링
        if scn.ngeom < scn.maxgeom:
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.015, 0, 0]),           # 작은 구체 반지름 (15mm)
                target_marker_pos,                # 목표 위치 3D 좌표
                np.eye(3).flatten(),
                np.array([1.0, 0.1, 0.1, 0.7])     # 선명한 반투명 붉은색 (RGBA)
            )
            scn.ngeom += 1

        mujoco.mjr_render(viewport, scn, con)
        glfw.swap_buffers(window)
        glfw.poll_events()

        time_until_next = mj_model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

    glfw.terminate()

if __name__ == "__main__":
    main()