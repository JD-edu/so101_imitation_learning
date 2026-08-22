"""
3_cvae_act_inference_glfw.py
SO101 CVAE + Action Chunking Transformer (ACT) Full Inference Loop
- CVAE Latent Token z=0 for Standard Mode Evaluation
- Receding Horizon (Every Step Re-planning) + Temporal Ensembling Filter
- Random Start & Random Target Poses (+-10 deg)
- Small Red Target Marker (Sphere r=0.015m)
- Mouse Left-Click Disturbance Injection
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
LATENT_DIM = 16
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
EXP_WEIGHT_M = 0.05  # Temporal Ensembling 지수 감쇠 계수
STEPS_PER_EP = 100
MODEL_PATH = "cvae_act_transformer.pt"
XML_PATH = "./scene.xml"

# GLFW 마우스 인터랙션 콜백 변수
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

    # 마우스 좌클릭 시 외란 인가 트리거
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

# --- 1. CVAE Encoder 구조 정의 (가중치 로드를 위해 필요) ---
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

# --- 2. CVAE Transformer ACT 모델 ---
class CVAE_ACT(nn.Module):
    def __init__(self, dof=DOF, history=HISTORY, chunk_size=CHUNK_SIZE, latent_dim=LATENT_DIM, 
                 d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS):
        super().__init__()
        self.dof = dof
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        self.cvae_encoder = CVAEEncoder(dof, history, chunk_size, latent_dim, d_model)

        self.state_projector = nn.Linear(dof, d_model)
        self.latent_projector = nn.Linear(latent_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, history + 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, dof)

    def forward(self, state_seq, action_chunk=None):
        batch_size = state_seq.size(0)

        # 추론 모드: z = 0 벡터 주입 (표준 대표 행동 모달리티)
        z = torch.zeros(batch_size, self.latent_dim, device=state_seq.device)

        z_token = self.latent_projector(z).unsqueeze(1)
        state_tokens = self.state_projector(state_seq)
        enc_input = torch.cat([z_token, state_tokens], dim=1)
        enc_input = enc_input + self.pos_embedding

        memory = self.transformer_encoder(enc_input)

        query = self.action_queries.expand(batch_size, -1, -1)
        dec_out = self.transformer_decoder(tgt=query, memory=memory)
        pred_actions = self.action_head(dec_out)

        return pred_actions, None, None

# Forward Kinematics를 이용해 특정 관절 각도의 말단(End-Effector) 3D 월드 좌표 계산
def get_target_3d_position(mj_model, target_qpos):
    temp_data = mujoco.MjData(mj_model)
    temp_data.qpos[:DOF] = target_qpos
    mujoco.mj_forward(mj_model, temp_data)

    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    if site_id != -1:
        return temp_data.site_xpos[site_id].copy()
    return temp_data.xpos[-1].copy()

# --- 3. 메인 추론 루프 ---
def main():
    global mj_model, cam, scn, trigger_disturbance

    # 1) 가중치 로드
    model = CVAE_ACT()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print(f"[*] CVAE-ACT 모델 로드 성공: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[!] 에러: '{MODEL_PATH}' 가중치 파일이 없습니다. 학습 코드를 먼저 실행하세요.")
        return

    model.eval()

    # 2) scene.xml 로드
    try:
        mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        mj_data = mujoco.MjData(mj_model)
        print(f"[*] MuJoCo XML 로드 성공: {XML_PATH}")
    except Exception as e:
        print(f"[!] XML 로드 실패: {e}")
        return

    # 3) GLFW 초기화
    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패")

    window = glfw.create_window(1200, 900, "SO101 Full ACT (CVAE + RH + TE + Disturbance)", None, None)
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
    DEG10_RAD = 10.0 * (math.pi / 180.0)

    weights = np.exp(-EXP_WEIGHT_M * np.arange(CHUNK_SIZE))

    # 랜덤 시작점 및 목표점 리셋 함수
    def reset_episode():
        joint_noise1 = (torch.rand(5) * 2.0 - 1.0) * DEG10_RAD
        gripper_noise1 = (torch.rand(1) * 2.0 - 1.0) * 0.005
        start_pos = base_pos1 + torch.cat([joint_noise1, gripper_noise1])

        joint_noise2 = (torch.rand(5) * 2.0 - 1.0) * DEG10_RAD
        gripper_noise2 = (torch.rand(1) * 2.0 - 1.0) * 0.005
        target_pos = base_pos2 + torch.cat([joint_noise2, gripper_noise2])

        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:DOF] = start_pos.numpy()
        mujoco.mj_forward(mj_model, mj_data)

        state_buf = [mj_data.qpos[:DOF].copy() for _ in range(HISTORY)]
        target_marker_pos = get_target_3d_position(mj_model, target_pos.numpy())
        all_time_actions = {}
        return state_buf, target_marker_pos, all_time_actions

    state_buffer, target_marker_pos, all_time_actions = reset_episode()
    current_time_step = 0

    print("\n" + "=" * 75)
    print("[*] Full ACT 추론 파이프라인 가동 (CVAE z=0 + RH + Temporal Ensembling)")
    print("[*] 마우스 좌클릭: 임펄스 외란(Disturbance) 토크 인가")
    print("[*] 작은 붉은 마커(r=0.015m): 현재 에피소드의 랜덤 도달 목표 위치")
    print("=" * 75 + "\n")

    while not glfw.window_should_close(window):
        step_start = time.time()

        # 1) 마우스 클릭 외란 처리
        if trigger_disturbance:
            disturbance_torque = (np.random.rand(DOF) - 0.5) * 8.0
            mj_data.qvel[:DOF] += disturbance_torque
            trigger_disturbance = False
            print(f"[!] 외란 인가됨: delta_qvel = {disturbance_torque.round(2)}")

        # 2) [Receding Horizon]: 매 스텝 미래 K=30스텝 일괄 재추론 (z=0)
        input_tensor = torch.tensor(state_buffer, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_chunk, _, _ = model(input_tensor)
            pred_chunk = pred_chunk.squeeze(0).numpy()

        # 3) [Temporal Ensembling]: 미래 스텝 예측을 가중치와 함께 버퍼에 누적
        for i in range(CHUNK_SIZE):
            future_t = current_time_step + 1 + i
            if future_t not in all_time_actions:
                all_time_actions[future_t] = []
            all_time_actions[future_t].append((pred_chunk[i], weights[i]))

        # 4) 현재 스텝 목표 제어값 산출 (지수 감쇠 가중 평균)
        target_t = current_time_step + 1
        actions_at_t = all_time_actions[target_t]
        
        weighted_sum = np.zeros(DOF)
        total_weight = 0.0
        for act, w in actions_at_t:
            weighted_sum += act * w
            total_weight += w
        final_action = weighted_sum / total_weight

        del all_time_actions[target_t]

        # 5) 액추에이터 제어 및 버퍼 갱신
        mj_data.ctrl[:DOF] = final_action
        mujoco.mj_step(mj_model, mj_data)

        state_buffer.pop(0)
        state_buffer.append(mj_data.qpos[:DOF].copy())
        current_time_step += 1

        # 120스텝 후 새로운 랜덤 에피소드로 리셋
        if current_time_step >= STEPS_PER_EP + 20:
            state_buffer, target_marker_pos, all_time_actions = reset_episode()
            current_time_step = 0

        # 6) 화면 렌더링 & 작은 적색 마커 표시
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)

        mujoco.mjv_updateScene(mj_model, mj_data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)

        # 목표 위치에 작은 붉은 구체(반지름 15mm) 시각화
        if scn.ngeom < scn.maxgeom:
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.015, 0, 0]),           # 작은 반지름 (15mm)
                target_marker_pos,                # 목표 위치 좌표
                np.eye(3).flatten(),
                np.array([1.0, 0.1, 0.1, 0.75])    # 선명한 반투명 적색 (RGBA)
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