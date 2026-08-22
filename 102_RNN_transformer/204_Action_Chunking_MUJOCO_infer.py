"""
2_action_chunking_inference_glfw.py
SO101 Action Chunking Transformer Inference on Unseen Start Pose with GLFW Viewer
"""

import time
import glfw
import torch
import torch.nn as nn
import mujoco

DOF = 6
HISTORY = 10
CHUNK_SIZE = 30
STEPS_PER_EP = 100
MODEL_PATH = "action_chunking_transformer.pt"

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

def main():
    global mj_model, cam, scn

    model = ActionChunkTransformer()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print(f"[*] 가중치 로드 성공: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[!] 에러: {MODEL_PATH} 파일이 없습니다. 학습 코드를 먼저 실행하세요.")
        return

    model.eval()

    mj_model = mujoco.MjModel.from_xml_path('./scene.xml')
    mj_data = mujoco.MjData(mj_model)

    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패")

    window = glfw.create_window(1200, 900, "SO101 Action Chunking Inference (GLFW)", None, None)
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

    UNSEEN_START_POS = torch.tensor([-0.10, -0.10, 0.15, 0.20, -0.10, 0.000], dtype=torch.float32)

    def reset_unseen():
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:DOF] = UNSEEN_START_POS.numpy()
        mujoco.mj_forward(mj_model, mj_data)
        return [mj_data.qpos[:DOF].copy() for _ in range(HISTORY)]

    state_buffer = reset_unseen()
    action_plan_queue = []
    step_cnt = 0

    print("\n" + "=" * 60)
    print(f"[*] 훈련 시작점: [-0.60, -0.80,  0.50, -0.30,  0.20, 0.010]")
    print(f"[*] 추론 시작점: {UNSEEN_START_POS.numpy().round(3).tolist()} (미학습 자세)")
    print(f"[*] Action Chunking 추론 방식: {CHUNK_SIZE}스텝마다 1회 추론 및 큐 실행")
    print("=" * 60 + "\n")

    while not glfw.window_should_close(window):
        step_start = time.time()

        # 1) Action Queue가 비었을 때만 K=30스텝 일괄 예측
        if len(action_plan_queue) == 0:
            input_tensor = torch.tensor(state_buffer, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_chunk = model(input_tensor).squeeze(0).numpy()
            action_plan_queue = list(pred_chunk)

        # 2) 큐에서 다음 행동 꺼내어 제어
        next_action = action_plan_queue.pop(0)
        mj_data.ctrl[:DOF] = next_action
        mujoco.mj_step(mj_model, mj_data)

        # 3) 롤링 버퍼 갱신
        state_buffer.pop(0)
        state_buffer.append(mj_data.qpos[:DOF].copy())

        step_cnt += 1
        if step_cnt >= STEPS_PER_EP + 20:
            state_buffer = reset_unseen()
            action_plan_queue.clear()
            step_cnt = 0

        # 4) GLFW 렌더링
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

if __name__ == "__main__":
    main()