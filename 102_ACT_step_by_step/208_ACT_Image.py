"""
208_ACT_Image.py
SO101 6-DoF Full Visual ACT (CVAE + ResNet-18 + Transformer Encoder-Decoder)
- Fix: OpenGL Context Initialization via Hidden GLFW Window before Data Generation
- Camera: Overhead/Foreground Fixed Camera (Offscreen RGB Rendering)
- Hyperparameters: KL Weight (0.01), Episodes (200), Epochs (100)
"""

import math
import time
import glfw
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import mujoco

# 시드 설정
torch.manual_seed(42)
np.random.seed(42)

# --- 하이퍼파라미터 정의 ---
DOF = 6
NUM_EPISODES = 200      # 데이터 증량
STEPS_PER_EP = 100
CHUNK_SIZE = 30         # 미래 K=30스텝 일괄 예측
LATENT_DIM = 16         # CVAE 잠재 변수 z 차원
D_MODEL = 128           # 이미지 피처 융합을 위해 d_model 확장
NHEAD = 4
NUM_LAYERS = 3
BATCH_SIZE = 32
EPOCHS = 100            # 에포크 증량
LR = 3e-4               # 안정적인 수렴을 위한 Learning Rate
KL_WEIGHT = 0.01        # KL Weight 하향 (Reconstruction 집중)
IMG_W, IMG_H = 128, 128
MODEL_PATH = "visual_act_final.pt"

# --- 1. SO-101 로봇 및 전경 카메라 XML ---

# --- 2. 이미지 + 궤적 수집 루틴 ---
def capture_foreground_camera(model, data, scn, con, cam_id, width=IMG_W, height=IMG_H):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cam_id
    opt = mujoco.MjvOption()

    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjr_render(viewport, scn, con)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, viewport, con)
    rgb = np.flipud(rgb)
    tensor_img = torch.tensor(rgb.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0
    return tensor_img

def generate_visual_p2p_dataset(model, data, scn, con):
    print(f"[*] MuJoCo 전경 카메라 렌더링 & {NUM_EPISODES}세트 멀티모달 데이터 수집 중...")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_cam")

    base_pos1 = torch.tensor([-0.6, -0.4, 0.5, -0.3, 0.2, 0.01], dtype=torch.float32)
    base_pos2 = torch.tensor([ 0.7,  0.5, -0.4,  0.6, -0.5, 0.035], dtype=torch.float32)
    DEG10_RAD = 10.0 * (math.pi / 180.0)

    all_images, all_joints, all_actions = [], [], []

    for ep in range(NUM_EPISODES):
        joint_noise1 = (torch.rand(5) * 2.0 - 1.0) * DEG10_RAD
        gripper_noise1 = (torch.rand(1) * 2.0 - 1.0) * 0.005
        start_pos = base_pos1 + torch.cat([joint_noise1, gripper_noise1])

        joint_noise2 = (torch.rand(5) * 2.0 - 1.0) * DEG10_RAD
        gripper_noise2 = (torch.rand(1) * 2.0 - 1.0) * 0.005
        target_pos = base_pos2 + torch.cat([joint_noise2, gripper_noise2])

        mujoco.mj_resetData(model, data)
        data.qpos[:DOF] = start_pos.numpy()
        data.qvel[:DOF] = 0.0
        mujoco.mj_forward(model, data)

        ep_imgs, ep_joints, ep_acts = [], [], []

        for step in range(STEPS_PER_EP):
            tau = step / (STEPS_PER_EP - 1)
            smooth_s = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
            target_ctrl = (1.0 - smooth_s) * start_pos + smooth_s * target_pos

            # 시각/관절 관측
            img = capture_foreground_camera(model, data, scn, con, cam_id)
            current_qpos = data.qpos[:DOF].copy()

            data.ctrl[:DOF] = target_ctrl.numpy()
            mujoco.mj_step(model, data)

            ep_imgs.append(img)
            ep_joints.append(torch.tensor(current_qpos, dtype=torch.float32))
            ep_acts.append(target_ctrl)

        all_images.append(torch.stack(ep_imgs))
        all_joints.append(torch.stack(ep_joints))
        all_actions.append(torch.stack(ep_acts))

    return torch.stack(all_images), torch.stack(all_joints), torch.stack(all_actions)

# --- 3. Dataset ---
class VisualACTDataset(Dataset):
    def __init__(self, images, joints, actions, chunk_size=CHUNK_SIZE):
        self.samples = []
        num_episodes = images.shape[0]

        for ep in range(num_episodes):
            ep_imgs = images[ep]
            ep_j = joints[ep]
            ep_a = actions[ep]
            last_t = len(ep_j) - chunk_size - 1
            for t in range(0, last_t + 1):
                self.samples.append((ep_imgs[t], ep_j[t], ep_a[t + 1 : t + 1 + chunk_size]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, joint, action_chunk = self.samples[idx]
        return img, joint, action_chunk

# --- 4. CVAE Encoder ---
class CVAEEncoder(nn.Module):
    def __init__(self, dof=DOF, chunk_size=CHUNK_SIZE, latent_dim=LATENT_DIM):
        super().__init__()
        in_dim = dof + (chunk_size * dof)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, current_joint, action_chunk):
        x = torch.cat([current_joint, action_chunk.flatten(start_dim=1)], dim=1)
        feat = self.net(x)
        mu = self.fc_mu(feat)
        logvar = torch.clamp(self.fc_logvar(feat), min=-10.0, max=10.0)
        return mu, logvar

# --- 5. Full Visual ACT Policy ---
class VisualACT(nn.Module):
    def __init__(self, dof=DOF, chunk_size=CHUNK_SIZE, latent_dim=LATENT_DIM, 
                 d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS):
        super().__init__()
        self.dof = dof
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        # 1) Vision Backbone (ResNet-18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.vision_projector = nn.Linear(512, d_model)

        # 2) CVAE & 토큰 투영기
        self.cvae_encoder = CVAEEncoder(dof, chunk_size, latent_dim)
        self.joint_projector = nn.Linear(dof, d_model)
        self.latent_projector = nn.Linear(latent_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 3, d_model))

        # 3) Transformer Encoder & Decoder
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, dof)

        nn.init.normal_(self.action_queries, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img, joint, action_chunk=None):
        batch_size = img.size(0)

        if self.training and action_chunk is not None:
            mu, logvar = self.cvae_encoder(joint, action_chunk)
            z = self.reparameterize(mu, logvar)
        else:
            mu, logvar = None, None
            z = torch.zeros(batch_size, self.latent_dim, device=img.device)

        z_tok = self.latent_projector(z).unsqueeze(1)
        joint_tok = self.joint_projector(joint).unsqueeze(1)
        
        vis_feat = self.vision_backbone(img).flatten(start_dim=1)
        vis_tok = self.vision_projector(vis_feat).unsqueeze(1)

        enc_input = torch.cat([z_tok, joint_tok, vis_tok], dim=1) + self.pos_embedding
        memory = self.transformer_encoder(enc_input)

        query = self.action_queries.expand(batch_size, -1, -1)
        dec_out = self.transformer_decoder(tgt=query, memory=memory)
        pred_actions = self.action_head(dec_out)

        return pred_actions, mu, logvar

def compute_loss(pred_actions, target_actions, mu, logvar, kl_weight=KL_WEIGHT):
    recon_loss = nn.functional.l1_loss(pred_actions, target_actions)
    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss

# --- 6. GLFW 실시간 Closed-Loop 추론 시연 루프 ---
def run_visual_glfw_eval(window, mj_model, mj_data, scn, con, model):
    glfw.show_window(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "front_cam")

    cam.azimuth, cam.elevation, cam.distance, cam.lookat = 90.0, -25.0, 1.2, [0.0, 0.0, 0.2]

    base_pos1 = torch.tensor([-0.6, -0.4, 0.5, -0.3, 0.2, 0.01])
    DEG10_RAD = 10.0 * (math.pi / 180.0)
    weights = np.exp(-0.05 * np.arange(CHUNK_SIZE))

    def reset_env():
        noise = (torch.rand(5) * 2.0 - 1.0) * DEG10_RAD
        g_noise = (torch.rand(1) * 2.0 - 1.0) * 0.005
        init_pos = base_pos1 + torch.cat([noise, g_noise])

        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:DOF] = init_pos.numpy()
        mujoco.mj_forward(mj_model, mj_data)
        return 0, {}

    curr_step, all_time_actions = reset_env()
    print("\n[*] Visual ACT 실시간 시연 중 (전경 카메라 관측 기반 제어)...")

    while not glfw.window_should_close(window):
        step_start = time.time()

        # 1) 전경 카메라 영상 및 관절 획득
        img_tensor = capture_foreground_camera(mj_model, mj_data, scn, con, cam_id).unsqueeze(0)
        joint_tensor = torch.tensor(mj_data.qpos[:DOF].copy(), dtype=torch.float32).unsqueeze(0)

        # 2) Visual ACT 추론 (z=0)
        with torch.no_grad():
            pred_chunk, _, _ = model(img_tensor, joint_tensor)
            pred_chunk = pred_chunk.squeeze(0).numpy()

        # 3) Temporal Ensembling 가중 평균
        for i in range(CHUNK_SIZE):
            t_future = curr_step + 1 + i
            if t_future not in all_time_actions:
                all_time_actions[t_future] = []
            all_time_actions[t_future].append((pred_chunk[i], weights[i]))

        target_t = curr_step + 1
        actions_at_t = all_time_actions[target_t]
        weighted_sum, total_w = np.zeros(DOF), 0.0
        for act, w in actions_at_t:
            weighted_sum += act * w
            total_w += w
        final_action = weighted_sum / total_w
        del all_time_actions[target_t]

        # 4) 로봇 제어
        mj_data.ctrl[:DOF] = final_action
        mujoco.mj_step(mj_model, mj_data)
        curr_step += 1

        if curr_step >= STEPS_PER_EP + 20:
            curr_step, all_time_actions = reset_env()

        # 5) 시각화 렌더링
        w, h = glfw.get_framebuffer_size(window)
        mujoco.mjv_updateScene(mj_model, mj_data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, w, h), scn, con)
        glfw.swap_buffers(window)
        glfw.poll_events()

        t_sleep = mj_model.opt.timestep - (time.time() - step_start)
        if t_sleep > 0:
            time.sleep(t_sleep)

    glfw.terminate()

# --- 7. 메인 파이프라인 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 사용 디바이스: {device}")

    # (중요) MuJoCo MjrContext 생성을 위한 OpenGL/GLFW 컨텍스트 초기화
    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE) # 데이터 수집 단계에서는 창을 숨김
    window = glfw.create_window(1200, 900, "SO101 Visual ACT Final Evaluation", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW 윈도우 생성 실패")
    glfw.make_context_current(window)

    mj_model = mujoco.MjModel.from_xml_path('./scene.xml')
    mj_data = mujoco.MjData(mj_model)

    scn = mujoco.MjvScene(mj_model, maxgeom=1000)
    con = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # 데이터 수집 (OpenGL Context가 활성화된 상태에서 정상 수행)
    images, joints, actions = generate_visual_p2p_dataset(mj_model, mj_data, scn, con)

    split = int(NUM_EPISODES * 0.8)
    train_ds = VisualACTDataset(images[:split], joints[:split], actions[:split])
    test_ds = VisualACTDataset(images[split:], joints[split:], actions[split:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"[*] 총 학습 샘플: {len(train_ds)} | 검증 샘플: {len(test_ds)}")

    model = VisualACT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n[*] Visual ACT 최종 학습 시작...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

        for img, joint, chunk in train_loader:
            img, joint, chunk = img.to(device), joint.to(device), chunk.to(device)
            pred_chunk, mu, logvar = model(img, joint, chunk)
            loss, recon_l, kl_l = compute_loss(pred_chunk, chunk, mu, logvar, kl_weight=KL_WEIGHT)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * img.size(0)
            total_recon += recon_l.item() * img.size(0)
            total_kl += kl_l.item() * img.size(0)

        scheduler.step()

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{EPOCHS} | Loss: {total_loss/len(train_ds):.5f} "
                  f"(Recon L1: {total_recon/len(train_ds):.5f}, KL: {total_kl/len(train_ds):.5f})")

    # 정량 평가
    model.eval()
    test_recon = 0.0
    with torch.no_grad():
        for img, joint, chunk in test_loader:
            img, joint, chunk = img.to(device), joint.to(device), chunk.to(device)
            pred_chunk, _, _ = model(img, joint)
            test_recon += nn.functional.l1_loss(pred_chunk, chunk).item() * img.size(0)

    print(f"\n[최종 Test Reconstruction L1 Loss (z=0)]: {test_recon / len(test_ds):.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[*] 모델 저장 완료: {MODEL_PATH}")

    # 학습 완료 후 숨겨둔 창을 표시하며 GLFW 실시간 시뮬레이션 평가 구동
    model.cpu()
    run_visual_glfw_eval(window, mj_model, mj_data, scn, con, model)

if __name__ == "__main__":
    main()