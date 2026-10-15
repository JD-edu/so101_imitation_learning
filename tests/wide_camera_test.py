import matplotlib.pyplot as plt
import numpy as np

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(14, 6), dpi=100)

# -------------------------------------------------------------
# 1. 2D 측면 뷰 (Side Elevation View): Pitch 각도 및 차폐 비교
# -------------------------------------------------------------
ax1 = fig.add_subplot(1, 2, 1)

# 작업대 및 타깃 물체
table_x = [0.0, 0.8]
table_z = [0.0, 0.0]
ax1.plot(table_x, table_z, 'k-', lw=3, label='Worktable')
ax1.plot([0.45], [0.05], 's', color='crimson', markersize=14, label='Target Object')

# 로봇 베이스 및 링크 단순화 (장애물/차폐 요소)
robot_base_x = 0.1
robot_base_z = 0.0
ax1.plot([robot_base_x, robot_base_x], [0.0, 0.25], color='#444444', lw=8, solid_capstyle='round')
ax1.plot([robot_base_x, 0.3], [0.25, 0.22], color='#e5b800', lw=7, solid_capstyle='round', label='Robot Link (Obstacle)')

# [현재 시점] 낮고 수평에 가까운 카메라
curr_cam = np.array([-0.2, 0.32])
target = np.array([0.45, 0.05])
ax1.plot(curr_cam[0], curr_cam[1], 'o', color='gray', markersize=10, label='Current Cam (Low Pitch)')
ax1.annotate('Current View\n(Link Occlusion!)', xy=(curr_cam[0], curr_cam[1]), xytext=(-0.35, 0.42),
             arrowprops=dict(arrowstyle="->", color='gray', lw=1.5), fontweight='bold', color='dimgray')
ax1.plot([curr_cam[0], target[0]], [curr_cam[1], target[1]], 'r--', lw=2, label='Current LOS (Occluded by arm)')

# [권장 시점] ALOHA 스타일 하향 40~45도 탑-사이드 카메라
rec_cam = np.array([-0.1, 0.65])
ax1.plot(rec_cam[0], rec_cam[1], 'o', color='#1f77b4', markersize=12, label='Recommended Top Cam')
ax1.annotate('Recommended View\n(Pitch: ~45° Downward)', xy=(rec_cam[0], rec_cam[1]), xytext=(-0.3, 0.75),
             arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.5), fontweight='bold', color='#1f77b4')
ax1.plot([rec_cam[0], target[0]], [rec_cam[1], target[1]], 'g-', lw=2.5, label='Recommended LOS (Clear View)')

# 차폐 영역 강조 표시
ax1.scatter([0.25], [0.2], s=180, facecolors='none', edgecolors='red', lw=2, linestyle=':')
ax1.text(0.26, 0.15, 'Occlusion Zone', color='red', fontsize=9, fontweight='bold')

ax1.set_xlim([-0.45, 0.85])
ax1.set_ylim([-0.05, 0.95])
ax1.set_aspect('equal')
ax1.set_title('[Side View] Pitch Angle & Line of Sight (LOS)', fontsize=12, fontweight='bold')
ax1.set_xlabel('X Distance [m]')
ax1.set_ylabel('Z Height [m]')
ax1.legend(loc='upper right', frameon=True, fontsize=8.5)

# -------------------------------------------------------------
# 2. 3D 공간 뷰 (Workspace Layout): 양팔 로봇 및 카메라 시점
# -------------------------------------------------------------
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# 테이블
tx, ty = np.meshgrid(np.linspace(0.1, 0.7, 5), np.linspace(-0.35, 0.35, 5))
tz = np.zeros_like(tx)
ax2.plot_wireframe(tx, ty, tz, color='lightgray', alpha=0.7)

# 양팔 베이스 (좌/우)
ax2.scatter([0.15, 0.15], [-0.2, 0.2], [0.0, 0.0], color='darkgoldenrod', s=120, label='Dual Arms Base')
# 암 개략 궤적 (좌/우)
ax2.plot([0.15, 0.25, 0.35], [-0.2, -0.15, -0.05], [0.0, 0.25, 0.1], color='goldenrod', lw=4)
ax2.plot([0.15, 0.25, 0.35], [0.2, 0.15, 0.05], [0.0, 0.25, 0.1], color='goldenrod', lw=4)

# 타깃 물체
ax2.scatter([0.45], [0.0], [0.03], color='crimson', s=100, marker='s', label='Target Object')

# 카메라 위치 및 시선 벡터
cam_top = np.array([-0.1, 0.0, 0.65])
target_3d = np.array([0.45, 0.0, 0.03])
los_vec = target_3d - cam_top

ax2.scatter(cam_top[0], cam_top[1], cam_top[2], color='#1f77b4', s=120, label='Top Camera (~45°)')
ax2.quiver(cam_top[0], cam_top[1], cam_top[2], 
           los_vec[0], los_vec[1], los_vec[2], 
           color='green', arrow_length_ratio=0.1, lw=2, linestyle='--', label='Camera FOV Center')

ax2.set_xlim([-0.3, 0.7])
ax2.set_ylim([-0.4, 0.4])
ax2.set_zlim([0.0, 0.8])
ax2.view_init(elev=25, azim=-125)
ax2.set_title('[3D View] ALOHA-Style Top Camera Layout', fontsize=12, fontweight='bold')
ax2.set_xlabel('X [m]')
ax2.set_ylabel('Y [m]')
ax2.set_zlabel('Z [m]')
ax2.legend(loc='upper left', frameon=True, fontsize=8.5)

plt.tight_layout()

# -------------------------------------------------------------
# 3. 권장 카메라 파라미터 계산 및 콘솔 출력
# -------------------------------------------------------------
dx = target_3d[0] - cam_top[0]
dy = target_3d[1] - cam_top[1]
dz = target_3d[2] - cam_top[2]
dist_horizontal = np.sqrt(dx**2 + dy**2)
pitch_deg = np.degrees(np.arctan2(-dz, dist_horizontal))
dist_total = np.linalg.norm(los_vec)

print("=" * 65)
print("       [ACT / miniVLA 학습용 권장 Top-Camera 세팅 파라미터]")
print("=" * 65)
print(f"1. 카메라 위치 (Pos [X, Y, Z])   : [{cam_top[0]:.2f}, {cam_top[1]:.2f}, {cam_top[2]:.2f}] m")
print(f"2. 시선 목표점 (LookAt [X, Y, Z]) : [{target_3d[0]:.2f}, {target_3d[1]:.2f}, {target_3d[2]:.2f}] m")
print(f"3. 하향 각도 (Downward Pitch)    : {pitch_deg:.1f}° (권장 범위: 40° ~ 50°)")
print(f"4. 카메라-타깃 거리 (Distance)   : {dist_total:.2f} m")
print(f"5. 권장 수직 화각 (FOV)          : 약 55° ~ 65°")
print("-" * 65)
print("[MuJoCo XML 예시 태그]")
print(f'  <camera name="top_cam" pos="{cam_top[0]} {cam_top[1]} {cam_top[2]}" fovy="60" mode="targetbody" target="table_center"/>')
print(f'  <!-- 또는 euler/quat 회전 직접 지정 시: pitch 약 {pitch_deg:.0f}도 하향 -->')
print("=" * 65)

plt.show()