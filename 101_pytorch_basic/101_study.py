import numpy as np
import matplotlib.pyplot as plt

# 1. 재현성을 위한 무작위 시드 설정 및 데이터 파라미터
np.random.seed(42)
num_samples = 1000  # 총 생성할 궤적(에피소드) 수
chunk_size = 20    # ACT 모델이 한 번에 예측하는 미래 타임슬롯 크기

# 2. 대화형 모드(Interactive Mode) 활성화 -> 실시간 스트리밍 플롯 가능하게 설정
plt.ion()

# 3. 3D 그래프 스케일 및 캔버스 초기 설정
fig = plt.subplots(figsize=(11, 9))[0]
ax = fig.add_subplot(111, projection='3d')

# 고정된 목표 위치 (빨간색 별: 물건이 놓인 위치)
target_x, target_y, target_z = 0.0, 0.0, 0.0
lift_z = 0.5  # 물건을 집어 올릴 목표 높이

# 목표 지점 플로팅 (가장 먼저 그려서 기준점 제시)
ax.scatter(target_x, target_y, target_z, color='red', s=200, marker='*', label='Target Object', zorder=10)

# 그래프 기본 레이아웃 설정
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-0.2, 2.2])
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title("ACT Dataset: 1,000 Trajectories Live Streaming")
ax.legend(loc='upper left')

print("실시간 라이브 스트리밍 시각화를 시작합니다...")

# 4. 1,000개의 궤적을 실시간으로 생성하며 그리는 루프
for i in range(num_samples):
    # 임의의 3차원 시작 위치 (Observation) 생성
    start_x = np.random.uniform(-1.0, 1.0)
    start_y = np.random.uniform(-1.0, 1.0)
    start_z = np.random.uniform(1.0, 2.0)
    
    # 삼차 보간법(Cubic Interpolation)을 이용한 20스텝 Action Chunk 생성
    t1 = np.linspace(0, 1, 10)
    t2 = np.linspace(0, 1, 10)
    
    # 1단계: 시작점 -> 물건 위치로 부드럽게 감속하며 하강
    x_part1 = start_x + (target_x - start_x) * (3 * t1**2 - 2 * t1**3)
    y_part1 = start_y + (target_y - start_y) * (3 * t1**2 - 2 * t1**3)
    z_part1 = start_z + (target_z - start_z) * (3 * t1**2 - 2 * t1**3)
    
    # 2단계: 물건 위치 -> 수직으로 Lift Up
    x_part2 = np.full(10, target_x)
    y_part2 = np.full(10, target_y)
    z_part2 = target_z + (lift_z - target_z) * (3 * t2**2 - 2 * t2**3)
    
    # 두 단계를 결합하여 [20, 3] 크기의 하나의 Action Chunk 완성
    x_traj = np.concatenate([x_part1, x_part2])
    y_traj = np.concatenate([y_part1, y_part2])
    z_traj = np.concatenate([z_part1, z_part2])
    
    # 현재 에피소드의 시작 상태값(파란색 점) 그리기
    ax.scatter(start_x, start_y, start_z, color='blue', s=15, alpha=0.5)
    
    # 해당 시작점과 연동된 미래 20스텝 전체 동선(선) 그리기
    ax.plot(x_traj, y_traj, z_traj, alpha=0.6, linewidth=1.0)
    
    # 실시간 모니터링을 위한 타이틀 업데이트 (현재 몇 번째 궤적인지 표시)
    ax.set_title(f"ACT Dataset: 1,000 Trajectories Live Streaming ({i+1}/1000)")
    
    # 실시간 스트리밍 핵심: 0.001초 동안 대기하며 화면을 갱신 (속도 조절 가능)
    plt.pause(0.001)

print("모든 궤적 스트리밍이 완료되었습니다.")

# 5. 대화형 모드를 해제하고 창을 유지시킴
plt.ioff()
plt.show()