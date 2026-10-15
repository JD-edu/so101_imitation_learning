# SO-101 MuJoCo 이미테이션 러닝

이 프로젝트는 다음 두 LeRobot ACT 파이프라인을 제공한다.

1. MuJoCo IK expert가 합성 시연을 생성하는 순수 MuJoCo 파이프라인
2. 실제 SO-101 리더로 MuJoCo 팔로워를 조작해 시연을 수집하는 파이프라인

두 파이프라인 모두 같은 데이터 의미와 LeRobot 공식 `ACTPolicy` 학습 구조를
사용한다.

## 폴더 구조

```text
09_10_so101_mujoco_imitation_learning/
├── 101_so101_load/                         # 기존 SO-101 모델 로드 예제
├── 102_robot_control/                      # 기존 MuJoCo 로봇 제어 예제
├── 103_so101_mujoco_ik/
│   └── so101_mujoco_ik/                    # 순수 MuJoCo IK/ACT 패키지
│       ├── assets/
│       │   ├── pick_place.xml
│       │   ├── so101_new_calib.xml
│       │   └── meshes/*.stl
│       ├── env.py                          # MuJoCo 환경과 물리 그리핑
│       ├── ik.py                           # 수치 IK
│       ├── expert.py                       # pick-and-place expert
│       ├── dataset.py                      # 합성 LeRobot 데이터 생성
│       ├── train.py                        # 공식 ACT 학습
│       ├── inference.py                    # ACT 추론
│       ├── viewer.py                       # GLFW viewer
│       └── cli.py
├── 104_so101_real_leader_mujoco_il/
│   └── so101_real_leader_mujoco_il/        # 실제 리더/가상 팔로워 패키지
│       ├── assets/
│       │   ├── full_arm_calibration_leader.json
│       │   ├── calibration_backups/*.json
│       │   ├── lift_cube_calibration.xml
│       │   ├── so101_new_calib.xml
│       │   └── meshes/*.stl
│       ├── calibration.py                  # 리더 칼리브레이션
│       ├── hardware.py                     # STS3215 serial driver
│       ├── runtime.py                      # MuJoCo follower와 값 변환
│       ├── dataset.py                      # 실제 리더 시연 수집
│       ├── inspect_data.py                 # 수집 episode 확인
│       ├── train.py                        # 공식 ACT 학습
│       ├── inference.py                    # ACT 추론
│       ├── viewer.py                       # GLFW viewer
│       └── cli.py
├── data/                                   # 생성/수집된 LeRobot 데이터셋
├── outputs/                                # 학습된 ACT 체크포인트
├── tests/                                  # 회귀 테스트
├── pyproject.toml
├── requirements.txt
├── README.md
├── word1.md                                # 이전 작업 기록
├── work2.md                                # 이전 작업 기록
└── work3.md                                # 실제 리더 파이프라인 작업 기록
```

`data/`에는 생성된 대용량 데이터셋이 있고 `outputs/`에는 학습된 체크포인트가
있으므로 보존했다. `tests/`는 구조 변경 후 두 파이프라인의 회귀 검증에
사용한다. 기존 `103_robot_pick_n_place`, `104_imitation_learning_ACT`, 캐시,
로그 및 중복 코드는 제거했다.

## 설치

저장소 루트에서 기존 LeRobot conda 환경을 활성화하고 editable install한다.

```bash
cd /home/robo/so101_imitation_learning/09_10_so101_mujoco_imitation_learning
conda activate lerobot
python -m pip install -e '.[train,test]' --no-build-isolation
```

CUDA 사용 여부는 다음 명령으로 확인한다.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 103: 순수 MuJoCo IK → ACT

### IK expert 확인

```bash
# Headless
python -m so101_mujoco_ik.cli demo --headless --seed 0

# GLFW UI
python -m so101_mujoco_ik.cli demo --seed 0
```

UI 명령은 IK 궤적을 계산한 뒤 약 7초 동안 GLFW 창에서 동작을 재생한다.
궤적 생성용 offscreen renderer와 GLFW 창을 연속 생성하면 GLFW가 먼저
종료될 수 있으므로, `demo`는 궤적 계산 중에는 이미지를 렌더링하지 않고
재생 단계에서만 창을 만든다.

창이 여전히 열리지 않는 경우 먼저 다음을 확인한다.

```bash
echo "$DISPLAY"
python -m so101_mujoco_ik.cli demo --headless --seed 0
```

첫 명령의 결과가 비어 있으면 현재 터미널에 X11/데스크톱 디스플레이가
연결되지 않은 것이다. SSH 접속이라면 X11 forwarding을 사용하거나 로봇의
데스크톱 터미널에서 UI 명령을 실행한다. `MUJOCO_GL=egl`을 셸에 전역으로
설정했다면 UI 실행 전 `unset MUJOCO_GL`도 수행한다.

### 합성 데이터 생성

```bash
python -m so101_mujoco_ik.cli generate \
  --root data/so101_pick_place_physics \
  --repo-id local/so101_mujoco_pick_place \
  --episodes 100
```

기존 데이터 보호를 위해 목적지에 파일이 있으면 생성을 거부한다.

### ACT 학습

```bash
python -m so101_mujoco_ik.cli train \
  --dataset-root data/so101_pick_place_physics \
  --repo-id local/so101_mujoco_pick_place \
  --output outputs/act_so101_physics \
  --steps 100000 --batch-size 32 --device cuda
```

### ACT 추론

현재 저장된 물리 그리핑 체크포인트는 다음 명령으로 실행한다.

```bash
python -m so101_mujoco_ik.cli infer \
  --checkpoint outputs/act_so101_physics \
  --episodes 10 --ui
```

CUDA 없이 실행하려면 `--device cpu`를 추가한다.

## 104: 실제 리더 → MuJoCo 팔로워 → ACT

전체 순서는 다음과 같다.

```text
calibrate → collect → inspect → train → infer
```

### 1. 실제 리더 칼리브레이션

```bash
python -m so101_real_leader_mujoco_il.cli calibrate \
  --port /dev/ttyACM0
```

각 관절마다 안내에 따라 다음 순서로 리더 자세를 지정한다.

1. MuJoCo `0 rad` 자세
2. MuJoCo 최소 각도 자세
3. MuJoCo 최대 각도 자세

최소/최대는 encoder 숫자가 아니라 MuJoCo 관절 각도 기준이다. 결과는 다음
패키지 asset에 직접 저장된다.

```text
104_so101_real_leader_mujoco_il/
└── so101_real_leader_mujoco_il/assets/full_arm_calibration_leader.json
```

기존 JSON은 `assets/calibration_backups/`에 타임스탬프 이름으로 백업된다.
칼리브레이션은 torque만 끄고 encoder를 읽으며 persistent homing-offset이나
hardware position limit를 변경하지 않는다.

### 2. 시연 데이터 수집

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_mujoco \
  --repo-id local/so101_real_leader_mujoco \
  --leader-port /dev/ttyACM0 \
  --fps 30
```

키 조작:

- `r`: 새 episode 녹화 시작
- `s`: episode 저장 후 환경 초기화
- `d`: 현재 episode 폐기 후 초기화
- `q`: 종료. 미완성 episode는 자동 저장하지 않음

이전 초기화 실패로 `total_episodes=0`, `total_frames=0`인 metadata만 남은
경우 다음 옵션을 한 번 추가한다.

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_mujoco \
  --repo-id local/so101_real_leader_mujoco \
  --leader-port /dev/ttyACM0 \
  --fps 30 --recover-empty
```

검증된 0-frame 골격은 삭제하지 않고 `.incomplete_날짜_시간` 이름으로
보존한다. 실제 frame이나 episode가 있으면 `--recover-empty`는 동작을 거부한다.

### 3. 수집 데이터 확인

```bash
python -m so101_real_leader_mujoco_il.cli inspect \
  --dataset-root data/real_leader_mujoco \
  --repo-id local/so101_real_leader_mujoco \
  --episode 0 --delay-ms 33
```

이미지 창에서 `q`를 누르면 종료한다.

### 4. ACT 학습

```bash
python -m so101_real_leader_mujoco_il.cli train \
  --dataset-root data/real_leader_mujoco \
  --repo-id local/so101_real_leader_mujoco \
  --output outputs/act_real_leader_mujoco \
  --steps 100000 --batch-size 32 --device cuda
```

CPU smoke test:

```bash
python -m so101_real_leader_mujoco_il.cli train \
  --dataset-root data/real_leader_mujoco \
  --repo-id local/so101_real_leader_mujoco \
  --output outputs/act_real_leader_mujoco_smoke \
  --steps 10 --batch-size 2 --device cpu
```

### 5. ACT 추론

```bash
python -m so101_real_leader_mujoco_il.cli infer \
  --checkpoint outputs/act_real_leader_mujoco \
  --device cuda --seed 1000
```

추론 UI에서 `r`은 환경과 ACT action queue를 초기화하고 `q`는 종료한다.

## 데이터 정의

두 파이프라인은 같은 관절 순서를 사용한다.

```text
shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper
```

실제 리더 데이터셋:

| Feature | Shape | 의미 |
|---|---:|---|
| `observation.images.overview` | `(3, 224, 224)` | MuJoCo RGB 관측 |
| `observation.state` | `(6,)` | MuJoCo follower 현재 관절 위치(rad) |
| `action` | `(6,)` | 칼리브레이션된 리더 목표 관절 위치(rad) |
| `task` | 문자열 | 작업 설명 |

state와 action은 모두 radian이다. normalization과 unnormalization은 저장된
dataset 통계를 사용하는 LeRobot processor가 수행한다.

## 주요 수정 원칙

- LeRobot 공식 `ACTConfig`와 `ACTPolicy` 사용
- 실제 미래 action delta timestamp 사용
- 학습과 추론에서 동일한 preprocessor/postprocessor 사용
- MuJoCo joint ID 대신 qpos/dof/actuator address를 명시적으로 사용
- position actuator에는 torque가 아닌 목표 관절 위치 입력
- 수집과 추론에서 동일한 이미지·state·action 의미 사용
- NaN/Inf action을 MuJoCo에 전달하기 전에 차단
- AMP는 CUDA 여부만으로 강제하지 않고 체크포인트 `use_amp` 설정을 따름
- 존재하지 않거나 불완전한 체크포인트를 로드 전에 검사
- 기존 데이터셋과 calibration을 자동 삭제하지 않음

## 테스트

```bash
python -m pytest -q
```

현재 회귀 테스트는 다음을 확인한다.

- MuJoCo 모델의 6축 address와 joint limit
- 랜덤 시작 위치의 IK pick-and-place
- bilateral physical contact와 no-weld 그리핑
- 실제 리더 calibration의 normal/reversed 방향 변환
- `center_raw`가 정확히 MuJoCo `0 rad`로 변환되는지 여부
- position actuator에 radian target이 입력되는지 여부

## 작업 기록

- `word1.md`: 최초 순수 MuJoCo 파이프라인 작업
- `work2.md`: LeRobot import 및 물리 그리핑 개선
- `work3.md`: 실제 리더 파이프라인, calibration, ACT 및 오류 수정 기록

작업 기록의 과거 경로와 명령은 당시 구조를 설명하기 위한 기록이므로 현재
실행 명령은 이 README를 기준으로 한다.
