# ArUco 기준점: 생성 → MuJoCo → ACT·miniVLA

이 폴더에는 ArUco 기준점의 생성, 검출, MuJoCo 렌더 검증 코드가 모여 있다. 기본 규격은 `DICT_6X6_250`, ID `23`, 바깥 흰 여백을 포함한 12 cm 정사각형이다.

## 문서 구성

이 폴더만 읽어도 마커 생성부터 학습·추론까지 재현할 수 있도록 문서를 다음과 같이 나눴다.

| 문서 | 내용 |
|---|---|
| [01_marker_generation.md](01_marker_generation.md) | 기존 그림 태그를 폐기한 이유, ArUco 규격, 생성·인쇄·단일 이미지 검증 |
| [02_mujoco_xml_and_camera.md](02_mujoco_xml_and_camera.md) | XML texture/body/sensor, 양팔 중앙 좌표, overview 카메라와 FOV |
| [03_act_minivla_usage.md](03_act_minivla_usage.md) | ACT 영상 방식·명시적 6차원 상태 방식, miniVLA 언어 grounding |
| [04_run_and_validation.md](04_run_and_validation.md) | 전체 명령, 출력 확인, 테스트 결과와 문제 해결 |

관련 Python 코드는 모두 이 폴더에 있다.

| 코드 | 역할 |
|---|---|
| `generate_marker.py` | ArUco PNG와 JSON 메타데이터 생성 |
| `detector.py` | ID 검출, 정규화 특징 6개 계산, preview 표시 |
| `inspect_image.py` | PNG·카메라 이미지의 마커 검출 확인 |
| `verify_scene.py` | 104 XML을 실제 overview 조건으로 렌더링하여 검출 확인 |
| `test_aruco_reference.py` | 생성·검출·XML·CLI 회귀 검사 |

## 빠른 실행

```bash
cd /home/robo/so101_imitation_learning/09_10_so101_mujoco_imitation_learning
conda activate lerobot

# 1. 마커 생성
python -m so101_real_leader_mujoco_il.aruco_reference.generate_marker

# 2. 실제 MuJoCo 카메라 조건으로 검출 확인
MUJOCO_GL=egl python \
  -m so101_real_leader_mujoco_il.aruco_reference.verify_scene \
  --output aruco_scene_check.png

# 3. ArUco 영상과 6차원 검출 상태를 포함해 데이터 수집
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_aruco_state \
  --repo-id local/so101_real_leader_aruco_state \
  --leader-port /dev/ttyUSB0 \
  --aruco-state --aruco-id 23 \
  --assist-view
```

`--assist-view`는 사람이 조작할 때만 보는 정면 사선 MuJoCo 창을 추가한다. 이 창은 정책 영상과 데이터셋에 포함되지 않으며 옵션을 생략하면 열리지 않는다.

## 1단계: 마커 생성

패키지 상위 폴더에서 실행한다.

```bash
cd /home/robo/so101_imitation_learning/09_10_so101_mujoco_imitation_learning
conda activate lerobot

python -m so101_real_leader_mujoco_il.aruco_reference.generate_marker
```

생성 파일:

```text
104_so101_real_leader_mujoco_il/so101_real_leader_mujoco_il/assets/
├── aruco_6x6_250_id23.png
└── aruco_6x6_250_id23.json
```

다른 ID나 출력 크기를 쓰려면 다음처럼 실행한다. XML과 수집·추론의 ID도 반드시 같은 값으로 바꿔야 한다.

```bash
python -m so101_real_leader_mujoco_il.aruco_reference.generate_marker \
  --marker-id 23 \
  --pixels 512 \
  --quiet-zone 56 \
  --physical-size-m 0.12
```

흰 quiet zone은 검출에 필요하다. 출력 PNG를 인쇄할 때 이미지 자동 자르기나 페이지 맞춤 확대를 끄고, JSON의 `physical_outer_size_m`와 실제 바깥 크기를 맞춘다.

단일 이미지 검출 확인:

```bash
python -m so101_real_leader_mujoco_il.aruco_reference.inspect_image \
  104_so101_real_leader_mujoco_il/so101_real_leader_mujoco_il/assets/aruco_6x6_250_id23.png \
  --output aruco_marker_check.png
```

## 2단계: MuJoCo XML

`assets/lift_cube_calibration.xml`에는 다음 요소가 추가되어 있다.

- `<texture>`: 생성한 PNG
- `<material>`: 반사 없는 마커 재질
- `aruco_reference_frame`: 알려진 월드 좌표의 기준 body
- `aruco_reference_marker`: 충돌하지 않는 12 cm 평면
- `aruco_reference_origin`: 마커 중심 site
- `aruco_reference_pos`, `aruco_reference_quat`: 시뮬레이션 디버깅용 ground-truth sensor

현재 104 모델에는 팔이 한 대뿐이고 베이스가 `Y=0`을 차지한다. 따라서 즉시 검출 가능한 기본 위치는 현재 카메라 작업 레인의 다음 좌표이다.

```xml
<body name="aruco_reference_frame" pos="0.25 -0.15 0.002">
```

양팔 XML에서 왼팔과 오른팔 베이스가 각각 `Y=-d`, `Y=+d`에 있다면 마커의 Y는 중앙값으로 바꾼다.

```text
marker_y = (left_arm_y + right_arm_y) / 2
```

예를 들어 두 팔이 `Y=-0.20`, `Y=+0.20`이면 다음과 같다.

```xml
<body name="aruco_reference_frame" pos="0.25 0.0 0.002">
```

마커가 로봇이나 작업물에 가리지 않는지 실제 overview 카메라로 검사한다.

```bash
MUJOCO_GL=egl python \
  -m so101_real_leader_mujoco_il.aruco_reference.verify_scene \
  --output aruco_scene_check.png
```

종료 코드 `0`과 JSON의 `"visible": true`가 나와야 한다. 출력 그림에는 검출 테두리와 중심점이 표시된다.

## 3단계: ACT 활용

### 방법 A: 영상 기준점

추가 옵션 없이 수집하면 마커는 `observation.images.overview` 안에 포함된다. ACT의 CNN은 물체와 로봇을 마커에 상대적인 영상 위치로 학습할 수 있다.

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_aruco_visual \
  --repo-id local/so101_real_leader_aruco_visual \
  --leader-port /dev/ttyUSB0
```

이 방식은 기존과 같은 관절 상태 6개를 유지한다. 마커를 넣기 전에 수집한 데이터와 섞지 않는다.

### 방법 B: 영상 + 명시적 ArUco 상태

`--aruco-state`를 켜면 매 프레임에서 ID 23을 검출하여 `observation.environment_state` 6개를 함께 저장한다.

```text
[visible, center_x_normalized, center_y_normalized,
 side_length_ratio, rotation_sin, rotation_cos]
```

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_aruco_state \
  --repo-id local/so101_real_leader_aruco_state \
  --leader-port /dev/ttyUSB0 \
  --aruco-state --aruco-id 23
```

마커가 가려지면 여섯 값은 모두 `0`이고 `visible=0`이 된다. 저장되는 정책 영상은 원본이며, 녹색 검출선은 사람에게 보여주는 preview에만 그린다.

학습 명령은 동일하다. 데이터 메타데이터를 읽은 ACT가 environment-state 입력을 자동으로 구성한다.

```bash
python -m so101_real_leader_mujoco_il.cli train \
  --dataset-root data/real_leader_aruco_state \
  --repo-id local/so101_real_leader_aruco_state \
  --output outputs/act_real_leader_aruco_state \
  --steps 100000 --batch-size 32 --device cuda
```

추론은 체크포인트 설정에 `observation.environment_state`가 있으면 검출기를 자동으로 켠다.

```bash
python -m so101_real_leader_mujoco_il.cli infer \
  --checkpoint outputs/act_real_leader_aruco_state \
  --device cuda --aruco-id 23
```

기존 ACT 체크포인트에는 environment-state가 없으므로 검출 벡터를 요구하지 않고 기존 방식으로 실행된다. `--aruco-state`로 수집한 데이터는 이 옵션 없이 수집한 데이터와 입력 schema가 다르므로 한 학습 데이터셋에 섞지 않는다.

## miniVLA 활용

현재 저장소에는 miniVLA 모델이나 실행기가 없다. 따라서 존재하지 않는 모델 API에 연결하는 코드는 추가하지 않았다. 대신 miniVLA가 사용할 두 입력은 이미 데이터에 들어간다.

1. `observation.images.overview`: ArUco가 보이는 RGB 영상
2. `task`: `Use ArUco marker 23 as the workspace reference...` 문장

miniVLA 학습에서는 이미지와 이 task 문장을 함께 사용한다. 작업 지시는 거리와 방향을 일관된 표현으로 기록한다.

```text
Use ArUco marker 23 as the workspace origin.
Move the red block 10 cm in front of the marker.
```

여기서 `앞`, `왼쪽`, `오른쪽`의 축 정의를 데이터셋 전체에서 바꾸지 않는다. miniVLA loader가 수치 상태도 지원하는 경우에만 `observation.environment_state`를 추가 입력으로 연결한다. 이미지·언어만 받는 모델에는 6차원 벡터를 억지로 전달하지 않는다.

## 운용 원칙

- 실제 작업대와 MuJoCo에서 같은 dictionary, ID, 바깥 크기와 방향을 사용한다.
- 카메라 위치, FOV, 렌더 해상도와 마커 위치를 확정한 후 데이터를 수집한다.
- 정적 overview 카메라에서 마커가 항상 같은 픽셀에만 있으면 정보량이 작다. 물체 목표를 마커 기준으로 바꾸거나 카메라·작업 배치를 변화시켜야 기준점의 장점이 생긴다.
- 마커 검출값은 2D 영상 기준이다. 정확한 3D 자세가 필요하면 카메라 내부 파라미터와 실제 인쇄 크기를 사용한 pose estimation을 별도로 추가한다.
- XML sensor 값은 시뮬레이션 ground truth이므로 실제 로봇 학습 입력으로 사용하지 않는다. sim-to-real에서 얻을 수 없는 privileged 정보이기 때문이다.
