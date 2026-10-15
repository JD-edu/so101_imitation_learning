# 104 실제 SO-101 리더 → MuJoCo 팔로워 실행 방법

이 폴더는 실제 SO-101 리더 암의 관절 값을 읽어 MuJoCo 팔로워를 움직이고, 시연 데이터를 LeRobot 형식으로 수집한 뒤 ACT를 학습·추론하는 패키지이다.

양팔 로봇의 뒤쪽 상단 카메라 위치, 방향과 화각을 조정하려면 [`so101_real_leader_mujoco_il/camera_view_setup.md`](so101_real_leader_mujoco_il/camera_view_setup.md)를 참고한다.

ArUco 기준점의 생성, XML 배치, ACT·miniVLA 사용법은 [`so101_real_leader_mujoco_il/aruco_reference/README.md`](so101_real_leader_mujoco_il/aruco_reference/README.md)를 참고한다.

실행 순서는 다음과 같다.

```text
환경 설치 → 포트 확인 → calibrate → collect → inspect → train → infer
```

## 1. 실행 위치와 환경 설치

명령은 `104_so101_real_leader_mujoco_il` 폴더 안이 아니라 `pyproject.toml`이 있는 상위 폴더에서 실행한다.

```bash
cd /home/robo/so101_imitation_learning/09_10_so101_mujoco_imitation_learning
conda activate lerobot
python -m pip install -e '.[train,test]' --no-build-isolation
```

정상 설치되면 다음 두 방식 중 하나로 같은 CLI를 실행할 수 있다.

```bash
# 설치된 콘솔 명령
so101-real-leader-il --help

# Python 모듈 실행 방식
python -m so101_real_leader_mujoco_il.cli --help
```

이 문서에서는 Python 환경이 분명하게 드러나는 두 번째 방식을 사용한다. `calibration.py`, `dataset.py`, `train.py`, `inference.py`를 파일 경로로 직접 실행하지 않는다. 이 파일들은 패키지 상대 import를 사용하며, 공개 실행 진입점은 `cli.py`이다.

## 2. 장치와 그래픽 환경 확인

리더 암을 연결한 뒤 실제 시리얼 포트를 확인한다.

```bash
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

코드의 기본 포트는 `/dev/ttyUSB0`이다. 장치가 `/dev/ttyACM0` 등으로 표시되면 아래 명령의 `--port`와 `--leader-port`를 실제 값으로 바꾼다. 기본 통신 속도는 `1000000` baud이다.

권한 오류가 나면 현재 사용자가 장치에 접근할 수 있는지 확인한다.

```bash
groups
ls -l /dev/ttyUSB0
```

`collect`, `inspect`, `infer`는 GLFW/OpenCV 창을 사용하므로 데스크톱 그래픽 환경에서 실행해야 한다.

```bash
echo "$DISPLAY"
```

결과가 비어 있으면 로컬 데스크톱 터미널이나 X11 전달이 설정된 세션을 사용한다. UI 실행 전에 `MUJOCO_GL=egl`을 전역 설정했다면 해제한다.

```bash
unset MUJOCO_GL
```

CUDA 사용 가능 여부는 다음과 같이 확인한다.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 3. 전체 CLI 확인

```bash
python -m so101_real_leader_mujoco_il.cli --help
python -m so101_real_leader_mujoco_il.cli calibrate --help
python -m so101_real_leader_mujoco_il.cli collect --help
python -m so101_real_leader_mujoco_il.cli inspect --help
python -m so101_real_leader_mujoco_il.cli train --help
python -m so101_real_leader_mujoco_il.cli infer --help
```

## 4. 리더 암 칼리브레이션

```bash
python -m so101_real_leader_mujoco_il.cli calibrate \
  --port /dev/ttyUSB0 \
  --baudrate 1000000
```

각 관절에서 프로그램 안내에 따라 다음 세 자세를 차례로 만든 뒤 `Enter`를 누른다.

1. MuJoCo 기준 `0 rad` 자세
2. MuJoCo 관절 최소 각도 자세
3. MuJoCo 관절 최대 각도 자세

프로그램은 모터 ID 1~6의 토크를 차례로 끄고 encoder 위치를 읽는다. 최소·최대는 encoder 숫자의 대소가 아니라 MuJoCo 관절 방향을 기준으로 잡아야 한다. 중앙값은 측정 범위 안에 있어야 한다.

기본 결과 파일은 다음 위치이다.

```text
104_so101_real_leader_mujoco_il/so101_real_leader_mujoco_il/
└── assets/full_arm_calibration_leader.json
```

같은 파일이 이미 있으면 기존 파일을 같은 디렉터리에 `full_arm_calibration_leader.backup_YYYYMMDD_HHMMSS.json` 이름으로 복사한 뒤 새 값을 저장한다. 다른 XML이나 출력 파일을 사용하려면 다음처럼 지정한다.

```bash
python -m so101_real_leader_mujoco_il.cli calibrate \
  --port /dev/ttyACM0 \
  --xml /absolute/path/to/model.xml \
  --output /absolute/path/to/leader_calibration.json
```

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--port` | `/dev/ttyUSB0` | 리더 암 시리얼 포트 |
| `--baudrate` | `1000000` | 시리얼 통신 속도 |
| `--xml` | 패키지의 `lift_cube_calibration.xml` | MuJoCo 모델 |
| `--output` | 패키지의 `full_arm_calibration_leader.json` | 저장할 칼리브레이션 JSON |

## 5. 시연 데이터 수집

처음에는 반드시 비어 있거나 존재하지 않는 새 데이터 폴더를 사용한다.

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_mujoco \
  --repo-id local/so101_real_leader_mujoco \
  --leader-port /dev/ttyUSB0 \
  --fps 30 --seed 0 \
  --assist-view
```

`--assist-view`를 추가하면 기존 뒤쪽 overview 창과 별도로 정면에서 30도 비껴 본 보조 MuJoCo 창이 열린다. 보조 창은 사람이 큐브를 잡기 쉽게 보는 용도이며 학습 데이터에 저장되지 않는다. 옵션을 생략하면 보조 창을 만들지 않는다.

ArUco ID 23의 2D 검출값을 ACT 입력에도 저장하려면 새 데이터셋에 다음 옵션을 추가한다.

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_aruco_state \
  --repo-id local/so101_real_leader_aruco_state \
  --leader-port /dev/ttyUSB0 \
  --fps 30 --seed 0 \
  --aruco-state --aruco-id 23
```

수집 창의 키는 다음과 같다.

| 키 | 동작 |
|---|---|
| `r` | 새 episode 녹화 시작 |
| `s` | 현재 episode 저장 후 MuJoCo 환경 초기화 |
| `d` | 현재 episode 폐기 후 환경 초기화 |
| `q` | 종료. 녹화 중이던 미완성 episode는 저장하지 않음 |

수집 데이터에는 MuJoCo RGB 영상 `(3, 224, 224)`, 현재 관절 상태 6개, 리더에서 변환한 목표 행동 6개가 저장된다. 기본 칼리브레이션과 XML 대신 다른 파일을 쓸 때만 명시한다.

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_mujoco_test \
  --repo-id local/so101_real_leader_mujoco_test \
  --leader-port /dev/ttyACM0 \
  --calibration /absolute/path/to/leader_calibration.json \
  --xml /absolute/path/to/model.xml \
  --fps 30 --seed 10
```

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--dataset-root` | 필수 | 데이터셋을 저장할 로컬 디렉터리 |
| `--repo-id` | `local/so101_real_leader_mujoco` | LeRobot 데이터셋 식별자 |
| `--leader-port` | `/dev/ttyUSB0` | 실제 리더 시리얼 포트 |
| `--calibration` | 패키지 기본 JSON | 리더 칼리브레이션 파일 |
| `--xml` | 패키지 기본 XML | MuJoCo 모델 |
| `--fps` | `30` | 데이터 저장 프레임률 |
| `--seed` | `0` | cube 초기 위치 난수 seed |
| `--recover-empty` | 꺼짐 | 0-frame 초기화 흔적만 복구 |
| `--aruco-state` | 꺼짐 | ArUco 검출값 6개를 environment-state로 저장 |
| `--aruco-id` | `23` | 검출할 ArUco ID |
| `--assist-view` | 꺼짐 | 정면 사선의 조작 보조 MuJoCo 창 표시 |

데이터 보호를 위해 `--dataset-root`가 비어 있지 않으면 수집을 거부한다. 초기화 실패로 `meta/info.json` 하나만 있고 `total_episodes=0`, `total_frames=0`인 경우에만 다음 명령을 한 번 사용한다.

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_mujoco \
  --repo-id local/so101_real_leader_mujoco \
  --leader-port /dev/ttyUSB0 \
  --recover-empty
```

검증된 빈 골격은 삭제하지 않고 `real_leader_mujoco.incomplete_날짜_시간`으로 옮긴다. 실제 frame이나 episode가 들어 있는 폴더에는 `--recover-empty`를 사용할 수 없다. 새 `--dataset-root`를 지정해야 한다.

## 6. 수집 데이터 확인

```bash
python -m so101_real_leader_mujoco_il.cli inspect \
  --dataset-root data/real_leader_mujoco \
  --repo-id local/so101_real_leader_mujoco \
  --episode 0 \
  --delay-ms 33
```

터미널에는 episode 수, frame 수, FPS, 관절 상태와 행동이 표시된다. OpenCV 창에는 저장된 RGB 영상이 재생되며 `q`를 누르면 종료한다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--dataset-root` | 필수 | 수집할 때 사용한 데이터 디렉터리 |
| `--repo-id` | `local/so101_real_leader_mujoco` | 수집할 때 사용한 동일 ID |
| `--episode` | `0` | 확인할 episode 번호 |
| `--delay-ms` | `33` | frame 사이의 표시 지연 시간(ms) |

## 7. ACT 학습

GPU 학습 예시는 다음과 같다.

```bash
python -m so101_real_leader_mujoco_il.cli train \
  --dataset-root data/real_leader_mujoco \
  --repo-id local/so101_real_leader_mujoco \
  --output outputs/act_real_leader_mujoco \
  --steps 100000 \
  --batch-size 32 \
  --device cuda
```

먼저 전체 파이프라인만 짧게 확인하려면 작은 CPU smoke test를 실행한다.

```bash
python -m so101_real_leader_mujoco_il.cli train \
  --dataset-root data/real_leader_mujoco \
  --repo-id local/so101_real_leader_mujoco \
  --output outputs/act_real_leader_mujoco_smoke \
  --steps 10 \
  --batch-size 2 \
  --device cpu
```

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--dataset-root` | 필수 | 학습 데이터 디렉터리 |
| `--repo-id` | `local/so101_real_leader_mujoco` | 수집할 때 사용한 동일 ID |
| `--output` | 필수 | 체크포인트를 저장할 새 디렉터리 |
| `--steps` | `100000` | optimizer update 횟수 |
| `--batch-size` | `32` | 학습 배치 크기 |
| `--device` | 자동 | CUDA가 있으면 `cuda`, 없으면 `cpu` |

데이터 frame 수가 `batch-size`보다 적으면 학습을 시작하지 않는다. 이 경우 더 많은 시연을 수집하거나 smoke test처럼 배치를 줄인다. `--device cuda`를 지정했는데 CUDA를 사용할 수 없으면 오류가 나므로 `--device cpu`로 실행한다.

학습이 끝나면 `--output` 폴더에 정책, 전처리기, 후처리기가 함께 저장된다. 추론에는 최소한 다음 파일들이 필요하다.

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
```

## 8. ACT 추론

```bash
python -m so101_real_leader_mujoco_il.cli infer \
  --checkpoint outputs/act_real_leader_mujoco \
  --device cuda \
  --seed 1000 \
  --assist-view
```

CUDA가 없으면 다음처럼 실행한다.

```bash
python -m so101_real_leader_mujoco_il.cli infer \
  --checkpoint outputs/act_real_leader_mujoco \
  --device cpu \
  --seed 1000
```

추론 창에서 `r`은 cube 위치, 로봇 상태, ACT action queue를 초기화하고 `q`는 종료한다. 추론은 실제 리더 암을 사용하지 않으며 학습된 정책이 MuJoCo 팔로워를 제어한다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--checkpoint` | 필수 | 학습이 만든 체크포인트 디렉터리 |
| `--device` | 자동 | CUDA가 있으면 `cuda`, 없으면 `cpu` |
| `--seed` | `1000` | cube 초기 위치 난수 seed |
| `--xml` | 패키지 기본 XML | 추론에 사용할 MuJoCo 모델 |
| `--aruco-id` | `23` | ArUco 상태를 사용한 체크포인트의 마커 ID |
| `--assist-view` | 꺼짐 | 정책 입력과 무관한 정면 사선 보조 창 표시 |

체크포인트는 단일 `.pt` 파일이 아니라 위 파일들이 들어 있는 디렉터리여야 한다. 다른 XML을 사용하려면 `--xml /absolute/path/to/model.xml`을 추가한다.

## 9. 소스 파일별 역할과 실행 여부

| 파일 | 역할 | 직접 실행 |
|---|---|---|
| `cli.py` | 다섯 명령을 연결하는 공식 진입점 | `python -m so101_real_leader_mujoco_il.cli ...` |
| `calibration.py` | encoder 범위 측정과 JSON 저장 | 보통 `cli calibrate` 사용 |
| `dataset.py` | 실제 리더 시연 수집과 LeRobot 저장 | 직접 실행하지 않음 |
| `inspect_data.py` | 저장 episode 영상·상태·행동 확인 | 직접 실행하지 않음 |
| `train.py` | 공식 LeRobot `ACTPolicy` 학습 | 직접 실행하지 않음 |
| `inference.py` | 체크포인트 로드와 MuJoCo 폐루프 추론 | 직접 실행하지 않음 |
| `hardware.py` | STS3215 최소 serial driver | 직접 실행하지 않음 |
| `runtime.py` | 리더 값 변환과 MuJoCo follower | 직접 실행하지 않음 |
| `viewer.py` | GLFW 렌더링과 영상 캡처 | 직접 실행하지 않음 |
| `__init__.py` | 관절 순서와 task 정의 | 직접 실행하지 않음 |

학습과 추론의 관절 순서는 항상 다음과 같다.

```text
shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper
```

`--repo-id`, 데이터 경로, 관절 순서, 관측 feature가 수집·학습·추론 사이에서 일치해야 한다.

## 10. 자주 발생하는 문제

- `Permission denied: /dev/tty...`: 시리얼 장치 권한과 사용 중인 포트를 확인한다.
- 모터 값을 읽지 못함: 포트, baudrate, 모터 ID 1~6, 전원과 케이블을 확인한다.
- 창이 열리지 않음: `DISPLAY`, X11 연결, `MUJOCO_GL` 설정을 확인한다.
- `dataset directory is not empty`: 기존 데이터를 덮어쓰지 않도록 새 `--dataset-root`를 사용한다.
- `dataset has fewer than batch_size`: 더 수집하거나 `--batch-size`를 줄인다.
- `CUDA is unavailable`: `--device cpu`를 지정한다.
- `checkpoint is incomplete`: `train`이 정상 종료된 출력 디렉터리를 전달했는지 확인한다.
- 추론에서 NaN/Inf 오류: 행동은 MuJoCo에 적용되지 않는다. CPU로 다시 확인하고 체크포인트와 데이터셋 조합을 점검한다.
