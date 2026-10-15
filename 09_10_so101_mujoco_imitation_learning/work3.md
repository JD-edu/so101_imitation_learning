# SO-101 실제 리더–MuJoCo 팔로워 ACT 파이프라인 작업 기록 3

## 1. 문서 범위

이 문서는 `word1.md`, `work2.md` 이후 진행한 작업을 기록한다. 이번 작업의
목표는 기존 `104_imitation_learning_ACT` 폴더의 실제 SO-101 리더–MuJoCo
팔로워 코드를 앞서 만든 순수 MuJoCo ACT 파이프라인과 같은 구조로 정리하는
것이었다.

사용자가 정상 동작한다고 확인한 독립 텔레오퍼레이션 코드는 유지하고, 문제가
있던 다음 영역을 중심으로 수정했다.

1. 실제 리더 칼리브레이션
2. LeRobot 데이터 수집
3. 수집 데이터 확인
4. 공식 LeRobot ACT 학습
5. 저장된 ACT 체크포인트 추론
6. 체크포인트 및 렌더러 오류 처리
7. 전체 CLI와 README 문서화

## 2. 기존 코드에서 확인한 주요 문제

### 비공식 ACT 구현

기존 `105_ACT_IL_training.py`에는 LeRobot 공식 ACT가 아닌 별도의 단순
Transformer/CVAE 형태 모델이 구현되어 있었다. 학습 코드와 추론 코드에
ACT 클래스가 중복되어 체크포인트 구조가 달라질 가능성도 있었다.

### 잘못된 미래 action chunk

데이터셋에서 미래 action chunk를 가져오지 못하면 현재 action 하나를 50번
반복하는 코드가 있었다. 이 방식은 ACT가 미래 행동 시퀀스를 학습하는 목적과
맞지 않는다.

### state와 action 의미 불일치

기존 단일 카메라 수집 코드는 실제 MuJoCo follower 관절 상태가 아니라 리더
명령의 정규화 비율을 `observation.state`와 `action` 양쪽에 저장했다. 반면
추론에서는 MuJoCo 관절 상태를 다시 계산해 사용하여 학습과 추론의 입력 의미가
일치하지 않았다.

### MuJoCo 주소 사용 오류

일부 코드가 MuJoCo joint ID를 그대로 `qpos` 인덱스로 사용했다. MuJoCo에서
joint ID, qpos address, DoF address 및 actuator ID는 서로 다른 주소 체계이므로
모델 구조가 바뀌면 잘못된 값을 읽을 수 있다.

### position actuator에 torque 입력

사용 중인 XML은 다음과 같은 position actuator를 사용한다.

```xml
<position name="shoulder_pan" joint="shoulder_pan" ... />
```

하지만 기존 코드는 외부에서 계산한 PD torque 값을 `data.ctrl`에 넣었다.
position actuator의 `ctrl`에는 torque가 아니라 목표 관절 위치(rad)를 넣어야
한다.

### 데이터 및 체크포인트 경로 문제

기존 수집 코드는 Hugging Face cache 아래의 기존 데이터셋을 자동 삭제했다.
추론에서는 존재하지 않는 로컬 체크포인트 경로를 LeRobot이 Hugging Face repo
ID로 오인하여 `HFValidationError`가 발생했다.

### LeRobot 0.4.4 추론 입력 형식

현재 환경의 `predict_action()`은 NumPy HWC 이미지를 받아 내부에서 Tensor
CHW로 변환한다. 기존 추론 코드는 이미 Tensor CHW로 변환한 값을 전달하여
타입 오류가 발생했다.

## 3. 새 실제 리더 파이프라인 구조

새 구현은 다음 패키지에 정리했다.

```text
so101_real_leader_mujoco_il/
```

### `calibration.py`

- 실제 리더의 각 관절에서 MuJoCo `0 rad`, 최소 각도, 최대 각도의 encoder
  위치를 측정한다.
- encoder 증가 방향과 MuJoCo 관절 증가 방향이 반대이면 `drive_mode=1`을
  자동 설정한다.
- `center_raw`, `range_min`, `range_max`, `id`, `drive_mode`,
  `homing_offset`을 저장한다.
- 기존 calibration JSON은 덮어쓰기 전에 타임스탬프가 붙은 파일로 백업한다.
- 임시 파일에 먼저 기록한 후 `os.replace()`로 원자적으로 교체한다.
- 리더 torque만 끄며 persistent homing-offset 및 position-limit 레지스터는
  변경하지 않는다.

### `hardware.py`

- calibration과 데이터 수집에서 공통으로 사용하는 최소 STS3215 serial
  driver다.
- 새 패키지는 더 이상 기존 `104_imitation_learning_ACT/motor_control.py`를
  동적으로 import하지 않는다.
- torque on/off, 현재 위치 읽기 및 serial close 기능을 제공한다.

### `runtime.py`

- 6개 관절 이름으로 joint ID, qpos address, DoF address 및 actuator ID를
  각각 계산한다.
- calibration JSON의 필수 관절과 필드를 검증한다.
- `center_raw`가 있는 새 JSON은 최소→0 및 0→최대 구간을 별도로 변환하여
  측정한 중앙값이 MuJoCo `0 rad`에 정확히 대응하도록 한다.
- `center_raw`가 없는 기존 JSON도 endpoint 선형 변환 방식으로 지원한다.
- position actuator의 `ctrl`에 목표 radian 값을 직접 넣는다.
- NaN/Inf 관절 명령을 actuator에 전달하기 전에 거부한다.

### `dataset.py`

- 실제 리더 encoder를 calibration JSON을 이용해 MuJoCo 목표 radian으로
  변환한다.
- MuJoCo follower의 실제 관절 위치를 `observation.state`로 저장한다.
- 리더에서 계산한 목표 관절 위치를 `action`으로 저장한다.
- 데이터와 action은 모두 6축 radian 단위다.
- 이미지는 `observation.images.overview`라는 LeRobot `image` feature로
  저장한다.
- 기존 데이터셋을 자동 삭제하지 않으며 목적지가 비어 있지 않으면 중단한다.
- 이전 초기화 실패로 `info.json`만 남은 0-frame 골격은 `--recover-empty`로
  백업 이름에 보존한 후 새로 시작할 수 있다. 실제 frame이 있으면 복구를
  거부한다.
- serial과 GLFW 초기화가 성공한 뒤에만 데이터셋을 디스크에 생성한다.
- 종료 중이던 미완성 episode는 자동 저장하지 않는다.

### `viewer.py`

- 데이터 수집과 추론에서 동일한 GLFW 카메라와 렌더링 경로를 사용한다.
- 정책 입력용 224×224 RGB 이미지를 동일한 viewer에서 캡처한다.

### `train.py`

- LeRobot 공식 `ACTConfig`와 `ACTPolicy`를 사용한다.
- dataset metadata에서 input/output feature를 만든다.
- dataset FPS와 `action_delta_indices`로 실제 미래 action timestamp를 구성한다.
- LeRobot 공식 preprocessor와 postprocessor를 사용한다.
- policy, config, normalization/unnormalization processor를 같은 출력 폴더에
  저장한다.

### `inference.py`

- `ACTPolicy.from_pretrained()`로 policy를 읽는다.
- 저장된 preprocessor와 postprocessor를 함께 복원한다.
- 수집과 동일한 이미지, state, action 단위 및 30 Hz 제어 주기를 사용한다.
- 체크포인트 존재 여부와 필수 파일을 먼저 검사한다.
- 체크포인트에 저장된 device와 관계없이 `--device` 인자로 CPU/CUDA를
  override할 수 있다.
- 정책 action에 NaN/Inf가 있으면 MuJoCo에 전달하지 않고 episode/step을
  포함한 오류로 중단한다.

### `cli.py`

다음 명령을 하나의 CLI로 제공한다.

```text
calibrate
collect
train
infer
```

## 4. 기존 번호 스크립트 연결

기존 실행 방법과의 호환성을 위해 다음 파일은 새 패키지를 호출하는 얇은
진입점으로 변경했다.

| 기존 파일 | 연결된 기능 |
|---|---|
| `100_calibration.py` | 실제 리더 calibration |
| `103_data_collect.py` | 실제 리더 시연 데이터 수집 |
| `104_data_check.py` | 로컬 LeRobot episode 확인 |
| `105_ACT_IL_training.py` | 공식 LeRobot ACT 학습 |
| `106_ACT_IL_inference.py` | 공식 LeRobot ACT 추론 |

`109_*`, `110_*` 듀얼 카메라 파일은 실험용으로 남겨 두었으며 이번 canonical
단일 overview ACT 파이프라인에는 연결하지 않았다.

## 5. 최종 데이터 스키마

```text
observation.images.overview : RGB image, (3, 224, 224)
observation.state           : MuJoCo follower joint position, float32 (6,), rad
action                      : calibrated leader target, float32 (6,), rad
task                        : pick-and-place task 문자열
```

관절 순서는 다음과 같다.

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

state와 action을 모두 radian으로 통일하여 수집, 학습 및 추론 사이에 0~1,
-100~100 등의 서로 다른 정규화 의미가 섞이지 않도록 했다. 실제 normalization은
LeRobot processor가 dataset 통계를 이용해 수행한다.

## 6. Calibration JSON 연결

기본 calibration 파일은 다음 경로다.

```text
104_imitation_learning_ACT/full_arm_calibration_leader.json
```

`calibrate` 명령이 이 파일을 생성하며 `collect` 명령이 같은 파일을 기본으로
읽는다. 기존 텔레오퍼레이션과 같은 calibration을 공유하기 위해 JSON 위치는
유지했다.

새 JSON의 예시는 다음과 같다.

```json
{
  "shoulder_pan": {
    "id": 1,
    "drive_mode": 0,
    "center_raw": 2047,
    "homing_offset": 0,
    "range_min": 700,
    "range_max": 3400
  }
}
```

기존 JSON처럼 `center_raw`가 없으면 endpoint 전체를 선형 변환한다. 새로
calibration하면 측정된 중앙값이 정확히 `0 rad`가 되도록 piecewise 변환한다.

## 7. 참조되는 다른 설정 및 모델 파일

실제 리더 파이프라인이 소스 측에서 읽는 파일은 다음과 같다.

- `full_arm_calibration_leader.json`: 수집에서 읽는 유일한 calibration JSON
- `lift_cube_calibration.xml`: 최상위 MuJoCo 작업 scene
- `so101_new_calib.xml`: scene이 include하는 SO-101 모델
- `meshes/*.stl`: 로봇 XML이 참조하는 mesh

다음 JSON들은 새 파이프라인에서 읽지 않는다.

- `hybrid_calibration.json`
- `full_arm_calibration_follower.json`
- `pan_calibration.json`
- `shoulder_pan_calibration_leader.json`

학습 출력 폴더의 다음 파일들은 추론 시 LeRobot이 자동으로 읽는다.

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
policy_*_processor.safetensors
```

## 8. 최종 CLI 실행 순서

모든 명령은 저장소 최상위 폴더에서 실행한다.

### 8.1 칼리브레이션

```bash
python -m so101_real_leader_mujoco_il.cli calibrate \
  --port /dev/ttyUSB0
```

각 관절에서 다음 순서로 자세를 지정한다.

1. MuJoCo `0 rad`
2. MuJoCo 최소 각도
3. MuJoCo 최대 각도

최소/최대는 encoder 숫자가 아니라 MuJoCo 관절 각도 기준이다.

### 8.2 데이터 수집

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_mujoco \
  --repo-id local/so101_real_leader_mujoco \
  --leader-port /dev/ttyUSB0 \
  --fps 30
```

키 조작:

- `r`: 녹화 시작
- `s`: episode 저장 및 초기화
- `d`: episode 폐기 및 초기화
- `q`: 종료

### 8.3 데이터 확인

```bash
python 104_imitation_learning_ACT/104_data_check.py \
  --dataset-root data/real_leader_mujoco \
  --repo-id local/so101_real_leader_mujoco \
  --episode 0 --delay-ms 33
```

### 8.4 ACT 학습

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

### 8.5 ACT 추론

```bash
python -m so101_real_leader_mujoco_il.cli infer \
  --checkpoint outputs/act_real_leader_mujoco \
  --device cuda --seed 1000
```

추론 키:

- `r`: 환경 및 ACT action queue 초기화
- `q`: 종료

## 9. 순수 MuJoCo 체크포인트 오류 수정

다음 명령에서 `outputs/act_so101` 폴더가 실제로 존재하지 않아 오류가
발생했다.

```bash
python -m so101_mujoco_il.cli infer \
  --checkpoint outputs/act_so101 \
  --episodes 10 --ui
```

LeRobot은 존재하지 않는 절대경로를 Hugging Face repo ID로 해석했고 다음
오류를 출력했다.

```text
HFValidationError: Repo id must be in the form ...
```

현재 추론 코드는 다음을 먼저 검사한다.

- 체크포인트 디렉터리 존재 여부
- `config.json`
- `model.safetensors`
- `policy_preprocessor.json`
- `policy_postprocessor.json`

없는 경로를 지정하면 현재 사용 가능한 로컬 체크포인트도 오류 메시지에
표시한다. 당시 실제 존재하는 체크포인트는 다음과 같았다.

```text
outputs/act_so101_physics
outputs/act_so101_smoke
```

물리 그리핑 체크포인트의 올바른 실행 명령은 다음과 같다.

```bash
python -m so101_mujoco_il.cli infer \
  --checkpoint outputs/act_so101_physics \
  --episodes 10 --ui
```

## 10. NaN actuator 및 GLFW 오류 수정

`act_so101_physics`를 CUDA UI 모드로 실행했을 때 다음 경고가 발생했다.

```text
WARNING: Nan, Inf or huge value in CTRL at ACTUATOR 0.
GLFWError: The GLFW library is not initialized
```

### NaN 원인과 수정

기존 추론 코드는 CUDA device라는 이유만으로 `predict_action()`의 AMP를
강제로 활성화했다. 그러나 해당 체크포인트의 `use_amp` 설정은 `false`였다.
CUDA fp16 추론 중 NaN action이 만들어질 가능성이 있어 다음과 같이 수정했다.

- AMP는 `device == cuda`이면서 checkpoint의 `use_amp == true`일 때만 사용
- action의 NaN/Inf를 MuJoCo에 전달하기 전에 검사
- 오류 메시지에 episode 및 step 표시
- 환경의 `step()`과 실제 리더 runtime 양쪽에서도 non-finite 값 거부

### GLFW 원인과 수정

UI 모드에서 정책 입력 이미지를 별도 offscreen renderer로 만들면서 GLFW
viewer도 동시에 사용했다. 종료 시 GLFW를 먼저 terminate한 뒤 renderer가
정리되어 경고가 발생할 수 있었다.

다음과 같이 수정했다.

- UI에서는 GLFW viewer의 overview camera 이미지를 정책 입력으로 직접 사용
- 별도 offscreen renderer를 만들지 않음
- MuJoCo renderer를 GLFW window보다 먼저 정리
- viewer `close()`를 idempotent하게 만들어 중복 종료 방지

## 11. 검증 결과

하드웨어 없이 가능한 검증 결과는 다음과 같다.

- MuJoCo joint/qpos/dof/actuator 주소 확인
- calibration normal/reversed 방향 변환 확인
- `center_raw` → MuJoCo `0 rad` 변환 확인
- position actuator 목표값 입력 확인
- NaN/Inf actuator 명령 거부 확인
- LeRobot 공식 ACT CPU 1-step 학습 성공
- policy 및 pre/postprocessor 저장 성공
- 체크포인트 재로드 성공
- `(1, 6)` action 추론 성공
- 기존 CUDA checkpoint를 CPU device로 override하여 로드 성공
- `act_so101_physics` 240-step headless 추론 성공

240-step 검증 결과:

```text
episode 1: success=True target_error=0.003 m
success rate: 1/1 (100.0%)
```

CLI help, Python 정적 컴파일 및 `git diff --check`도 통과했다.

## 12. 남은 실제 장비 확인 항목

다음 항목은 실제 리더와 화면이 연결된 환경에서 최종 확인해야 한다.

1. `/dev/ttyUSB0`에서 6개 motor encoder 측정
2. 각 관절의 MuJoCo 최소/최대 자세 입력 방향
3. 생성된 `center_raw`와 endpoint 범위
4. 실제 리더 움직임과 MuJoCo follower의 방향 일치
5. 여러 정상 pick-and-place episode 수집
6. CUDA UI 추론에서 NaN 경고가 사라졌는지 확인
7. 창을 닫거나 `q`를 눌렀을 때 GLFW 경고 없이 종료되는지 확인

## 13. 문서화

루트 `README.md`를 한국어 실행 매뉴얼로 다시 작성했다. 다음 내용을 포함한다.

- 환경 설치
- calibration부터 inference까지 전체 CLI
- 데이터 수집 및 추론 키 조작
- 데이터 feature와 단위
- 각 모듈 역할
- 기존 번호 스크립트 대응표
- 이번 리뷰에서 수정한 문제
- JSON/XML/mesh/checkpoint 파일 관계
- calibration 안전 주의사항
- 순수 MuJoCo 기준 파이프라인 명령

## 14. 사용자 기존 변경사항

작업 시작 전부터 존재하던 다음 archive 이미지 변경은 이번 작업에서 수정하지
않았다.

```text
../archive/103_ACT_step_by_step/104_output/viz_TE_fixed_comparison.gif
../archive/103_ACT_step_by_step/104_output/viz_TE_fixed_final_frame.png
```

문서 작성 시점에는 실제 calibration 실행으로 생성된 것으로 보이는 다음 파일도
작업 트리에 존재한다. 이 문서를 작성하면서 해당 파일을 수정하거나 삭제하지
않았다.

```text
104_imitation_learning_ACT/full_arm_calibration_leader.json
104_imitation_learning_ACT/full_arm_calibration_leader.backup_20260923_151610.json
104_imitation_learning_ACT/full_arm_calibration_leader.backup_20260923_171529.json
MUJOCO_LOG.TXT
```

## 15. 후속 디렉터리 정리

후속 요청에 따라 최상위 코드 폴더를 다음 네 단계로 정리했다.

```text
101_so101_load/
102_robot_control/
103_so101_mujoco_ik/
104_so101_real_leader_mujoco_il/
```

- 기존 `so101_mujoco_il` 패키지는 이름을 `so101_mujoco_ik`로 바꾸고
  `103_so101_mujoco_ik/` 아래로 이동했다.
- 실제 리더 패키지는 `104_so101_real_leader_mujoco_il/` 아래로 이동했다.
- 두 패키지에 필요한 MuJoCo XML과 STL asset을 각각 패키지 내부에 배치했다.
- 현재 calibration과 이전 backup JSON은 새 104 패키지의 `assets/`로 옮겼다.
- 데이터 확인 기능은 새 실제 리더 CLI의 `inspect` 명령으로 통합했다.
- 기존 `103_robot_pick_n_place/`, `104_imitation_learning_ACT/`, 캐시 및 로그를
  제거했다.
- `data/`의 실제 데이터셋, `outputs/`의 ACT 체크포인트 및 `tests/`는 필요한
  산출물과 검증 자산이므로 보존했다.
- `pyproject.toml`, 테스트 import 및 README 명령을 새 패키지 경로에 맞췄다.

과거 절의 기존 경로는 당시 작업 과정을 기록하기 위해 그대로 두며, 현재 실행
경로와 명령은 루트 `README.md`를 기준으로 한다.

## 16. IK demo GLFW 창 미표시 수정

`python -m so101_mujoco_ik.cli demo --seed 0`에서 궤적은 정상 생성되지만
창이 열리지 않고 `The GLFW library is not initialized` 경고가 발생하는 문제를
수정했다.

원인은 demo 궤적 생성 중 `mujoco.Renderer`를 먼저 만든 후 GLFW 재생 창을
연속 생성하면서 OpenGL/GLFW 컨텍스트 수명이 충돌한 것이었다. demo는 이미지
데이터가 필요하지 않으므로 궤적 생성에는 `capture=False`를 사용하고, 계산된
action을 재생할 때에만 GLFW 창을 생성하도록 변경했다. 데이터셋 `generate`의
카메라 이미지 수집 경로는 그대로 유지된다.

검증 결과:

- headless demo: 208 frame 생성, 경고 없이 완료
- 최종 블록 위치: `[0.204, -0.162, 0.015]`
- 테스트에 궤적 생성 후 offscreen renderer가 생성되지 않았음을 확인하는
  회귀 조건 추가
- 이 샌드박스의 X 서버 접근 제한 때문에 실제 창 표시는 사용자 데스크톱에서
  최종 확인 필요
