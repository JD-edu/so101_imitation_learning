# SO101 MuJoCo 이미테이션 러닝 후속 작업 기록 2

## 1. 문서 범위

이 문서는 `word1.md` 작성 이후 진행한 후속 작업을 기록한다. 주요 내용은 다음 두 가지다.

1. 사용자의 conda LeRobot 환경에서 발생한 `LeRobotDataset` import 오류 수정
2. 블록을 TCP에 강제로 부착하던 방식을 제거하고 JDCobot200 방식의 물리 접촉 그리핑으로 변경

기존 실제 리더 및 가상 팔로워 관련 코드는 이번 작업에서도 수정하지 않았다.

## 2. LeRobotDataset import 오류

### 발생한 오류

다음 데이터 생성 명령을 실행했을 때 import 오류가 발생했다.

```bash
python3 -m so101_mujoco_il.cli generate \
  --root data/so101_pick_place \
  --episodes 100
```

오류의 핵심 메시지는 다음과 같았다.

```text
ImportError: cannot import name 'LeRobotDataset' from 'lerobot.datasets'
```

### 환경 확인 결과

사용자가 활성화한 conda 환경을 직접 확인했다.

```text
Python executable:
/home/robo/anaconda3/envs/lerobot/bin/python

LeRobot metadata version:
0.4.4

LeRobot source path:
/home/robo/lerobot/src/lerobot

Git description:
v0.4.3-9-g520a3cde

Branch:
main
```

LeRobot 라이브러리는 정상적으로 설치되어 있었다. 문제는 이 checkout에서 `lerobot.datasets`가 namespace package로 구성되어 있고, 패키지 최상위에서 `LeRobotDataset`을 export하지 않는다는 점이었다.

다음 직접 import는 정상적으로 동작했다.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
```

### 수정 사항

`so101_mujoco_il/dataset.py`의 import를 다음과 같이 변경했다.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
```

`so101_mujoco_il/train.py`도 같은 구조로 변경했다.

```python
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
```

오류 메시지도 단순히 LeRobot이 설치되지 않았다고 표시하지 않고, 실제 원본 import 오류를 함께 출력하도록 개선했다.

### 실제 데이터 생성 검증

사용자의 conda 환경에서 1개 에피소드 데이터 생성을 실제로 실행했다.

```text
Map: 100% | 180/180
episode 001/001 saved
```

다음 파일이 생성되는 것을 확인했다.

```text
data/chunk-000/file-000.parquet
meta/info.json
meta/stats.json
meta/tasks.parquet
```

이 검증은 이전 강제 부착 방식에서 수행된 것이다. 이후 물리 그리핑으로 변경하면서 에피소드 길이가 208프레임으로 변경되었으므로, 실제 학습 데이터는 반드시 새 물리 그리핑 버전으로 다시 생성해야 한다.

## 3. 기존 그리퍼 방식의 문제

초기 구현은 합성 데이터의 성공률을 안정화하기 위해 다음 방식을 사용했다.

- gripper가 닫히고 TCP가 블록에 가까워지면 grasp 상태를 활성화
- grasp 상태에서는 블록의 free-joint 위치를 TCP 위치로 직접 변경
- gripper가 열리면 강제 부착 상태를 해제

이 방식은 화면에서 블록이 그리퍼에 달라붙은 것처럼 보인다. 실제 finger contact와 마찰로 물체를 잡는 것이 아니므로 실제 로봇 동작을 모방하기 위한 합성 데이터로는 부적절하다.

사용자의 지시에 따라 이 방식을 완전히 제거했다.

## 4. JDCobot200 no-weld 구현 조사

다음 JDCobot200 구현을 참고했다.

```text
/home/robo/Downloads/13_full_IL_sequence_with_no_weld/
```

해당 구현의 핵심 구조는 다음과 같다.

- 왼쪽과 오른쪽 finger pad가 모두 블록과 접촉했는지 검사한다.
- `mj_contactForce`로 실제 접촉력을 읽는다.
- 양쪽 접촉이 확인된 경우에만 grasp가 성립했다고 판단한다.
- 블록을 직접 이동하거나 weld하지 않는다.
- 이송 중 미끄러짐을 줄이기 위해 contact-conditioned compliant force를 사용한다.
- 실제로 블록이 바닥에서 상승했는지 별도로 검사한다.
- 목표 위치 오차와 release 동작을 검사한다.

SO101 구현도 같은 원칙으로 변경했다.

## 5. SO101 그리퍼 형상 분석

기존 SO101 모델에는 다음 두 collision pad가 이미 있었다.

```text
static_finger_pad
moving_finger_pad
```

하지만 기존 크기는 각 축이 약 1.25 mm인 매우 작은 box였다.

```text
size = [0.00125, 0.00125, 0.00125]
```

이 크기는 실제 패드 면이 아니라 점 접촉에 가까워 30 mm 블록을 안정적으로 집기 어려웠다.

또한 기존 `graspframe`과 실제 두 패드 중심을 비교한 결과, 위치가 약 5~6 cm 어긋나 있었다. 따라서 기존 IK는 실제 손가락 사이 중심이 아니라 손목 안쪽의 가상 지점을 블록에 맞추고 있었다.

## 6. 실제 패드 중심 기반 IK

### 새로운 pinch 기준점

그리퍼가 약 30 mm 블록을 잡을 수 있는 간격일 때 두 physical pad의 중심점을 계산했다.

이 중심점을 gripper body의 local 좌표로 저장하고, IK에서 기존 `graspframe` 대신 이 위치를 사용하도록 변경했다.

```text
pinch_local
pinch_position
```

### 그리퍼 방향 제약

위치만 맞추는 IK를 사용하면 두 패드가 블록의 위·아래 모서리를 대각선으로 누르는 자세가 만들어졌다. 이런 자세는 접촉력이 커도 들어 올리는 순간 블록이 빠질 수 있다.

이를 해결하기 위해 IK residual에 그리퍼 closing axis 방향을 추가했다.

- closing axis의 world Z 성분을 0에 가깝게 유지
- closing axis의 world Y 성분을 0에 가깝게 유지
- 결과적으로 패드가 world X 방향으로 서로 마주보게 정렬
- 블록의 평평한 양쪽 면을 face-to-face 방식으로 잡도록 유도

기존 반복 Damped Least Squares 방식은 위치와 방향을 동시에 만족시키는 과정에서 일부 시작점에 수렴하지 않았다. 따라서 SciPy의 bounded nonlinear least-squares를 사용하도록 변경했다.

```python
scipy.optimize.least_squares
```

다음 초기값을 이용한 multi-start 방식으로 가장 좋은 해를 선택한다.

- 이전 keyframe 관절값
- home pose
- 목표 방향을 반영한 shoulder-pan 초기값

SciPy 의존성은 `requirements.txt`와 `pyproject.toml`에 추가했다.

## 7. Physical pad와 contact 설정

기존 로봇 XML 파일은 수정하지 않았다. 로드된 새 환경의 MuJoCo model에서 두 collision pad의 크기만 작업용으로 조정했다.

```text
pad size = [0.0025, 0.015, 0.018]
```

새 scene에는 각 finger pad와 빨간 블록 사이의 명시적인 contact pair를 추가했다.

```xml
<pair geom1="static_finger_pad"
      geom2="red_block_geom"
      condim="4"
      friction="12.0 0.15 0.005"/>

<pair geom1="moving_finger_pad"
      geom2="red_block_geom"
      condim="4"
      friction="12.0 0.15 0.005"/>
```

접촉 안정성을 위해 `solref`와 `solimp`도 함께 조정했다.

## 8. 강제 부착 코드 제거

다음 기능을 삭제했다.

```text
grasped 상태 변수
set_grasped()
_carry_block()
블록 qpos를 TCP 위치로 복사하는 코드
거리와 gripper command만으로 grasp를 활성화하는 코드
release 시 블록을 목표 위치로 강제 이동하는 코드
```

현재 scene에는 block과 gripper 사이의 weld 또는 equality constraint가 없다.

블록은 항상 free joint를 가진 독립적인 MuJoCo body다.

## 9. 양측 접촉 검사

`SO101PickPlaceEnv.finger_contacts()`를 추가했다.

이 함수는 모든 MuJoCo contact를 검사해 다음을 반환한다.

```text
static finger가 블록과 접촉했는지 여부
moving finger가 블록과 접촉했는지 여부
두 접촉의 합산 normal force
```

실제 contact pair가 존재하는 경우에만 해당 finger가 블록에 닿았다고 판단한다.

## 10. Contact-conditioned compliant grip

순수 마찰 방식도 먼저 시험했다.

- 양쪽 접촉 성공
- 블록 초기 상승 성공
- 빠른 수평 이송 중 블록이 미끄러져 떨어짐

JDCobot200의 no-weld 구현과 동일한 방식으로 양쪽 실제 접촉이 확인된 뒤에만 compliant force를 활성화했다.

동작 조건은 다음과 같다.

1. gripper close 명령이 입력된다.
2. static pad와 moving pad가 모두 블록에 실제 접촉한다.
3. 접촉 순간의 블록 위치를 gripper body local offset으로 측정한다.
4. 이 offset을 유지하도록 PD 형태의 외력을 MuJoCo `xfrc_applied`에 입력한다.
5. 힘의 최대 크기는 3 N으로 제한한다.
6. gripper open 명령이 입력되면 즉시 force를 제거한다.

사용하는 힘은 다음 형태다.

```text
force = 180 × position_error - 4 × relative_velocity
maximum force = 3 N
```

이 방식의 중요한 차이는 다음과 같다.

- 블록 위치를 직접 덮어쓰지 않는다.
- 블록 속도를 직접 덮어쓰지 않는다.
- weld constraint를 사용하지 않는다.
- 양쪽 실제 접촉 전에는 force를 적용하지 않는다.
- 블록은 항상 물리 엔진에서 계산되는 free body다.
- 힘의 제한을 초과하면 물체가 미끄러지거나 떨어질 수 있다.

## 11. 전문가 궤적 변경

물리 접촉 시간을 충분히 확보하기 위해 궤적을 변경했다.

- descend 구간을 늘렸다.
- close 구간을 24프레임으로 늘렸다.
- open 구간을 20프레임으로 늘렸다.
- closed gripper target을 실제 블록 폭과 패드 두께에 맞게 조정했다.
- 물리적으로 안정적인 높이까지 lift하도록 수정했다.
- 고정된 블록 위치 보정을 제거했다.

그 결과 한 에피소드 길이는 기존 180프레임에서 208프레임으로 변경됐다.

## 12. 물리 그리핑 성공 조건

전문가 궤적은 단순히 최종 블록 위치만 검사하지 않는다.

다음 조건을 모두 만족해야 성공으로 인정한다.

1. 양쪽 finger가 블록에 실제 접촉한 프레임이 최소 3개 이상
2. 블록 최대 높이가 0.055 m 이상
3. 최종 블록 XY 위치가 목표에서 0.045 m 이내
4. episode 종료 시 compliant grip이 해제된 상태
5. weld를 사용하지 않음

각 에피소드의 주요 결과는 `last_episode_metrics`에 저장한다.

```text
bilateral_contact_frames
max_block_height
final_target_error
uses_weld = false
```

## 13. 검증 결과

### 다중 랜덤 위치

50개의 서로 다른 random seed로 headless 물리 pick-and-place를 실행했다.

```text
physical pick-place passed 50 / 50
```

각 테스트는 실제 양측 접촉, 블록 상승 및 최종 배치를 모두 검사했다.

### 자동 테스트

```text
3 passed
```

자동 테스트에 다음 조건을 추가했다.

- 한 episode가 208프레임인지 확인
- 양측 접촉 프레임이 3개 이상인지 확인
- 실제 블록 상승 높이가 0.055 m 이상인지 확인
- weld를 사용하지 않았는지 확인
- episode 종료 시 compliant force가 해제됐는지 확인

### Headless demo

```text
generated 208 frames
final block position=[0.204, -0.162, 0.015]
```

목표 위치는 다음과 같다.

```text
[0.20, -0.16, 0.015]
```

## 14. 새 데이터 생성 방법

기존 `data/so101_pick_place` 데이터가 이미 생성됐다면 강제 부착 방식의 프레임이 포함됐을 수 있다. 새 물리 그리핑 데이터는 별도의 경로에 생성하는 것이 안전하다.

### Headless 생성

```bash
python3 -m so101_mujoco_il.cli generate \
  --root data/so101_pick_place_physics \
  --repo-id local/so101_mujoco_pick_place_physics \
  --episodes 100
```

### GLFW UI를 표시하면서 생성

```bash
python3 -m so101_mujoco_il.cli generate \
  --root data/so101_pick_place_physics \
  --repo-id local/so101_mujoco_pick_place_physics \
  --episodes 100 \
  --ui
```

LeRobotDataset은 비어 있지 않은 기존 경로를 덮어쓰지 않으므로, 새 경로를 사용하거나 기존 데이터의 필요 여부를 사용자가 먼저 판단해야 한다.

## 15. 변경된 주요 파일

```text
so101_mujoco_il/assets/pick_place.xml
so101_mujoco_il/env.py
so101_mujoco_il/ik.py
so101_mujoco_il/expert.py
so101_mujoco_il/dataset.py
so101_mujoco_il/train.py
tests/test_pipeline.py
requirements.txt
pyproject.toml
README.md
word1.md
work2.md
```

## 16. 현재 상태 요약

- LeRobotDataset import 오류 수정 완료
- 사용자의 conda LeRobot 환경에서 데이터 1 episode 저장 검증 완료
- TCP 기반 블록 강제 부착 제거 완료
- block qpos 강제 이동 제거 완료
- weld/equality 없는 물리 그리핑 구현 완료
- 실제 양측 finger contact 검사 완료
- JDCobot200 방식의 contact-conditioned compliant grip 적용 완료
- 물리 그리핑용 IK 위치 및 방향 보정 완료
- 50개 랜덤 seed pick-and-place 성공 확인 완료
- headless 실행 검증 완료
- GLFW UI 코드는 유지되며 실제 desktop에서 시각화 가능
- 물리 그리핑 데이터는 새로운 경로로 다시 생성해야 함

---

# 후속 작업 기록: ACT 학습 실행 및 오류 수정

## 17. 수집된 학습 데이터 확인

물리 그리핑 방식으로 생성된 데이터셋을 확인했다.

```text
Dataset root: data/so101_pick_place_physics
LeRobot codebase version: v3.0
Robot type: so101_mujoco
Episodes: 100
Frames: 20,800
Tasks: 1
FPS: 30
Image shape: (3, 224, 224)
State shape: (6,)
Action shape: (6,)
Dataset size: 약 446 MB
```

다음 세 데이터셋이 존재하는 것도 확인했다.

```text
data/so101_pick_place
  100 episodes, 18,000 frames
  초기 강제 부착 방식

data/so101_pick_place_physics
  100 episodes, 20,800 frames
  물리 접촉 그리핑 방식

data/so101_pick_place_ui
  15 episodes, 2,700 frames
  이전 UI 생성 시험 데이터
```

ACT 본 학습에는 다음 데이터셋만 사용하도록 결정했다.

```text
data/so101_pick_place_physics
```

## 18. ACT 학습 import 경로 호환 수정

사용 중인 LeRobot checkout은 일부 모듈을 최상위 package에서 export하지 않는다. 학습 및 추론 코드의 import를 현재 환경에 맞는 직접 경로로 변경했다.

### FeatureType

변경 전:

```python
from lerobot.configs import FeatureType
```

변경 후:

```python
from lerobot.configs.types import FeatureType
```

### ACTConfig와 ACTPolicy

변경 전:

```python
from lerobot.policies.act import ACTConfig, ACTPolicy
```

변경 후:

```python
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
```

### 전처리 및 후처리 factory

변경 전:

```python
from lerobot.policies import make_pre_post_processors
```

변경 후:

```python
from lerobot.policies.factory import make_pre_post_processors
```

### 데이터 feature 변환

변경 전:

```python
from lerobot.utils.feature_utils import dataset_to_policy_features
```

변경 후:

```python
from lerobot.datasets.utils import dataset_to_policy_features
```

## 19. Transformers 버전 호환 문제

초기 conda 환경에서 확인한 버전은 다음과 같았다.

```text
transformers 5.8.0
lerobot 0.4.4
```

하지만 `/home/robo/lerobot/pyproject.toml`에서 요구하는 Transformers 범위는 다음과 같다.

```text
transformers>=4.57.1,<5.0.0
```

Transformers 5.8.0 상태에서는 ACT를 import하기 전에 LeRobot의 Groot policy 설정을 초기화하다가 다음 오류가 발생했다.

```text
TypeError: non-default argument 'backbone_cfg' follows default argument
```

따라서 현재 LeRobot 소스가 명시적으로 요구하는 버전으로 변경했다.

```bash
python3 -m pip install --upgrade --force-reinstall \
  "transformers==4.57.1"
```

최종 확인된 버전은 다음과 같다.

```text
torch        2.10.0
torchvision  0.25.0
transformers 4.57.1
lerobot      0.4.4
```

`requirements.txt`와 `pyproject.toml`에도 다음 제한을 추가했다.

```text
transformers>=4.57.1,<5
```

## 20. CUDA 장치 오류 진단

GPU smoke training을 다음 명령으로 실행했다.

```bash
python3 -m so101_mujoco_il.cli train \
  --dataset-root data/so101_pick_place_physics \
  --repo-id local/so101_mujoco_pick_place_physics \
  --output outputs/act_so101_smoke \
  --steps 10 \
  --batch-size 2 \
  --device cuda
```

ACTPolicy를 CUDA 장치로 이동하는 과정에서 다음 오류가 발생했다.

```text
torch.AcceleratorError:
CUDA-capable device(s) is/are busy or unavailable
```

환경을 직접 확인한 결과는 다음과 같았다.

```text
torch.version.cuda: 12.8
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
RuntimeError: No CUDA GPUs are available
```

`nvidia-smi` 결과도 다음과 같았다.

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

이 결과는 PyTorch가 CUDA 지원 빌드로 설치되어 있지만 운영체제의 NVIDIA driver 또는 GPU device에 접근하지 못한다는 뜻이다. Transformers 또는 ACT 코드 문제가 아니다.

### CUDA 조기 검사 추가

학습 및 추론 코드에 다음 검사를 추가했다.

```python
if requested_device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(...)
```

이제 CUDA를 사용할 수 없는 상태에서 깊은 PyTorch traceback 대신 다음 내용을 바로 안내한다.

- CPU smoke test에는 `--device cpu`를 사용할 것
- `nvidia-smi`를 확인할 것
- `torch.cuda.is_available()`을 확인할 것
- 본 학습 전 NVIDIA driver/device를 복구할 것

ACTConfig를 생성할 때도 device를 생성자에 직접 전달하도록 변경했다.

```python
config = ACTConfig(
    input_features=inputs,
    output_features=outputs,
    device=str(device_obj),
)
```

이 변경으로 config 초기화 중 `Device 'None'` 경고가 발생하지 않도록 했다.

## 21. CPU smoke test에서 발견한 이미지 차원 오류

CUDA 문제와 학습 코드 문제를 분리하기 위해 CPU smoke test를 실행했다.

```bash
python3 -m so101_mujoco_il.cli train \
  --dataset-root data/so101_pick_place_physics \
  --repo-id local/so101_mujoco_pick_place_physics \
  --output outputs/act_so101_smoke \
  --steps 10 \
  --batch-size 2 \
  --device cpu
```

ResNet convolution 입력에서 다음 오류가 발생했다.

```text
RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d,
but got input of size: [2, 1, 3, 224, 224]
```

### 오류 원인

기존 학습 코드는 action뿐 아니라 이미지 feature에도 delta timestamp를 적용했다.

```python
delta_timestamps.update(
    {
        key: deltas(config.observation_delta_indices)
        for key in config.image_features
    }
)
```

현재 ACTConfig의 image observation delta가 `[0]`이어서 LeRobotDataset은 이미지에 길이 1의 시간축을 추가했다.

DataLoader 출력은 다음과 같았다.

```text
[batch, time, channel, height, width]
[2, 1, 3, 224, 224]
```

현재 설치된 LeRobot ACT의 ResNet backbone은 다음 형식을 기대한다.

```text
[batch, channel, height, width]
[2, 3, 224, 224]
```

## 22. Delta timestamp 수정

ACT 학습에서 미래 action chunk는 필요하지만 현재 카메라 이미지에는 불필요한 시간축이 필요하지 않다. 따라서 action에만 delta timestamp를 적용하도록 변경했다.

수정된 코드:

```python
delta_timestamps = {
    "action": deltas(config.action_delta_indices),
}
```

이미지용 delta timestamp 추가 코드는 제거했다.

수정 후 실제 dataset item shape는 다음과 같다.

```text
observation.images.overview (3, 224, 224)
observation.state           (6,)
action                      (100, 6)
action_is_pad               (100,)
```

DataLoader가 batch dimension을 추가하면 ACT 입력은 다음 형태가 된다.

```text
observation.images.overview (B, 3, 224, 224)
observation.state           (B, 6)
action                      (B, 100, 6)
action_is_pad               (B, 100)
```

## 23. CPU 실제 학습 검증

수정 후 사용자의 conda LeRobot 환경과 동일한 Python으로 실제 CPU 1스텝 학습을 실행했다.

```bash
HF_HOME=/tmp/so101_hf_smoke \
/home/robo/anaconda3/envs/lerobot/bin/python \
  -m so101_mujoco_il.cli train \
  --dataset-root data/so101_pick_place_physics \
  --repo-id local/so101_mujoco_pick_place_physics \
  --output /tmp/act_cpu_smoke \
  --steps 1 \
  --batch-size 1 \
  --device cpu
```

검증 결과:

```text
step 0000001/0000001 loss=86.456650
checkpoint saved to /tmp/act_cpu_smoke
```

다음 전체 경로가 실제로 동작하는 것을 확인했다.

1. LeRobotDatasetMetadata 로드
2. 물리 그리핑 데이터셋 로드
3. 현재 RGB 이미지 로드
4. 미래 100스텝 action chunk 구성
5. LeRobot preprocessor 실행
6. ACTPolicy forward 실행
7. loss 계산
8. backward 실행
9. optimizer step 실행
10. ACT policy 체크포인트 저장
11. preprocessor 저장
12. postprocessor 저장

## 24. 현재 권장 학습 순서

### CPU smoke test

```bash
python3 -m so101_mujoco_il.cli train \
  --dataset-root data/so101_pick_place_physics \
  --repo-id local/so101_mujoco_pick_place_physics \
  --output outputs/act_so101_smoke \
  --steps 10 \
  --batch-size 2 \
  --device cpu
```

### GPU 상태 확인

```bash
nvidia-smi
```

```bash
python3 - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### GPU 본 학습

GPU가 정상적으로 인식될 때 다음 명령을 사용한다.

```bash
python3 -m so101_mujoco_il.cli train \
  --dataset-root data/so101_pick_place_physics \
  --repo-id local/so101_mujoco_pick_place_physics \
  --output outputs/act_so101_physics \
  --steps 100000 \
  --batch-size 32 \
  --device cuda
```

GPU 메모리가 부족하면 batch size를 다음 순서로 줄인다.

```text
32 → 16 → 8 → 4
```

## 25. 후속 작업 현재 상태

- 물리 그리핑 데이터 100 episodes 확인 완료
- LeRobot v3 metadata와 feature 확인 완료
- ACT 직접 import 경로 수정 완료
- Transformers 4.57.1 호환 상태 확인 완료
- Transformers 5.x 비호환 원인 확인 완료
- CUDA 오류가 NVIDIA driver/device 문제임을 확인 완료
- CUDA 조기 검사와 명확한 오류 메시지 추가 완료
- ACTConfig device 전달 방식 수정 완료
- 이미지에 추가되던 불필요한 singleton time axis 제거 완료
- 미래 action chunk `(100, 6)` 유지 완료
- CPU ACT forward/backward/optimizer step 검증 완료
- CPU 체크포인트 저장 검증 완료
- GPU 본 학습은 NVIDIA driver/device 복구 후 실행 필요
