# SO101 MuJoCo 이미테이션 러닝 개선 작업 기록

## 1. 작업 목적

기존 폴더의 MuJoCo 기반 SO101 ACT 코드를 검토하고, 다음 작업을 수행할 수 있는 새로운 파이프라인을 구축했다.

1. SO101 로봇이 IK를 이용해 빨간 블록을 집는다.
2. 블록을 지정된 초록색 목표 위치에 놓는다.
3. 블록 시작 위치를 랜덤하게 바꾸면서 기본 100개의 합성 궤적을 만든다.
4. 합성 데이터를 LeRobot 데이터셋으로 저장한다.
5. LeRobot의 ACT 정책 클래스로 학습한다.
6. 학습한 체크포인트로 MuJoCo 추론을 실행한다.
7. 데이터 생성과 추론을 headless 및 UI 모드에서 모두 실행할 수 있게 한다.
8. UI에는 `mujoco.viewer` 대신 GLFW를 사용한다.
9. 기존 실제 리더 및 가상 팔로워 코드는 수정하지 않는다.

## 2. 기존 코드에서 확인한 문제

기존 `104_imitation_learning_ACT` 코드는 다음과 같은 문제가 있었다.

- 학습 코드가 미래 액션 chunk를 데이터셋에서 올바르게 구성하지 않고 단일 액션을 반복하는 방식이었다.
- 학습과 추론 사이의 이미지, 관절 상태 및 액션 전처리가 완전히 일치하지 않았다.
- MuJoCo joint ID를 그대로 `qpos` 인덱스로 사용하는 부분이 있어 모델 구조가 변경되면 잘못된 관절을 읽을 가능성이 있었다.
- 추론 코드가 시뮬레이션에 필요하지 않은 실제 리더 모터 및 캘리브레이션 코드에 의존했다.
- 학습 코드와 추론 코드에 별도의 ACT 구현이 중복되어 체크포인트 호환성을 유지하기 어려웠다.
- 기존 pick-and-place 코드는 정해진 관절 포즈 사이를 보간할 뿐, 블록 위치에 대한 IK를 수행하지 않았다.

이 파일들은 사용자의 요청에 따라 수정하지 않았다. 새 구현은 `so101_mujoco_il` 디렉터리에 분리했다.

## 3. 새로 추가한 구조

### `so101_mujoco_il/assets/pick_place.xml`

- 기존 SO101 MuJoCo 모델을 include 방식으로 읽어온다.
- 빨간색 3 cm 블록을 추가했다.
- 초록색 원통형 place target을 추가했다.
- 고정 overview 카메라와 조명, 바닥을 추가했다.
- 기존 로봇 XML과 mesh 파일은 읽기만 하며 수정하지 않는다.

### `so101_mujoco_il/env.py`

- SO101 pick-and-place 전용 MuJoCo 환경이다.
- 6개 관절 이름과 MuJoCo address table을 이용해 정확한 `qpos`, `qvel`, actuator 인덱스를 구한다.
- 블록 시작 위치를 지정된 workspace 안에서 랜덤화한다.
- 관절 target position을 액션으로 받는다.
- RGB observation, 관절 상태 및 블록 위치를 제공한다.
- headless 환경에서는 MuJoCo offscreen renderer를 사용할 수 있다.
- 양쪽 finger pad와 블록의 실제 MuJoCo 접촉을 검사한다.
- 양쪽 접촉이 확인된 뒤에만 최대 3 N의 제한된 compliant force로 마찰 이송을 안정화한다.
- 블록의 `qpos`를 직접 변경하거나 equality weld를 사용하지 않는다.
- gripper가 열리면 compliant force를 제거하고 자유 물리 상태로 복귀한다.

### `so101_mujoco_il/ik.py`

- MuJoCo의 `mj_jacSite`로 `graspframe`의 위치 Jacobian을 계산한다.
- Damped Least Squares 방식으로 목표 TCP 위치에 해당하는 관절값을 구한다.
- 관절 제한 범위를 적용한다.
- 한 초기값에서 수렴하지 않을 때 home pose와 목표 방향 기반 초기값으로 다시 시도한다.
- 별도의 `MjData`를 사용해 IK 계산이 실제 실행 중인 환경 상태를 오염시키지 않게 했다.

### `so101_mujoco_il/expert.py`

다음 keyframe 순서로 전문가 시연을 생성한다.

1. 블록 위로 접근
2. 블록까지 하강
3. gripper 닫기
4. 블록 들어 올리기
5. 목표 지점 위로 이동
6. 목표 지점까지 하강
7. gripper 열기
8. 위로 후퇴

각 keyframe 사이는 cubic smooth-step 보간을 사용한다. 물리 그리핑 버전의 한 에피소드는 현재 208프레임이다.

각 프레임에 다음 값을 기록한다.

- 현재 6축 관절 상태
- 다음 6축 관절 target action
- overview RGB 이미지

### `so101_mujoco_il/viewer.py`

- `mujoco.viewer`를 사용하지 않는 GLFW 전용 뷰어다.
- GLFW window에 MuJoCo scene을 직접 그린다.
- 같은 GLFW context에서 데이터셋용 RGB 이미지를 캡처할 수 있다.
- 데이터 생성 UI와 추론 UI에 공통으로 사용할 수 있다.

### `so101_mujoco_il/dataset.py`

- IK 전문가 궤적을 LeRobot 데이터셋으로 저장한다.
- 기본 episode 수는 CLI에서 100개로 설정했다.
- feature 구성은 다음과 같다.

```text
observation.images.overview : uint8 RGB image, (3, 224, 224)
observation.state           : float32, (6,)
action                      : float32, (6,)
task                        : pick-and-place task 문자열
```

- 이미지는 video 대신 PNG 기반 LeRobot `image` feature로 저장한다.
- 기존 데이터셋을 실수로 덮어쓰지 않도록 대상 폴더가 비어 있지 않으면 중단한다.
- 에피소드마다 `save_episode()`를 호출한다.
- LeRobot v3에서 제공되는 경우 마지막에 `finalize()`를 호출한다.

### `so101_mujoco_il/train.py`

- 자체 구현한 비공식 ACT 모델 대신 LeRobot의 `ACTConfig`와 `ACTPolicy`를 사용한다.
- LeRobot 데이터셋 metadata에서 policy input/output feature를 구성한다.
- ACT action chunk에 필요한 delta timestamp를 구성한다.
- LeRobot의 공식 preprocessor와 postprocessor를 사용한다.
- 학습 완료 후 policy, preprocessor, postprocessor를 같은 출력 디렉터리에 저장한다.

### `so101_mujoco_il/inference.py`

- `ACTPolicy.from_pretrained()`로 저장된 정책을 읽는다.
- 학습할 때 저장된 전처리 및 후처리 설정을 불러온다.
- 매 스텝 현재 RGB 이미지와 6축 관절 상태로 액션을 예측한다.
- headless 또는 GLFW UI 방식으로 실행할 수 있다.
- 각 에피소드 종료 후 블록과 목표 사이 XY 오차를 측정한다.
- 목표에서 4.5 cm 이내에 놓인 경우 성공으로 판정한다.

### `so101_mujoco_il/cli.py`

다음 명령을 하나의 CLI로 제공한다.

```text
demo      IK 전문가 동작 확인
generate  랜덤 합성 LeRobot 데이터 생성
train     LeRobot ACT 학습
infer     학습된 ACT 정책 추론
```

headless 명령에서는 `MUJOCO_GL=egl`을 기본으로 설정한다.

## 4. LeRobot import 오류 수정

사용자의 conda 환경에서 다음 오류가 발생했다.

```text
ImportError: cannot import name 'LeRobotDataset' from 'lerobot.datasets'
```

확인된 환경은 다음과 같다.

```text
Python: /home/robo/anaconda3/envs/lerobot/bin/python
LeRobot metadata version: 0.4.4
LeRobot source: /home/robo/lerobot/src/lerobot
Git 상태: v0.4.3-9-g520a3cde, main branch
```

이 LeRobot 소스는 `lerobot.datasets`가 namespace package로 구성되어 있어 최상위에서 `LeRobotDataset`을 재노출하지 않는다. 따라서 다음과 같이 직접 모듈에서 import하도록 수정했다.

```python
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
```

오류 메시지도 단순히 라이브러리가 설치되지 않았다고 표시하지 않고 원래 import 오류를 함께 보여주도록 개선했다.

## 5. 현재까지의 검증 결과

### IK와 pick-and-place

- 서로 다른 50개의 랜덤 seed로 전문가 궤적을 실행했다.
- 50개 모두 IK 생성과 블록 배치를 완료했다.
- 물리 그리핑 버전의 한 에피소드는 208프레임이다.
- 대표 최종 블록 위치는 약 `[0.202, -0.151, 0.015]`였다.
- 목표 위치는 `[0.20, -0.16]`이다.

### Headless 렌더링

- EGL 기반 offscreen 렌더링을 실행했다.
- 생성 이미지 크기: `(224, 224, 3)`
- dtype: `uint8`
- 픽셀 범위가 정상적으로 생성되는 것을 확인했다.

### 자동 테스트

새로 추가한 `tests/test_pipeline.py`에서 다음을 검사한다.

- SO101 관절 및 actuator 개수
- 관절 상태가 joint limit 안에 있는지 여부
- 전문가 궤적 길이와 state/action shape
- 랜덤 시작점에서 목표 위치 배치 성공 여부

시스템 Python 환경의 실행 결과는 다음과 같다.

```text
3 passed
```

### 실제 LeRobot 데이터 저장

사용자의 conda `lerobot` 환경에서 1개 에피소드를 실제 생성했다.

```text
Map: 100% | 180/180
episode 001/001 saved
```

다음 LeRobot 데이터 파일이 생성되는 것을 확인했다.

```text
data/chunk-000/file-000.parquet
meta/info.json
meta/stats.json
meta/tasks.parquet
```

테스트 데이터는 `/tmp` 아래에 생성한 후 제거했으며, 사용자의 실제 데이터 폴더는 변경하지 않았다.

## 6. 현재 실행 명령

### IK headless 확인

```bash
python3 -m so101_mujoco_il.cli demo --headless --seed 0
```

### GLFW UI 확인

```bash
python3 -m so101_mujoco_il.cli demo --seed 0
```

### 100개 데이터 생성

```bash
python3 -m so101_mujoco_il.cli generate \
  --root data/so101_pick_place \
  --repo-id local/so101_mujoco_pick_place \
  --episodes 100
```

### UI를 표시하면서 데이터 생성

```bash
python3 -m so101_mujoco_il.cli generate \
  --root data/so101_pick_place \
  --repo-id local/so101_mujoco_pick_place \
  --episodes 100 \
  --ui
```

### ACT 학습

```bash
python3 -m so101_mujoco_il.cli train \
  --dataset-root data/so101_pick_place \
  --repo-id local/so101_mujoco_pick_place \
  --output outputs/act_so101 \
  --steps 100000 \
  --batch-size 32 \
  --device cuda
```

### Headless 추론

```bash
python3 -m so101_mujoco_il.cli infer \
  --checkpoint outputs/act_so101 \
  --episodes 10
```

### GLFW UI 추론

```bash
python3 -m so101_mujoco_il.cli infer \
  --checkpoint outputs/act_so101 \
  --episodes 10 \
  --ui
```

## 7. 남아 있는 확인 사항

### ACT 모듈 import 상태

현재 `/home/robo/lerobot` checkout에서 ACT 관련 모듈을 불러오면 다음 오류가 별도로 발생하는 것을 확인했다.

```text
TypeError: non-default argument 'backbone_cfg' follows default argument
```

이 오류는 이번에 수정한 데이터셋 import 오류와는 별개다. 현재 LeRobot checkout에서 `lerobot.policies`를 초기화할 때 여러 정책 설정을 한꺼번에 import하며 발생한다. 따라서 데이터 생성은 정상 동작하지만 실제 ACT 학습을 시작하기 전에 이 LeRobot checkout과 설치 의존성의 호환 상태를 추가로 정리해야 할 수 있다.

### GLFW 실행 검증

Codex 작업 환경에는 X11 display가 없었기 때문에 GLFW 창을 실제로 표시하는 테스트는 수행하지 못했다. GLFW 렌더러와 이미지 캡처 코드는 구현했으며, 문법 검사와 headless 렌더링은 통과했다. 실제 desktop session에서 `--ui` 실행 확인이 필요하다.

### 전체 100개 데이터와 학습

- 1개 LeRobot episode 저장은 실제 검증했다.
- 100개 전체 생성은 실행 시간이 길어 자동 검증 단계에서는 수행하지 않았다.
- ACT 전체 학습과 학습 체크포인트 추론은 현재 LeRobot policy import 문제를 해결한 후 최종 검증해야 한다.

## 8. 기존 파일 보호 상태

기존 실제 리더/가상 팔로워 관련 코드는 수정하지 않았다. 작업 시작 전부터 상위 `archive` 디렉터리에 존재하던 GIF 및 PNG 변경 사항도 건드리지 않았다.

이번 작업에서 추가하거나 수정한 주요 대상은 다음과 같다.

```text
so101_mujoco_il/
tests/test_pipeline.py
README.md
requirements.txt
pyproject.toml
.gitignore
word1.md
```
