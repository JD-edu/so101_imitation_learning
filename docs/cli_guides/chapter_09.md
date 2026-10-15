## 실행 안내: IK 시연 생성·공식 ACT CLI

이 장은 `argparse` 기반 CLI를 제공한다. 기본 형태는 `python -m so101_mujoco_ik.cli <명령> [옵션]`이다. 옵션은 `generate`, `train` 같은 **하위 명령 뒤에** 쓴다. 숫자는 양의 episodes/steps/batch-size를 사용한다. `--ui`, `--headless`는 값을 받지 않는 스위치이므로 `--ui True`처럼 쓰지 않는다.

### 환경과 설치 위치

```bash
conda activate lerobot
cd /home/robo/so101_imitation_learning/09_10_so101_mujoco_imitation_learning
python -m pip install --no-deps --no-build-isolation -e .
```

이 설치 명령은 **필요한 의존성이 이미 준비된 환경**에서 현재 저장소를 editable 패키지로 연결한다. 새로운 환경에서 프로젝트에 선언된 학습 의존성도 설치하려면 같은 폴더에서 다음을 사용한다.

```bash
python -m pip install -e '.[train]'
```

두 번째 명령은 네트워크를 사용하여 의존성을 설치하거나 변경할 수 있다. CPU/CUDA에 맞는 PyTorch 환경을 사용한다. 다음으로 현재 Python과 장치 상태를 확인한다.

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

아래 상대 경로는 모두 이 프로젝트 폴더 기준이다. `--device`를 생략하면 사용 가능한 CUDA, 아니면 CPU를 선택한다. `--device cuda`를 명시했는데 CUDA가 없으면 자동 CPU 대체 대신 오류가 난다. CPU 실행은 `--device cpu`로 지정한다.

### 도움말

```bash
python -m so101_mujoco_ik.cli --help
python -m so101_mujoco_ik.cli demo --help
python -m so101_mujoco_ik.cli generate --help
python -m so101_mujoco_ik.cli train --help
python -m so101_mujoco_ik.cli infer --help
```

`-e .` 설치 후에는 `so101-ik` 명령도 같은 진입점이다. 본문에서는 어떤 Python 환경을 쓰는지 명확히 하기 위해 `python -m`을 사용한다.

### 1. expert 데모 확인: `demo`

| 옵션 | 기본값/필수 | 역할 |
|---|---|---|
| `--seed` | `0` | 물체·목표 초기화의 난수 seed |
| `--headless` | `False` | 창 없이 한 번의 expert 시연 생성; 생략하면 GUI 재생 |

```bash
python -m so101_mujoco_ik.cli demo --seed 0 --headless
```

데이터셋이나 checkpoint를 저장하지 않고 프레임 수·마지막 블록 위치를 출력한다. 현재 expert의 기본 시연은 208프레임이다. GUI에서 보려면 다음을 실행한다.

```bash
python -m so101_mujoco_ik.cli demo --seed 0
```

`demo`에는 `--ui`가 없으며 기본이 GUI, `--headless`가 창 생략이다. 뒤의 `generate`·`infer`는 이와 반대로 `--ui`를 붙일 때 창을 표시한다.

### 2. 학습 데이터 생성: `generate`

| 옵션 | 기본값/필수 | 역할 |
|---|---|---|
| `--root` | `data/so101_pick_place` | 생성할 로컬 데이터셋 디렉터리; 비어 있거나 없어야 함 |
| `--repo-id` | `local/so101_mujoco_pick_place` | LeRobot 데이터 식별자; 뒤 train과 같게 설정 |
| `--episodes` | `100` | 저장할 시연 에피소드 수 |
| `--seed` | `0` | 시연 환경의 난수 seed |
| `--ui` | `False` | 시연 생성 과정을 GUI에서 보기; 생략하면 창 없이 이미지 렌더링 |

처음에는 2개 에피소드로 저장 경로와 데이터 구조를 확인한다.

```bash
MUJOCO_GL=egl python -m so101_mujoco_ik.cli generate \
  --root data/book09_demo \
  --repo-id local/book09_demo \
  --episodes 2 \
  --seed 0
```

줄 끝의 `\`는 같은 명령의 다음 줄이라는 뜻이며 뒤에 공백을 넣지 않는다. `MUJOCO_GL=egl`은 해당 프로세스의 offscreen 렌더링 환경변수이다. EGL 환경이 필요하며 단순한 `--device cpu` 선택이 렌더링 문제를 해결하지는 않는다.

GUI가 가능한 환경에서는 별도의 빈 디렉터리로 다음과 같이 수집할 수 있다.

```bash
python -m so101_mujoco_ik.cli generate \
  --root data/book09_gui \
  --repo-id local/book09_gui \
  --episodes 2 \
  --seed 0 \
  --ui
```

출력은 `--root` 아래의 LeRobot 메타데이터·관측 이미지·상태·행동이다. 실제 하드웨어는 필요하지 않다. 기존 데이터가 있는 폴더는 `FileExistsError`가 발생하므로 새 경로를 지정한다. 생성 CLI에는 `--fps`, `--camera`, `--xml`, `--recover-empty`가 없다. fps는 현재 함수 기본값 30이며 모델·카메라 구성은 패키지 코드와 asset에서 정한다.

### 3. 짧은 학습 확인과 본 학습: `train`

| 옵션 | 기본값/필수 | 역할 |
|---|---|---|
| `--dataset-root` | `data/so101_pick_place` | 위 generate의 --root와 같은 로컬 경로 |
| `--repo-id` | `local/so101_mujoco_pick_place` | generate와 같은 데이터 식별자 |
| `--output` | `outputs/act_so101` | 정책과 전처리·후처리를 저장할 디렉터리 |
| `--steps` | `100000` | epoch 수가 아닌 optimizer 업데이트 횟수 |
| `--batch-size` | `32` | 업데이트당 표본 수; 메모리 부족 시 낮춤 |
| `--device` | `자동 선택` | cpu 또는 cuda 등의 torch 장치; 생략 시 자동 선택 |

```bash
python -m so101_mujoco_ik.cli train \
  --dataset-root data/book09_demo \
  --repo-id local/book09_demo \
  --output outputs/book09_smoke \
  --steps 2 \
  --batch-size 2 \
  --device cpu
```

이것은 경로·배치·저장 확인용 설정이며 작업을 수행할 정책을 얻기 위한 학습량이 아니다. 본 학습은 충분한 시연을 별도 수집한 다음 수행한다.

```bash
MUJOCO_GL=egl python -m so101_mujoco_ik.cli generate \
  --root data/book09_train \
  --repo-id local/book09_train \
  --episodes 100 \
  --seed 0
python -m so101_mujoco_ik.cli train \
  --dataset-root data/book09_train \
  --repo-id local/book09_train \
  --output outputs/book09_act \
  --steps 100000 \
  --batch-size 32 \
  --device cuda
```

CUDA 없는 환경은 `--device cpu`로 바꾼다. loader가 `drop_last=True`이므로 사용 가능한 표본 수보다 batch를 크게 잡지 않는다. 현재 CLI에는 `--lr`, `--epochs`, `--chunk-size`, `--resume`가 없다. 모델·optimizer는 `train.py`의 `ACTConfig`와 그 preset에서 구성한다. 이를 바꾸려면 소스 구성을 수정해야 한다. 중간 checkpoint나 자동 resume 기능은 없고 마지막에 `--output`으로 저장하므로 실험마다 출력 경로를 나눈다.

### 4. 저장 모델 실행: `infer`

| 옵션 | 기본값/필수 | 역할 |
|---|---|---|
| `--checkpoint` | `outputs/act_so101` | train의 --output 디렉터리; 단일 .pth 파일이 아님 |
| `--episodes` | `10` | 평가할 에피소드 수 |
| `--seed` | `1000` | 평가 초기화 seed; 학습과 다른 값을 사용 가능 |
| `--ui` | `False` | GUI 표시; 생략하면 offscreen 실행 |
| `--device` | `자동 선택` | 정책 추론 장치; 생략 시 자동 선택 |
| `--max-steps` | `240` | 에피소드별 최대 제어 스텝 수 |

```bash
MUJOCO_GL=egl python -m so101_mujoco_ik.cli infer \
  --checkpoint outputs/book09_act \
  --episodes 10 \
  --seed 1000 \
  --max-steps 240 \
  --device cpu
```

GUI 평가 예시는 다음과 같다.

```bash
python -m so101_mujoco_ik.cli infer \
  --checkpoint outputs/book09_act \
  --episodes 3 \
  --seed 1000 \
  --max-steps 240 \
  --device cpu \
  --ui
```

checkpoint에는 적어도 `config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json`이 필요하고, 전처리 통계 파일 등 저장 폴더 전체를 함께 보존한다. `--checkpoint`에 `checkpoints/last`를 자동 덧붙이지 않는다. 이 코드의 저장 위치는 `--output` 그 자체이다.

`ModuleNotFoundError`는 활성 환경과 editable 설치 경로, 데이터 오류는 root/repo-id 조합, checkpoint 오류는 학습 종료와 폴더 전체를 확인한다. 학습·추론의 이미지와 행동 전처리를 수동으로 한 번 더 적용하지 않는다.
