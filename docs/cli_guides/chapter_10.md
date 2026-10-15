## 실행 안내: 실제 리더 수집부터 가상 팔로워 추론까지

명령 형태는 `python -m so101_real_leader_mujoco_il.cli <명령> [옵션]`이다. calibration·collect는 실제 리더를 사용하고, train·infer는 수집된 데이터만 있으면 리더 연결 없이 실행한다. 실제 팔로워 모터가 아니라 MuJoCo 팔로워를 제어하는 장이다.

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
python -m so101_real_leader_mujoco_il.cli --help
python -m so101_real_leader_mujoco_il.cli calibrate --help
python -m so101_real_leader_mujoco_il.cli collect --help
python -m so101_real_leader_mujoco_il.cli inspect --help
python -m so101_real_leader_mujoco_il.cli train --help
python -m so101_real_leader_mujoco_il.cli infer --help
```

설치 후 `so101-real-leader-il`도 같은 진입점이다. 아래 예시는 데이터 `data/book10_demo`, repo-id `local/book10_demo`, calibration `data/book10_leader.json`을 이어 사용한다. 자신이 바꾸는 값은 후속 명령에도 동일하게 반영한다.

### 1. 리더 포트와 중심 정렬: `calibrate`

```bash
python -m serial.tools.list_ports -v
```

포트를 확인한 뒤 calibration을 실행한다.

| 옵션 | 기본값/필수 | 역할 |
|---|---|---|
| `--port` | `/dev/ttyUSB0` | 실제 리더 serial 포트 |
| `--baudrate` | `1000000` | 리더 통신 속도; collect의 기본 드라이버는 1,000,000 사용 |
| `--xml` | `자동 선택` | 생략하면 패키지 assets/lift_cube_calibration.xml |
| `--output` | `자동 선택` | 생략하면 패키지 assets/full_arm_calibration_leader.json |

```bash
python -m so101_real_leader_mujoco_il.cli calibrate \
  --port /dev/ttyUSB0 \
  --baudrate 1000000 \
  --output data/book10_leader.json
```

각 관절을 MuJoCo 0 rad 자세, 최소 각도 자세, 최대 각도 자세로 옮기며 터미널의 Enter 요청을 따른다. 같은 output이 있으면 기존 calibration의 timestamp 백업을 만든다. 이 단계의 중심 정보는 7장의 JSON을 이름만 바꿔 대체하는 것과 다르다.

### 2. 리더로 시연 기록: `collect`

| 옵션 | 기본값/필수 | 역할 |
|---|---|---|
| `--dataset-root` | **필수** | 새로 수집할 로컬 데이터셋 경로; 필수 |
| `--repo-id` | `local/so101_real_leader_mujoco` | 데이터 식별자; inspect/train과 동일하게 사용 |
| `--leader-port` | `/dev/ttyUSB0` | 실제 리더 포트; calibrate에서 쓰는 이름은 --port인 점에 주의 |
| `--calibration` | `자동 선택` | 직전에 저장한 JSON; 생략 시 패키지 기본 JSON |
| `--xml` | `자동 선택` | 가상 팔로워 XML; 생략 시 기본 asset |
| `--fps` | `30` | 목표 수집 빈도와 데이터 메타데이터; 양의 정수 |
| `--seed` | `0` | 가상 장면 초기화 seed |
| `--recover-empty` | `False` | 검증된 0프레임 초기화 잔여 폴더만 보관 후 다시 생성 |

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/book10_demo \
  --repo-id local/book10_demo \
  --leader-port /dev/ttyUSB0 \
  --calibration data/book10_leader.json \
  --fps 30 \
  --seed 0
```

MuJoCo 창과 OpenCV 미리보기 창을 사용한다. 키 입력은 OpenCV 이미지 창에 포커스를 둔 상태에서 한다.

| 키 | 동작 |
|---|---|
| `r` | 기록 시작 |
| `s` | 현재 에피소드 저장 후 장면 reset |
| `d` | 현재 기록 폐기 후 reset |
| `q` | 종료; 미완성 에피소드 자동 저장 없음 |

여러 에피소드는 프로그램 실행 중 `r`→시연→`s`를 반복해 기록한다. CLI에 `--episodes`나 기존 데이터에 이어 붙이는 `--resume`가 없다. 데이터가 든 경로로 프로그램을 다시 실행하면 거부되므로 새로운 수집 세션은 새 root를 사용한다.

`--recover-empty`는 `meta/info.json`만 있고 total episodes/frames가 모두 0인 초기화 잔여 폴더에만 사용할 수 있다. 백업 후 재생성하며 정상 데이터의 덮어쓰기나 이어 수집 옵션은 아니다.

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/book10_demo \
  --repo-id local/book10_demo \
  --leader-port /dev/ttyUSB0 \
  --calibration data/book10_leader.json \
  --fps 30 \
  --seed 0 \
  --recover-empty
```

복구 조건이 맞을 때만 위 명령을 사용한다. `collect`에는 `--baudrate`가 없고 드라이버 기본 1,000,000을 사용한다. `--fps`를 바꿔 수집할 수 있지만 현재 infer 루프는 30Hz이므로 처음에는 30을 유지한다. `--headless`·`--ui`도 없으며 collect는 GUI 경로이다.

### 3. 저장한 에피소드 검사: `inspect`

| 옵션 | 기본값/필수 | 역할 |
|---|---|---|
| `--dataset-root` | **필수** | collect의 root |
| `--repo-id` | `local/so101_real_leader_mujoco` | collect의 repo-id |
| `--episode` | `0` | 확인할 0부터 시작하는 에피소드 번호 |
| `--delay-ms` | `33` | OpenCV 프레임 표시 대기 ms; 양수 권장, 0은 키 입력 대기 |

```bash
python -m so101_real_leader_mujoco_il.cli inspect \
  --dataset-root data/book10_demo \
  --repo-id local/book10_demo \
  --episode 0 \
  --delay-ms 33
```

이미지와 state/action 값을 비교하고 `q`로 종료한다. 하드웨어를 열지는 않지만 표시용 GUI는 필요하다. 저장 에피소드가 없으면 inspect에서 진행하지 말고 collect의 `s` 저장 결과부터 확인한다.

### 4. 학습: `train`

| 옵션 | 기본값/필수 | 역할 |
|---|---|---|
| `--dataset-root` | **필수** | 검사한 데이터셋 로컬 경로 |
| `--repo-id` | `local/so101_real_leader_mujoco` | 수집·검사와 같은 식별자 |
| `--output` | **필수** | 정책·전처리·후처리 저장 디렉터리 |
| `--steps` | `100000` | optimizer 업데이트 횟수; epoch가 아님 |
| `--batch-size` | `32` | 업데이트당 표본 수; 유효 프레임 수 이하로 설정 |
| `--device` | `자동 선택` | 생략 시 CUDA 가능 여부로 선택; cpu/cuda 명시 가능 |

먼저 작은 설정으로 데이터→저장 경로를 확인한다.

```bash
python -m so101_real_leader_mujoco_il.cli train \
  --dataset-root data/book10_demo \
  --repo-id local/book10_demo \
  --output outputs/book10_smoke \
  --steps 2 \
  --batch-size 2 \
  --device cpu
```

충분한 시연을 모은 뒤 본 학습을 별도 출력 디렉터리로 실행한다.

```bash
python -m so101_real_leader_mujoco_il.cli train \
  --dataset-root data/book10_demo \
  --repo-id local/book10_demo \
  --output outputs/book10_act \
  --steps 100000 \
  --batch-size 32 \
  --device cuda
```

GPU가 없으면 CPU로 바꾼다. 메모리 부족은 batch를 먼저 낮춘다. 데이터가 batch보다 적으면 오류가 발생한다. 현재 CLI는 `--lr`, `--epochs`, `--chunk-size`, `--resume`를 제공하지 않는다. `train.py`의 `ACTConfig`·optimizer preset이 구조와 학습률을 정하며, 마지막에 output 폴더로 저장한다. 예전 checkpoint를 자동 이어 학습하지 않는다.

### 5. 리더 없이 정책 실행: `infer`

| 옵션 | 기본값/필수 | 역할 |
|---|---|---|
| `--checkpoint` | **필수** | train의 output 디렉터리; 필수 |
| `--device` | `자동 선택` | 정책 추론 장치; 생략 시 자동 선택 |
| `--seed` | `1000` | 가상 장면 초기화 seed |
| `--xml` | `자동 선택` | 수집과 호환되는 모델; 생략 시 기본 asset |

```bash
python -m so101_real_leader_mujoco_il.cli infer \
  --checkpoint outputs/book10_act \
  --device cpu \
  --seed 1000
```

MuJoCo·OpenCV 창을 사용하며 OpenCV 창에서 `r`로 reset, `q`로 종료한다. 리더 serial 포트와 calibration을 읽는 명령이 아니므로 `--leader-port`를 붙이지 않는다. 9장의 infer와 달리 `--episodes`, `--max-steps`, `--ui`, `--headless`가 없다. 창을 닫거나 종료할 때까지 실행한다.

최소 checkpoint 구성은 `config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json`이고 통계 파일 등 폴더 전체를 보존한다. 모델 XML을 바꾸려면 calibrate·collect·infer의 `--xml`을 일관되게 설정하고 관절 순서·단위·카메라 의미가 학습 데이터와 맞는지 확인한다. 기본 모델을 그대로 쓰는 첫 실습은 `--xml`을 생략한다.
