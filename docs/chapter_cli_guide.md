# 장별 실행·CLI 안내

4장 구조 확인은 독립 CLI, 5·6장의 기존 학습·추론은 소스 내부 설정, 9~10장은 패키지 CLI를 사용한다.

# 1장

## 실행 안내: 터미널과 Python 환경 준비

1장은 Tensor·자동미분의 개념과 독립 코드 조각을 다룬다. 이 장 전체를 실행하는 `chapter1.py`는 없다. 단독 예제는 `.py`로 저장하여 실행하고, 데이터셋 학습은 2장의 실제 파일을 사용한다.

### 환경과 작업 위치

저장소가 `/home/robo/so101_imitation_learning`에 있고 기존 `lerobot` conda 환경을 쓰는 경우의 명령이다. 다른 환경·경로라면 해당 부분을 바꾼다.

```bash
conda activate lerobot
cd /home/robo/so101_imitation_learning
python -c "import sys, torch; print(sys.executable); print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

`sys.executable`은 실제 실행 중인 Python 경로이다. CUDA가 `False`면 자동 장치 선택 예제는 CPU를 사용한다. `python -m pip`는 현재 Python에 연결된 패키지 설치 도구이다. 서로 다른 환경의 `pip`와 `python`을 섞지 않는다. 9~10장 패키지의 설치 명령은 해당 장에 있다.

### 다운로드 없이 자동미분 확인

아래 `python -`는 표준입력에서 코드를 실행하는 **Python 자체 옵션**이다. `PY` 사이의 코드를 터미널에 함께 붙여 넣는다.

```bash
python - <<'PY'
import torch
x = torch.tensor([2.0], requires_grad=True)
y = x.square().sum()
y.backward()
print('y =', y.item(), 'gradient =', x.grad.item())
PY
```

기대 출력은 `y = 4.0 gradient = 4.0`이다. 파일 저장·데이터셋·웹캠을 사용하지 않는다. 본문의 설명용 조각 중에는 앞에서 정의한 `model`, `optimizer`, `dataset`을 이어 쓰는 것도 있으므로 모든 블록이 단독 프로그램은 아니다.

### 실제 학습으로 넘어가기

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 101_pytorch_mnist.py
```

이 명령은 MNIST 다운로드와 기본 5 epoch 학습을 실행하고 `mnist_mlp.pth`를 현재 폴더에 저장한다. Tensor 연습만 하려면 자동미분 예제까지만 실행한다. 이 학습 파일에는 `--epochs`, `--device`, `--help` 처리가 없으므로 옵션을 붙이지 않는다. 다음 장에서 실제 설정 위치를 설명한다.


---

# 2장

## 실행 안내: 학습 파일과 웹캠 추론 파일

이 장의 단독 `.py`는 CLI 옵션을 해석하지 않는다. `--epochs`, `--device`, `--help`를 붙여도 설정이나 도움말로 처리되지 않으며 학습·카메라 코드가 실행될 수 있다. 실행 인자는 붙이지 않고 아래의 변수·함수 인자를 소스에서 바꾼다.

### 준비와 작업 폴더

```bash
conda activate lerobot
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python -c "import torch, torchvision, numpy, cv2, matplotlib, torchsummary; print(torch.__version__)"
```

MNIST/CIFAR-10과 사전학습 VGG/ResNet 가중치는 캐시가 없으면 다운로드한다. 커스텀 예제는 `images/` 아래의 클래스 폴더를 읽는다. `ImageFolder.class_to_idx`의 순서와 추론의 `class_names`를 맞춘다.

### 학습 후 대응 추론 실행

각 쌍에서 학습이 끝나 가중치가 저장된 뒤 추론을 실행한다. 필요한 예제 하나씩 선택한다.

```bash
python 101_pytorch_mnist.py
python 101_pytorch_mnist_inference.py
```

```bash
python 102_pytorch_CNN_cifar10.py
python 102_pytorch_CNN_cifar10_inference.py
```

```bash
python 103_pytorch_miniVGGnet.py
python 103_pytorch_miniVGGnet_inference.py
```

```bash
python 104_pytorch_VGGnet_transfer_feature_can.py
python 104_pytorch_VGGnet_transfer_feature_inference.py
```

```bash
python 105_pytorch_Resnet_transfer_fine.py
python 105_pytorch_Resnet_transfer_fine_inference.py
```

### CLI 대신 설정하는 위치

| 파일 | 기본값·수정 위치 | 결과물 |
|---|---|---|
| 101 MNIST | 상단 `batch_size=64`, `learning_rate=0.001`, `epochs=5` | `mnist_mlp.pth` |
| 102 CIFAR-10 | `main()`의 `batch_size=64`, optimizer `lr=0.001`, epoch 반복 `range(10)` | `cifar10_cnn.pth` |
| 103 MiniVGG | `if __name__ == "__main__":` 블록의 `IMG_SIZE=64`, `BATCH_SIZE=32`, `LEARNING_RATE=0.001`, `EPOCHS=100` | `miniVGGnet.pth` |
| 104 VGG16 | 상단 `IMG_SIZE=224`, `BATCH_SIZE=16`, `EPOCHS=20`, optimizer `lr=0.0001` | `vgg16_transfer_model.pth` |
| 105 ResNet18 | 상단 `IMG_SIZE=224`, `BATCH_SIZE=16`, `DATA_PATH='./images'`; `train_model(10, "Feature-Ex")`, `train_model(15, "Fine-Tuning")` | `resnet18_final.pth` |

예를 들어 MiniVGG의 실행만 점검하려면 실행 블록의 `EPOCHS`를 2로 바꾸고 같은 명령을 실행한다. 입력 크기나 클래스 수를 바꾸면 추론 구조도 맞춰야 하므로 먼저 epoch/batch부터 조정한다. 결과 이름은 고정이므로 재실행하면 같은 파일에 저장한다.

자동 장치 선택을 CPU로 제한하는 방법은 다음과 같다. `CUDA_VISIBLE_DEVICES`는 환경변수이며 스크립트 CLI 옵션이 아니다.

```bash
CUDA_VISIBLE_DEVICES="" python 101_pytorch_mnist.py
```

MiniVGG·VGG16·ResNet18은 그래프 창을 닫은 뒤 가중치를 저장한다. 추론은 카메라 0을 사용하며 다른 카메라는 소스의 `cv2.VideoCapture(0)`을 바꾼다. 이미지 창에서 `q`로 종료한다. 카메라가 없어도 학습은 가능하지만 웹캠 추론은 할 수 없다.

가중치 파일을 못 찾으면 학습 종료·그래프 종료·현재 폴더를 확인한다. GPU 메모리가 부족하면 batch를 낮춘다. `size mismatch`이면 클래스 수·학습/추론 구조·가중치 파일명을 함께 확인한다.


---

# 3장

## 실행 안내: MuJoCo 예제 선택과 설정

이 장의 파일은 `--xml`, `--headless`, `--help`를 해석하지 않는다. 옵션 없이 실행하며 모델·카메라·명령값은 Python 또는 XML에서 바꾼다.

```bash
conda activate lerobot
cd /home/robo/so101_imitation_learning/03_mujoco_basic
python -c "import mujoco, glfw, numpy; print('MuJoCo:', mujoco.__version__)"
```

본편의 GUI에는 디스플레이와 OpenGL이 필요하다. 창 없이 모델만 확인할 때는 부록 A의 `MjModel.from_xml_path` 진단 예제를 사용한다.

### 기본 예제 실행

각 블록의 `cd`부터 실행한다. 한 예제의 창을 닫은 다음 다음 예제를 실행한다.

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/101_mujoco_glfw_window
python 101_mujoco_glfw_window.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/102_mujoco_mjcf_load
python 102_mujoco_mjcf_load.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/103_mujoco_viewer
python 103_mujoco_viewer.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/104_glfw_mouse_move
python 104_glfw_mouse_move.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/105_glfw_asdw_move
python 105_glfw_asdw_move.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/106_mujoco_imu_sensor
python 106_mujoco_imu_sensor.py
```

### 로봇 모델 표시 후 움직임 실행

각 `mjcf_load.py` 창을 종료한 뒤 같은 폴더의 `robot_move.py`를 실행한다.

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/107_jdcobot100_MUJOCO_load/answer
python mjcf_load.py
python robot_move.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/108_puppypi_mujoco_load/answer
python mjcf_load.py
python robot_move.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/109_so101_mujoco_load/answer
python mjcf_load.py
python robot_move.py
```

### 동작을 바꾸는 위치

| 항목 | 실제 수정 위치 |
|---|---|
| 창 크기 | `glfw.create_window(width, height, ...)` |
| 카메라 시점 | Python의 카메라 distance·azimuth·elevation |
| 모델 | Python이 읽는 XML과 `scene.xml`의 `include` |
| 물리 시간 간격 | 실제 입력 XML의 `<option timestep="...">` |
| 105/106 바퀴 명령 | `velocity=40.0`, `left_handycap=0.7` |
| 107/109 목표 자세 | `robot_move.py`의 `pose_A`, `pose_B`, `switch_interval=2.0` |
| 108 Puppy 자세 | 최초 스텝 후 `qpos[7:]`에서 복사하는 목표; 현재 보행 옵션 없음 |

101의 빈 창은 정상이다. 104는 마우스 시점 조작, 105/106은 창에 포커스를 두고 W/A/S/D를 사용한다. 사용자 정의 GLFW 예제는 창 닫기로 종료한다. 103은 벽시계 20초 이후 정지 명령을 계속 유지한다.

URDF 변환은 선택 과정이다. 107/108/109의 `answer`에서 `python urdf_to_mjcf.py`를 실행하면 각각 `jdcobot100_converted.xml`, `puppy2_converted.xml`, `so101_converted.xml`을 만든다. 기존 scene이 자동으로 이 결과를 읽도록 바뀌지는 않으며, actuator 없는 변환 결과와 완성 제어 모델을 구분한다.


---

# 4장

## 실행 안내: 4장 전용 순차 모델 폴더

4장은 `04_sequence_models/`에서 RNN·Transformer의 구조와 입력·출력 shape만 확인한다. MuJoCo·GLFW·카메라·학습 데이터는 필요하지 않으며 PyTorch를 사용한다. 5장의 학습·추론 파일과 6장의 ACT 파일은 각각 별도 폴더에 있다.

```bash
conda activate lerobot
cd /home/robo/so101_imitation_learning/04_sequence_models
python 101_model_shapes.py --help
python 101_model_shapes.py
```

기본 실행 결과는 다음과 같다.

```text
VanillaRNN_BC: (2, 10, 6) -> (2, 6)
TransformerEncoderBC: (2, 10, 6) -> (2, 6)
```

### 이 실행 파일의 CLI 옵션

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--batch-size` | 2 | 동시에 확인할 입력 표본 수, 양의 정수 |
| `--history` | 10 | 상태 이력 길이, 양의 정수 |
| `--help` | 없음 | 도움말을 표시하고 종료 |

```bash
python 101_model_shapes.py --batch-size 3 --history 12
```

이 경우 두 출력은 `(3, 12, 6) -> (3, 6)`이다. Transformer 위치 표현도 주어진 이력 길이로 새로 만든다. 이 프로그램은 임의 초기 모델의 구조 점검이며 기존 가중치를 읽거나 학습·저장하지 않는다. 출력 후 자동 종료한다.

`sequence_models.py`에는 `VanillaRNN_BC`, `TransformerEncoderBC` 두 클래스가 있다. 5장 202/203의 모델 정의와 같은 구조를 독립된 학습용 설명 모듈로 옮겼으며, 같은 가중치를 넣었을 때 출력이 일치하는지 테스트한다.

실제 BC 학습은 다음 폴더로 이동하여 5장 안내를 따른다.

```bash
cd /home/robo/so101_imitation_learning/05_mujoco_bc
python 202_RNN_MUJOCO.py
```

5장의 기존 학습 파일에는 4장의 `--history`·`--batch-size` CLI가 없다. 5장은 소스의 상수를 조정하며 6장 ACT는 `06_mujoco_act/`에서 실행한다.


---

# 5장

## 실행 안내: MLP·RNN·Transformer BC

이 장의 스크립트에는 CLI parser가 없다. `--epochs`, `--checkpoint`, `--device`, `--help`를 붙여도 설정·도움말로 처리되지 않는다. 아래 변수는 소스에서 바꾼 뒤 옵션 없이 실행한다.

### 준비와 실행 순서

```bash
conda activate lerobot
cd /home/robo/so101_imitation_learning/05_mujoco_bc
python -c "import torch, mujoco, glfw; print(torch.__version__, mujoco.__version__)"
```

학습 파일은 합성 시연 생성→학습→checkpoint 저장→GUI 평가 순서로 실행한다. 외부 CSV를 입력받지 않으며 생성 시연은 메모리에 보관한다. 별도의 LeRobot 데이터셋 파일은 만들지 않는다. 필요한 단계 하나씩 실행하고, 학습의 평가 창을 닫은 뒤 별도 추론을 실행한다.

```bash
python 201_MLP_MUJOCO.py
python 201_MLP_MUJOCO_inference.py
```

```bash
python 202_RNN_MUJOCO.py
python 202_RNN_MUJOCO_inference.py
```

```bash
python 203_Transformer_MUJOCO.py
python 203_Transformer_MUJOCO_infer.py
```

### 학습 파일 상단의 설정

| 파일 | `NUM_EPISODES` | `STEPS_PER_EP` | `HISTORY` | `BATCH_SIZE` | `EPOCHS` | `LR` |
|---|---:|---:|---:|---:|---:|---:|
| 201 | 100 | 100 | 없음 | 64 | 40 | 0.001 |
| 202 | 100 | 100 | 10 | 64 | 35 | 0.001 |
| 203 | 100 | 100 | 10 | 64 | 40 | 0.001 |

빠른 실행 점검은 `NUM_EPISODES=10`, `EPOCHS=2`처럼 학습량부터 줄인다. 입력·출력 구조는 유지한다. 작은 설정으로는 정책 성능을 평가할 수 없다. 에피소드 분할 뒤 학습·평가 데이터가 모두 남도록 하고, 이력보다 짧은 시연을 만들지 않는다.

### 저장물과 추론

| 단계 | 코드 폴더에 저장되는 checkpoint |
|---|---|
| 201 | `so101_mlp_bc_p2p_v2.pt` |
| 202 | `so101_rnn_bc_p2p_v2.pt` |
| 203 | `transformer_encoder_bc_v2.pt` |

`MODEL_PATH`와 `XML_PATH`는 스크립트 폴더 기준이다. 다른 파일을 쓰려면 학습·추론 양쪽 상수를 맞춘다. 학습은 CUDA 가능 여부를 자동 판단하고, 별도 추론은 CPU로 읽는다. `_v2.pt`는 현재 시간 정렬로 새로 학습한 가중치이며 옛 파일의 이름만 바꿔 대체하지 않는다.

`--headless`나 `--no-eval`은 없다. GUI 없는 환경은 학습 종료 후 평가 창 생성에서 실패할 수 있으므로 4장의 구조 검사와 전체 실행을 구분한다. 창 닫기로 종료한다. 같은 설정으로 재학습하면 같은 checkpoint에 저장한다.


---

# 6장

## 실행 안내: Action Chunking·CVAE·시각 ACT

이 장의 파일에는 CLI parser가 없다. `--chunk-size`, `--epochs`, `--goal`, `--kl-weight`, `--checkpoint`를 지원하지 않는다. 상단 상수 또는 초기화 함수를 수정하고 옵션 없이 실행한다.

```bash
conda activate lerobot
cd /home/robo/so101_imitation_learning/06_mujoco_act
python -c "import torch, torchvision, mujoco, glfw, numpy, matplotlib; print(torch.__version__)"
```

학습 후 저장·GUI 평가까지 진행한다. 각 평가 창을 닫은 다음 추론 파일을 실행한다. 208은 데이터 수집 단계부터 숨겨진 GLFW 창을 만들어 OpenGL context를 사용한다. `MUJOCO_GL=egl`만 설정해도 전체 스크립트가 headless로 바뀌는 것은 아니다.

### 단계별 명령

```bash
python 204_Action_Chunking_MUJOCO.py
python 204_Action_Chunking_MUJOCO_infer.py
```

```bash
python 205_Action_Various_start.py
python 205_Action_Various_start_infer.py
```

```bash
python 206_ACT_RH_TE.py
python 206_ACT_RH_TE_infer.py
```

```bash
python 207_ACT_CAVE.py
python 207_ACT_CVAE_infer.py
```

```bash
python 208_ACT_Image.py
python 208_ACT_Image_infer.py
```

207 학습 파일의 실제 철자는 `CAVE`이고 대응 추론은 `CVAE_infer`이다. 이름이 비슷한 과거 실험 파일을 대신 사용하지 않는다.

### 학습 설정과 결과물

| 파일 | 에피소드 | epoch | batch | LR | 이력 H | 청크 K | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---|
| 204 | 100 | 45 | 64 | 0.001 | 10 | 30 | `action_chunking_transformer_v2.pt` |
| 205 | 100 | 45 | 64 | 0.001 | 10 | 30 | `action_chunking_transformer_multistart_v2.pt` |
| 206 | 100 | 45 | 64 | 0.001 | 10 | 30 | `action_chunking_rh_te_v2.pt` |
| 207 | 400 | 80 | 64 | 0.001 | 10 | 30 | `cvae_act_goal_transformer_v2.pt` |
| 208 | 200 | 100 | 32 | 0.0003 | 현재 영상·관절 | 30 | `visual_act_final_v2.pt` |

변수 이름은 `NUM_EPISODES`, `EPOCHS`, `BATCH_SIZE`, `LR`, `HISTORY`, `CHUNK_SIZE`이다. 모든 `STEPS_PER_EP` 기본값은 100이다. 207/208의 `KL_WEIGHT=0.01`, 208의 `IMG_W=128`, `IMG_H=128`도 학습 파일에서 설정한다.

짧은 실행 점검에서는 `NUM_EPISODES=10`, `EPOCHS=2`, `BATCH_SIZE=8`처럼 학습량을 줄인다. 구조 변경은 학습·추론을 함께 맞추고 재학습해야 한다. 204~207은 `STEPS_PER_EP >= HISTORY + CHUNK_SIZE - 1`, 208은 `STEPS_PER_EP >= CHUNK_SIZE`여야 완전한 표본을 만들 수 있다. 체크포인트는 코드 폴더의 `MODEL_PATH`에 저장하며 같은 이름은 덮어쓴다.

207 목표·시작 조건과 206~208의 temporal ensembling 계수는 해당 추론 파일 상수와 초기화 함수를 확인해 바꾼다. 학습량만 줄이는 것과 목표 분포·모델 구조를 바꾸는 것을 구분한다.

### 분포 그림을 함께 보는 실행

```bash
python 205_1_Action_Various_start.py
```

이 파일은 205 학습 흐름에 분포 그림을 추가하고 `205_1_synthetic_data_distribution.png`를 저장한다. 별도 모델 파일이 아니라 205와 같은 checkpoint를 사용한다. 그림 창·평가 창을 순서대로 닫는다. 저장 성공이나 loss 감소를 작업 성공률로 해석하지 않는다.


---

# 7장

## 실행 안내: 포트 설정부터 칼리브레이션·텔레오퍼레이션까지

이 장은 실제 모터에 명령을 보내는 단독 스크립트이다. `--port`, `--baudrate`, `--motor-id`, `--help`를 받는 CLI는 없다. 파일 하단의 `if __name__ == "__main__":` 호출부를 수정한 뒤 실행한다. **함수 정의의 기본값보다 실제 호출부의 인자가 우선**한다.

### 1. 환경과 포트 확인

```bash
conda activate lerobot
cd /home/robo/so101_imitation_learning/07_so101_control_calibration
python -c "import serial; print(serial.__version__)"
python -m serial.tools.list_ports -v
```

마지막 명령은 pyserial이 제공하는 포트 목록 조회 도구이며 모터를 움직이지 않는다. `/dev/ttyUSB0`와 `/dev/ttyACM0`은 예시 이름이므로 리더·팔로워가 실제로 연결된 포트를 확인한다. 드라이버 기본 baudrate는 1,000,000이다. 포트 열기에서 실패하면 존재 여부·다른 프로세스의 사용 여부·현재 사용자의 장치 접근 권한을 먼저 확인한다.

### 2. 실행 전에 바꾸는 값

| 파일 | 실행 시 실제 포트·설정 | 바꿀 위치 |
|---|---|---|
| `201_motor_id_setting.py` | USB0, ID는 터미널 입력 | 마지막 `set_motor_id()`를 `set_motor_id(port="실제 포트")`로 변경 |
| `202_feetech_motor_con.py` | USB0, ID 1, 목표 raw 1000 | 하단 `MiniFeetechDriver`, `MOTOR_ID`, `set_position` |
| `203_motor_1_calibration_leader.py` | USB0, ID 1 | 하단 `calibrate_single_motor_shoulder_pan(...)` |
| `203_motor_1_calibration_follow.py` | USB0, **ID 6** | 같은 함수 호출의 `port`, `motor_id`, `out_json_path` |
| `204_motor_1_teleoperate.py` | ACM0→ACM1, `hz=50` | 하단 `teleop_motor1_shoulder_pan(...)` |
| `205_all_motor_calibration_leader.py` | USB0 | 하단 `FullArmCalibrator(port=...)` |
| `205_all_motor_calibration_follower.py` | USB1 | 하단 `FullArmCalibrator(port=...)` |
| `206_motor_all_teleoperate.py` | ACM0→ACM1, `config_dir="./"` | 하단 `Teleoperator(...)` |
| `position_test.py` | USB0, ID 1 | 하단 `monitor_motor_position(...)` |

203의 팔로워 파일은 함수 기본 ID가 1이어도 하단 호출에서 **ID 6**을 전달한다. shoulder_pan 단일 관절 실습이라면 실제 shoulder_pan ID(기본 배치에서는 1)에 맞춰 `motor_id`를 먼저 바꾼다. 기본 포트도 USB0이다. 파일명이 follower라고 자동으로 USB1을 선택하지 않는다. 전체 팔 텔레옵 206은 고정 Hz 옵션 없이 반복하며 측정 Hz를 출력한다. 단일 관절 204의 `hz`와 혼동하지 않는다.

### 3. 실행 단계

ID 변경이 필요한 모터는 **하나씩 연결**하고 다음을 실행한다. 터미널에서 현재 ID와 새 ID를 입력한다. 여러 모터를 연결한 채 방송 ID 254로 바꾸지 않는다.

```bash
python 201_motor_id_setting.py
```

위치 읽기를 먼저 확인하려면 다음을 실행한다. 해당 모터의 토크를 끄고 현재 raw 위치를 표시하며 `Ctrl+C`로 종료한다.

```bash
python position_test.py
```

단일 관절 경로를 학습하려면 포트와 저장 파일명을 맞춘 다음 순서대로 실행한다. 각 calibration 프로그램은 중앙·최소·최대 자세에서 Enter 입력을 받는다.

```bash
python 203_motor_1_calibration_leader.py
python 203_motor_1_calibration_follow.py
python 204_motor_1_teleoperate.py
```

결과는 `shoulder_pan_calibration_leader.json`, `shoulder_pan_calibration_follower.json`이다. 한 관절 텔레옵은 이 두 파일을 읽고 `Ctrl+C`로 종료한다.

전체 팔 경로는 다음과 같다.

```bash
python 205_all_motor_calibration_leader.py
python 205_all_motor_calibration_follower.py
python 206_motor_all_teleoperate.py
```

결과는 현재 폴더의 `full_arm_calibration_leader.json`, `full_arm_calibration_follower.json`이다. `config_dir`은 현재 구현이 문자열로 파일명을 붙이므로 `./`처럼 끝에 `/`가 있는 디렉터리를 지정한다. 206은 `Ctrl+C` 종료 처리에서 팔로워 토크를 끈다.

`202_feetech_motor_con.py`는 선택적인 저수준 명령 실습이며, 실행하면 raw 1000을 전송한다. 실제 관절의 가동 범위를 확인하고 목표를 정한 뒤 실행한다. 205의 JSON은 8장으로 전달하지만, 10장은 별도 중심 정렬 calibration을 사용한다.


---

# 8장

## 실행 안내: 실제 로봇 데이터 수집→학습→추론

이 장의 커스텀 스크립트에는 CLI parser가 없다. `--repo-id`, `--epochs`, `--camera`, `--port`, `--checkpoint`, `--help`는 지원하지 않는다. 아래 소스 설정을 맞추고 인자 없이 실행한다. 9~10장의 공식 ACT CLI 옵션을 이 파일에 붙여도 적용되지 않는다.

### 1. 데이터 수집 준비

```bash
conda activate lerobot
cd /home/robo/so101_imitation_learning/08_so101_imitation_learning/101_data_collect
python -c "import torch, cv2, serial; from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('imports OK')"
python -m serial.tools.list_ports -v
```

7장에서 해당 리더·팔로워로 측정한 최신 두 calibration JSON을 이 작업 폴더에 둔다. 같은 이름의 예제 파일이 있더라도 자신의 로봇에 맞는지 확인한다. 최신 결과를 복사하는 명령은 다음과 같다.

```bash
cp ../../07_so101_control_calibration/full_arm_calibration_leader.json .
cp ../../07_so101_control_calibration/full_arm_calibration_follower.json .
```

수집 파일 하단의 `TeleopRecorder(leader_port=..., follower_port=..., repo_id=...)`를 설정한다. 포트 기본 호출은 USB0→USB1이고 repo-id 기본값은 `my_robot_task`이다. **현재 수집기는 시작할 때 같은 repo-id의 `~/.cache/huggingface/lerobot/<repo_id>`를 삭제한다.** 이어 수집 옵션이 없으므로 기존 결과를 보존하려면 새 repo-id로 바꾸거나 먼저 별도로 보관한다.

| 설정 | 실제 위치와 기본값 |
|---|---|
| 리더·팔로워 포트 | 하단 `TeleopRecorder(...)` 호출 |
| repo-id | 생성자 기본 `my_robot_task`; 호출 시 명시 가능 |
| 카메라 | `run()`의 `cv2.VideoCapture(0)` |
| 데이터 fps 메타데이터 | 생성자의 `self.fps=30` |
| 목표 EMA 계수 | 생성자의 `self.alpha=0.3` |

`fps=30`을 지정한 것만으로 실제 수집 루프가 정확히 30Hz로 동기화되는 것은 아니다. CLI의 `--fps`도 없다. 본문의 시간축 설명을 함께 확인한다.

```bash
python 101_data_collect_using_teleop.py
```

이미지 창에 포커스를 두고 `r`로 녹화를 시작하고, `s`로 에피소드를 저장하고, `q`로 종료한다. 저장하지 않은 에피소드가 `q`로 자동 저장된다고 가정하지 않는다.

### 2. 저장 데이터 확인

```bash
python 102_collected_data_check.py
```

검사 파일 하단의 `repo_id`도 수집과 같게 바꾼다. 기본은 0번 에피소드 재생이고 `q`로 종료한다. `delay_ms=100` 인자가 있지만 현재 구현의 표시 루프는 `cv2.waitKey(1)`을 사용하므로 인자만 바꿔 재생 간격이 바뀌지는 않는다.

수집 repo-id를 바꾸면 검사 파일, MLP의 `train(repo_id=...)`, ACT의 `train_act(repo_id=...)`도 같은 값으로 맞춘다. 각 학습은 repo-id로 LeRobot 캐시의 데이터를 읽으며 별도 `--dataset-root` 옵션이 없다.

### 3. MLP 학습과 추론

```bash
cd /home/robo/so101_imitation_learning/08_so101_imitation_learning/102_MLP_imitation_learning
python 101_MLP_training.py
```

`train()` 내부 기본값은 batch 64, worker 4, AdamW LR 0.0001, `range(50)`의 50 epoch이다. 학습량은 이 부분을 수정한다. CUDA 가능 여부를 자동 판단하며 마지막에 현재 폴더의 `robot_model_final.pth`를 저장한다.

MLP 추론 폴더에는 `motor_control.py`가 없으므로 데이터 수집 폴더를 검색 경로로 연결한다. 먼저 최신 팔로워 calibration을 준비한 뒤 실행한다.

```bash
cp ../../07_so101_control_calibration/full_arm_calibration_follower.json .
PYTHONPATH=../101_data_collect python 102_MLP_inference.py
```

`PYTHONPATH`는 해당 명령의 환경변수이며 CLI 인자가 아니다. 추론 파일 하단의 `RobotInference(follower_port="/dev/ttyUSB1", model_path="robot_model_final.pth")`에서 실제 포트와 모델 경로를 맞춘다. 생성자의 `model_path` 기본값은 위 파일명이다. 카메라 0과 현재 폴더의 팔로워 JSON을 사용하며 창에서 `q`로 종료한다.

### 4. 커스텀 ACT 학습과 추론

```bash
cd /home/robo/so101_imitation_learning/08_so101_imitation_learning/103_ACT_imitation_learning
python 101_ACT_training.py
```

하단 `train_act(repo_id="my_robot_task")`를 수집 값에 맞춘다. `train_act()`의 기본값은 `chunk_size=50`, loader batch 64, AdamW LR 0.0001, 100 epoch이다. 결과는 현재 폴더의 `act_robot_model.pth`이다. chunk size를 바꾸면 추론의 모델 생성도 함께 바꾸고 재학습한다.

```bash
cp ../../07_so101_control_calibration/full_arm_calibration_follower.json .
python 102_ACT_inference.py
```

추론 상단의 `load_model("act_robot_model.pth", device)`, `MiniFeetechDriver(port="/dev/ttyUSB1")`, `cv2.VideoCapture(0)`을 실제 장치에 맞춘다. 이 폴더에는 자체 `motor_control.py`가 있다. 실제 팔로워 명령이 전송되며 `q`로 종료한다.

### 실행 전에 확인할 현재 구현의 범위

이 안내는 현재 프로그램을 실행하는 방법이다. MLP/ACT 학습의 이미지 `/255.0` 중복 가능성과 커스텀 ACT의 미래 행동 대신 현재 행동을 반복하는 fallback 문제는 본문의 8.6·8.9에서 설명한다. 데이터 배치의 이미지 범위와 action 시간축을 먼저 확인해야 한다. ACT 추론은 출력 범위 clamp나 토크 활성화 설정을 갖춘 완성형 하드웨어 실행기로 보장되지 않으므로 가중치 저장 성공만으로 실제 동작 준비가 끝났다고 판단하지 않는다.

정식 LeRobot ACT의 미래 timestamp와 전처리 저장을 사용하는 CLI 흐름은 9~10장에서 별도로 다룬다. 이번 원고 보강에서는 실제 모터 구동이나 재학습을 수행하지 않았다.


---

# 9장

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


---

# 10장

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
