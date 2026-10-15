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
