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
