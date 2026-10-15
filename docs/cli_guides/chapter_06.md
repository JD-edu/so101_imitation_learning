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
