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
