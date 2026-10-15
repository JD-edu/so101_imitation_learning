# 로보틱스를 위한 이미테이션 러닝 — 4~10장 통합 원고

4·5·6장을 독립 코드 폴더로 분리한 최신 게시 원고이다. [실행 안내](chapter_cli_guide.md)와 [장별 소스 대응표](chapter_source_numbering.md)를 함께 확인한다.

# 04장. RNN과 Transformer: 로봇의 상태 이력 읽기

> **4장 소스코드:** `04_sequence_models/`. 폴더 앞번호는 책의 장 번호입니다. 4·5·6장은 각각 독립된 폴더를 사용합니다. 내부 파일 번호는 본문의 실행 파일 표에 표시한 예제 ID입니다.

<!-- CLI_GUIDE_START -->
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
<!-- CLI_GUIDE_END -->

---

이 장은 `04_sequence_models/sequence_models.py`의 RNN·Transformer 구조를 읽고 `101_model_shapes.py`로 shape를 확인한다. 5장의 학습 모델과 같은 구조를 독립된 설명 모듈로 제공한다. 3장에서 배운 관측–명령–물리 진행을 바탕으로, 하나의 상태와 상태 이력을 입력으로 쓰는 차이를 이해한다. 5장에서는 데이터 생성부터 학습·폐루프 평가까지 실행하고, 6장에서는 action chunking과 CVAE·시각 입력으로 확장한다.

## 4.1 분류에서 로봇 정책으로

2장의 분류기는 이미지에서 클래스별 로짓을 출력한다. 정책은 관측에서 로봇에 보낼 행동을 출력한다. 출력 크기가 같더라도 학습 목표와 단위는 다르다.

| 구분 | 입력 | 출력 | 대표 손실 |
|---|---|---|---|
| 이미지 분류 | 이미지 | 클래스별 로짓 | CrossEntropyLoss |
| 단일 상태 BC | 현재 상태 `s_t` | 행동 `a_t` | MSELoss |
| 이력 기반 BC | 최근 상태 묶음 | 현재 행동 `a_t` | MSELoss |
| Action chunking | 관측 또는 이력 | 현재부터 여러 행동 | 청크 재구성 손실 |

201~203의 상태는 현재 관절 위치 6개이고, 행동은 position actuator에 줄 목표 위치 6개이다. 행동을 다음 관측 `s_(t+1)`이나 직접 출력 토크라고 해석하지 않는다. 5장 `05_mujoco_bc/`의 `scene.xml`이 읽는 `so101_new_calib.xml`은 그리퍼를 포함한 여섯 관절 모두 hinge이므로 여섯 값 모두 rad이다. 그리퍼의 0.01이나 0.035도 미터 단위의 벌어진 폭이 아니다. 3장의 109 모델 역시 그리퍼가 hinge이다.

Behavior Cloning은 시연에서 관측과 행동의 대응을 회귀하는 지도학습이다. 5장의 시연은 사람이 조종해 수집한 파일이 아니라 코드가 보간 목표를 생성하고 MuJoCo에 적용해 만든 데이터이다. 이 구분은 무엇을 모방하는지 설명할 때 필요하다.

## 4.2 상태 하나와 상태 이력의 차이

현재 관절 위치만으로는 같은 위치를 지나가는 두 움직임의 방향을 구별하기 어렵다. 직전 위치들을 함께 보면 변화 방향과 속도를 추론하는 데 도움이 된다. 그러나 이력을 넣었다고 속도가 명시적으로 관측되거나 접촉·외력까지 완전히 식별되는 것은 아니다.

이 책의 202/203 예제는 `HISTORY=10`이다. 현재 관측을 포함하여 다음 이력을 만든다.

$$
X_t=[s_{t-9},s_{t-8},\ldots,s_t],\qquad \hat a_t=\pi_\theta(X_t)
$$

PyTorch의 `batch_first=True`에 맞춰 입력 shape는 `[B,10,6]`이다. `B`는 서로 다른 학습 표본의 수이고, 10은 이력 길이이며, 6은 각 상태의 성분 수이다. 출력 `[B,6]`은 표본마다 **현재 행동 하나**를 뜻한다. `[B,10,6]`이 입력이라고 행동도 열 개 나오는 것은 아니다.

```python
# 두 파일 P2PTrajectoryDataset의 핵심: 에피소드별로 수행
for t in range(history - 1, len(ep_s)):
    xs.append(ep_s[t - history + 1 : t + 1])
    ys.append(ep_a[t])
```

파이썬 슬라이스의 끝은 제외되므로 `t+1`까지 잘라야 `s_t`를 포함한다. 정답은 `a_t`이다. 다른 에피소드의 마지막 상태와 새 에피소드의 첫 상태를 한 이력으로 잇지 않는다. 시연 생성에서는 명령 적용 **전** 상태 `s_t`를 기록한 뒤 `ctrl=a_t`, `mj_step`을 수행한다. 이 시간 정렬이 틀리면 모델의 종류를 바꿔도 잘못된 대응을 학습한다.

에피소드 길이가 100이고 이력이 10이면 표본 수는 `100-10+1=91`이다. 현재 두 스크립트의 기본값인 100개 에피소드에서는 9,100개 표본이 만들어진다. 이웃 표본은 상태 대부분을 공유하므로 표본 단위 무작위 분할만으로 독립 평가가 보장되지는 않는다.

## 4.3 Vanilla RNN의 계산

RNN은 이력의 앞에서 뒤로 숨은 상태를 갱신한다. 입력 투영을 `e_i`, 숨은 상태를 `h_i`라고 하면 현재 코드의 흐름은 다음과 같다.

$$
e_i=\operatorname{ReLU}(W_p s_i+b_p)
$$

$$
h_i=\tanh(W_{ih}e_i+W_{hh}h_{i-1}+b_{ih}+b_{hh})
$$

$$
\hat a_t=W_o h_t+b_o
$$

4장의 `sequence_models.py` 및 5장의 `202_RNN_MUJOCO.py`에 있는 `VanillaRNN_BC`는 상태 6개를 Linear로 64차원에 투영하고 ReLU를 적용한다. `nn.RNN`은 기본 `tanh` 비선형성을 사용하며 은닉 크기 64, 1층, 단방향이다. **입력 투영의 ReLU와 RNN 내부의 tanh를 구분한다.**

```python
class VanillaRNN_BC(nn.Module):
    def __init__(self, dof=6, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.projector = nn.Linear(dof, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, dof)

    def forward(self, x):
        embedded = torch.relu(self.projector(x))
        _, hidden = self.rnn(embedded)
        return self.head(hidden[-1])
```

위 조각은 실제 모델의 핵심을 보여준다. `hidden`은 `[1,B,64]`이고 `hidden[-1]`은 마지막 층의 숨은 상태 `[B,64]`이다. 출력층을 거쳐 `[B,6]`이 된다. `hidden[-1]`의 첫 축은 시간축이 아니라 층·방향 축이라는 점에 주의한다.

현재 `forward`는 이전 호출의 hidden을 인자로 전달하지 않는다. 따라서 매 호출마다 기본 영 상태에서 최근 10개 상태를 다시 읽는다. 시뮬레이터 전체 실행 동안 무제한 기억을 누적하는 구현이 아니다. 추론 시작에는 같은 초기 상태를 10번 채운 버퍼를 사용하고 물리 스텝 후 새 상태를 넣는다.

## 4.4 역전파와 LSTM을 구분해서 이해하기

RNN 학습도 PyTorch의 자동미분을 사용한다. 이력 안에서 반복된 연산을 따라 역전파하므로 긴 구간에서 기울기가 작아지거나 커질 수 있다. 이력이 길수록 항상 정확도가 높아지는 것은 아니며 학습 난이도와 비용도 바뀐다.

LSTM과 GRU는 정보를 유지하거나 갱신할 게이트를 추가한 다른 순환 모델이다. LSTM에는 hidden과 cell state가 있고, Vanilla RNN의 hidden 하나와 같지 않다. 이 절은 모델 간 차이를 설명하기 위한 개념 소개이며 **202의 구현은 LSTM/GRU가 아니라 `nn.RNN`**이다. 클래스를 교체하려면 반환값, hidden 처리와 학습·추론 구조를 함께 바꾸고 재학습해야 한다.

이 책에서 다음 단계로 Transformer를 보는 이유는 같은 이력의 요소를 attention으로 결합하는 방식을 비교하기 위해서다. RNN보다 항상 우수하다는 성능 결론은 이 예제만으로 내리지 않는다.

## 4.5 Self-attention의 역할

RNN이 순서대로 숨은 상태를 갱신한다면, self-attention은 이력 안의 표현들이 서로 어느 정도를 참조할지 계산한다. 한 head의 핵심 계산은 다음과 같다.

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

softmax는 각 query가 참조할 key들의 축으로 적용된다. 내적 점수로 가중치를 만들고 value의 가중합을 얻는다. 단순히 가장 큰 점수의 상태 하나를 고르는 연산이 아니다. Multi-head attention은 서로 다른 투영으로 얻은 표현을 결합한다.

4장의 `sequence_models.py`와 5장의 `203_Transformer_MUJOCO.py`에 있는 모델은 encoder이다. decoder나 교차 attention, CVAE 잠재변수가 없다. Transformer를 썼다는 이유만으로 이 모델을 완성된 ACT라고 부르지 않는다.

## 4.6 실제 Transformer encoder의 shape

`TransformerEncoderBC`의 기본 구조는 다음과 같다.

| 단계 | 실제 구현 | 출력 shape |
|---|---|---|
| 상태 투영 | Linear(6,64) | `[B,10,64]` |
| 위치 표현 더하기 | 학습 파라미터 `[1,10,64]` | `[B,10,64]` |
| encoder | 2층, head 4개, FFN 128, dropout 0.1 | `[B,10,64]` |
| 마지막 위치 선택 | `x[:, -1]` | `[B,64]` |
| 행동 head | Linear(64,6) | `[B,6]` |

```python
x = self.state_projector(state_seq) + self.pos_embedding
x = self.transformer_encoder(x)
return self.head(x[:, -1])
```

현재 위치 표현은 `nn.Parameter(torch.zeros(1, history, d_model))`로 선언하여 학습한다. 사인·코사인 표를 계산하는 sinusoidal positional encoding이 아니다. 위치 표현은 이력의 슬롯을 구분하게 해 주며 초기 영 값이 학습으로 바뀐다. 일반적인 생성형 언어 모델의 토큰 임베딩과 달리 여기서는 연속 관절값을 Linear로 투영한다.

이 모델에는 causal mask가 없다. 입력은 모두 행동 결정 시점까지 관측한 `s_(t-9)..s_t`이며 현재 행동 하나만 감독하므로 이 구성 자체가 미래 관측을 사용하는 것은 아니다. 이력 각 위치마다 그 시점의 행동을 예측하도록 목적을 바꾸면 뒤쪽 관측을 참조하는 문제를 다시 검토해야 한다.

위치 파라미터의 길이는 기본 10에 고정되어 있다. 추론에서 이력 길이만 마음대로 늘리면 덧셈 shape 또는 체크포인트 shape가 맞지 않는다. 모델과 데이터셋, 추론 버퍼의 설정을 함께 맞춘다. 평가에서는 `model.eval()`로 dropout을 끄고 gradient 기록을 끈다.

## 4.7 GUI·학습 없이 실제 모델 읽기

다음은 책에 추가한 구조 확인용 예제이다. 저장소의 실제 클래스를 import하여 작은 영 입력의 shape만 확인한다. `main()`을 호출하지 않으므로 시연 수집·학습·GUI를 시작하거나 가중치 파일을 저장하지 않는다. 영 입력과 무작위 초기 모델의 출력은 유효한 로봇 제어 결과가 아니다.

```bash
cd /home/robo/so101_imitation_learning
python - <<'PY'
from pathlib import Path
import importlib.util
import torch

root = Path('04_sequence_models')
cases = [
    ('sequence_models.py', 'VanillaRNN_BC'),
    ('sequence_models.py', 'TransformerEncoderBC'),
]
for index, (filename, classname) in enumerate(cases):
    spec = importlib.util.spec_from_file_location(f'book_shape_{index}', root / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = getattr(module, classname)().eval()
    with torch.no_grad():
        result = model(torch.zeros(2, 10, 6))
    print(filename, tuple(result.shape))
PY
```

두 줄 모두 `(2, 6)`이 나오면 해당 환경에서 실제 클래스의 입력·출력 연결이 확인된 것이다. 이 구조 확인에는 PyTorch만 필요하다. 모델 구조가 같아도 임의 초기 출력은 학습된 정책의 행동이 아니다.

## 4.8 단일 행동에서 action chunk까지

RNN/Transformer의 선택과 출력 시간 범위의 선택은 서로 다른 축이다. 203은 Transformer여도 현재 행동 하나만 낸다. 204부터는 출력 범위를 청크로 확장한다.

$$
\hat A_t=[\hat a_t,\hat a_{t+1},\ldots,\hat a_{t+K-1}]
$$

이때 출력 shape는 `[B,K,6]`이다. **상태 이력 길이 H와 행동 청크 길이 K는 다른 값**이며 같은 시간축이라고 취급하지 않는다. H는 과거·현재 관측을 몇 개 읽는지, K는 현재부터 명령을 몇 개 예측하는지 나타낸다.

| 파일 계열 | 다음에 배우는 변화 |
|---|---|
| 201 | 현재 상태 → MLP → 현재 행동 |
| 202 | 상태 이력 → Vanilla RNN → 현재 행동 |
| 203 | 상태 이력 → Transformer encoder → 현재 행동 |
| 204~205 | 여러 행동 출력과 시작 분포 변경 |
| 206 | 매 스텝 재계획과 겹치는 예측 결합 |
| 207 | 목표 조건·CVAE 잠재변수 |
| 208 | 현재 영상과 관절을 함께 사용하는 시각 ACT |

201~206의 목표 입력이 없는 모델은 고정 목표 시연을 학습한다. 사용자가 추론할 때 임의 목표를 전달할 수 있는 정책으로 설명하지 않는다. 207은 현재 상태와 목표를 연결한 12차원 조건을 사용하는 별도 단계이다. 각 단계의 정확한 구조와 시간 정렬은 5~6장에서 실제 파일을 따라 확인한다.

## 4.9 확인 문제와 실습 연결

1. `[B,10,6]`에서 각 축은 무엇이며 현재 행동 하나의 출력 shape는 무엇인가?
2. `xs`의 마지막 상태가 `s_t`라면 왜 정답을 `a_t`와 맞추는가?
3. 202의 입력 ReLU와 RNN 내부 tanh는 어디에 적용되는가?
4. 202는 호출 사이의 hidden을 유지하는가, 최근 이력을 다시 읽는가?
5. 203의 위치 표현은 어떻게 만들어지며 학습되는가?
6. 203에 causal mask가 없다는 사실과 미래 관측 누설 여부를 어떻게 구분하는가?
7. Transformer encoder, action chunking, CVAE는 각각 무엇을 바꾸는가?

이제 [5장](https://wikidocs.net/439087)에서 동일한 모델에 실제 시연 데이터를 넣어 학습하고 MuJoCo 폐루프에서 평가한다. 손실이 작아지는 것과 스스로 생성한 상태에서 안정적으로 동작하는 것은 별도로 확인한다. 이후 [6장](https://wikidocs.net/439088)에서 청크 실행·재계획·시각 입력을 순서대로 다룬다.


---

# 05장. MuJoCo에서 Behavior Cloning, RNN, Transformer 배우기

> **5장 소스코드:** `05_mujoco_bc/`. 폴더 앞번호는 책의 장 번호입니다. 4·5·6장은 각각 독립된 폴더를 사용합니다. 내부 파일 번호는 본문의 실행 파일 표에 표시한 예제 ID입니다.

이 폴더에는 201~203의 학습·추론과 자체 XML·mesh가 있다. 204~208은 `06_mujoco_act/`로 분리했다. 기존 Python 파일과 모델·가중치의 내용은 유지했으며, 가중치는 각 장의 코드 폴더에서 읽고 저장한다. 두 장의 XML·mesh는 분리 시점에는 동일하지만 이후 편집은 각 폴더에 독립적으로 적용된다.

<!-- CLI_GUIDE_START -->
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
<!-- CLI_GUIDE_END -->

---

## 5.1 이 장에서 만드는 정책

이 장의 실습 폴더는 `05_mujoco_bc/`이다. SO-101 시뮬레이터에서 시작 자세와 목표 자세를 잇는 시연을 만들고, 그때 사용한 위치 명령을 신경망이 모방하게 한다. 201은 현재 상태 하나를 사용하는 MLP, 202는 짧은 상태 이력을 사용하는 RNN, 203은 같은 이력을 읽는 Transformer encoder이다.

Behavior Cloning(BC)은 시연의 관측–행동 쌍을 지도학습하는 **학습 방식**이다. MLP, RNN, Transformer는 정책을 표현하는 **모델 구조**이다. 따라서 RNN이나 Transformer로 학습해도 BC이다. 이 단계에는 보상을 최대화하는 강화학습 알고리즘이 없다.

이 장과 다음 장은 코드 오류를 수정한 v2 구현을 기준으로 한다. 기존 `.pt` 파일은 보관하고, 수정본은 이름 끝에 `_v2.pt`를 붙여 저장한다. 관측과 정답의 시점, 목표 분포 등이 바뀌었으므로 **각 학습 파일로 재학습한 뒤 대응 추론 파일을 실행한다. 기존 가중치의 이름만 바꾸지 않는다.**

## 5.2 실행 파일과 모델 파일

| 단계 | 학습 파일 | 추론 파일 | 저장 파일 |
|---|---|---|---|
| 201 MLP | `201_MLP_MUJOCO.py` | `201_MLP_MUJOCO_inference.py` | `so101_mlp_bc_p2p_v2.pt` |
| 202 RNN | `202_RNN_MUJOCO.py` | `202_RNN_MUJOCO_inference.py` | `so101_rnn_bc_p2p_v2.pt` |
| 203 Transformer | `203_Transformer_MUJOCO.py` | `203_Transformer_MUJOCO_infer.py` | `transformer_encoder_bc_v2.pt` |

```bash
cd /home/robo/so101_imitation_learning/05_mujoco_bc
python 201_MLP_MUJOCO.py
python 201_MLP_MUJOCO_inference.py
```

첫 프로그램에서 데이터 생성과 학습을 수행하고, 학습 후 시연 창을 닫은 다음 두 번째 프로그램을 실행한다. 이 패턴은 이후 예제에서도 같다. 학습 파일은 CUDA를 사용할 수 있으면 GPU를 선택하고, 시연 단계에서는 모델을 CPU로 옮긴다. 별도 추론 파일도 CPU에서 실행한다.

수정본은 Python 파일의 위치를 기준으로 `scene.xml`과 가중치 경로를 계산한다. 따라서 저장소 루트에서 `python 05_mujoco_bc/201_MLP_MUJOCO.py`로 실행해도 같은 모델을 읽고 같은 폴더에 가중치를 저장한다. 이 경로 변경은 5·6장 코드 폴더에 적용한 것이며 다른 장의 상대 경로 예제에 그대로 일반화하지 않는다.

이 스크립트들은 데이터 수집·학습·GLFW 시연을 함께 실행하는 교육용 프로그램이다. 별도의 `--train-only` 옵션은 없다. GUI 실행에는 디스플레이와 OpenGL 환경이 필요하다.

## 5.3 상태, 명령, 관절 단위를 먼저 맞춘다

`scene.xml`은 `so101_new_calib.xml`을 포함한다. 현재 모델은 고정 베이스의 6개 hinge 관절이며 `nq=nv=nu=6`이다. 관절과 명령 배열의 순서는 다음과 같다.

```text
shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper
```

**이 모델의 그리퍼도 회전 관절이다. 여섯 값 모두 rad 단위이다.** 그리퍼의 `0.01`, `0.035`를 미터 단위의 벌어진 폭으로 해석하지 않는다. 예를 들어 `(rand - 0.5) * 0.08`은 ±0.04 rad, 약 ±2.29° 범위의 잡음이다. ±4 cm를 뜻하지 않는다.

현재 관절 위치를 $q_t$, 정책에 들어가는 관측을 $o_t$, actuator에 보낼 목표 위치를 $a_t$라고 정의한다. position actuator에 연결된 `data.ctrl`은 목표 위치 명령을 받는다. 전류나 토크 배열이 아니다.

$$
o_t=q_t,\qquad \hat a_t=\pi_\theta(o_t)
$$

목표 명령 $a_t$를 적용한 뒤 관측되는 실제 위치 $q_{t+1}$는 동역학, 접촉, 힘 제한 등의 영향을 받는다. 둘이 같다고 가정하면 안 된다. 정책은 **다음 실제 관절 위치 자체가 아니라 현재 적용할 위치 명령**을 학습한다.

### 5.3.1 관측–행동의 정확한 시간 순서

모든 데이터 생성기는 다음 순서를 사용한다.

실제 핵심 실행문은 다음 세 줄이다.

```python
current_state = data.qpos[:DOF].copy()
data.ctrl[:DOF] = target_ctrl.numpy()
mujoco.mj_step(model, data)
```

수집 배열의 `states[t]`와 `actions[t]`는 같은 시점의 관측과 명령이다. 따라서 단일 행동은 `states[t] → actions[t]`, 청크는 현재 시점부터 `actions[t:t+K]`로 구성한다. 기존 코드의 `actions[t+1]`은 다음 시점의 명령을 한 스텝 일찍 실행하게 하는 불일치였으며 수정했다. 새 표기에서 `a_t`는 $t$에서 $t+1$로 진행할 때 적용하는 행동이다.

## 5.4 합성 시연을 만드는 방법

기본 시작·목표 벡터는 다음과 같다.

```python
q_start = [-0.6, -0.4,  0.5, -0.3,  0.2, 0.010]
q_goal  = [ 0.7,  0.5, -0.4,  0.6, -0.5, 0.035]
```

각 에피소드를 초기화하고, 100개 시점에 대해 다음 보간식을 사용한다.

$$
\tau=\frac{t}{T-1},\quad s(\tau)=3\tau^2-2\tau^3
$$

$$
a_t=(1-s(\tau))q_{start}+s(\tau)q_{goal}+\epsilon_t
$$

이 식은 **3차 smoothstep**이다. 양 끝에서 1차 미분이 0이다. 5차 smootherstep인 $6\tau^5-15\tau^4+10\tau^3$ 또는 minimum-jerk 궤적과 같은 식이라고 설명하지 않는다. 매 스텝 독립적인 잡음 $\epsilon_t$를 더하면 최종 명령의 매끄러움은 보간식만 사용할 때와 달라진다.

| 예제 | 시작 자세 | 목표 자세 | 매 스텝 명령 잡음 |
|---|---|---|---|
| 201 | 고정 | 고정 | 관절별 ±0.003 rad |
| 202·203 | 기본 자세 주변 관절별 ±0.04 rad | 고정 | 관절별 ±0.0025 rad |

202·203의 수정 전 코드는 목표도 무작위로 바꾸면서 정책에는 목표를 알려주지 않았다. 수정본은 목표 입력이 없는 모델의 과제를 고정 목표 도달로 정의한다. 여러 목표를 선택하는 문제는 207에서 목표 벡터를 명시적으로 입력하면서 다룬다. 시작점 다양성과 목표 조건의 제공은 서로 다른 설계 항목이다.

시연은 사람이 조종해 기록한 데이터가 아니라 보간기로 생성한 합성 데이터이다. 작은 잡음을 추가했다고 여러 작업 방식이나 물체 조작 기술이 자동으로 포함되는 것은 아니다.

## 5.5 MLP BC: 현재 관측에서 행동 하나로

`BehaviorCloningMLP`의 구조는 다음과 같다.

```text
[B, 6] → Linear(6,128) → ReLU
       → Linear(128,128) → ReLU
       → Linear(128,6) → [B,6]
```

출력은 여섯 관절의 연속적인 목표 위치이므로 softmax를 붙이지 않는다. MSE 손실은 예측 명령과 시연 명령의 제곱 오차를 평균한다.

$$
\mathcal L_{MSE}=\frac{1}{BD}\sum_{b=1}^{B}\sum_{j=1}^{D}
(\hat a_{b,j}-a_{b,j})^2
$$

201의 `generate_p2p_episodes`는 각 에피소드의 관측과 같은 인덱스의 명령을 연결한다. 100에피소드 × 100스텝이므로 총 **10,000쌍**이다. 수정 전의 9,900쌍과 다르다. 앞 8,000개와 뒤 2,000개를 학습·평가로 나누며, 이 설정에서는 정확히 80개/20개 에피소드 경계와 일치한다.

201의 시작 자세는 매번 같다. 별도 추론의 다른 시작 자세에서 움직임이 나쁘더라도 이를 모든 MLP의 한계로 단정하지 않는다. 학습 분포, 상태에 포함된 정보, 시연의 동역학적 품질이 함께 영향을 준다.

## 5.6 RNN BC: 최근 상태 이력을 입력한다

`202_RNN_MUJOCO.py`의 `P2PTrajectoryDataset`은 길이 `HISTORY=10`의 상태 이력을 만든다.

$$
X_t=[q_{t-9},\ldots,q_t]\in\mathbb R^{10\times6},\qquad y_t=a_t
$$

실제 슬라이싱은 다음과 같다.

```python
for t in range(history - 1, len(ep_s)):
    xs.append(ep_s[t - history + 1:t + 1])
    ys.append(ep_a[t])
```

100스텝 에피소드에서 `t=9..99`를 사용하므로 에피소드당 91개이다. 학습은 80 × 91 = **7,280개**, 평가는 20 × 91 = **1,820개**이다. 학습 전에 에피소드를 나누므로 서로 다른 에피소드의 상태를 하나의 이력으로 연결하지 않는다.

`VanillaRNN_BC`는 다음 구조를 사용한다.

```text
[B,10,6] → Linear(6,64) → ReLU
         → nn.RNN(input_size=64, hidden_size=64)
         → 마지막 hidden [B,64] → Linear(64,6)
```

최근 자세의 변화는 이동 방향 등을 추정할 단서를 준다. 하지만 입력에 실제 `qvel`이 추가되는 것은 아니다. 이 구현은 매 호출마다 최근 10개 상태를 다시 읽으며, 숨은 상태를 에피소드 전체에 걸쳐 유지하는 방식이 아니다.

```bash
python 202_RNN_MUJOCO.py
python 202_RNN_MUJOCO_inference.py
```

추론 초기에는 관측 이력이 없으므로 초기 자세를 10번 복사해 버퍼를 만든다. 한 스텝 실행할 때마다 새 `qpos`를 추가하고 가장 오래된 상태를 제거한다. 학습 Dataset은 에피소드 앞부분을 패딩하지 않으므로 이러한 초기 반복 버퍼와 학습 윈도우 사이에는 차이가 있다.

## 5.7 Transformer encoder BC: 이력 안의 관계를 읽는다

203은 RNN과 같은 `[B,10,6] → [B,6]` 입출력 계약을 사용한다. `TransformerEncoderBC`는 상태를 64차원으로 투영하고 학습 가능한 위치 임베딩을 더한다. attention head는 4개, encoder는 2층, feed-forward 차원은 128, dropout은 0.1이다.

```text
상태 이력 [B,10,6]
→ 상태 투영 + 위치 임베딩 [B,10,64]
→ Transformer encoder 2층
→ 마지막 시점 표현 [B,64]
→ 선형 출력 [B,6]
```

이력 안의 모든 값은 이미 관측한 과거와 현재이다. 이 구현은 미래 정답 관측을 입력하지 않으며, encoder에 별도의 causal mask를 넣지 않는다. 위치 정보는 고정 사인파가 아니라 학습 가능한 파라미터이다.

```bash
python 203_Transformer_MUJOCO.py
python 203_Transformer_MUJOCO_infer.py
```

Transformer를 사용해도 출력은 여전히 행동 하나이다. 행동 청크와 CVAE, 시각 입력은 다음 장에서 단계적으로 추가한다. RNN보다 반드시 성공률이 높다는 결론은 이 코드만으로 나오지 않는다.

## 5.8 학습 설정과 시간축을 확인한다

| 예제 | 에피소드 × 스텝 | 학습/평가 샘플 | batch | epoch | learning rate | 손실 |
|---|---|---|---|---|---|---|
| 201 | 100 × 100 | 8,000 / 2,000 | 64 | 40 | 0.001 | MSE |
| 202 | 100 × 100 | 7,280 / 1,820 | 64 | 35 | 0.001 | MSE |
| 203 | 100 × 100 | 7,280 / 1,820 | 64 | 40 | 0.001 | MSE |

모델을 비교할 때 학습 데이터와 추론 시작 조건도 함께 기록한다. 이 파일들은 모든 변수를 통제한 벤치마크가 아니다. 평가 loss는 저장된 시연 명령에 대한 오차이며, 목표 도달 성공률이 아니다.

현재 XML을 읽은 `model.opt.timestep`은 **0.002초**이다. 수집과 추론은 명령마다 `mj_step`을 한 번 호출한다. 따라서 100스텝의 물리 시간은 **0.2초**이다. 실제 관절이 그 시간 안에 명령 목표까지 도달했는지 별도로 측정해야 한다. 시작과 목표가 크게 떨어진 궤적의 끝 명령이 목표와 같다고 해서 로봇도 목표에 도달한 것은 아니다.

GLFW 창은 `swap_interval(1)`로 화면 갱신에 동기화하고, 네트워크 추론과 렌더링에도 시간이 든다. 화면에서 100스텝을 보는 벽시계 시간과 물리 시간은 다를 수 있다. 이 코드를 실제 500 Hz 제어 성능을 검증한 예제로 설명하지 않는다. 제어 주기를 바꾸려면 데이터 수집과 추론 양쪽의 물리 진행 횟수 및 청크의 시간 범위를 함께 맞춘 뒤 재학습해야 한다.

## 5.9 폐루프 평가와 확인 문제

추론은 `관측 → 정책 → 위치 명령 → 물리 진행 → 새 관측`을 반복한다. 작은 행동 오차도 다음 관측을 바꾸므로, 시연 데이터의 평가 오차가 작아도 폐루프에서는 시연 분포를 벗어날 수 있다.

실험 기록에는 가중치 파일, 시작 자세, 목표 자세, 물리 timestep, 총 실행 스텝, 최종 관절 오차와 관절별 궤적을 남긴다. GUI에서 움직였다는 사실과 과제를 완수했다는 사실을 구분한다.

1. `actions[t]`와 `qpos[t+1]`가 서로 다른 이유는 무엇인가?
2. 수정본 MLP의 데이터가 10,000쌍이고 RNN은 9,100윈도우인 이유는 무엇인가?
3. 시작점과 목표점을 모두 바꾸려면 정책 입력에서 어떤 정보를 확인해야 하는가?
4. 그리퍼의 `0.035`를 3.5 cm라고 해석하면 왜 잘못되는가?
5. 시뮬레이션 100스텝과 화면에서 관찰한 시간이 같지 않을 수 있는 이유는 무엇인가?

---

# 06장. Action Chunking에서 시각 기반 ACT까지

> **6장 소스코드:** `06_mujoco_act/`. 폴더 앞번호는 책의 장 번호입니다. 4·5·6장은 각각 독립된 폴더를 사용합니다. 내부 파일 번호는 본문의 실행 파일 표에 표시한 예제 ID입니다.

이 폴더에는 204~208의 학습·추론, 205 분포 그림 예제와 자체 XML·mesh가 있다. 201~203은 `05_mujoco_bc/`에 있다. 기존 Python 파일과 모델·가중치의 내용은 유지했으며, 가중치는 각 장의 코드 폴더에서 읽고 저장한다. 두 장의 XML·mesh는 분리 시점에는 동일하지만 이후 편집은 각 폴더에 독립적으로 적용된다.

<!-- CLI_GUIDE_START -->
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
<!-- CLI_GUIDE_END -->

---

## 6.1 단계별로 무엇이 추가되는가

이 장도 `06_mujoco_act/`의 코드를 사용한다. 다음 표는 모델의 입력과 출력뿐 아니라, 예측한 행동을 어떻게 실행하는지도 구분한다.

| 번호 | 학습 파일 | 정책 입력 | 출력 | 실행 방식 |
|---|---|---|---|---|
| 204 | `204_Action_Chunking_MUJOCO.py` | 최근 관절 10프레임 | `[B,30,6]` | 30개 큐 소진 후 재추론 |
| 205 | `205_Action_Various_start.py` | 최근 관절 10프레임 | `[B,30,6]` | 시작 분포 확대, 큐 실행 |
| 206 | `206_ACT_RH_TE.py` | 최근 관절 10프레임 | `[B,30,6]` | 매 스텝 재추론 + 시간 앙상블 |
| 207 | `207_ACT_CAVE.py` | 최근 `[관절6, 목표6]` 10프레임 | `[B,30,6]` | 목표 조건 + CVAE + encoder/decoder + 시간 앙상블 |
| 208 | `208_ACT_Image.py` | 현재 RGB와 현재 관절6 | `[B,30,6]` | 시각 CVAE 정책 + 시간 앙상블 |

204~206은 encoder의 마지막 표현에서 청크를 만드는 모델이며 Transformer decoder나 CVAE가 없다. 207은 목표 조건을 가진 관절 기반 CVAE 정책이고, 208은 고정 목표 과제에 영상 입력을 추가한다. **208은 207의 12차원 목표 이력을 그대로 유지하는 누적 확장이 아니다.** 두 모델이 요구하는 관측을 구별해야 한다.

이들은 ACT의 구성 원리를 익히는 커스텀 교육 구현이다. 공식 ACT/LeRobot 구현과 동일한 네트워크, 입력 토큰, 학습 설정이라고 설명하지 않는다. 공식 ACT를 사용하는 파이프라인은 9~10장에서 다룬다.

## 6.2 청크 Dataset과 시간 인덱스

현재 관측을 읽은 시점 $t$에서 정책은 현재 실행할 행동을 포함한 다음 묶음을 출력한다.

$$
\hat A_t=[\hat a_{t|t},\hat a_{t+1|t},\ldots,\hat a_{t+K-1|t}]
$$

$\hat a_{s|t}$는 시점 $t$에 예측한 실행 시점 $s$의 행동이다. 기본값은 `HISTORY=10`, `CHUNK_SIZE=30`이다.

```python
last_t = len(ep_s) - chunk_size
for t in range(history - 1, last_t + 1):
    xs.append(ep_s[t - history + 1:t + 1])
    ys.append(ep_a[t:t + chunk_size])
```

길이 100인 에피소드에서 `t=9..70`을 사용할 수 있다. 에피소드당 샘플 수는 다음과 같다.

$$
N=T-H-K+2=100-10-30+2=62
$$

마지막 샘플은 관측 `states[61:71]`과 행동 `actions[70:100]`이다. Python 슬라이스 끝은 포함되지 않는다. 부족한 미래 행동을 복사해서 패딩하지 않으며, 이력과 청크가 에피소드 경계를 넘지 않는다. 204~206은 학습 4,960개, 평가 1,240개이다. 목표 조건 207도 같은 시간 인덱스를 사용한다.

`[B,30,6]`은 같은 행동 하나를 30번 반복한 텐서가 아니다. 서로 다른 30개 실행 시점에 대응하는 명령이다. 기본 물리 timestep에서는 청크 30개가 0.06초의 물리 진행에 해당한다.

## 6.3 204: Transformer 출력 범위를 늘린다

`ActionChunkTransformer`의 encoder는 203과 같이 64차원, 4 head, 2층이다. 마지막 시점 표현을 출력 head에서 128차원으로 바꾼 뒤 `30 × 6 = 180`개 값을 생성한다.

```text
[B,10,6] → 투영 + 위치 임베딩 → encoder
→ 마지막 표현 [B,64]
→ Linear(64,128) → ReLU → Linear(128,180)
→ reshape [B,30,6]
```

```bash
cd /home/robo/so101_imitation_learning/06_mujoco_act
python 204_Action_Chunking_MUJOCO.py
python 204_Action_Chunking_MUJOCO_infer.py
```

행동 큐가 비었을 때만 새 청크를 추론한다. 매 물리 스텝에는 큐에서 행동 하나를 꺼내 적용하고, 실제 관측 이력은 계속 갱신한다. 새 관측이 있어도 큐가 남아 있는 동안 정책을 다시 호출하지 않으므로 청크 내부의 실행은 개루프이다. 청크 사이에서 다시 관측을 반영한다.

긴 청크는 정책 호출 횟수를 줄이지만, 외란 이후 새 계획을 반영하기까지 기다리는 시간을 늘릴 수 있다. 예측 길이와 재추론 주기를 따로 다루어야 하는 이유이다.

## 6.4 205: 시작 분포를 확대한다

205의 첫 에피소드는 기본 시작 자세를 그대로 사용한다. 나머지는 첫 다섯 관절에 ±10°(약 ±0.1745 rad), 그리퍼에 ±0.005 rad의 시작 잡음을 준다. 목표는 `[0.7,0.5,-0.4,0.6,-0.5,0.035]`로 고정하고, 매 스텝 명령 잡음은 ±0.0015 rad이다.

```bash
python 205_Action_Various_start.py
python 205_Action_Various_start_infer.py
```

모델 구조를 복잡하게 만드는 것과 학습 데이터가 다루는 시작 영역을 넓히는 것은 서로 다른 변화이다. 학습 범위 안과 밖의 시작 자세를 나누어 비교해야 한다.

분포 그림이 필요하면 다음 보조 예제를 실행한다.

```bash
python 205_1_Action_Various_start.py
```

이 파일은 205 학습 파일을 동적으로 불러와 데이터 생성 함수에 그림 저장을 결합한 뒤 원래 `main()`을 실행한다. 이미지 파일만 보는 프로그램이 아니며 학습과 시연도 수행한다. `graspframe`의 순기구학 좌표로 시작 분포와 궤적을 그려 `205_1_synthetic_data_distribution.png`를 저장한다.

## 6.5 206: 매 스텝 다시 예측한다

206은 무작위 시작점과 고정 목표를 사용하며, 수집 중 매 스텝 명령 잡음은 없다. 네트워크는 여전히 관절 이력 기반 `ActionChunkTransformer`이다. Receding Horizon은 모델 계층이 아니라 **추론 결과를 실행하는 방법**의 변화이다.

```text
t:   현재 관측 → [a(t|t), a(t+1|t), …] 예측 → 현재 행동 결정
 t+1: 새 관측 → [a(t+1|t+1), a(t+2|t+1), …] 예측 → 현재 행동 결정
```

예측 길이는 30이지만, 물리 진행은 한 스텝이다. 새로운 관측을 매번 반영하므로 외란에 대한 반응 기회를 늘린다. 다만 정책이 외란 상황에서 적절한 행동을 학습했다는 보장은 별개이다.

```bash
python 206_ACT_RH_TE.py
python 206_ACT_RH_TE_infer.py
```

수정본에서는 학습 후 내장 시연도 별도 추론처럼 매 스텝 재추론과 시간 앙상블을 사용한다. 수정 전에는 내장 시연이 30개 큐를 소진하는 방식이라 이름과 실행 방식이 달랐다.

206 가중치는 `action_chunking_rh_te_v2.pt`이다. 204의 `action_chunking_transformer_v2.pt`와 구별하므로 서로 덮어쓰지 않는다.

## 6.6 Temporal Ensembling의 정확한 결합 방향

여러 시점에 만든 청크에는 같은 실행 시점의 예측이 중복된다. 현재 시점 $t$에 적용할 값은 이 중복 예측만 모아 평균한다.

$$
\bar a_t=\frac{\sum_{j\in J_t}\exp[-m(t-j)]\hat a_{t|j}}
{\sum_{j\in J_t}\exp[-m(t-j)]},\qquad m=0.05
$$

여기서 $j$는 청크를 생성한 시점이며 $0\le t-j<K$이다. 이 코드의 가중치는 청크 내부 거리 `i`에 대해 `exp(-0.05*i)`이다. 같은 실행 시점에 대해서는 **더 최근에 생성한 예측에 더 큰 가중치**가 붙는다. 예를 들어 새 예측의 가중치는 1이고, 직전 청크의 두 번째 값은 `exp(-0.05)`이다. 이는 이 저장소가 선택한 가중 방식이며 모든 ACT 구현의 가중치 방향이 동일하다는 뜻이 아니다.

버퍼에는 `buffer[t+i]`에 청크의 `i`번째 행동을 추가하고, `buffer[t]`만 꺼내 평균한 뒤 삭제한다. 다음 물리 스텝에서 `t`를 증가시킨다. 다른 실행 시점의 행동을 한꺼번에 평균하지 않는다. 에피소드 초기화 때는 상태 이력뿐 아니라 미래 예측 버퍼도 비운다.

별도 206·207·208 추론에서 마우스 왼쪽 클릭은 `qvel`에 무작위 값을 더한다. 물리적인 토크를 정해진 시간 동안 인가하는 코드는 아니다. 단위도 관절 속도 변화인 rad/s이다. 복원 여부는 관찰할 실험 결과이며 보장된 기능이 아니다.

## 6.7 207: 목표 조건과 CVAE를 도입한다

여러 목표 중 하나를 선택하려면 정책이 어떤 목표인지 알아야 한다. `207_ACT_CAVE.py`는 현재 관절6과 목표 관절6을 연결한다.

$$
o_t=[q_t,g]\in\mathbb R^{12},\quad X_t\in\mathbb R^{10\times12}
$$

시작과 목표의 첫 다섯 관절에는 각각 ±10°, 그리퍼에는 각각 ±0.005 rad의 잡음을 준다. 한 에피소드 동안 목표 $g$는 고정하며 모든 관측 프레임에 같은 목표 벡터를 붙인다. 명령에는 별도의 매 스텝 잡음을 더하지 않는다.

CVAE의 posterior encoder인 `GoalCVAEEncoder`는 이력 `[B,10,12]`와 정답 청크 `[B,30,6]`를 평탄화해 연결한다. 입력 300차원을 MLP `300 → 256 → 128`로 처리한 뒤 16차원의 평균과 로그 분산을 구한다. **이 posterior encoder 자체는 Transformer가 아니라 MLP이다.**

$$
z=\mu+\exp(\tfrac12\log\sigma^2)\odot\epsilon,
\quad\epsilon\sim\mathcal N(0,I)
$$

`GoalCVAE_ACT`의 정책 경로는 다음과 같다.

```text
[z] 토큰 1개 + [관절6, 목표6] 토큰 10개
→ 위치 임베딩을 더한 [B,11,64]
→ Transformer encoder 2층
→ memory [B,11,64]
→ 학습 가능한 action query [B,30,64] + Transformer decoder 2층
→ Linear(64,6) → [B,30,6]
```

**이 구현에서 `z`는 encoder 입력에 포함된다.** 따라서 encoder의 self-attention에도 관여한다. `z`를 encoder 바깥에서 decoder에만 전달한다고 설명하면 실제 `torch.cat([z_token, state_tokens], dim=1)`과 맞지 않는다. Transformer encoder와 CVAE posterior encoder라는 두 용어도 혼동하지 않는다.

## 6.8 CVAE의 학습과 추론을 나눈다

학습 중에는 정답 청크를 posterior encoder에 주고 $z$를 샘플링한다. 예측 청크의 L1 재구성 오차와 KL 손실을 함께 줄인다.

$$
\mathcal L=\operatorname{mean}|\hat A-A|
+0.01\left[-\tfrac12\operatorname{mean}(1+\log\sigma^2-\mu^2-\exp(\log\sigma^2))\right]
$$

이 구현의 KL은 배치와 잠재 차원에 대해 평균한다. 잠재 차원을 합산하는 다른 구현과 계수 0.01만 비교하면 손실의 상대 크기를 잘못 이해할 수 있다.

추론에서는 `model.eval()`을 호출하고 정답 청크 없이 입력 관측만 전달한다. 이 경로의 잠재 벡터는 0이며 반환값은 `(pred_actions, None, None)`이다. eval 모드에서는 dropout도 비활성화된다. 입력이 같을 때 결정론적으로 같은 출력을 만드는 경로이지, 추론마다 새로운 동작 스타일을 샘플링하는 경로가 아니다.

```bash
python 207_ACT_CAVE.py
python 207_ACT_CVAE_infer.py
```

새 추론 파일은 학습 파일에서 `GoalCVAE_ACT`를 불러와 같은 구조와 같은 `cvae_act_goal_transformer_v2.pt`를 사용한다. 리셋할 때 목표를 새로 만들고, 초기 이력과 매 스텝 새 관측에 모두 목표를 연결한다. 기존 공백 포함 파일명도 호환 실행용으로 남겼다.

```bash
python "207_ACT_CAVE_ inter.py"
```

이 명령도 이제 같은 새 추론으로 연결된다. 수정 전의 6차원 `CVAE_ACT` 및 `cvae_act_transformer.pt`와는 다르다. 학습용 `207_ACT_CAVE.py`의 기존 파일명은 유지했으며 개념의 올바른 약어는 CVAE이다.

CVAE가 누락된 목표를 알아내거나 외란 복원을 자동으로 보장하는 것은 아니다. 또한 규칙적인 단일 보간 경로는 행동 스타일의 다양성이 적다. 이 예제에서 CVAE를 학습할 수 있다는 사실과 CVAE가 성능을 높인다는 실험적 결론을 구별한다.

## 6.9 208: 현재 영상과 관절을 함께 사용한다

`208_ACT_Image.py`는 `front_cam`의 128×128 RGB 영상과 현재 관절 6개를 입력한다. 과거 영상이나 관절10프레임, 207의 목표 벡터를 입력하지 않는다. 목표 정보가 없는 상태로 랜덤 목표를 선택하던 문제를 수정해, 208은 무작위 시작점에서 **고정 목표 자세**로 가는 과제로 정의했다.

목표 지점의 붉은 마커는 사람이 보는 뷰어에 추가한 장식이다. 카메라 캡처는 장면을 다시 구성하며 목표 마커를 추가하지 않는다. 따라서 표시용 마커를 모델이 목표 입력으로 보고 있다고 설명하지 않는다. 여러 목표를 영상으로 지정하려면 학습 영상부터 목표 물체나 표시를 포함하도록 과제를 새로 설계해야 한다.

### 6.9.1 영상 수집과 전처리

카메라 함수는 RGB를 읽고 OpenGL 화면 좌표의 상하 방향을 뒤집어 `[3,128,128]`의 `uint8` 텐서로 반환한다. `mj_step` 뒤에는 `qpos`와 파생 위치가 갱신되는 시점 차이가 있으므로, 캡처 전에 `mj_forward`를 호출해 영상과 관절 관측을 맞춘다.

이미지는 Dataset까지 0~255 정수로 보관하고, 모델 안에서 float 변환과 정규화를 수행한다.

```text
uint8 RGB [0,255] → float RGB [0,1]
→ (RGB - mean) / std
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

학습과 추론은 같은 모델 및 캡처 함수를 공유한다. 이미 `[0,1]` float인 영상도 모델에 전달할 수 있지만, 외부에서 정규화를 한 번 더 수행하지 않는다. ImageNet 사전학습 가중치의 채널 정규화를 사용하되, 이 교육 코드는 입력을 128×128로 유지한다. torchvision 분류 평가용 256 resize·224 crop 전체를 그대로 적용한 파이프라인은 아니다. [torchvision ResNet18 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

200 × 100장의 원본 영상은 약 0.983 GB의 uint8 저장 공간을 사용한다. float32이면 약 3.932 GB이다. 수집 중 리스트와 `stack` 결과가 함께 존재하고 모델·활성값도 필요하므로 실제 최대 메모리는 원본 크기보다 크다. 디스크에서 스트리밍하는 Dataset은 아니다.

### 6.9.2 토큰과 네트워크 구조

| 입력 | 처리 | encoder 토큰 수 |
|---|---|---|
| RGB `[B,3,128,128]` | ResNet18 + global average pooling → 512 → 128 | 1 |
| 관절 `[B,6]` | Linear(6,128) | 1 |
| 잠재 변수 `[B,16]` | Linear(16,128) | 1 |

세 토큰은 `[z, joint, image]` 순서로 연결해 `[B,3,128]`이 된다. 위치 임베딩을 더한 뒤 4 head, 3층 encoder와 3층 decoder를 사용한다. decoder의 query는 `[B,30,128]`, 출력은 `[B,30,6]`이다.

이 시각 백본은 이미지 전체를 토큰 하나로 요약한다. ViT의 patch 토큰이나 공식 ACT의 공간 특징 토큰과 동일하다고 설명하지 않는다. 영상 하나를 읽는 모델이지만 CVAE posterior encoder는 이미지가 아닌 현재 관절과 정답 청크를 사용한다.

학습 시 ResNet18은 ImageNet 가중치로 초기화하고 전체 정책과 함께 갱신한다. 처음 실행할 때 가중치가 캐시에 없으면 다운로드가 필요하다. 별도 추론은 `pretrained_backbone=False`로 구조만 만든 뒤 저장된 정책의 backbone까지 로드하므로 불필요한 ImageNet 다운로드를 하지 않는다.

## 6.10 시각 ACT의 Dataset과 실행

영상 예제는 과거 이력을 요구하지 않으므로 `t=0..70`을 사용한다.

```python
sample = (images[t], joints[t], actions[t:t + 30])
```

에피소드당 71개이며 총 14,200개, 학습 160 × 71 = **11,360개**, 평가 40 × 71 = **2,840개**이다. 204~207의 이력 기반 샘플 수와 다르다.

```bash
python 208_ACT_Image.py
python 208_ACT_Image_infer.py
```

학습은 숨겨진 GLFW 창으로 OpenGL context를 준비하고 영상을 수집한 뒤 수행한다. 학습 후 같은 창을 표시하며 시연한다. 창을 숨긴다고 그래픽 context가 불필요해지는 것은 아니다. 추론은 최신 영상과 관절로 매번 청크를 예측하고 시간 앙상블로 현재 행동을 결정한다.

학습 손실은 207과 같은 L1 + 0.01 KL이며, 평가에서는 `z=0`의 L1 재구성 오차를 측정한다. ResNet18에 영상이 들어간다는 사실만으로 물체를 찾아 집는 기술을 배웠다고 볼 수 없다. 이 scene과 시연 생성기는 관절 P2P 과제이며 물체 집기 성공 판정은 없다.

## 6.11 기본 설정과 체크포인트 대응표

| 예제 | 에피소드 | 학습/평가 샘플 | epoch | batch | learning rate | 손실 |
|---|---|---|---|---|---|---|
| 204 | 100 | 4,960 / 1,240 | 45 | 64 | 0.001 | MSE |
| 205 | 100 | 4,960 / 1,240 | 45 | 64 | 0.001 | MSE |
| 206 | 100 | 4,960 / 1,240 | 45 | 64 | 0.001 | MSE |
| 207 | 400 | 19,840 / 4,960 | 80 | 64 | 0.001 | L1 + KL |
| 208 | 200 | 11,360 / 2,840 | 100 | 32 | 0.0003 | L1 + KL |

| 학습 번호 | 저장 파일 | 별도 추론 파일 |
|---|---|---|
| 204 | `action_chunking_transformer_v2.pt` | `204_Action_Chunking_MUJOCO_infer.py` |
| 205 | `action_chunking_transformer_multistart_v2.pt` | `205_Action_Various_start_infer.py` |
| 206 | `action_chunking_rh_te_v2.pt` | `206_ACT_RH_TE_infer.py` |
| 207 | `cvae_act_goal_transformer_v2.pt` | `207_ACT_CVAE_infer.py` |
| 208 | `visual_act_final_v2.pt` | `208_ACT_Image_infer.py` |

가중치 파일은 모델 구조뿐 아니라 시간 정렬, 데이터 분포, 영상 전처리와 한 쌍이다. 같은 `state_dict` shape로 읽힌다는 사실만으로 행동 의미까지 호환되는 것은 아니다. 수정 전 가중치는 그대로 보존했지만 이 표의 v2 가중치를 만들어 놓은 것은 아니다. 먼저 학습을 실행해야 한다.

## 6.12 검증 방법과 실험의 한계

저장소 루트에서 다음 회귀 검사를 실행할 수 있다.

```bash
MUJOCO_GL=egl python -m unittest discover -s tests -p test_mujoco_bc_to_act.py -v
```

MuJoCo, PyTorch, torchvision, NumPy, GLFW가 설치된 환경을 사용한다. 검사는 데이터 시간 정렬·에피소드 경계, 8개 모델의 역전파·가중치 로드·추론 출력 일치, 목표 입력 이력, 시간 앙상블과 리셋, 실제 EGL 영상 수집을 다룬다. 기존 `.pt`를 덮어쓰지 않으며 사전학습 가중치를 다운로드하지 않는다. EGL을 사용할 수 없으면 환경 변수 없이 실행해 그래픽 검사를 제외한 검사를 수행한다.

검사 통과는 인터페이스와 실행 경로의 오류를 확인한 결과이다. 기본 epoch 전체를 재학습한 성공률 보고서가 아니다. 모델 간 우열, 실제 창의 마우스 조작 품질, 외란 이후 복원, 실제 로봇 성능은 별도의 실험으로 검증해야 한다. 특히 5장의 0.2초 시연 시간과 목표 추종 오차를 함께 확인한다.

1. `T=100`, `H=10`, `K=30`일 때 샘플이 62개인 이유를 설명하라.
2. 204의 큐 실행과 206의 매 스텝 재추론은 어떤 상황에서 다르게 동작하는가?
3. 이 구현에서 같은 실행 시점의 새 예측과 오래된 예측 중 어느 쪽의 가중치가 큰가?
4. 207의 목표 벡터와 CVAE의 잠재 변수는 각각 무엇을 알려주는가?
5. `z`가 policy encoder의 attention에 들어가는 코드 위치를 찾아라.
6. 208이 207과 달리 고정 목표 과제로 정의된 이유는 무엇인가?
7. 학습 loss, 명령 끝점, 실제 관절 도달 오차를 구별해 설명하라.

---

# 07장. 실제 SO-101 모터 제어와 칼리브레이션

> **7장 소스코드:** `07_so101_control_calibration/`. 폴더 앞번호는 책의 장 번호입니다. 4·5·6장은 각각 독립된 폴더를 사용합니다. 내부 파일 번호는 본문의 실행 파일 표에 표시한 예제 ID입니다.

<!-- CLI_GUIDE_START -->
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
<!-- CLI_GUIDE_END -->

---

## 7.1 정책보다 먼저 제어 경로를 확인한다

시뮬레이션에서는 XML이 관절과 구동기의 관계를 정의한다. 실제 로봇에서는 모터 ID, 통신 포트, 엔코더 값과 장착 방향이 그 역할의 일부를 담당한다. 학습된 정책을 연결하기 전에 사람이 보낸 명령이 올바른 관절을 움직이는지 확인해야 한다.

이 장의 기준 폴더는 `07_so101_control_calibration/`이다. 학습 데이터는 아직 수집하지 않는다. 단일 모터 통신에서 시작해 한 관절의 칼리브레이션과 텔레오퍼레이션을 확인한 뒤 전체 팔로 확장한다.

## 7.2 코드와 실습 순서

| 순서 | 파일 | 역할 |
|---|---|---|
| 1 | `201_motor_id_setting.py` | 모터 ID 설정 |
| 2 | `202_feetech_motor_con.py` | Feetech 통신과 모터 제어 예제 |
| 3 | `203_motor_1_calibration_leader.py` | 리더 단일 관절 칼리브레이션 |
| 3 | `203_motor_1_calibration_follow.py` | 팔로워 단일 관절 칼리브레이션 |
| 4 | `204_motor_1_teleoperate.py` | 한 관절의 리더–팔로워 대응 |
| 5 | `205_all_motor_calibration_leader.py` | 리더 전체 관절 칼리브레이션 |
| 5 | `205_all_motor_calibration_follower.py` | 팔로워 전체 관절 칼리브레이션 |
| 6 | `206_motor_all_teleoperate.py` | 전체 관절 텔레오퍼레이션 |
| 참고 | `motor_control.py`, `position_test.py` | 통신 드라이버와 위치 읽기 |

모터 ID 변경은 주소를 실제로 쓰는 작업이다. 먼저 스크립트의 입력 과정과 연결 대상을 확인한다. ID가 같은 여러 모터가 연결된 상태에서 주소를 바꾸는 것은 단일 모터 설정과 다르다. 이 장의 실제 실행은 연결된 장비를 확인한 상태에서 독자가 수행한다.

## 7.3 포트 이름과 관절 ID

리더와 팔로워는 각각 통신 포트를 사용한다. 파일마다 `/dev/ttyUSB0`, `/dev/ttyUSB1` 또는 `/dev/ttyACM0`, `/dev/ttyACM1` 기본값이 들어 있다. 같은 장비라도 연결 방식에 따라 이름이 달라질 수 있으므로 실제 장치를 확인한다.

```bash
ls -l /dev/serial/by-id/
```

이 명령은 연결된 직렬 장치의 식별 링크를 확인하기 위한 것이다. 장치가 없으면 디렉터리가 없을 수 있다. 파일에 적힌 숫자를 보고 어느 쪽이 리더인지 추측하지 않는다.

기존 단독 스크립트에는 공통 `--port` 옵션이 없다. 따라서 원고의 명령을 실행하기 전에 파일 끝의 생성자 호출과 함수 인자를 확인한다. 이번 재편성에서는 포트 기본값을 수정하지 않았다. 포트 조정이 필요한 환경에서는 로컬 실행 설정을 먼저 맞춘다.

관절 순서는 다음을 기본으로 사용한다.

```text
shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper
```

모터 ID는 이 이름과 연결되어야 한다. 파일에 배열로 저장된 여섯 값의 순서와 통신에 사용하는 ID 순서가 다르면 올바른 값도 다른 관절로 전달된다.

## 7.4 MiniFeetechDriver가 맡는 일

`motor_control.py`의 `MiniFeetechDriver`는 직렬 통신을 통해 위치를 읽고, 토크 상태를 설정하고, 위치 명령을 보낸다. 상위 코드가 계산한 숫자를 실제 모터 프로토콜로 전달하는 경계이다.

코드를 읽을 때는 다음 경로를 따라간다.

```text
관절 이름 → calibration의 모터 ID → raw 위치 읽기
목표 위치 계산 → 모터 ID와 목표값 묶기 → 위치 명령 전송
```

`get_position`이 값을 읽지 못해 `None`을 반환하는 경우도 고려해야 한다. 통신 실패 값을 정상 숫자처럼 정규화하면 잘못된 명령이 만들어진다. 각 상위 예제가 읽기 실패를 어떻게 다루는지 확인한다.

여러 모터를 제어할 때 `sync_write_position`은 ID 목록과 목표 목록을 함께 받는다. 두 목록의 길이와 순서를 맞추는 것이 중요하다. 동시 쓰기 기능이 있어도 실제 운동 응답이 완전히 같다는 뜻은 아니다.

## 7.5 왜 칼리브레이션이 필요한가

같은 모델의 모터라도 장착 위치와 조립 방향이 다르면 동일한 raw 값이 같은 로봇 자세를 뜻하지 않을 수 있다. 따라서 raw 값의 범위를 관절 운동 범위와 연결해야 한다.

가장 단순한 정규화는 다음과 같다.

$$
r=\frac{p-p_{min}}{p_{max}-p_{min}}
$$

여기서 $p$는 모터의 raw 위치이며 $r$은 범위 안에서의 비율이다. 이 비율을 팔로워 범위로 옮기면 다음 식이 된다.

$$
p_f^*=p_{f,min}+r(p_{f,max}-p_{f,min})
$$

리더와 팔로워의 raw 범위가 달라도 비율을 이용해 대응시킬 수 있다. 단, 양 끝값이 같으면 분모가 0이다. 칼리브레이션 중 실수로 같은 자세를 두 번 기록한 결과인지 확인해야 한다.

방향이 반대인 관절은 $r$ 대신 $1-r$을 사용하는 등의 방향 처리가 필요하다. `204_motor_1_teleoperate.py`는 `drive_mode`를 읽는 변환을 포함한다. 반면 현재 `206_motor_all_teleoperate.py`는 min/max 기반 비율 대응을 사용하며 같은 방식의 방향 처리가 구현되어 있다고 가정할 수 없다. 한 관절에서 확인한 동작을 전체 팔 코드에서도 다시 확인한다.

## 7.6 단일 관절에서 전체 팔로

먼저 단일 관절 칼리브레이션 파일을 읽는다. 안내하는 위치에서 값을 기록하고 JSON에 저장하는 흐름을 따라간다. 이번 실습의 칼리브레이션 코드는 소프트웨어 파일만 쓰는 것이 아니라 하드웨어 위치 제한을 쓰는 경로도 포함한다. 따라서 10장의 읽기 중심 칼리브레이션과 같은 절차라고 취급하지 않는다.

전체 팔 칼리브레이션의 실행 위치는 다음과 같다. 실제 연결 포트가 코드와 일치하는 것을 먼저 확인한다.

```bash
cd /home/robo/so101_imitation_learning/07_so101_control_calibration
python 205_all_motor_calibration_leader.py
python 205_all_motor_calibration_follower.py
```

주요 결과 파일은 다음 두 개이다.

```text
full_arm_calibration_leader.json
full_arm_calibration_follower.json
```

다른 사람이 제공한 파일이나 과거 장착 상태의 파일이 현재 장비에 맞는다고 단정하지 않는다. 각 관절에 ID와 측정 범위가 들어 있는지 확인하고, 저장 시점의 장착 상태를 기록한다. JSON을 고쳤다면 이후 데이터셋의 정규화 의미에도 영향이 있다.

## 7.7 전체 팔 텔레오퍼레이션

텔레오퍼레이션에서는 사람이 리더를 움직이고 팔로워가 따라간다. 리더는 위치를 읽는 입력 장치이고 팔로워는 목표 위치 명령을 받는 출력 장치이다.

```bash
cd /home/robo/so101_imitation_learning/07_so101_control_calibration
python 206_motor_all_teleoperate.py
```

현재 파일 끝의 기본 포트는 ACM 계열이다. 전체 칼리브레이션 예제의 USB 계열 기본값과 다르므로, 앞 단계가 실행되었다는 이유만으로 이번 단계의 포트도 맞다고 판단하지 않는다.

코드는 먼저 리더 관절들을 읽고 이어서 팔로워에 명령을 쓴다. 각 관절을 조금씩 움직이며 리더의 방향, 팔로워의 방향, 끝 범위, 값 누락을 관찰한다. 출력되는 loop Hz는 통신 루프가 실행된 속도이다. 다음 장의 데이터셋 메타데이터 `fps=30`과 자동으로 일치하는 값이 아니다.

텔레오퍼레이션에서 통신 실패나 방향 반전이 보인다면 학습 데이터를 모으기 전에 원인을 구분한다. 잘못된 동작을 많이 기록하면 그 잘못된 동작이 정책의 정답이 된다.

## 7.8 다음 장으로 가져갈 것

다음 장에 필요한 결과는 두 calibration 파일과, 리더·팔로워가 의도한 관절 순서로 대응한다는 확인이다. 파일만 존재한다고 준비가 끝난 것은 아니다. 실제 읽은 raw 값이 기록한 범위와 맞는지도 확인한다.

다음 장의 데이터 수집 코드는 자기 작업 폴더에서 JSON을 읽는다. 이 장에서 만든 파일을 사용할 때는 8장의 안내에 따라 필요한 위치에 연결한다. 원본 폴더를 이동하는 것만으로 실행 중 생성될 미래의 calibration 파일이 다른 폴더에 자동 전달되지는 않는다.

## 7.9 확인 문제

1. 리더 raw 값을 팔로워 raw 값으로 그대로 보내면 어떤 문제가 생길 수 있는가?
2. calibration의 min/max가 같으면 어떤 계산이 불가능한가?
3. 관절 이름 순서와 모터 ID 순서를 따로 기록해야 하는 이유는 무엇인가?
4. 한 관절 예제와 전체 팔 예제의 방향 처리가 같은지 어디서 확인하는가?
5. 7장 calibration과 10장 calibration을 서로 바꿔 쓰면 안 되는 이유를 예상해 보라.

---

# 08장. 실제 SO-101 데이터 수집, MLP·ACT 학습과 추론

> **8장 소스코드:** `08_so101_imitation_learning/`. 폴더 앞번호는 책의 장 번호입니다. 4·5·6장은 각각 독립된 폴더를 사용합니다. 내부 파일 번호는 본문의 실행 파일 표에 표시한 예제 ID입니다.

<!-- CLI_GUIDE_START -->
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
<!-- CLI_GUIDE_END -->

---

## 8.1 하나의 파이프라인으로 보기

이 장에서는 실제 리더와 실제 팔로워를 사용한다. 사람이 리더를 조작하고, 팔로워 상태와 카메라 영상을 기록한 다음 행동 정책을 학습한다. 마지막에는 리더 대신 정책이 팔로워 목표값을 만든다.

기준 폴더는 `08_so101_imitation_learning/`이다.

```text
101_data_collect/
  101_data_collect_using_teleop.py
  102_collected_data_check.py
  motor_control.py
102_MLP_imitation_learning/
  101_MLP_training.py
  102_MLP_inference.py
103_ACT_imitation_learning/
  101_ACT_training.py
  102_ACT_inference.py
  motor_control.py
```

이 장은 기존 커스텀 코드를 해설한다. 특히 커스텀 ACT의 미래 행동 구성에는 아래에서 설명하는 제약이 있다. 파일명이 ACT라는 이유만으로 9~10장의 공식 ACT 파이프라인과 같은 구현으로 취급하지 않는다.

## 8.2 관측과 행동의 의미를 먼저 정하기

수집 코드의 `TeleopRecorder`는 카메라 영상, 팔로워의 현재 상태, 팔로워에 보낼 목표 행동을 묶어 저장한다.

| feature | shape | 현재 코드의 의미 |
|---|---|---|
| `observation.image` | `(3, 224, 224)` | 실제 카메라 RGB 이미지 |
| `observation.state` | `(6,)` | 팔로워 현재 raw 위치를 0~1 범위로 변환 |
| `action` | `(6,)` | 필터링된 팔로워 목표 raw 위치를 0~1 범위로 변환 |
| `task` | 문자열 | 작업 설명 |

state와 action의 shape가 같다고 두 값의 의미가 같은 것은 아니다. state는 명령 이전에 읽은 팔로워 위치이고 action은 앞으로 보내는 목표이다. 팔로워가 아직 목표에 도달하지 않았다면 두 값은 다르다.

여기서 사용하는 0~1 값은 radian이 아니다. 물리 단위의 각도로 저장하는 9~10장 데이터와 직접 교환할 수 없다. 같은 관절 이름과 같은 차원이라도 단위와 feature 키가 다르면 모델의 입력 계약이 달라진다.

## 8.3 데이터 수집 전에 준비할 파일

수집 프로그램은 현재 작업 폴더에서 두 calibration JSON을 읽는다. 다음 명령은 7장에서 확인한 파일을 수집 폴더에 준비하는 예이다. `-n`은 이미 존재하는 목적지 파일을 덮어쓰지 않기 위한 옵션이다. 파일이 이미 있다면 어느 장비의 측정 결과인지 먼저 비교한다.

```bash
cd /home/robo/so101_imitation_learning/08_so101_imitation_learning/101_data_collect
cp -n ../../07_so101_control_calibration/full_arm_calibration_leader.json .
cp -n ../../07_so101_control_calibration/full_arm_calibration_follower.json .
```

수집 파일의 기본 포트는 `/dev/ttyUSB0`과 `/dev/ttyUSB1`이며 카메라 인덱스는 0이다. 7장의 전체 텔레오퍼레이션 파일과 기본 포트가 다르다. 실제 연결 및 촬영 대상을 확인한다.

현재 `_setup_dataset()`은 같은 `repo_id`의 캐시 디렉터리가 이미 있으면 `shutil.rmtree`로 삭제한다. 기본 `repo_id`는 `my_robot_task`이다. 따라서 이 예제는 이전 데이터에 계속 이어서 녹화하는 코드로 설명할 수 없다. 이전 시연을 보존하려면 실행 전에 데이터셋을 별도 위치에 보관하고, 새 수집의 저장 대상을 확인해야 한다. 이 동작은 원본 코드 그대로이며 이번 재편성에서 실행하거나 수정하지 않았다.

## 8.4 텔레오퍼레이션과 프레임 저장

연결과 저장 대상을 확인한 뒤 다음 위치에서 수집한다.

```bash
cd /home/robo/so101_imitation_learning/08_so101_imitation_learning/101_data_collect
python 101_data_collect_using_teleop.py
```

| 키 | 동작 |
|---|---|
| `r` | 새 녹화 시작 |
| `s` | 현재 episode 저장 |
| `q` | 프로그램 종료 |

현재 에피소드가 필요하다면 `s`로 저장한 다음 종료한다. 이 코드는 `q`에서 미완성 녹화를 저장하는 흐름을 별도로 제공하지 않는다.

목표값에는 EMA 필터를 사용한다.

$$
p_t^{filtered}=\alpha p_t^{target}+(1-\alpha)p_{t-1}^{filtered}
$$

현재 계수는 `alpha=0.3`이다. 정답 action은 이 필터를 거친 목표에 대응한다. 필터 전 리더 위치를 정답으로 저장한다고 읽으면 학습 목표를 잘못 해석하게 된다.

영상은 OpenCV의 BGR, HWC 배열에서 RGB, CHW 텐서로 바꾼다. 관절 하나라도 읽기에 실패하면 해당 루프에서 완전한 프레임을 기록하지 않는다. 그 결과 데이터셋의 `fps=30` 선언이 실제 카메라·통신 루프의 정확한 30Hz 동작을 보장하지는 않는다. 이 수집 루프에는 그 주기를 강제하는 명시적인 대기 제어가 보이지 않으므로, 동작 시간과 저장 프레임 수를 함께 점검해야 한다.

## 8.5 데이터셋을 확인한 뒤 학습하기

```bash
cd /home/robo/so101_imitation_learning/08_so101_imitation_learning/101_data_collect
python 102_collected_data_check.py
```

이 파일에서 사용하는 `repo_id`가 수집 대상과 같은지 확인한다. 메타데이터 요약과 에피소드 영상을 보면서 손으로 의도한 동작이 올바르게 기록되었는지 살펴본다.

검사할 것은 총 프레임 수만이 아니다. 다음을 서로 연결해서 본다.

- 영상의 로봇 동작과 state의 변화 방향이 일치하는가?
- action이 먼저 변하고 state가 뒤따라가는 구간이 자연스러운가?
- 여섯 관절의 순서가 calibration 파일과 같은가?
- 에피소드마다 작업 시작과 끝이 분명한가?
- 값이 계속 0이나 1에 붙어 있다면 범위나 방향이 잘못되지 않았는가?

확인 코드의 이미지 재생에는 실제 `cv2.waitKey(1)` 호출이 사용된다. 함수의 `delay_ms` 인자 이름만 보고 실시간 속도로 재생된다고 생각하지 않는다. 빠르게 지나가는 영상이 반드시 기록 시간 오류를 뜻하지는 않는다.

## 8.6 MLP 정책: 이미지 특징과 관절 상태 결합

`101_MLP_training.py`의 `RobotPolicy`는 ResNet18의 분류층을 제거해 512차원 이미지 특징을 얻는다. 여기에 관절 상태 6개를 붙여 518차원 입력을 만든 다음 MLP로 행동을 예측한다.

```text
이미지 → ResNet18 → 512차원 특징 ┐
                              ├→ 518 → 256 → 128 → 6
현재 관절 상태 → 6차원 벡터 ────┘
```

이 정책은 이미지와 관절 상태를 함께 사용하지만 행동 출력은 한 시점의 6차원 값이다. 5장의 관절 전용 MLP와 입력은 다르지만, 연속적인 행동 정답에 MSE 손실을 적용하는 BC라는 점은 같다.

```bash
cd /home/robo/so101_imitation_learning/08_so101_imitation_learning/102_MLP_imitation_learning
python 101_MLP_training.py
```

기본 학습은 `my_robot_task` 데이터셋, batch size 64, 50 epoch를 사용한다. 결과는 `robot_model_final.pth`에 저장한다. 실제 수집 대상이 다르면 파일의 `train` 인자와 데이터셋 위치를 맞춘다.

이미지 전처리는 특별히 확인해야 한다. 현재 학습 코드는 로더의 이미지에 `/255.0`을 적용한다. 그런데 로더가 이미 0~1 float 이미지를 반환한다면 한 번 더 나누는 셈이다. 수집할 때 uint8을 전달했다는 사실만으로 로더 출력의 범위를 단정하지 않는다. 실제 샘플의 dtype과 min/max를 확인하고 추론 입력과 비교한다. 원본을 유지하는 이번 개정에서는 전처리 코드를 바꾸지 않았다.

## 8.7 MLP 추론과 드라이버 경로

추론은 실시간 이미지를 RGB로 변환하고 0~1 범위 및 ResNet 표준 정규화를 적용한다. 팔로워 현재 위치도 수집 때와 같은 calibration으로 정규화한다. 예측된 행동은 다시 raw 위치로 환산해 모터에 보낸다.

현재 MLP 폴더에는 `motor_control.py`가 없지만 추론 파일은 그 모듈을 import한다. 기존 코드를 수정하지 않고 실행할 때는 수집 폴더의 드라이버를 `PYTHONPATH`로 지정할 수 있다. calibration 파일도 준비한다.

```bash
cd /home/robo/so101_imitation_learning/08_so101_imitation_learning/102_MLP_imitation_learning
cp -n ../../07_so101_control_calibration/full_arm_calibration_follower.json .
PYTHONPATH=../101_data_collect python 102_MLP_inference.py
```

이 명령은 실제 팔로워를 움직이는 프로그램이다. 해당 장비의 calibration과 포트가 확인된 상태에서 수행한다. 모델 파일은 바로 앞 학습에서 생성한 것이어야 한다.

현재 MLP 추론은 raw 목표값을 calibration 범위로 제한한다. 그러나 모델 출력 자체가 항상 0~1이라고 수학적으로 보장되는 것은 아니다. 마지막 계층이 선형층이기 때문이다. 정답을 0~1로 학습한 것과 출력 범위를 강제한 것은 다른 조건이다.

## 8.8 커스텀 ACT의 모델 구조 읽기

`103_ACT_imitation_learning/101_ACT_training.py`의 클래스 이름도 `ACTPolicy`이지만, 파일 안에서 직접 정의한 PyTorch 모델이다. LeRobot에서 import한 공식 `ACTPolicy`와 구분한다.

이 모델은 ResNet18 이미지 특징, 관절 상태 임베딩, CVAE 잠재 변수를 세 토큰으로 쌓는다. `nn.Transformer` 객체를 만들지만 실제 forward에서는 `.encoder(src)`를 호출하고 첫 토큰의 출력에 action head를 적용한다. 기본 출력 shape는 `[B, 50, 6]`이다. Transformer decoder를 실제 사용하는 6장의 `208` 코드와 같은 구조가 아니다.

학습 때는 이미지 특징과 펼친 행동 묶음을 CVAE encoder에 넣는다. 추론에서는 `z=0`을 사용하고, 예측된 50개 중 첫 행동 하나를 선택한다. 현재 실제 로봇 추론에는 시간 앙상블을 적용하는 코드가 없다.

## 8.9 미래 행동 구성의 구현 제약

현재 `get_dataloader`는 데이터셋을 만든 후 `dataset.delta_timesteps` 속성에 정수 목록을 대입한다. 이 설정만으로 실제 미래 행동 chunk가 반환된다고 보장할 수 없다. 9~10장의 코드는 생성자에 `delta_timestamps`를 넘기고 프레임 인덱스를 fps로 나눠 초 단위 시간차로 구성한다.

또한 현재 커스텀 ACT 코드는 로더가 `[B, 6]` 행동 하나를 반환하면 그것을 50번 반복한다. 이 처리로 shape는 맞출 수 있지만, 미래 50개 행동의 서로 다른 시간적 변화를 학습하는 정답은 만들어지지 않는다.

따라서 이 파일을 실행해 loss가 줄어든다고 해서 올바른 ACT 궤적 학습을 완료했다고 결론내리지 않는다. 실제 배치의 action shape와 각 시간 위치의 값을 확인해야 한다. 교육적으로는 모델 구조를 읽고 이 한계를 이해하는 데 사용하고, 실제 미래 행동을 이용한 공식 학습 파이프라인은 9~10장에서 익힌다.

파일의 진입 명령과 결과물은 다음과 같다. 아래 명령은 현재 구현을 실행하는 방법이며, 앞의 제약이 해결되었다는 뜻이 아니다.

```bash
cd /home/robo/so101_imitation_learning/08_so101_imitation_learning/103_ACT_imitation_learning
python 101_ACT_training.py
```

저장되는 가중치는 `act_robot_model.pth`이다. 이 모델을 하드웨어에 연결하기 전에는 학습 데이터의 의미와 전처리 일치부터 확인한다. 추론 파일 `102_ACT_inference.py`는 같은 폴더에 있는 `motor_control.py`와 현재 폴더의 `full_arm_calibration_follower.json`을 사용하며, 기본 포트는 `/dev/ttyUSB1`이다.

현재 ACT 추론에는 MLP 추론과 동일한 명시적 목표 범위 제한이 없고, 위치 읽기 실패와 카메라 읽기 실패 처리에도 제약이 있다. 따라서 단순히 MLP의 실행 파일만 ACT로 바꾸면 같은 수준의 제어 검사가 적용된다고 설명할 수 없다. 이 사실을 확인하지 않은 상태의 무조건적인 모터 실행 절차는 제공하지 않는다.

## 8.10 진동과 실패를 나누어 보기

로봇이 흔들릴 때 모델 종류부터 바꾸기보다 원인별로 확인한다.

| 현상 | 먼저 확인할 내용 |
|---|---|
| 한 관절만 반대로 움직임 | 관절 ID·배열 순서·장착 방향 |
| 시작 즉시 범위 끝으로 이동 | calibration, 단위, 출력 역정규화 |
| 학습은 되지만 추론이 크게 다름 | RGB/BGR, 이미지 값 범위, 정규화 |
| 정지한 장면에서도 명령이 요동 | 관측 노이즈, 입력 상태, 학습 데이터 |
| chunk 출력이 거의 같은 값 반복 | 실제 미래 정답인지 반복 복제인지 |
| 동작 속도가 기록과 다름 | 수집·추론 주기와 데이터셋 fps |

여기서 발견한 기능 문제를 고치는 작업은 폴더 구조 개편과 별개의 변경이다. 이 원고는 기존 구현을 사실대로 설명하고 수정이 필요한 지점을 구분한다.

## 8.11 확인 문제

1. state와 action을 모두 팔로워 현재 위치로 저장하면 어떤 학습 문제가 생기는가?
2. EMA를 적용한 목표를 정답으로 저장하는 이유는 무엇인가?
3. uint8 이미지를 저장한 뒤 로더가 float 이미지를 반환할 수 있다는 사실이 왜 중요한가?
4. `[B, 6]` 행동을 반복해 `[B, 50, 6]`으로 만드는 것과 미래 시점을 읽는 것은 어떻게 다른가?
5. 8장 모델의 체크포인트를 10장 시뮬레이터에 그대로 쓸 수 없는 이유를 세 가지 제시하라.

---

# 09장. MuJoCo IK 시연으로 공식 ACT 학습하기

> **9장 소스코드:** `09_10_so101_mujoco_imitation_learning/`. 폴더 앞번호는 책의 장 번호입니다. 4·5·6장은 각각 독립된 폴더를 사용합니다. 내부 파일 번호는 본문의 실행 파일 표에 표시한 예제 ID입니다.

<!-- CLI_GUIDE_START -->
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
<!-- CLI_GUIDE_END -->

---

## 9.1 관절 궤적 예제에서 물체 조작 과제로

6장에서는 정책 구조를 차례로 확장했다. 이번에는 작업을 명확히 정한다. 빨간 블록을 집어 초록 목표 위치에 내려놓는 것이다. 관절의 목표 자세만 맞추는 문제와 달리, 그리퍼 접촉과 블록 이동도 확인해야 한다.

시연은 사람이 아니라 IK 기반 expert가 만든다. IK는 inverse kinematics, 역기구학이다. 원하는 손끝 위치에 대응하는 관절 자세를 계산한다. expert가 만든 관측–행동 데이터를 LeRobot 형식으로 저장한 다음 공식 ACT 정책을 학습한다.

```text
작업공간의 블록 위치
  → IK 기반 작업 단계와 관절 목표
  → MuJoCo 물리 실행 및 이미지·상태 기록
  → LeRobot 데이터셋
  → 공식 ACT 학습
  → 학습한 정책의 MuJoCo 실행
```

IK expert가 성공했다는 것과 학습 정책이 성공했다는 것을 구별한다. 전자는 시연 생성기의 성능이고 후자는 시연을 학습한 모델의 성능이다.

## 9.2 프로젝트 구조와 설치

이 장과 다음 장은 `09_10_so101_mujoco_imitation_learning/`에 있는 하나의 설치 프로젝트를 공유한다.

```text
09_10_so101_mujoco_imitation_learning/
  101_so101_load/
  102_robot_control/
  103_so101_mujoco_ik/
    so101_mujoco_ik/
      assets/
      env.py
      ik.py
      expert.py
      dataset.py
      train.py
      inference.py
      viewer.py
      cli.py
  104_so101_real_leader_mujoco_il/
    so101_real_leader_mujoco_il/
  tests/
  pyproject.toml
  requirements.txt
  data/
  outputs/
```

101과 102 하위 폴더는 모델 로드와 미리 지정한 순서의 제어 예제이다. 이 장의 IK 정책 파이프라인은 103 하위 패키지를 사용한다. 패키지명과 하위 폴더를 유지했으므로 기존 `pyproject.toml`의 검색 경로도 그대로 유효하다.

기존 `lerobot` conda 환경을 사용하는 설치 예는 다음과 같다.

```bash
cd /home/robo/so101_imitation_learning/09_10_so101_mujoco_imitation_learning
conda activate lerobot
python -m pip install -e '.[train,test]' --no-build-isolation
```

이 명령은 환경에 패키지를 설치하는 단계이다. 프로젝트의 명세는 Python 3.10 이상을 요구하며, LeRobot 범위를 `>=0.4,<0.5`로 지정한다. 설치 명세에 적힌 호환 범위가 모든 조합에 대한 실행 검증을 뜻하지는 않는다. 기존에 동작한 환경이 있다면 해당 환경 정보를 보존한다.

GPU 사용 가능 여부를 확인한다.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

이후 명령에서 `--device cuda`를 쓰려면 위 결과와 실제 CUDA 접근이 맞아야 한다. CPU 검증은 `--device cpu`로 명시할 수 있다. CLI의 하위 명령과 옵션은 다음으로 확인한다.

```bash
python -m so101_mujoco_ik.cli --help
```

## 9.3 환경이 제공하는 상태와 행동

`env.py`의 `SO101PickPlaceEnv`는 로봇, 블록, 목표 영역과 카메라를 관리한다. 모델은 패키지 `assets`의 XML과 mesh를 사용한다. 관절 이름으로 joint ID를 찾은 뒤 qpos 주소, dof 주소, actuator ID를 각각 저장한다.

행동은 여섯 관절의 목표 위치이다. 이 환경의 로봇 관절값은 radian으로 다룬다. `ctrl`에는 목표 위치가 들어가며 외부에서 계산한 토크를 위치 actuator에 전달하지 않는다.

블록은 로봇과 별도의 물리 물체이다. 따라서 전체 `qpos`의 앞이나 뒤 여섯 값을 임의로 로봇 상태라고 고르지 않는다. 명시적으로 찾은 주소로 관절 상태를 추출한다. 4장에서 배운 주소 구분이 실제 구현으로 이어지는 지점이다.

## 9.4 IK 계산을 읽는 방법

`ik.py`에는 `DampedLeastSquaresIK`라는 클래스가 있다. 이름만 보고 특정 Jacobian 역행렬 갱신식을 구현했다고 설명하지 않는다. 현재 실제 계산은 SciPy `least_squares`를 사용하며, 관절 범위 제약 안에서 손끝 위치 및 그리퍼 닫힘 축의 오차를 줄인다.

코드는 로봇 팔의 앞 다섯 관절을 최적화하고 그리퍼 값은 작업 단계에서 정한다. 서로 다른 초기 후보로 최적화를 실행해 잔차가 작은 결과를 선택한다. 손끝 위치뿐 아니라 손가락이 블록 면을 집도록 축 정렬에 관한 잔차도 포함한다.

위치 목표가 작업공간 밖에 있거나 원하는 자세가 관절 범위 안에서 만들어지지 않으면 실패할 수 있다. 현재 코드는 오차가 기준을 넘는 결과를 그대로 시연으로 저장하지 않고 예외를 발생시킨다. 잘못된 expert 궤적을 많이 모으는 대신 실패 원인을 확인해야 한다.

`solve`의 인자 이름이나 클래스명보다 실제 잔차 계산과 종료 조건을 근거로 동작을 해석한다. 선언된 모든 인자가 현재 구현에서 같은 수준으로 사용되는 것은 아니다.

## 9.5 집기 동작을 작은 단계로 나누기

`expert.py`의 `IKPickPlaceExpert`는 작업을 다음 순서로 구성한다.

| 단계 | 의미 | 기록 스텝 수 |
|---|---|---:|
| approach | 블록 위로 접근 | 30 |
| descend | 집을 높이로 내려감 | 24 |
| close | 그리퍼를 닫음 | 24 |
| lift | 블록을 들어 올림 | 25 |
| transfer | 목표 위로 이동 | 35 |
| lower | 목표 높이로 내려감 | 26 |
| open | 블록을 놓음 | 20 |
| retreat | 손끝을 위로 뺌 | 24 |

합계는 208프레임이다. 각 단계의 관절 목표 사이를 smoothstep으로 보간한다. 시연 프레임에는 행동 적용 전 상태, 해당 목표 행동, 관측 이미지가 들어간다. 마지막에는 블록이 안정되도록 추가 물리 스텝을 진행하는데, 그 안정화 구간은 위 208개 기록 프레임과 구별한다.

현재 expert는 양쪽 손가락 접촉 프레임 수, 블록의 최대 높이, 최종 XY 위치 오차를 검사한다. 코드의 기준은 양쪽 접촉 3프레임 이상, 최대 높이 0.055m 이상, 목표 XY 오차 0.045m 이하이다. 이는 현재 교육 과제의 통과 기준이며 모든 집기 작업에 보편적인 정확도 기준은 아니다.

환경은 블록을 equality weld로 손에 고정하지 않는다. 다만 접촉 이후 제한된 안정화 힘을 사용하는 구현이 있으므로, 아무 보조 없이 마찰만으로 물체를 운반한다고 표현하지 않는다. 시뮬레이션 성공을 실제 그리퍼의 물리 성능으로 곧바로 해석하지 않는 이유이기도 하다.

## 9.6 먼저 expert를 확인하기

```bash
cd /home/robo/so101_imitation_learning/09_10_so101_mujoco_imitation_learning
python -m so101_mujoco_ik.cli demo --headless --seed 0
```

이 명령은 창 없이 한 시연의 물리 실행을 확인한다. 화면으로 보려면 다음을 사용한다.

```bash
python -m so101_mujoco_ik.cli demo --seed 0
```

UI demo는 먼저 이미지 렌더링 없이 expert 궤적을 계산한 다음 창에서 재생한다. 데이터 생성 때 이미지를 저장하는 경로와 렌더러 사용 방식이 다르다. `demo --headless`가 성공했다고 이미지 생성용 OpenGL 환경까지 검증된 것은 아니다.

## 9.7 합성 시연 저장하기

이번 원고의 실험 이름에는 `book09`를 붙여 기존 작업 결과와 구분한다. 이미 같은 폴더에 데이터가 있으면 새 이름을 선택한다.

```bash
python -m so101_mujoco_ik.cli generate \
  --root data/book09_ik_pick_place \
  --repo-id local/book09_ik_pick_place \
  --episodes 100 \
  --seed 0
```

`--root`는 로컬 저장 경로이고 `--repo-id`는 데이터셋 식별자이다. 이 명령 자체를 Hub 게시 명령으로 해석하지 않는다. 현재 구현은 로컬 데이터를 생성한다. 생성 대상이 비어 있지 않으면 기존 파일을 삭제하지 않고 실행을 거부한다.

기록 feature는 다음과 같다.

| feature | shape | 단위·의미 |
|---|---|---|
| `observation.images.overview` | `(3, 224, 224)` | MuJoCo RGB 이미지 |
| `observation.state` | `(6,)` | 로봇 현재 관절 위치, rad |
| `action` | `(6,)` | 로봇 목표 관절 위치, rad |
| `task` | 문자열 | 블록 집기·놓기 설명 |

이미지는 LeRobot의 image feature로 저장하며 `use_videos=False`를 사용한다. 8장의 video feature와 보관 방식도 다르다. 프레임을 만드는 구현을 기준으로 저장 형식을 확인한다.

## 9.8 공식 ACT와 미래 시간차

`train.py`는 LeRobot의 `ACTConfig`, `ACTPolicy`를 import한다. 모델 구조와 입출력 feature는 dataset metadata에서 구성한다. 정규화는 dataset 통계를 사용하는 preprocessor와 postprocessor가 담당한다.

미래 행동은 `config.action_delta_indices`에서 얻은 인덱스를 metadata의 fps로 나눠 만든다.

$$
\Delta t_i=\frac{i}{fps}
$$

예를 들어 fps가 30이면 3프레임 뒤는 0.1초이다. 이 값들을 `delta_timestamps`로 전달해야 서로 다른 미래 시점의 action을 요청할 수 있다. 같은 행동을 반복 복제하는 8장의 fallback과 목적이 다르다.

현재 카메라에는 별도의 시간축을 추가하지 않는다. 공식 정책이 현재 이미지 배치를 BCHW로 기대하는 상황에서 이미지 시간축을 하나 더 넣으면 BTCHW가 되어 다른 shape가 되기 때문이다. 미래 행동에는 시간축이 필요하지만 현재 이미지에도 반드시 같은 시간축을 추가해야 하는 것은 아니다.

## 9.9 작은 학습 확인과 본 학습

먼저 적은 스텝으로 데이터 로딩, 모델 forward, 저장 경로를 확인할 수 있다.

```bash
python -m so101_mujoco_ik.cli train \
  --dataset-root data/book09_ik_pick_place \
  --repo-id local/book09_ik_pick_place \
  --output outputs/book09_ik_smoke \
  --steps 10 --batch-size 2 --device cpu
```

이 실험의 목표는 프로그램 연결 확인이다. 열 번의 학습으로 작업 성공률을 주장하지 않는다. 학습 DataLoader는 `drop_last=True`를 사용하므로 유효 샘플 수가 batch size보다 적으면 배치를 만들 수 없다. 작은 데이터로 시험할 때도 최소 한 배치가 가능한지 확인한다.

본 학습 예는 다음과 같다.

```bash
python -m so101_mujoco_ik.cli train \
  --dataset-root data/book09_ik_pick_place \
  --repo-id local/book09_ik_pick_place \
  --output outputs/book09_ik_act \
  --steps 100000 --batch-size 32 --device cuda
```

`--steps`는 optimizer 갱신 횟수이며 epoch와 다르다. 코드가 DataLoader를 다 사용하면 iterator를 다시 만들고 계속 학습한다. 같은 출력 폴더에 재학습 결과를 저장하면 기존 저장물을 바꿀 수 있으므로 실험마다 출력 이름을 구분한다.

가중치뿐 아니라 정책 설정과 전처리·후처리도 저장된다. 추론에서는 이 묶음을 함께 사용해야 정규화된 행동을 원래 단위로 되돌릴 수 있다.

## 9.10 정책 추론과 성능 해석

```bash
python -m so101_mujoco_ik.cli infer \
  --checkpoint outputs/book09_ik_act \
  --episodes 10 --seed 1000 --ui
```

GPU를 사용할 수 없으면 `--device cpu`를 추가한다. 창 없이 실행하려면 `--ui`를 생략한다. 다만 이미지 관측을 만드는 offscreen renderer의 실행 환경은 여전히 필요하다.

평가 기록에는 모델 checkpoint, 평가 seed, 에피소드 수, 작업 성공 여부와 실패 양상을 남긴다. 블록 근처에 도달했는지, 양쪽 손가락으로 집었는지, 이동 중 떨어뜨렸는지, 목표에 놓았는지를 구분하면 원인을 찾기 쉽다.

expert 검증의 기준과 정책 추론의 성공 판정 구현이 같은지도 확인한다. 목표 XY 거리만 가까운 상태와 실제로 집어서 이동한 상태는 다를 수 있다. 회귀 테스트가 통과하는 것은 환경과 expert의 지정 조건에 대한 확인이며, 새로 학습한 ACT의 성공률 검증을 대신하지 않는다.

## 9.11 확인 문제

1. IK expert가 사용하는 블록 위치 정보와 ACT가 받는 관측은 어떻게 다른가?
2. expert 시연 생성에 실패하면 학습을 시작하기 전에 무엇을 확인해야 하는가?
3. 30Hz에서 15프레임 뒤 행동을 요청하려면 시간차는 얼마인가?
4. 정책 가중치만 복사하고 정규화 통계를 잃으면 어떤 문제가 생기는가?
5. headless demo 성공과 이미지 데이터 생성 성공이 서로 다른 검증인 이유는 무엇인가?

---

# 10장. 실제 리더와 MuJoCo 팔로워의 이미테이션 러닝

> **10장 소스코드:** `09_10_so101_mujoco_imitation_learning/`. 폴더 앞번호는 책의 장 번호입니다. 4·5·6장은 각각 독립된 폴더를 사용합니다. 내부 파일 번호는 본문의 실행 파일 표에 표시한 예제 ID입니다.

<!-- CLI_GUIDE_START -->
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
<!-- CLI_GUIDE_END -->

---

## 10.1 사람이 시연하는 가상 로봇

9장에서는 IK expert가 시연을 만들었다. 이번에는 실제 SO-101 리더를 사람이 움직여 시연한다. 리더의 엔코더 값을 가상 로봇의 관절 목표로 바꾸고 MuJoCo 팔로워가 따라간다. 기록되는 영상은 실제 카메라가 아니라 MuJoCo의 관측 이미지이다.

```text
실제 리더 엔코더
  → 리더–MuJoCo calibration
  → 가상 팔로워 목표 관절 위치
  → MuJoCo 동작과 영상
  → LeRobot 시연 데이터
  → 공식 ACT 학습
  → MuJoCo에서 정책 추론
```

실제 팔로워는 이 파이프라인에 필요하지 않다. 따라서 8장의 실제 리더–실제 팔로워 실습과 연결 장비가 다르다. 이 장의 마지막 추론은 가상 팔로워를 움직이며 실제 팔로워에 명령을 전송하지 않는다.

## 10.2 파일별 역할

기준 패키지는 `09_10_so101_mujoco_imitation_learning/104_so101_real_leader_mujoco_il/so101_real_leader_mujoco_il/`이다. 설치와 명령 실행 위치는 9장과 동일하다.

| 파일 | 역할 |
|---|---|
| `hardware.py` | 실제 리더 직렬 통신 |
| `calibration.py` | 엔코더 값과 MuJoCo 자세 대응 측정 |
| `runtime.py` | 값 변환, MuJoCo 상태와 목표 적용 |
| `viewer.py` | 가상 팔로워 표시와 이미지 획득 |
| `dataset.py` | 사람이 만든 시연의 녹화·저장 |
| `inspect_data.py` | 에피소드 이미지와 상태 확인 |
| `train.py` | LeRobot 공식 ACT 학습 |
| `inference.py` | checkpoint 로드와 MuJoCo 추론 |
| `cli.py` | 각 기능의 명령행 진입점 |

명령 순서는 `calibrate → collect → inspect → train → infer`이다. 이미 측정한 calibration이 현재 리더와 XML에 맞는다면 무조건 다시 측정할 필요는 없지만, 그 대응을 확인한 뒤 사용한다.

## 10.3 7장과 다른 칼리브레이션 목표

7장은 리더 raw 범위를 실제 팔로워 raw 범위로 옮겼다. 이번에는 리더 raw 값을 MuJoCo 관절의 radian 목표로 옮긴다. 범위의 양 끝뿐 아니라 MuJoCo 0 rad 자세가 실제 리더에서 어느 raw 값인지 필요하다.

각 관절에서 세 자세를 기록한다.

1. MuJoCo 관절 0 rad에 해당하는 자세.
2. MuJoCo 최소 관절각에 해당하는 자세.
3. MuJoCo 최대 관절각에 해당하는 자세.

여기서 최소·최대는 raw 숫자의 대소가 아니라 가상 관절각 기준이다. 물리 장착 방향에 따라 최소 각도 자세의 raw 값이 더 클 수 있다. 코드는 그 대응에서 방향 정보를 구한다.

```bash
cd /home/robo/so101_imitation_learning/09_10_so101_mujoco_imitation_learning
python -m so101_real_leader_mujoco_il.cli calibrate \
  --port /dev/ttyACM0
```

실제 리더 포트가 USB 계열이면 해당 포트로 바꾼다. 기본 결과 위치는 다음과 같다.

```text
104_so101_real_leader_mujoco_il/
  so101_real_leader_mujoco_il/
    assets/full_arm_calibration_leader.json
```

기존 JSON은 `assets/calibration_backups/`에 보관된다. 현재 이 calibration 구현은 리더 토크를 끄고 위치를 읽으며 persistent homing-offset과 하드웨어 위치 제한을 변경하지 않는다. 7장의 전체 calibration 예제와 하드웨어 쓰기 동작이 다르다.

## 10.4 중심을 정확히 맞추는 구간별 변환

단순한 전체 범위 선형 대응만 쓰면 raw 중점과 MuJoCo 0 rad가 일치하지 않을 수 있다. 실제 관절의 양쪽 이동 범위가 비대칭이면 이 차이가 더 중요해진다.

`raw_to_joint_target`은 `center_raw`가 있으면 중심을 기준으로 두 구간을 나눈다. 방향 보정을 마친 raw 값을 $p$, 중심을 $c$, 범위를 $[l,h]$, MuJoCo 범위를 $[q_{min},q_{max}]$라고 하면 다음과 같다.

$$
q(p)=
\begin{cases}
q_{min}+\dfrac{p-l}{c-l}(0-q_{min}), & p\le c \\
\dfrac{p-c}{h-c}q_{max}, & p>c
\end{cases}
$$

이 식은 $l\mapsto q_{min}$, $c\mapsto0$, $h\mapsto q_{max}$를 맞춘다. 반전 방향이면 raw 값과 중심을 함께 뒤집은 다음 계산한다. 범위를 벗어난 raw 값은 먼저 제한하며 중심이 양 끝 사이에 있는지도 검사한다.

`center_raw`가 없는 이전 JSON에는 전체 범위 선형 변환 경로를 사용한다. 호환 경로가 있다는 사실이 이전 파일의 중심까지 정확히 보장한다는 뜻은 아니다. 현재 측정 형식과 사용 XML을 함께 확인한다.

## 10.5 가상 팔로워의 상태와 목표

`MujocoFollower`는 여섯 관절 이름으로 ID를 찾고 qpos 주소와 actuator ID를 별도로 관리한다. `state`는 실제 시뮬레이션 관절 위치를 읽어 반환한다. `apply_target`은 목표 shape가 6인지 확인하고 NaN/Inf를 거른 뒤 actuator의 허용 범위에 맞춰 `ctrl`에 넣는다.

리더의 목표를 state에 복사하는 것이 아니다. 리더를 갑자기 움직이면 가상 팔로워는 물리 계산을 거쳐 따라가므로 현재 상태와 목표 사이에 차이가 생긴다. 이 차이를 그대로 기록해야 관측과 행동의 의미가 유지된다.

관절 순서는 9장과 같다. 그러나 XML과 물체 배치가 같은 실험이라고 가정하지 않는다. feature schema가 같아도 환경 분포가 달라지면 학습 정책의 동작이 달라질 수 있다.

## 10.6 실제 리더로 데이터 수집하기

새 실험 폴더를 사용해 수집한다.

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/book10_real_leader \
  --repo-id local/book10_real_leader \
  --leader-port /dev/ttyACM0 \
  --fps 30
```

| 키 | 동작 |
|---|---|
| `r` | 현재 장면에서 녹화 시작 |
| `s` | 현재 episode 저장 후 환경 초기화 |
| `d` | 현재 episode 폐기 후 환경 초기화 |
| `q` | 종료, 미완성 episode는 자동 저장하지 않음 |

녹화 전에는 가상 로봇이 리더를 따라가는지 먼저 확인한다. 녹화가 시작되면 작업의 시작부터 완료까지 한 에피소드에 담는다. 실패한 동작을 버리려면 `d`를 사용한다. `s` 이후에는 환경이 초기화되므로 이전 물체 배치에서 그대로 이어지는 동작으로 생각하지 않는다.

수집 루프는 리더 명령을 적용하고 시뮬레이션을 진행한 뒤 영상과 상태를 읽어 그때 유지 중인 목표와 함께 저장한다. 9장 expert의 행동 적용 전 프레임 구성과 세부 순서에 차이가 있다. 두 데이터셋의 feature 의미는 대응하지만 시간 정렬까지 완전히 같다고 가정하지 않는다. 서로 합쳐 쓰기 전에는 수집 시점의 정의를 확인해야 한다.

## 10.7 수집 데이터의 단위와 시간

| feature | shape | 의미 |
|---|---|---|
| `observation.images.overview` | `(3, 224, 224)` | 가상 카메라의 RGB 이미지 |
| `observation.state` | `(6,)` | 가상 팔로워 현재 관절 위치, rad |
| `action` | `(6,)` | calibration으로 변환한 리더 목표 위치, rad |
| `task` | 문자열 | 작업 설명 |

현재 수집 코드는 시뮬레이션 시간을 기준으로 기록 간격을 결정한다. `fps=30`은 데이터 시퀀스의 목표 시간 간격이다. PC에서 한 프레임을 처리하는 벽시계 시간이 항상 정확히 1/30초라는 뜻은 아니다. 리더 읽기가 유효하지 않은 루프는 기록하지 않으므로 통신 상태도 데이터의 시간적 성질에 영향을 준다.

녹화 프레임이 계속 누락되거나 리더 동작과 화면 반응 사이 지연이 크면 장시간 시연을 늘리기 전에 원인을 확인한다. 손으로 움직이는 속도와 화면의 물리 진행이 어떻게 대응하는지도 관찰한다.

## 10.8 빈 데이터셋 초기화 실패에서 복구하기

수집 대상 폴더에 파일이 있으면 기본적으로 실행을 거부한다. 초기화 도중 실패해 `meta/info.json`만 있고 에피소드와 프레임이 모두 0인 경우에는 `--recover-empty`를 사용할 수 있다.

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/book10_real_leader \
  --repo-id local/book10_real_leader \
  --leader-port /dev/ttyACM0 \
  --fps 30 --recover-empty
```

이 옵션은 임의의 기존 데이터셋을 지우거나 이어 쓰는 기능이 아니다. 검증된 빈 골격만 `.incomplete_날짜_시간` 이름으로 보존한 후 다시 시작한다. 다른 데이터 파일이 있거나 실제 프레임이 있으면 거부한다. 이미 정상 에피소드가 있는 데이터에 추가 수집하려는 경우 이 옵션을 해결책으로 사용하지 않는다.

## 10.9 저장된 에피소드 검사

```bash
python -m so101_real_leader_mujoco_il.cli inspect \
  --dataset-root data/book10_real_leader \
  --repo-id local/book10_real_leader \
  --episode 0 --delay-ms 33
```

에피소드 번호를 바꿔 성공한 시연과 실패한 시연을 각각 확인한다. 영상 창에서 `q`로 종료한다. 여기서는 최소한 이미지에 블록과 그리퍼가 보이는지, state와 action이 유한한 여섯 값인지, 관절 방향과 범위가 의도와 같은지를 확인한다.

이 검사는 단지 화면이 열린다는 것을 넘어 데이터의 의미를 검토하는 단계이다. 리더 raw 값이 그대로 `action`에 저장되었다면 radian 범위와 크게 다른 숫자가 나타날 수 있다. 정규화된 0~1 값만 보인다면 8장 자료와 혼동하지 않았는지도 확인한다. 다만 작은 각도 구간의 radian 값도 우연히 0~1 안에 있을 수 있으므로 범위만으로 단위를 단정하지 않는다.

## 10.10 학습과 저장물

먼저 작은 실행으로 입력 연결을 확인한다.

```bash
python -m so101_real_leader_mujoco_il.cli train \
  --dataset-root data/book10_real_leader \
  --repo-id local/book10_real_leader \
  --output outputs/book10_real_leader_smoke \
  --steps 10 --batch-size 2 --device cpu
```

본 학습 예는 다음과 같다.

```bash
python -m so101_real_leader_mujoco_il.cli train \
  --dataset-root data/book10_real_leader \
  --repo-id local/book10_real_leader \
  --output outputs/book10_real_leader_act \
  --steps 100000 --batch-size 32 --device cuda
```

이 학습도 공식 `ACTPolicy`를 사용한다. Dataset은 초 단위 `delta_timestamps`로 실제 미래 행동을 가져오며 processor가 metadata 통계로 정규화한다. 같은 key 이름을 사용하는 것만으로 다른 데이터셋 통계를 재사용하지 않는다.

추론 코드가 검사하는 주요 파일은 다음 네 개이다.

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
```

그 외 processor 관련 파일이 생성되면 함께 보존한다. 전달 단위는 checkpoint 디렉터리 전체이다. `--checkpoint`에는 단일 `.pth` 파일이 아니라 이 디렉터리를 지정한다.

## 10.11 실제 리더 없이 정책 추론하기

```bash
python -m so101_real_leader_mujoco_il.cli infer \
  --checkpoint outputs/book10_real_leader_act \
  --device cuda --seed 1000
```

추론에서는 리더 입력 대신 정책이 목표 관절값을 만든다. 가상 카메라 이미지와 가상 팔로워 현재 상태를 읽어 `predict_action`에 전달한다. 저장된 preprocessor/postprocessor를 거쳐 행동을 원래 단위로 되돌린 뒤 MuJoCo에 적용한다.

UI에서 `r`을 누르면 환경과 정책의 내부 action queue를 함께 초기화한다. `q`는 종료이다. 이 CLI의 실제 리더 파이프라인 추론에는 9장의 `--ui`나 `--episodes` 옵션이 없다. 명령이 비슷해 보여도 각 패키지의 `--help`를 확인한다.

현재 추론은 NaN/Inf 행동을 MuJoCo에 보내기 전에 차단한다. AMP 사용은 CUDA 여부만으로 강제하지 않고 checkpoint의 `use_amp` 설정도 확인한다. CPU로 실행하고 싶다면 `--device cpu`로 바꾼다.

## 10.12 오류를 파이프라인의 위치로 찾기

| 증상 | 확인 위치 | 구체적인 확인 사항 |
|---|---|---|
| 실제 리더를 읽지 못함 | `hardware.py`, 포트 설정 | 포트, 권한, ID, 전원, 응답 |
| 중심 자세에서 가상 관절이 어긋남 | `calibration.py`, `runtime.py` | `center_raw`, 방향, 사용 XML |
| 창이 열리지 않음 | `viewer.py`, 그래픽 환경 | GUI 세션, DISPLAY, OpenGL |
| 데이터 경로가 비어 있지 않다는 오류 | `dataset.py` | 기존 정상 데이터인지 빈 골격인지 |
| 미래 행동 shape가 맞지 않음 | `train.py`, Dataset | fps, action 시간차, feature |
| checkpoint 로드 실패 | `inference.py` | 디렉터리와 필수 파일, 정책 종류 |
| 정책 출력이 NaN/Inf | 모델·전처리 | checkpoint와 통계 일치, 입력값, device |
| 리셋 뒤 이전 동작이 이어짐 | 추론의 reset 경로 | 환경과 정책 상태 동시 초기화 |

포트 연결 실패를 모델 학습으로 해결하려고 하거나, 데이터 단위 오류를 epoch 증가로 해결하려고 하지 않는다. 입력에서 출력까지 어느 경계에서 의미가 달라졌는지 찾는 것이 우선이다.

## 10.13 시뮬레이션 결과를 실제 로봇과 연결해서 생각하기

실제 리더를 사용했다고 해서 학습 정책이 이미 실제 팔로워에서 검증된 것은 아니다. 여기서 영상과 운동 응답은 시뮬레이터에서 만들어졌다. 실제 팔로워로 옮기려면 카메라 조건, 관절 단위, 제어 주기, 마찰과 백래시, 그리퍼 접촉 등의 차이를 다뤄야 한다.

특히 이 파이프라인의 출력은 radian 목표이고, 8장의 커스텀 팔로워 코드는 calibration 비율에서 raw 값으로 복원한다. 숫자 여섯 개라는 이유만으로 출력선을 바로 연결할 수 없다. 새로운 변환과 검증이 필요한 별도 구현 과제이다.

이 장의 완료 기준은 실제 리더 시연을 가상 환경에서 기록하고, 그 데이터로 학습한 공식 ACT를 같은 의미의 가상 관측에 연결해 실행하는 것이다. 실제 팔로워 전이는 이번 코드의 자동 기능이 아니다.

## 10.14 최종 실습 기록과 확인 문제

한 실험을 마칠 때 다음 정보를 기록하면 다시 실행할 수 있다.

| 항목 | 기록 예 |
|---|---|
| 시연 방식 | 실제 리더 / IK expert |
| 환경 | 사용 XML, 카메라와 관절 순서 |
| calibration | 사용 파일과 측정 시점 |
| 데이터셋 | 로컬 경로, repo-id, 에피소드 수, fps |
| 학습 | device, steps, batch size, 출력 폴더 |
| 추론 | checkpoint, seed, reset 방식 |
| 결과 | 작업 성공 여부, 실패 단계, 관측된 지연 |

1. 실제 리더의 raw 값, 가상 팔로워 state, 데이터셋 action의 차이를 설명하라.
2. 중심 raw 값이 범위의 중점이 아닐 때 구간별 변환이 필요한 이유는 무엇인가?
3. 9장과 10장의 feature schema가 같아도 데이터를 무조건 합치면 안 되는 이유는 무엇인가?
4. 환경 리셋과 정책 action queue 리셋을 함께 해야 하는 이유는 무엇인가?
5. 이 장의 정책을 실제 팔로워로 옮기려면 어떤 경계를 새로 구현해야 하는가?
