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
