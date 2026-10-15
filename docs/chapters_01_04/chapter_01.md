# 01장. PyTorch 개론: 로봇 AI 학습 파이프라인의 최소 이해

> **1장 소스코드:** `01_02_pytorch_basic/`. 폴더 앞번호는 책의 장 번호입니다. 4·5·6장은 각각 독립된 폴더를 사용합니다. 내부 파일 번호는 본문의 실행 파일 표에 표시한 예제 ID입니다.

<!-- CLI_GUIDE_START -->
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
<!-- CLI_GUIDE_END -->

---

이 책의 중심 주제는 로봇을 구동하는 정책(policy)과, 그 정책을 구성하는 imitation learning(IL), Transformer, reinforcement learning(RL), 하이브리드 제어이다. PyTorch는 이러한 정책을 구현하고 학습하는 데 사용하는 공통 도구이다. 따라서 이 장의 목적은 PyTorch 자체를 깊이 있게 설명하는 데 있지 않으며, 이후 장의 로봇 AI 코드를 이해하는 데 필요한 최소한의 학습 파이프라인을 정리하는 데 있다.

이 장에서 우선적으로 이해해야 할 항목은 다음 다섯 가지이다.

- `tensor`
- `nn.Linear`
- `Dataset / DataLoader`
- `train loop`
- `GPU`

이 다섯 요소를 하나의 흐름으로 이해하면, 이후의 로봇 AI 코드를 다음과 같은 구조로 해석할 수 있다.

> 데이터를 텐서로 표현하고, 배치 단위로 불러와 모델에 입력한 뒤, loss를 계산하고, 역전파와 파라미터 업데이트를 반복한다.

이 장의 개념 예제와 실제 실행 파일을 구별한다. 실행 실습은 `01_02_pytorch_basic/`의 코드를 사용하는 2장으로 이어진다. 현재 파일명과 실행 위치를 기준으로 연결을 정리했다.


---

## 1.1 PyTorch와 Tensor 기초

### 1.1 PyTorch와 Tensor 기초

#### 1.1.1 PyTorch란 무엇인가

PyTorch는 파이썬 중심의 오픈소스 딥러닝 프레임워크이다. 텐서 연산, 자동미분(autograd), 신경망 모듈, 데이터 처리, GPU 가속 학습 기능을 하나의 환경에서 제공한다. 연구용 프로토타입 작성부터 실제 서비스와 로봇 시스템 개발까지 널리 사용된다.

PyTorch는 Facebook AI Research(현재 Meta AI)를 중심으로 개발되었으며, 초기 개발에는 Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan 등이 참여하였다. 이후 오픈소스 생태계와 함께 빠르게 확장되었고, 현재는 학계와 산업계 전반에서 널리 사용하는 딥러닝 개발 도구가 되었다.

현재 PyTorch는 컴퓨터 비전, 자연어 처리, 음성, 추천 시스템, 생성 모델, 자율주행, 의료 AI, 과학 계산, 로보틱스 등 다양한 분야에서 사용된다. 특히 로보틱스 분야에서는 imitation learning, reinforcement learning, vision-language-action 모델, world model, diffusion policy, transformer policy 구현에 폭넓게 활용된다.

#### 1.1.2 왜 로봇 AI 책에서 먼저 PyTorch를 다루는가

로봇 제어 문제는 형태가 다양하다.

- 관절 상태만으로 액션을 출력하는 정책
- 카메라 이미지와 상태를 함께 사용하는 조작 정책
- 시계열(sequence)을 입력으로 받아 다음 행동을 생성하는 transformer policy
- 보상을 기준으로 정책을 업데이트하는 RL
- 정책 출력과 역기구학(IK)을 함께 사용하는 하이브리드 제어

그러나 구현 관점에서 보면 대부분 다음과 같은 공통 구조를 가진다.

```text
데이터 또는 rollout 수집
→ tensor로 표현
→ batch 구성
→ 모델 forward
→ loss 또는 objective 계산
→ backward
→ optimizer step
→ 반복
```

즉, PyTorch는 별도의 독립 주제라기보다 로봇 AI를 구현하는 공통 실행 환경으로 이해하는 것이 적절하다.

#### 1.1.3 이 장에서 어디까지 알면 충분한가

이 장은 다음 내용을 목표로 하지 않는다.

- PyTorch 내부 구현 원리의 상세 분석
- autograd의 수학적 세부사항 설명
- CUDA 최적화의 고급 튜닝
- 다양한 신경망 레이어와 API의 포괄적 정리

이 장에서 필요한 수준은 다음과 같다.

1. 텐서와 shape를 보고 데이터의 의미를 해석할 수 있다.
2. 간단한 MLP나 이미지 모델이 입력을 출력으로 변환하는 과정을 이해한다.
3. `Dataset`과 `DataLoader`가 데이터를 학습 가능한 배치 형태로 구성한다는 점을 이해한다.
4. train loop에서 실제 학습이 어떤 순서로 진행되는지 설명할 수 있다.
5. CPU/GPU와 device 개념으로 인해 발생하는 기본적인 오류를 피할 수 있다.

이 정도의 이해가 있으면 이후의 IL, Transformer, RL, IL+IK 코드를 읽고 수정하는 데 필요한 기반을 갖추게 된다.

#### 1.1.4 실습 코드와의 연결

이 책의 `01_02_pytorch_basic` 예제는 형식상 일반적인 분류 예제로 보이지만, 내부 구조는 이후의 로봇 정책 학습 코드와 매우 유사하다.

- `101_pytorch_mnist.py`
- `102_pytorch_CNN_cifar10.py`
- `105_pytorch_Resnet_transfer_fine.py`

각 예제의 의미는 다음과 같다.

- `101_pytorch_mnist.py`: MLP 기반 최소 학습 구조를 보여준다. 로봇 관점에서는 `joint state → action` 정책과 유사하다.
- `102_pytorch_CNN_cifar10.py`: 이미지 입력을 처리하는 CNN 구조를 보여준다. 로봇 관점에서는 카메라 기반 조작 정책의 기초에 해당한다.
- `105_pytorch_Resnet_transfer_fine.py`: ResNet18의 사전학습 가중치, 특징 추출과 전체 미세 조정, GPU 사용을 다룬다. 로봇 관점에서는 CNN 기반 시각 특징 추출기의 학습 준비에 해당한다.

---

---

#### 들어가며

PyTorch를 공부할 때 가장 먼저 익혀야 하는 대상은 모델이나 학습 알고리즘이 아니라 **Tensor**입니다.
PyTorch의 대부분의 연산은 Tensor를 중심으로 이루어지며, 입력 데이터, 가중치, 출력값, 손실값까지 모두 Tensor로 표현됩니다.

이 내용을 이 절에서 자세히 다루는 이유는, 처음 학습할 때 전체 흐름을 방해하지 않으면서도 필요할 때 바로 펼쳐 볼 수 있는 **핵심 참고 자료**로 활용하기에 적합하기 때문입니다.
즉, 이 절은 Tensor를 처음 접하는 학습자에게는 기초 개념 정리용으로, 이미 학습을 진행한 독자에게는 빠른 복습용으로 도움을 주기 위한 문서입니다.

---

#### 학습 목표

이 절을 학습한 뒤에는 다음 내용을 이해할 수 있습니다.

- Tensor가 무엇인지 설명할 수 있다.
- Tensor가 PyTorch의 기본 데이터 구조임을 이해한다.
- Tensor와 NumPy 배열의 공통점과 차이점을 구분할 수 있다.
- `torch.Tensor` 객체의 구조를 이해할 수 있다.
- Tensor의 주요 속성과 메서드를 읽고 사용할 수 있다.
- Tensor를 생성하고, 형태를 바꾸고, 장치를 이동하는 기본 작업을 수행할 수 있다.

---

#### 1. Tensor란 무엇인가?

Tensor는 **다차원 수치 데이터를 표현하는 객체**입니다.
쉽게 말하면, 숫자를 하나 저장할 수도 있고, 숫자의 목록을 저장할 수도 있으며, 행렬이나 그보다 더 높은 차원의 데이터도 저장할 수 있습니다.

예를 들어 다음과 같이 생각할 수 있습니다.

- 스칼라(scalar): 하나의 숫자 → 0차원 Tensor
- 벡터(vector): 1차원 숫자 배열 → 1차원 Tensor
- 행렬(matrix): 2차원 숫자 배열 → 2차원 Tensor
- 이미지, 시계열, 배치 데이터 등 → 3차원 이상 Tensor

PyTorch에서는 이러한 Tensor가 **기본 데이터 구조**입니다.
즉, PyTorch에서 계산을 한다는 것은 대부분 Tensor를 만들고, 변형하고, 연산하는 과정이라고 볼 수 있습니다.

중요한 점은 Tensor가 단순한 숫자 묶음이 아니라는 것입니다. Tensor는 다음 요소를 함께 가지는 **객체(object)** 입니다.

- **데이터(data)**: 실제 숫자 값
- **shape**: 데이터의 형태
- **dtype**: 데이터 자료형
- **device**: CPU에 있는지 GPU에 있는지
- **requires_grad**: 자동미분 대상인지 여부
- **연산 기능(methods)**: 형태 변경, 장치 이동, 복사, 변환 등의 기능

따라서 Tensor는 단순한 “배열”이 아니라, **데이터와 그 데이터에 대한 정보, 그리고 연산 기능을 함께 가진 계산 단위**라고 이해하는 것이 좋습니다.

---

#### 2. `torch.Tensor` 클래스 이해하기

파이썬에서 클래스는 객체를 만들기 위한 설계도입니다.
PyTorch에서 Tensor는 `torch.Tensor`라는 클래스와 관련된 객체로 다뤄집니다.

예를 들어 다음 코드를 보겠습니다.

```python
import torch

x = torch.tensor([[1, 2], [3, 4]])
print(x)
```

출력 결과는 보통 다음과 비슷합니다.

```python
tensor([[1, 2],
        [3, 4]])
```

겉으로 보기에는 단순히 표처럼 숫자가 들어 있는 것처럼 보이지만, 실제로는 **Tensor 객체 하나가 생성된 것**입니다.

즉,

```python
x = torch.tensor([[1, 2], [3, 4]])
```

는 다음 의미를 가집니다.

- `[[1, 2], [3, 4]]`라는 데이터를 바탕으로
- PyTorch가 Tensor 객체를 하나 생성하고
- 그 객체를 변수 `x`가 참조하도록 한다

이 객체는 단순히 값을 저장하는 것에서 끝나지 않습니다. 다음과 같은 다양한 속성과 메서드를 제공합니다.

##### 2.1 Tensor 객체가 가진 것

```text
Tensor 객체
├── 데이터
├── shape
├── dtype
├── device
├── requires_grad
└── 다양한 연산 메서드
```

예를 들어 Tensor 객체는 다음과 같은 질문에 답할 수 있습니다.

- 이 Tensor의 크기는 몇인가? → `shape`, `size()`
- 몇 차원인가? → `ndim`
- 정수형인가, 실수형인가? → `dtype`
- CPU에 있는가, GPU에 있는가? → `device`
- 기울기 계산이 필요한가? → `requires_grad`

또한 다음과 같은 작업도 스스로 수행할 수 있습니다.

- 다른 장치로 이동 → `to()`, `cuda()`, `cpu()`
- NumPy 배열로 변환 → `numpy()`
- 형태 변경 → `reshape()`, `view()`
- 차원 순서 변경 → `permute()`, `transpose()`
- 복사 또는 그래프 분리 → `clone()`, `detach()`

즉, Tensor는 **데이터를 담는 그릇**이면서 동시에 **계산에 참여하는 능동적인 객체**입니다.

---

#### 3. Tensor의 내부 구조

다음 Tensor를 예로 들어 보겠습니다.

```python
import torch

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
```

이때 `x`는 다음과 같은 정보를 함께 가집니다.

| 항목          | 값                   | 설명                   |
| ------------- | -------------------- | ---------------------- |
| 데이터        | `[[1, 2], [3, 4]]` | 실제 저장된 숫자 값    |
| shape         | `(2, 2)`           | 2행 2열 구조           |
| ndim          | `2`                | 2차원 Tensor           |
| dtype         | `torch.float32`    | 32비트 부동소수점      |
| device        | `cpu`              | 현재 CPU 메모리에 저장 |
| requires_grad | `False`            | 자동미분 추적 비활성화 |

다음 코드를 통해 직접 확인할 수 있습니다.

```python
import torch

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

print("x =", x)
print("shape =", x.shape)
print("ndim =", x.ndim)
print("dtype =", x.dtype)
print("device =", x.device)
print("requires_grad =", x.requires_grad)
```

실행 결과 예시는 다음과 같습니다.

```python
x = tensor([[1., 2.],
            [3., 4.]])
shape = torch.Size([2, 2])
ndim = 2
dtype = torch.float32
device = cpu
requires_grad = False
```

##### 3.1 각 구성 요소의 의미

###### 데이터

Tensor 안에 실제로 저장된 숫자입니다. 모델 입력값, 가중치, 출력값 모두 결국 숫자 데이터입니다.

###### shape

각 차원의 크기를 나타냅니다.
예를 들어 `(2, 2)`는 2행 2열을 뜻합니다.

###### ndim

Tensor가 몇 차원인지 나타냅니다.
`(2, 2)`는 2차원이므로 `ndim`은 2입니다.

###### dtype

데이터의 자료형입니다.
예를 들어 `torch.int64`, `torch.float32`, `torch.float64` 등이 있습니다.

###### device

Tensor가 어느 장치에 저장되어 있는지 나타냅니다.
일반적으로 `cpu` 또는 `cuda:0` 같은 형태로 표시됩니다.

###### requires_grad

자동미분 대상인지 알려 줍니다.
딥러닝 학습에서 가중치를 업데이트하려면 보통 이 값이 중요한 역할을 합니다.

---

#### 4. Tensor와 NumPy 배열의 차이

PyTorch의 Tensor는 NumPy 배열과 매우 비슷해 보입니다. 실제로 형태, 자료형, 차원 개념 등 많은 부분이 유사합니다.
그래서 NumPy를 먼저 배운 학습자는 Tensor를 비교적 빠르게 익힐 수 있습니다.

하지만 두 객체는 목적과 기능에서 중요한 차이를 가집니다.

##### 4.1 공통점

- 다차원 배열을 표현한다.
- `shape`, `dtype`, 인덱싱, 슬라이싱 개념이 비슷하다.
- 수치 계산에 사용된다.
- 배열 형태 변경, 축 변환 등의 기능이 있다.

##### 4.2 차이점

| 항목                   | NumPy 배열               | PyTorch Tensor             |
| ---------------------- | ------------------------ | -------------------------- |
| 주 용도                | 일반 수치 계산           | 딥러닝 및 텐서 연산        |
| GPU 사용               | 기본적으로 지원하지 않음 | 지원함                     |
| 자동미분               | 직접 지원하지 않음       | 지원함                     |
| 딥러닝 프레임워크 통합 | 제한적                   | PyTorch 전체와 긴밀히 통합 |
| 신경망 학습 활용       | 별도 구현 필요           | 바로 사용 가능             |

##### 4.3 핵심 차이 3가지

###### 1) 자동미분 지원

PyTorch Tensor는 `requires_grad=True`로 설정하면 연산 과정을 추적하여 기울기를 계산할 수 있습니다.
이는 신경망 학습에서 매우 중요합니다.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()

print(x.grad)
```

###### 2) GPU 사용 가능

Tensor는 GPU로 이동하여 대규모 계산을 빠르게 수행할 수 있습니다.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])

if torch.cuda.is_available():
    x = x.cuda()

print(x.device)
```

###### 3) 딥러닝 프레임워크와의 통합

PyTorch의 모델, 손실 함수, 옵티마이저는 모두 Tensor를 중심으로 동작합니다.
즉, Tensor를 이해하면 PyTorch 전체 구조를 이해하기가 훨씬 쉬워집니다.

---

#### 5. 주요 속성

Tensor를 다룰 때 자주 확인하는 대표 속성과 관련 메서드를 정리해 보겠습니다.

##### 5.1 `shape`

Tensor의 형태를 나타냅니다.

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.shape)
```

출력 예:

```python
torch.Size([2, 3])
```

---

##### 5.2 `dtype`

데이터 자료형을 나타냅니다.

```python
import torch

x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(x.dtype)
```

출력 예:

```python
torch.float32
```

---

##### 5.3 `device`

Tensor가 저장된 장치를 나타냅니다.

```python
import torch

x = torch.tensor([1, 2, 3])
print(x.device)
```

출력 예:

```python
cpu
```

---

##### 5.4 `ndim`

Tensor의 차원 수를 나타냅니다.

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.ndim)
```

출력 예:

```python
2
```

---

##### 5.5 `requires_grad`

자동미분 추적 여부를 나타냅니다.

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
print(x.requires_grad)
```

출력 예:

```python
True
```

---

##### 5.6 `T` 또는 전치 관련 속성

2차원 Tensor에서 행과 열을 바꾼 결과를 확인할 때 자주 사용합니다.

```python
import torch

x = torch.tensor([[1, 2], [3, 4]])
print(x.T)
```

출력 예:

```python
tensor([[1, 3],
        [2, 4]])
```

단, 전치와 관련해서는 `transpose()` 메서드를 함께 익혀 두는 것이 좋습니다.

---

##### 5.7 `size()`

`shape`와 비슷한 정보를 메서드 형태로 반환합니다.

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.size())
```

출력 예:

```python
torch.Size([2, 3])
```

`shape`는 속성이고, `size()`는 메서드라는 점만 다를 뿐 초급 단계에서는 거의 같은 용도로 이해해도 됩니다.

---

#### 6. 주요 메서드

이번에는 Tensor를 다룰 때 자주 사용하는 메서드를 살펴보겠습니다.

##### 6.1 `to()`

자료형이나 장치를 변경할 때 사용합니다.

```python
import torch

x = torch.tensor([1, 2, 3])
y = x.to(torch.float32)

print(y)
print(y.dtype)
```

또는 장치 이동에도 사용할 수 있습니다.

```python
if torch.cuda.is_available():
    y = x.to("cuda")
    print(y.device)
```

**주의:** `to()`는 원본을 직접 바꾸지 않고, 변경된 Tensor를 반환하는 경우가 많으므로 결과를 다시 변수에 저장하는 습관이 좋습니다.

---

##### 6.2 `cpu()`

Tensor를 CPU로 이동합니다.

```python
x = torch.tensor([1.0, 2.0])
x_cpu = x.cpu()
print(x_cpu.device)
```

GPU에서 작업한 뒤 NumPy로 변환하기 전에 자주 사용합니다.

---

##### 6.3 `cuda()`

Tensor를 GPU로 이동합니다.

```python
import torch

x = torch.tensor([1.0, 2.0])

if torch.cuda.is_available():
    x = x.cuda()
    print(x.device)
else:
    print("CUDA를 사용할 수 없습니다.")
```

**주의:** GPU가 없는 환경에서 바로 `cuda()`를 호출하면 오류가 발생할 수 있습니다.

---

##### 6.4 `numpy()`

Tensor를 NumPy 배열로 변환합니다.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
arr = x.numpy()

print(type(arr))
print(arr)
```

**주의:** 보통 CPU Tensor에서 사용합니다. GPU Tensor라면 먼저 `cpu()`로 옮겨야 합니다.

---

##### 6.5 `item()`

원소가 하나뿐인 Tensor를 파이썬의 기본 숫자형으로 꺼낼 때 사용합니다.

```python
import torch

x = torch.tensor(3.14)
print(x.item())
```

**주의:** 원소가 여러 개인 Tensor에는 사용할 수 없습니다.

---

##### 6.6 `reshape()`

Tensor의 형태를 바꿉니다.

```python
import torch

x = torch.arange(12)
y = x.reshape(3, 4)

print(y)
print(y.shape)
```

형태만 바꾸고 데이터 수는 같아야 합니다.

---

##### 6.7 `view()`

`reshape()`와 비슷하게 형태를 바꿉니다.

```python
import torch

x = torch.arange(12)
y = x.view(3, 4)
print(y)
```

**주의:** `view()`는 메모리 배치가 연속적(contiguous)이어야 하는 경우가 있어, 상황에 따라 오류가 날 수 있습니다. 초급 단계에서는 우선 `reshape()`를 더 안전하게 사용하는 것도 좋습니다.

---

##### 6.8 `permute()`

차원의 순서를 원하는 방식으로 재배치합니다.

```python
import torch

x = torch.randn(2, 3, 4)
y = x.permute(2, 0, 1)

print(x.shape)
print(y.shape)
```

출력 예:

```python
torch.Size([2, 3, 4])
torch.Size([4, 2, 3])
```

이미지 데이터처럼 `(배치, 높이, 너비, 채널)`과 `(배치, 채널, 높이, 너비)`를 바꿀 때 자주 사용합니다.

---

##### 6.9 `transpose()`

두 차원의 위치를 맞바꿉니다.

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = x.transpose(0, 1)

print(y)
```

`permute()`가 여러 차원의 전체 순서를 바꾸는 데 적합하다면, `transpose()`는 두 차원만 바꾸는 데 적합합니다.

---

##### 6.10 `clone()`

Tensor를 복사합니다.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = x.clone()

print(x)
print(y)
```

원본과 별도의 Tensor를 만들고 싶을 때 사용합니다.

---

##### 6.11 `detach()`

자동미분 그래프에서 분리된 Tensor를 만듭니다.

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
z = y.detach()

print(y.requires_grad)
print(z.requires_grad)
```

출력 예:

```python
True
False
```

학습 과정 중 중간 결과를 그래프와 분리해서 다루고 싶을 때 사용합니다.

---

#### 7. 예제 코드로 확인하기

아래 예제는 초보자가 직접 실행해 보며 Tensor의 핵심 개념을 확인할 수 있도록 구성했습니다.

##### 예제 1. Tensor 생성과 속성 확인

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

print("Tensor:")
print(x)
print("shape:", x.shape)
print("size():", x.size())
print("ndim:", x.ndim)
print("dtype:", x.dtype)
print("device:", x.device)
print("requires_grad:", x.requires_grad)
```

이 예제에서는 Tensor가 단순한 값 묶음이 아니라 여러 메타정보를 함께 가진 객체임을 확인할 수 있습니다.

---

##### 예제 2. 형태 변경과 전치

```python
import torch

x = torch.arange(1, 13)
print("원본:", x)
print("원본 shape:", x.shape)

y = x.reshape(3, 4)
print("\nreshape(3, 4):")
print(y)

z = y.transpose(0, 1)
print("\ntranspose(0, 1):")
print(z)

print("\nT 속성:")
print(y.T)
```

이 예제에서는 1차원 Tensor를 2차원으로 바꾸고, 행과 열을 교환하는 과정을 살펴볼 수 있습니다.

---

##### 예제 3. NumPy 변환과 장치 이동

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])

if torch.cuda.is_available():
    x = x.cuda()
    print("현재 장치:", x.device)

    x_cpu = x.cpu()
    arr = x_cpu.numpy()
else:
    print("CUDA를 사용할 수 없어 CPU에서 진행합니다.")
    arr = x.numpy()

print("NumPy 배열:", arr)
print("자료형:", type(arr))
```

이 예제에서는 GPU 사용 가능 여부를 먼저 확인한 뒤, NumPy 배열로 변환하는 안전한 흐름을 보여 줍니다.

---

#### 8. 자주 하는 실수

Tensor를 처음 다룰 때는 작은 차이 때문에 오류가 발생하는 경우가 많습니다. 아래 실수들을 미리 알아 두면 학습이 훨씬 수월해집니다.

##### 8.1 정수 Tensor와 실수 Tensor를 혼동하는 경우

```python
import torch

x = torch.tensor([1, 2, 3])
print(x.dtype)
```

이 코드는 보통 정수형 Tensor를 만듭니다.
하지만 딥러닝 계산에서는 실수형이 필요한 경우가 많습니다.

```python
x = torch.tensor([1, 2, 3], dtype=torch.float32)
```

처럼 자료형을 명시하는 습관이 중요합니다.

---

##### 8.2 GPU가 없는 환경에서 `cuda()`를 바로 호출하는 경우

```python
x = torch.tensor([1.0, 2.0]).cuda()
```

이 코드는 CUDA 환경이 없으면 오류가 날 수 있습니다.
반드시 다음처럼 확인 후 사용하는 것이 안전합니다.

```python
if torch.cuda.is_available():
    x = x.cuda()
```

---

##### 8.3 `view()` 사용 시 메모리 연속성 문제

일부 연산 후에는 Tensor의 메모리 배치가 연속적이지 않을 수 있습니다. 이런 경우 `view()`는 실패할 수 있습니다.

```python
y = x.transpose(0, 1)
# y.view(...) 는 오류가 날 수 있음
```

이럴 때는 `reshape()`를 사용하거나, 필요하면 `contiguous()`를 거친 뒤 `view()`를 사용합니다.

---

##### 8.4 GPU Tensor에 바로 `numpy()`를 호출하는 경우

`numpy()`는 일반적으로 CPU Tensor에서 사용해야 합니다.

```python
arr = x.numpy()   # x가 GPU Tensor라면 문제 발생 가능
```

안전한 방식은 다음과 같습니다.

```python
arr = x.cpu().numpy()
```

---

##### 8.5 `item()`을 여러 원소 Tensor에 사용하는 경우

```python
x = torch.tensor([1.0, 2.0])
x.item()
```

이 코드는 오류가 발생합니다. `item()`은 원소가 **하나인 Tensor**에만 사용할 수 있습니다.

---

#### 9. 정리

이번 절에서는 PyTorch의 가장 기본적인 데이터 구조인 Tensor를 간단하지만 체계적으로 살펴보았습니다.

핵심을 다시 정리하면 다음과 같습니다.

- Tensor는 **다차원 수치 데이터를 표현하는 객체**이다.
- Tensor는 단순한 숫자 묶음이 아니라 **데이터 + 메타정보 + 연산 기능**을 함께 가진다.
- `torch.Tensor`는 PyTorch 계산의 중심이 되는 객체이다.
- Tensor는 NumPy 배열과 비슷하지만, **자동미분**, **GPU 사용**, **딥러닝 프레임워크와의 통합**이라는 중요한 차이를 가진다.
- `shape`, `dtype`, `device`, `ndim`, `requires_grad` 같은 속성과 `reshape()`, `to()`, `numpy()` 같은 메서드는 반드시 익혀 두어야 한다.

Tensor를 정확히 이해하면 이후에 배우게 될 데이터 처리, 신경망 구성, 손실 계산, 역전파 과정을 훨씬 자연스럽게 받아들일 수 있습니다.

---

#### 10. 실습 문제

##### 실습 1

`1`부터 `12`까지의 값을 이용해 `3 x 4` 형태의 Tensor를 생성하고, 해당 Tensor의 `shape`를 출력해 보세요.

##### 실습 2

실수형 `float32` 자료형을 갖는 Tensor를 하나 생성한 뒤, 그 Tensor의 `dtype`을 출력해 보세요.

##### 실습 3

Tensor를 생성한 뒤, CUDA를 사용할 수 있는 환경이면 GPU로 이동시키고, 그렇지 않으면 현재 장치가 CPU임을 출력해 보세요.

---

이제 Tensor를 바탕으로, 이후 장에서는 PyTorch의 연산과 자동미분, 그리고 신경망 모델 구성으로 자연스럽게 나아갈 수 있습니다.


---

## 1.2 Tensor 연산과 shape 다루기

### 1.2 Tensor 연산과 shape 다루기

앞 절에서 Tensor 연산의 일부를 간단히 살펴보았다면, 이번 절에서는 같은 핵심 주제를 더 자세히 다룹니다. 특히 NumPy와 PyTorch Tensor 연산을 비교하면서, 형태는 비슷하지만 목적과 활용 방식이 어떻게 다른지 정리합니다. 또한 실제로 바로 실행해 볼 수 있는 예제 코드를 통해 기본 연산, 행렬곱, broadcasting, 차원 변환을 단계적으로 익힙니다.

---

#### 학습 목표

이 장을 학습한 뒤에는 다음을 할 수 있어야 합니다.

- Tensor 연산을 이해합니다.
- NumPy와의 차이를 이해합니다.
- Broadcasting을 이해합니다.

---

#### Tensor와 NumPy 비교

NumPy 배열과 PyTorch Tensor는 문법이 매우 비슷합니다. 실제로 PyTorch는 NumPy 사용자가 자연스럽게 적응할 수 있도록 많은 부분에서 유사한 인터페이스를 제공합니다.하지만 두 도구의 **목적**은 다릅니다.

- **NumPy**는 범용 수치 계산 라이브러리입니다.
- **PyTorch Tensor**는 수치 계산 기능에 더해, **GPU 사용**과 **자동미분(autograd)** 기능을 갖춘 딥러닝 중심 자료구조입니다.

##### 비교 표

| 항목           | NumPy                  | PyTorch Tensor                            |
| -------------- | ---------------------- | ----------------------------------------- |
| 실행 장치      | CPU만 사용             | CPU + GPU 사용 가능                       |
| 자동미분       | 지원하지 않음          | 지원함 (`requires_grad=True`)           |
| 딥러닝 친화성  | 낮음                   | 매우 높음                                 |
| 데이터 생성    | `np.array()`         | `torch.tensor()`                        |
| 기본 연산      | 가능                   | 가능                                      |
| 행렬곱 표기    | `@`, `np.matmul()` | `@`, `torch.matmul()`, `torch.mm()` |
| 브로드캐스팅   | 지원                   | 지원                                      |
| 메모리 공유    | 배열 중심              | Tensor ↔ NumPy 간 일부 메모리 공유 가능  |
| 모델 학습 활용 | 직접 구현 필요         | 신경망 학습에 최적화                      |

##### 메모리 공유 관련 차이

PyTorch와 NumPy는 경우에 따라 메모리를 공유할 수 있습니다. 예를 들어 CPU 상의 Tensor를 NumPy 배열로 바꿀 때, 복사가 아니라 같은 메모리를 참조할 수 있습니다.

```python
import numpy as np
import torch

t = torch.tensor([1.0, 2.0, 3.0])
n = t.numpy()

n[0] = 100.0
print(t)  # tensor([100.,   2.,   3.])
```

즉, 한쪽을 수정하면 다른 쪽에도 반영될 수 있습니다.
이 성질은 편리하지만, 초보자에게는 혼란을 줄 수 있으므로 항상 의식하고 사용해야 합니다.

##### 핵심 정리

- 문법은 비슷합니다.
- 둘 다 다차원 배열 연산을 지원합니다.
- 그러나 PyTorch Tensor는 딥러닝 학습을 위해 설계되었기 때문에 GPU, 자동미분, 신경망 모듈과 자연스럽게 연결됩니다.

---

#### Tensor 생성과 기본 연산

가장 먼저 데이터 생성 방법부터 비교해 보겠습니다.

##### 1. 배열/텐서 생성

```python
import numpy as np
import torch

a_np = np.array([[1, 2], [3, 4]])
a_torch = torch.tensor([[1, 2], [3, 4]])

print(a_np)
print(a_torch)
```

예상 실행 결과:

```text
[[1 2]
 [3 4]]
tensor([[1, 2],
        [3, 4]])
```

둘 다 2행 2열의 2차원 데이터를 만듭니다.
형태는 비슷하지만 자료형 표기와 출력 형식은 약간 다를 수 있습니다.

---

##### 2. 기본 사칙연산

NumPy 배열과 Tensor 모두 다음 연산을 지원합니다.

- 덧셈: `+`
- 뺄셈: `-`
- 곱셈: `*`
- 나눗셈: `/`

이때 기본적으로는 **원소별(element-wise) 연산**이 수행됩니다.

> element-wise 연산이란 같은 위치에 있는 원소끼리 각각 계산하는 방식입니다.

예를 들어 다음 두 행렬을 생각해 보겠습니다.

```python
import numpy as np
import torch

x_np = np.array([[1, 2], [3, 4]])
y_np = np.array([[10, 20], [30, 40]])

x_t = torch.tensor([[1, 2], [3, 4]])
y_t = torch.tensor([[10, 20], [30, 40]])

print("NumPy 덧셈")
print(x_np + y_np)

print("Torch 덧셈")
print(x_t + y_t)
```

예상 실행 결과:

```text
NumPy 덧셈
[[11 22]
 [33 44]]
Torch 덧셈
tensor([[11, 22],
        [33, 44]])
```

출력 결과는 다음과 같은 의미를 가집니다.

- `(1 + 10)`, `(2 + 20)`
- `(3 + 30)`, `(4 + 40)`

즉, 같은 위치의 값끼리 더해집니다.

---

##### 3. NumPy와 Tensor를 나란히 비교하기

###### 덧셈

```python
import numpy as np
import torch

a_np = np.array([1, 2, 3])
b_np = np.array([4, 5, 6])

a_t = torch.tensor([1, 2, 3])
b_t = torch.tensor([4, 5, 6])

print("NumPy:", a_np + b_np)
print("Torch:", a_t + b_t)
```

예상 실행 결과:

```text
NumPy: [5 7 9]
Torch: tensor([5, 7, 9])
```

###### 뺄셈

```python
import numpy as np
import torch

a_np = np.array([10, 20, 30])
b_np = np.array([1, 2, 3])

a_t = torch.tensor([10, 20, 30])
b_t = torch.tensor([1, 2, 3])

print("NumPy:", a_np - b_np)
print("Torch:", a_t - b_t)
```

예상 실행 결과:

```text
NumPy: [ 9 18 27]
Torch: tensor([ 9, 18, 27])
```

###### 곱셈

```python
import numpy as np
import torch

a_np = np.array([1, 2, 3])
b_np = np.array([10, 20, 30])

a_t = torch.tensor([1, 2, 3])
b_t = torch.tensor([10, 20, 30])

print("NumPy:", a_np * b_np)
print("Torch:", a_t * b_t)
```

예상 실행 결과:

```text
NumPy: [10 40 90]
Torch: tensor([10, 40, 90])
```

###### 나눗셈

```python
import numpy as np
import torch

a_np = np.array([10, 20, 30])
b_np = np.array([2, 4, 5])

a_t = torch.tensor([10, 20, 30], dtype=torch.float32)
b_t = torch.tensor([2, 4, 5], dtype=torch.float32)

print("NumPy:", a_np / b_np)
print("Torch:", a_t / b_t)
```

예상 실행 결과:

```text
NumPy: [5. 5. 6.]
Torch: tensor([5., 5., 6.])
```

##### 주의할 점

- `*`는 기본적으로 **원소별 곱셈**입니다.
- 행렬곱은 별도의 연산자를 사용해야 합니다.
- Tensor에서는 GPU 이동, 자동미분 같은 기능이 추가로 가능합니다.

---

#### 행렬곱과 원소별 곱의 차이

초보자가 가장 자주 혼동하는 부분 중 하나가 바로 **원소별 곱**과 **행렬곱**입니다.

##### 1. 원소별 곱

원소별 곱은 같은 위치의 값끼리 곱하는 연산입니다.

- NumPy: `*`
- PyTorch: `*`

```python
import numpy as np
import torch

a_np = np.array([[1, 2], [3, 4]])
b_np = np.array([[10, 20], [30, 40]])

a_t = torch.tensor([[1, 2], [3, 4]])
b_t = torch.tensor([[10, 20], [30, 40]])

print("NumPy 원소별 곱")
print(a_np * b_np)

print("Torch 원소별 곱")
print(a_t * b_t)
```

예상 실행 결과:

```text
NumPy 원소별 곱
[[ 10  40]
 [ 90 160]]
Torch 원소별 곱
tensor([[ 10,  40],
        [ 90, 160]])
```

---

##### 2. 행렬곱

행렬곱은 선형대수에서 사용하는 연산으로, 단순히 같은 위치끼리 곱하는 것이 아닙니다.

- NumPy: `@`, `np.matmul()`
- PyTorch: `@`, `torch.matmul()`, `torch.mm()`

```python
import numpy as np
import torch

a_np = np.array([[1, 2], [3, 4]])
b_np = np.array([[5, 6], [7, 8]])

a_t = torch.tensor([[1, 2], [3, 4]])
b_t = torch.tensor([[5, 6], [7, 8]])

print("NumPy 행렬곱 (@)")
print(a_np @ b_np)

print("NumPy matmul")
print(np.matmul(a_np, b_np))

print("Torch 행렬곱 (@)")
print(a_t @ b_t)

print("Torch matmul")
print(torch.matmul(a_t, b_t))

print("Torch mm")
print(torch.mm(a_t, b_t))
```

예상 실행 결과:

```text
NumPy 행렬곱 (@)
[[19 22]
 [43 50]]
NumPy matmul
[[19 22]
 [43 50]]
Torch 행렬곱 (@)
tensor([[19, 22],
        [43, 50]])
Torch matmul
tensor([[19, 22],
        [43, 50]])
Torch mm
tensor([[19, 22],
        [43, 50]])
```

---

##### 3. 왜 shape가 맞아야 할까?

행렬곱은 다음 규칙을 따릅니다.

- `(m, n)` shape의 행렬
- `(n, p)` shape의 행렬

이 두 개는 곱할 수 있고, 결과는 `(m, p)`가 됩니다.

즉, **앞 행렬의 열 개수**와 **뒤 행렬의 행 개수**가 같아야 합니다.

예를 들어:

- `(2, 3)` × `(3, 4)` → 가능, 결과는 `(2, 4)`
- `(2, 3)` × `(2, 4)` → 불가능

```python
import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6]])      # shape: (2, 3)
b = torch.tensor([[1, 2], [3, 4], [5, 6]])    # shape: (3, 2)

print(torch.matmul(a, b))  # 결과 shape: (2, 2)
```

예상 실행 결과:

```text
tensor([[22, 28],
        [49, 64]])
```

##### `matmul()`과 `mm()`의 차이

- `torch.mm()`은 **2차원 행렬끼리의 곱**에 사용합니다.
- `torch.matmul()`은 더 일반적이며, 벡터/행렬/배치 행렬곱까지 폭넓게 처리합니다.

초보자 단계에서는 다음처럼 기억하셔도 충분합니다.

- **2차원 행렬곱**: `torch.mm()` 또는 `@`
- **더 일반적인 상황**: `torch.matmul()`

---

#### Broadcasting

Broadcasting은 Tensor 연산에서 **매우 중요합니다**.
실제 딥러닝 코드에서는 매번 같은 shape를 직접 맞추지 않고, broadcasting을 이용해 간결하게 계산하는 경우가 많습니다.

NumPy와 PyTorch는 거의 동일한 broadcasting 규칙을 사용합니다.
따라서 NumPy broadcasting을 이해하면 PyTorch Tensor 연산도 훨씬 쉽게 이해할 수 있습니다.

---

##### Broadcasting이란?

shape가 완전히 같지 않더라도, 특정 조건을 만족하면 **더 작은 배열/텐서를 자동으로 확장한 것처럼 간주하여 연산**하는 기능입니다.

예를 들어 다음을 생각해 보겠습니다.

- `A`: shape `(3, 4)`
- `B`: shape `(4,)`

이 둘을 더하면 어떻게 될까요?

직접 보면 다음과 같습니다.

```python
import torch

A = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])  # shape: (3, 4)

B = torch.tensor([100, 200, 300, 400])  # shape: (4,)

C = A + B
print(C)
```

예상 실행 결과:

```text
tensor([[101, 202, 303, 404],
        [105, 206, 307, 408],
        [109, 210, 311, 412]])
```

결과는 다음과 같은 의미를 가집니다.

```python
tensor([
    [101, 202, 303, 404],
    [105, 206, 307, 408],
    [109, 210, 311, 412]
])
```

즉, `B`가 각 행에 반복 적용된 것처럼 계산됩니다.

---

##### `(3, 4) + (4,)`가 가능한 이유

핵심은 `(4,)`를 내부적으로 `(1, 4)`처럼 보고, 이것을 다시 `(3, 4)`로 확장할 수 있기 때문입니다.

###### 단계 1. 원래 shape 확인

- `A.shape = (3, 4)`
- `B.shape = (4,)`

###### 단계 2. 뒤에서부터 차원 비교

broadcasting은 **뒤쪽 차원부터** 비교합니다.

- `A`: `(3, 4)`
- `B`: `(4,)`

`B`는 1차원이므로, 비교를 쉽게 하기 위해 다음처럼 생각할 수 있습니다.

- `A`: `(3, 4)`
- `B`: `(1, 4)`

###### 단계 3. 각 차원을 비교

오른쪽부터 비교하면:

- 마지막 차원: `4`와 `4` → 같으므로 가능
- 그 앞 차원: `3`과 `1` → `1`은 확장 가능하므로 가능

따라서 `B`는 마치 다음처럼 확장된 것처럼 동작합니다.

```python
[
    [100, 200, 300, 400],
    [100, 200, 300, 400],
    [100, 200, 300, 400]
]
```

이렇게 보면 `A + B`가 왜 가능한지 자연스럽게 이해할 수 있습니다.

---

##### Broadcasting 규칙 정리

다음 규칙을 기억해 두시면 좋습니다.

- shape는 **뒤에서부터 비교**합니다.
- 각 차원이
  - 서로 같거나
  - 둘 중 하나가 `1`이면
    broadcasting이 가능합니다.
- 없는 차원은 필요하면 앞쪽에 `1`이 붙은 것처럼 간주할 수 있습니다.
- 조건을 만족하지 않으면 연산할 수 없습니다.

---

##### 예제 1. `(3, 4) + (4,)`

###### PyTorch

```python
import torch

A = torch.arange(12).reshape(3, 4)
B = torch.tensor([10, 20, 30, 40])

print("A shape:", A.shape)
print("B shape:", B.shape)
print(A + B)
```

예상 실행 결과:

```text
A shape: torch.Size([3, 4])
B shape: torch.Size([4])
tensor([[10, 21, 32, 43],
        [14, 25, 36, 47],
        [18, 29, 40, 51]])
```

###### NumPy

```python
import numpy as np

A = np.arange(12).reshape(3, 4)
B = np.array([10, 20, 30, 40])

print("A shape:", A.shape)
print("B shape:", B.shape)
print(A + B)
```

예상 실행 결과:

```text
A shape: (3, 4)
B shape: (4,)
[[10 21 32 43]
 [14 25 36 47]
 [18 29 40 51]]
```

---

##### 예제 2. `(3, 4) + (3, 1)`

이번에는 열 방향으로 broadcasting이 일어나는 예를 살펴보겠습니다.

```python
import torch

A = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])  # (3, 4)

B = torch.tensor([
    [100],
    [200],
    [300]
])  # (3, 1)

print(A + B)
```

예상 실행 결과:

```text
tensor([[101, 102, 103, 104],
        [205, 206, 207, 208],
        [309, 310, 311, 312]])
```

`B`는 실제로는 `(3, 1)`이지만, 각 행의 값이 열 방향으로 반복된 것처럼 확장되어 `(3, 4)`와 더해집니다.

결과는 다음과 같은 의미다.

```python
[
    [101, 102, 103, 104],
    [205, 206, 207, 208],
    [309, 310, 311, 312]
]
```

---

##### 예제 3. 스칼라와의 연산

스칼라도 broadcasting의 한 예입니다.

```python
import numpy as np
import torch

a_np = np.array([[1, 2], [3, 4]])
a_t = torch.tensor([[1, 2], [3, 4]])

print(a_np + 10)
print(a_t + 10)
```

예상 실행 결과:

```text
[[11 12]
 [13 14]]
tensor([[11, 12],
        [13, 14]])
```

숫자 `10`이 모든 위치에 더해진 것처럼 동작합니다.

---

##### broadcasting이 실패하는 경우

예를 들어 다음 shape는 서로 맞지 않습니다.

- `(3, 4)`
- `(2, 4)`

뒤에서 비교하면:

- 마지막 차원: `4`와 `4` → 가능
- 그 앞 차원: `3`과 `2` → 같지 않고, 어느 쪽도 `1`이 아님 → 불가능

```python
import torch

A = torch.randn(3, 4)
B = torch.randn(2, 4)

# print(A + B)  # 오류 발생
```

##### broadcasting을 이해해야 하는 이유

딥러닝에서는 다음과 같은 상황에서 매우 자주 사용됩니다.

- bias 더하기
- 정규화 연산
- 채널별 스케일링
- 배치 단위 연산
- 이미지/특징맵 처리

즉, broadcasting을 이해하면 코드가 짧아질 뿐 아니라, 모델 내부 계산을 훨씬 쉽게 읽을 수 있습니다.

---

#### 차원 변환과 형태 변경

Tensor를 다룰 때는 값 자체보다도 **shape**를 올바르게 다루는 능력이 매우 중요합니다.
특히 딥러닝에서는 데이터 형식을 모델이 기대하는 shape로 맞추는 작업이 자주 필요합니다.

이 절에서는 다음 연산을 살펴봅니다.

- `reshape`
- `view`
- `flatten`
- `unsqueeze`
- `squeeze`
- `permute`

또한 NumPy에서 대응되는 함수도 함께 비교합니다.

---

##### 1. `reshape`

`reshape`는 Tensor의 전체 원소 수를 유지하면서 shape만 바꿉니다.

- PyTorch: `tensor.reshape(...)`
- NumPy: `array.reshape(...)`

```python
import numpy as np
import torch

a_np = np.arange(12)
a_t = torch.arange(12)

print(a_np.reshape(3, 4))
print(a_t.reshape(3, 4))
```

예상 실행 결과:

```text
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
```

###### 핵심 포인트

- 전체 원소 수는 같아야 합니다.
- 예: 12개 원소 → `(3, 4)`, `(2, 6)`, `(12,)` 등으로 변경 가능

---

##### 2. `view`

`view`는 PyTorch에서 shape를 바꾸는 대표적인 방법입니다.

```python
import torch

x = torch.arange(12)
y = x.view(3, 4)

print(y)
```

예상 실행 결과:

```text
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
```

###### `view`와 `reshape`의 차이

초보자 관점에서는 다음처럼 이해하시면 좋습니다.

- `view`:
  - 가능하면 **같은 메모리를 공유**하면서 shape만 바꿉니다.
  - 따라서 더 가볍게 동작할 수 있습니다.
  - 하지만 메모리 배치가 맞지 않으면 사용할 수 없는 경우가 있습니다.
- `reshape`:
  - 가능하면 메모리를 공유하려고 시도합니다.
  - 필요하면 내부적으로 복사하여 결과를 만듭니다.
  - 그래서 더 유연하게 사용할 수 있습니다.

즉, 실전에서는 `reshape`가 더 편한 경우가 많고,
`view`는 "메모리 구조가 맞을 때 빠르게 모양만 바꾼다"라고 생각하면 됩니다.

---

##### 3. `flatten`

`flatten`은 여러 차원의 Tensor를 1차원으로 펼칩니다.

- PyTorch: `tensor.flatten()`
- NumPy: `array.flatten()`

```python
import numpy as np
import torch

a_np = np.array([[1, 2], [3, 4]])
a_t = torch.tensor([[1, 2], [3, 4]])

print(a_np.flatten())
print(a_t.flatten())
```

예상 실행 결과:

```text
[1 2 3 4]
tensor([1, 2, 3, 4])
```

###### 언제 사용할까요?

- CNN 출력을 fully connected layer에 넣기 전
- 다차원 데이터를 일렬로 정리하고 싶을 때

---

##### 4. `unsqueeze`

`unsqueeze`는 크기 1인 차원을 새로 추가합니다.

- PyTorch: `tensor.unsqueeze(dim)`
- NumPy 대응: `np.expand_dims(array, axis=...)`

```python
import numpy as np
import torch

a_t = torch.tensor([1, 2, 3])      # shape: (3,)
b_t = a_t.unsqueeze(0)             # shape: (1, 3)
c_t = a_t.unsqueeze(1)             # shape: (3, 1)

print(b_t.shape)
print(c_t.shape)

a_np = np.array([1, 2, 3])
b_np = np.expand_dims(a_np, axis=0)
c_np = np.expand_dims(a_np, axis=1)

print(b_np.shape)
print(c_np.shape)
```

예상 실행 결과:

```text
torch.Size([1, 3])
torch.Size([3, 1])
(1, 3)
(3, 1)
```

###### 왜 필요할까요?

- 배치 차원 추가
- broadcasting을 위한 shape 맞춤
- 모델 입력 형식 맞추기

---

##### 5. `squeeze`

`squeeze`는 크기가 1인 차원을 제거합니다.

- PyTorch: `tensor.squeeze()`
- NumPy: `np.squeeze(array)`

```python
import numpy as np
import torch

a_t = torch.tensor([[[1, 2, 3]]])   # shape: (1, 1, 3)
print(a_t.shape)

b_t = a_t.squeeze()
print(b_t.shape)

a_np = np.array([[[1, 2, 3]]])
print(a_np.shape)
print(np.squeeze(a_np).shape)
```

예상 실행 결과:

```text
torch.Size([1, 1, 3])
torch.Size([3])
(1, 1, 3)
(3,)
```

###### 언제 사용할까요?

- 필요 없는 배치 차원 제거
- 연산 후 남은 불필요한 축 정리

---

##### 6. `permute`

`permute`는 차원의 순서를 바꿉니다.

- PyTorch: `tensor.permute(...)`
- NumPy 대응: `transpose(...)`

이 연산은 **CNN에서 매우 많이 사용됩니다**.
특히 이미지 데이터 형식을 바꿀 때 중요합니다.

일반적으로 이미지 데이터는 다음 두 형식 중 하나를 사용합니다.

- **HWC**: `(높이, 너비, 채널)`
- **CHW**: `(채널, 높이, 너비)`

PyTorch의 CNN 계열 연산은 보통 **CHW** 또는 배치 포함 시 **NCHW** 형식을 기대합니다.
따라서 이미지가 HWC 형식으로 들어오면 CHW로 바꿔야 합니다.

```python
import numpy as np
import torch

img_t = torch.randn(224, 224, 3)  # HWC
img_chw_t = img_t.permute(2, 0, 1)

print("Torch HWC:", img_t.shape)
print("Torch CHW:", img_chw_t.shape)

img_np = np.random.randn(224, 224, 3)  # HWC
img_chw_np = np.transpose(img_np, (2, 0, 1))

print("NumPy HWC:", img_np.shape)
print("NumPy CHW:", img_chw_np.shape)
```

예상 실행 결과:

```text
Torch HWC: torch.Size([224, 224, 3])
Torch CHW: torch.Size([3, 224, 224])
NumPy HWC: (224, 224, 3)
NumPy CHW: (3, 224, 224)
```

###### `permute(2, 0, 1)`의 의미

원래 축이 `(0, 1, 2)`라고 가정해 보겠습니다.

- `0`: H
- `1`: W
- `2`: C

이를 `(2, 0, 1)`로 재배열하면

- 새 0번 축 ← 원래 2번 축(C)
- 새 1번 축 ← 원래 0번 축(H)
- 새 2번 축 ← 원래 1번 축(W)

즉, `HWC → CHW`가 됩니다.

---

##### 차원 변환 종합 예제

```python
import torch

x = torch.arange(24)
print("원본:", x.shape)

x1 = x.reshape(2, 3, 4)
print("reshape(2, 3, 4):", x1.shape)

x2 = x1.flatten()
print("flatten():", x2.shape)

x3 = x2.unsqueeze(0)
print("unsqueeze(0):", x3.shape)

x4 = x3.squeeze()
print("squeeze():", x4.shape)
```

예상 실행 결과:

```text
원본: torch.Size([24])
reshape(2, 3, 4): torch.Size([2, 3, 4])
flatten(): torch.Size([24])
unsqueeze(0): torch.Size([1, 24])
squeeze(): torch.Size([24])
```

---

##### NumPy 대응 함수 정리

| 목적                        | PyTorch         | NumPy                       |
| --------------------------- | --------------- | --------------------------- |
| shape 변경                  | `reshape()`   | `reshape()`               |
| 메모리 공유 기반 shape 변경 | `view()`      | 직접적으로 동일 개념은 다름 |
| 1차원 펼치기                | `flatten()`   | `flatten()`               |
| 차원 추가                   | `unsqueeze()` | `expand_dims()`           |
| 크기 1 차원 제거            | `squeeze()`   | `squeeze()`               |
| 차원 순서 변경              | `permute()`   | `transpose()`             |

---

#### 실습

이제 직접 Tensor shape와 broadcasting을 다뤄 보겠습니다.
각 실습은 **문제 → 힌트 → 정답 예시** 순서로 구성되어 있습니다.

---

##### 실습 1. 3x4 Tensor를 2x6으로 바꾸기

###### 문제

다음 3x4 Tensor를 2x6 Tensor로 바꾸어 보겠습니다.

```python
import torch

x = torch.arange(12).reshape(3, 4)
print(x)
print(x.shape)
```

###### 힌트

- 전체 원소 수는 유지되어야 합니다.
- `3 x 4 = 12`
- `2 x 6 = 12`
- `reshape()` 또는 `view()`를 사용할 수 있습니다.

###### 정답 예시

```python
import torch

x = torch.arange(12).reshape(3, 4)
y = x.reshape(2, 6)

print(y)
print(y.shape)
```

예상 실행 결과:

```text
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])
torch.Size([2, 6])
```

또는

```python
import torch

x = torch.arange(12).reshape(3, 4)
y = x.view(2, 6)

print(y)
print(y.shape)
```

예상 실행 결과:

```text
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])
torch.Size([2, 6])
```

---

##### 실습 2. RGB 이미지를 HWC에서 CHW로 변경하기

###### 문제

다음 RGB 이미지 Tensor는 HWC 형식입니다. 이를 CHW 형식으로 바꾸어 보겠습니다.

```python
import torch

img = torch.randn(64, 64, 3)  # HWC
print(img.shape)
```

###### 힌트

- HWC는 `(높이, 너비, 채널)`입니다.
- CHW는 `(채널, 높이, 너비)`입니다.
- 차원 순서를 바꾸어야 하므로 `permute()`를 사용합니다.

###### 정답 예시

```python
import torch

img = torch.randn(64, 64, 3)  # HWC
img_chw = img.permute(2, 0, 1)

print(img_chw.shape)  # torch.Size([3, 64, 64])
```

예상 실행 결과:

```text
torch.Size([3, 64, 64])
```

NumPy로 하면 다음과 같습니다.

```python
import numpy as np

img = np.random.randn(64, 64, 3)  # HWC
img_chw = np.transpose(img, (2, 0, 1))

print(img_chw.shape)  # (3, 64, 64)
```

예상 실행 결과:

```text
(3, 64, 64)
```

---

##### 실습 3. Broadcasting을 이용하여 `(3,4) + (4,)` 연산 수행하기

###### 문제

다음 두 Tensor를 broadcasting을 이용하여 더해 보겠습니다.

- `A`: shape `(3, 4)`
- `B`: shape `(4,)`

```python
import torch

A = torch.arange(12).reshape(3, 4)
B = torch.tensor([10, 20, 30, 40])
```

###### 힌트

- 두 Tensor의 shape가 완전히 같지 않아도 됩니다.
- `(4,)`는 `(1, 4)`처럼 생각할 수 있습니다.
- 각 행에 같은 벡터가 더해지는 결과가 나온다.

###### 정답 예시

```python
import torch

A = torch.arange(12).reshape(3, 4)
B = torch.tensor([10, 20, 30, 40])

C = A + B

print("A shape:", A.shape)
print("B shape:", B.shape)
print(C)
```

예상 실행 결과:

```text
A shape: torch.Size([3, 4])
B shape: torch.Size([4])
tensor([[10, 21, 32, 43],
        [14, 25, 36, 47],
        [18, 29, 40, 51]])
```

NumPy 버전도 확인해 보겠습니다.

```python
import numpy as np

A = np.arange(12).reshape(3, 4)
B = np.array([10, 20, 30, 40])

C = A + B

print("A shape:", A.shape)
print("B shape:", B.shape)
print(C)
```

예상 실행 결과:

```text
A shape: (3, 4)
B shape: (4,)
[[10 21 32 43]
 [14 25 36 47]
 [18 29 40 51]]
```

---

#### 정리

이번 절에서 다룬 핵심 내용을 정리하면 다음과 같습니다.

- NumPy 배열과 PyTorch Tensor는 문법이 매우 비슷합니다.
- NumPy는 범용 수치 계산용이고, PyTorch Tensor는 딥러닝 학습에 더 적합합니다.
- Tensor는 CPU뿐 아니라 GPU에서도 연산할 수 있습니다.
- PyTorch Tensor는 자동미분 기능을 지원합니다.
- `+`, `-`, `*`, `/`는 기본적으로 **원소별 연산**입니다.
- 행렬곱은 `@`, `matmul()`, `mm()` 등으로 수행합니다.
- 행렬곱에서는 **앞 행렬의 열 수 = 뒤 행렬의 행 수** 조건이 필요합니다.
- broadcasting은 shape가 완전히 같지 않아도 자동으로 확장하여 연산하게 해 주는 중요한 기능입니다.
- broadcasting 규칙은 **뒤에서부터 비교**하며, 각 차원이 **같거나 1이면** 확장 가능합니다.
- `(3, 4) + (4,)`는 `(4,)`를 `(1, 4)`처럼 보고 `(3, 4)`로 확장할 수 있기 때문에 가능합니다.
- `reshape`, `view`, `flatten`, `unsqueeze`, `squeeze`, `permute`는 Tensor shape를 다룰 때 매우 자주 사용됩니다.
- 특히 `permute`는 CNN에서 이미지 형식인 **HWC → CHW** 변환에 매우 중요합니다.

Tensor 연산은 딥러닝의 거의 모든 계산의 기반이 됩니다.
따라서 단순히 문법만 외우기보다, **shape가 어떻게 바뀌는지**, **broadcasting이 왜 가능한지**, **원소별 연산과 행렬곱이 어떻게 다른지**를 직접 손으로 확인하며 익히는 것이 중요합니다.


---

## 1.3 자동미분과 PyTorch 학습 파이프라인

### 1.3 자동미분과 PyTorch 학습 파이프라인

이 절은 **PyTorch의 핵심 개념 중 하나인 자동미분(Automatic Differentiation)** 을 설명합니다.
사실 딥러닝 프레임워크를 제대로 이해하려면, 이 부분이 가장 중요하다고 해도 과언이 아닙니다.
모델을 만들고, loss를 계산하고, `backward()`를 호출하고, `optimizer.step()`을 수행하는 모든 흐름의 중심에는 자동미분이 있습니다.

특히 초급 학습자들은 다음과 같은 코드를 자주 외웁니다.

```python
loss.backward()
optimizer.step()
```

하지만 이 두 줄은 단순한 암기 대상이 아니라, **각각 완전히 다른 역할**을 갖습니다.
이 장에서는 그 차이를 분명히 이해하고, 자동미분이 신경망 학습에서 어떻게 사용되는지 차근차근 살펴보겠습니다.

---

#### 학습목표

이 장을 학습한 뒤에는 다음을 설명할 수 있어야 합니다.

- 자동미분이 왜 필요한지 이해한다.
- `requires_grad`의 의미를 이해한다.
- `Computational Graph`가 무엇인지 이해한다.
- `backward()`가 정확히 무엇을 하는지 이해한다.
- `optimizer.step()`이 어떤 역할을 하는지 이해한다.
- 자동미분과 역전파법의 관계를 구분해서 설명할 수 있다.

---

#### 1. 왜 자동미분이 필요한가

가장 단순한 함수부터 생각해 봅시다.

$$
y = x^2
$$

사람은 쉽게 미분할 수 있습니다.

$$
\frac{dy}{dx} = 2x
$$

예를 들어 $x = 3$이면,

$$
\frac{dy}{dx} = 6
$$

입니다.

이 정도는 직접 손으로 계산할 수 있습니다.
그런데 딥러닝 모델은 상황이 완전히 다릅니다.

예를 들어 CNN이나 Transformer는 다음과 같은 특징을 가집니다.

- 파라미터 수가 수만 개, 수십만 개, 때로는 수백만 개 이상이다.
- 함수 구조가 매우 복잡하다.
- 층(layer)이 여러 개로 연결되어 있다.
- 각 파라미터가 loss에 미치는 영향을 모두 계산해야 한다.

이때 사람이 모든 가중치에 대해 미분식을 직접 만들고 계산하는 것은 사실상 불가능합니다.

그래서 PyTorch 같은 딥러닝 프레임워크는 **계산 과정을 그래프로 기록하고**,
그 그래프를 따라 **자동으로 gradient를 계산**합니다.

이것이 자동미분입니다.

즉, 자동미분이 필요한 이유는 간단합니다.

- 간단한 함수는 사람이 미분할 수 있다.
- 복잡한 모델은 사람이 직접 미분하기 어렵다.
- 따라서 컴퓨터가 계산 과정을 기억하고 자동으로 미분해야 한다.

---

#### 2. 자동미분의 기본 개념

자동미분은 다음 한 문장으로 요약할 수 있습니다.

> **자동미분은 함수의 출력이 입력이나 파라미터에 대해 얼마나 변하는지를 자동으로 계산하는 기술이다.**

여기서 중요한 표현은 **“얼마나 변하는지”** 입니다.
이것이 바로 미분값, 즉 **gradient(기울기)** 입니다.

예를 들어 함수

$$
y = 3x^2 + 2x + 1
$$

이 있을 때, 우리는 $x$가 조금 변하면 $y$가 얼마나 변하는지 알고 싶습니다.
이때 미분하면

$$
\frac{dy}{dx} = 6x + 2
$$

가 됩니다.

자동미분은 이 결과를 사람이 손으로 계산하지 않아도,
컴퓨터가 연산 과정을 추적하여 자동으로 구하게 해 줍니다.

##### 자동미분의 핵심 아이디어

자동미분은 단순히 “수치적으로 대충 미분하는 것”이 아닙니다.
또한 미리 정리된 미분 공식을 문자열처럼 외워서 적용하는 것도 아닙니다.

자동미분은 다음 두 가지를 바탕으로 합니다.

1. **계산 과정을 그래프로 저장한다.**
2. **연쇄법칙(Chain Rule)** 을 이용해 뒤에서부터 gradient를 계산한다.

즉, 자동미분은 계산 과정의 기록과 연쇄법칙의 결합입니다.

---

#### 3. 자동미분은 정확히 무엇을 미분하는가

이 부분은 학생들이 가장 자주 헷갈리는 부분입니다.특히 다음과 같은 질문이 많이 나옵니다.

- 자동미분은 함수를 미분하는 것인가?
- 미분방정식을 푸는 것인가?
- 미분방정식에서 나온 수치를 다시 미분하는 것인가?

이 질문을 정확히 정리해 봅시다.

##### 3.1 자동미분은 “함수의 출력”을 “입력 또는 파라미터”에 대해 미분한다

예를 들어 다음과 같은 함수가 있다고 합시다.

$$
y = f(x) = x^2
$$

자동미분은 여기서 $y$를 $x$에 대해 미분하여

$$
\frac{dy}{dx}
$$

를 계산합니다.

또 딥러닝에서는 보통 loss를 weight에 대해 미분합니다.

$$
L = L(w)
$$

이때 자동미분이 계산하는 것은

$$
\frac{dL}{dw}
$$

입니다.

즉, **출력이 입력이나 파라미터에 대해 얼마나 민감하게 변하는지**를 계산합니다.

---

##### 3.2 자동미분은 미분방정식을 푸는 기능이 아니다

매우 중요합니다.

> **자동미분은 미분방정식을 푸는 기능이 아니라, 계산 그래프를 따라 연쇄법칙을 적용하여 gradient를 구하는 기능이다.**

미분방정식은 보통 다음과 같은 형태입니다.

$$
\frac{dy}{dx} = 2x
$$

또는

$$
\frac{dy}{dt} = ky
$$

이런 식은 “미분 자체가 식 안에 포함된 방정식”입니다.
이런 문제는 보통 **미분방정식 풀이** 또는 **수치해석적 적분**의 대상입니다.

반면 자동미분은 이런 방정식을 푸는 것이 목적이 아닙니다.
자동미분은 이미 정의된 계산 과정, 예를 들어

$$
L = (wx - 10)^2
$$

같은 함수가 있을 때, 이 함수의 출력 $L$이 파라미터 $w$에 대해 얼마나 변하는지를 계산합니다.

즉,

- 미분방정식 풀이: 미분이 들어간 방정식을 만족하는 함수를 구하는 것
- 자동미분: 주어진 계산식의 출력에 대한 gradient를 구하는 것

입니다.

두 개념은 전혀 다릅니다.

---

##### 3.3 자동미분은 단순한 수치 차분과도 다르다

미분값을 근사하는 가장 단순한 방법은 다음과 같습니다.

$$
\frac{f(x+h) - f(x)}{h}
$$

이것은 **수치미분(numerical differentiation)** 입니다.
작은 $h$를 넣어서 기울기를 대략 계산하는 방법입니다.

하지만 자동미분은 이와 다릅니다.

- 수치미분: 근사값
- 자동미분: 계산 그래프를 이용한 정확한 gradient 계산(부동소수점 오차 제외)

즉, 자동미분은 “대충 계산하는 방법”이 아니라,
**계산 그래프를 기반으로 체계적으로 gradient를 계산하는 방법**입니다.

---

#### 4. Computational Graph와 `requires_grad`

자동미분을 이해하려면 **계산 그래프(Computational Graph)** 를 이해해야 합니다.

##### 4.1 계산 그래프란 무엇인가

계산 그래프는 연산 과정을 노드와 간선으로 표현한 것입니다.

예를 들어 다음 식을 생각해 봅시다.

`y = 3x^2 + 2x + 1`

이 식은 한 번에 계산되는 것이 아니라, 내부적으로 여러 단계의 연산으로 나눌 수 있습니다.

- `x^2`를 계산한다.
- `3 * x^2`를 계산한다.
- `2 * x`를 계산한다.
- 앞의 두 결과를 더한다.
- 마지막으로 `1`을 더한다.

이런 계산 흐름이 그래프로 연결됩니다.

간단히 그리면 다음과 같은 느낌입니다.

```text
x
↓
x**2
↓
3*x**2      2*x
   ↓         ↓
      더하기
         ↓
       + 1
         ↓
         y
```

PyTorch는 `forward`를 수행할 때 단순히 숫자만 계산하지 않습니다.
동시에 **“이 값이 어떤 연산을 통해 만들어졌는가”** 를 기억합니다.

이것이 바로 계산 그래프입니다.

---

##### 4.2 forward는 계산만 하는 것이 아니다

초급 학습자들이 자주 놓치는 점이 있습니다.

> **Forward는 단순 계산만 하는 것이 아니라, 미분에 필요한 계산 그래프를 함께 저장하는 과정이다.**

예를 들어 다음을 보겠습니다.

- 입력: $x = 2$
- 가중치: $w = 3$
- 출력: $y = wx$
- 손실: $loss = (y - 10)^2$

Forward에서는 다음 계산이 일어납니다.

$$
y = w \cdot x
$$

$$
loss = (y - 10)^2
$$

그런데 PyTorch는 단순히 `loss = 16`이라는 숫자만 얻는 것이 아닙니다.
동시에 다음과 같은 계산 관계를 기억합니다.

```text
loss
↑
(y - 10)^2
↑
y = w*x
↑
w, x
```

이 기록 덕분에 나중에 `backward()`를 호출했을 때
loss에서 시작해서 거꾸로 올라가며 gradient를 계산할 수 있습니다.

---

##### 4.3 `requires_grad`란 무엇인가

PyTorch에서 어떤 텐서에 대해 gradient를 추적하려면 `requires_grad=True`로 설정해야 합니다.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
print(x.requires_grad)
```

출력:

```python
True
```

이 설정의 의미는 다음과 같습니다.

- 이 텐서는 학습 대상이거나,
- gradient를 계산해야 하므로,
- 연산 과정을 추적하라는 뜻입니다.

반대로 `requires_grad=False`이면 PyTorch는 그 텐서를 미분 대상으로 보지 않습니다.

---

##### 4.4 가장 쉬운 예제

```python
import torch

x = torch.tensor(3.0, requires_grad=True)

y = x**2
print(y)

y.backward()

print(x.grad)
```

출력 결과는 다음과 같습니다.

```python
tensor(9., grad_fn=<PowBackward0>)
tensor(6.)
```

왜 `6`일까요?

$$
y = x^2
$$

이므로

$$
\frac{dy}{dx} = 2x
$$

이고, $x=3$이므로

$$
2 \times 3 = 6
$$

입니다.

즉, `x.grad`에는 $\frac{dy}{dx}$ 값이 저장됩니다.

---

##### 4.5 조금 확장한 예제

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = 3*x**2 + 2*x + 1
y.backward()

print(x.grad)
```

직접 미분하면

$$
y = 3x^2 + 2x + 1
$$

$$
\frac{dy}{dx} = 6x + 2
$$

이므로 $x=2$일 때

$$
6 \times 2 + 2 = 14
$$

입니다.

PyTorch의 출력도 `14`가 나와야 합니다.

이 예제는 학생들에게 매우 좋은 연습이 됩니다.
먼저 손으로 미분한 뒤, PyTorch 결과와 비교하면 자동미분의 의미가 훨씬 분명해집니다.

---

#### 5. `backward()`는 무엇을 하는가

이제 자동미분의 핵심 명령인 `backward()`를 보겠습니다.

많은 학생들이 `backward()`를 “loss를 줄이는 함수”처럼 오해합니다.하지만 실제 역할은 훨씬 더 정확하게 말해야 합니다.

> `backward()`는 손실함수를 각 파라미터에 대해 미분하여 gradient를 계산하고, 그 값을 `.grad`에 저장한다.

즉, `backward()`는 **미분을 수행하는 함수**입니다.
아직 파라미터를 바꾸지는 않습니다.

---

##### 5.1 `backward()`는 언제 무엇을 계산하는가

예를 들어 다음 상황을 생각해 봅시다.

- 입력 $x=2$
- 가중치 $w=3$
- 출력 $y=wx$
- 손실 $loss=(y-10)^2$

코드로 쓰면:

```python
import torch

x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)

y = w * x
loss = (y - 10)**2

print(loss)   # 16
```

현재 값은 다음과 같습니다.

- $x = 2$
- $w = 3$
- $y = 6$
- $loss = (6 - 10)^2 = 16$

여기서 `loss.backward()`를 호출하면 PyTorch는 계산 그래프를 거꾸로 따라 올라가며

$$
\frac{d(loss)}{dw}
$$

를 계산합니다.

---

##### 5.2 연쇄법칙으로 직접 계산해 보기

손으로도 계산해 봅시다.

$$
y = wx
$$

$$
loss = (y - 10)^2
$$

우리가 구하고 싶은 것은

$$
\frac{d(loss)}{dw}
$$

입니다.

연쇄법칙을 적용하면

$$
\frac{d(loss)}{dw}
=
\frac{d(loss)}{dy}
\times
\frac{dy}{dw}
$$

먼저,

$$
loss = (y - 10)^2
$$

이므로

$$
\frac{d(loss)}{dy} = 2(y - 10)
$$

또,

$$
y = wx
$$

이므로

$$
\frac{dy}{dw} = x
$$

따라서

$$
\frac{d(loss)}{dw} = 2(y - 10)x
$$

입니다.

이제 숫자를 대입해 봅시다.

- $y = 6$
- $x = 2$

이므로

$$
\frac{d(loss)}{dw} = 2(6 - 10)\times 2 = 2(-4)\times2 = -16
$$

즉, gradient는 `-16`입니다.

PyTorch로 확인하면:

```python
import torch

x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)

y = w * x
loss = (y - 10)**2

loss.backward()

print(w.grad)
```

출력:

```python
tensor(-16.)
```

정확히 일치합니다.

---

##### 5.3 `backward()`의 핵심 정리

`backward()`를 한 줄로 정리하면 다음과 같습니다.

- loss부터 시작한다.
- 계산 그래프를 거꾸로 따라간다.
- 각 연산의 미분값을 연쇄법칙으로 연결한다.
- 최종적으로 각 파라미터의 gradient를 구한다.
- 그 값을 `.grad`에 저장한다.

즉,

> **backward는 계산, step은 수정**

입니다.

이 문장은 꼭 기억해 두어야 합니다.

---

#### 6. 역전파법과 자동미분의 관계

이 둘은 비슷하게 들리지만 완전히 같은 말은 아닙니다.

##### 6.1 역전파법이란 무엇인가

역전파법(Backpropagation)은 신경망 학습에서
**출력층에서 입력 방향으로 거꾸로 gradient를 전달하는 절차**를 말합니다.

즉, 신경망의 구조 전체를 대상으로 한 학습 알고리즘의 흐름입니다.

##### 6.2 자동미분은 무엇인가

자동미분은 그 역전파 과정 안에서 실제 gradient를 계산하는 메커니즘입니다.

즉,

- **역전파**: 신경망에서 gradient를 뒤로 전달하는 학습 절차의 이름
- **자동미분**: 그 과정에서 각 연산의 gradient를 실제 계산하는 기술

이라고 이해하면 됩니다.

##### 6.3 둘의 관계를 한 문장으로 정리

> **역전파는 신경망 학습 절차의 이름이고, 자동미분은 그 안에서 gradient를 실제 계산하는 메커니즘이다.**

즉, 딥러닝에서 역전파를 수행할 때 자동미분이 내부적으로 사용됩니다.

---

##### 6.4 왜 뒤에서부터 계산하는가

신경망의 출력은 수많은 연산의 조합입니다.
앞에서부터 계산한 값을 다시 뒤집어 올라가며 미분하면,
연쇄법칙을 매우 효율적으로 적용할 수 있습니다.

예를 들어

```text
x → layer1 → layer2 → layer3 → loss
```

라는 구조가 있다면, 역전파는

```text
loss → layer3 → layer2 → layer1
```

방향으로 이동하면서 gradient를 계산합니다.

이때 각 층은 자기 앞 단계에 gradient를 넘겨 줍니다.
이것이 바로 역전파 내부에서 자동미분이 동작하는 방식입니다.

---

#### 7. `Optimizer.step()`과의 연결

자동미분의 계산 결과는 결국 파라미터를 업데이트하는 데 사용됩니다.

전체 흐름은 다음과 같습니다.

```text
forward
↓
loss 계산
↓
backward()
↓
gradient 생성
↓
optimizer.step()
↓
weight 수정
```

여기서 가장 중요한 점은 다음입니다.

- `backward()`는 **미분만 한다**
- `optimizer.step()`은 **실제로 값을 바꾼다**

---

##### 7.1 표로 정리하기

| 함수                 | 역할                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `loss.backward()`  | 각 파라미터에 대해$\partial Loss / \partial Weight$를 계산하여 `.grad`에 저장한다. 가중치는 아직 바뀌지 않는다. |
| `optimizer.step()` | `.grad`에 저장된 gradient를 사용하여 실제 가중치를 업데이트한다.                                                  |

이 차이는 반드시 구분해야 합니다.

> **backward는 계산, step은 수정**

---

##### 7.2 SGD의 예

가장 기본적인 SGD는 다음 공식을 사용합니다.

$$
\text{새 weight} = \text{기존 weight} - \text{learning rate} \times \text{gradient}
$$

앞의 예제에서 gradient가 `-16`이고, learning rate가 `0.1`이라고 합시다.

그러면

$$
\text{새 weight} = 3 - 0.1 \times (-16)
$$

$$
= 3 + 1.6 = 4.6
$$

즉, 가중치는 `3`에서 `4.6`으로 바뀝니다.

중요한 점은 이 변화가 `backward()`에서 일어나는 것이 아니라,
`optimizer.step()`에서 일어난다는 것입니다.

---

#### 8. 예제와 실습

이제 직접 손으로 계산하고, PyTorch로 확인해 봅시다.

---

##### 8.1 실습 1

다음을 계산하세요.

$$
y = x^3,\quad x = 2
$$

###### 직접 계산

$$
\frac{dy}{dx} = 3x^2
$$

따라서 $x=2$일 때

$$
3 \times 2^2 = 12
$$

정답: **12**

---

##### 8.2 실습 2

다음을 PyTorch로 확인하세요.

$$
y = 5x^2 + 3x,\quad x = 4
$$

###### 직접 계산

$$
\frac{dy}{dx} = 10x + 3
$$

따라서 $x=4$이면

$$
10 \times 4 + 3 = 43
$$

###### PyTorch 코드

```python
import torch

x = torch.tensor(4.0, requires_grad=True)
y = 5*x**2 + 3*x

y.backward()

print(x.grad)
```

출력:

```python
tensor(43.)
```

---

##### 8.3 실습 3

다음 모델을 만드세요.

- `Linear(1,1)`
- weight를 출력하고
- `requires_grad`를 확인하세요.

```python
import torch

layer = torch.nn.Linear(1, 1)

print(layer.weight)
print(layer.weight.requires_grad)
```

출력 예:

```python
Parameter containing:
tensor([[0.5271]], requires_grad=True)
True
```

여기서 꼭 확인해야 할 점은 다음입니다.

- `Linear`의 weight도 결국 **Tensor**
- 더 정확히는 학습 가능한 **Parameter**
- 그리고 `requires_grad=True`

즉, 신경망의 가중치는 자동미분의 대상입니다.

---

##### 8.4 가장 쉬운 자동미분 예제 다시 보기

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x**2

y.backward()

print("y =", y.item())
print("dy/dx =", x.grad.item())
```

출력:

```python
y = 9.0
dy/dx = 6.0
```

---

##### 8.5 다항식 예제

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = 3*x**2 + 2*x + 1

y.backward()

print("gradient =", x.grad.item())
```

직접 계산한 값 $6x+2$에 $x=2$를 넣으면 `14`이므로,
PyTorch 결과와 비교해 보세요.

---

##### 8.6 종합 실습: weight가 실제로 바뀌는지 확인하기

이 실습은 학생들이 가장 흥미로워하는 실습입니다.
정말로 `optimizer.step()` 이후 weight가 바뀌는지 눈으로 확인할 수 있습니다.

###### 목표

- 입력 $x=3$
- `Linear(1,1)` 통과
- loss 계산
- `backward()`
- gradient 출력
- `optimizer.step()` 전후의 weight 출력

###### 코드

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 입력과 정답
x = torch.tensor([[3.0]])
target = torch.tensor([[10.0]])

# 모델
model = nn.Linear(1, 1)

# 옵티마이저
optimizer = optim.SGD(model.parameters(), lr=0.1)

# step 전 weight 확인
print("Before step")
print("weight =", model.weight.data)
print("bias   =", model.bias.data)

# forward
output = model(x)
loss = (output - target).pow(2).mean()

print("output =", output.item())
print("loss   =", loss.item())

# backward
optimizer.zero_grad()
loss.backward()

print("\nGradient")
print("weight.grad =", model.weight.grad)
print("bias.grad   =", model.bias.grad)

# step
optimizer.step()

# step 후 weight 확인
print("\nAfter step")
print("weight =", model.weight.data)
print("bias   =", model.bias.data)
```

###### 관찰 포인트

1. `forward`에서 출력과 loss가 계산된다.
2. `loss.backward()`를 하면 gradient가 생긴다.
3. 이때 weight 값은 아직 그대로다.
4. `optimizer.step()`을 실행하면 weight가 실제로 바뀐다.

이 실습을 통해 학생들은 다음 사실을 직접 확인하게 됩니다.

- gradient는 계산된 값이다.
- gradient 자체가 weight는 아니다.
- weight를 바꾸는 주체는 optimizer다.

---

#### 9. 자주 헷갈리는 질문

##### Q1. 자동미분은 수식을 상징적으로 미분하는 것인가요?

아닙니다.
자동미분은 수학 소프트웨어처럼 일반적인 수식을 기호적으로 정리하는 것이 아닙니다.

자동미분은 **실제로 수행된 연산 기록**을 바탕으로 gradient를 계산합니다.
즉, 계산 그래프를 따라가며 연쇄법칙을 적용하는 방식입니다.

---

##### Q2. 자동미분은 미분방정식을 푸는 기능인가요?

아닙니다.

> **자동미분은 미분방정식을 푸는 기능이 아니다.**

자동미분이 하는 일은, 어떤 계산식의 출력이 입력이나 파라미터에 대해 얼마나 변하는지를 구하는 것입니다.

예를 들어

$$
L = (wx - 10)^2
$$

가 있을 때,

$$
\frac{dL}{dw}
$$

를 구하는 것이 자동미분입니다.

반면 미분방정식 풀이란

$$
\frac{dy}{dt} = ky
$$

와 같은 식을 만족하는 함수 $y(t)$를 찾는 문제입니다.
둘은 문제의 성격이 다릅니다.

---

##### Q3. `backward()`를 하면 weight가 자동으로 바뀌나요?

아닙니다.

`backward()`는 gradient만 계산해서 `.grad`에 저장합니다.
실제 weight 수정은 `optimizer.step()`이 수행합니다.

다시 강조하면:

> **backward는 계산, step은 수정**

---

##### Q4. Forward는 그냥 출력만 계산하는 단계 아닌가요?

아닙니다.

Forward는 출력과 loss를 계산하는 동시에,
나중에 미분하기 위해 **계산 그래프를 저장**하는 단계이기도 합니다.

즉, forward는 단순 계산이 아니라
“미분 준비까지 포함한 계산”입니다.

---

##### Q5. 모든 Tensor가 자동으로 미분되나요?

아닙니다.
`requires_grad=True`인 텐서만 gradient 추적 대상이 됩니다.

예를 들어 학습 가능한 weight는 보통 자동으로 `requires_grad=True`입니다.
하지만 입력 데이터나 상수는 그렇지 않을 수 있습니다.

---

##### Q6. `Linear`의 weight는 왜 Tensor인가요?

PyTorch의 신경망 파라미터는 결국 모두 Tensor입니다.
다만 학습 가능한 텐서로 관리되기 위해 `Parameter` 형태로 저장됩니다.

즉,

- weight는 Tensor이고
- 자동미분 대상이며
- gradient가 계산되고
- optimizer가 업데이트합니다.

---

##### Q7. gradient 값이 음수이면 무슨 뜻인가요?

gradient가 음수라는 것은,
해당 파라미터를 조금 증가시켰을 때 loss가 감소하는 방향일 수 있다는 뜻입니다.

예를 들어 gradient가 `-16`이라면 SGD 업데이트는

$$
w = w - lr \times (-16)
$$

이 되어, 결과적으로 $w$가 증가합니다.

즉, optimizer는 gradient의 부호와 크기를 보고
loss를 줄이는 방향으로 이동합니다.

---

##### Q8. 왜 loss에 대해 미분하나요?

딥러닝 학습의 목표는 보통 **loss를 최소화하는 것**입니다.
따라서 각 weight가 loss에 얼마나 영향을 주는지 알아야 합니다.

즉, 궁금한 것은

$$
\frac{\partial Loss}{\partial Weight}
$$

입니다.

이 값이 있어야 optimizer가
“어느 방향으로 얼마나 weight를 움직여야 하는가”를 결정할 수 있습니다.

---

#### 10. 정리

이 장의 핵심을 다시 정리해 봅시다.

##### 핵심 개념 1: 자동미분의 의미

> **자동미분은 함수의 출력이 입력이나 파라미터에 대해 얼마나 변하는지를 자동으로 계산하는 기술이다.**

즉, 복잡한 계산식에 대해 gradient를 자동으로 구해 주는 기능입니다.

---

##### 핵심 개념 2: 자동미분은 무엇을 미분하는가

자동미분은 **주어진 함수의 출력**을 **입력이나 파라미터**에 대해 미분합니다.

예를 들어:

- $y = x^2$에서 $\frac{dy}{dx}$
- $L = (wx - 10)^2$에서 $\frac{dL}{dw}$

를 계산합니다.

그리고 매우 중요하게,

> **자동미분은 미분방정식을 푸는 기능이 아니라, 계산 그래프를 따라 연쇄법칙을 적용하여 gradient를 구하는 기능이다.**

---

##### 핵심 개념 3: 계산 그래프와 `requires_grad`

- `forward`를 수행하면 PyTorch는 계산 그래프를 만든다.
- `requires_grad=True`인 텐서는 gradient 추적 대상이 된다.
- `backward()`는 그래프를 거꾸로 따라가며 gradient를 계산한다.

---

##### 핵심 개념 4: 역전파와 자동미분의 관계

- 역전파는 신경망 학습 절차의 이름이다.
- 자동미분은 그 안에서 gradient를 실제 계산하는 메커니즘이다.

즉, 역전파 내부에서 자동미분이 작동한다고 이해하면 됩니다.

---

##### 핵심 개념 5: `backward()`와 `optimizer.step()`

이 둘은 반드시 구분해야 합니다.

- `backward()` : gradient 계산
- `optimizer.step()` : weight 수정

한 문장으로 외우면:

> **backward는 계산, step은 수정**

---

자동미분을 이해하면 PyTorch 코드가 더 이상 마법처럼 보이지 않습니다.
`loss.backward()`가 왜 필요한지, `optimizer.step()`이 왜 따로 존재하는지,
그리고 신경망의 weight가 어떻게 학습되는지가 한 흐름으로 연결되기 시작합니다.

이 절의 내용을 확실히 이해했다면, 앞으로 만나는 모든 딥러닝 모델에서
**학습이 실제로 어떻게 이루어지는가**를 훨씬 더 깊이 이해할 수 있을 것입니다.

---

#### 1.3.1 Tensor와 shape

PyTorch에서 대부분의 데이터는 tensor로 표현된다. 숫자, 벡터, 이미지, 시계열, 로봇 상태, 액션, 보상 모두 tensor 형식으로 다룬다.

로봇 AI에서 자주 사용하는 shape는 다음과 같다.

- 벡터형 상태
  - 단일 상태: `[D]`
  - 배치 상태: `[B, D]`
- 이미지
  - 단일 이미지: `[C, H, W]`
  - 배치 이미지: `[B, C, H, W]`
- 시퀀스
  - 단일 시퀀스: `[T, D]`
  - 배치 시퀀스: `[B, T, D]`

예를 들어 15차원 상태를 배치 크기 32로 묶으면 shape는 `[32, 15]`가 된다. RGB 이미지 224×224를 16장 배치로 묶으면 `[16, 3, 224, 224]`가 된다. 최근 10스텝의 상태 시퀀스를 배치 32개로 묶으면 `[32, 10, 15]`가 된다.

이러한 shape 해석은 특히 transformer policy에서 중요하다. 차원 하나를 잘못 해석하면 모델 입력 전체가 맞지 않게 된다.

간단한 로봇 예시는 다음과 같다.

```python
# joint state → action
state.shape   # [B, 13]
action.shape  # [B, 7]

# image + state → action
image.shape   # [B, 3, 224, 224]
state.shape   # [B, 13]
action.shape  # [B, 7]

# sequence → next action
state_seq.shape  # [B, T, D]
action.shape     # [B, A]
```

#### 1.3.2 `nn.Linear`: 가장 단순한 정책의 시작점

`nn.Linear`는 입력을 받아 다른 차원의 출력으로 변환하는 가장 기본적인 신경망 레이어이다. 로봇 관점에서 가장 단순한 정책은 다음과 같이 표현할 수 있다.

> 현재 상태를 입력받아 다음 액션을 출력하는 함수

예를 들어 입력 상태가 13차원이고 액션이 7차원이라면 다음과 같이 정의할 수 있다.

```python
policy = nn.Linear(13, 7)
```

실제 구현에서는 보통 비선형층을 추가한 MLP를 사용한다.

```python
policy = nn.Sequential(
    nn.Linear(13, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 7)
)
```

이 구조는 이후의 MLP imitation learning과 직접 연결된다. `101_pytorch_mnist.py`의 MLP 구조도 같은 관점에서 읽을 수 있다. 차이는 이미지 분류를 하느냐, 상태에서 액션을 예측하느냐에 있다.

#### 1.3.3 `Dataset`과 `DataLoader`

딥러닝 학습에서는 전체 데이터를 한 번에 넣기보다 batch 단위로 나누어 학습하는 것이 일반적이다. PyTorch에서는 `Dataset`과 `DataLoader`가 이 역할을 담당한다.

- `Dataset`: 샘플 하나를 어떻게 읽을지 정의
- `DataLoader`: 샘플을 batch로 묶고, 섞고, 반복해서 제공

예를 들어 로봇 시연 데이터의 한 샘플이 다음과 같다고 하자.

```python
{
    "state": 현재 관절 상태,
    "image": 카메라 이미지,
    "action": 시연 액션
}
```

`DataLoader`는 이를 다음과 같은 배치 형태로 구성한다.

```python
state_batch.shape   # [32, 13]
image_batch.shape   # [32, 3, 224, 224]
action_batch.shape  # [32, 7]
```

로봇 분야에서는 데이터 형식이 다양하다.

- CSV 기반 상태-액션 로그
- 이미지 폴더와 메타데이터
- episode 단위 시퀀스
- HDF5, numpy, rosbag 등

따라서 `Dataset`과 `DataLoader`를 이해하는 것은 단순 문법 습득이 아니라, 로봇 데이터를 학습 가능한 형태로 구성하는 능력과 연결된다.

#### 1.3.4 Train loop와 학습 모드

PyTorch 학습의 핵심은 train loop이다.

```python
for epoch in range(num_epochs):
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

각 줄의 의미는 다음과 같다.

- `model.train()`: 모델을 학습 모드로 전환한다.
- `pred = model(x)`: 입력에 대한 예측을 계산한다.
- `loss = loss_fn(pred, y)`: 예측과 정답의 차이를 계산한다.
- `optimizer.zero_grad()`: 이전 step의 gradient를 초기화한다.
- `loss.backward()`: 역전파를 수행하여 gradient를 계산한다.
- `optimizer.step()`: 계산된 gradient를 사용해 파라미터를 업데이트한다.

로봇 imitation learning에서는 이를 다음과 같이 볼 수 있다.

```python
for batch in loader:
    obs, action_gt = batch
    action_pred = policy(obs)
    loss = mse(action_pred, action_gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

즉, 정책이 시연자의 행동을 더 잘 재현하도록 파라미터를 업데이트하는 과정이다.

학습과 추론은 구분해야 한다. 추론 시에는 일반적으로 `eval()`과 `torch.no_grad()`를 함께 사용한다.

```python
model.eval()
with torch.no_grad():
    output = model(x)
```

이렇게 하면 메모리 사용량을 줄이고, 불필요한 gradient 계산을 방지할 수 있다. 실시간 로봇 제어 단계에서도 추론은 대체로 이 방식으로 수행한다.

#### 1.3.5 GPU, device, VRAM, batch size

작은 예제는 CPU로도 실행할 수 있지만, 로봇 AI에서는 GPU 사용이 빠르게 중요해진다. 특히 다음과 같은 경우 GPU가 사실상 필요하다.

- 이미지 입력 사용
- CNN encoder 사용
- Transformer policy 사용
- 큰 batch 사용
- diffusion policy, world model 등 대형 모델 사용

PyTorch에서는 모델과 데이터가 같은 장치에 있어야 한다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
y = y.to(device)
```

입문 단계에서 자주 발생하는 오류가 `device mismatch`이다. 예를 들어 모델은 GPU에 있고 입력 데이터는 CPU에 있으면 연산을 수행할 수 없다.

`105_pytorch_Resnet_transfer_fine.py`는 GPU와 메모리 문제를 이해하는 데 좋은 예제이다. 입력 이미지 크기가 224이고 사전학습 ResNet18을 사용하므로, MLP 예제보다 계산량과 메모리 사용량이 크다. 여기서 자연스럽게 다음 관계를 확인할 수 있다.

- 모델이 클수록 VRAM 사용량이 증가한다.
- 이미지 해상도가 커질수록 VRAM 사용량이 증가한다.
- 시퀀스 길이 `T`가 길수록 메모리 사용량이 증가한다.
- batch size가 커질수록 메모리 사용량이 증가한다.

OOM(Out of Memory)이 발생하면 일반적으로 다음 항목을 먼저 조정한다.

- `batch_size` 줄이기
- 이미지 해상도 줄이기
- sequence 길이 줄이기
- 더 작은 모델 사용
- 추론 시 `torch.no_grad()` 사용

정리하면, `tensor`, `nn.Linear`, `Dataset / DataLoader`, train loop, GPU는 서로 분리된 기능 목록이 아니라 하나의 학습 파이프라인을 구성하는 연결된 요소이다.


---

## 1.4 회귀에서 로봇 정책 학습까지

> 이 절의 회귀·로봇 샘플은 개념을 익히는 독립 코드 조각이다. 현재 위치→다음 위치 예제는 상태 예측이며, 실제 5~6장의 BC는 현재 관측→현재 적용할 목표 위치 명령을 학습한다. `01_02_pytorch_basic/`의 파일은 분류 예제이므로 이미 로봇 정책을 구현한 파일로 해석하지 않는다.

### 1.4 회귀에서 로봇 정책 학습까지

이 절에서는 회귀는 **Behavior Cloning**을 이해하고, 더 나아가 **ACT(Action Chunking Transformer)** 를 이해하기 위한 핵심 징검다리 임을 이해합니다. 

많은 입문서가 다음 순서를 따릅니다.

```text
Linear Regression
↓
Logistic Regression
↓
CNN
```

하지만 이 책의 최종 목적은 다릅니다.

```text
Behavior Cloning
↓
ACT
↓
Robot Learning
```

따라서 이 장에서는 “회귀”를 **분류 이전의 일반적인 머신러닝 주제**로 다루지 않고,
**연속적인 로봇 행동(action)을 예측하는 관점**에서 설명합니다.

---

#### 학습목표

이 챕터를 마치면 다음 내용을 이해할 수 있습니다.

- 회귀와 분류의 차이를 설명할 수 있다.
- `Linear Layer`가 연속적인 값을 예측하는 원리를 이해할 수 있다.
- `MSELoss`를 왜 사용하는지 설명할 수 있다.
- MLP가 비선형 함수를 근사하는 이유를 이해할 수 있다.
- 로봇의 관절각, 속도, 토크, 그리퍼 개방률이 모두 회귀 문제임을 이해할 수 있다.
- Behavior Cloning이 관측으로부터 행동을 예측하는 회귀 문제임을 설명할 수 있다.
- ACT 역시 연속적인 action chunk를 예측하는 회귀 문제로 볼 수 있음을 이해할 수 있다.

---

#### Section 1. 분류와 회귀의 차이

머신러닝을 처음 배우면 많은 학생들이 다음처럼 생각하기 쉽습니다.

> “AI는 사진을 보고 고양이인지 개인지 맞히는 것 아닌가?”

물론 그런 문제도 중요합니다. 하지만 **로봇 제어**의 관점에서는 다릅니다.
로봇은 대부분 “클래스를 맞히는 문제”가 아니라, **연속적인 값을 예측하는 문제**를 풀어야 합니다.

##### 1.1 분류와 회귀 비교

| 문제                    | 출력        | 종류 |
| ----------------------- | ----------- | ---- |
| 손글씨 숫자 인식        | 0~9 중 하나 | 분류 |
| 고양이 / 개 구분        | 클래스 라벨 | 분류 |
| 사람 키 예측            | 173.2 cm    | 회귀 |
| 자동차 속도 예측        | 58.4 km/h   | 회귀 |
| 로봇 관절각 예측        | 32.7°      | 회귀 |
| 로봇 그리퍼 개방률 예측 | 0.63        | 회귀 |

분류는 보통 **정해진 카테고리 중 하나**를 선택합니다.
반면 회귀는 **연속적인 수치값**을 예측합니다.

예를 들어 다음 두 문제를 비교해 보겠습니다.

- **분류 문제**: “이 이미지는 컵인가, 병인가?”
- **회귀 문제**: “이 상태에서 로봇 관절 6개의 목표 각도는 얼마인가?”

첫 번째는 클래스 선택입니다.
두 번째는 실수(real number) 6개를 출력해야 하므로 회귀입니다.

##### 1.2 왜 이 책에서는 회귀가 더 중요한가?

이 책의 목표는 이미지 분류 모델을 만드는 것이 아니라,
**로봇이 관측을 바탕으로 적절한 행동을 내도록 학습하는 것**입니다.

로봇의 행동은 보통 다음과 같이 표현됩니다.

- 관절각(joint angle)
- 관절 속도(joint velocity)
- 토크(torque)
- 말단 위치 변화량(delta pose)
- 그리퍼 개방률(gripper openness)

이 값들은 대부분 **연속값**입니다.

> **이미테이션 러닝은 대부분 “정답 클래스를 맞히는 문제”가 아니라 “연속적인 행동(Action)을 예측하는 회귀 문제”이다.**

이 문장은 이 장 전체를 관통하는 핵심 메시지입니다.
그리고 이 관점이 있어야 이후의 **Behavior Cloning**, **ACT**, **Robot Learning**이 자연스럽게 연결됩니다.

---

#### Section 2. 가장 간단한 회귀

회귀의 가장 기본 형태는 1차 함수입니다.

$$
y = ax + b
$$

여기서

- $a$: 기울기
- $b$: 절편

입니다.

##### 2.1 PyTorch의 `Linear(1,1)`은 무엇을 의미할까?

PyTorch의 가장 기본적인 선형 레이어는 다음과 같습니다.

```python
nn.Linear(1, 1)
```

이 의미는 다음과 같습니다.

- 입력 차원: 1개
- 출력 차원: 1개

수식으로 쓰면 다음과 같습니다.

$$
\hat{y} = wx + b
$$

즉,

- `weight = w`
- `bias = b`

이고, 이것은 곧 우리가 배운

$$
y = ax + b
$$

와 같은 형태입니다.

즉, `Linear(1,1)`은 **1차 방정식을 학습하는 층**이라고 생각할 수 있습니다.

##### 2.2 예제 데이터

다음과 같은 간단한 데이터를 생각해 봅시다.

| x | y |
| - | - |
| 0 | 1 |
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |

이 데이터는 정확히 다음 규칙을 따릅니다.

$$
y = 2x + 1
$$

우리는 모델이 이 규칙을 데이터로부터 스스로 찾아내도록 학습시킬 수 있습니다.

---

##### 예제 1. 직선 회귀: `y = 2x + 1`

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 데이터 준비
x = torch.tensor([[0.0],
                  [1.0],
                  [2.0],
                  [3.0],
                  [4.0]])

y = torch.tensor([[1.0],
                  [3.0],
                  [5.0],
                  [7.0],
                  [9.0]])

# 2. 모델 정의: Linear(1,1) = y = wx + b
model = nn.Linear(1, 1)

# 3. 손실 함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 4. 학습 루프
for epoch in range(500):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# 5. 학습된 파라미터 확인
weight = model.weight.item()
bias = model.bias.item()

print(f"Learned weight: {weight:.4f}")
print(f"Learned bias  : {bias:.4f}")

# 6. 예측 확인
test_x = torch.tensor([[5.0]])
test_y = model(test_x)
print(f"x=5.0 -> predicted y={test_y.item():.4f}")
```

###### 실행 결과 예시

```text
Epoch   0 | Loss: 18.742931
Epoch 100 | Loss: 0.015284
Epoch 200 | Loss: 0.000121
Epoch 300 | Loss: 0.000001
Epoch 400 | Loss: 0.000000
Learned weight: 1.9998
Learned bias  : 1.0004
x=5.0 -> predicted y=10.9994
```

> 위 숫자는 실행 환경과 초기값에 따라 조금씩 달라질 수 있지만, 학습이 잘 되면 `weight`는 2에, `bias`는 1에 가까워집니다.

##### 2.3 왜 `MSELoss`를 사용할까?

회귀에서는 보통 예측값 $\hat{y}$ 와 정답 $y$ 의 차이를 작게 만드는 것이 목표입니다.
가장 대표적인 손실 함수가 **MSE(Mean Squared Error)** 입니다.

$$
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

제곱을 하는 이유는 다음과 같습니다.

1. 오차가 클수록 더 큰 패널티를 주기 위해
2. 양수/음수 오차가 서로 상쇄되지 않게 하기 위해
3. 미분이 가능해 최적화에 유리하기 때문에

##### 2.4 학습 결과 해석

학습이 잘 되었다면 다음과 비슷한 결과가 나옵니다.

```text
Learned weight: 1.9998
Learned bias  : 1.0004
```

이 값은 우리가 알고 있는 정답

$$
y = 2x + 1
$$

과 거의 같습니다.

즉, 모델은 단순히 숫자를 외운 것이 아니라,
데이터 속의 **선형 관계**를 학습한 것입니다.

---

#### Section 3. 왜 MLP가 필요한가?

이제 중요한 질문을 해봅시다.

> 직선 하나만으로 모든 함수를 표현할 수 있을까요?

답은 **아니오**입니다.

예를 들어 다음 함수를 생각해 보겠습니다.

$$
y = x^2
$$

이 함수는 포물선입니다.
직선 하나로는 이 곡선을 정확히 표현할 수 없습니다.

##### 3.1 선형 모델의 한계

`Linear(1,1)`은 다음 형태만 표현할 수 있습니다.

$$
\hat{y} = wx + b
$$

즉, 항상 **직선**입니다.
입력이 아무리 많아져도 선형 조합만으로는 복잡한 곡선을 표현하는 데 한계가 있습니다.

##### 3.2 Hidden Layer가 필요한 이유

비선형 관계를 배우기 위해서는 중간에 **은닉층(hidden layer)** 과 **비선형 활성화 함수**가 필요합니다.

가장 간단한 구조는 다음과 같습니다.

```text
Linear
↓
ReLU
↓
Linear
```

이 구조를 흔히 **MLP(Multi-Layer Perceptron)** 라고 부릅니다.

##### 3.3 `ReLU`의 역할

ReLU는 다음과 같은 함수입니다.

$$
\text{ReLU}(x) = \max(0, x)
$$

이 함수가 들어가면 모델은 단순한 직선 하나가 아니라,
여러 조각의 선형 함수를 조합한 더 복잡한 함수를 만들 수 있습니다.

직관적으로 생각하면,

- `Linear`: 입력을 어떤 방향으로 섞고
- `ReLU`: 비선형적으로 꺾어 주고
- 다시 `Linear`: 원하는 출력 형태로 결합

하는 과정입니다.

##### 3.4 MLP는 왜 비선형 함수를 근사할 수 있을까?

MLP는 많은 은닉 뉴런을 사용하면, 곡선을 **잘게 나눈 선분들의 조합**처럼 근사할 수 있습니다.완벽한 수학적 증명은 더 뒤에서 다루더라도, 지금은 다음 직관이 중요합니다.

> **선형 층만 있으면 직선밖에 못 만들지만, 선형 층 사이에 비선형 함수가 들어가면 곡선을 만들 수 있다.**

---

##### 예제 2. `y = x^2` 를 MLP로 근사하기

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 생성
x = torch.linspace(-2, 2, 100).unsqueeze(1)
y = x ** 2

# MLP 모델
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(2000):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
```

###### 실행 결과 예시

```text
Epoch    0 | Loss: 2.184731
Epoch  400 | Loss: 0.041592
Epoch  800 | Loss: 0.012847
Epoch 1200 | Loss: 0.005238
Epoch 1600 | Loss: 0.002116
```

loss가 점점 감소하는 것을 보면, MLP가 직선으로는 표현하기 어려운 $x^2$ 곡선을 점차 근사하고 있음을 확인할 수 있습니다.

이 예제의 핵심은 다음과 같습니다.

- 선형 회귀만으로는 $x^2$ 를 표현하기 어렵다.
- MLP는 비선형성을 도입하여 곡선 형태를 근사할 수 있다.
- 즉, **회귀 모델은 직선에서 멈추지 않고 곡선으로 확장될 수 있다.**

---

#### Section 4. `sin` 함수 근사

학생들이 MLP의 힘을 가장 직관적으로 느끼는 예제 중 하나는 `sin` 함수입니다.

$$
y = \sin(x)
$$

이 함수는 단순한 포물선보다도 더 복잡합니다.
주기적으로 위아래로 진동하기 때문입니다.

##### 4.1 왜 `sin` 예제가 좋은가?

`sin(x)` 예제가 좋은 이유는 다음과 같습니다.

1. **직선으로는 절대 설명할 수 없다.**
2. **곡선이 반복적으로 변한다.**
3. 학습이 되면 학생들이 “신경망이 정말 곡선을 배우는구나”를 강하게 느낄 수 있다.

즉, `sin` 예제는 MLP가 단순한 선형 회귀의 연장이 아니라,
**비선형 함수를 근사하는 일반적인 회귀 모델**이라는 점을 잘 보여줍니다.

---

##### 예제 3. `sin(x)` 를 MLP로 근사하기

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math

# 데이터 생성
x = torch.linspace(-2 * math.pi, 2 * math.pi, 400).unsqueeze(1)
y = torch.sin(x)

# 모델 정의
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습
for epoch in range(3000):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# 테스트
test_x = torch.tensor([[0.0], [math.pi / 2], [math.pi]])
test_pred = model(test_x)

for tx, ty in zip(test_x, test_pred):
    print(f"x={tx.item():.4f}, predicted sin(x)={ty.item():.4f}")
```

###### 실행 결과 예시

```text
Epoch    0 | Loss: 0.521438
Epoch  500 | Loss: 0.087214
Epoch 1000 | Loss: 0.021503
Epoch 1500 | Loss: 0.006941
Epoch 2000 | Loss: 0.002118
Epoch 2500 | Loss: 0.000913
x=0.0000, predicted sin(x)=0.0021
x=1.5708, predicted sin(x)=0.9984
x=3.1416, predicted sin(x)=0.0068
```

여기서 특히 중요한 점은, $0$, $\pi/2$, $\pi$ 같은 대표 지점에서 예측값이 실제 `sin` 값과 매우 비슷해졌다는 것입니다.

##### 4.2 학습 포인트

이 예제에서 꼭 기억해야 할 점은 다음과 같습니다.

- MLP는 **연속적인 함수의 형태 자체**를 학습할 수 있다.
- 입력이 숫자 1개여도, 출력이 복잡한 곡선일 수 있다.
- 즉, 회귀는 단순한 직선 맞추기가 아니라 **함수 근사(function approximation)** 이다.

이 관점은 로봇 제어에 매우 중요합니다.
왜냐하면 로봇의 상태와 행동 사이의 관계는 대부분 단순한 직선이 아니기 때문입니다.

---

#### Section 5. 출력이 여러 개인 회귀

지금까지는 출력이 1개인 회귀만 보았습니다.
하지만 로봇에서는 한 번에 하나의 숫자만 예측하는 경우보다,
**여러 개의 연속값을 동시에 예측하는 경우**가 훨씬 많습니다.

##### 5.1 `Linear(1,6)` 의 의미

예를 들어 다음 레이어를 생각해 봅시다.

```python
nn.Linear(1, 6)
```

이것은 입력 1개를 받아 출력 6개를 내는 선형 레이어입니다.

수식으로 쓰면 다음과 같습니다.

$$
\hat{\mathbf{y}} = W\mathbf{x} + \mathbf{b}
$$

출력이 벡터가 되므로, 결과는 다음처럼 여러 값을 가집니다.

$$
[\hat{y}_1, \hat{y}_2, \hat{y}_3, \hat{y}_4, \hat{y}_5, \hat{y}_6]
$$

##### 5.2 로봇 관절 6개와의 연결

6축 로봇 팔을 생각해 보면, 출력 6개는 자연스럽게 다음을 의미할 수 있습니다.

| 출력 차원 | 의미         |
| --------- | ------------ |
| 1         | Joint 1 각도 |
| 2         | Joint 2 각도 |
| 3         | Joint 3 각도 |
| 4         | Joint 4 각도 |
| 5         | Joint 5 각도 |
| 6         | Joint 6 각도 |

즉, 하나의 입력 또는 하나의 상태로부터 **여러 관절의 목표값을 동시에 예측**할 수 있습니다.

##### 5.3 왜 회귀가 여러 값을 동시에 예측할 수 있을까?

회귀의 본질은 “출력값이 연속적인 수치인가?”에 있습니다.출력 차원이 1개인지 6개인지는 본질이 아닙니다.

- 출력 1개: 하나의 연속값 예측
- 출력 6개: 6개의 연속값을 동시에 예측
- 출력 100개: 100개의 연속값을 동시에 예측

모두 회귀입니다.

---

##### 예제 4. 하나의 입력으로 6개 관절값 예측하기

아래는 단순한 예시입니다.
입력 $x$ 하나로부터 6개 출력값을 예측합니다.

```python
import torch
import torch.nn as nn

model = nn.Linear(1, 6)

x = torch.tensor([[2.0]])   # 입력 1개
y_pred = model(x)

print(y_pred.shape)  # torch.Size([1, 6])
print(y_pred)
```

조금 더 현실적으로는 다음과 같이 상태 벡터를 입력으로 넣을 수 있습니다.

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 6)
)

state = torch.randn(4, 10)   # 배치 크기 4, 상태 차원 10
joint_pred = model(state)

print(joint_pred.shape)  # torch.Size([4, 6])
```

###### 실행 결과 예시

```text
torch.Size([1, 6])
tensor([[ 0.4123, -0.7811,  0.0534,  0.9240, -0.3375,  0.1188]],
       grad_fn=<AddmmBackward0>)

torch.Size([4, 6])
```

> 첫 번째 `tensor` 값은 학습 전 임의 초기화 상태이므로 실행할 때마다 달라질 수 있습니다. 여기서 핵심은 **출력 shape가 `[batch_size, 6]`** 이라는 점입니다.

여기서 출력의 각 원소는 각 관절의 연속적인 목표값이 됩니다.

---

#### Section 6. 로봇으로 연결

이제 회귀를 로봇에 직접 연결해 봅시다.

##### 6.1 로봇에서 회귀란 무엇인가?

로봇에서는 다음과 같은 구조가 자주 등장합니다.

```text
입력 상태
↓
신경망
↓
연속적인 제어값 출력
```

예를 들어,

```text
(x, y, z)
↓
MLP
↓
Joint1, Joint2, Joint3, Joint4, Joint5, Joint6
```

처럼 말단 위치와 관련된 정보를 입력으로 넣고,
각 관절의 목표각을 출력하게 할 수 있습니다.

여기서 중요한 것은 **정확한 역기구학(IK) 수식**이 아닙니다.이 장에서는 다음 관점이 중요합니다.

> **입력에서 여러 개의 연속 제어값을 직접 예측하면 그것은 회귀다.**

##### 6.2 로봇에서 회귀가 사용되는 대표 사례

| 입력 예시            | 출력 예시     | 의미                           |
| -------------------- | ------------- | ------------------------------ |
| 현재 로봇 상태       | 다음 관절각   | 다음 자세 예측                 |
| 카메라 이미지 + 상태 | 그리퍼 개방률 | 잡기 동작 제어                 |
| 목표 위치            | 관절 속도     | 목표로 이동하기 위한 속도 명령 |
| 관측 시퀀스          | action chunk  | 미래 행동 묶음 예측            |

##### 6.3 실제적 직관을 주는 짧은 예시

###### 예시 A. 목표 위치에서 관절각 예측

- 입력: 물체의 목표 위치 $(x, y, z)$
- 출력: 로봇 팔 6개 관절의 목표 각도

이 경우 출력은 다음과 같은 연속값 벡터입니다.

$$
[\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6]
$$

이는 명백한 회귀입니다.

###### 예시 B. 카메라와 상태로 그리퍼 개방률 예측

- 입력: 현재 이미지 + 손목 위치 + 속도
- 출력: 그리퍼 개방률 $g \in [0, 1]$

예를 들어,

- 0.0: 완전히 닫힘
- 1.0: 완전히 열림

처럼 표현할 수 있습니다.
이 역시 연속값 예측이므로 회귀입니다.

###### 예시 C. 토크 제어

- 입력: 현재 관절각, 속도, 목표 위치
- 출력: 각 관절에 가할 토크

토크는 실수값이므로 분류가 아니라 회귀입니다.

##### 6.4 중요한 관점

로봇에서 신경망이 하는 일은 종종 다음 한 문장으로 요약됩니다.

> **관측(observation)을 받아서 연속적인 행동(action)을 출력한다.**

이 관점은 이후 Behavior Cloning과 ACT를 이해하는 핵심이 됩니다.

---

#### Section 7. Behavior Cloning

Behavior Cloning(BC)은 가장 기본적인 이미테이션 러닝 방법 중 하나입니다.핵심 아이디어는 매우 간단합니다.

> 전문가(expert)의 상태-행동 데이터를 모아서,
> **관측값으로부터 행동값을 예측하도록 모델을 학습한다.**

##### 7.1 기본 구조

Behavior Cloning의 구조는 다음과 같습니다.

```text
Observation
↓
Neural Network
↓
Action
```

여기서 중요한 점은 **Action이 연속값인 경우가 많다**는 것입니다.

예를 들어 액션이 다음과 같다면 모두 회귀입니다.

- 관절각 벡터
- 속도 명령
- gripper opening
- end-effector delta pose

##### 7.2 상태 기반 BC와 이미지 기반 BC

###### 상태 기반 BC

입력으로 수치 상태를 직접 넣습니다.

예:

- 현재 관절각
- 현재 관절 속도
- 물체 위치
- end-effector pose

구조:

```text
Robot State
↓
MLP
↓
Action
```

###### 이미지 기반 BC

입력으로 카메라 이미지를 사용합니다.

구조:

```text
Image / Multi-view Image
↓
CNN or Transformer Encoder
↓
MLP / Policy Head
↓
Action
```

두 경우 모두 본질은 같습니다.

- 입력이 무엇이든
- 최종적으로 연속적인 행동을 예측하면
- 그것은 회귀 문제입니다.

##### 7.3 아주 단순한 로봇 행동 예측 예제

다음 데이터는 “현재 관절각”으로부터 “다음 관절각”을 예측하는 아주 단순한 예시입니다.

| 현재 각도 | 다음 각도 |
| --------- | --------- |
| 10        | 12        |
| 12        | 14        |
| 14        | 16        |
| 16        | 18        |

이 규칙은 매우 단순합니다.
하지만 교육적으로는 매우 강력합니다.

학생은 이 예제를 통해 다음을 느낄 수 있습니다.

- MLP가 숫자를 예측할 수 있다.
- 그 숫자가 로봇의 상태 변화와 연결될 수 있다.
- 즉, **신경망이 로봇의 다음 동작을 예측할 수 있다.**

---

##### 예제 5. 현재 관절각 → 다음 관절각

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 데이터
x = torch.tensor([[10.0],
                  [12.0],
                  [14.0],
                  [16.0]])

y = torch.tensor([[12.0],
                  [14.0],
                  [16.0],
                  [18.0]])

# 간단한 MLP
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# 테스트
test_angle = torch.tensor([[18.0]])
next_angle = model(test_angle)
print(f"현재 각도 18.0 -> 예측된 다음 각도 {next_angle.item():.4f}")
```

###### 실행 결과 예시

```text
Epoch    0 | Loss: 287.531982
Epoch  200 | Loss: 0.684219
Epoch  400 | Loss: 0.052347
Epoch  600 | Loss: 0.004918
Epoch  800 | Loss: 0.000412
현재 각도 18.0 -> 예측된 다음 각도 19.9987
```

즉, 모델은 `현재 각도 + 2` 라는 패턴을 학습해 **다음 관절각을 회귀로 예측**하고 있음을 볼 수 있습니다.

이 예제는 단순하지만 매우 중요합니다.
왜냐하면 이것이 바로 “상태에서 다음 행동을 예측하는 모델”의 가장 작은 형태이기 때문입니다.

##### 7.4 다중 관절 행동 예측으로 확장하기

실제 로봇은 관절이 여러 개입니다.
따라서 다음과 같이 확장할 수 있습니다.

```text
현재 상태 벡터
↓
MLP
↓
다음 Joint1~Joint6
```

예를 들면,

| 입력 상태               | 출력 행동                |
| ----------------------- | ------------------------ |
| 현재 6개 관절각         | 다음 6개 관절각          |
| 현재 pose + 속도        | 다음 joint velocity      |
| 이미지 + proprioception | gripper 포함 전체 action |

이 모든 것이 Behavior Cloning의 회귀 문제입니다.

---

#### Section 8. ACT로 연결

이제 이 장의 핵심 목적지인 **ACT(Action Chunking Transformer)** 로 연결해 봅시다.

##### 8.1 회귀에서 ACT까지의 흐름

우리가 지금까지 본 흐름은 다음과 같습니다.

```text
Linear Regression
↓
MLP Regression
↓
Robot Regression
↓
Behavior Cloning
↓
CNN/Transformer 기반 Behavior Cloning
↓
ACT
```

이 흐름에서 중요한 점은, ACT가 갑자기 완전히 다른 문제로 등장하는 것이 아니라는 점입니다.

##### 8.2 단계별 입력과 출력 비교

| 단계              | 입력                 | 출력         |
| ----------------- | -------------------- | ------------ |
| Linear Regression | $x$                | $y$        |
| MLP               | $x$                | $y$        |
| Robot MLP         | Robot State          | Joint        |
| Behavior Cloning  | Observation          | Action       |
| ACT               | Observation Sequence | Action Chunk |

이 표를 보면 입력과 출력의 형태는 점점 복잡해지지만,핵심은 계속 같습니다.

> **입력으로부터 연속적인 출력값을 예측한다.**

즉, 본질은 계속 회귀입니다.

##### 8.3 ACT는 무엇을 확장했는가?

ACT는 다음 두 가지를 크게 확장합니다.

1. **입력의 복잡성 증가**

   - 단일 숫자 $x$ 가 아니라
   - 이미지, 상태, 시간적 문맥을 포함한 **observation sequence**를 사용
2. **출력의 구조 증가**

   - 단일 action이 아니라
   - 여러 시점의 행동 묶음인 **action chunk**를 예측

즉, ACT는 다음처럼 이해할 수 있습니다.

```text
단순 회귀:
하나의 입력 -> 하나의 연속값

MLP 회귀:
복잡한 입력 -> 하나의 연속값

로봇 회귀:
상태 -> 여러 연속 action 값

Behavior Cloning:
관측 -> 현재 action

ACT:
관측 시퀀스 -> 여러 미래 action 값(action chunk)
```

##### 8.4 ACT에서 회귀 파트의 역할

ACT를 Transformer 기반 정책으로 보더라도, 마지막 목표는 여전히 명확합니다.

> **각 시점 또는 chunk에 대해 연속적인 action 값을 정확하게 예측하는 것**

예를 들어 action chunk가 길이 $H$ 이고, 각 시점 action 차원이 $D$ 라면
모델은 다음과 같은 출력을 예측합니다.

$$
[\mathbf{a}_{t}, \mathbf{a}_{t+1}, \mathbf{a}_{t+2}, \dots, \mathbf{a}_{t+H-1}]
$$

각 $\mathbf{a}_{t}$ 는 관절각, 속도, pose delta, gripper 값을 포함하는 연속 벡터일 수 있습니다.

따라서 ACT에서 회귀 파트는 다음 역할을 합니다.

- 각 미래 시점의 **연속적인 action 값**을 예측한다.
- 예측한 chunk와 전문가의 chunk 사이 오차를 줄인다.
- 정책이 실제 로봇 제어에 사용할 수 있는 **수치 행동**을 생성한다.

즉, Transformer는 구조를 더 강력하게 만들지만,
최종 목적함수의 중요한 부분은 여전히 **회귀 손실**입니다.

##### 8.5 이 장의 정체성 다시 확인하기

이 장은 분류를 배우기 위한 일반적인 입문 장이 아닙니다.이 장의 목적은 다음과 같습니다.

- **연속값 예측의 감각**을 익히고
- 그것이 로봇 행동과 어떻게 연결되는지 이해하고
- 그 관점에서 **Behavior Cloning**을 이해하며
- 최종적으로 **ACT를 “확장된 회귀 모델”로 받아들일 수 있게 하는 것**

즉, 이 장은 **ACT를 배우기 위한 준비 챕터**입니다.

---

#### 예제 모음 요약

이 장에서 다룬 핵심 예제들을 다시 정리하면 다음과 같습니다.

##### 예제 A. 직선 회귀

- 문제: $y = 2x + 1$
- 모델: `Linear(1,1)`
- 핵심: 선형층 = 1차 함수

##### 예제 B. $x^2$ 근사

- 문제: 곡선 함수
- 모델: MLP
- 핵심: Hidden Layer + ReLU가 비선형성 제공

##### 예제 C. $\sin(x)$ 근사

- 문제: 반복적이고 복잡한 비선형 함수
- 모델: MLP
- 핵심: MLP는 복잡한 연속 함수를 근사할 수 있음

##### 예제 D. 현재 관절각 → 다음 관절각

- 문제: 로봇의 다음 상태 예측
- 모델: MLP
- 핵심: 로봇 행동 예측은 회귀다

##### 예제 E. 다중 출력 관절 예측

- 문제: 6개 관절 동시 예측
- 모델: `Linear(..., 6)` 또는 MLP
- 핵심: 회귀는 여러 연속값을 동시에 출력할 수 있음

---

#### 정리

- 분류는 클래스 선택 문제이고, 회귀는 연속값 예측 문제이다.
- 로봇의 관절각, 속도, 토크, 그리퍼 개방률은 대부분 연속값이므로 회귀 문제이다.
- `nn.Linear(1,1)`은 수식으로 보면 $\hat{y}=wx+b$ 이며, 1차 방정식 회귀와 같다.
- 회귀에서는 예측값과 정답 사이의 수치 오차를 줄이기 위해 `MSELoss`를 자주 사용한다.
- 선형 모델은 직선만 표현할 수 있으므로 $x^2$, $\sin(x)$ 같은 곡선을 표현하는 데 한계가 있다.
- MLP는 은닉층과 비선형 활성화 함수를 통해 복잡한 비선형 함수를 근사할 수 있다.
- Behavior Cloning은 관측값으로부터 연속적인 행동을 예측하는 회귀 문제로 이해할 수 있다.
- ACT는 완전히 새로운 문제가 아니라, 회귀를 더 강력한 네트워크와 시간적 action chunk 예측으로 확장한 모델이다.

---

#### 연습문제

##### 개념 문제

1. 분류와 회귀의 차이를 설명하고, 로봇 제어에서 왜 회귀가 더 자주 등장하는지 서술하시오.
2. `nn.Linear(1,1)` 이 왜 $y=ax+b$ 와 같은 의미인지 설명하시오.이때 `weight`와 `bias`가 각각 무엇을 의미하는지도 함께 쓰시오.
3. ACT를 “확장된 회귀 모델”로 볼 수 있는 이유를 입력, 출력, 목적함수 관점에서 설명하시오.

##### 구현 문제

1. `y = 3x - 2` 데이터를 직접 만들고, `nn.Linear(1,1)` 과 `MSELoss`를 사용해 학습하시오.학습 후 `weight`와 `bias`를 출력하고 정답과 비교하시오.
2. `y = x^2` 데이터를 생성하고, 선형 모델과 MLP를 각각 학습하시오.어떤 모델이 더 잘 맞는지 loss를 비교하시오.
3. 다음과 같은 다중 출력 데이터를 직접 만들어 보시오.

   - 입력: 현재 관절각 2개 $[j_1, j_2]$
   - 출력: 다음 관절각 2개 $[j_1 + 1, j_2 + 1]$

   `nn.Sequential`로 작은 MLP를 만들고 학습하여,
   새로운 입력에 대해 다음 관절각을 예측하시오.

---

#### 마무리

이 장에서 우리는 단순한 직선 회귀에서 시작해, MLP를 이용한 비선형 함수 근사, 다중 출력 회귀, 그리고 로봇 행동 예측까지 확장해 보았습니다.

이 흐름은 단순한 수학 연습이 아닙니다.
이것은 다음 장에서 배우게 될 내용을 위한 준비입니다.

```text
회귀
↓
로봇 행동 예측
↓
Behavior Cloning
↓
ACT
```

2장에서 분류 학습·추론을 실행하고 3장에서 MuJoCo를 익힌 뒤, 4~6장에서 상태 이력과 로봇 정책 학습을 다룹니다.

---

#### 1.4.1 세 가지 기본 문제 형태

로봇 정책 학습은 입력과 출력의 형태에 따라 크게 세 가지로 정리할 수 있다.

##### 1) joint state → action

가장 단순한 imitation learning 형태이다.

- 입력: 현재 관절 상태 `[B, D]`
- 출력: 다음 액션 `[B, A]`
- 모델: MLP
- loss: MSE

이는 `101_pytorch_mnist.py`의 MLP 구조와 유사한 방식으로 이해할 수 있다.

##### 2) image + state → action

비전 기반 조작의 기본 형태이다.

- 입력1: 이미지 `[B, C, H, W]`
- 입력2: 상태 `[B, D]`
- 출력: 액션 `[B, A]`
- 모델: CNN 또는 ViT encoder + MLP head

이는 `102_pytorch_CNN_cifar10.py`, `105_pytorch_Resnet_transfer_fine.py`를 로봇 정책으로 확장한 구조이다.

##### 3) sequence → next action

Transformer policy의 기본 형태이다.

- 입력: 상태 또는 토큰 시퀀스 `[B, T, D]`
- 출력: 다음 액션 또는 action sequence
- 모델: Transformer 또는 시계열 모델

이때 `B`는 batch, `T`는 time, `D`는 feature 또는 token dimension을 의미한다.

#### 1.4.2 IL, Transformer, RL, IL+IK와의 연결

로봇 AI의 주요 학습 방식은 모델 형태와 목적함수가 다르지만, PyTorch 관점의 핵심 구조는 유사하다.

- IL: 시연 데이터셋을 사용하여 정책이 정답 액션을 재현하도록 학습한다.
- Transformer policy: 입력이 시퀀스로 확장되고 shape 해석과 GPU 사용의 중요성이 커진다.
- RL: 정답 레이블 대신 환경 rollout과 reward를 사용하지만, 텐서 저장, 배치 샘플링, forward, loss 계산, backward, update 구조는 유지된다.
- IL + IK 하이브리드 제어: 정책이 직접 joint command를 출력하지 않고 목표 pose나 residual을 예측한 뒤 IK와 결합할 수 있다.

하이브리드 제어의 개념은 다음과 같이 표현할 수 있다.

```python
pose_target = policy(obs)          # 학습 가능한 부분
joint_cmd = ik_solver(pose_target) # 기구학/제어 부분
```

즉, PyTorch는 제어기 전체를 대체하는 것이 아니라 학습이 필요한 부분을 구현하는 도구로 사용된다.

#### 1.4.3 제공된 예제 코드를 로봇 관점에서 읽는 방법

`101_pytorch_mnist.py`는 다음 요소를 간결하게 보여준다.

- `device` 설정
- `DataLoader`
- `nn.Module`
- `nn.Linear` 기반 MLP
- 손실 함수와 optimizer
- 학습 루프
- `model.eval()`
- `torch.no_grad()`
- `state_dict` 저장

로봇 관점에서는 이를 “분류 문제 대신 행동 예측 문제로 바꾼 가장 단순한 정책 학습 코드”로 읽을 수 있다.

`102_pytorch_CNN_cifar10.py`는 이미지 입력을 다루는 기본 구조를 보여준다. 로봇에서는 CIFAR-10 이미지를 작업 장면 카메라 이미지로 바꾸고, 클래스 예측을 행동 또는 물체 상태 예측으로 바꾸면 된다.

`105_pytorch_Resnet_transfer_fine.py`는 CNN 기반 시각 백본인 ResNet18의 전이학습과 미세 조정을 보여준다. 이 파일은 Transformer나 ViT 구현이 아니다. 로봇에서는 이를 다음과 같이 확장할 수 있다.

- ResNet18로 이미지 특징 추출
- 상태 벡터와 결합
- policy head로 action 출력
- 이 특징을 Transformer 정책의 입력으로 사용

#### 1.4.4 로봇형 샘플 코드

다음 코드는 `joint state → action` 형태의 가장 단순한 behavior cloning 예시이다.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RobotDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states      # [N, 13]
        self.actions = actions    # [N, 7]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class PolicyMLP(nn.Module):
    def __init__(self, state_dim=13, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        return self.net(state)

states = torch.randn(1000, 13)
actions = torch.randn(1000, 7)

dataset = RobotDataset(states, actions)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolicyMLP().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    for state, action_gt in loader:
        state = state.to(device)
        action_gt = action_gt.to(device)

        action_pred = model(state)
        loss = criterion(action_pred, action_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

이 구조를 이미지 입력으로 확장하면 `image + state → action` 형태가 된다.

```python
class VisionStatePolicy(nn.Module):
    def __init__(self, state_dim=13, action_dim=7):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, image, state):
        image_feat = self.image_encoder(image)   # [B, 32]
        state_feat = self.state_encoder(state)   # [B, 32]
        fused = torch.cat([image_feat, state_feat], dim=1)
        return self.head(fused)                  # [B, 7]
```

시퀀스 문제는 다음과 같이 표현할 수 있다.

```python
class SequencePolicy(nn.Module):
    def __init__(self, state_dim=13, hidden_dim=64, action_dim=7):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_seq):
        # state_seq: [B, T, 13]
        x = self.input_proj(state_seq)   # [B, T, 64]
        h, _ = self.temporal(x)          # [B, T, 64]
        last = h[:, -1]                  # [B, 64]
        return self.head(last)           # [B, 7]
```

이 세 예시를 통해 확인할 수 있는 점은 다음과 같다.

- 입력이 state, image, sequence 중 무엇이든 결국 tensor로 표현된다.
- 정책은 입력 tensor를 출력 tensor로 변환하는 함수이다.
- `Dataset`과 `DataLoader`는 로봇 데이터를 batch 형태로 공급한다.
- train loop는 정책이 시연 또는 목표를 더 잘 따르도록 업데이트하는 과정이다.
- Transformer, RL, IL+IK로 확장되더라도 기본 구조는 유지된다.


---

## 1.5 실전 점검과 다음 장 연결

### 1.5 실전 점검과 다음 장 연결

#### 1.5.1 자주 보는 에러와 실수

초기 단계에서 자주 나타나는 문제는 다음과 같다.

##### shape mismatch

- `[B, D]`가 와야 하는데 `[D]`가 들어가는 경우
- 이미지가 `[B, C, H, W]`여야 하는데 `[B, H, W, C]`로 들어가는 경우
- transformer가 `[B, T, D]`를 기대하는데 다른 축 순서로 들어가는 경우

해결 방법은 간단하다.

```python
print(x.shape)
```

각 축의 의미를 주석으로 함께 적어 두면 도움이 된다.

##### device mismatch

- 모델은 `cuda`
- 데이터는 `cpu`

해결 방법은 모델과 데이터를 동일한 `device`로 보내는 것이다.

```python
x = x.to(device)
model = model.to(device)
```

##### `optimizer.zero_grad()` 누락

이전 gradient가 누적되어 의도하지 않은 학습이 진행될 수 있다.

##### `model.eval()`를 호출하지 않고 추론

Dropout이나 BatchNorm이 포함된 경우 결과가 불안정해질 수 있다.

##### `torch.no_grad()` 없이 추론

메모리 사용량이 증가하고 속도도 저하될 수 있다.

##### batch size를 과도하게 크게 설정

VRAM 부족으로 OOM이 발생할 수 있다. Transformer나 이미지 모델에서 특히 자주 나타난다.

#### 1.5.2 이 장을 마치면 할 수 있어야 하는 것

이 장을 마치면 다음 내용을 수행하거나 설명할 수 있어야 한다.

- PyTorch 학습 코드의 전체 흐름을 설명할 수 있다.
- `tensor`와 shape를 보고 데이터 의미를 해석할 수 있다.
- `[B, D]`, `[B, C, H, W]`, `[B, T, D]`의 차이를 이해할 수 있다.
- `nn.Linear` 기반 MLP가 가장 단순한 정책이 될 수 있는 이유를 설명할 수 있다.
- `Dataset`과 `DataLoader`의 역할을 설명할 수 있다.
- `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`의 역할을 설명할 수 있다.
- `model.train()`, `model.eval()`, `torch.no_grad()`의 차이를 이해할 수 있다.
- CPU/GPU와 device mismatch 문제를 피할 수 있다.
- `101_pytorch_mnist.py`, `102_pytorch_CNN_cifar10.py`, `105_pytorch_Resnet_transfer_fine.py`를 로봇 정책 학습의 관점에서 해석할 수 있다.

#### 1.5.3 다음 장으로의 연결

2장은 `01_02_pytorch_basic/`의 MNIST, CIFAR-10, MiniVGGNet, VGG16, ResNet18 학습·웹캠 추론을 다룬다. 3장에서 MuJoCo의 관측·명령·물리 진행을 익히고, 4장에서 상태 이력과 순차 모델의 원리를 연결한다. 5장은 `05_mujoco_bc/`의 MLP·RNN·Transformer BC 실습, 6장은 `06_mujoco_act/`의 ACT 실습이다.

분류 예제의 모델을 이름만 바꿔 로봇에 연결하는 것은 아니다. 분류 정답은 클래스 인덱스이고 `CrossEntropyLoss`를 사용한다. 관절 BC의 정답은 적용할 연속 위치 명령이며 MSE 또는 L1을 사용한다. 입력 shape, 행동 단위, 데이터 시점을 새 과제에 맞춰야 한다.

#### 1.5.4 저장소에서 실제 파일 확인하기

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

이 장의 실제 연결 파일은 `101_pytorch_mnist.py`, `102_pytorch_CNN_cifar10.py`, `105_pytorch_Resnet_transfer_fine.py`이다. 원리 설명을 위한 Tensor·회귀 코드 조각은 본문 안의 독립 예제이며, 같은 이름의 별도 실행 파일이 제공된다는 뜻이 아니다.

학습 파일 일부와 웹캠 추론 파일은 import만 해도 데이터 읽기·학습·카메라 초기화를 수행하는 최상위 실행문을 포함한다. 모델 정의만 살펴보려는 용도로 전체 파일을 무심코 import하지 않는다. 실제 실행은 2장의 준비·경로 안내를 따른다.
