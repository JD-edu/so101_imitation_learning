# 02장. PyTorch 비전 모델의 학습과 웹캠 추론

> **2장 소스코드:** `01_02_pytorch_basic/`. 폴더 앞번호는 책의 장 번호입니다. 4·5·6장은 각각 독립된 폴더를 사용합니다. 내부 파일 번호는 본문의 실행 파일 표에 표시한 예제 ID입니다.

<!-- CLI_GUIDE_START -->
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
<!-- CLI_GUIDE_END -->

---

# 이 장의 실습 계약

`01_02_pytorch_basic/`의 실제 10개 Python 파일을 기준으로 학습과 추론을 연결한다. 분류 입력과 로봇 정책 입력을 구별하고, 각 실습의 데이터 분할·정규화·checkpoint 이름까지 함께 확인한다.

| 단계 | 학습 파일 | 추론 파일 | 가중치 |
|---|---|---|---|
| MNIST | `101_pytorch_mnist.py` | `101_pytorch_mnist_inference.py` | `mnist_mlp.pth` |
| CIFAR-10 | `102_pytorch_CNN_cifar10.py` | `102_pytorch_CNN_cifar10_inference.py` | `cifar10_cnn.pth` |
| MiniVGG | `103_pytorch_miniVGGnet.py` | `103_pytorch_miniVGGnet_inference.py` | `miniVGGnet.pth` |
| VGG16 | `104_pytorch_VGGnet_transfer_feature_can.py` | `104_pytorch_VGGnet_transfer_feature_inference.py` | `vgg16_transfer_model.pth` |
| ResNet18 | `105_pytorch_Resnet_transfer_fine.py` | `105_pytorch_Resnet_transfer_fine_inference.py` | `resnet18_final.pth` |

## 실행 위치와 준비

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python -c "import torch, torchvision, cv2, matplotlib, torchsummary; print(torch.__version__)"
```

필요한 패키지는 PyTorch, torchvision, NumPy, OpenCV, Matplotlib, torchsummary이다. 설치 환경은 사용하는 CPU/CUDA 구성에 맞춰 준비한다. `images/`, `data/`, `.pth`는 이 폴더를 작업 디렉터리로 실행한다는 전제의 상대 경로이다. 저장소 루트에서 파일 경로만 지정하면 다른 곳의 데이터를 읽거나 결과를 저장할 수 있다.

MNIST·CIFAR-10 데이터와 ImageNet 사전학습 가중치는 로컬 캐시가 없으면 다운로드가 필요하다. 커스텀 예제는 `images/cans`, `images/cups`, `images/pets`를 사용한다. 실제 클래스 순서는 `ImageFolder.class_to_idx`로 확인한다.

학습이 끝나고 가중치가 저장된 뒤 추론 명령을 실행한다. MiniVGG/VGG16/ResNet18은 학습 그래프의 `plt.show()`가 끝난 뒤 저장하므로 그래프 창도 닫아야 한다. 추론은 모두 카메라 인덱스 0을 사용하며 `q`로 종료한다. ROI가 정상적으로 잘리려면 카메라 해상도가 설정된 ROI 크기 이상이어야 한다.

## 입력 크기와 검증의 차이

| 단계 | 입력 | 평가 |
|---|---|---|
| MNIST | `[B,1,28,28]` | MNIST test split |
| CIFAR-10 | `[B,3,32,32]` | CIFAR-10 test split |
| MiniVGG | `[B,3,64,64]` | 현재 파일에는 별도 검증 분할 없음 |
| VGG16 | `[B,3,224,224]` | 현재 파일에는 별도 검증 분할 없음 |
| ResNet18 | `[B,3,224,224]` | 파일 단위 80/20 분할, 검증 증강 제외 |

한 물체를 연속 촬영한 이미지들이 양쪽 분할에 섞이면 파일이 달라도 비슷한 장면이 공유될 수 있다. 사람·물체·촬영 세션이 독립적인 평가를 자동으로 수행하는 코드는 아니다. 웹캠 결과나 softmax 점수를 테스트셋 정확도와 혼동하지 않는다.

## 이번 코드 대조의 범위

MLP·CNN·MiniVGG 모델의 작은 배치 역전파와 학습/추론 구조 일치, VGG16·ResNet18 추론 로더의 가중치 복원, ResNet의 분할·전처리·BatchNorm 처리를 검사했다. 카메라 구동, 전체 epoch 재학습, 최종 정확도 측정은 수행하지 않았다. 다음 3장은 MuJoCo 실행, 4장은 상태 이력과 순차 모델 원리를 다룬다.


---

## 2.1 MNIST로 시작하는 PyTorch

### 요약

이 예제는 가장 단순한 형태의 이미지 분류 파이프라인이다. **입력 텐서, 모델 정의, 손실 함수, 옵티마이저, 학습 루프, 평가, 가중치 저장**이라는 PyTorch의 핵심 흐름을 한 번에 익힐 수 있다. 또한 웹캠 입력을 MNIST 스타일로 바꾸는 과정을 통해 **학습 데이터와 추론 데이터의 형식 일치**가 얼마나 중요한지도 확인할 수 있다.

### 소스코드 개요 및 역할 설명

**`101_pytorch_mnist.py`**

- MNIST 데이터셋 로드
- MLP 모델 정의
- 학습 및 테스트 정확도 계산
- 가중치 저장: `mnist_mlp.pth`

**`101_pytorch_mnist_inference.py`**

- 저장된 MLP 가중치 로드
- 웹캠 중앙 ROI 추출
- 흑백 변환 및 이진화 반전
- 실시간 숫자 분류

### 코드 블록 상세 설명

#### 1) 하이퍼파라미터 및 장치 설정

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 0.001
epochs = 5
```

이 부분은 학습을 시작하기 전에 **계산 환경과 학습 방식의 기본 규칙**을 정하는 단계다. 모델 구조 못지않게 중요한 설정이며, 이후 학습 속도와 안정성에 직접적인 영향을 준다.

**`device` 설정: 계산 유닛 선택**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

이 코드는 현재 컴퓨터에서 **NVIDIA GPU(CUDA)**를 사용할 수 있는지 먼저 확인한 뒤, 가능하면 GPU를 쓰고 그렇지 않으면 CPU를 쓰겠다는 의미다.

- CUDA(GPU)는 수천 개의 코어로 대량의 행렬 연산을 동시에 처리하므로 딥러닝 학습 속도가 매우 빠르다.
- CPU는 일반 작업에는 충분하지만 대규모 텐서 연산에서는 상대적으로 느리다.

쉽게 비유하면 GPU는 딥러닝 학습을 위한 **고속 전용 차량**, CPU는 범용 작업에 적합한 **일반 차량**에 가깝다.

가장 중요한 점은 **장치 불일치(Device Mismatch)**를 피하는 것이다. 예를 들어 모델은 GPU에 올라가 있는데 입력 데이터는 CPU에 남아 있으면 연산이 되지 않는다. 그래서 학습 코드에서 다음처럼 **모델과 데이터 모두를 같은 장치로 이동**시키는 습관이 매우 중요하다.

```python
model = model.to(device)
data, target = data.to(device), target.to(device)
```

**`batch_size`: 한 번에 공부할 분량**

```python
batch_size = 64
```

딥러닝 모델은 전체 데이터를 한 번에 모두 보는 것이 아니라 일정 개수씩 나누어 학습한다. `batch_size = 64`는 **데이터 64개를 한 묶음으로 모델에 넣고 한 번 업데이트한다**는 뜻이다.

- 너무 크면 GPU 메모리나 시스템 메모리가 부족해 실행이 멈출 수 있다.
- 너무 작으면 학습 속도가 느려지고 gradient 업데이트가 들쭉날쭉해져 학습이 불안정해질 수 있다.

즉 batch size는 **속도, 메모리 사용량, 학습 안정성** 사이의 균형점이다.

**`learning_rate`: 학습 속도, 즉 보폭**

```python
learning_rate = 0.001
```

학습률은 모델이 오차를 발견했을 때 **파라미터를 얼마나 크게 수정할지**를 정하는 값이다.

- 값이 너무 크면 빠르게 움직이지만 최적점을 지나쳐 버려 학습이 불안정해질 수 있다.
- 값이 너무 작으면 매우 천천히 학습되어 시간이 오래 걸리고 충분히 좋은 해에 도달하기 어려울 수 있다.

따라서 `0.001`은 Adam 옵티마이저에서 자주 쓰이는 **무난한 시작점**으로 볼 수 있다. 실험을 진행하면서 loss가 너무 흔들리거나 거의 줄지 않는다면 가장 먼저 의심해 볼 값이 바로 learning rate다.

**`epochs`: 전체 데이터 반복 횟수**

```python
epochs = 5
```

`epochs = 5`는 전체 학습 데이터셋을 처음부터 끝까지 훑는 과정을 총 5번 반복한다는 뜻이다. 예를 들어 MNIST 학습 데이터 6만 장을 batch 단위로 나누어 모두 학습하는 과정을 5회 수행하는 셈이다.

- epoch 수가 너무 적으면 아직 충분히 학습되지 않아 성능이 낮을 수 있다.
- epoch 수가 너무 많으면 학습 데이터를 지나치게 외워 버리는 **과적합(Overfitting)**이 발생할 수 있다.

결국 epoch는 단순한 반복 횟수가 아니라 **얼마나 충분히 학습시키되 과적합은 피할 것인가**를 결정하는 중요한 축이다.

정리하면 이 네 줄은 단순한 초기 설정이 아니라 **어디서 계산할지(device), 얼마나 나누어 학습할지(batch size), 얼마나 크게 수정할지(learning rate), 몇 번 반복할지(epoch)**를 한꺼번에 결정하는 핵심 제어판이다.

#### 2) 데이터셋과 정규화

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

이 부분은 모델에 이미지를 넣기 전에 **입력 데이터를 학습하기 좋은 형태로 바꾸는 전처리 단계**다. 겉보기에는 코드가 짧지만 실제로는 이미지 분류 성능과 학습 안정성에 매우 큰 영향을 준다.

**`transforms.Compose`: 작업 지시서 묶기**

`Compose`는 여러 개의 변환 과정을 **정해진 순서대로 차례차례 실행**하도록 묶어 주는 도구다. 여기서는 다음 두 작업을 순서대로 수행하라는 뜻이다.

1. `ToTensor()`
2. `Normalize()`

즉 이미지를 바로 모델에 넣는 것이 아니라 **텐서 변환 -> 정규화**라는 파이프라인을 먼저 거치게 만든다.

**`transforms.ToTensor()`: 이미지를 숫자로 변환**

```python
transforms.ToTensor()
```

컴퓨터는 이미지 자체를 이해하지 못하고 결국 **숫자 배열**로 바뀐 형태만 계산할 수 있다. `ToTensor()`는 일반 이미지 데이터를 PyTorch가 다룰 수 있는 **Tensor** 형식으로 바꿔 준다.

이 과정에서 두 가지가 동시에 일어난다.

1. 형식 변환: 이미지 데이터를 텐서 형태로 바꾼다.
2. 스케일 변환: 원래 `0~255` 범위의 픽셀 값을 `0.0~1.0` 범위의 실수로 바꾼다.

예를 들어 원래 픽셀 값이 255라면 1.0에 가깝게, 0이라면 0.0으로 변환된다. 즉 흑백 이미지 기준으로 **0은 검은색, 1은 흰색에 가까운 값**이 된다.

**`transforms.Normalize((0.1307,), (0.3081,))`: 수치 표준화**

```python
transforms.Normalize((0.1307,), (0.3081,))
```

이 부분이 전처리의 핵심이다. `Normalize`는 입력 데이터를 **평균이 0 근처, 표준편차가 1 근처가 되도록 재조정**한다.

여기서 사용된 값의 의미는 다음과 같다.

- `0.1307`: MNIST 데이터셋 전체 픽셀의 평균값
- `0.3081`: MNIST 데이터셋 전체 픽셀의 표준편차

MNIST는 1채널 이미지이므로 평균과 표준편차도 각각 값 하나만 들어간다.

정규화는 내부적으로 대략 다음과 같은 계산을 수행한다.

```python
x_normalized = (x - mean) / std
```

즉 각 픽셀값에서 평균을 빼고 표준편차로 나누어 입력 분포를 더 다루기 쉬운 형태로 바꾸는 것이다.

**왜 이렇게 정규화하는가: 학습 효율과 안정성**

정규화를 하는 이유는 단순히 숫자를 바꾸기 위해서가 아니라 **모델이 더 안정적이고 빠르게 학습하도록 만들기 위해서**다.

1. **입력 수치 스케일 조정**어떤 입력값은 매우 크고 어떤 값은 매우 작으면, 모델이 특정 구간의 값에 지나치게 민감하게 반응할 수 있다. 입력을 평균 0 근처로 맞추면 특정 방향으로 치우친 반응을 줄이는 데 도움이 된다.
2. **최적화 효율 향상**딥러닝 모델의 가중치 업데이트는 입력값이 지나치게 한쪽으로 쏠려 있지 않을 때 더 안정적으로 진행되는 경우가 많다. 즉 데이터가 0 근처를 중심으로 고르게 분포하면, 경사하강법 기반 최적화가 더 잘 작동한다.
3. **학습 속도 개선**
   정규화된 입력은 손실 감소가 더 부드럽게 진행되는 경우가 많아, 같은 epoch 수에서도 더 좋은 결과를 얻는 데 도움이 된다.

MNIST는 **1채널 grayscale 28x28 이미지**다. 따라서 이 transform은 결국 "흑백 손글씨 이미지를 텐서로 바꾸고, MNIST 통계값 기준으로 표준화해서 모델이 학습하기 좋은 숫자 형태로 정리한다"는 의미로 이해하면 된다.

> 주의: 정규화에 사용한 평균과 표준편차는 학습과 추론에서 반드시 동일해야 한다. 학습 때는 `Normalize((0.1307,), (0.3081,))`를 썼는데 추론 때 다른 값을 사용하면 입력 분포가 달라져 성능이 흔들릴 수 있다.

#### 3) MLP 모델 구조

```python
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

이 모델은 가장 전형적인 **다층 퍼셉트론(MLP)** 구조다. 핵심은 2차원 이미지를 1차원 벡터로 펼친 뒤 여러 개의 선형층과 활성화 함수를 거쳐 최종 클래스 점수를 만드는 데 있다.

전체 흐름은 다음처럼 이해하면 된다.

**28x28 이미지 -> 784차원 벡터 -> 512차원 특징 -> 256차원 특징 -> 10개 클래스 로짓**

**`nn.Flatten()`: 2차원을 1차원으로 바꾸기**

MNIST 이미지는 가로 28, 세로 28의 **2차원 격자 형태**다. 이 모델은 `Flatten`으로 이미지를 펼친 뒤 `nn.Linear`의 마지막 입력 차원 784에 전달한다.

```python
self.flatten = nn.Flatten()
```

이 연산은 `28 x 28` 행렬을 `784`개의 숫자가 나열된 긴 벡터로 바꾼다.

비유하면 **격자무늬 바닥 타일을 하나씩 떼어서 길게 한 줄로 이어 붙이는 것**과 비슷하다. 이렇게 해야 뒤의 선형층이 각 픽셀값을 입력으로 받아 계산할 수 있다.

**`nn.Linear(28*28, 512)`: 첫 번째 특징 추출**

```python
nn.Linear(28*28, 512)
```

이 층은 784개의 픽셀 정보를 받아 **512개의 새로운 특징값**으로 요약한다. 단순히 차원을 줄이는 것이 아니라 모델이 "어떤 픽셀 조합이 숫자 형태를 구분하는 데 의미가 있는가"를 학습하는 단계다.

내부적으로는 각 픽셀에 가중치(weight)를 곱하고 더하는 연산을 수행한다. 즉 원본 픽셀을 그대로 보존하는 것이 아니라 **숫자의 획 모양이나 전체 윤곽을 판별하기 위한 중간 재료**를 만들어 내는 과정이라고 볼 수 있다.

**`nn.ReLU()`: 비선형성 추가**

```python
nn.ReLU()
```

ReLU는 신경망에서 매우 중요한 **비선형 활성화 함수**다. 역할은 간단하다.

- 0보다 작은 값은 0으로 만든다.
- 0보다 큰 값은 그대로 통과시킨다.

만약 신경망이 선형층만 여러 개 쌓인 구조라면 결국 전체적으로는 하나의 큰 선형변환과 크게 다르지 않다. 그렇게 되면 복잡한 패턴을 배우기 어렵다. ReLU를 넣으면 모델이 **복잡하고 구부러진 관계**, 즉 단순 직선으로 설명할 수 없는 패턴을 학습할 수 있게 된다.

**`nn.Linear(512, 256) -> nn.Linear(256, 10)`: 점진적 요약과 최종 분류**

```python
nn.Linear(512, 256)
nn.ReLU()
nn.Linear(256, 10)
```

이 부분은 앞에서 만든 특징을 더 압축하고 정리해 마지막에는 **0부터 9까지 어떤 숫자인지 점수 형태로 출력**하는 단계다.

- `512 -> 256`: 중간 특징을 한 번 더 요약
- `256 -> 10`: 최종 10개 클래스 점수 출력

여기서 마지막 `10`은 우리가 구분하려는 정답의 개수, 즉 **숫자 0~9의 10개 클래스**를 의미한다.

마지막 층의 출력은 확률이 아니라 **로짓(logits)**이다. 예를 들어 다음과 같은 값이 나올 수 있다.

```python
[10.2, 1.3, -0.8, 0.5, ...]
```

이는 "0번 클래스 점수는 높고, 1번이나 2번 클래스 점수는 상대적으로 낮다"는 식의 **날것의 점수**다. 이후 `CrossEntropyLoss`가 내부적으로 softmax를 포함해 이를 학습에 적합한 형태로 사용한다.

**MLP와 공간 구조의 관계**

Flatten은 정해진 순서로 픽셀을 펼치므로 픽셀 값이나 위치 인덱스를 삭제하지 않는다. 이 예제의 문제는 위치 정보가 사라진다는 것이 아니라, Linear 층이 지역 연결과 공간적 가중치 공유를 기본으로 제공하지 않는다는 점이다. 이미지를 옮기면 다른 입력 인덱스의 가중치가 사용된다. 다음 절의 CNN은 작은 영역에 같은 합성곱 필터를 적용한다.

#### 4) 손실 함수와 옵티마이저

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

이 부분은 모델이 **얼마나 틀렸는지 측정하고**, 그 오차를 바탕으로 **가중치를 어떻게 수정할지 결정하는 단계**다. 즉 손실 함수는 오답의 크기를 재는 자이고, 옵티마이저는 그 오답을 줄이기 위해 움직이는 방법을 정하는 조정자다.

**`nn.CrossEntropyLoss()`: 정답과의 거리 측정기**

```python
criterion = nn.CrossEntropyLoss()
```

`CrossEntropyLoss`는 다중 클래스 분류에서 가장 널리 쓰이는 손실 함수다. MNIST처럼 **0부터 9까지 여러 선택지 중 정답 하나를 맞히는 문제**에 특히 적합하다.

모델은 마지막 선형층에서 클래스별 점수, 즉 **로짓(logits)**을 출력한다. 손실 함수는 이 점수들을 바탕으로 모델이 정답 클래스에 얼마나 높은 점수를 주었는지를 계산해 **오차(loss)**로 바꿔 준다.

예를 들어 실제 정답이 `3`인데

- 모델이 `3`에 높은 점수를 주면 손실이 낮다.
- 모델이 `7` 같은 엉뚱한 클래스에 높은 점수를 주면 손실이 커진다.

즉 `CrossEntropyLoss`는 "모델이 정답을 얼마나 잘 맞히고 있는가"를 **하나의 숫자**로 요약해 주는 도구다.

**왜 마지막에 Softmax를 직접 넣지 않는가**

입문자가 가장 많이 헷갈리는 지점이다. 일반적으로 분류 문제에서는 클래스별 점수를 확률처럼 보고 싶기 때문에 softmax를 떠올리게 된다.

Softmax는 로짓을 받아 전체 합이 1이 되도록 바꾸는 함수다. 예를 들어 다음처럼 해석 가능한 확률 형태를 만든다.

- 3일 확률 0.90
- 5일 확률 0.05
- 8일 확률 0.01

하지만 PyTorch의 `nn.CrossEntropyLoss()`는 내부적으로 **softmax 계산 + 로그 연산 + 오차 계산**을 한 번에 처리하도록 설계되어 있다. 그래서 학습 단계에서는 모델의 마지막 층 뒤에 `nn.Softmax()`를 직접 넣지 않아도 된다.

이렇게 설계된 이유는 두 가지다.

1. **수학적 안정성**softmax를 따로 적용한 뒤 로그를 취하면 확률값이 매우 작을 때 수치적으로 불안정해질 수 있다. `CrossEntropyLoss`는 이를 더 안정적인 방식으로 계산한다.
2. **코드 간결성**
   모델의 마지막 출력은 로짓 그대로 두고 손실 함수가 필요한 계산을 알아서 처리하므로 구현이 단순해진다.

> 주의: 모델 마지막에 `nn.Softmax()`를 직접 넣고 또 `nn.CrossEntropyLoss()`를 사용하면 softmax가 사실상 중복 적용되어 학습이 비정상적으로 진행될 수 있다.

**`optim.Adam`: 영리한 길찾기 알고리즘**

```python
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

옵티마이저는 손실을 줄이기 위해 **모델의 가중치를 어떤 방향으로, 얼마나 수정할지**를 결정하는 알고리즘이다. 쉽게 말해 손실 함수가 "얼마나 틀렸는가"를 알려 준다면 옵티마이저는 "그 틀림을 줄이려면 어디로 움직여야 하는가"를 결정하는 **내비게이터**다.

`Adam`은 현재 딥러닝에서 가장 대중적으로 쓰이는 옵티마이저 중 하나다. 이름은 **Adaptive Moment Estimation**의 약자다.

Adam의 강점은 크게 두 가지로 이해하면 좋다.

1. **방향 기억(Momentum)**이전 업데이트 방향을 어느 정도 기억하면서 이동하므로, 들쭉날쭉한 경로를 줄이고 더 효율적으로 손실이 낮아지는 방향으로 나아간다.
2. **보폭 자동 조절(Adaptive Learning Rate)**
   어떤 파라미터는 더 크게, 어떤 파라미터는 더 조심스럽게 업데이트할 수 있어 학습이 안정적이다. 비유하면 목표 지점에서 멀 때는 비교적 크게 움직이고 가까워질수록 더 섬세하게 조정하는 느낌에 가깝다.

정리하면 `CrossEntropyLoss`는 **정답과의 거리 측정기**, `Adam`은 **그 거리를 줄이기 위한 영리한 길찾기 알고리즘**이다. 둘이 함께 동작하면서 모델은 점점 더 정답에 가까운 출력을 내도록 학습된다.

#### 5) 학습 루프의 핵심

```python
optimizer.zero_grad()
output = model(data)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

이 다섯 줄이 바로 PyTorch 학습의 심장부다. 앞에서 정의한 모델, 손실 함수, 옵티마이저가 실제로 연결되어 **예측 -> 오차 계산 -> 오차를 줄이는 방향 계산 -> 가중치 수정**의 흐름을 만든다.

학습 루프를 한 단계씩 풀어 보면 다음과 같다.

**`optimizer.zero_grad()`: 이전 기울기 초기화**

```python
optimizer.zero_grad()
```

PyTorch는 기본적으로 gradient를 자동으로 누적한다. 그래서 이전 반복에서 계산된 gradient를 지우지 않으면 이번 step의 결과와 섞여 버린다. 학습을 시작할 때마다 먼저 기울기를 비워 주는 이유가 여기에 있다.

**`output = model(data)`: 순전파(Forward Pass)**

```python
output = model(data)
```

입력 이미지 `data`를 모델에 넣어 예측값을 얻는다. 이때 출력은 확률이 아니라 **로짓(logits)**이며 각 클래스에 대한 점수라고 이해하면 된다.

**`loss = criterion(output, target)`: 오차 계산**

```python
loss = criterion(output, target)
```

모델의 예측 결과 `output`과 실제 정답 `target`을 비교해 손실값을 계산한다. 이 손실은 현재 모델이 얼마나 틀렸는지를 나타내는 숫자이며, 학습은 결국 이 값을 줄이는 방향으로 진행된다.

**`loss.backward()`: 역전파(Backpropagation)**

```python
loss.backward()
```

이 단계에서는 손실이 각 가중치에 대해 얼마나 민감한지, 즉 **어느 파라미터를 어느 방향으로 얼마나 바꿔야 하는지**에 대한 정보가 gradient 형태로 계산된다. 다시 말해 "틀린 이유를 모델 내부로 거꾸로 전파하는 과정"이라고 볼 수 있다.

**`optimizer.step()`: 가중치 업데이트**

```python
optimizer.step()
```

마지막으로 옵티마이저가 gradient 정보를 이용해 실제 가중치를 수정한다. 여기서 Adam은 단순히 현재 gradient만 보는 것이 아니라 이전 방향과 크기 정보까지 활용해 더 안정적으로 업데이트를 수행한다.

이 다섯 줄을 하나의 흐름으로 보면 다음과 같다.

1. 이전 gradient를 지운다.
2. 현재 입력으로 예측한다.
3. 정답과 비교해 오차를 계산한다.
4. 오차를 줄이기 위한 방향을 계산한다.
5. 실제로 파라미터를 수정한다.

즉 학습이란 결국 이 과정을 **미니배치마다 반복하면서 손실을 점점 낮추는 작업**이다.

> 팁: 학습이 전혀 되지 않거나 loss가 이상하게 흔들릴 때는, 먼저 이 다섯 줄의 순서가 맞는지 확인하는 것이 좋다. 특히 `zero_grad()` 누락, `backward()`와 `step()` 순서 오류는 초보자가 자주 겪는 실수다.

#### 6) 추론 코드의 핵심: 웹캠 입력을 MNIST 스타일로 맞추기

```python
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_, thresh_roi = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY_INV)
input_tensor = transform(thresh_roi).unsqueeze(0).to(device)
```

이 부분이 매우 중요하다. MNIST는 **검은 배경에 흰 숫자** 형태에 가깝기 때문에 웹캠에서 가져온 이미지를 그대로 넣으면 분포가 크게 달라진다. 따라서 ROI를 흑백으로 바꾸고 `THRESH_BINARY_INV`를 사용해 **배경은 검게, 숫자는 밝게** 만들어 학습 데이터와 비슷한 형태로 맞춘다.

### 데이터 형태 및 전처리 설명

- 입력 데이터: `(N, 1, 28, 28)`
- 채널 수: 1
- 클래스 수: 10
- 정규화: `Normalize((0.1307,), (0.3081,))`

웹캠 추론에서는 다음 순서를 따른다.

1. ROI 추출
2. BGR -> Grayscale
3. Threshold Binary Inverse
4. 28x28 리사이즈
5. Tensor 변환
6. 정규화

> 주의: **학습 때 사용한 전처리와 추론 때 사용한 전처리가 다르면** 성능이 급격히 떨어진다. 특히 MNIST처럼 단순한 데이터셋은 배경색과 글자색 반전만 바뀌어도 결과가 크게 흔들린다.

### 학습 실행 방법

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 101_pytorch_mnist.py
```

학습이 끝나면 다음 파일이 저장된다.

```bash
mnist_mlp.pth
```

### 추론 실행 방법

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 101_pytorch_mnist_inference.py
```

- 웹캠 중앙 사각형 영역에 숫자를 보여준다.
- 화면 상단에 예측 숫자와 신뢰도가 표시된다.
- `q` 키로 종료한다.

### 실무 관점 해설 및 주의사항

MLP는 단순하지만 **PyTorch 훈련 구조를 배우기에 가장 좋은 출발점**이다. 이후 로봇 제어에서도 상태 벡터를 다루는 정책 네트워크는 MLP 기반인 경우가 많다. 즉 이 예제는 이미지 분류이면서 동시에 **정책 네트워크의 기초**도 함께 익히는 셈이다.

> 팁: MNIST 추론을 안정적으로 하고 싶다면 배경을 단순하게 만들고 숫자를 두껍고 명확하게 써 주는 것이 좋다.
### 실제 웹캠 입력과 평가 범위

중앙 ROI는 200×200이고, 회색조 변환 후 임계값 128의 `THRESH_BINARY_INV`로 흰 글씨·검은 배경을 만든다. 이후 28×28과 학습 정규화로 맞춘다. 학습 데이터와 동일한 숫자 분포를 자동으로 보장하는 전처리는 아니다. 카메라의 조명·획 두께·배경에 따라 결과가 달라진다.

`nn.Linear`는 마지막 차원에 작용하므로 다차원 텐서에도 적용할 수 있다. 이 모델은 이미지 전체를 한 벡터로 읽기 위해 명시적으로 `Flatten`을 사용하는 것이다.


---

## 2.2  Pytorch로 CNN 활용하기

### 요약

MNIST MLP가 "픽셀 벡터 분류"였다면, 이 예제는 본격적으로 **이미지의 공간 구조를 활용하는 CNN 분류기**를 다룬다. Conv-BN-ReLU 블록, MaxPool, Dropout, Fully Connected 계층을 조합해 작은 컬러 이미지 분류의 표준 구조를 익히게 된다.

### 소스코드 개요 및 역할 설명

**`102_pytorch_CNN_cifar10.py`**

- CIFAR-10 데이터셋 로드
- ImprovedCNN 정의
- 학습과 테스트 정확도 측정
- 가중치 저장: `cifar10_cnn.pth`

**`102_pytorch_CNN_cifar10_inference.py`**

- 학습된 CNN 로드
- 웹캠 중앙 ROI 추출
- 32x32 리사이즈 후 분류
- 클래스명과 확률 표시

### 코드 블록 상세 설명

#### 1) CIFAR-10 전처리

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)
```

이 부분은 단순히 이미지 변환만 하는 코드가 아니라, **CIFAR-10 학습용 데이터와 평가용 데이터를 준비하고 이를 미니배치 단위로 공급하는 전체 입력 파이프라인**이다.

CIFAR-10은 **32x32 RGB 이미지**로 이루어진 대표적인 소형 컬러 이미지 분류 데이터셋이다. `transform`에서는 먼저 `ToTensor()`로 이미지를 텐서로 바꾸고, 이어서 각 채널에 대해 `(0.5, 0.5, 0.5)` 평균과 표준편차를 사용해 정규화한다. 이렇게 하면 대략 각 채널 값이 `[-1, 1]` 범위 근처로 맞춰져 학습 안정성에 도움이 된다.

#### `torchvision.datasets.CIFAR10(...)`: 데이터셋 로드

```python
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
```

여기서 `train=True`는 학습용 데이터, `train=False`는 테스트용 데이터를 의미한다. `download=True`를 주면 데이터가 없을 때 자동으로 내려받기 때문에 실습 환경에서 매우 편리하다. `transform=transform`을 지정했기 때문에 이미지를 꺼낼 때마다 앞에서 정의한 전처리가 자동 적용된다.

#### `DataLoader`: 배치 단위 공급기

```python
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)
```

`DataLoader`는 데이터셋을 바로 모델에 넣기 좋은 형태로 묶어 주는 역할을 한다.

- `batch_size=64`: 한 번에 64장씩 모델에 전달한다.
- `shuffle=True`: 학습 데이터 순서를 매 epoch마다 섞는다. 이렇게 해야 특정 순서에 모델이 편향되지 않고 더 안정적으로 학습할 수 있다.
- `shuffle=False`: 테스트 데이터는 순서를 섞지 않아도 되므로 보통 그대로 둔다.
- `num_workers=2`: 데이터를 미리 읽어 오는 보조 프로세스 수다. CPU가 여유 있으면 데이터 공급 병목을 줄이는 데 도움이 된다.

#### 학습용과 테스트용을 분리하는 이유

학습 데이터는 모델이 직접 보고 파라미터를 업데이트하는 데 사용된다. 반면 테스트 데이터는 학습에 참여하지 않고 **모델이 처음 보는 데이터에서 얼마나 잘 일반화되는지**를 확인하기 위한 기준점이 된다.

정리하면 이 코드는 단순히 CIFAR-10을 불러오는 것이 아니라 **전처리된 학습셋과 테스트셋을 만들고, 이를 배치 단위로 안정적으로 공급하는 단계**다. 이후의 학습 루프는 바로 이 `trainloader`와 `testloader`를 중심으로 돌아가게 된다.

> 주의: 학습용과 테스트용 모두 같은 정규화 기준을 사용해야 한다. 학습셋에는 한 전처리를 쓰고 테스트셋에는 다른 전처리를 쓰면 평가 결과가 왜곡될 수 있다.

#### 2) ImprovedCNN 구조

```python
self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
self.bn1 = nn.BatchNorm2d(32)
self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
self.bn2 = nn.BatchNorm2d(32)

self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
self.bn3 = nn.BatchNorm2d(64)
self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
self.bn4 = nn.BatchNorm2d(64)
```

ImprovedCNN은 단순히 레이어를 나열한 구조가 아니라, **특징 추출(Feature Extraction)**과 **최종 분류(Classification)**라는 두 역할이 계층적으로 나뉜 CNN이다. 앞부분에서는 이미지에서 유의미한 시각 패턴을 뽑아내고, 뒷부분에서는 그 특징을 종합해 최종 클래스를 결정한다.

이 모델은 크게 다음과 같이 이해하면 된다.

- 특징 추출부: 합성곱 블록 2개 스테이지
- 분류부: 펼치기(flatten) 이후 fully connected 층

즉 앞에서는 "무엇이 보이는가"를 분석하고, 뒤에서는 "그래서 이것이 어떤 클래스인가"를 판정한다.

#### `Conv2d`: 특징의 필터링

합성곱층 `nn.Conv2d(입력채널, 출력채널, 커널크기)`는 이미지 안에서 특정 패턴을 찾는 필터 역할을 한다.

```python
self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
```

각 스테이지의 의미는 다음과 같다.

1. **스테이지 1 (`3 -> 32`)**입력은 RGB 이미지이므로 채널 수가 3이다. 여기서 32개의 필터를 사용해 선, 색 대비, 간단한 경계 같은 **기초 시각 특징**을 추출한다.
2. **스테이지 2 (`32 -> 64`)**
   앞에서 얻은 32개의 특징맵을 다시 조합하여 더 복잡한 64개의 상위 특징을 만든다. 예를 들어 단순한 선과 색 경계가 결합되어 바퀴 형태, 동물의 윤곽, 얼굴 일부 같은 **조합 특징**으로 발전할 수 있다.

즉 CNN은 처음부터 "자동차"나 "고양이"를 바로 찾는 것이 아니라 **단순한 특징에서 복잡한 특징으로 계층적으로 쌓아 올리며** 인식한다.

#### `Conv -> BN -> ReLU`: 비선형 특징 추출 블록

이 코드에서 자주 반복되는 패턴은 다음과 같다.

```python
x = F.relu(self.bn1(self.conv1(x)))
```

이는 사실상 **Conv -> BatchNorm -> ReLU**의 한 세트이며, CNN에서 매우 자주 쓰이는 표준 블록이다.

- **Conv**: 이미지에서 패턴의 강도를 계산한다.
- **BatchNorm(BN)**: 계산 결과가 너무 크거나 작게 치우치지 않도록 분포를 안정화한다.
- **ReLU**: 의미 없는 음수 영역은 0으로 잘라내고 중요한 양수 신호를 다음 층으로 전달한다.

특히 BatchNorm은 로봇 제어에서 센서값을 정규화해 다루기 쉽게 만드는 것과 비슷하게 이해할 수 있다. 입력 분포를 더 안정적으로 맞춰 줌으로써 학습이 덜 흔들리고 수렴도 빨라지는 경우가 많다.

#### 공간 정보의 요약과 통합

각 스테이지 끝에는 `MaxPool2d`와 `Dropout`이 배치되어 있다.

```python
self.pool = nn.MaxPool2d(2, 2)
self.dropout_conv = nn.Dropout(0.3)
```

**`MaxPool2d`: 공간 압축**

풀링은 feature map의 크기를 줄이며 작은 위치 변화나 미세한 노이즈에 덜 민감하도록 만든다. 32x32 이미지는 두 번의 풀링을 거치면서 **32 -> 16 -> 8**로 줄어든다.

이것은 단순한 축소가 아니라 사소한 조명 변화나 먼지 같은 노이즈는 덜 중요하게 보고, **"이 위치 근처에 확실히 의미 있는 특징이 있다"**는 정보는 남기는 과정이라고 볼 수 있다.

**`view()`와 FC: 최종 판단을 위한 변환**

공간 특징 추출이 끝나면 최종 feature map은 `64 x 8 x 8` 형태가 된다. 이 3차원 구조를 다음 단계에서 fully connected 층에 넣기 위해 1차원 벡터로 펼친다.

```python
x = x.view(-1, 64 * 8 * 8)
```

이는 시각 분석을 마친 뒤 이제 최종 판단을 위해 모든 정보를 한 장의 데이터 시트처럼 정리하는 단계라고 이해하면 된다. 이후 FC 레이어는 이 요약된 정보를 바탕으로 "이것은 어떤 클래스인가"를 결정한다.

#### 실무자가 꼭 볼 부분: `padding=1`의 의미

코드의 모든 합성곱층에는 `padding=1`이 들어 있다.

```python
nn.Conv2d(..., kernel_size=3, padding=1)
```

3x3 필터를 적용하면 원래는 가장자리 정보가 잘리면서 출력 크기가 줄어들 수 있다. `padding=1`은 입력 가장자리에 0을 한 줄 둘러 **가장자리 정보가 너무 빨리 손실되지 않게 해 주는 설정**이다.

장점은 다음과 같다.

- 출력 feature map 크기를 보다 안정적으로 유지할 수 있다.
- 이미지 가장자리에 있는 정보도 끝까지 학습에 활용할 수 있다.
- 여러 개의 합성곱층을 쌓을 때 구조 설계가 단순해진다.

정리하면 이 ImprovedCNN은 **작은 필터로 기초 특징을 찾고, BN과 ReLU로 안정적이고 비선형적인 표현을 만들며, MaxPool로 공간 정보를 압축하고, 마지막 FC로 판단을 내리는 구조**다. MLP가 픽셀을 한 줄로 펴서 처리했다면, CNN은 이미지의 2차원 구조를 유지한 채 점점 더 높은 수준의 특징으로 올라간다는 점이 핵심이다.

#### 3) 순전파 구조

```python
x = F.relu(self.bn1(self.conv1(x)))
x = F.relu(self.bn2(self.conv2(x)))
x = self.pool(x)
x = self.dropout_conv(x)

x = F.relu(self.bn3(self.conv3(x)))
x = F.relu(self.bn4(self.conv4(x)))
x = self.pool(x)
x = self.dropout_conv(x)

x = x.view(-1, 64 * 8 * 8)
x = F.relu(self.bn5(self.fc1(x)))
x = self.dropout_fc(x)
x = self.fc2(x)
```

32x32 이미지는 두 번의 `MaxPool(2, 2)`을 거치면서 **32 -> 16 -> 8**로 줄어든다. 따라서 최종 feature map 크기는 `64 x 8 x 8`이 되고, 이를 펼쳐 FC 층으로 연결한다.

#### 4) BatchNorm과 Dropout의 의미

- **BatchNorm**: 학습을 안정화하고 수렴을 빠르게 돕는다.
- **Dropout**: 일부 뉴런을 무작위로 비활성화하여 과적합을 줄인다.

이 조합은 작은 데이터셋을 다룰 때 매우 자주 등장한다.

### 데이터 형태 및 전처리 설명

- 입력 데이터: `(N, 3, 32, 32)`
- 채널 수: 3
- 클래스 수: 10
- 클래스: `plane, car, bird, cat, deer, dog, frog, horse, ship, truck`

웹캠 추론 시 ROI는 원래 더 큰 크기로 잘라내지만 모델 입력에 맞춰 **32x32로 축소**된다.

> 주의: CIFAR-10은 매우 작은 이미지이기 때문에 웹캠에서 실제 물체를 바로 넣으면 도메인 차이 때문에 잘 맞지 않을 수 있다. 이 예제는 **실시간 추론 파이프라인을 체험하는 목적**으로 보는 것이 좋다.

### 학습 실행 방법

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 102_pytorch_CNN_cifar10.py
```

학습 후 저장 파일은 다음과 같다.

```bash
cifar10_cnn.pth
```

### 추론 실행 방법

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 102_pytorch_CNN_cifar10_inference.py
```

- 중앙 ROI를 잘라서 32x32로 변환한다.
- 가장 높은 확률의 클래스를 출력한다.
- `q` 키로 종료한다.

### 실무 관점 해설 및 주의사항

CNN은 이미지의 지역 패턴을 잘 포착하므로 MLP보다 실제 시각 인식 문제에 훨씬 적합하다. 로봇 비전에서도 간단한 물체 상태 분류, 표면 결함 검출, 버튼/레버 상태 판별 같은 과제는 여전히 CNN 계열이 매우 강하다.

> 팁: 웹캠 추론 결과가 불안정하다면 ROI 안에 하나의 대상만 크게 들어오도록 구성하고 배경을 단순하게 만들어 보자.
### 실제 학습 설정과 색상 처리

학습은 batch 64, 10 epoch, Adam learning rate 0.001이다. `DataLoader`의 worker 수는 2이다. 합성곱 dropout은 0.3, FC dropout은 0.5이고 마지막 출력은 로짓 10개이다. 정규화는 채널별 mean/std 0.5이다.

웹캠의 중앙 ROI는 300×300이다. `ToPILImage()`가 BGR을 자동으로 RGB로 바꾸는 것은 아니다. 실제 루프의 `cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)`가 그 변환을 수행한다. `model.eval()`은 BatchNorm을 제거하지 않고 학습 중 저장한 통계로 평가하게 한다.


---

## 2.3 Pytorch 사용자 데이터셋으로 확장하기

### 요약

이 예제부터는 공개 벤치마크를 넘어 **직접 준비한 이미지 폴더**를 학습에 사용한다. 즉 실제 프로젝트에서 가장 자주 마주치는 "내 데이터로 분류기를 만드는 과정"이 시작된다. 또한 데이터 증강을 통해 적은 데이터에서도 어느 정도 일반화를 확보하는 방법을 배운다.

### 소스코드 개요 및 역할 설명

**`103_pytorch_miniVGGnet.py`**

- `ImageFolder` 기반 커스텀 데이터셋 로드
- MiniVGGNet 정의
- 데이터 증강 적용
- 학습 그래프 출력
- 가중치 저장: `miniVGGnet.pth`

**`103_pytorch_miniVGGnet_inference.py`**

- MiniVGGNet 가중치 로드
- 웹캠 중앙 ROI 분류
- 클래스명과 확률 표시

### 코드 블록 상세 설명

#### 1) 데이터셋 구성

```python
dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
```

`ImageFolder`는 폴더 구조만 맞으면 별도의 CSV 없이도 데이터셋을 만들 수 있다. 이 저장소의 실제 클래스 폴더는 다음과 같다.

- `images/cans`
- `images/cups`
- `images/pets`

따라서 클래스 인덱스는 보통 **알파벳순**으로 다음처럼 결정된다.

```python
['cans', 'cups', 'pets']
```

#### 2) 데이터 증강

```python
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

커스텀 데이터는 대개 수량이 적다. 이런 경우 같은 이미지를 조금씩 변형해 보여 주는 **augmentation**이 매우 중요하다.

- `RandomHorizontalFlip()`: 좌우 반전
- `RandomRotation(15)`: 회전
- `ColorJitter()`: 밝기와 대비 변화

이는 **조명 변화, 카메라 각도 변화, 배치 오차**에 강한 모델을 만드는 데 도움을 준다.

#### 3) MiniVGGNet 구조

```python
self.block1 = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout(0.3)
)
```

이 지점에서는 먼저 **VGGNet이 무엇인지**, 그리고 **MiniVGGNet이 왜 필요한지**를 이해하는 것이 중요하다. 단순히 "합성곱층을 몇 개 쌓은 모델"로 보기보다, **작은 필터를 반복적으로 쌓아 표현력을 높이는 설계 철학**으로 이해하면 이후 전이학습 파트와도 자연스럽게 연결된다.

#### VGGNet이란 무엇인가

VGGNet은 옥스퍼드 Visual Geometry Group에서 제안한 대표적인 CNN 계열 모델로, **작은 3x3 합성곱 필터를 여러 층 반복해서 쌓는 구조**로 유명하다. 이전 세대 CNN들이 상대적으로 큰 커널을 쓰는 경우가 많았다면, VGG는 작은 커널을 깊게 쌓아 더 풍부한 비선형 표현을 만들었다.

VGG 계열의 대표적 특징은 다음과 같다.

1. **구조가 단순하고 규칙적이다.**Conv 블록을 반복하고, 중간중간 MaxPool로 해상도를 줄인 뒤, 마지막에 Fully Connected 층으로 분류한다.
2. **3x3 작은 필터를 여러 번 쌓는다.**작은 커널을 연속으로 사용하면 큰 커널 하나를 쓰는 것과 비슷한 수용영역을 얻으면서도 중간에 ReLU 같은 비선형성을 더 자주 삽입할 수 있다.
3. **비전 모델 설계의 교과서 같은 구조다.**
   CNN 입문 단계에서 "블록을 쌓아 깊이를 늘리고, 풀링으로 공간 크기를 줄이며, 마지막에 분류한다"는 감각을 익히기에 매우 좋다.

#### MiniVGGNet이란 무엇인가

MiniVGGNet은 말 그대로 **VGG 스타일의 핵심 아이디어만 남기고 가볍게 축약한 버전**이라고 이해하면 된다. 원래 VGG16, VGG19는 층 수가 깊고 파라미터 수가 많아 작은 커스텀 데이터셋이나 실습 환경에서는 다소 무겁다. 그래서 교육용 예제나 소규모 프로젝트에서는 **구조의 장점은 살리되 계산량은 줄인 MiniVGG 형태**를 많이 사용한다.

이 예제의 MiniVGGNet은 다음 성격을 가진다.

- 작은 3x3 합성곱 반복
- 블록 단위 설계
- MaxPool로 공간 축소
- BatchNorm과 Dropout으로 학습 안정성 및 일반화 보강
- 최종 Fully Connected 분류기 연결

즉 **VGG의 구조적 사고방식을 학습하기 위한 현실적인 축소판**이라고 볼 수 있다.

#### 왜 굳이 VGG 스타일을 배우는가: 존재 의의

MiniVGGNet의 존재 이유는 단순하다. MLP보다 훨씬 이미지답게 동작하면서도 ResNet처럼 구조가 복잡하지 않기 때문이다. 다시 말해 다음 단계로 넘어가기 전, CNN 블록 설계의 감각을 익히는 **중간 다리 역할**을 한다.

존재 의의를 정리하면 다음과 같다.

1. **CNN 블록 설계 원리를 배우기 좋다.**Conv-ReLU-BN-Pool-Dropout 같은 부품이 어떻게 조합되는지 이해하기 쉽다.
2. **커스텀 데이터셋에 바로 적용하기 좋다.**공개 벤치마크가 아니라 `images/` 같은 직접 수집한 데이터에도 비교적 가볍게 적용할 수 있다.
3. **전이학습 이전 단계의 좋은 연습 모델이다.**이후 VGG16, ResNet18 같은 더 큰 사전학습 모델을 다루기 전에 CNN이 특징을 어떻게 쌓아 가는지 감각을 먼저 익힐 수 있다.
4. **실무에서도 작은 분류 문제에 여전히 유용하다.**
   데이터 규모가 크지 않고 클래스 수가 적은 문제에서는 무거운 대형 모델보다 이런 중간 규모 CNN이 더 빠르고 다루기 쉬운 경우가 많다.

#### 이 예제의 MiniVGGNet 구조 읽기

```python
self.block1 = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout(0.3)
)
```

첫 번째 블록은 RGB 입력에서 32개의 특징맵을 만들고, 다시 같은 채널 수에서 한 번 더 특징을 정제한 뒤 풀링으로 공간 크기를 줄인다. 두 번째 블록에서는 채널 수를 64로 늘려 더 복잡한 특징을 표현한다.

이 구조는 다음처럼 읽으면 된다.

- 초기 층: 에지, 색 대비, 단순 윤곽
- 중간 층: 물체 형태, 표면 패턴, 부분 구조
- 후반 분류기: 이 특징들을 종합해 클래스 결정

#### 입력 크기와 최종 차원 변화

입력이 64x64일 때 두 번의 풀링을 거치면 공간 크기는 다음처럼 줄어든다.

- `64 -> 32 -> 16`

따라서 마지막 feature map은 `64 x 16 x 16`이 되고 이를 펼쳐 Fully Connected 층에 넣는다. 그래서 마지막 FC 입력 차원이 `64 * 16 * 16`이 된다.

정리하면 **VGGNet은 작은 3x3 필터를 깊게 쌓는 고전적이면서도 매우 중요한 CNN 설계 철학**이고, **MiniVGGNet은 그 철학을 소형 데이터셋과 교육용 실습에 맞게 경량화한 모델**이다. 이 예제에서 MiniVGGNet을 이해해 두면 다음 절의 VGG16 전이학습이 왜 자연스럽게 이어지는지도 훨씬 명확해진다.

#### 4) 학습 곡선 기록

```python
history = {'loss': [], 'acc': []}
```

이 예제는 epoch별 loss와 accuracy를 저장하고 matplotlib로 그래프를 그린다. 작은 데이터셋 실험에서는 단순히 마지막 정확도만 보는 것보다 **학습 곡선의 모양**을 함께 보는 것이 훨씬 중요하다.

### 데이터 형태 및 전처리 설명

- 입력 데이터: `(N, 3, 64, 64)`
- 데이터 소스: `./images`
- 클래스 폴더: `cans`, `cups`, `pets`
- 정규화: `(0.5, 0.5, 0.5)` 평균과 표준편차

> 주의: 추론 코드의 `class_names`는 **학습 데이터의 클래스 인덱스 순서와 반드시 일치**해야 한다. 예를 들어 폴더가 `cans`, `cups`, `pets`라면 표시 문자열은 `['can', 'cup', 'pet']`처럼 약간 달라도 되지만 **순서만큼은 동일**해야 한다.

#### 5) 추론 파일의 전체 구조 이해하기

`103_pytorch_miniVGGnet_inference.py`는 단순히 모델을 한 번 불러와 예측만 하는 코드가 아니다. **학습 때 사용한 구조를 다시 만들고, 저장된 가중치를 읽어 온 뒤, 웹캠 입력을 같은 전처리 규칙으로 변환해 실시간 분류를 수행하는 전체 추론 파이프라인**이다.

대략적인 흐름은 다음과 같다.

1. 학습 코드와 동일한 `MiniVGGNet` 구조 정의
2. `class_names` 설정
3. 저장된 `miniVGGnet.pth` 가중치 로드
4. `model.eval()`로 추론 모드 전환
5. 학습 때와 맞춘 전처리 정의
6. `cv2.VideoCapture(0)`로 웹캠 실행
7. 중앙 ROI를 잘라 RGB 변환 후 텐서화
8. `torch.no_grad()`에서 예측 수행
9. softmax 확률, 클래스명, 신뢰도 표시
10. `q` 키 입력 시 종료

핵심 코드 흐름은 다음과 같이 정리할 수 있다.

```python
model = MiniVGGNet(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cap = cv2.VideoCapture(0)
```

이 구조에서 특히 중요한 점은 **학습 코드와 추론 코드의 모델 구조가 완전히 같아야 한다는 것**이다. `state_dict`는 가중치 값만 저장하므로 추론 파일에서도 동일한 네트워크 뼈대를 먼저 정의해야 가중치를 올바르게 복원할 수 있다.

또한 추론 루프 내부에서는 다음 단계가 반복된다.

- 화면 중앙 ROI 추출
- `BGR -> RGB` 변환
- `Resize(64, 64)`
- Tensor 변환 및 정규화
- 배치 차원 추가: `unsqueeze(0)`
- 모델 출력 계산
- `softmax`로 확률 해석
- 가장 높은 클래스와 confidence 표시

예를 들어 핵심 추론 부분은 다음과 같은 형태다.

```python
img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
input_tensor = transform(img_rgb).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, 1)
```

여기서 `unsqueeze(0)`는 매우 중요하다. 단일 이미지라도 모델은 보통 **배치 차원까지 포함한 4차원 입력** `(N, C, H, W)`를 기대하므로 `64 x 64` 이미지 1장을 **배치 크기 1개짜리 입력**으로 바꿔 주는 단계다.

결국 이 추론 파일은 "카메라에서 이미지를 읽어 와서, 학습 때와 같은 규칙으로 전처리한 뒤, 저장된 모델로 즉시 분류하는 구조"라고 이해하면 된다. 로봇 응용 관점에서는 이것이 곧 **실시간 시각 인식 노드의 가장 단순한 형태**다.

### 학습 실행 방법

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 103_pytorch_miniVGGnet.py
```

학습 후 저장 파일은 다음과 같다.

```bash
miniVGGnet.pth
```

### 추론 실행 방법

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 103_pytorch_miniVGGnet_inference.py
```

추론 전에 `class_names`를 자신의 데이터셋 순서에 맞게 확인한다.

예시 1:

```python
class_names = ['cans', 'cups', 'pets']
```

예시 2:

```python
class_names = ['can', 'cup', 'pet']
```

단, **순서는 반드시 `cans -> cups -> pets`에 대응**해야 한다.

### 실무 관점 해설 및 주의사항

이 예제는 로봇 비전 실무와 매우 가깝다. 예를 들어 분류 대상이 "캔/컵/반려동물"이 아니라 "부품A/부품B/불량품"이어도 구조는 그대로다. 중요한 것은 모델보다 먼저 **데이터 폴더 구조, 클래스 라벨 일관성, 전처리, 증강 전략**을 안정적으로 잡는 일이다.

> 팁: 클래스별 이미지 수가 너무 불균형하면 정확도 수치가 높아도 실제 성능은 나쁠 수 있다. 커스텀 데이터셋에서는 클래스 균형을 반드시 확인하자.
### 현재 구현의 설정과 평가 범위

기본값은 `IMG_SIZE=64`, `BATCH_SIZE=32`, learning rate 0.001, 100 epoch이다. 두 번째 블록의 dropout은 0.4, FC의 dropout은 0.5이며 FC는 `64*16*16 → 512 → 클래스 수`이다. 실제 순서는 Conv→ReLU→BatchNorm이다.

현재 이 스크립트는 `ImageFolder` 전체로 학습하며 별도 검증 분할을 만들지 않는다. 그래프의 accuracy는 **학습 정확도**이다. 일반화 성능을 확인하려면 별도 평가 데이터가 필요하다. `plt.show()` 창을 닫은 뒤 `miniVGGnet.pth` 저장문에 도달한다.

추론 코드의 기본 표시 이름은 `['CAN', 'CUP', 'PET']`이다. 학습 폴더의 정렬 순서 `cans, cups, pets`에 대응한다. 클래스 이름·순서는 checkpoint에 자동 저장되지 않는다. 학습 데이터의 `class_to_idx`를 확인하고 다른 클래스로 바꾸었다면 추론도 함께 바꿔야 한다.

학습 모드에서 FC의 BatchNorm1d는 배치에 표본이 하나뿐이면 실패한다. 데이터 수를 바꿔 마지막 배치가 1개가 된다면 배치 구성을 조정해야 한다. 원래 예제에는 `drop_last=True` 설정이 없다.


---

## 2.4 Pytorch VGGnet 전이학습 활용하기

### 요약

작은 데이터셋에서는 처음부터 큰 모델을 학습시키는 것보다 **대규모 데이터로 미리 학습된 모델의 특징 추출 능력**을 활용하는 편이 훨씬 안정적이다. 이 예제는 VGG16을 불러와 앞부분은 고정하고, classifier의 마지막 출력층을 교체하는 **전이학습의 가장 전형적인 방식**을 보여 준다.

### 소스코드 개요 및 역할 설명

**`104_pytorch_VGGnet_transfer_feature_can.py`**

- VGG16 사전학습 모델 로드
- `features` 고정
- 마지막 classifier 층 교체
- 커스텀 데이터셋 학습
- 가중치 저장: `vgg16_transfer_model.pth`

**`104_pytorch_VGGnet_transfer_feature_inference.py`**

- VGG16 구조 재생성
- 저장된 가중치 로드
- 웹캠 ROI를 224x224로 맞춰 실시간 추론

### 코드 블록 상세 설명

#### 1) VGG16 사전학습 모델 로드

```python
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
```

이는 ImageNet으로 사전학습된 VGG16을 불러온다는 뜻이다. 이미 모델은 일반적인 에지, 질감, 형태 패턴을 어느 정도 배운 상태다.

#### 2) 특징 추출기 고정

```python
for param in model.features.parameters():
    param.requires_grad = False
```

합성곱 feature extractor의 파라미터를 고정하고 classifier 전체를 새로운 문제에 맞춰 학습한다. 이 전략은 데이터가 적을 때 특히 유리하다.

#### 3) 마지막 층 교체

```python
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, num_classes)
```

VGG16의 마지막 출력층은 원래 ImageNet 1000클래스를 위한 것이다. 이를 현재 문제의 클래스 수에 맞게 바꿔야 한다.

#### 4) ImageNet 정규화의 의미

```python
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

이 값은 ImageNet 데이터셋의 평균과 표준편차다. 사전학습 모델은 이 통계에 맞춰 입력을 받아 왔기 때문에 전이학습에서도 같은 정규화를 쓰는 것이 매우 중요하다.

### 데이터 형태 및 전처리 설명

- 입력 크기: `224 x 224`
- 채널 수: 3
- 정규화: ImageNet mean/std
- 클래스 수: `len(dataset.classes)`

224x224를 쓰는 이유는 VGG16이 이 크기에서 가장 자연스럽게 동작하도록 설계되었기 때문이다. 반드시 정확히 224여야 하는 것은 아니지만 이 예제에서는 구조와 관례를 맞추기 위해 224x224를 사용한다.

> 주의: **사전학습 모델은 입력 정규화가 다르면 성능이 크게 흔들릴 수 있다.** 커스텀 데이터라도 전이학습 시에는 먼저 ImageNet 기준 전처리를 따르는 것이 기본이다.

### 학습 실행 방법

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 104_pytorch_VGGnet_transfer_feature_can.py
```

학습 후 저장 파일은 다음과 같다.

```bash
vgg16_transfer_model.pth
```

### 추론 실행 방법

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 104_pytorch_VGGnet_transfer_feature_inference.py
```

- 중앙 ROI를 잘라 `224 x 224`로 리사이즈한다.
- ImageNet 기준 정규화를 적용한 뒤 분류를 수행한다.
- 예측 클래스와 신뢰도를 화면에 표시한다.
- `q` 키로 종료한다.

### 실무 관점 해설 및 주의사항

이 예제의 핵심은 **모든 층을 처음부터 다시 학습하지 않아도 충분히 강한 출발점을 얻을 수 있다**는 점이다. 작은 커스텀 데이터셋에서는 이런 전이학습 방식이 학습 안정성, 수렴 속도, 최종 성능 면에서 매우 유리한 경우가 많다.

또한 VGG16은 구조가 단순해 전이학습의 원리를 설명하기에 좋다. 즉 이 절은 단지 VGG16을 배우는 것이 아니라, **사전학습 표현을 새로운 작업으로 옮겨 오는 사고방식 자체**를 익히는 단계라고 볼 수 있다.

> 팁: 데이터가 매우 적을 때는 처음부터 모든 층을 풀기보다, 먼저 현재 코드처럼 classifier를 학습해 보고 성능을 확인한 뒤 필요한 경우에만 더 깊은 층을 미세 조정하는 것이 안전하다.
### 이 예제에서 실제로 갱신하는 범위

`classifier[6]`만 새 Linear 층으로 교체하지만 optimizer는 `model.classifier.parameters()`를 받는다. 따라서 **classifier의 앞쪽 기존 Linear 층도 함께 학습**한다. 마지막 Linear 하나만 학습한다고 설명하면 코드와 다르다. `features` 파라미터만 `requires_grad=False`이다.

기본값은 batch 16, 20 epoch, Adam learning rate 0.0001이다. 학습 증강은 좌우 반전과 ±10도 회전이다. `ReduceLROnPlateau`는 patience 3, factor 0.1이며 **학습 loss**를 받는다. 이 파일은 별도 검증 데이터셋을 만들지 않는다. 그래프를 닫으면 `vgg16_transfer_model.pth`가 저장된다.

추론은 400×400 중앙 ROI를 224×224로 바꾸며, 표시 색상의 기준은 softmax 최고 점수 0.7이다. 이 임계값은 학습 정확도나 확률 보정이 검증되었다는 뜻이 아니다.


---

## 2.5 ResNet으로 파인튜닝까지 적용하기

### 요약

이 예제는 전이학습을 한 단계 더 발전시켜, **사전학습된 ResNet18을 먼저 특징 추출기로 사용한 뒤 이후 전체 네트워크를 조금 더 세밀하게 미세 조정하는 2단계 학습 전략**을 다룬다. 데이터가 많지 않은 상황에서도 비교적 안정적으로 적응할 수 있어, 실무에서 매우 자주 쓰이는 접근이다.

### ResNet의 간단한 역사와 VGG 대비 장점

ResNet은 2015년 Microsoft Research에서 발표된 모델로, 매우 깊은 신경망 학습을 실용적으로 만든 전환점 같은 구조다. 핵심 아이디어는 **residual connection**, 즉 입력 정보를 중간 블록을 건너뛰어 뒤쪽으로 전달하는 skip connection에 있다.

이 구조 덕분에 깊은 네트워크에서도 gradient가 비교적 잘 흐르고, 층이 깊어질수록 오히려 학습이 어려워지는 문제를 크게 완화할 수 있었다. 그래서 ResNet 계열은 이후 수많은 비전 모델의 기본 백본으로 자리 잡았다.

VGG와 비교하면 차이가 더 분명하다.

- **VGG**는 구조가 단순하고 직관적이어서 학습용 설명에 매우 좋다.
- **ResNet**은 더 깊어도 안정적으로 학습되며, 실제 전이학습 실무에서 훨씬 널리 쓰인다.
- **VGG**는 파라미터 수가 많고 무거운 편이다.
- **ResNet**은 residual block 덕분에 깊이에 비해 효율적이고, 성능과 실용성의 균형이 좋다.

즉 VGG가 "전이학습의 원리를 배우는 교과서"에 가깝다면, ResNet은 "실제로 현업에서 더 자주 꺼내 드는 실전형 백본"에 가깝다.

### 소스코드 개요 및 역할 설명

**`105_pytorch_Resnet_transfer_fine.py`**

- ResNet18 사전학습 모델 로드
- 마지막 FC 층을 현재 클래스 수에 맞게 교체
- 1단계: feature extraction 학습
- 2단계: 전체 네트워크 fine-tuning
- 학습/검증 정확도 그래프 기록
- 가중치 저장: `resnet18_final.pth`

**`105_pytorch_Resnet_transfer_fine_inference.py`**

- ResNet18 구조 재생성
- 저장된 가중치 로드
- 웹캠 ROI 기반 실시간 추론
- 클래스명과 신뢰도 표시

### 코드 블록 상세 설명

#### 1) ResNet18 불러오기와 마지막 층 교체

```python
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
```

ResNet18은 ImageNet으로 사전학습된 대표적인 경량 백본이다. 마지막 `fc` 층은 원래 1000개 클래스를 분류하도록 만들어져 있으므로, 현재 데이터셋의 클래스 수에 맞게 새로 바꿔야 한다.

이때 앞부분은 일반 시각 특징을 이미 잘 배운 상태이고, 마지막 FC는 현재 문제에 맞게 새로 적응해야 하는 부분이라고 이해하면 된다.

#### 2) 1단계 학습: Feature Extraction

```python
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True
```

첫 단계에서는 대부분의 층을 고정하고 마지막 분류기만 학습한다. 이렇게 하면 작은 데이터셋에서도 학습이 덜 흔들리고, 이미 배워 둔 일반 특징 표현을 안정적으로 활용할 수 있다.

이 단계의 목적은 다음과 같다.

- 기존 사전학습 표현을 최대한 보존한다.
- 새 데이터셋의 클래스 경계만 먼저 빠르게 맞춘다.
- 작은 데이터에서 과도한 가중치 변형을 방지한다.

#### 3) 2단계 학습: Fine-Tuning

```python
for param in model.parameters():
    param.requires_grad = True
```

이후에는 전체 레이어를 다시 열고 더 작은 학습률로 미세 조정한다. 즉 첫 단계가 "마지막 분류기 적응"이라면, 두 번째 단계는 **전체 표현 공간을 현재 데이터에 조금 더 맞춤 조정하는 과정**이다.

이 예제에서는 보통 다음과 같은 전략을 사용한다.

- 1단계: 비교적 큰 학습률로 FC만 학습
- 2단계: 더 작은 학습률로 전체 네트워크 fine-tuning

이 전략이 중요한 이유는, 처음부터 모든 층을 큰 learning rate로 건드리면 사전학습된 좋은 표현이 쉽게 무너질 수 있기 때문이다.

#### 4) 학습/검증 분리와 정확도 추적

```python
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_split = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)
eval_dataset = datasets.ImageFolder(root=DATA_PATH, transform=eval_transform)
val_dataset = Subset(eval_dataset, val_split.indices)
```

수정본은 학습용 ImageFolder와 검증용 ImageFolder를 별도로 만든다. `eval_transform`은 Resize(224,224), ToTensor, ImageNet Normalize만 포함하며 랜덤 반전을 사용하지 않는다. `Subset`으로 동일한 파일 목록에서 서로 겹치지 않는 인덱스를 선택하고 두 단계 학습 동안 정확도를 추적한다. 작은 데이터셋에서는 특히 **학습 정확도만 보는 것이 위험**하다. 학습셋에서는 좋아 보여도 실제 일반화 성능은 낮을 수 있기 때문이다.

그래서 train accuracy와 val accuracy를 함께 기록하면 다음을 더 쉽게 판단할 수 있다.

- 과적합이 시작되는 시점
- fine-tuning이 실제로 도움이 되었는지 여부
- epoch를 더 늘릴지 멈출지에 대한 판단 근거

#### 5) 추론 파일의 구조

`105_pytorch_Resnet_transfer_fine_inference.py`는 학습된 ResNet18을 다시 만들고, 저장된 `resnet18_final.pth`를 불러온 뒤, 웹캠 중앙 ROI를 `224 x 224` 입력으로 맞춰 실시간 분류를 수행한다.

대략적인 흐름은 다음과 같다.

1. ResNet18 구조 정의 및 마지막 FC 교체
2. 저장된 가중치 로드
3. `model.eval()` 설정
4. ImageNet 기준 전처리 정의
5. 중앙 ROI 추출
6. `torch.no_grad()`에서 예측 수행
7. softmax 확률 계산 후 클래스와 confidence 표시

핵심은 VGG16 전이학습 절과 마찬가지로, **사전학습 백본을 사용할 때는 입력 크기와 정규화 규칙을 반드시 맞춰야 한다**는 점이다.

### 데이터 형태 및 전처리 설명

- 입력 크기: `224 x 224`
- 채널 수: 3
- 데이터 소스: `./images`
- 클래스 수: `len(dataset.classes)`
- 정규화: ImageNet mean/std

ResNet18 역시 ImageNet 사전학습 모델이므로 `224 x 224`와 ImageNet 정규화를 그대로 따르는 것이 기본이다. 추론 코드에서 중앙 ROI를 더 크게 자른 뒤 다시 `224 x 224`로 맞추는 이유도 여기에 있다.

> 주의: fine-tuning에서는 학습률을 지나치게 크게 두면 사전학습된 표현이 빠르게 손상될 수 있다. 특히 작은 데이터셋일수록 두 번째 단계의 learning rate는 더 조심스럽게 설정하는 편이 좋다.

### 학습 실행 방법

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 105_pytorch_Resnet_transfer_fine.py
```

학습 후 저장 파일은 다음과 같다.

```bash
resnet18_final.pth
```

### 추론 실행 방법

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 105_pytorch_Resnet_transfer_fine_inference.py
```

- 중앙 ROI를 잘라 `224 x 224` 입력으로 맞춘다.
- 가장 높은 확률의 클래스를 표시한다.
- confidence 기준에 따라 박스 색을 다르게 표시할 수 있다.
- `q` 키로 종료한다.

### 실무 관점 해설 및 주의사항

이 예제는 작은 데이터셋에서 **전이학습과 fine-tuning을 어떻게 단계적으로 적용하는지**를 잘 보여 준다. 로봇 비전에서도 새로운 부품 분류, 작업 상태 판별, 간단한 조립 단계 인식처럼 데이터가 많지 않은 문제에서는 이런 접근이 매우 효과적이다.

특히 "먼저 마지막 층만 학습 -> 이후 전체 미세 조정"이라는 흐름은, Transformer 계열 비전 인코더를 다룰 때도 그대로 응용되는 매우 중요한 실전 감각이다.

> 팁: 검증 정확도가 더 이상 오르지 않거나 오히려 떨어지기 시작하면, epoch를 더 늘리기보다 learning rate와 증강 전략, 데이터 수를 먼저 점검하는 편이 더 효과적이다.

---

## 공통 실행 명령 모음

아래는 이 장의 예제를 빠르게 실행하기 위한 명령 모음이다.

### 101. MNIST MLP

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 101_pytorch_mnist.py
python 101_pytorch_mnist_inference.py
```

### 102. CIFAR-10 CNN

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 102_pytorch_CNN_cifar10.py
python 102_pytorch_CNN_cifar10_inference.py
```

### 103. MiniVGGNet 커스텀 데이터셋

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 103_pytorch_miniVGGnet.py
python 103_pytorch_miniVGGnet_inference.py
```

### 104. VGG16 전이학습

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 104_pytorch_VGGnet_transfer_feature_can.py
python 104_pytorch_VGGnet_transfer_feature_inference.py
```

### 105. ResNet18 2단계 학습

```bash
cd /home/robo/so101_imitation_learning/01_02_pytorch_basic
python 105_pytorch_Resnet_transfer_fine.py
python 105_pytorch_Resnet_transfer_fine_inference.py
```

---

## 장 마무리

이 장에서는 MLP에서 출발해 CNN, MiniVGGNet, VGG16 전이학습, ResNet18 fine-tuning까지 이어지는 흐름을 따라가며 **비전 모델이 어떻게 점진적으로 강해지는지**를 살펴보았다.

핵심은 단순히 모델 이름을 외우는 것이 아니라 다음 원리를 연결해서 이해하는 데 있다.

- 입력 전처리는 학습과 추론에서 일관되어야 한다.
- 모델 구조가 깊어질수록 더 풍부한 특징을 표현할 수 있다.
- 데이터가 적을수록 사전학습 표현을 활용하는 것이 유리하다.
- 작은 데이터셋에서는 단계적 fine-tuning이 특히 중요하다.
- 실시간 추론 코드는 결국 학습 파이프라인과 같은 규칙 위에서 동작해야 한다.

이 기초가 잡혀 있어야 3장의 MuJoCo 관측·명령과 4~6장의 순차 정책 학습 구조도 자연스럽게 받아들일 수 있다.
## 수정본의 실제 설정과 동작

1단계는 FC만 Adam learning rate 0.001로 10 epoch 학습한다. 2단계는 전체 파라미터를 Adam learning rate 0.0001로 15 epoch 학습한다. batch size는 16이고 학습/검증 비율은 80/20이다. seed 42는 분할용 generator에 적용되며 전체 학습의 완전한 재현성을 보장하는 설정은 아니다.

`requires_grad=False`만으로 BatchNorm의 running mean/variance가 고정되지는 않는다. 수정본은 `Feature-Ex` 단계에서 `model.train()` 다음 BatchNorm2d만 `eval()`로 바꿔 통계를 고정한다. Fine-Tuning에서는 전체 학습 모드로 통계도 갱신한다.

모델 summary는 최종 FC 교체와 `model.to(device)` 다음에 `device=device.type`으로 실행한다. 모델과 summary 입력의 CPU/CUDA 불일치를 피하기 위한 수정이다. summary 직전에는 `model.eval()`을 적용하고 이후 `model.train()`으로 복구하여 구조 확인용 임의 입력이 사전학습 BatchNorm 통계를 바꾸지 않게 했다. 이 수정은 CPU에서 실제 summary 호출과 통계 보존까지 검사했으며 GPU 실기 검증을 수행한 것은 아니다.

검증 정확도 최고 시점의 모델을 따로 저장하는 구현은 없다. `resnet18_final.pth`는 마지막 epoch의 가중치이며, 그래프 창을 닫은 뒤 저장한다. 수정본으로 재학습하면 같은 파일명에 저장하므로 기존 실험 결과를 유지하려면 먼저 복사해 둔다. 이번 검증에서는 기존 가중치를 덮어쓰지 않았다.
