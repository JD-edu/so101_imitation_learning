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
