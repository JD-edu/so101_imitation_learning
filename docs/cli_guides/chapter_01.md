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
