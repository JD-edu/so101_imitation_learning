# MuJoCo 양팔 로봇 카메라 뷰 설정

이 문서는 저장소 최상위의 `tests/wide_camera_test.py`가 제안하는 카메라 배치와 `so101_real_leader_mujoco_il/viewer.py`의 실제 MuJoCo 카메라 설정을 연결해서 설명한다.

ArUco 기준점의 생성·XML 배치·ACT 및 miniVLA 연결은 [`aruco_reference/README.md`](aruco_reference/README.md)를 참고한다.

## 1. 권장 배치

`tests/wide_camera_test.py`는 양팔 로봇의 뒤쪽 위에 카메라를 놓고 작업대 앞쪽을 내려다보는 ALOHA 스타일 시점을 제안한다.

```text
카메라 위치: [-0.10, 0.00, 0.65] m
시선 목표점: [ 0.45, 0.00, 0.03] m
하향 각도: 약 48.4도
카메라-목표 거리: 약 0.83 m
권장 수직 화각: 55~65도
```

이 모델에서 로봇의 작업 방향은 `+X`이다. 따라서 카메라는 로봇 뒤쪽인 `-X`에 있고 `+X` 방향을 바라봐야 한다. `Y=0`은 좌우 양팔 사이의 중앙이다.

```text
                       +X, 작업대 앞쪽
                              ↑
 카메라(-X, 높은 Z)  →  좌측 팔  작업물  우측 팔
                              │
                         Y=0 중앙선
```

## 2. 실제로 수정할 파일

데이터 수집과 ACT 추론이 사용하는 카메라는 `viewer.py`의 `GLFWFollowerViewer`에 있다.

```text
09_10_so101_mujoco_imitation_learning/
└── 104_so101_real_leader_mujoco_il/
    └── so101_real_leader_mujoco_il/
        └── viewer.py
```

양팔이 `Y=0`을 중심으로 좌우에 배치된 경우 권장 초기값은 다음과 같다.

```python
self.camera = mujoco.MjvCamera()
self.camera.lookat[:] = [0.45, 0.0, 0.03]
self.camera.distance = 0.83
self.camera.azimuth = 0
self.camera.elevation = -48.4
```

각 값의 의미는 다음과 같다.

| 설정 | 의미 | 값을 바꿨을 때 |
|---|---|---|
| `lookat=[x,y,z]` | 화면 중앙이 바라보는 작업점 | 작업 영역 전체를 평행 이동해서 맞춘다. |
| `distance` | 카메라와 `lookat` 사이 거리 | 키우면 더 넓게 보이고 로봇이 작아진다. |
| `azimuth` | Z축 주위의 수평 회전 각도 | `0`은 뒤쪽 `-X`에서 앞쪽 `+X`를 본다. |
| `elevation` | 수평면 기준 상하 각도 | 음수의 절댓값을 키우면 더 위에서 내려다본다. |

현재 `viewer.py`에는 다음 값이 들어 있다.

```python
self.camera.lookat[:] = [0.45, -0.15, 0.03]
self.camera.distance = 0.83
self.camera.azimuth = 0
self.camera.elevation = -48.4
```

이 설정은 뒤쪽 상단에서 앞을 바라보는 방향은 맞지만 화면 중심이 `Y=-0.15`로 치우쳐 있다. 한쪽 팔을 중심으로 볼 때는 적절할 수 있다. 양팔이 `Y=-0.2`, `Y=+0.2`처럼 대칭 배치되어 있다면 다음과 같이 중앙값 `Y=0.0`으로 바꾼다.

```python
self.camera.lookat[:] = [0.45, 0.0, 0.03]
```

두 팔의 중심이 `Y=0`이 아니라면 실제 두 베이스 또는 작업 영역의 Y 좌표 평균을 사용한다.

```python
center_y = (left_arm_y + right_arm_y) / 2
self.camera.lookat[:] = [0.45, center_y, 0.03]
```

## 3. `wide_camera_test.py` 값과 MuJoCo 값의 관계

테스트 그림의 카메라 위치와 목표점 차이는 다음과 같다.

```text
dx =  0.45 - (-0.10) =  0.55
dy =  0.00 -   0.00  =  0.00
dz =  0.03 -   0.65  = -0.62
```

전체 거리는 약 `0.83 m`이고 하향 각도는 약 `48.4도`이다. 이를 MuJoCo 자유 카메라로 표현하면 다음 값이 된다.

```text
lookat   = [0.45, 0.00, 0.03]
distance = 0.83
azimuth  = 0
elevation = -48.4
```

반대로 `azimuth=180`이면 카메라가 작업대 앞쪽 `+X`로 이동하여 로봇 쪽 `-X`을 바라본다. 뒤에서 앞을 보는 양팔 시점과 반대가 되므로 이 구성에서는 사용하지 않는다.

현재 모델로 계산한 대략적인 카메라 위치는 다음과 같다.

| `azimuth` | 카메라 위치 | 바라보는 수평 방향 |
|---:|---|---|
| `0` | `[-0.10, lookat_y, 0.65]` | `+X`, 로봇 뒤에서 앞 |
| `90` | `[0.45, -0.55, 0.65]` | `+Y`, 측면 |
| `180` | `[1.00, lookat_y, 0.65]` | `-X`, 로봇 앞에서 뒤 |
| `270` | `[0.45, +0.55, 0.65]` | `-Y`, 반대 측면 |

## 4. 화각 설정

`tests/wide_camera_test.py`는 수직 화각 `55~65도`를 권장한다. 현재 자유 카메라의 화각은 `lift_cube_calibration.xml`의 `<visual><global>`에서 설정한다.

```xml
<visual>
    <global azimuth="90"
            elevation="-20"
            fovy="60"
            offwidth="1280"
            offheight="720"/>
</visual>
```

여기서 `fovy="60"`을 추가하면 된다. `viewer.py`가 별도의 `MjvCamera`를 넘기므로 XML의 `azimuth`와 `elevation`은 최종 수집 시점을 결정하지 않는다. 자유 카메라의 방향은 `viewer.py` 값이 결정하고, `fovy`는 XML의 전역 시각 설정을 사용한다.

화면 양옆에서 팔이 잘리면 다음 순서로 조정한다.

1. `lookat`의 Y를 두 팔 사이 중앙으로 맞춘다.
2. `distance`를 `0.83 → 0.95 → 1.05`처럼 조금씩 늘린다.
3. 필요하면 `fovy`를 `60 → 65`로 늘린다.
4. 로봇 링크가 물체를 가리면 `elevation`을 `-48.4 → -52`처럼 더 낮춘다.

화각을 지나치게 넓히면 가장자리 왜곡이 커지므로 먼저 중심과 거리를 조절한다.

## 5. XML에 있는 다른 카메라가 적용되지 않는 이유

현재 데이터 수집과 추론은 `viewer.py`에서 다음과 같이 자유 카메라 객체를 직접 전달한다.

```python
mujoco.mjv_updateScene(
    self.model,
    self.data,
    self.option,
    None,
    self.camera,
    mujoco.mjtCatBit.mjCAT_ALL.value,
    self.scene,
)
```

따라서 현재 파이프라인의 `observation.images.overview`에는 다음 XML 카메라가 자동으로 사용되지 않는다.

- `lift_cube_calibration.xml`의 `origin_cam`
- `so101_new_calib.xml`의 `wrist_camera_sensor`
- `<visual><global>`의 `azimuth`, `elevation`

고정 XML 카메라를 사용하려면 `MjvCamera`의 타입과 고정 카메라 ID를 선택하도록 렌더링 코드를 별도로 바꿔야 한다. 현재 구조에서는 `viewer.py`의 자유 카메라를 조절하는 것이 가장 직접적이다.

## 6. 개념도를 실행하는 방법

`tests/wide_camera_test.py`는 실제 MuJoCo 장면을 렌더링하는 테스트가 아니다. 권장 카메라 위치, 시선과 차폐 관계를 Matplotlib 그림으로 보여주는 설계 보조 도구이다.

저장소 최상위에서 실행한다.

```bash
cd /home/robo/so101_imitation_learning
conda activate lerobot
python tests/wide_camera_test.py
```

실행하면 권장 위치와 각도가 터미널에 출력되고 2D 측면도와 3D 배치도가 열린다. 이 파일의 값을 바꿔도 MuJoCo 카메라가 자동으로 바뀌지는 않는다. 확정한 값을 `viewer.py`와 필요하면 XML의 `fovy`에 직접 반영해야 한다.

## 7. 실제 MuJoCo 화면으로 확인하는 방법

카메라 값을 바꾼 뒤 수집 명령으로 화면을 확인한다.

```bash
cd /home/robo/so101_imitation_learning/09_10_so101_mujoco_imitation_learning
conda activate lerobot

python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/camera_view_test \
  --repo-id local/camera_view_test \
  --leader-port /dev/ttyUSB0
```

`r`을 누르지 않으면 episode가 녹화되지 않는다. 화면 구도만 확인한 뒤 `q`로 종료한다. 실제 장치가 `/dev/ttyACM0`이면 `--leader-port`를 바꾼다.

확인할 항목은 다음과 같다.

- 두 팔의 베이스, 손목과 그리퍼가 화면 안에 들어오는가?
- 작업 물체와 양팔 사이의 조작 공간이 중앙에 있는가?
- 로봇 링크가 물체를 오래 가리지 않는가?
- 팔을 작업 영역 끝까지 움직여도 화면 밖으로 나가지 않는가?
- 물체 크기가 ACT 입력 해상도 `224×224`에서도 구별되는가?

## 8. 데이터 수집과 추론의 시점 일치

`dataset.py`와 `inference.py`는 모두 같은 `GLFWFollowerViewer.render_and_capture()`를 사용한다. 따라서 `viewer.py`를 수정하면 이후 수집 영상과 추론 영상에 모두 적용된다.

## 조작용 정면 사선 보조 창

정책이 사용하는 기존 overview 시점은 유지하면서, 사람이 큐브를 잡을 때만 정면 사선 보조 창을 추가로 열 수 있다.

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_assist \
  --repo-id local/so101_real_leader_assist \
  --leader-port /dev/ttyUSB0 \
  --assist-view
```

보조 카메라는 `viewer.py`의 다음 값으로 설정한다.

```python
ASSIST_LOOKAT = (0.32, 0.0, 0.12)
ASSIST_DISTANCE = 0.65
ASSIST_AZIMUTH = 150.0
ASSIST_ELEVATION = -30.0
```

완전 정면인 `azimuth=180`에서 30도 비껴 본 각도라서 그리퍼와 큐브의 전후 관계를 보기 쉽다. 보조 창은 별도 `MjvScene`과 `MjvCamera`를 사용하며 화면 표시만 담당한다. `observation.images.overview`, ArUco 검출과 데이터셋에는 기존 뒤쪽 카메라만 들어간다.

`--assist-view`를 생략하면 두 번째 창을 생성하지 않는다. 보조 창만 닫으면 overview 창과 데이터 수집은 계속되고, 기존 OpenCV observation 창에서 `q`를 누르면 전체 실행을 종료한다.

카메라를 확정한 뒤 새 데이터셋을 수집해야 한다. 기존 데이터가 `azimuth=180` 또는 다른 `lookat`으로 수집되었다면 새 시점으로 학습할 데이터도 다시 수집한다. 학습 영상과 추론 영상의 시점이 다르면 ACT가 학습하지 않은 영상 분포를 입력받아 성능이 크게 떨어질 수 있다.

같은 실험에서는 다음 설정을 고정해서 기록한다.

```text
lookat, distance, azimuth, elevation, fovy,
렌더링 해상도, 로봇 배치, 작업대와 물체 위치
```

권장 시작값은 다음과 같다.

```python
# viewer.py
self.camera.lookat[:] = [0.45, 0.0, 0.03]
self.camera.distance = 0.83
self.camera.azimuth = 0
self.camera.elevation = -48.4
```

```xml
<!-- lift_cube_calibration.xml -->
<global azimuth="90" elevation="-20" fovy="60"
        offwidth="1280" offheight="768"/>
```
