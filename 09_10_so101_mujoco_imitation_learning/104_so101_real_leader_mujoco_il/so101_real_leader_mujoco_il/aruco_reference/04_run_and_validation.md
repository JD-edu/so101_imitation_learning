# 4. 실행·검증·문제 해결

## 환경

모든 명령은 `pyproject.toml`이 있는 폴더에서 실행한다.

```bash
cd /home/robo/so101_imitation_learning/09_10_so101_mujoco_imitation_learning
conda activate lerobot
python -m pip install -e '.[train,test]' --no-build-isolation
```

확인:

```bash
which python
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"
```

사용한 환경에서는 OpenCV 4.13.0과 `cv2.aruco`를 확인했다. `aruco=False`이면 ArUco 모듈이 포함된 OpenCV 빌드가 필요하다.

## 순서별 명령

### 1. PNG 생성

```bash
python -m so101_real_leader_mujoco_il.aruco_reference.generate_marker
```

### 2. PNG 자체 검출

```bash
python -m so101_real_leader_mujoco_il.aruco_reference.inspect_image \
  104_so101_real_leader_mujoco_il/so101_real_leader_mujoco_il/assets/aruco_6x6_250_id23.png \
  --output aruco_marker_check.png
```

### 3. MuJoCo 렌더 검출

```bash
MUJOCO_GL=egl python \
  -m so101_real_leader_mujoco_il.aruco_reference.verify_scene \
  --output aruco_scene_check.png
```

### 4. 실제 리더 데이터 수집

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_aruco_state \
  --repo-id local/so101_real_leader_aruco_state \
  --leader-port /dev/ttyUSB0 \
  --aruco-state --aruco-id 23 \
  --assist-view
```

장치가 `/dev/ttyACM0`이면 포트를 바꾼다. `r`은 녹화, `s`는 저장, `d`는 폐기, `q`는 종료다.

`--assist-view`는 정면 사선의 두 번째 MuJoCo 창을 표시한다. 화면 확인용이며 데이터에는 저장되지 않는다. 필요 없으면 옵션을 생략한다. 보조 창만 닫아도 수집은 계속된다.

### 5. episode 확인

```bash
python -m so101_real_leader_mujoco_il.cli inspect \
  --dataset-root data/real_leader_aruco_state \
  --repo-id local/so101_real_leader_aruco_state \
  --episode 0 --delay-ms 33
```

### 6. ACT 학습

```bash
python -m so101_real_leader_mujoco_il.cli train \
  --dataset-root data/real_leader_aruco_state \
  --repo-id local/so101_real_leader_aruco_state \
  --output outputs/act_real_leader_aruco_state \
  --steps 100000 --batch-size 32 --device cuda
```

### 7. ACT 추론

```bash
python -m so101_real_leader_mujoco_il.cli infer \
  --checkpoint outputs/act_real_leader_aruco_state \
  --device cuda --aruco-id 23
```

## 자동 검사

ArUco 전용 검사:

```bash
python -m unittest -v \
  so101_real_leader_mujoco_il.aruco_reference.test_aruco_reference
```

검사 항목:

1. 생성한 marker를 ID 23으로 검출
2. marker가 없으면 `visible=0`인 영벡터 반환
3. XML에 reference body, geom과 sensor 존재
4. collect·infer CLI에 ArUco 및 보조 카메라 옵션 존재
5. 보조 카메라가 완전 정면에서 비껴 난 앞쪽 위치인지 확인

headless 장면 검증은 전용 unittest와 별도로 실행한다.

```bash
MUJOCO_GL=egl python \
  -m so101_real_leader_mujoco_il.aruco_reference.verify_scene \
  --output /tmp/aruco_scene_check.png
```

## 2026-09-30 검증 결과

| 검사 | 결과 |
|---|---|
| 생성 PNG에서 ID 23 검출 | 통과 |
| 104 XML 로드 | 통과 |
| 1024×768 렌더 후 224×224 변환 영상 검출 | 통과 |
| ArUco·보조 카메라 단위 검사 | 5개 통과 |
| 기존 103·104 파이프라인 함수 검사 | 6개 통과 |
| 6차원 environment-state를 포함한 ACT forward/backward | 통과 |
| overview·정면 사선 카메라 headless 렌더 | 통과 |
| 수정 패키지 Python compile | 통과 |

실제 리더 하드웨어 수집과 100,000-step 전체 학습은 수행하지 않았다. 현재 자동화 환경은 X11 창 접근이 차단되어 두 GLFW 창을 화면에 띄우는 검사는 수행하지 못했고, 두 카메라의 scene·방향·렌더 결과를 headless로 검사했다. 현재 환경에는 pytest 모듈이 없어 기존 pytest 형식 함수 6개를 같은 Python 환경에서 직접 실행했다.

검증된 MuJoCo frame의 예시 특징은 다음과 같았다.

```text
visible = 1
center_x_normalized = 0.0
center_y_normalized ≈ 0.574
side_length_ratio ≈ 0.127
```

## 문제 해결

### `cv2`에 `aruco`가 없음

현재 Python과 OpenCV 설치를 확인한다.

```bash
which python
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"
```

시스템 `/usr/bin/python3` 대신 `lerobot` 환경의 Python을 사용한다.

### XML에서 texture를 읽지 못함

- JPG가 아닌 생성된 PNG를 사용한다.
- PNG가 `assets/lift_cube_calibration.xml`과 같은 디렉터리에 있는지 확인한다.
- 파일명과 XML `file=` 값이 같은지 확인한다.

### `visible=false`

- XML의 marker ID와 `--aruco-id`를 확인한다.
- 마커 네 변과 흰 quiet zone이 카메라에 모두 보이는지 확인한다.
- 로봇 링크나 cube가 마커를 가리는지 확인한다.
- `viewer.py`의 camera 중심이 marker 작업 레인과 맞는지 확인한다.
- 마커가 너무 작으면 거리를 줄이거나 physical size를 늘린다.

### 양팔 중앙 마커가 팔에 가려짐

현재 XML이 실제로 두 팔을 `Y=±offset`에 배치했는지 먼저 확인한다. 단일 팔이 `Y=0`에 남아 있으면 중앙 마커를 가린다. 양팔 배치가 끝난 뒤 `aruco_reference_frame`의 Y를 두 팔 베이스의 평균으로 설정한다.

### 기존 ACT 체크포인트 성능이 떨어짐

마커 추가는 영상 분포 변경이다. 마커가 없는 영상으로 학습한 체크포인트에 마커가 있는 장면을 바로 사용하지 않는다. 새 XML과 카메라 조건으로 데이터를 다시 수집하고 학습한다.

### 학습과 추론 feature 불일치

`--aruco-state` 데이터는 `observation.environment_state`를 포함한다. 해당 데이터로 만든 체크포인트만 명시적 ArUco 상태 추론에 사용한다. 시각 전용 데이터와 상태 포함 데이터를 한 dataset root에 섞지 않는다.
