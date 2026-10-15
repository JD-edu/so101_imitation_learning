# 3. ACT와 miniVLA 활용

## 방식 A: 영상 기준점

XML에 ArUco가 있으므로 추가 옵션 없이 수집해도 `observation.images.overview`에 마커가 들어간다.

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_aruco_visual \
  --repo-id local/so101_real_leader_aruco_visual \
  --leader-port /dev/ttyUSB0
```

ACT의 CNN은 마커, 물체와 로봇의 상대적인 영상 배치를 학습한다. 이 방식의 `observation.state`는 기존과 같은 관절 6개이며 기존 데이터 schema를 유지한다.

고정 카메라와 고정 마커에서 마커가 항상 같은 픽셀에 있으면 새로운 정보가 거의 없다. 기준점의 이점을 얻으려면 물체 목표를 마커 기준으로 정의하거나, 카메라·작업대 배치가 조금 변해도 마커를 이용해 관계를 해석하도록 데이터를 구성한다.

## 방식 B: 영상과 명시적 검출 상태

`--aruco-state`를 사용하면 ID 23을 매 frame 검출해 다음 6차원 벡터를 저장한다.

```text
observation.environment_state = [
    visible,
    center_x_normalized,
    center_y_normalized,
    side_length_ratio,
    rotation_sin,
    rotation_cos,
]
```

| 값 | 범위·의미 |
|---|---|
| `visible` | 검출 성공 1, 실패 0 |
| `center_x_normalized` | 영상 왼쪽 -1, 중앙 0, 오른쪽 +1 |
| `center_y_normalized` | 영상 위 -1, 중앙 0, 아래 +1 |
| `side_length_ratio` | 평균 변 길이 ÷ 영상의 짧은 변 |
| `rotation_sin`, `rotation_cos` | 첫 번째 마커 변의 영상 평면 회전 |

검출 실패 시 여섯 값은 모두 0이다. 녹색 테두리와 텍스트는 OpenCV preview 복사본에만 그리며 정책에 저장되는 원본 영상에는 넣지 않는다.

수집:

```bash
python -m so101_real_leader_mujoco_il.cli collect \
  --dataset-root data/real_leader_aruco_state \
  --repo-id local/so101_real_leader_aruco_state \
  --leader-port /dev/ttyUSB0 \
  --fps 30 --seed 0 \
  --aruco-state --aruco-id 23
```

확인:

```bash
python -m so101_real_leader_mujoco_il.cli inspect \
  --dataset-root data/real_leader_aruco_state \
  --repo-id local/so101_real_leader_aruco_state \
  --episode 0 --delay-ms 33
```

`inspect`는 관절 상태, 행동과 함께 `aruco=[...]`를 출력한다.

학습:

```bash
python -m so101_real_leader_mujoco_il.cli train \
  --dataset-root data/real_leader_aruco_state \
  --repo-id local/so101_real_leader_aruco_state \
  --output outputs/act_real_leader_aruco_state \
  --steps 100000 --batch-size 32 --device cuda
```

LeRobot metadata의 `observation.environment_state`는 `FeatureType.ENV`로 변환되고 ACT encoder의 environment-state token이 된다. 수집할 때의 feature 순서와 shape를 추론에서도 그대로 사용한다.

추론:

```bash
python -m so101_real_leader_mujoco_il.cli infer \
  --checkpoint outputs/act_real_leader_aruco_state \
  --device cuda --seed 1000 --aruco-id 23
```

추론 코드는 체크포인트의 입력 feature를 읽는다.

- `observation.environment_state`가 있으면 ID 23을 검출해 6차원 벡터를 전달한다.
- 해당 feature가 없는 기존 체크포인트는 검출 벡터 없이 실행한다.
- environment-state shape가 6이 아니면 잘못된 체크포인트 결합으로 판단해 실행을 중단한다.

영상 방식 데이터와 명시적 상태 방식 데이터는 schema가 다르므로 같은 데이터셋 디렉터리에 섞지 않는다. 마커 추가 전 영상으로 학습한 체크포인트도 새 마커 장면과 영상 분포가 다르므로 새 데이터로 재학습한다.

## ACT가 얻는 정보의 한계

현재 6차원 특징은 2D 영상 정보다. 마커와 카메라 사이의 정확한 3D 변환이나 물체의 마커 기준 3D 좌표를 직접 제공하지 않는다. 정확한 pose가 필요하면 다음 항목이 추가로 필요하다.

- 실제 검은 marker edge의 물리 길이
- 카메라 내부 파라미터 `fx, fy, cx, cy`
- 실제 카메라 distortion 계수
- `solvePnP` 또는 ArUco pose estimation
- MuJoCo 카메라와 실제 카메라의 동일한 축 정의

## miniVLA 활용

현재 저장소에는 miniVLA 모델, loader 또는 추론 API가 없다. 특정 라이브러리를 임의로 가정하는 실행 코드는 추가하지 않았다. 대신 miniVLA가 일반적으로 사용하는 다음 입력은 데이터 수집 코드에 반영했다.

1. ArUco가 보이는 `observation.images.overview`
2. ArUco ID를 명시한 task 문자열
3. loader가 수치 environment-state를 지원할 때 사용할 선택적 6차원 특징

현재 task:

```text
Use ArUco marker 23 as the workspace reference.
Pick up the red block and place it at the target location.
```

언어 지시를 만들 때 방향과 거리를 구체적으로 고정한다.

```text
Use ArUco marker 23 as the workspace origin.
Move the red block 10 cm in front of the marker.
```

데이터 전체에서 다음 정의를 바꾸지 않는다.

- marker ID
- 마커의 위쪽 방향
- `앞/뒤/왼쪽/오른쪽`에 대응하는 월드 축
- 거리 단위
- 카메라 시점

이미지와 언어만 받는 miniVLA에는 6차원 벡터를 전달하지 않는다. proprioception이나 environment state를 별도로 받는 구현에서만 해당 feature를 loader에 연결한다.

## 실제 마커와 MuJoCo 마커 일치

sim-to-real을 위해 다음 조건을 맞춘다.

- 같은 `DICT_6X6_250`, ID 23
- 같은 인쇄 방향
- 같은 외곽 크기 12 cm
- 작업대 기준 위치와 높이
- 카메라가 보는 상대 크기
- 조명과 반사 조건

마커만 같다고 실제 카메라와 MuJoCo 영상이 자동으로 일치하지는 않는다. 렌즈 왜곡, 그림자, 노출과 로봇 외형 차이도 데이터 augmentation 또는 실제 데이터로 다뤄야 한다.
