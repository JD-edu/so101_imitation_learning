# 2. MuJoCo XML과 카메라 설정

## 수정한 XML

기본 장면은 `../assets/lift_cube_calibration.xml`이다. `<asset>`에는 PNG texture와 반사가 없는 material을 추가했다.

```xml
<texture name="aruco_reference_texture"
         type="2d"
         file="aruco_6x6_250_id23.png"/>

<material name="aruco_reference_material"
          texture="aruco_reference_texture"
          texrepeat="1 1"
          texuniform="false"
          rgba="1 1 1 1"
          specular="0"
          shininess="0"
          reflectance="0"/>
```

`<worldbody>`에는 12 cm 평면을 추가했다. MuJoCo `size`의 X·Y는 전체 길이의 절반이므로 `0.06 0.06`은 12×12 cm다.

```xml
<body name="aruco_reference_frame" pos="0.25 -0.15 0.002">
    <geom name="aruco_reference_marker"
          type="plane"
          size="0.06 0.06 0.001"
          material="aruco_reference_material"
          contype="0"
          conaffinity="0"/>
    <site name="aruco_reference_origin"
          type="sphere"
          pos="0 0 0.003"
          size="0.003"
          rgba="1 0 0 0"/>
</body>
```

Z를 2 mm 올린 것은 작업대 평면과 같은 깊이에서 깜빡이는 z-fighting을 피하기 위해서다. `contype=0`, `conaffinity=0`이므로 cube나 로봇과 충돌하지 않는다.

## 시뮬레이션 ground truth

다음 sensor도 추가했다.

```xml
<framepos name="aruco_reference_pos"
          objtype="body"
          objname="aruco_reference_frame"/>
<framequat name="aruco_reference_quat"
           objtype="body"
           objname="aruco_reference_frame"/>
```

이 값은 XML 배치 확인과 시뮬레이션 디버깅용이다. 실제 카메라에서는 같은 sensor를 얻을 수 없으므로 ACT·miniVLA 학습 입력에는 넣지 않는다. 실제로 얻을 수 없는 ground truth를 사용하면 sim-to-real 입력 조건이 달라진다.

## 단일 팔과 양팔 위치의 차이

현재 104 XML은 아직 단일 MuJoCo 팔이고 베이스가 `Y=0`에 있다. 마커를 계획한 양팔 중앙 `Y=0`에 놓으면 현재 팔이 마커를 가린다. 그래서 현재 XML에서 검출 가능한 기본 위치를 다음처럼 설정했다.

```text
[X, Y, Z] = [0.25, -0.15, 0.002] m
```

양팔 모델에서는 실제 두 베이스 사이의 중앙값으로 옮긴다.

```text
marker_x = 원하는 기준점 X
marker_y = (left_arm_y + right_arm_y) / 2
marker_z = table_z + 0.002
```

예를 들어 왼팔과 오른팔이 `Y=-0.20`, `Y=+0.20`이면 다음과 같다.

```xml
<body name="aruco_reference_frame" pos="0.25 0.0 0.002">
```

마커는 두 팔 사이에 있다는 이유만으로 좋은 위치가 되는 것은 아니다. 양팔의 링크·그리퍼·작업물이 작업 중 마커를 오래 가리지 않는 위치를 선택한다.

## 실제로 학습되는 카메라

XML의 `origin_cam`이나 손목의 `wrist_camera_sensor`는 현재 `observation.images.overview`를 만들지 않는다. 데이터 수집과 추론은 `../viewer.py`의 `MjvCamera`를 직접 사용한다.

현재 값:

```python
OVERVIEW_LOOKAT = (0.45, -0.15, 0.03)
OVERVIEW_DISTANCE = 0.83
OVERVIEW_AZIMUTH = 0.0
OVERVIEW_ELEVATION = -48.4
```

이 값은 카메라를 로봇 뒤쪽 `-X` 상단에 놓고 작업 방향 `+X`를 바라보게 한다. 대략적인 위치는 `[-0.10, -0.15, 0.65]`이다.

사람이 리더 암을 조작할 때는 `--assist-view`로 정면 사선 보조 창을 선택적으로 연다.

```python
ASSIST_LOOKAT = (0.32, 0.0, 0.12)
ASSIST_DISTANCE = 0.65
ASSIST_AZIMUTH = 150.0
ASSIST_ELEVATION = -30.0
```

완전 정면 `180도`에서 30도 비껴 본 카메라다. 보조 창은 사람이 큐브와 그리퍼의 전후 관계를 확인하는 용도이고, 정책 영상·ArUco 특징·데이터셋에는 포함되지 않는다. 옵션을 생략하면 생성하지 않는다.

양팔이 Y축에 대칭이면 화면 중심도 다음처럼 바꾼다.

```python
OVERVIEW_LOOKAT = (0.45, 0.0, 0.03)
```

| 값 | 역할 |
|---|---|
| `LOOKAT` | 영상 중앙의 작업 좌표 |
| `DISTANCE` | 카메라와 중앙점 사이 거리 |
| `AZIMUTH=0` | 뒤쪽 `-X`에서 앞쪽 `+X`를 봄 |
| `ELEVATION=-48.4` | 약 48.4도 하향 시점 |

두 팔이 잘리면 중심 Y를 먼저 맞추고, `distance`를 늘린 다음 FOV를 조절한다. 기존 `azimuth=180`은 작업대 앞쪽에서 로봇 뒤를 보는 반대 방향이다.

## 렌더링 크기

GLFW 화면은 1024×768이고 정책 영상은 224×224로 변환된다. XML offscreen buffer 높이도 768로 맞췄다.

```xml
<global azimuth="90" elevation="-20"
        offwidth="1280" offheight="768"/>
```

XML의 전역 `azimuth`, `elevation`은 `viewer.py`가 전달하는 자유 카메라 방향을 대체하지 않는다. offscreen buffer와 전역 시각 속성에만 사용된다.

1024×768을 224×224로 직접 줄이면 종횡비가 변한다. ACT가 같은 변환으로 학습·추론하면 입력 일관성은 유지되지만, 정밀한 마커 pose estimation에는 불리하다. 정확한 기하 추정을 추가할 때는 정사각형 렌더링 또는 중앙 정사각형 crop으로 바꾸고 데이터를 다시 수집한다.

## MuJoCo 장면 검증

```bash
cd /home/robo/so101_imitation_learning/09_10_so101_mujoco_imitation_learning
conda activate lerobot

MUJOCO_GL=egl python \
  -m so101_real_leader_mujoco_il.aruco_reference.verify_scene \
  --output aruco_scene_check.png
```

이 명령은 실제 파이프라인처럼 1024×768로 렌더한 뒤 224×224로 변환해서 검출한다. JSON의 `visible=true`와 종료 코드 0을 확인한다.
