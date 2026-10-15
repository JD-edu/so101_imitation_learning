## 실행 안내: MuJoCo 예제 선택과 설정

이 장의 파일은 `--xml`, `--headless`, `--help`를 해석하지 않는다. 옵션 없이 실행하며 모델·카메라·명령값은 Python 또는 XML에서 바꾼다.

```bash
conda activate lerobot
cd /home/robo/so101_imitation_learning/03_mujoco_basic
python -c "import mujoco, glfw, numpy; print('MuJoCo:', mujoco.__version__)"
```

본편의 GUI에는 디스플레이와 OpenGL이 필요하다. 창 없이 모델만 확인할 때는 부록 A의 `MjModel.from_xml_path` 진단 예제를 사용한다.

### 기본 예제 실행

각 블록의 `cd`부터 실행한다. 한 예제의 창을 닫은 다음 다음 예제를 실행한다.

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/101_mujoco_glfw_window
python 101_mujoco_glfw_window.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/102_mujoco_mjcf_load
python 102_mujoco_mjcf_load.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/103_mujoco_viewer
python 103_mujoco_viewer.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/104_glfw_mouse_move
python 104_glfw_mouse_move.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/105_glfw_asdw_move
python 105_glfw_asdw_move.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/106_mujoco_imu_sensor
python 106_mujoco_imu_sensor.py
```

### 로봇 모델 표시 후 움직임 실행

각 `mjcf_load.py` 창을 종료한 뒤 같은 폴더의 `robot_move.py`를 실행한다.

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/107_jdcobot100_MUJOCO_load/answer
python mjcf_load.py
python robot_move.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/108_puppypi_mujoco_load/answer
python mjcf_load.py
python robot_move.py
```

```bash
cd /home/robo/so101_imitation_learning/03_mujoco_basic/109_so101_mujoco_load/answer
python mjcf_load.py
python robot_move.py
```

### 동작을 바꾸는 위치

| 항목 | 실제 수정 위치 |
|---|---|
| 창 크기 | `glfw.create_window(width, height, ...)` |
| 카메라 시점 | Python의 카메라 distance·azimuth·elevation |
| 모델 | Python이 읽는 XML과 `scene.xml`의 `include` |
| 물리 시간 간격 | 실제 입력 XML의 `<option timestep="...">` |
| 105/106 바퀴 명령 | `velocity=40.0`, `left_handycap=0.7` |
| 107/109 목표 자세 | `robot_move.py`의 `pose_A`, `pose_B`, `switch_interval=2.0` |
| 108 Puppy 자세 | 최초 스텝 후 `qpos[7:]`에서 복사하는 목표; 현재 보행 옵션 없음 |

101의 빈 창은 정상이다. 104는 마우스 시점 조작, 105/106은 창에 포커스를 두고 W/A/S/D를 사용한다. 사용자 정의 GLFW 예제는 창 닫기로 종료한다. 103은 벽시계 20초 이후 정지 명령을 계속 유지한다.

URDF 변환은 선택 과정이다. 107/108/109의 `answer`에서 `python urdf_to_mjcf.py`를 실행하면 각각 `jdcobot100_converted.xml`, `puppy2_converted.xml`, `so101_converted.xml`을 만든다. 기존 scene이 자동으로 이 결과를 읽도록 바뀌지는 않으며, actuator 없는 변환 결과와 완성 제어 모델을 구분한다.
