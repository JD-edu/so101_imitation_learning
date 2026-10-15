## 실행 안내: 포트 설정부터 칼리브레이션·텔레오퍼레이션까지

이 장은 실제 모터에 명령을 보내는 단독 스크립트이다. `--port`, `--baudrate`, `--motor-id`, `--help`를 받는 CLI는 없다. 파일 하단의 `if __name__ == "__main__":` 호출부를 수정한 뒤 실행한다. **함수 정의의 기본값보다 실제 호출부의 인자가 우선**한다.

### 1. 환경과 포트 확인

```bash
conda activate lerobot
cd /home/robo/so101_imitation_learning/07_so101_control_calibration
python -c "import serial; print(serial.__version__)"
python -m serial.tools.list_ports -v
```

마지막 명령은 pyserial이 제공하는 포트 목록 조회 도구이며 모터를 움직이지 않는다. `/dev/ttyUSB0`와 `/dev/ttyACM0`은 예시 이름이므로 리더·팔로워가 실제로 연결된 포트를 확인한다. 드라이버 기본 baudrate는 1,000,000이다. 포트 열기에서 실패하면 존재 여부·다른 프로세스의 사용 여부·현재 사용자의 장치 접근 권한을 먼저 확인한다.

### 2. 실행 전에 바꾸는 값

| 파일 | 실행 시 실제 포트·설정 | 바꿀 위치 |
|---|---|---|
| `201_motor_id_setting.py` | USB0, ID는 터미널 입력 | 마지막 `set_motor_id()`를 `set_motor_id(port="실제 포트")`로 변경 |
| `202_feetech_motor_con.py` | USB0, ID 1, 목표 raw 1000 | 하단 `MiniFeetechDriver`, `MOTOR_ID`, `set_position` |
| `203_motor_1_calibration_leader.py` | USB0, ID 1 | 하단 `calibrate_single_motor_shoulder_pan(...)` |
| `203_motor_1_calibration_follow.py` | USB0, **ID 6** | 같은 함수 호출의 `port`, `motor_id`, `out_json_path` |
| `204_motor_1_teleoperate.py` | ACM0→ACM1, `hz=50` | 하단 `teleop_motor1_shoulder_pan(...)` |
| `205_all_motor_calibration_leader.py` | USB0 | 하단 `FullArmCalibrator(port=...)` |
| `205_all_motor_calibration_follower.py` | USB1 | 하단 `FullArmCalibrator(port=...)` |
| `206_motor_all_teleoperate.py` | ACM0→ACM1, `config_dir="./"` | 하단 `Teleoperator(...)` |
| `position_test.py` | USB0, ID 1 | 하단 `monitor_motor_position(...)` |

203의 팔로워 파일은 함수 기본 ID가 1이어도 하단 호출에서 **ID 6**을 전달한다. shoulder_pan 단일 관절 실습이라면 실제 shoulder_pan ID(기본 배치에서는 1)에 맞춰 `motor_id`를 먼저 바꾼다. 기본 포트도 USB0이다. 파일명이 follower라고 자동으로 USB1을 선택하지 않는다. 전체 팔 텔레옵 206은 고정 Hz 옵션 없이 반복하며 측정 Hz를 출력한다. 단일 관절 204의 `hz`와 혼동하지 않는다.

### 3. 실행 단계

ID 변경이 필요한 모터는 **하나씩 연결**하고 다음을 실행한다. 터미널에서 현재 ID와 새 ID를 입력한다. 여러 모터를 연결한 채 방송 ID 254로 바꾸지 않는다.

```bash
python 201_motor_id_setting.py
```

위치 읽기를 먼저 확인하려면 다음을 실행한다. 해당 모터의 토크를 끄고 현재 raw 위치를 표시하며 `Ctrl+C`로 종료한다.

```bash
python position_test.py
```

단일 관절 경로를 학습하려면 포트와 저장 파일명을 맞춘 다음 순서대로 실행한다. 각 calibration 프로그램은 중앙·최소·최대 자세에서 Enter 입력을 받는다.

```bash
python 203_motor_1_calibration_leader.py
python 203_motor_1_calibration_follow.py
python 204_motor_1_teleoperate.py
```

결과는 `shoulder_pan_calibration_leader.json`, `shoulder_pan_calibration_follower.json`이다. 한 관절 텔레옵은 이 두 파일을 읽고 `Ctrl+C`로 종료한다.

전체 팔 경로는 다음과 같다.

```bash
python 205_all_motor_calibration_leader.py
python 205_all_motor_calibration_follower.py
python 206_motor_all_teleoperate.py
```

결과는 현재 폴더의 `full_arm_calibration_leader.json`, `full_arm_calibration_follower.json`이다. `config_dir`은 현재 구현이 문자열로 파일명을 붙이므로 `./`처럼 끝에 `/`가 있는 디렉터리를 지정한다. 206은 `Ctrl+C` 종료 처리에서 팔로워 토크를 끈다.

`202_feetech_motor_con.py`는 선택적인 저수준 명령 실습이며, 실행하면 raw 1000을 전송한다. 실제 관절의 가동 범위를 확인하고 목표를 정한 뒤 실행한다. 205의 JSON은 8장으로 전달하지만, 10장은 별도 중심 정렬 calibration을 사용한다.
