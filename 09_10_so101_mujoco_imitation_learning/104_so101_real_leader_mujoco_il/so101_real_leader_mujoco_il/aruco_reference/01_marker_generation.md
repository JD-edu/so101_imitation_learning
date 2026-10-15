# 1. ArUco 마커 생성

## 기존 그림 태그를 사용하지 않는 이유

처음 제공된 흑백 그림은 ACT가 영상 패턴으로 학습하는 기준점으로는 사용할 수 있지만, 표준 ID와 검출 규칙이 없다. OpenCV가 자동으로 ID, 네 모서리와 중심을 구할 수도 없다. 촬영된 JPEG에는 흰 작업대, 원근 왜곡, 오염과 기울기도 포함되어 있었다.

이번 구현에서는 기존 그림을 학습 자산으로 사용하지 않고 표준 ArUco로 교체했다. ArUco는 다음 정보를 일관되게 얻을 수 있다.

- dictionary와 marker ID
- 영상 내 네 모서리 좌표
- 영상 중심 위치
- 영상에서 차지하는 크기
- 영상 평면상의 회전
- 가시 여부

## 확정 규격

| 항목 | 값 |
|---|---|
| Dictionary | `DICT_6X6_250` |
| Marker ID | `23` |
| PNG 해상도 | 512×512 |
| 검은 코드 영역 | 400×400 pixel |
| 흰 quiet zone | 가장자리마다 56 pixel |
| 검은 border bit | 1 |
| 시뮬레이션 외곽 크기 | 0.12 m |

생성 결과는 `../assets/aruco_6x6_250_id23.png`이고 같은 이름의 JSON에 규격을 저장한다. XML package 배포를 위해 `pyproject.toml`에도 `assets/*.png`가 포함되어 있다.

## 생성 명령

```bash
cd /home/robo/so101_imitation_learning/09_10_so101_mujoco_imitation_learning
conda activate lerobot

python -m so101_real_leader_mujoco_il.aruco_reference.generate_marker
```

모든 옵션 확인:

```bash
python -m so101_real_leader_mujoco_il.aruco_reference.generate_marker --help
```

같은 규격을 명시해서 생성:

```bash
python -m so101_real_leader_mujoco_il.aruco_reference.generate_marker \
  --dictionary DICT_6X6_250 \
  --marker-id 23 \
  --pixels 512 \
  --quiet-zone 56 \
  --border-bits 1 \
  --physical-size-m 0.12
```

ID나 dictionary를 바꾸면 다음 항목을 모두 같이 바꿔야 한다.

1. 생성 PNG와 JSON
2. XML의 texture 파일명
3. 수집의 `--aruco-id`
4. 추론의 `--aruco-id`
5. miniVLA task 문장

## PNG를 사용하는 이유

현재 설치된 MuJoCo에서 원본 JPEG를 `<texture file="...jpg">`로 읽으면 사용자 정의 binary texture로 해석되어 크기 오류가 발생했다. PNG는 정상 로드되는 것을 확인했다. 따라서 MuJoCo asset에는 PNG만 사용한다.

## 인쇄 방법

실제 작업대에 붙일 마커도 MuJoCo와 같은 dictionary·ID·방향을 사용한다.

- 페이지 맞춤이나 자동 확대를 끈다.
- 흰 quiet zone을 자르지 않는다.
- 바깥 흰 정사각형의 실제 크기를 12 cm로 맞춘다.
- 평평하고 반사가 적은 재질로 출력한다.
- 구김, 코팅 반사와 로봇 그림자를 줄인다.

현재 JSON의 `physical_outer_size_m=0.12`는 흰 여백까지 포함한 외곽 크기다. 향후 3D pose estimation을 추가할 때는 OpenCV에 전달하는 marker length가 검은 코드 영역의 실제 길이인지 API 정의를 확인하고 별도로 측정해야 한다.

## 생성 이미지 검출 확인

```bash
python -m so101_real_leader_mujoco_il.aruco_reference.inspect_image \
  104_so101_real_leader_mujoco_il/so101_real_leader_mujoco_il/assets/aruco_6x6_250_id23.png \
  --output aruco_marker_check.png
```

정상 결과는 JSON에 다음 값이 포함된다.

```json
{
  "marker_id": 23,
  "visible": true
}
```

검출되지 않으면 명령은 종료 코드 2를 반환한다.
