# 1~4장 소스 동기화와 온라인 게시 결과

> 번호 정렬 후 현재 코드 경로는 [장 번호·소스 대응표](chapter_source_numbering.md)를 기준으로 한다. 아래 이전 검토·게시 기록의 101~106 경로는 당시 경로이다. 온라인 1~10장과 최신 통합 원고는 새 경로로 갱신했다.

1~3장의 현재 소스를 기준으로 본문을 수정하고 위키독스에 게시했다. MuJoCo 설명은 3장으로 통합하고 중복된 4장은 현재 202/203 코드에 대응하는 RNN·Transformer 기초로 재구성했다. 수정한 원고 26페이지가 온라인 본문·제목·공개 상태·계층과 일치함을 게시 후 다시 조회하여 확인했다.

| 장 | 현재 내용 | 온라인 |
|---|---|---|
| 1 | Tensor·자동미분·회귀의 개념과 실제 PyTorch 예제 연결 | [343982](https://wikidocs.net/343982) |
| 2 | 실제 10개 파일의 비전 학습·웹캠 추론 | [343391](https://wikidocs.net/343391) |
| 3 | `102_mujoco_basic`의 101~109, 센서·위치 제어·변환 | [371945](https://wikidocs.net/371945) |
| 4 | `103_mujoco_bc_to_act`의 RNN·Transformer 구조와 시간축 | [439086](https://wikidocs.net/439086) |

## 책에서 바로잡은 내용

- 1장에 남아 있던 없는 ViT 파일 참조를 실제 ResNet 파일과 설명으로 교체했다. 독립 회귀 예제, 상태 예측, 실제 정책의 목표 위치 명령을 구분했다.
- 2장의 MiniVGG/VGG/ResNet 파일명을 실제 경로와 맞췄다. 학습·검증 분할 유무, 전처리, 학습 범위, batch/epoch, 저장 파일과 GUI 그래프 종료 이후 저장 순서를 설명했다.
- 3장은 실제 읽는 XML, nq/nv/nu, 시간축, 좌우 속도 비율, 센서 주소, 관절 이름, actuator 입력을 대조했다. Puppy는 초기 자세 유지이며 보행이 아니다. CAD 조립 그림은 선택 자료로 보존하고, 저장소에 없는 export 설정을 실행 파일로 소개하지 않았다.
- 4장은 실제 RNN의 ReLU 투영·tanh·hidden 처리, Transformer의 학습 위치 표현·encoder·출력 shape를 설명한다. 현재 관측→현재 명령의 시간 정렬, 이력 길이와 청크 길이를 구분했다. 두 모델 모두 그리퍼를 포함한 여섯 관절의 값은 rad이다.

## 먼저 수정한 소스

| 파일 | 수정 |
|---|---|
| `101_pytorch_basic/105_pytorch_Resnet_transfer_fine.py` | 검증용 증강 분리, 분할 generator seed 42, Feature-Ex의 BatchNorm 통계 고정, summary 장치 일치와 통계 보존 |
| `102_mujoco_basic/106_mujoco_imu_sensor/106_mujoco_imu_sensor.py` | sensor ID 대신 `sensor_adr`·`sensor_dim`으로 실제 출력 구간 읽기 |
| `102_mujoco_basic/107_jdcobot100_MUJOCO_load/answer/robot_move.py` | position actuator에 목표 위치 직접 전달 |
| `102_mujoco_basic/109_so101_mujoco_load/answer/robot_move.py` | position actuator에 목표 위치 직접 전달 |
| 위 107·108·109의 `answer/urdf_to_mjcf.py` 세 파일 | `_converted.xml`로 저장하여 제어용 XML과 분리, 성공 메시지 일치 |

101/102의 기존 파일 522개를 변경 전후 SHA-256으로 대조했다. Python 7개만 변경됐고 누락은 없다. 기존 가중치·데이터·XML·URDF·mesh는 보존했다. 초기 폴더 재편성 시점의 코드 불변 기록과 이번 오류 수정 기록은 구별한다.

## 검증

`tests/test_intro_sync.py`의 unittest **9개 통과**:

1. Python 25개 AST 문법 검사.
2. MLP/CNN/MiniVGG의 작은 배치 역전파와 학습·추론 모델 가중치 호환.
3. VGG16/ResNet18의 실제 추론 로더에 같은 구조의 메모리 가중치를 전달하여 출력 비교.
4. ResNet 학습·검증 분할의 비중·중복 없음·파일 대응·검증 증강 제외.
5. ResNet Feature-Ex의 BatchNorm 통계 유지와 Fine-Tuning 갱신.
6. 실제 summary 호출 전후 BatchNorm 통계 보존.
7. 실제 센서 읽기 코드를 구분 가능한 가상 센서 값으로 실행하여 주소 검증.
8. 107/109의 실제 명령문으로 각 2,200 물리 스텝 진행, 목표 명령 일치와 qpos/qvel 유한값 검사.
9. 세 변환기의 출력명 검사 및 원본 URDF를 임시 경로로 변환, 결과 파일과 actuator 수 0 확인.

4장의 구조 확인 명령도 실행했고 두 실제 클래스의 출력은 모두 `(2, 6)`이었다. 앞선 MuJoCo 기본 검토에서 로드한 8개 모델의 XML은 이번에도 변경하지 않았다.

전체 epoch 재학습, 웹캠, GUI 입력·overlay, 실제 모터 구동 및 작업 성공률 측정은 수행하지 않았다. 짧은 물리 유한값 검사는 장시간 안정성·추종 성능 검증이 아니다. 저장된 기존 checkpoint의 학습 품질도 이번에 새로 평가하지 않았다. 소스코드는 로컬에 수정했으며 GitHub push는 수행하지 않았다.

## 온라인 원문 보존과 확인

- 갱신 26페이지: 1장 6, 2장 6, 3장 13, 4장 1.
- 교체 원문 26페이지를 [동기화 이전 원문](https://wikidocs.net/439094)에 제목·본문·공개 상태 그대로 복사했다. 현재 장의 기존 URL은 유지했다.
- 0장, 5~10장, 기존 보관 자료를 포함한 보호 대상 63페이지의 본문·제목·공개 상태·부모가 그대로임을 확인했다.
- 전체 페이지 수는 90→117이다. 원문 사본 26개와 보관 그룹 1개가 추가됐다.
- 최상위 목차는 0~10장과 [이전 원고 보관함](https://wikidocs.net/439093) 순서이다.
- 로컬 4~10장 통합본에서 5~10장 본문은 변경하지 않았다.

## 결과 파일

- [1~4장 통합 원고](chapters_01_04.md)
- [갱신된 4~10장 단일 Markdown](chapters_04_10.md)
- [소스 해시·테스트 로그](chapters_01_04/source_verification.json)
- [온라인 재조회 검증·원문 사본 ID](chapters_01_04/online_verification.json)
- [이번 변경 전 책 전체 백업](wikidocs_before_chapters_01_04_sync.json)
- [장별 원고·진행 기록 폴더](chapters_01_04/)
