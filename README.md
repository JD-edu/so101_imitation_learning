# 로보틱스를 위한 이미테이션 러닝 — 코드와 개정 원고

학습 순서는 **PyTorch → MuJoCo 기초 → MuJoCo에서 BC·RNN·ACT → 실제 SO-101 커스텀 이미테이션 러닝 → 실제 리더와 MuJoCo 연동**입니다.

| 학습 단계 | 코드 폴더 | 책의 대응 장 |
|---|---|---|
| 1. PyTorch 학습 | `01_02_pytorch_basic/` | 1~2장, 소스 동기화 |
| 2. MuJoCo 기초 | `03_mujoco_basic/` | 3장 |
| 3-1. RNN·Transformer 구조 | `04_sequence_models/` | 4장 |
| 3-2. MuJoCo BC 학습·추론 | `05_mujoco_bc/` | 5장 |
| 3-3. MuJoCo Chunking·ACT | `06_mujoco_act/` | 6장 |
| 4-1. 실제 SO-101 모터 제어·칼리브레이션 | `07_so101_control_calibration/` | 새 7장 |
| 4-2. 실제 SO-101 데이터 수집·학습·추론 | `08_so101_imitation_learning/` | 새 8장 |
| 5. MuJoCo IK 시연과 실제 리더 연동 | `09_10_so101_mujoco_imitation_learning/` | 새 9~10장 |

- [각 장의 실행 명령·CLI 옵션 안내](docs/chapter_cli_guide.md)
- [장 번호와 코드 경로 대응표](docs/chapter_source_numbering.md)
- [위키독스 게시 결과와 장별 링크](docs/wikidocs_publication.md)
- [1~4장 통합 원고](docs/chapters_01_04.md)
- [1~4장 소스 동기화·게시 검증](docs/chapters_01_04_sync_review.md)
- [4~10장 통합 개정 원고](docs/chapters_04_10.md)
- [MuJoCo BC→ACT: 코드 수정·책 대조 결과](docs/mujoco_bc_to_act_review.md)
- [MuJoCo 기본: 위키독스·코드 대조 결과](docs/mujoco_basic_review.md)
- [구조 변경 내역 및 검증 결과](docs/reorganization.md)
- [기존 파일별 이동 경로와 SHA-256](docs/reorganization_manifest.json)
- [MuJoCo IK·실제 리더 파이프라인 실행 안내](09_10_so101_mujoco_imitation_learning/README.md)

후속 요청으로 1~3장을 현재 소스에 동기화하고, 3·4장의 MuJoCo 중복을 정리했습니다. 0장은 보존했습니다. 이전 순차 모델 예제는 `archive/102_RNN_transformer/`에 그대로 있습니다. `archive/103_ACT_step_by_step/`는 이전 ACT 설명 예제를 보관한 자료입니다. 현재 본편 실습은 위 표의 순서로 진행합니다.

최상위 폴더의 앞번호는 위키독스 장 번호이다. `01_02`는 1~2장, `09_10`은 9~10장의 공유 코드를 뜻한다. 내부 파일의 101~109와 201~208은 예제 식별자이며 각 장의 실행 파일 표에 대응한다. 파일 간 동적 import와 XML·mesh 상대 위치를 보존하여 프로그램 동작을 유지했다.

단독 Python 예제는 원고에 안내된 **해당 실습 폴더로 이동한 다음** 실행합니다. `./scene.xml`과 체크포인트를 현재 작업 폴더에서 찾는 코드가 있기 때문입니다. 9~10장 패키지는 기존 `09_10_so101_mujoco_imitation_learning/pyproject.toml`을 사용합니다.

4~6장 코드의 후속 검토에서는 관측–행동 정렬, 207 학습/추론 연결, 목표 조건, 영상 전처리 등을 수정하고 통합 원고의 5~6장을 갱신했습니다. BC·ACT 예제는 스크립트 위치에서 XML과 가중치를 찾습니다. 수정본 실행에는 각 단계의 재학습이 필요하며 기존 가중치의 이름만 바꿔 사용하지 않습니다. 새 207 추론 명령은 `python 207_ACT_CVAE_infer.py`입니다.

소규모 학습·추론 및 실제 MuJoCo/EGL 영상 수집 검사를 수행했습니다. 기본 설정 전체 재학습과 실제 모터 구동은 수행하지 않았습니다. 후속 요청에 따라 위키독스에는 4~10장을 장별로 게시하고 기존 원고 53페이지를 보관함으로 옮겼습니다.

1~4장 동기화에서는 ResNet의 검증 전처리·BatchNorm·summary, MuJoCo IMU 주소·두 로봇팔의 위치 명령·URDF 변환 출력명을 수정했습니다. 관련 자동 검사 9개와 4장의 실제 RNN·Transformer shape 예제가 통과했습니다. 온라인 26페이지를 갱신하고 원문 26페이지는 별도 보관했습니다. GitHub push는 수행하지 않았습니다.

4·5·6장은 각각 독립 폴더로 분리했습니다. 4장은 PyTorch 모델 구조 확인 예제를 제공하고, 5·6장은 각각 XML·mesh·가중치를 둡니다. [분리 및 검증 기록](docs/split_chapters_04_06_review.md)을 참조하세요.
