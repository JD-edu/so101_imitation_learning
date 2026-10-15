# 코드 재편성 및 원고 개정 기록

> 번호 정렬 후 현재 코드 경로는 [장 번호·소스 대응표](chapter_source_numbering.md)를 기준으로 한다. 아래 이전 검토·게시 기록의 101~106 경로는 당시 경로이다. 온라인 1~10장과 최신 통합 원고는 새 경로로 갱신했다.

> 아래 폴더 이동·초기 원고 검증은 당시 기록이다. 후속 온라인 게시와 1~4장 동기화로 현재 MuJoCo는 3장, 순차 모델 기초는 4장이다. [현재 장별 링크](wikidocs_publication.md)와 [동기화 결과](chapters_01_04_sync_review.md)를 참조한다.

## 변경 범위

요청한 순서인 PyTorch 학습 → MuJoCo 기초 → MuJoCo BC·RNN·ACT → 실제 SO-101 커스텀 제어·학습 → 실제 리더와 MuJoCo 연동에 맞춰 최상위 폴더를 재배치했다. 4~10장 본문은 [하나의 Markdown 파일](chapters_04_10.md)에 작성했다.

0~3장 원고와 기존 보관 자료는 수정하지 않았다. 온라인 위키독스 페이지는 게시·수정하지 않았다. 공개 목차는 [로보틱스를 위한 이미테이션 러닝](https://wikidocs.net/book/19620)에서 확인했다.

## 폴더 변경표

| 이전 | 이후 | 학습 위치 |
|---|---|---|
| `101_pytorch_basic/` | 동일 | 1. PyTorch |
| `105_MUJOCO_basic/` | `102_mujoco_basic/` | 2. MuJoCo 기초 |
| `102_ACT_step_by_step/` | `103_mujoco_bc_to_act/` | 3. MuJoCo BC·RNN·ACT |
| `103_so101_control_caibration/` | `104_so101_control_calibration/` | 4-1. 실제 모터 제어·칼리브레이션 |
| `104_so101_imitation_learning/` | `105_so101_imitation_learning/` | 4-2. 실제 데이터 수집·학습·추론 |
| `106_so101_MUJOCO_imitation_learning/` | 동일 | 5. MuJoCo IK·실제 리더 연동 |

다섯 학습 단계 중 실제 SO-101 단계는 제어·칼리브레이션과 데이터·학습의 두 폴더로 구성한다. 기존 실습 단위와 상대 경로를 유지하면서 최상위 번호를 101~106의 연속 순서로 정리했다.

내부 예제 번호, Python 파일명, 패키지명, XML, mesh, 데이터, 가중치 파일명은 유지했다. `205_1_Action_Various_start.py`의 동적 파일 import와 `pyproject.toml`의 패키지 검색 경로 등 기존 연결을 보존하기 위해서다. 새 장 번호는 원고에서 각 파일과 매핑한다.

`archive/102_RNN_transformer/`는 기존 3장의 자료이므로 이동하지 않았다. 그 안의 ViT 자료는 새 6장에서 선택 참고 자료로 안내한다. `archive/103_ACT_step_by_step/`와 루트 `data/`도 그대로 보존했다.

## 새 4~10장의 구성

| 새 장 | 중심 내용 | 기존 구성과의 관계 |
|---|---|---|
| 4 | MuJoCo 기초, MJCF, viewer, 입력, 센서, 모델 로드 | 기존 후반 MuJoCo 기초를 앞으로 배치 |
| 5 | MLP BC → RNN → Transformer encoder | 기존 3장 본문은 그대로 두고 로봇 적용 과정을 새로 설명 |
| 6 | Chunking → 다양한 시작점 → RH·TE → CVAE → Visual ACT | 단계별 MuJoCo 코드를 중심으로 ACT 설명 재구성 |
| 7 | 모터 ID, 드라이버, calibration, teleoperation | 실제 로봇 제어 준비 통합 |
| 8 | LeRobot 데이터, MLP, 커스텀 ACT, 실제 추론 | 기존 데이터·MLP·ACT 흐름 통합 및 구현 제약 명시 |
| 9 | IK expert, 합성 시연, 공식 ACT, 시뮬레이션 평가 | 현재 IK 패키지 기반 본문 신설 |
| 10 | 실제 리더 calibration, 가상 팔로워 수집·학습·추론 | 현재 실제 리더 패키지 기반 본문 신설 |

원고는 총 7개 장의 설명, 데이터 의미·수식, 실행 위치·명령, 결과물, 구현 제약, 확인 문제를 포함한다. 기존 11장 온라인 페이지는 이번 변경 대상이 아니므로 그대로 두며, 새 10장과의 온라인 목차 정리는 실제 게시 작업 시 별도로 판단할 수 있다.

## 파일 보존 검증

이동 전에 존재하던 일반 파일 967개의 경로·크기·SHA-256을 기록했다. 이동 후 매핑된 위치의 967개를 모두 다시 읽어 검증했다.

- 내용 변경 또는 누락: **0개**.
- 경로가 변경된 기존 파일: **289개**.
- 기존 Python 파일: **112개**, 모두 바이트 단위 내용 동일.
- 기존 데이터셋, checkpoint, calibration, XML/mesh, 작업 기록 포함 보존.
- 작업 시작 전에 Git 상태에 있던 삭제 91건은 그대로 유지. 삭제된 예전 구현을 복원하지 않음.
- 작업 시작 전에 있던 수정 파일과 미추적 파이프라인도 현재 내용을 보존.

파일별 매핑과 해시는 [reorganization_manifest.json](reorganization_manifest.json)에 있다. 이 manifest는 **작업 전 존재하던 파일**을 대상으로 하며 새 README와 새 문서는 포함하지 않는다. 검증이 끝난 뒤 사용자가 데이터를 생성하거나 학습하면 해당 출력 파일의 해시는 달라질 수 있다.

파일 이동은 일반 디렉터리 rename으로 수행했고 Git index는 변경하지 않았다. 따라서 staging 전에는 Git이 이전 위치의 삭제와 새 위치의 미추적 파일로 표시할 수 있다. 기존 변경을 포함해 임의 staging·commit하지 않았다.

## 실행·정적 검증

1. 이동된 MuJoCo 기초의 대표 XML 8개와 BC·ACT `scene.xml` 1개, 총 **9개 모델 로딩 성공**. 상대 include·mesh 경로가 유지됨을 확인했다.
2. `106_so101_MUJOCO_imitation_learning/tests/test_pipeline.py`의 **기존 테스트 함수 6개 모두 통과**. 모델 주소·범위, 물리 집기 expert, 10개 seed의 IK 회귀, 실제 리더 매핑, 중심 0 rad 대응을 확인했다.
3. conda `lerobot` 환경에는 `pytest`가 없어 `python -m pytest`는 실행되지 않았다. 해당 테스트들은 fixture 없이 호출 가능한 함수이므로 같은 환경에서 `runpy.run_path`로 불러와 여섯 함수를 직접 실행했다. pytest runner 전체가 실행되었다고 주장하지 않는다. 환경에 패키지를 추가 설치하지 않았다.
4. 기존 Python 112개를 AST 파싱했다. 111개는 통과했고 `archive/102_RNN_transformer/107_transformer_inference_study.py`의 1행 `unexpected indent`는 작업 전부터 존재하는 오류이다. 해시 동일성을 확인했으며 원본 유지 요청에 따라 수정하지 않았다.
5. 원고의 장 번호 4~10, 코드 블록 구분, 명시된 `cd` 경로, 문서 링크와 단독 스크립트 진입 경로를 점검했다.

모터 구동, calibration 측정, 새 데이터 수집, 전체 모델 재학습, GUI 화면 상호작용은 수행하지 않았다. 회귀 검증은 저장소 구조와 기존 환경·expert에 대한 검증이며 학습 정책의 성능 측정이 아니다.

## 원고에 반영한 기존 구현 제약

코드는 수정하지 않고 다음 사항을 해당 장에서 설명했다.

- 204/206 예제의 checkpoint 파일명 중복.
- 207 학습 파일과 별도 추론 파일의 모델 구조·checkpoint 불일치.
- 실제 calibration·teleoperation 스크립트 사이의 포트 기본값 및 방향 처리 차이.
- 기존 실제 데이터 수집 코드의 동일 repo-id 캐시 삭제 동작.
- MLP 추론 폴더에 없는 `motor_control.py`를 `PYTHONPATH`로 연결하는 실행 방법.
- 실제 학습 이미지의 값 범위와 `/255.0` 중복 가능성.
- 커스텀 ACT에서 현재 행동을 반복하는 fallback과 올바른 미래 action timestamp의 차이.
- 실제 리더 시뮬레이션 데이터의 시간 정렬, simulation time과 wall-clock time 구분.
- 공식 ACT checkpoint의 정책·전처리·후처리 파일을 함께 보존해야 하는 이유.

## 후속 수정 기록

위 파일 보존 결과와 manifest는 초기 폴더 재편성 시점의 기록이다. 이후 사용자 요청으로 `103_mujoco_bc_to_act/`의 코드를 수정하고 원고 5~6장을 갱신했다. 이 폴더의 현재 코드 해시는 초기 manifest와 달라진다. 변경 내역과 새 검증 기준은 [BC→ACT 검토 보고서](mujoco_bc_to_act_review.md)를 참조한다. 기존 가중치·XML·mesh는 보존했다.

후속 1~4장 동기화에서는 `101_pytorch_basic/` 1개와 `102_mujoco_basic/` 6개 Python 파일을 수정했다. 기존 데이터·가중치·XML·URDF·mesh는 보존했으며 변경 전후 해시는 `chapters_01_04/source_verification.json`에 기록했다. 기존 초기 보존 결과는 수정 이후 파일 해시가 같다는 의미가 아니다.
