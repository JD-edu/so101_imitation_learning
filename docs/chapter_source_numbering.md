# 위키독스 장 번호와 소스코드 번호

책의 1~10장과 장마다 온라인 페이지 하나인 구성을 유지하고, 코드 최상위 폴더의 번호를 대응 장 번호로 정렬했다. 여러 장이 공유하는 코드는 범위를 사용한다. 범위는 `01_02` = 1~2장, `09_10` = 9~10장이다.

| 위키독스 장 | 현재 코드 폴더 | 실행 파일 범위 |
|---|---|---|
| [1장](https://wikidocs.net/343982)·[2장](https://wikidocs.net/343391) | `01_02_pytorch_basic/` | PyTorch 101~105 학습·추론 파일 |
| [3장](https://wikidocs.net/371945) | `03_mujoco_basic/` | MuJoCo 101~109 예제 |
| [4장](https://wikidocs.net/439086) | `04_sequence_models/` | `101_model_shapes.py`, `sequence_models.py` |
| [5장](https://wikidocs.net/439087) | `05_mujoco_bc/` | 201~203 BC 학습·추론 |
| [6장](https://wikidocs.net/439088) | `06_mujoco_act/` | 204~208 Chunking·ACT |
| [7장](https://wikidocs.net/439089) | `07_so101_control_calibration/` | 모터 제어·칼리브레이션 |
| [8장](https://wikidocs.net/439090) | `08_so101_imitation_learning/` | 데이터 수집·실제 정책 학습·추론 |
| [9장](https://wikidocs.net/439091) | `09_10_so101_mujoco_imitation_learning/` | `103_so101_mujoco_ik/` 패키지 |
| [10장](https://wikidocs.net/439092) | `09_10_so101_mujoco_imitation_learning/` | `104_so101_real_leader_mujoco_il/` 패키지 |

0장은 서문이며 별도 실행 코드 폴더가 없다. 내부 Python 파일의 101~109, 201~208 등은 예제 ID이다. 예를 들어 책 5장의 MLP 예제는 `05_mujoco_bc/201_MLP_MUJOCO.py`이다. Python의 동적 파일 import, XML include, mesh와 checkpoint의 상대 위치를 보존하기 위해 내부 예제 이름은 유지했다.

## 이전 경로에서 이동한 위치

| 이전 폴더 | 현재 폴더 |
|---|---|
| `101_pytorch_basic` | `01_02_pytorch_basic` |
| `102_mujoco_basic` | `03_mujoco_basic` |
| `103_mujoco_bc_to_act` | `05_mujoco_bc`(201~203), `06_mujoco_act`(204~208) |
| `104_so101_control_calibration` | `07_so101_control_calibration` |
| `105_so101_imitation_learning` | `08_so101_imitation_learning` |
| `106_so101_MUJOCO_imitation_learning` | `09_10_so101_mujoco_imitation_learning` |

온라인 1~10장의 코드 경로와 장 첫머리의 소스 안내를 갱신했다. 0장과 보관함은 유지했다. 로컬 최신 원고와 README, 테스트의 경로도 맞췄다. 과거 백업·manifest·검토 기록의 옛 경로는 작업 이력으로 보존한다.

4장은 모델 구조만 확인하는 `04_sequence_models`로 독립했다. 5·6장에는 각각 XML과 mesh를 두었다. [최신 분리 검증](split_chapters_04_06_review.md)을 참조한다.

## 이전 번호 정렬 당시 환경과 검증

현재 `lerobot` 환경의 SO-101 editable 패키지를 새 폴더로 다시 연결했다. 다른 환경에서 옛 경로로 설치했다면 해당 환경을 활성화하고 다음을 실행한다.

```bash
cd /home/robo/so101_imitation_learning/09_10_so101_mujoco_imitation_learning
python -m pip install --no-deps --no-build-isolation -e .
```

- 이동 전 파일 888개를 SHA-256으로 대조: 누락 0, Python·데이터·가중치·XML·URDF·mesh 내용 변경 0. 이 파일 집합에서 바뀐 내용은 패키지 README의 경로뿐이다.
- 자동 unittest 20개 통과: PyTorch, MuJoCo 기본, BC·ACT의 기존 검사 포함.
- IK·리더 연동 기존 테스트 함수 6개 직접 실행 통과. pytest runner를 사용한 결과는 아니다.
- 두 SO-101 패키지가 새 경로에서 import됨을 확인했다.
- 온라인 1~10장 재조회: 게시 원고 일치, 각 장의 하위 페이지 없음, 0장·보관 자료 보존.
- 원고의 모든 절대 `cd` 경로가 실제 폴더와 일치함을 확인했다.

전체 학습과 실제 로봇 구동은 수행하지 않았다. GitHub push는 수행하지 않았으므로 이번 번호 정렬은 로컬 소스와 위키독스에 반영된 상태이다.

[파일 보존 검증](numbering_alignment_source_verification.json), [온라인 검증](numbering_alignment_online_verification.json), [변경 전 파일 목록](numbering_alignment_before_manifest.json), [변경 전 책 백업](wikidocs_before_numbering_alignment.json)을 함께 보관했다.
