# 4·5·6장 소스 폴더 분리 및 위키독스 반영

2026-09-29 기준으로 세 장의 코드를 다음과 같이 분리했다.

| 장 | 독립 폴더 | 내용 |
|---|---|---|
| [4장](https://wikidocs.net/439086) | `04_sequence_models/` | RNN·Transformer 구조와 shape 확인 CLI |
| [5장](https://wikidocs.net/439087) | `05_mujoco_bc/` | 201~203 MLP·RNN·Transformer BC 학습·추론 |
| [6장](https://wikidocs.net/439088) | `06_mujoco_act/` | 204~208 Chunking·ACT 학습·추론 |

기존 `04_06_mujoco_bc_to_act/`의 파일 60개는 새 위치에서 SHA-256이 모두 일치한다. 기존 학습·추론 코드와 가중치 내용은 변경하지 않았다. 5·6장은 각각 XML·mesh를 가지며 복사한 자산 30개도 원본과 일치한다. 이후 한 장의 자산을 수정해도 다른 장에는 자동 반영되지 않는다.

4장은 5장의 모델 정의를 그대로 추출한 독립 PyTorch 모듈과 실행 파일을 추가했다. 같은 가중치를 넣었을 때 5장 모델과 출력이 일치하는 것을 검사했다. `--batch-size`(기본 2), `--history`(기본 10)로 입력 크기를 설정하며 학습이나 가중치 저장은 하지 않는다.

장별 README, 최신 통합 Markdown, CLI 안내와 번호 대응표를 갱신했다. 온라인 4·5·6장의 본문·실행 명령을 수정하고, 나머지 1~10장에 있던 공통 경로 안내도 맞췄다. 다시 조회한 10개 페이지의 본문이 게시 원고와 정확히 일치했다. 각 장당 한 페이지 구성을 유지했고, 0장과 보관함을 포함한 나머지 112개 페이지의 본문·제목·부모·공개 상태는 보존했다.

검증 결과:

- unittest 21개 통과: 기존 PyTorch·MuJoCo·BC·ACT 검사와 4장 모델 동등성 검사.
- 새 CLI의 기본 실행, `--help`, 사용자 설정, 잘못된 입력 거부 확인.
- 게시 원고의 모든 절대 `cd` 경로가 실제 폴더와 일치.
- 전체 기본 설정 재학습과 실제 로봇 구동은 수행하지 않았다. GitHub push는 수행하지 않았다.

[파일 이동 manifest](split_chapters_04_06_manifest.json), [검증 결과](split_chapters_04_06_verification.json), [게시 전 원문 백업](wikidocs_before_split_chapters_04_06.json), [게시한 원고](split_chapters_04_06_pages.json)를 보관했다.
