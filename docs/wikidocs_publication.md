# 위키독스 게시 결과

> 4·5·6장 코드를 독립 폴더로 분리하고 온라인 경로·명령을 갱신했다. [분리 및 검증 기록](split_chapters_04_06_review.md)을 참조한다.

> 최신 본문 보강: 1~10장 첫머리에 실행 위치·설정값·CLI 옵션·명령·결과물·종료 방법을 추가했다. [실행 안내 모음](chapter_cli_guide.md)과 [검증 기록](cli_guides/validation.json)을 참조한다. 0장과 기존 보관 페이지는 유지했다.

> 번호 정렬 후 현재 코드 경로는 [장 번호·소스 대응표](chapter_source_numbering.md)를 기준으로 한다. 아래 이전 검토·게시 기록의 101~106 경로는 당시 경로이다. 온라인 1~10장과 최신 통합 원고는 새 경로로 갱신했다.

> 최신 목차 변경: 0~3장을 **각 장당 온라인 한 페이지**로 통합했다. 기존 하위 25페이지는 URL과 본문을 보존하여 [페이지 통합 이전 원문](https://wikidocs.net/439121)으로 이동했다. 4~10장은 유지했다. [통합 게시 검증](chapters_00_03_single_page_verification.json)과 [0~3장 단일 Markdown](chapters_00_03.md)을 참조한다. 아래 29/26페이지 관련 기록은 통합 이전 게시 이력이다.

[책 목차](https://wikidocs.net/book/19620)의 1~3장을 소스코드와 동기화하고, MuJoCo 중복을 정리하여 4장을 RNN·Transformer 기초로 교체했다. 이번 26페이지의 게시 결과는 온라인 재조회로 확인했다. 0장과 5~10장의 본문은 유지했다.

| 장 | 게시 페이지 |
|---|---|
| 1 | [01장. PyTorch 개론](https://wikidocs.net/343982) |
| 2 | [02장. PyTorch 비전 모델의 학습과 웹캠 추론](https://wikidocs.net/343391) |
| 3 | [03장. MuJoCo 기초와 로봇 모델 실행](https://wikidocs.net/371945) |
| 4 | [04장. RNN과 Transformer: 로봇의 상태 이력 읽기](https://wikidocs.net/439086) |
| 5 | [05장. MuJoCo에서 Behavior Cloning, RNN, Transformer 배우기](https://wikidocs.net/439087) |
| 6 | [06장. Action Chunking에서 시각 기반 ACT까지](https://wikidocs.net/439088) |
| 7 | [07장. 실제 SO-101 모터 제어와 칼리브레이션](https://wikidocs.net/439089) |
| 8 | [08장. 실제 SO-101 데이터 수집, MLP·ACT 학습과 추론](https://wikidocs.net/439090) |
| 9 | [09장. MuJoCo IK 시연으로 공식 ACT 학습하기](https://wikidocs.net/439091) |
| 10 | [10장. 실제 리더와 MuJoCo 팔로워의 이미테이션 러닝](https://wikidocs.net/439092) |

이번에 교체한 26페이지의 원문을 [1~4장 동기화 이전 원문](https://wikidocs.net/439094)에 복사했다. 앞선 게시에서 이동한 53페이지도 [이전 원고 보관함](https://wikidocs.net/439093)에 그대로 있다. 비공개였던 원고는 비공개 상태를 유지했다. 현재 전체 페이지는 117개이다.

- [현재 동기화 결과·검증 범위](chapters_01_04_sync_review.md)
- [현재 온라인 일치·보존 검증](chapters_01_04/online_verification.json)
- [현재 1~4장 통합 원고](chapters_01_04.md)
- [현재 4~10장 단일 원고](chapters_04_10.md)
- [이번 수정 전 전체 책 백업](wikidocs_before_chapters_01_04_sync.json)

앞선 4~10장 최초 게시 당시에는 0~3장을 유지하고 기존 원고 53페이지를 옮겨 전체 90페이지였다. 그 시점의 [백업](wikidocs_backup_before_publish.json), [검증 기록](wikidocs_publish_verification.json), [게시 계획](wikidocs_publish_plan.json)은 이력으로 보존한다. 그 뒤의 변경은 위 현재 결과를 기준으로 한다.

위키독스 온라인 반영은 완료했으며, 소스코드의 GitHub push는 수행하지 않았다.
