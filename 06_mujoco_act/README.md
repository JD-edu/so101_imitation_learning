# 6장 — MuJoCo Chunking·ACT

각 행의 학습을 완료한 다음 같은 행의 추론을 실행합니다.

| 학습 | 추론 |
|---|---|
| `204_Action_Chunking_MUJOCO.py` | `204_Action_Chunking_MUJOCO_infer.py` |
| `205_Action_Various_start.py` | `205_Action_Various_start_infer.py` |
| `206_ACT_RH_TE.py` | `206_ACT_RH_TE_infer.py` |
| `207_ACT_CAVE.py` | `207_ACT_CVAE_infer.py` |
| `208_ACT_Image.py` | `208_ACT_Image_infer.py` |

```bash
cd /home/robo/so101_imitation_learning/06_mujoco_act
python 204_Action_Chunking_MUJOCO.py
python 204_Action_Chunking_MUJOCO_infer.py
```

이 스크립트에는 CLI 옵션 파서가 없습니다. 학습량 등의 설정은 해당 파일의 상수를 수정합니다. 상세 설정과 모든 실행 명령은 [위키독스 6장](https://wikidocs.net/439088)을 참조하세요.

MuJoCo·NumPy·PyTorch가 필요하며 뷰어 실행에는 그래픽 환경이 필요합니다. XML·mesh와 가중치는 이 폴더를 기준으로 찾습니다. 5·6장 XML과 mesh는 같은 원본의 독립 복사본이므로 이후 한쪽을 수정해도 다른 장에 적용되지 않습니다.

기존 가중치는 보존했습니다. 현재 코드의 `_v2` 체크포인트는 해당 학습 코드로 생성해야 하며, 이전 가중치의 이름만 바꿔 사용하지 않습니다.

`205_1_Action_Various_start.py`는 205 실행 래퍼이며 같은 체크포인트를 사용합니다. 207의 정식 추론 파일은 `207_ACT_CVAE_infer.py`입니다.
