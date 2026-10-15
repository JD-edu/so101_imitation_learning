# 5장 — MuJoCo BC

각 행의 학습을 완료한 다음 같은 행의 추론을 실행합니다.

| 학습 | 추론 |
|---|---|
| `201_MLP_MUJOCO.py` | `201_MLP_MUJOCO_inference.py` |
| `202_RNN_MUJOCO.py` | `202_RNN_MUJOCO_inference.py` |
| `203_Transformer_MUJOCO.py` | `203_Transformer_MUJOCO_infer.py` |

```bash
cd /home/robo/so101_imitation_learning/05_mujoco_bc
python 201_MLP_MUJOCO.py
python 201_MLP_MUJOCO_inference.py
```

이 스크립트에는 CLI 옵션 파서가 없습니다. 학습량 등의 설정은 해당 파일의 상수를 수정합니다. 상세 설정과 모든 실행 명령은 [위키독스 5장](https://wikidocs.net/439087)을 참조하세요.

MuJoCo·NumPy·PyTorch가 필요하며 뷰어 실행에는 그래픽 환경이 필요합니다. XML·mesh와 가중치는 이 폴더를 기준으로 찾습니다. 5·6장 XML과 mesh는 같은 원본의 독립 복사본이므로 이후 한쪽을 수정해도 다른 장에 적용되지 않습니다.

기존 가중치는 보존했습니다. 현재 코드의 `_v2` 체크포인트는 해당 학습 코드로 생성해야 하며, 이전 가중치의 이름만 바꿔 사용하지 않습니다.
