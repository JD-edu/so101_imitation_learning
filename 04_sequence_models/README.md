# 4장 — RNN·Transformer 구조

`sequence_models.py`는 5장의 RNN·Transformer와 같은 구조를 담은 독립 PyTorch 예제입니다. MuJoCo 없이 입력과 출력 형태를 확인합니다. 학습과 가중치 저장은 하지 않습니다.

```bash
cd /home/robo/so101_imitation_learning/04_sequence_models
python 101_model_shapes.py --help
python 101_model_shapes.py
python 101_model_shapes.py --batch-size 3 --history 12
```

`--batch-size`는 배치 크기(기본 2), `--history`는 시퀀스 길이(기본 10)이며 양의 정수입니다. 두 모델 모두 `(배치, 길이, 6)`을 받아 `(배치, 6)`을 출력합니다. 실제 BC 학습은 `../05_mujoco_bc/`에서 진행합니다.

[위키독스 4장](https://wikidocs.net/439086)
