# TRAINER_TOOL (Full) — Kaggle/Colab

You previously only had pack-builder scripts on Kaggle. This version includes:
- `src/train.py`
- `src/dataset.py`
- `src/model.py`
- `src/export_onnx.py`

## Kaggle usage
In a Kaggle notebook cell:

```bash
!pip install -r TRAINER_TOOL/requirements_kaggle.txt
%cd /kaggle/working/TRAINER_TOOL

!python -m src.train \
  --data "/kaggle/input/<YOUR_DATASET>/약모음 20250902" \
  --out "/kaggle/working/outputs/run1" \
  --classes 16 \
  --epochs 80 \
  --img 448 \
  --batch 16 \
  --lr 0.0002
```

Export ONNX:

```bash
!python -m src.export_onnx \
  --ckpt "/kaggle/working/outputs/run1/best.pt" \
  --out "/kaggle/working/outputs/runtime_bundle" \
  --opset 17
```

Copy `outputs/runtime_bundle/pill_cls.onnx` + `classes.json` into:
`RUNTIME_APP/models/current/` on Windows.
