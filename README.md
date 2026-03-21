# Pill 2차 System: Full-Resolution Trainer + New Runtime

This bundle is for your **2차 system**:
1. Kaggle trainer for a **full-resolution, multi-scale** model
2. New runtime that preserves the original image until multi-scale extraction

## Main idea

Do not destroy detail too early.

Instead of only shrinking the whole image and hoping the network remembers everything, this system uses:

- a **global branch** for whole-pill context
- a **tile branch** for local texture / imprint / edge detail
- **full-resolution decode**
- **ROI-aware tile extraction**
- **embedding prototypes + thresholds**
- **2-image set consistency**

This is the practical engineering version of “see like a human”.

## Kaggle training

```bash
cd trainer

python train_kaggle_fullres.py   --data-root /kaggle/input/pill-dataset/약모음\ 20250902   --outdir /kaggle/working/fullres_run01   --epochs 30   --batch-size 4   --accum-steps 4   --num-workers 4   --global-size 1024   --tile-size 512   --num-tiles 4   --emb-dim 512   --lr 3e-4   --val-ratio 0.15
```

It saves:
- `best.ckpt`
- `last.ckpt`
- `labels.json`
- `train_config.json`
- `prototypes.npz`

Export the runtime bundle:

```bash
python export_runtime_bundle.py   --checkpoint /kaggle/working/fullres_run01/best.ckpt   --prototypes /kaggle/working/fullres_run01/prototypes.npz   --labels /kaggle/working/fullres_run01/labels.json   --output-dir /kaggle/working/runtime_bundle
```

## Runtime

```bash
cd runtime
set PILL_MODEL_BUNDLE=models\runtime_bundle
set PILL_RUNTIME_CONFIG=models\runtime_bundle\runtime_config.json
uvicorn app.core.api:app --host 127.0.0.1 --port 9000
```

## Exact 2-image tester

```bash
cd runtime

python tools\pair_tester.py ^
  --data-root "C:\Users\SHIN\Desktop\약모음 20250902 (2)\약모음 20250902" ^
  --url http://127.0.0.1 ^
  --port 9000 ^
  --mode both ^
  --out pair_test_results.json
```

## Main adaptation points

Because I do not have your exact current private checkpoint / model-export format, the likely places you may adapt are:

- `trainer/export_runtime_bundle.py`
- `runtime/app/core/predictor.py` → `_run_model(...)`
