# Kaggle Trainer

This folder trains the three-tower inference bundle used by the production runtime.

## Main entry points

- `train_tower.py` - DDP trainer for `tower_a`, `tower_b`, `tower_c`
- `train_prototype.py` - prototype model + prototype library
- `build_profiles_and_ood.py` - class profiles + OOD index
- `build_calibrator.py` - linear meta-score + isotonic calibrator
- `export_runtime_pack.py` - exports runtime-compatible pack
- `notebook_launcher.py` - single launcher used by the Kaggle notebook

## Dual T4 launch pattern

```bash
torchrun --standalone --nproc_per_node=2 train_tower.py --tower tower_a --data-root /kaggle/input/your-pill-dataset --outdir /kaggle/working/pill_run/tower_a
```

## Expected dataset style

The scanner assumes a class-first directory layout, such as:

```text
dataset/
  CLASS_A/
    pair_001/
      U_001.png
      D_001.png
    pair_002/
      U_002.png
      D_002.png
  CLASS_B/
    pair_001/
      U_001.png
      D_001.png
```

If pair folders do not exist, the loader falls back to inferring pair IDs from filename prefixes before `_`.
