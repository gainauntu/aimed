# Architecture coverage report — revised and bug-fixed

## Bug fixes applied

| ID | Severity | Symptom | Fix location |
|---|---|---|---|
| Bug 1 | 🔴 Crash | `--tower-checkpoint` wrong arg name in launcher | `notebook_launcher.py` |
| Bug 2 | 🔴 Silent failure | Constraint profiles from raw frames, not ROI crops | `build_profiles_and_ood.py` |
| Bug 3 | 🔴 Silent failure | OOD index from raw frame features | `build_profiles_and_ood.py` |
| Bug 4 | 🔴 Gate 7 always rejects | Empty signals.jsonl fed to calibrator | `generate_calibration_signals.py` (new) |
| Issue 5 | 🟡 DINO partially ignored | 3-ch DINO weights can't load into 9-ch tower stems | `models.py`, `train_dino_ssl.py` |
| Issue 6 | 🟡 Train/inference mismatch | Training on raw frames, inference on ROI crops | `pair_dataset.py` (LightROIPreprocessor) |

## Components included

### Preprocessing (app/preprocess/)
- Stage 0 raw frame triage: blur (Laplacian ≥120), artifact, pill signal, resolution
- Stage 1 full pipeline: background subtract, anomaly pre-check, localize anywhere in frame,
  physical stats extract, shape classify (ROUND/OVAL/OBLONG/CAPSULE),
  damage detect (contour irregularity, crack scan, coating outlier),
  PCA canonicalize (ROUND skip), Lanczos resize to 288×288
- Stage 1.9 cross-image pre-check: shape match, color distance, size ratio, SSIM

### Inference (app/inference/)
- Symmetric fusion: [A+B, |A-B|, A⊙B] — mathematically swap-invariant
- TTA engine: 8 rotations × 4 photometric = 32 views
- Tower adapters: TorchScriptTower, DummyTower

### Verification (app/verifiers/)
- Stage 3 OOD: FAISS k-NN on 3-ch ROI features, p99 threshold, per-image A and B
- Stage 4 prototypical verifier: cosine sim, p99 distance, second-class margin
- Stage 5 constraint gate: 14 arithmetic checks (7 per image × 2)

### Decision (app/decision/)
- 7 hard gates: agreement, TTA stability, OOD, prototype, cross-image, constraint, calibrated P
- 12-signal isotonic meta-calibrator
- Full audit logging every decision

### Training (trainer/)
- `train_dino_ssl.py`: Phase T1 — DINOAdaptBackbone (9-ch) on fused pairs
- `train_tower.py`: Phase T2+T3 — frozen head → unfreeze top10 → unfreeze top10-30, LOOCV
- `train_prototype.py`: Phase T4 — DenseNet-169, 50k episodes, swap-aware, hard episode mining
- `build_profiles_and_ood.py`: constraint profiles + OOD index from preprocessed ROI crops
- `generate_calibration_signals.py`: LOOCV inference pass → signals.jsonl (NEW — Bug 4 fix)
- `build_calibrator.py`: multivariate isotonic calibration + variance thresholds
- `export_runtime_pack.py`: TorchScript export of all models + bundle manifest

### Models (trainer/models.py)
- TowerA: ConvNeXt-Base global + 5×5 tile branch + TileAttentionPool + CrossGate
- TowerB: EfficientNet-B5 global
- TowerC: ViT-B/8 (patch_size=8, img_size=288 → 1296 patches)
- PrototypeNet: DenseNet-169
- DINOAdaptBackbone: in_chans=9 (for Phase T1 and tower weight init)
- DINOFeatureBackbone: in_chans=3 (for Stage 3 OOD at inference)

### Losses (trainer/losses.py)
- ArcMarginProduct (s=64, m=0.5)
- SupConLoss (temperature=0.07)
- ProxyNCALoss

### Runtime service (runtime_service/)
- FastAPI: POST /predict, POST /predict_debug
- request_id + image_paths schema (Windows Unicode path safe)
- Windows .bat launch scripts

### Monitoring (trainer/monitoring_alerts.py)
- Alert 1: class rejection rate rising (7-day rolling, 2× baseline)
- Alert 2: global TTA variance rising (14-day, 1.20× baseline)
- Alert 3: per-tower disagreement rate rising
- Alert 4: OOD rejection rate spike (1-day, 3× baseline)
- Alert 5: constraint gate failing after tower agreement
- Alert 6: calibration P-value inflation (reviewed accuracy vs predicted P)

## Not included (artifact-only — produced by training run)

These exist after the Kaggle training job completes on your real dataset:
- Trained tower weights (tower_a.pt, tower_b.pt, tower_c.pt)
- Prototype model weights and library
- OOD FAISS index and threshold
- Class constraint profiles
- Calibrator and variance thresholds
- Background reference (captured by you from your hardware)
- Full runtime bundle (runtime_bundle/)
