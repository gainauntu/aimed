# TRAINER_TOOL - Kaggle rebuild baseline (single-image encoder + multi-image runtime fusion)

This package is the trainer/export side of the rebuild.

## Goal
Train a single-image encoder on class-folder data, then use that encoder in the runtime to:
- score each input frame independently
- fuse multiple frames at inference time
- reject unknown / inconsistent samples as `UNDECIDED`

## Current design
- training: single-image supervised learning
- runtime: multi-image fusion
- model outputs:
  - logits
  - embedding
- target environment:
  - Kaggle notebook
  - T4 x2
  - `torchrun --nproc_per_node=2`

## Expected dataset structure
dataset_root/
- MI 5/
- EL/
- SPP RM/
- ...

Each image is treated as a valid labeled view of the class.

## Outputs
outputs/run1/
- best.pt
- last.pt
- classes.json
- metrics.json

EXPORT/runtime_bundle/
- pill_cls.onnx
- classes.json
- pack_manifest.json
- prototypes.npz
- gallery_refs.npz
