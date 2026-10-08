# TDPFusion

Text-guided infrared and visible image fusion. This release provides the model
and inference code for MSRS, LLVIP, and RoadScene, with a separate checkpoint
for each dataset.

Repository: [wang-jindong/TDPFusion](https://github.com/wang-jindong/TDPFusion).

**Release status:** source code is available; pretrained checkpoint download links
are pending. The training entry point is a reconstructed reference implementation,
not a verified reproduction of the released model weights. See
[TRAINING.md](TRAINING.md) for its scope and limitations.

## Installation

Use Python 3.10 and install a matching PyTorch/torchvision build for your hardware,
then install the remaining dependencies:

```bash
git clone https://github.com/wang-jindong/TDPFusion.git
cd TDPFusion
pip install -r requirements.txt
```

The inference code has been run with PyTorch 2.5.1, torchvision 0.20.1,
transformers 4.36.2, and NumPy 1.26.4.

## Pretrained models

Place the fusion checkpoints at the paths below. Download links are pending.

| Dataset | Checkpoint path | Original experiment |
|---|---|---|
| MSRS | `weights/msrs/best_checkpoint.pth` | `TextIF_Ours_Full_20260225-183631` |
| LLVIP | `weights/llvip/best_checkpoint.pth` | `LLVIP_Ours_20260225-003151` |
| RoadScene | `weights/roadscene/latest_model.pth` | `RoadScene_Semantic_0226_1407` |

Put the CLIP model and tokenizer files in `clip_weights/`: `config.json`,
`pytorch_model.bin` (or `model.safetensors`), `vocab.json`, `merges.txt`, and
`tokenizer_config.json`. The model loads CLIP locally. Use `--clip-path` to
select another directory.

Fusion weights, CLIP, datasets, and generated results are excluded from Git.

## Inference

Prepare paired images with matching filenames:

```text
dataset/
├── ir/
└── vi/
```

MSRS also supports `Infrared/Visible`; RoadScene also supports
`cropinfrared/crop_HR_visible`. Input images must be aligned and have the same
size. No resizing is performed.

Run from the repository root:

```bash
python -X utf8 infer.py --dataset msrs --data-root /path/to/MSRS --output results/msrs
python -X utf8 infer.py --dataset llvip --data-root /path/to/LLVIP --output results/llvip
python -X utf8 infer.py --dataset roadscene --data-root /path/to/RoadScene --output results/roadscene
```

Output directories must be new or empty. Add `--check` to check required paths
without loading the model. Optional arguments include `--weights`,
`--clip-path`, `--prompt-path`, `--prompt-mode fixed`, and `--gpu-id`.
MSRS supports `--save-mask` for saving auxiliary masks.

### Dataset settings

| Dataset | Input range | Output conversion | Text |
|---|---|---|---|
| MSRS | `[-1,1]` | Clip to `[-1,1]`, then map to uint8 | `prompts/msr_test_prompt.json` |
| LLVIP | `[0,1]` | Clip to `[0,1]`, then map to uint8 | Default text with the supplied JSON; see below |
| RoadScene | `[-1,1]` | Global 0.5/99.5 percentile clipping and stretching | `a clear road scene fusion` |

MSRS and RoadScene use reflection padding with a 32-pixel border and alignment
to 16. LLVIP pads only the right and bottom edges to a multiple of 16.

The supplied LLVIP JSON is a list, while the original inference script looks up
numeric IDs as dictionary keys. This preserved behavior uses the default
`infrared and visible image fusion` text for every image. Converting this JSON
into a dictionary would change inference behavior. For MSRS, unmatched filenames
also use this default text.

These are dataset-specific inference settings and checkpoints; the release does
not claim a single checkpoint for all three datasets.

## Repository layout

```text
infer.py          Inference entry point
train.py          Reconstructed training entry point (draft)
training/         Training data and Chapter 3 objective
model/            Fusion network and text encoder
inference/        Dataset-specific image processing and inference
configs/          Checkpoint paths and text defaults
prompts/          Supplied prompt files
weights/          Local fusion checkpoints (ignored by Git)
clip_weights/     Local CLIP assets (ignored by Git)
```

The original pretrained inference workflow is preserved. A minimal training
draft reconstructed from thesis Chapter 3 is provided in `train.py` and
`training/`; see [TRAINING.md](TRAINING.md) for commands, required resources,
implementation choices and validation limits. It is not a recovered historical
training implementation and is not verified to reproduce the released weights.
Experimental snapshots and local investigation records remain outside the release.

## Attribution

The original MIT license (Copyright 2024 Xunpeng Yi) is retained. The code uses
Text-IF/Restormer-style components and Hugging Face CLIP. Paper metadata and citation
information will be added when available.
