# Training from the existing source

The training code is an organization of the supplied, currently existing code.
The previous Chapter 3 reconstruction has been removed. No loss, gradient path,
sampling rule, augmentation, optimizer schedule or checkpoint selection rule is
reimplemented from the thesis. This does not establish that the existing scripts
are the historical versions which produced the released weights.

## Source mapping

| Release file | Existing source |
| --- | --- |
| `training/msrs.py` | `train_fusion.py` |
| `training/llvip.py` | `train_fusion_llvip.py` |
| `training/roadscene.py` | `trian_fusion_roadscene.py` from textif3 |
| `training/patent_dataset.py` | `patent_dataset.py` |
| `training/roadscene_dataset.py` | `roadscene_dataset.py` |
| `training/losses.py` | `scripts/patent_loss.py` |
| `training/semantic_loss.py` | `scripts/semantic_loss.py` |
| `training/utils.py` | Only `setup_seed` and `create_lr_scheduler`, copied from `scripts/utils.py` |

Training loop bodies, loss calculations and dataset operations are preserved.
Changes are limited to module imports, configurable resource/output paths and
extracting the original argument parsers for a shared launcher. The model's only
source change is its configurable CLIP path. Original tokenizer/model fallback
behavior is retained. Supply the same CLIP assets for comparable results.

The server copies of the three textif3 trainers match the local originals apart
from line endings. Historical loss/data dependency snapshots are not established.
In particular, the RoadScene source writes `RoadScene_FullFixed_*` runs; it has
not been established as the source of the released `RoadScene_Semantic_*` weights.
The alternative textif4 trainer is not substituted or combined with it.

## Required resources and commands

Run from the repository root. Install `requirements.txt`, including TensorBoard.
Supply paired training IR/VIS images, the source-format masks/pseudo-labels,
training prompt JSON, full CLIP ViT-B/32 weights and the appropriate pretrained
Judge. MSRS/RoadScene use the 9-class Judge; LLVIP uses the 2-class Judge. These
Judge checkpoints are separate from the released fusion checkpoints. Their public
download links and provenance are still pending.

```bash
python -X utf8 train.py --dataset msrs --ir_path /data/MSRS_train/ir --vis_path /data/MSRS_train/vi --mask_path /data/MSRS_train/Label --prompt_path /data/MSRS_train/prompts.json --judge_weights /models/best_judge.pth --clip-path /models/clip_weights
python -X utf8 train.py --dataset llvip --ir_path /data/LLVIP_train/ir --vis_path /data/LLVIP_train/vi --mask_path /data/LLVIP_train/masks/train --judge_weights /models/best_judge_llvip.pth --clip-path /models/clip_weights
python -X utf8 train.py --dataset roadscene --ir_path /data/RoadScene_train/ir --vis_path /data/RoadScene_train/vi --mask_path /data/RoadScene_train/masks/pseudo_labels --prompt_path /data/RoadScene_train/road_train_prompt.json --judge_weights /models/best_judge.pth --clip-path /models/clip_weights
```

Use `python train.py --dataset NAME --help` for the original dataset-specific
options. `--output-root` changes only the parent of the original timestamped
experiment folders (default `./experiments`). RoadScene retains its optional
`--pretrained_weights` argument; its default is empty. Relative paths are resolved
from the working directory. No training split or sample is downloaded or generated.

The removed reconstruction's `--data-root`, `--split-file`, `--image-size`,
`--resize-mismatched-pairs`, `--max-batches`, `--check` and `--output` options do not
apply. The original trainers use size 256 and four DataLoader workers. Use the
actual training pool; do not substitute held-out test data.

## Preserved behavior

| Setting | MSRS | LLVIP | RoadScene candidate |
| --- | --- | --- | --- |
| Default epochs | 70 | 70 | 70 |
| Batch size / accumulation | 2 / 8 | 1 / 16 | 2 / no accumulation |
| AdamW nominal LR / decay | 1e-4 / 1e-4 | 1e-4 / 1e-4 | 1e-4 / 1e-4 |
| LR scheduler | Original LambdaLR, stepped once per epoch | Same | Same |
| Sampling / split | Filesystem order, first 90% train | Sort, take every tenth pair, first 90% train | Sort, first 90% train; validation list unused |
| Training images | Resize and horizontal flip | Resize and horizontal flip | Original minimum-size resize, random crop and flip |
| Text | Original numeric-ID lookup and fallback | Fixed default text, as in the existing trainer | Original JSON lookup and fallback |
| External Judge coefficient | 0.1 | 0.5 | 0.02 |
| Saved weights | Best fusion validation loss, epochs 0,5,... | Same; fixed validation text | Latest after every epoch |

The fusion objective retains intensity weight 5, gradient weight 5, SSIM weight
1, Cb/Cr color weight 8 and CLIP semantic weight 0.5. Intensity uses the original
gray-based whole-pixel selector, gradients use grayscale, and IR SSIM compares
grayscale images. Original SSIM padding, dynamic-range inference and numerical
epsilon are retained. Auxiliary mask BCE (weight 0.2) is retained in all trainers.
RoadScene also retains output clipping to [-1.02,1.02] and the 0.05 mean-square
brightness penalty. Inputs retain the source training normalization [-1,1].

Several existing behaviors need to remain visible, without silently correcting
them as part of organization:

- Judge forward passes are inside `torch.no_grad()`. Their losses contribute to
  the reported total but do not backpropagate through the Judge to fusion images.
- In `patent_dataset.py`, image flips are not applied to the mask. Missing masks
  can become zero masks and failed image reads can advance to another sample.
- LLVIP's training prompt mapping is empty even though its parser exposes a
  prompt path. The trainer uses its fixed default prompt.
- The original scheduler is parameterized by the number of batches but stepped
  once per epoch. The final partial gradient-accumulation window still uses the
  original fixed divisor. Neither behavior is corrected here.
- RoadScene retains its permissive Judge loading and original handling of
  mismatched image dimensions. The supplied training images have differing IR/VIS
  dimensions; this release does not infer or add a registration/resize procedure.
- Checkpoints contain the original model state payload, without optimizer or RNG
  state for exact training resumption. No new validation objective or final-epoch
  selection rule is added.

These are descriptions of existing code, not endorsements of its correctness or
proof of the paper's historical protocol. Full retraining and reproduction of the
published metrics have not been established. Existing dataset-specific inference
presets remain unchanged; their suitability for newly trained weights is not
established by organizing the training code.

## Organization checks (2026-10-15)

All existing function/class syntax trees were compared after normalizing only
documented path and module-location changes; training computation trees and
non-path argument defaults matched. All release Python sources compiled, and all
three dataset command-line help paths loaded successfully.

On CPU with the actual local CLIP model, the source and organized `PatentLoss`
produced exactly equal losses and fused-image gradients on a synthetic batch.
Both dataset classes produced equal image, label and token tensors for training
and validation, including their existing missing-mask and size-mismatch behavior.
MSRS/LLVIP training-loop checks with small test models matched parameter updates,
partial accumulation, Judge no-grad behavior and scheduler steps. RoadScene's
training body was checked structurally; no full RoadScene training run was made.
OpenCV image encoding matched byte-for-byte through the Unicode-path adapter.
These checks establish organization consistency, not historical metric reproduction.
