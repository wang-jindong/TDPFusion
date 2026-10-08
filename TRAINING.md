# Minimal training implementation (reconstructed)

`train.py` reconstructs the training objective and settings described in Chapter 3
of the source thesis. It is **not a recovered copy of the historical training
program**, and has not been shown to reproduce the released checkpoints or paper
metrics. The supplied experiment logs record 70 epochs; the thesis specifies 100.
The RoadScene historical training source could not be recovered from the supplied
files, and neither server project has a Git repository.

This entry point is a training draft pending validation on confirmed training
splits and confirmation of the preprocessing and implementation choices below.
It reuses the unchanged released network, including its supplied SSGA fallback.

## Data and resources

Use a training pool with aligned pairs; exclude all held-out test images:

```text
training_pool/
  ir/00001.png
  vi/00001.png
  mask/00001.png    # MSRS only: class IDs 0..8, ignore index 255
```

IR and VIS filenames must match. Labels use the same stem with `.png` and must
contain class indices, not RGB visualization colors. MSRS needs labels for both
training and validation. LLVIP and RoadScene do not use labels or a Judge in this
implementation, following Chapter 3.

Different IR/VIS dimensions are rejected by default. If the intended preprocessing
resizes both modalities independently, explicitly add `--resize-mismatched-pairs`.
This resizes each image to `--image-size` and records the choice in `config.json`;
it does not estimate geometric registration. A successful smoke run with this
option is not evidence that the original pairs are registered correctly or that
the historical training pipeline has been recovered.

Provide a JSON mapping image stems/filenames to scene-specific text, or a list:

```json
[{"file_name": "00001.png", "text": "Preserve visible road textures and highlight infrared pedestrians."}]
```

Every selected pair needs a prompt; missing prompts are errors. The included
20-image inference prompt files are not full training annotations. Supply the
actual training text data. LLaVA generation and cleaning are not implemented here;
the generator version, prompting procedure and complete annotations still need
to be supplied for reproducing that data-preparation stage.

The local `clip_weights/` must contain the full CLIP ViT-B/32 model (visual and text
encoders), not only text weights. MSRS additionally requires the pretrained
9-class DeepLabV3-ResNet50 Judge with auxiliary classifier. Its download link and
training provenance still need to be provided. Model weights load locally without
silently downloading or substituting pretrained assets.

## Commands

Run from the repository root with the same dependencies as inference:

```bash
python train.py --dataset msrs --data-root /data/MSRS_train --prompt-path /data/MSRS_train/prompts.json --mask-dir /data/MSRS_train/mask --judge-weights /models/best_judge.pth --output runs/msrs
python train.py --dataset llvip --data-root /data/LLVIP_train --prompt-path /data/LLVIP_train/prompts.json --output runs/llvip
python train.py --dataset roadscene --data-root /data/RoadScene_train --prompt-path /data/RoadScene_train/prompts.json --output runs/roadscene
```

Add `--check` to read all selected images and validate prompt/label coverage without
loading fusion, CLIP model or Judge weights. It loads only the CLIP tokenizer.
Use `--epochs 1 --max-batches 1` for a limited training/validation smoke run in a
separate output directory. A CPU smoke check can additionally use `--device cpu
--image-size 32`; this is not an experimental training configuration.

Default settings: image size 256, AdamW, learning rate 1e-4, weight decay 1e-4,
100 epochs, batch size 2, gradient accumulation 8, seed 42. The final accumulation
window is normalized by its actual sample count. Each dataset is trained
independently; fusion parameters start from random initialization, with frozen
pretrained CLIP. There is no implicit initialization from released fusion weights.

The default split sorts filenames and takes the first 90% for training, the rest
for validation. For a known split, pass `--split-file split.json` containing
`{"train": ["00001.png"], "val": ["00002.png"]}`. Lists must not overlap. Historical
sample lists have not been recovered; the deterministic default is not claimed
to be the original split. Choose the appropriate split before reporting results.

## Objective and explicit implementation choices

The objective is intensity + gradient + SSIM + color/edge + 0.1 * CLIP semantic
loss + 0.2 * Judge cross-entropy (MSRS only). It follows equations 3-27 through
3-37; historical scripts contain different objectives and weights.

- All training inputs and fusion outputs use [-1,1]. IR is expanded across RGB
  channels when comparing source/output images. Pixel losses use mean reductions.
- Both Sobel axes are summed. The color/edge term compares signed RGB Sobel
  responses as in equation 3-33, rather than the historical Cb/Cr loss.
- SSIM uses an 11x11 Gaussian window, sigma 1.5, data range 2, and replicated
  borders. Images resize to 256x256; horizontal flips are shared by both images
  and the semantic label. These discretization/augmentation details are explicit
  reconstruction choices, not recovered historical settings.
- CLIP and Judge parameters stay frozen and in evaluation mode. Gradients through
  their image branches remain enabled so their losses can update the fusion model.
- No auxiliary mask BCE or RoadScene brightness penalty is added. LLVIP prompts
  are read from the provided training JSON rather than replaced with fixed text.
- Learning rate is constant because Chapter 3 does not specify a scheduler.
  Validation uses the same total objective at epochs 0,5,10,... and the last epoch.
  This validation criterion and final check are reconstruction choices.

## Outputs and verification limits

Each new/empty run directory receives `config.json`, `split.json`, the selected
`prompts.json`, per-component `loss.csv`, and `best_checkpoint.pth` /
`latest_checkpoint.pth`. Checkpoints store model state, epoch, best validation
loss and protocol metadata. They do not include optimizer/RNG state for exact
training resumption. The original published checkpoint files are never replaced.

Small synthetic checks cover synchronized labels, gradient flow, one CPU
optimization/validation step with the real network, CLIP and MSRS Judge, and
checkpoint loading. Loading checks also passed on four supplied MSRS pairs and
four LLVIP pairs at 256x256. Those supplied directories are labeled as test data;
no optimization was performed on them. Supplied RoadScene files include 221 pairs
and complete prompts, but all pairs have different IR/VIS dimensions; the historical
registration/preprocessing procedure is unknown. Four sampled pairs passed the
256x256 loading check with `--resize-mismatched-pairs`; a disposable 64x64 CPU run
then passed one optimization batch, one validation batch, a parameter-update
check and strict checkpoint reloading. Judge supervision was disabled, as required
for RoadScene by Chapter 3. This confirms program execution only, not geometric
alignment, model quality or the historical training protocol.
This is not validation of training on the actual training split or a full run.

The existing `infer.py` remains for the original checkpoints and dataset-specific
preprocessing. Do not assume its preserved presets apply to newly trained weights:
in particular, original LLVIP inference uses [0,1] input and RoadScene applies
percentile stretching. A matching inference protocol for the reconstructed
training workflow has not been added in this draft. For direct tensor use, supply
IR/VIS in [-1,1], the matching CLIP tokens, and use the first model output as RGB
in [-1,1]. New-checkpoint inference and real-data training checks remain required
before presenting this as a complete reproducible training release.
