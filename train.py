"""Minimal Chapter 3 training; not a recovered historical training snapshot."""
import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from training.data import TrainingPairs, paired_names, read_prompts
from training.losses import Chapter3Loss

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=['msrs', 'llvip', 'roadscene'])
    p.add_argument('--data-root', type=Path, required=True, help='Training pool containing ir/ and vi/, excluding test images.')
    p.add_argument('--prompt-path', type=Path, required=True)
    p.add_argument('--mask-dir', type=Path, help='MSRS class-index labels; required for training and validation.')
    p.add_argument('--judge-weights', type=Path, help='MSRS 9-class DeepLabV3-ResNet50 weights.')
    p.add_argument('--clip-path', type=Path, default=ROOT / 'clip_weights')
    p.add_argument('--output', type=Path, required=True, help='New or empty run directory.')
    p.add_argument('--split-file', type=Path, help='Optional JSON with train and val filename lists.')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--accumulation-steps', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--resize-mismatched-pairs', action='store_true',
                   help='Explicitly resize different IR/VIS dimensions to image-size. This does not perform registration.')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-batches', type=int, help='Limit train/validation batches for a smoke check only.')
    p.add_argument('--check', action='store_true', help='Read all data and check labels/prompts without loading model weights.')
    a = p.parse_args()
    for key in ('epochs', 'batch_size', 'accumulation_steps', 'image_size'):
        if getattr(a, key) <= 0:
            p.error(f'{key} must be positive')
    if a.image_size < 32 or a.image_size % 16 or a.lr <= 0 or a.workers < 0:
        p.error('image-size must be >=32 and divisible by 16; lr>0; workers>=0')
    if a.max_batches is not None and a.max_batches < 1:
        p.error('max-batches must be positive')
    if a.dataset == 'msrs':
        if not a.mask_dir or not a.judge_weights or not a.judge_weights.is_file():
            p.error('MSRS requires --mask-dir and an existing --judge-weights file')
    elif a.mask_dir or a.judge_weights:
        p.error('Chapter 3 disables Judge supervision for LLVIP and RoadScene')
    if a.output.exists() and (not a.output.is_dir() or any(a.output.iterdir())):
        p.error('output must be new or empty')
    data, output = a.data_root.resolve(), a.output.resolve()
    if output == data or output in data.parents or data in output.parents:
        p.error('output must be separate from the input dataset')
    return a


def split_names(names, path):
    if path:
        split = json.loads(path.read_text(encoding='utf-8'))
        if set(split) != {'train', 'val'}:
            raise ValueError('Split JSON must have exactly train and val keys.')
    else:
        boundary = max(1, min(len(names) - 1, int(len(names) * 0.9)))
        split = {'train': names[:boundary], 'val': names[boundary:]}
    train, val = split['train'], split['val']
    if not train or not val or len(set(train + val)) != len(train + val):
        raise ValueError('Need nonempty, nonoverlapping train/val lists without duplicates.')
    if not set(train + val).issubset(names):
        raise ValueError('Split lists contain filenames absent from the paired inputs.')
    return split


def run_epoch(model, objective, loader, device, optimizer=None, accumulation=1, max_batches=None):
    training = optimizer is not None
    model.train(training)
    model.text_encoder.eval()
    objective.eval()
    steps = min(len(loader), max_batches) if max_batches else len(loader)
    total_samples = min(len(loader.dataset), steps * loader.batch_size)
    sums, seen = {}, 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader):
            if step >= steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            fused, _ = model(batch['ir'], batch['vi'], batch['input_ids'], batch['attention_mask'])
            loss, parts = objective(fused, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at batch {step}')
            size = batch['ir'].shape[0]
            if training:
                window_start = (step // accumulation) * accumulation * loader.batch_size
                window_size = min(accumulation * loader.batch_size, total_samples - window_start)
                (loss * size / window_size).backward()
                if (step + 1) % accumulation == 0 or step + 1 == steps:
                    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float('inf'))
                    if not torch.isfinite(norm):
                        raise RuntimeError('Non-finite parameter gradients')
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            for key, value in {'total': loss, **parts}.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * size
            seen += size
    return {key: value / seen for key, value in sums.items()}


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    names = paired_names(a.data_root)
    split = split_names(names, a.split_file)
    prompts = read_prompts(a.prompt_path)
    tokenizer = CLIPTokenizer.from_pretrained(str(a.clip_path), local_files_only=True)
    datasets = {key: TrainingPairs(a.data_root, files, prompts, tokenizer, a.image_size,
                                 augment=key == 'train', masks=a.mask_dir,
                                 resize_mismatched_pairs=a.resize_mismatched_pairs)
                for key, files in split.items()}
    if a.resize_mismatched_pairs:
        print('Explicit independent resize enabled; geometric registration is not verified.')
    print(f"Training: {len(datasets['train'])}; validation: {len(datasets['val'])}")
    if a.check:
        for ds in datasets.values():
            for i in range(len(ds)):
                ds[i]
        print('All pairs, prompts and labels passed checks. No model was loaded.')
        return
    os.environ['TDPFUSION_CLIP_PATH'] = str(a.clip_path.resolve())
    from model.Text_IF_model import Text_IF
    device = torch.device(a.device)
    model = Text_IF(inp_A_channels=1, inp_B_channels=3).to(device)
    objective = Chapter3Loss(a.clip_path, a.judge_weights).to(device).eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    loaders = {key: DataLoader(ds, batch_size=a.batch_size if key == 'train' else 1,
                              shuffle=key == 'train', num_workers=a.workers)
               for key, ds in datasets.items()}
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(protocol='chapter3_reconstruction', historical_training_verified=False,
                  input_range=[-1, 1], output_range=[-1, 1], weight_decay=1e-4,
                  semantic_weight=0.1, judge_weight=0.2 if a.dataset == 'msrs' else 0.0,
                  learning_rate_schedule='constant', validation_every=5)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (a.output / 'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    (a.output / 'prompts.json').write_text(json.dumps({Path(n).stem: prompts[Path(n).stem]
                                                   for n in split['train'] + split['val']}, indent=2), encoding='utf-8')
    best = float('inf')
    with (a.output / 'loss.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'phase', 'total', 'intensity', 'gradient', 'structure', 'color', 'semantic', 'judge'])
        writer.writeheader()
        for epoch in range(a.epochs):
            stats = run_epoch(model, objective, loaders['train'], device, optimizer, a.accumulation_steps, a.max_batches)
            writer.writerow(dict(epoch=epoch, phase='train', **stats))
            print(f"Epoch {epoch + 1}/{a.epochs}: loss={stats['total']:.6f}", flush=True)
            improved = False
            if epoch % 5 == 0 or epoch + 1 == a.epochs:
                val = run_epoch(model, objective, loaders['val'], device, max_batches=a.max_batches)
                writer.writerow(dict(epoch=epoch, phase='val', **val))
                improved = val['total'] < best
                best = min(best, val['total'])
            state = {'model': model.state_dict(), 'epoch': epoch, 'config': config, 'best_val_loss': best}
            torch.save(state, a.output / 'latest_checkpoint.pth')
            if improved:
                torch.save(state, a.output / 'best_checkpoint.pth')
            stream.flush()


if __name__ == '__main__':
    main()
