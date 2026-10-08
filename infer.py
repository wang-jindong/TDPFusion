"""Run TDPFusion with the preserved dataset-specific inference settings."""
import argparse
import importlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['msrs', 'llvip', 'roadscene'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New or empty output directory.')
    parser.add_argument('--weights', type=Path, help='Override the dataset checkpoint.')
    parser.add_argument('--clip-path', type=Path, default=ROOT / 'clip_weights')
    parser.add_argument('--prompt-path', type=Path)
    parser.add_argument('--prompt-mode', choices=['file', 'fixed'])
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--save-mask', action='store_true', help='MSRS only.')
    parser.add_argument('--check', action='store_true', help='Check required paths without loading the model.')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs' / f'{args.dataset}.json').read_text(encoding='utf-8'))
    data, output = args.data_root.resolve(), args.output.resolve()
    args.weights = str((args.weights or ROOT / config['weights']).resolve())
    args.clip_path = str(args.clip_path.resolve())
    prompt = args.prompt_path or (ROOT / config['prompt_path'] if config['prompt_path'] else None)
    mode = args.prompt_mode or ('file' if prompt else 'fixed')
    if mode == 'fixed' and args.prompt_path:
        parser.error('--prompt-path cannot be used with --prompt-mode fixed')
    args.prompt_path = str(prompt.resolve()) if mode == 'file' and prompt else ''
    args.default_text = config['default_text']
    args.dataset_path, args.save_path = str(data), str(output)

    directories = {
        'msrs': [('ir', 'vi'), ('Infrared', 'Visible')],
        'llvip': [('ir', 'vi')],
        'roadscene': [('cropinfrared', 'crop_HR_visible'), ('ir', 'vi')],
    }[args.dataset]
    pair = next(((data / ir, data / vi) for ir, vi in directories
                 if (data / ir).is_dir() and (data / vi).is_dir()), None)
    if pair is None:
        parser.error(f'Missing paired input folders under {data}; expected {directories}')
    args.ir_path, args.vis_path = map(str, pair)
    if not Path(args.weights).is_file():
        parser.error(f'Missing checkpoint: {args.weights}')
    clip = Path(args.clip_path)
    if not all((clip / name).is_file() for name in ['config.json', 'vocab.json', 'merges.txt']):
        parser.error(f'Missing CLIP model/tokenizer configuration: {clip}')
    if not any((clip / name).is_file() for name in ['pytorch_model.bin', 'model.safetensors']):
        parser.error(f'Missing CLIP weights: {clip}')
    if mode == 'file' and not Path(args.prompt_path).is_file():
        parser.error(f'Missing prompt JSON: {args.prompt_path}')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f'Output must be new or empty: {output}')
    if output == data or data in output.parents or output in data.parents:
        parser.error('Output must be separate from the input directory.')
    if args.save_mask and args.dataset != 'msrs':
        parser.error('--save-mask is supported only for MSRS')
    return args


def main():
    args = parse_args()
    print(f'Dataset: {args.dataset}\nWeights: {args.weights}\nOutput: {args.save_path}')
    if args.check:
        print('Required paths exist. No model was loaded.')
        return
    os.environ['TDPFUSION_CLIP_PATH'] = args.clip_path
    pipeline = importlib.import_module(f'inference.{args.dataset}')
    pipeline.main(args)
    images = [p for p in Path(args.save_path).iterdir()
              if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
              and not p.name.startswith('mask_')]
    if not images:
        raise RuntimeError('No fused images were saved. Check paired inputs and checkpoint loading.')


if __name__ == '__main__':
    main()
