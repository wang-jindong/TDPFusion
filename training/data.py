"""Aligned training pairs with explicit prompts and synchronized augmentation."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def read_prompts(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if isinstance(raw, dict):
        entries = list(raw.items())
    elif isinstance(raw, list):
        entries = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError('Each prompt record must be an object.')
            name = next((item[k] for k in ('file_name', 'image', 'id') if k in item), None)
            text = next((item[k] for k in ('text', 'caption', 'description') if k in item), None)
            entries.append((name, text))
    else:
        raise ValueError('Prompt JSON must be a mapping or list of records.')
    prompts = {}
    for name, text in entries:
        if name is None or not isinstance(text, str) or not text.strip():
            raise ValueError(f'Invalid prompt record: {name!r}')
        key = Path(str(name)).stem
        if key in prompts:
            raise ValueError(f'Duplicate prompt ID: {key}')
        prompts[key] = text.strip()
    return prompts


def paired_names(root):
    root = Path(root)
    groups = []
    for folder in ('ir', 'vi'):
        directory = root / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        groups.append({p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS})
    if groups[0] != groups[1]:
        raise ValueError('IR/VIS filenames do not match; repair the pairs before training.')
    if not groups[0]:
        raise ValueError('No input image pairs found.')
    names = sorted(groups[0])
    if len({Path(name).stem for name in names}) != len(names):
        raise ValueError('Image stems must be unique for prompt/label lookup.')
    return names


class TrainingPairs(Dataset):
    def __init__(self, root, names, prompts, tokenizer, image_size, augment=False, masks=None,
                 resize_mismatched_pairs=False):
        self.root, self.names = Path(root), names
        self.size, self.augment = image_size, augment
        self.resize_mismatched_pairs = resize_mismatched_pairs
        self.masks = Path(masks) if masks else None
        missing = [n for n in names if Path(n).stem not in prompts]
        if missing:
            raise ValueError(f'Missing training prompts for {missing[:5]}')
        if self.masks:
            for name in names:
                if not (self.masks / (Path(name).stem + '.png')).is_file():
                    raise FileNotFoundError(f'Missing semantic label: {name}')
        self.tokens = tokenizer([prompts[Path(n).stem] for n in names],
                                padding='max_length', truncation=True,
                                max_length=77, return_tensors='pt')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.root / 'ir' / name) as image:
            ir = image.convert('L')
        with Image.open(self.root / 'vi' / name) as image:
            vi = image.convert('RGB')
        if ir.size != vi.size and not self.resize_mismatched_pairs:
            raise ValueError(f'Different IR/VIS dimensions: {name}. Confirm preprocessing before using --resize-mismatched-pairs.')
        mask = None
        if self.masks:
            with Image.open(self.masks / (Path(name).stem + '.png')) as image:
                mask = image.copy()
            if mask.size != ir.size or np.asarray(mask).ndim != 2:
                raise ValueError(f'Expected aligned single-channel class IDs: {name}')
        ir = ir.resize((self.size, self.size), Image.Resampling.BILINEAR)
        vi = vi.resize((self.size, self.size), Image.Resampling.BILINEAR)
        if mask is not None:
            mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        if self.augment and random.random() < 0.5:
            ir, vi = ir.transpose(Image.Transpose.FLIP_LEFT_RIGHT), vi.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        data = {'ir': to_tensor(ir) * 2 - 1, 'vi': to_tensor(vi) * 2 - 1,
                'input_ids': self.tokens['input_ids'][index],
                'attention_mask': self.tokens['attention_mask'][index]}
        if mask is not None:
            ids = np.array(mask, dtype=np.int64)
            if not np.all(((ids >= 0) & (ids < 9)) | (ids == 255)):
                raise ValueError(f'MSRS class IDs must be 0..8 or 255: {name}')
            data['mask'] = torch.from_numpy(ids)
        return data
