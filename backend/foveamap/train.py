import argparse
import json
import os
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import RangeNet
from .range_image import range_image

# 0=drivable, 1=non-drivable terrain, 2=static obstacle, 3=dynamic object, -1=ignore
RELLIS_TO_4CLASS = {
    0: -1, 1: 0, 3: 1, 4: 2, 5: 2, 6: 1, 7: -1, 8: 3, 9: 2, 10: 0,
    12: 2, 15: 2, 17: 3, 18: 2, 19: 2, 23: 0, 27: 2, 29: 1, 30: 1,
    31: 2, 32: 1, 33: 2, 34: 2,
}

# Precompute the lookup table once globally instead of re-allocating inside the loader loop
_MAX_RELLIS_KEY = max(RELLIS_TO_4CLASS.keys())
_RELLIS_LUT = np.full(_MAX_RELLIS_KEY + 1, -1, dtype=np.int64)
for k, v in RELLIS_TO_4CLASS.items():
    _RELLIS_LUT[k] = v


def load_rellis_as_4class(path, count):
    if Path(path).stat().st_size != count * 4:
        raise ValueError(f'RELLIS label count mismatch for {path}')
    raw = np.fromfile(path, dtype='<u4') & 0xFFFF
    
    # Direct vectorized mask lookup without creating LUT arrays per scan
    out = np.full(raw.shape, -1, dtype=np.int64)
    valid = raw < len(_RELLIS_LUT)
    out[valid] = _RELLIS_LUT[raw[valid]]
    return out


def load_bin(path):
    if Path(path).stat().st_size % 16:
        raise ValueError(f'{path}: KITTI scans require 16-byte float32 records')
    points = np.fromfile(path, dtype='<f4').reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f'{path}: non-finite points')
    return points


def discover_sequences(bin_root, label_root, bin_subdir, label_subdir):
    bin_root, label_root = Path(bin_root), Path(label_root)
    sequences = {}
    for seq_dir in sorted(bin_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq = seq_dir.name
        bin_dir = seq_dir / bin_subdir
        label_dir = label_root / seq / label_subdir
        if not bin_dir.is_dir():
            continue
        pairs = []
        for bin_path in sorted(bin_dir.glob('*.bin')):
            label_path = label_dir / (bin_path.stem + '.label')
            if label_path.exists():
                pairs.append((bin_path, label_path))
            else:
                print(f'WARNING: no label for {bin_path}, skipping')
        if pairs:
            sequences[seq] = pairs
    if not sequences:
        raise ValueError(f'No sequences with paired bin/label found under {bin_root}')
    return sequences


def split_sequences(sequences, val_sequences=None, val_fraction=0.2, seed=42):
    names = sorted(sequences.keys())
    if val_sequences:
        missing = set(val_sequences) - set(names)
        if missing:
            raise ValueError(f'--val-sequences not found: {sorted(missing)}')
        val_names = list(val_sequences)
    else:
        rng = random.Random(seed)
        shuffled = names[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val_names = shuffled[:n_val]
    train_names = [n for n in names if n not in val_names]
    if not train_names or not val_names:
        raise ValueError('Split produced an empty train or val set')
    train_pairs = [p for n in train_names for p in sequences[n]]
    val_pairs = [p for n in val_names for p in sequences[n]]
    print(f'Train sequences: {train_names} ({len(train_pairs)} scans)')
    print(f'Val sequences:   {val_names} ({len(val_pairs)} scans)')
    return train_pairs, val_pairs


class RellisScans(Dataset):
    def __init__(self, pairs, projection):
        self.pairs = pairs
        self.projection = projection

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        bin_path, label_path = self.pairs[i]
        p = load_bin(bin_path)
        y = load_rellis_as_4class(label_path, len(p))
        x, target, *_ = range_image(p, y, **self.projection)
        if not (target >= 0).any():
            raise ValueError(f'{bin_path}: no labels within configured field of view')
        return torch.from_numpy(x).float(), torch.from_numpy(target).long()


def main():
    p = argparse.ArgumentParser(description='Train the four-class range-image CNN on RELLIS-3D scans')
    p.add_argument('--bin-root', required=True, help='Root containing sequence folders')
    p.add_argument('--label-root', required=True, help='Root containing matching sequence folders for labels')
    p.add_argument('--bin-subdir', default='os1_cloud_node_kitti_bin')
    p.add_argument('--label-subdir', default='os1_cloud_node_semantickitti_label_id')
    p.add_argument('--val-sequences', nargs='*', default=None, help='Sequence names to hold out')
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--output', default='checkpoints/best.pt')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4, help='DataLoader CPU worker processes')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp', action='store_true', default=True, help='Use PyTorch Automatic Mixed Precision (AMP)')
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--height', type=int, default=64)
    p.add_argument('--fov-up', type=float, default=15)
    p.add_argument('--fov-down', type=float, default=-25)
    p.add_argument('--class-weights', type=float, nargs=4, default=[1, 1, 1, 1])
    a = p.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    
    if 'cuda' in a.device and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    sequences = discover_sequences(a.bin_root, a.label_root, a.bin_subdir, a.label_subdir)
    train_pairs, val_pairs = split_sequences(sequences, a.val_sequences, a.val_fraction, a.split_seed)

    projection = dict(height=a.height, width=a.width, fov_up=a.fov_up, fov_down=a.fov_down)
    train_ds = RellisScans(train_pairs, projection)
    val_ds = RellisScans(val_pairs, projection)

    use_cuda = 'cuda' in a.device
    loader_kwargs = dict(
        batch_size=a.batch_size,
        num_workers=a.workers,
        pin_memory=use_cuda,
        persistent_workers=(a.workers > 0)
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = RangeNet().to(a.device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=-1,
        weight=torch.tensor(a.class_weights, dtype=torch.float32, device=a.device)
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(use_cuda and a.amp))

    best = float('inf')
    log = []
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(a.epochs):
        epoch_stats = {}
        for phase, loader, is_train in [('train', train_loader, True), ('val', val_loader, False)]:
            model.train(is_train)
            total_loss = 0.0
            total_samples = 0

            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch + 1:02d}/{a.epochs:02d} [{phase}]",
                unit="batch",
                dynamic_ncols=True
            )

            for x, y in pbar:
                x = x.to(a.device, non_blocking=use_cuda)
                y = y.to(a.device, non_blocking=use_cuda)

                with torch.set_grad_enabled(is_train):
                    with torch.amp.autocast('cuda', enabled=(use_cuda and a.amp)):
                        out_logits = model(x)
                        loss = loss_fn(out_logits, y)

                    if is_train:
                        optim.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(optim)
                        scaler.update()

                bs = len(x)
                total_loss += float(loss.detach()) * bs
                total_samples += bs
                
                # Live moving average loss on the progress bar
                pbar.set_postfix({'loss': f"{total_loss / total_samples:.4f}"})

            epoch_stats[phase] = total_loss / total_samples

        record = {'epoch': epoch + 1, **epoch_stats}
        log.append(record)
        print(f"--> Epoch {epoch + 1} complete | Train Loss: {epoch_stats['train']:.4f} | Val Loss: {epoch_stats['val']:.4f}", flush=True)

        if epoch_stats['val'] < best:
            best = epoch_stats['val']
            torch.save({
                'architecture': 'foveamap-range-v1',
                'state_dict': model.state_dict(),
                'projection': projection,
                'validation_loss': best,
                'epoch': epoch + 1,
                'label_source': 'rellis-3d-to-4class',
                'class_map': RELLIS_TO_4CLASS,
            }, out)
            print(f"    [*] Checkpoint saved to {out} (Val loss: {best:.4f})")

    out.with_suffix('.training.json').write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()