import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import RangeNet
from .range_image import range_image

RELLIS_TRAIN_CLASSES = {
    1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 8: 5, 9: 6, 10: 7, 12: 8,
    15: 9, 17: 10, 18: 11, 19: 12, 23: 13, 27: 14, 29: 15,
    30: 16, 31: 17, 32: 15, 33: 17, 34: 17,
}

NUM_CLASSES = 18
CLASS_NAMES = ['dirt','grass','tree','pole','water','vehicle','object','asphalt',
               'building','log','person','fence','bush','concrete','barrier',
               'puddle','mud','rubble']

_MAX_RELLIS_KEY = max(RELLIS_TRAIN_CLASSES.keys())
_RELLIS_LUT = np.full(_MAX_RELLIS_KEY + 1, -1, dtype=np.int64)
for k, v in RELLIS_TRAIN_CLASSES.items():
    _RELLIS_LUT[k] = v


def load_rellis_labels(path, count):
    if Path(path).stat().st_size != count * 4:
        raise ValueError(f'RELLIS label count mismatch for {path}')
    raw = np.fromfile(path, dtype='<u4') & 0xFFFF
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
        y = load_rellis_labels(label_path, len(p))
        x, target, *_ = range_image(p, y, **self.projection)
        if not (target >= 0).any():
            raise ValueError(f'{bin_path}: no labels within configured field of view')
        return torch.from_numpy(x).float(), torch.from_numpy(target).long()


def main():
    p = argparse.ArgumentParser(description='Train the range-image CNN on RELLIS-3D scans')
    p.add_argument('--bin-root', required=True)
    p.add_argument('--label-root', required=True)
    p.add_argument('--bin-subdir', default='os1_cloud_node_kitti_bin')
    p.add_argument('--label-subdir', default='os1_cloud_node_semantickitti_label_id')
    p.add_argument('--val-sequences', nargs='*', default=None)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--output', default='checkpoints/best.pt')
    p.add_argument('--resume-from', default=None, help='Checkpoint to resume training from (e.g. checkpoints/last.pt)')
    p.add_argument('--epochs', type=int, default=20, help='Total epochs to reach across all resumed runs, not additional epochs this run')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--height', type=int, default=64)
    p.add_argument('--fov-up', type=float, default=15)
    p.add_argument('--fov-down', type=float, default=-25)
    p.add_argument('--class-weights', type=float, nargs=NUM_CLASSES, default=[1.0] * NUM_CLASSES)
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
        persistent_workers=(a.workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = RangeNet(classes=NUM_CLASSES).to(a.device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=-1,
        weight=torch.tensor(a.class_weights, dtype=torch.float32, device=a.device),
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(use_cuda and a.amp))

    start_epoch = 0
    best = float('inf')
    log = []

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    last_path = out.with_name(out.stem + '_last' + out.suffix)  # e.g. checkpoints/best_last.pt

    resume_path = a.resume_from or (last_path if last_path.exists() else None)
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=a.device, weights_only=True)
        if ckpt.get('architecture') != 'foveamap-range-v1':
            raise ValueError(f'{resume_path}: unsupported checkpoint architecture')
        if ckpt.get('num_classes', NUM_CLASSES) != NUM_CLASSES:
            raise ValueError(f'{resume_path}: checkpoint has {ckpt.get("num_classes")} classes, expected {NUM_CLASSES}')
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optim.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best = ckpt.get('validation_loss', float('inf'))
        training_json = out.with_suffix('.training.json')
        if training_json.exists():
            log = json.loads(training_json.read_text())
        print(f'Resumed from {resume_path} at epoch {start_epoch} (best val loss so far: {best:.4f})')

    if start_epoch >= a.epochs:
        print(f'Checkpoint already at epoch {start_epoch} >= --epochs {a.epochs}; nothing to do. '
              f'Pass a higher --epochs to continue training.')
        return

    def save_checkpoint(path, epoch, val_loss):
        torch.save({
            'architecture': 'foveamap-range-v1',
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'projection': projection,
            'validation_loss': val_loss,
            'epoch': epoch,
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES,
            'label_source': 'rellis-3d-full-ontology',
            'class_map': RELLIS_TRAIN_CLASSES,
        }, path)

    for epoch in range(start_epoch, a.epochs):
        epoch_stats = {}
        for phase, loader, is_train in [('train', train_loader, True), ('val', val_loader, False)]:
            model.train(is_train)
            total_loss = 0.0
            total_samples = 0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1:02d}/{a.epochs:02d} [{phase}]", unit="batch", dynamic_ncols=True)
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
                pbar.set_postfix({'loss': f"{total_loss / total_samples:.4f}"})
            epoch_stats[phase] = total_loss / total_samples

        record = {'epoch': epoch + 1, **epoch_stats}
        log.append(record)
        print(f"--> Epoch {epoch + 1} complete | Train Loss: {epoch_stats['train']:.4f} | Val Loss: {epoch_stats['val']:.4f}", flush=True)

        # Save every epoch to `_last` so you can always resume from the most recent point
        save_checkpoint(last_path, epoch + 1, epoch_stats['val'])
        print(f"    [*] Resume checkpoint saved to {last_path}")

        # Separately keep the best-val-loss checkpoint for actual deployment/inference
        if epoch_stats['val'] < best:
            best = epoch_stats['val']
            save_checkpoint(out, epoch + 1, best)
            print(f"    [*] Best checkpoint saved to {out} (Val loss: {best:.4f})")

        out.with_suffix('.training.json').write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()