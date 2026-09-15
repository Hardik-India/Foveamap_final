from dataclasses import dataclass, asdict
from time import perf_counter
import math
import numpy as np

@dataclass(frozen=True)
class GridConfig:
    near: float = 10.
    mid: float = 30.
    far: float = 100.
    near_size: float = .05
    mid_size: float = .15
    far_size: float = .5
    def validate(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Configuration must be finite')
        if not (0 < self.near < self.mid < self.far <= 100):
            raise ValueError('Require 0 < near < mid < far <= 100')
        if not (.05 <= self.near_size <= self.mid_size <= self.far_size <= 2):
            raise ValueError('Require increasing cell sizes in [0.05, 2] metres')

def geometry(points):
    z = points[:, 2]
    return np.where(z < -.15, 1, np.where(z < .18, 0, np.where(z < .45, 1, 2))).astype(np.int64)

def addresses(points, config=GridConfig()):
    config.validate()
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('Expected N x 3 or N x 4 points')
    radius = np.hypot(points[:, 0], points[:, 1])
    valid = np.isfinite(points[:, :3]).all(axis=1) & (radius <= config.far)
    p, radius = points[valid], radius[valid]
    tier = np.where(radius < config.near, 0, np.where(radius < config.mid, 1, 2))
    start = np.asarray([0., config.near, config.mid])[tier]
    end = np.asarray([config.near, config.mid, config.far])[tier]
    size = np.asarray([config.near_size, config.mid_size, config.far_size])[tier]
    ring = np.minimum(np.ceil((end-start)/size).astype(np.int64)-1, np.floor((radius-start)/size).astype(np.int64))
    r0 = start + ring * size
    r1 = np.minimum(end, r0 + size)
    sectors = np.maximum(1, np.ceil(2*np.pi*(r0+r1)/2/size)).astype(np.int64)
    angle = np.mod(np.arctan2(p[:, 1], p[:, 0])+2*np.pi, 2*np.pi)
    sector = np.minimum(sectors-1, np.floor(angle/(2*np.pi)*sectors).astype(np.int64))
    return valid, np.column_stack([tier, ring, sector]), r0, r1, sectors

def build_grid(points, labels=None, config=GridConfig()):
    started = perf_counter()
    points = np.asarray(points, dtype=np.float64)
    valid, keys, r0, r1, sectors = addresses(points, config)
    predicted = geometry(points) if labels is None else np.asarray(labels)
    if predicted.shape != (len(points),):
        raise ValueError('One label is required per point')
    if not np.isin(predicted, [0, 1, 2, 3]).all():
        raise ValueError('Grid labels must be integer class IDs 0..3')
    p, predicted = points[valid], predicted[valid].astype(int)
    unique, first, inverse, counts = np.unique(keys, axis=0, return_index=True, return_inverse=True, return_counts=True)
    n = len(unique)
    minimum, maximum = np.full(n, np.inf), np.full(n, -np.inf)
    np.minimum.at(minimum, inverse, p[:, 2]); np.maximum.at(maximum, inverse, p[:, 2])
    total = np.bincount(inverse, weights=p[:, 2], minlength=n)
    mean = total / counts if n else total
    variance = np.bincount(inverse, weights=(p[:, 2]-mean[inverse])**2, minlength=n) / counts if n else total
    votes = np.zeros((n, 4), dtype=np.int64)
    np.add.at(votes, (inverse, predicted), 1)
    dominant = votes.argmax(axis=1) if n else np.array([], dtype=int)
    a0 = unique[:, 2]/sectors[first]*2*np.pi
    a1 = (unique[:, 2]+1)/sectors[first]*2*np.pi
    rm, am = (r0[first]+r1[first])/2, (a0+a1)/2
    cells = []
    for i, key in enumerate(unique):
        cells.append(dict(key=':'.join(map(str, key)), tier=int(key[0]), x=float(rm[i]*np.cos(am[i])), y=float(rm[i]*np.sin(am[i])), r0=float(r0[first[i]]), r1=float(r1[first[i]]), a0=float(a0[i]), a1=float(a1[i]), min=float(minimum[i]), max=float(maximum[i]), mean=float(mean[i]), roughness=float(np.sqrt(variance[i])), count=int(counts[i]), label=int(dominant[i]), confidence=float(votes[i, dominant[i]]/counts[i]), votes=votes[i].tolist()))
    uniform = len(np.unique(np.floor(p[:, :2]/config.near_size).astype(np.int64), axis=0))
    elapsed = (perf_counter()-started)*1000
    return {'config': asdict(config), 'cells': cells, 'metrics': {'input_points': len(points), 'accepted': int(valid.sum()), 'dropped': int((~valid).sum()), 'assigned': int(counts.sum()), 'cells': n, 'grid_and_comparison_ms': elapsed, 'estimated_packed_bytes': n*64, 'uniform_occupied_packed_bytes': uniform*64, 'memory_saving_percent': 100*(1-n/uniform) if uniform else 0.}}

def evaluate(truth, pred, points, config=GridConfig()):
    truth, pred, points = np.asarray(truth), np.asarray(pred), np.asarray(points)
    if truth.shape != pred.shape or truth.shape != (len(points),):
        raise ValueError('Labels must match point count')
    radius = np.hypot(points[:, 0], points[:, 1])
    valid = np.isin(truth, range(4)) & np.isin(pred, range(4)) & np.isfinite(points[:, :3]).all(axis=1) & (radius <= config.far)
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (truth[valid].astype(int), pred[valid].astype(int)), 1)
    union = matrix.sum(0)+matrix.sum(1)-matrix.diagonal()
    iou = [float(matrix[i, i]/u) if u else None for i, u in enumerate(union)]
    existing = [v for v in iou if v is not None]
    buckets = []
    for i, (lo, hi) in enumerate(zip([0, config.near, config.mid], [config.near, config.mid, config.far])):
        mask = valid & (radius >= lo) & ((radius <= hi) if i == 2 else (radius < hi))
        buckets.append({'range': [lo, hi], 'count': int(mask.sum()), 'accuracy': float((truth[mask] == pred[mask]).mean()) if mask.any() else None})
    return {'confusion': matrix.tolist(), 'iou': iou, 'miou': sum(existing)/len(existing) if existing else None, 'distance_buckets': buckets}
