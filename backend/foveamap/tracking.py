from collections import deque
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class TrackingConfig:
    ground_grid: float = 1.0
    ground_margin: float = 0.35
    cluster_size: float = 0.9
    min_cluster_points: int = 8
    max_cluster_points: int = 20000
    max_detections: int = 80
    gate_distance: float = 3.0
    max_missed: int = 2
    moving_enter_mps: float = 1.2
    moving_exit_mps: float = 0.7
    stationary_sensor: bool = False


def _as_points(points):
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError('points must be an Nx3 or Nx4 array')
    if not np.isfinite(arr[:, :3]).all():
        raise ValueError('points contain non-finite coordinates')
    return arr[:, :3]


def separate_ground(points, config=TrackingConfig()):
    """Local-min height filter for uneven outdoor terrain.

    The lowest point in each XY grid cell is treated as local terrain. Points
    above that local minimum by ground_margin are candidate obstacles.
    """
    p = _as_points(points)
    if len(p) == 0:
        return np.zeros(0, dtype=bool)
    keys = np.floor(p[:, :2] / config.ground_grid).astype(int)
    mins = {}
    for key, z in zip(map(tuple, keys), p[:, 2]):
        mins[key] = min(mins.get(key, z), z)
    ground = np.array([z <= mins[tuple(key)] + config.ground_margin for key, z in zip(keys, p[:, 2])])
    return ground


def extract_objects(points, labels=None, semantic_names=None, config=TrackingConfig()):
    p = _as_points(points)
    if len(p) == 0:
        return []
    ground = separate_ground(p, config)
    candidate_idx = np.flatnonzero(~ground)
    if labels is not None:
        labels = np.asarray(labels)
        if len(labels) == len(p):
            # Ignore terrain-like labels when annotations are present, but do not
            # infer motion from class names.
            terrain = np.isin(labels, [0, 1, 3, 10, 23, 31, 33])
            candidate_idx = np.flatnonzero(~ground & ~terrain)
    if not len(candidate_idx):
        return []
    q = p[candidate_idx]
    bins = {}
    for local_i, key in enumerate(np.floor(q / config.cluster_size).astype(int)):
        bins.setdefault(tuple(key), []).append(local_i)
    objects = []
    while bins and len(objects) < config.max_detections:
        key = next(iter(bins))
        queue = deque([key])
        ids = bins.pop(key)
        while queue:
            base = queue.popleft()
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nb = (base[0] + dx, base[1] + dy, base[2] + dz)
                        if nb in bins:
                            ids.extend(bins.pop(nb))
                            queue.append(nb)
        if not config.min_cluster_points <= len(ids) <= config.max_cluster_points:
            continue
        original = candidate_idx[ids]
        cluster = p[original]
        mn, mx = cluster.min(0), cluster.max(0)
        obj = {
            'centroid': cluster.mean(0).tolist(),
            'min': mn.tolist(),
            'max': mx.tolist(),
            'dimensions': (mx - mn).tolist(),
            'points': int(len(cluster)),
            'source': 'geometric-estimate',
            'label': None,
        }
        if labels is not None and len(labels) == len(p):
            vals, counts = np.unique(labels[original], return_counts=True)
            best = int(vals[np.argmax(counts)])
            obj['label'] = best
            if semantic_names:
                obj['className'] = semantic_names.get(best, f'rellis-{best}')
                obj['source'] = 'ground-truth-annotation'
        objects.append(obj)
    objects.sort(key=lambda x: (-x['points'], x['centroid'][0], x['centroid'][1]))
    return objects


def clusters(points, labels=None, size=.8, minimum=8):
    """Compatibility wrapper used by the original CLI/tests."""
    cfg = TrackingConfig(cluster_size=size, min_cluster_points=minimum, ground_margin=-math.inf)
    if labels is not None:
        labels = np.asarray(labels)
        p = np.asarray(points)[labels == 3, :3] if len(labels) == len(points) else np.asarray(points)[:, :3]
    else:
        p = np.asarray(points)[:, :3]
    return extract_objects(p, config=cfg)


class Tracker:
    """Pose-aware gated one-to-one centroid tracker.

    With valid world_from_lidar poses, matching and velocity are computed in a
    common world frame. Without poses, velocity is reported as sensor-relative
    and uncompensated unless stationary_sensor=True.
    """
    def __init__(self, max_distance=3.0, ttl=1.0, config=None):
        self.config = config or TrackingConfig(gate_distance=max_distance, max_missed=max(1, int(math.ceil(ttl))))
        self.next_id = 1
        self.tracks = {}
        self.last_frame = None

    def reset(self):
        self.next_id = 1
        self.tracks = {}
        self.last_frame = None

    def _frame_point(self, centroid, pose):
        c = np.asarray([*centroid, 1.0], dtype=float)
        return (pose @ c)[:3] if pose is not None else c[:3]

    def update(self, objects, timestamp, frame_index=None, pose=None, reset=False):
        if reset or (frame_index is not None and self.last_frame is not None and frame_index != self.last_frame + 1):
            self.reset()
        self.last_frame = frame_index
        if pose is not None:
            pose = np.asarray(pose, dtype=float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError('pose must be a finite 4x4 world_from_lidar matrix')
        if not self.config.stationary_sensor and pose is None:
            frame_mode = 'sensor-relative-uncompensated'
            speed_valid = False
        elif pose is None:
            frame_mode = 'stationary-sensor-assumed'
            speed_valid = True
        else:
            frame_mode = 'world-compensated'
            speed_valid = True

        detections = []
        for obj in objects:
            world = self._frame_point(obj['centroid'], pose)
            detections.append((obj, world))

        pairs = []
        for di, (_, centre) in enumerate(detections):
            for tid, tr in self.tracks.items():
                dt = max(0.0, float(timestamp) - tr['timestamp'])
                pred = tr['world'] + tr['velocity'] * dt
                d = float(np.linalg.norm(centre - pred))
                if d <= self.config.gate_distance:
                    pairs.append((d, di, tid))
        pairs.sort()
        used_d, used_t, outputs = set(), set(), []
        for _, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            obj, centre = detections[di]
            tr = self.tracks[tid]
            dt = max(1e-6, float(timestamp) - tr['timestamp'])
            velocity = (centre - tr['world']) / dt
            status = tr.get('motionStatus', 'unknown')
            speed = float(np.linalg.norm(velocity))
            if speed_valid:
                if status == 'moving':
                    status = 'stationary' if speed < self.config.moving_exit_mps else 'moving'
                else:
                    status = 'moving' if speed > self.config.moving_enter_mps else 'stationary'
            else:
                status = 'unknown'
            history = [*tr['history'][-19:], obj['centroid']]
            state = {'id': tid, 'world': centre, 'velocity': velocity, 'timestamp': float(timestamp), 'missed': 0, 'history': history, 'motionStatus': status}
            self.tracks[tid] = state
            outputs.append(self._decorate(obj, state, frame_mode, speed_valid))
            used_d.add(di); used_t.add(tid)

        for di, (obj, centre) in enumerate(detections):
            if di in used_d:
                continue
            tid = self.next_id; self.next_id += 1
            state = {'id': tid, 'world': centre, 'velocity': np.zeros(3), 'timestamp': float(timestamp), 'missed': 0, 'history': [obj['centroid']], 'motionStatus': 'unknown'}
            self.tracks[tid] = state
            outputs.append(self._decorate(obj, state, frame_mode, speed_valid))

        for tid in list(self.tracks):
            if tid not in used_t and all(o['trackId'] != tid for o in outputs):
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > self.config.max_missed:
                    del self.tracks[tid]
        outputs.sort(key=lambda x: x['trackId'])
        return outputs

    def _decorate(self, obj, state, frame_mode, speed_valid):
        v = state['velocity']
        speed = float(np.linalg.norm(v)) if speed_valid else None
        return {
            **obj,
            'trackId': state['id'],
            'velocity': v.tolist() if speed_valid else None,
            'speedMps': speed,
            'motionStatus': state['motionStatus'] if speed_valid else 'unknown',
            'motionFrame': frame_mode,
            'trackHistory': state['history'],
            'trackingStatus': 'active',
        }
