from pathlib import Path
import csv
import re
import numpy as np

# Four coarse classes used by the legacy demo pipeline.
SEMANTIC_MAP = {40:0, 60:0, 44:1, 48:1, 49:1, 72:1, 10:2, 11:2, 13:2, 15:2, 16:2, 18:2, 20:2, 50:2, 51:2, 52:2, 70:2, 71:2, 80:2, 81:2, 99:2, 30:3, 31:3, 32:3, 252:3, 253:3, 254:3, 255:3, 256:3, 257:3, 258:3, 259:3}

# RELLIS-3D uses its own ontology. Keep names separate from SemanticKITTI and
# from motion state. Unknown IDs are still displayed as named raw labels.
RELLIS_ONTOLOGY = {
    0: "void", 1: "dirt", 3: "grass", 4: "tree", 5: "pole", 6: "water",
    7: "sky", 8: "vehicle", 9: "object", 10: "asphalt", 12: "building",
    15: "log", 17: "person", 18: "fence", 19: "bush", 23: "concrete",
    27: "barrier", 31: "puddle", 33: "mud", 34: "rubble",
}
_FRAME_RE = re.compile(r"(\d+)(?=\.[^.]+$)")

def load_scan(path):
    path = Path(path)
    if path.suffix.lower() == '.bin':
        if path.stat().st_size % 16:
            raise ValueError('KITTI scans require 16-byte float32 records')
        points = np.fromfile(path, dtype='<f4').reshape(-1, 4)
        labels = None
    elif path.suffix.lower() == '.csv':
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            if not {'x', 'y', 'z'} <= set(reader.fieldnames or []):
                raise ValueError('CSV requires x,y,z headers')
            rows = list(reader)
        points = np.array([[float(r['x']), float(r['y']), float(r['z']), float(r.get('intensity') or 0)] for r in rows], dtype=np.float32).reshape(-1, 4)
        labels = np.array([float(r['label']) for r in rows]) if rows and all(r.get('label') not in (None, '') for r in rows) else None
        if labels is not None and not np.isin(labels, [0, 1, 2, 3]).all():
            raise ValueError('CSV class labels must be integers 0..3')
        if labels is not None: labels = labels.astype(np.int64)
    else:
        raise ValueError('Supported scans: .csv and .bin')
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Scan must contain finite points')
    return points, labels

def load_kitti_bin_bytes(payload, max_points=250000):
    if len(payload) == 0 or len(payload) % 16:
        raise ValueError('KITTI scans require 16-byte float32 records')
    count = len(payload) // 16
    if count > max_points:
        raise ValueError(f'Scan exceeds {max_points} point request limit')
    points = np.frombuffer(payload, dtype='<f4').reshape(-1, 4).copy()
    if not np.isfinite(points).all():
        raise ValueError('Scan contains non-finite values')
    return points

def numeric_frame_key(path):
    name = Path(path).name
    match = _FRAME_RE.search(name)
    return (int(match.group(1)) if match else float('inf'), name)

def load_semantickitti(path, count):
    if Path(path).stat().st_size != count*4:
        raise ValueError('SemanticKITTI label count does not match scan')
    raw = np.fromfile(path, dtype='<u4') & 0xFFFF
    return np.array([SEMANTIC_MAP.get(int(v), -1) for v in raw], dtype=np.int64)

def load_rellis_labels(path, count):
    path = Path(path)
    if path.stat().st_size != count * 4:
        raise ValueError('RELLIS label count does not match scan point count')
    raw = np.fromfile(path, dtype='<u4')
    semantic = (raw & 0xFFFF).astype(np.int64)
    instance = (raw >> 16).astype(np.int64)
    names = [RELLIS_ONTOLOGY.get(int(v), f'rellis-{int(v)}') for v in semantic]
    return {'semantic': semantic, 'instance': instance, 'names': names, 'source': 'rellis-ground-truth'}

def pair_rellis_label(scan_path, roots):
    scan = Path(scan_path)
    stem = scan.stem
    sequence = next((part for part in scan.parts if part.isdigit() and len(part) == 5), None)
    candidates = []
    for root in roots:
        root = Path(root)
        if sequence:
            candidates.extend(root.glob(f'**/{sequence}/**/{stem}.label'))
        candidates.extend(root.glob(f'**/{stem}.label'))
    found = sorted(set(candidates))
    if len(found) > 1:
        raise ValueError(f'Ambiguous labels for {scan.name}: {found[:3]}')
    return found[0] if found else None

def load_poses_json(path):
    data = __import__('json').loads(Path(path).read_text())
    if data.get('coordinate_frame') not in ('world_from_lidar', 'map_from_lidar'):
        raise ValueError('poses coordinate_frame must be world_from_lidar or map_from_lidar')
    frames = data.get('frames')
    if not isinstance(frames, list):
        raise ValueError('poses JSON requires a frames array')
    out = {}
    for item in frames:
        stem = str(item.get('frame', ''))
        mat = np.asarray(item.get('world_from_lidar') or item.get('map_from_lidar'), dtype=float)
        if mat.shape != (4, 4) or not np.isfinite(mat).all():
            raise ValueError(f'pose for {stem} must be a finite 4x4 matrix')
        out[stem] = mat
    return out

def write_csv(path, points, labels):
    with Path(path).open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['x','y','z','intensity','label'])
        for p, label in zip(points, labels): writer.writerow([*p[:4], int(label)])
