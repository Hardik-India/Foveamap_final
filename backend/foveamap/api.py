from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import argparse
import base64
import binascii
import hashlib
import json
import os
import threading
import time
import uuid
import numpy as np
import torch
from pathlib import Path

from .model import RangeNet, load_model
from .range_image import range_image
from .io import load_kitti_bin_bytes, load_rellis_labels, RELLIS_ONTOLOGY
from .tracking import TrackingConfig, Tracker, extract_objects

MAX_BODY = 16 * 1024 * 1024
MAX_POINTS = 250000
_sessions = {}
_lock = threading.Lock()

_model = None
_projection = None
_class_names = None
_checkpoint_path = None
_checkpoint_hash = None
_device = 'cpu'
MODEL_ID = 'foveamap-range-v1'

# Names for the checkpoint's class indices (0..N-1) -> your training CLASS_NAMES list,
# not the raw RELLIS IDs. Falls back to numbered names if the checkpoint didn't store them.
def _model_class_names():
    if _class_names:
        return {i: name for i, name in enumerate(_class_names)}
    return None


def _load_checkpoint(path, device='cpu'):
    global _model, _projection, _class_names, _checkpoint_hash
    _model, _projection, _class_names = load_model(path, device)
    _checkpoint_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _run_model(points):
    features, _, row, col, valid = range_image(points, **_projection)
    with torch.inference_mode():
        labels = _model(torch.from_numpy(features[None]).to(_device)).argmax(1)[0].cpu().numpy()
    out = np.full(len(points), -1, dtype=np.int64)
    out[valid] = labels[row[valid], col[valid]]
    return out, {'neural_points': int(valid.sum()), 'unlabeled_points': int((~valid).sum())}

def _cors(origin='*'):
    return {
        'Access-Control-Allow-Origin': origin,
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'content-type',
        'Access-Control-Max-Age': '600',
    }


def _json(handler, status, payload):
    body = json.dumps(payload).encode('utf-8')
    handler.send_response(status)
    for k, v in _cors().items():
        handler.send_header(k, v)
    handler.send_header('content-type', 'application/json')
    handler.send_header('content-length', str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler, status, message, code='bad_request'):
    _json(handler, status, {'error': {'code': code, 'message': message}})


def _decode_scan(req):
    if 'points' in req:
        arr = np.asarray(req['points'], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError('points must be an array of [x,y,z,intensity?]')
        if len(arr) > MAX_POINTS:
            raise ValueError(f'maximum {MAX_POINTS} points per request')
        if arr.shape[1] == 3:
            arr = np.column_stack([arr, np.zeros(len(arr), dtype=np.float32)])
        if not np.isfinite(arr[:, :4]).all():
            raise ValueError('points contain non-finite values')
        return arr[:, :4]
    encoded = req.get('binBase64')
    if not encoded:
        raise ValueError('request requires points or binBase64')
    return load_kitti_bin_bytes(base64.b64decode(encoded), MAX_POINTS)


def _pose(req):
    value = req.get('worldFromLidar')
    if value is None:
        return None
    mat = np.asarray(value, dtype=float)
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        raise ValueError('worldFromLidar must be a finite 4x4 matrix')
    return mat


def _session(session_id, stationary):
    with _lock:
        sid = session_id or str(uuid.uuid4())
        state = _sessions.get(sid)
        if state is None:
            state = {'tracker': Tracker(config=TrackingConfig(stationary_sensor=stationary)), 'updated': time.time()}
            _sessions[sid] = state
        state['updated'] = time.time()
        return sid, state['tracker']


class Handler(BaseHTTPRequestHandler):
    server_version = 'FoveaMapAPI/1.0'

    def log_message(self, fmt, *args):
        if os.environ.get('FOVEAMAP_API_LOGS'):
            super().log_message(fmt, *args)

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in _cors().items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/health':
            _json(self, 200, {
                'status': 'ok',
                'capabilities': {
                    'tracking': True,
                    'rellisLabels': True,
                    'poses': True,
                    'trainedModelLoaded': _model is not None,
                    'modelClassCount': len(_class_names) if _class_names else None,
                    'maxPointsPerFrame': MAX_POINTS,
                    'maxBodyBytes': MAX_BODY,
                    'motionModes': ['world-compensated', 'stationary-sensor-assumed', 'sensor-relative-uncompensated'],
                },
            })
        else:
            _error(self, 404, 'unknown endpoint', 'not_found')

    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/session':
            sid, _ = _session(None, False)
            _json(self, 200, {'sessionId': sid})
            return
        if path != '/track':
            _error(self, 404, 'unknown endpoint', 'not_found')
            return
        length = int(self.headers.get('content-length') or '0')
        if length > MAX_BODY:
            _error(self, 413, f'request body exceeds {MAX_BODY} bytes', 'too_large')
            return
        try:
            req = json.loads(self.rfile.read(length).decode('utf-8'))
            stationary = bool(req.get('stationarySensorAssumption'))
            sid, tracker = _session(req.get('sessionId'), stationary)
            if req.get('reset'):
                tracker.reset()
            points = _decode_scan(req)
            semantic = None
            names = None
            semantic_source_override = None
            inference_meta = None

            label_payload = req.get('rellisLabelBase64')
            if label_payload:
                raw = base64.b64decode(label_payload)
                if len(raw) != len(points) * 4:
                    raise ValueError('RELLIS label count does not match scan point count')
                vals = np.frombuffer(raw, dtype='<u4')
                semantic = (vals & 0xFFFF).astype(np.int64)
                names = RELLIS_ONTOLOGY
            elif _model is not None:
                semantic, inference_meta = _run_model(points)
                names = _model_class_names()
                semantic_source_override = 'model-prediction'

            cfg = TrackingConfig(
                ground_grid=float(req.get('groundGrid', 1.0)),
                ground_margin=float(req.get('groundMargin', 0.35)),
                cluster_size=float(req.get('clusterSize', 0.9)),
                min_cluster_points=int(req.get('minClusterPoints', 8)),
                gate_distance=float(req.get('gateDistance', 3.0)),
                max_missed=int(req.get('maxMissed', 2)),
                moving_enter_mps=float(req.get('movingEnterMps', 1.2)),
                moving_exit_mps=float(req.get('movingExitMps', 0.7)),
                stationary_sensor=stationary,
            )
            tracker.config = cfg
            objects = extract_objects(points, semantic, names, cfg)
            timestamp = float(req.get('timestamp', 0.0))
            tracks = tracker.update(objects, timestamp, req.get('frameIndex'), _pose(req), bool(req.get('jumped')))
            response = {
                'sessionId': sid,
                'frameIndex': req.get('frameIndex'),
                'timestamp': timestamp,
                'objects': tracks,
                'limits': {'maxPointsPerFrame': MAX_POINTS, 'maxDetections': cfg.max_detections},
                'semanticSource': (
                    'rellis-ground-truth' if label_payload
                    else semantic_source_override or 'geometric-estimate'
                ),
            }
            if semantic_source_override == 'model-prediction':
                response['modelId'] = MODEL_ID
                response['checkpointHash'] = _checkpoint_hash
                response['inferenceMeta'] = inference_meta
            _json(self, 200, response)
        except (ValueError, json.JSONDecodeError, binascii.Error) as exc:
            _error(self, 400, str(exc))
        except Exception as exc:
            _error(self, 500, str(exc), 'internal_error')


def main(argv=None):
    parser = argparse.ArgumentParser(description='FoveaMap local tracking API')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    global _checkpoint_path, _device
    _device = args.device
    if args.checkpoint:
        _load_checkpoint(args.checkpoint, args.device)
        print(f'Loaded checkpoint {args.checkpoint} (hash {_checkpoint_hash}, {len(_class_names)} classes)')
    _checkpoint_path = args.checkpoint
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'FoveaMap API listening on http://{args.host}:{args.port}')
    server.serve_forever()

if __name__ == '__main__':
    main()