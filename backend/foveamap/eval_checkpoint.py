import torch
import numpy as np
from pathlib import Path
from foveamap.model import load_model
from foveamap.range_image import range_image
from foveamap.grid import geometry, evaluate
from foveamap.train import discover_sequences, load_bin, load_rellis_as_4class

model, projection = load_model('checkpoints/best.pt', device='cpu')

sequences = discover_sequences(
    r"C:\Users\anubh\Downloads\Rellis_3D_os1_cloud_node_kitti_bin\Rellis-3D",
    r"C:\Users\anubh\Downloads\Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614\Rellis-3D",
    "os1_cloud_node_kitti_bin", "os1_cloud_node_semantickitti_label_id",
)
val_pairs = sequences['00004'][:200]  # sample 200 scans for a quick check, not the full 2059

all_truth, all_pred = [], []
for bin_path, label_path in val_pairs:
    p = load_bin(bin_path)
    y = load_rellis_as_4class(label_path, len(p))
    features, _, row, col, valid = range_image(p, **projection)
    with torch.inference_mode():
        labels = model(torch.from_numpy(features[None])).argmax(1)[0].numpy()
    pred = geometry(p)
    pred[valid] = labels[row[valid], col[valid]]
    mask = y >= 0
    all_truth.append(y[mask])
    all_pred.append(pred[mask])

truth = np.concatenate(all_truth)
pred = np.concatenate(all_pred)
dummy_points = np.zeros((len(truth), 3))  # evaluate() needs points for distance buckets; radius unused if we just want mIoU
result = evaluate(truth, pred, dummy_points)
print('Per-class IoU:', result['iou'])
print('mIoU:', result['miou'])
print('Confusion matrix:')
print(np.array(result['confusion']))