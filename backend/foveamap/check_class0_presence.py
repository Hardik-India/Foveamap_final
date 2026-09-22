# check_class0_presence.py
import numpy as np
from pathlib import Path
from foveamap.train import discover_sequences, load_rellis_as_4class

sequences = discover_sequences(
    r"C:\Users\anubh\Downloads\Rellis_3D_os1_cloud_node_kitti_bin\Rellis-3D",
    r"C:\Users\anubh\Downloads\Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614\Rellis-3D",
    "os1_cloud_node_kitti_bin", "os1_cloud_node_semantickitti_label_id",
)

for seq in ['00000', '00001', '00002', '00003', '00004']:
    counts = np.zeros(4, dtype=np.int64)
    for _, label_path in sequences[seq][::20]:  # sample every 20th scan
        n = Path(label_path).stat().st_size // 4
        y = load_rellis_as_4class(label_path, n)
        valid = y >= 0
        counts += np.bincount(y[valid], minlength=4)
    print(f'{seq}: drivable={counts[0]}, non-drivable={counts[1]}, static={counts[2]}, dynamic={counts[3]}')