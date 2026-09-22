from foveamap.model import infer
from foveamap.train import discover_sequences, load_bin

sequences = discover_sequences(
    r"C:\Users\anubh\Downloads\Rellis_3D_os1_cloud_node_kitti_bin\Rellis-3D",
    r"C:\Users\anubh\Downloads\Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614\Rellis-3D",
    "os1_cloud_node_kitti_bin", "os1_cloud_node_semantickitti_label_id",
)
bin_path, _ = sequences['00004'][0]
points = load_bin(bin_path)

labels, meta = infer(points, 'checkpoints/best.pt', device='cpu')
print('points:', len(points))
print('labels shape:', labels.shape)
print('unique labels:', set(labels.tolist()))
print('meta:', meta)