# FoveaMap backend

## Windows PowerShell

```powershell
cd C:\Users\Lenovo\OneDrive\Desktop\foveamap\backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m foveamap.api --host 127.0.0.1 --port 8000
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

Optional PyTorch range-image CNN dependencies:

```powershell
python -m pip install -e ".[ml]"
```

ROS2 dependencies must be installed through your ROS distribution before running `python -m foveamap.ros2_node`.

## Local API

- `GET /health` returns capability and limit information.
- `POST /session` creates an isolated tracking session.
- `POST /track` accepts one frame and returns tracked objects.

`POST /track` request fields:

- `sessionId`: optional browser/session identifier.
- `frameIndex`, `timestamp`: frame ordering and capture time in seconds.
- `binBase64`: KITTI little-endian float32 `x,y,z,intensity` records.
- `rellisLabelBase64`: optional RELLIS `.label` payload, one uint32 per point.
- `worldFromLidar`: optional 4x4 transform for world-compensated tracking.
- `stationarySensorAssumption`: explicit fixed-sensor mode when no poses exist.
- Tracking parameters: `groundGrid`, `groundMargin`, `clusterSize`, `minClusterPoints`, `gateDistance`, `maxMissed`, `movingEnterMps`, `movingExitMps`.

The API validates request size, point count, binary shape, label count, and pose shape. It does not load trained model weights silently.

## Pose Schema

Pose files used by CLI/import tooling must be explicit:

```json
{
  "coordinate_frame": "world_from_lidar",
  "frames": [
    {
      "frame": "000000",
      "timestamp": 0.0,
      "world_from_lidar": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }
  ]
}
```

Apply dataset-specific RELLIS calibration before creating these matrices. The backend refuses unknown pose conventions instead of guessing.

## Model Integration

Future Colab-trained checkpoints should be loaded behind the API after metadata validation. Expected preprocessing is metres, feature order `x,y,z,intensity`, and one prediction per input point. Checkpoint metadata should include architecture, label ontology, preprocessing version, and checkpoint hash.

Current capabilities are geometric extraction, optional RELLIS ground-truth label display, and tracking. Compatible trained RELLIS semantic prediction is not implemented yet.
