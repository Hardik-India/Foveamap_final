# FoveaMap - SIH 2026

FoveaMap is a dark LiDAR perception workspace with synthetic playback, single KITTI `.bin` / CSV uploads, RELLIS-style sequence playback, geometric object extraction, and persistent tracking. The included neural demo is synthetic-only; it is not a trained RELLIS-3D recogniser.

## Run on Windows PowerShell

Terminal 1 - backend API:

```powershell
cd C:\Users\Lenovo\OneDrive\Desktop\foveamap\backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m foveamap.api --host 127.0.0.1 --port 8000
```

Terminal 2 - frontend:

```powershell
cd C:\Users\Lenovo\OneDrive\Desktop\foveamap\frontend
corepack enable
corepack pnpm install --frozen-lockfile
corepack pnpm dev
```

Open [http://localhost:5173](http://localhost:5173). The frontend is a standard Next.js + React + TypeScript app. If `pnpm` is already on PATH, `pnpm install --frozen-lockfile` and `pnpm dev` are equivalent.

## Uploads

Use **Point cloud library** to load:

- A single `.csv` with `x,y,z,intensity,label` columns.
- A single KITTI-format `.bin` with little-endian float32 `x,y,z,intensity` records.
- A folder containing one or more RELLIS-style sequences such as `00000\os1_cloud_node_kitti_bin\000000.bin`.

Sequence frames are sorted numerically by filename. Different sequence folders are kept separate and selectable; they are not concatenated. Frames are loaded lazily through a small bounded browser cache with prefetching, so thousands of scans are not loaded into memory at once.

Optional RELLIS `.label` files are paired by sequence and frame stem. The loader validates point counts and displays RELLIS semantic names as ground-truth annotations. Semantic class is kept separate from motion state.

## Playback and Tracking

Controls include play/pause, previous/next, restart, timeline seek, playback speed, looping, capture frequency, current filename, and frame count. Capture frequency is used for timestamps when real timestamps are unavailable; playback speed only changes wall-clock playback and does not alter velocity calculations.

The backend and browser fallback run geometric tracking:

- Local terrain separation for uneven outdoor scenes.
- Above-ground obstacle clustering.
- Bounding boxes, centroids, persistent IDs, histories, and velocity estimates.
- Gated one-to-one matching with track expiry.
- Tracker reset on sequence switches, backward seeks, and non-consecutive jumps.

Without valid poses, real sequence motion is shown as sensor-relative/uncompensated or unknown. Enable the stationary-sensor assumption only when the sensor frame is actually fixed. Do not interpret semantic class names as moving/stationary labels.

## Poses

Future pose import should use a JSON file with validated world-from-LiDAR transforms:

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

Dataset-specific calibration must be applied before passing poses to FoveaMap. The code does not guess RELLIS coordinate conventions.

## Future Model Integration

The stable semantic adapter is in `frontend/lib/fovea/semanticAdapter.ts`. Predictions must preserve point order, use metres, use feature order `x,y,z,intensity`, and include metadata such as `modelId` and `checkpointHash` for model predictions. Incompatible or missing metadata is rejected by the adapter.

A future Colab-trained RELLIS model should be loaded behind the backend API and return per-point predictions with one label per input point. The current implementation supports imported annotations and geometric estimates; compatible trained-model inference is intentionally not claimed.

## Verification

Useful checks:

```powershell
cd C:\Users\Lenovo\OneDrive\Desktop\foveamap\backend
python -m unittest discover -s tests -v

cd C:\Users\Lenovo\OneDrive\Desktop\foveamap\frontend
node --experimental-strip-types --test tests/engine.test.mjs
.\node_modules\.bin\tsc.cmd --noEmit
.\node_modules\.bin\eslint.cmd . --ignore-pattern dist --ignore-pattern .next
.\node_modules\.bin\next.cmd build
```

Generated test fixtures are small synthetic scans. Passing tests do not establish accuracy on real RELLIS-3D data.
