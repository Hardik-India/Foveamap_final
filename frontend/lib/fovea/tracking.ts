import type { Point } from './engine.ts';

export type MotionStatus = 'moving' | 'stationary' | 'unknown';
export type TrackObject = {
  trackId: number;
  centroid: [number, number, number];
  min: [number, number, number];
  max: [number, number, number];
  dimensions: [number, number, number];
  points: number;
  velocity: [number, number, number] | null;
  speedMps: number | null;
  motionStatus: MotionStatus;
  motionFrame: 'world-compensated' | 'stationary-sensor-assumed' | 'sensor-relative-uncompensated';
  trackHistory: [number, number, number][];
  source: 'geometric-estimate' | 'ground-truth-annotation' | 'model-prediction';
  className?: string;
  trackingStatus: string;
};

export type TrackingConfig = { groundGrid: number; groundMargin: number; clusterSize: number; minClusterPoints: number; gateDistance: number; maxMissed: number; movingEnterMps: number; movingExitMps: number; stationarySensor: boolean };
export const TRACKING_DEFAULTS: TrackingConfig = { groundGrid: 1, groundMargin: .35, clusterSize: .9, minClusterPoints: 8, gateDistance: 3, maxMissed: 2, movingEnterMps: 1.2, movingExitMps: .7, stationarySensor: false };

type State = { id: number; centroid: [number, number, number]; velocity: [number, number, number]; timestamp: number; missed: number; history: [number, number, number][]; motionStatus: MotionStatus; object: TrackObject };
const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const add = (a: number[], b: number[]) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]] as [number, number, number];
const mul = (a: number[], s: number) => [a[0] * s, a[1] * s, a[2] * s] as [number, number, number];
const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]] as [number, number, number];

function separateGround(points: Point[], cfg: TrackingConfig) {
  const mins = new Map<string, number>();
  for (const p of points) { const k = `${Math.floor(p.x / cfg.groundGrid)}:${Math.floor(p.y / cfg.groundGrid)}`; mins.set(k, Math.min(mins.get(k) ?? p.z, p.z)); }
  return points.map(p => p.z <= (mins.get(`${Math.floor(p.x / cfg.groundGrid)}:${Math.floor(p.y / cfg.groundGrid)}`) ?? p.z) + cfg.groundMargin);
}

const TERRAIN_LABELS = new Set([
  0, 1, 3, 6, 7, 10, 23, 29, 30, 31, 32, 33, 34,
]);

const DYNAMIC_SEMANTIC_LABELS = new Set([8, 17]);

function classNameFromGeometry(dimensions: [number, number, number]): string {
  const [width, depth, height] = dimensions;
  const length = Math.max(width, depth);
  const thickness = Math.min(width, depth);

  if (height > 1.2 && height < 2.4 && length < 1.2) {
    return 'person';
  }

  if (height > 1.1 && height < 3.2 && length > 2.0 && length < 7.5) {
    return 'vehicle';
  }

  if (height > 1.0 && length > 3.0 && thickness < 0.8) {
    return 'wall/fence';
  }

  if (height > 2.0 && thickness < 1.0) {
    return 'pole/tree';
  }

  if (height < 1.2 && length > 1.0) {
    return 'barrier/object';
  }

  return 'unclassified obstacle';
}

export function extractObjects(points: Point[], semantic?: number[], names?: Record<number, string>, cfg: TrackingConfig = TRACKING_DEFAULTS): TrackObject[] {
  const ground = separateGround(points, cfg), idx: number[] = [];
  for (let i = 0; i < points.length; i++) {
    const terrain = semantic && TERRAIN_LABELS.has(semantic[i]);
    if (!ground[i] && !terrain) idx.push(i);
  }
  const bins = new Map<string, number[]>();
  for (const i of idx) { const p = points[i]; const key = `${Math.floor(p.x / cfg.clusterSize)}:${Math.floor(p.y / cfg.clusterSize)}:${Math.floor(p.z / cfg.clusterSize)}`; bins.set(key, [...(bins.get(key) || []), i]); }
  const out: TrackObject[] = [];
  while (bins.size && out.length < 80) {
    const first = bins.keys().next().value as string, queue = [first], ids = bins.get(first)!; bins.delete(first);
    while (queue.length) { const [x, y, z] = queue.shift()!.split(':').map(Number); for (let dx = -1; dx <= 1; dx++)for (let dy = -1; dy <= 1; dy++)for (let dz = -1; dz <= 1; dz++) { const k = `${x + dx}:${y + dy}:${z + dz}`, v = bins.get(k); if (v) { ids.push(...v); bins.delete(k); queue.push(k); } } }
    if (ids.length < cfg.minClusterPoints) continue;
    const ps = ids.map(i => points[i]);
    const xs = ps.map(p => p.x), ys = ps.map(p => p.y), zs = ps.map(p => p.z);
    const min = [Math.min(...xs), Math.min(...ys), Math.min(...zs)] as [number, number, number];
    const max = [Math.max(...xs), Math.max(...ys), Math.max(...zs)] as [number, number, number];
    const centroid = [xs.reduce((a, b) => a + b, 0) / ps.length, ys.reduce((a, b) => a + b, 0) / ps.length, zs.reduce((a, b) => a + b, 0) / ps.length] as [number, number, number];
    const obj: TrackObject = { trackId: 0, centroid, min, max, dimensions: sub(max, min), points: ps.length, velocity: null, speedMps: null, motionStatus: 'unknown', motionFrame: cfg.stationarySensor ? 'stationary-sensor-assumed' : 'sensor-relative-uncompensated', trackHistory: [centroid], source: semantic ? 'ground-truth-annotation' : 'geometric-estimate', trackingStatus: 'detected' };
    if (semantic) {
      const counts = new Map<number, number>();

      for (const i of ids) {
        counts.set(semantic[i], (counts.get(semantic[i]) || 0) + 1);
      }

      const best = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];

      if (best !== undefined) {
        const baseName = names?.[best] || `rellis-${best}`;
        obj.className = DYNAMIC_SEMANTIC_LABELS.has(best)
          ? `dynamic ${baseName}`
          : baseName;
      }
    } else {
      obj.className = classNameFromGeometry(obj.dimensions);
    }
    out.push(obj);
  }
  return out.sort((a, b) => b.points - a.points);
}

export class LocalTracker {
  private tracks = new Map<number, State>();
  private nextId = 1;
  private lastFrame: number | null = null;
  reset() { this.tracks.clear(); this.nextId = 1; this.lastFrame = null; }
  update(objects: TrackObject[], timestamp: number, frameIndex: number, cfg: TrackingConfig = TRACKING_DEFAULTS, jumped = false) {
    if (jumped || this.lastFrame !== null && frameIndex !== this.lastFrame + 1) this.reset();
    this.lastFrame = frameIndex;
    const speedValid = cfg.stationarySensor, mode: TrackObject['motionFrame'] = speedValid ? 'stationary-sensor-assumed' : 'sensor-relative-uncompensated';
    const pairs: { d: number; oi: number; id: number }[] = [];
    for (let oi = 0; oi < objects.length; oi++)for (const [id, t] of this.tracks) { const dt = Math.max(0, timestamp - t.timestamp), pred = add(t.centroid, mul(t.velocity, dt)), d = dist(objects[oi].centroid, pred); if (d <= cfg.gateDistance) pairs.push({ d, oi, id }); }
    pairs.sort((a, b) => a.d - b.d); const usedO = new Set<number>(), usedT = new Set<number>(), out: TrackObject[] = [];
    for (const p of pairs) { if (usedO.has(p.oi) || usedT.has(p.id)) continue; const prev = this.tracks.get(p.id)!, obj = objects[p.oi], dt = Math.max(1e-6, timestamp - prev.timestamp), vel = mul(sub(obj.centroid, prev.centroid), 1 / dt), speed = Math.hypot(...vel); let status: MotionStatus = 'unknown'; if (speedValid) status = prev.motionStatus === 'moving' ? (speed < cfg.movingExitMps ? 'stationary' : 'moving') : (speed > cfg.movingEnterMps ? 'moving' : 'stationary'); const history = [...prev.history.slice(-19), obj.centroid]; const tracked = { ...obj, trackId: p.id, velocity: speedValid ? vel : null, speedMps: speedValid ? speed : null, motionStatus: status, motionFrame: mode, trackHistory: history, trackingStatus: 'active' }; this.tracks.set(p.id, { id: p.id, centroid: obj.centroid, velocity: vel, timestamp, missed: 0, history, motionStatus: status, object: tracked }); out.push(tracked); usedO.add(p.oi); usedT.add(p.id); }
    for (let i = 0; i < objects.length; i++)if (!usedO.has(i)) { const id = this.nextId++, obj = { ...objects[i], trackId: id, motionFrame: mode, trackingStatus: 'active' }; this.tracks.set(id, { id, centroid: obj.centroid, velocity: [0, 0, 0], timestamp, missed: 0, history: [obj.centroid], motionStatus: 'unknown', object: obj }); out.push(obj); }
    for (const [id, t] of [...this.tracks]) if (!usedT.has(id) && !out.some(o => o.trackId === id)) { t.missed++; if (t.missed > cfg.maxMissed) this.tracks.delete(id); }
    return out.sort((a, b) => a.trackId - b.trackId);
  }
}
