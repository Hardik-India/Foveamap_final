import type { LoadedFrame } from "./sequence.ts";
import type { TrackObject, TrackingConfig } from "./tracking.ts";

export type BackendHealth = {
  status: "ok";
  capabilities: Record<string, unknown>;
};
export type TrackResponse = {
  sessionId:string;
  objects:TrackObject[];
  semanticSource:string;
  limits:Record<string,number>;
  modelId?:string;
  checkpointHash?:string;
};
const DEFAULT_URL = "http://127.0.0.1:8000";

export function backendUrl() {
  return (
    (
      globalThis as typeof globalThis & {
        process?: { env?: Record<string, string | undefined> };
      }
    ).process?.env?.NEXT_PUBLIC_FOVEAMAP_BACKEND_URL || DEFAULT_URL
  );
}

async function fileToBase64(file: File) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000)
    s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

export async function health(signal?: AbortSignal): Promise<BackendHealth> {
  const res = await fetch(`${backendUrl()}/health`, { signal });
  if (!res.ok) throw Error(`Backend health failed: ${res.status}`);
  return res.json();
}

export async function trackFrame(
  frame: LoadedFrame,
  opts: {
    sessionId?: string;
    timestamp: number;
    frameIndex: number;
    jumped: boolean;
    config: TrackingConfig;
    signal?: AbortSignal;
  },
): Promise<TrackResponse> {
  const body: Record<string, unknown> = {
    sessionId: opts.sessionId,
    frameIndex: opts.frameIndex,
    timestamp: opts.timestamp,
    jumped: opts.jumped,
    stationarySensorAssumption: opts.config.stationarySensor,
    groundGrid: opts.config.groundGrid,
    groundMargin: opts.config.groundMargin,
    clusterSize: opts.config.clusterSize,
    minClusterPoints: opts.config.minClusterPoints,
    gateDistance: opts.config.gateDistance,
    maxMissed: opts.config.maxMissed,
    movingEnterMps: opts.config.movingEnterMps,
    movingExitMps: opts.config.movingExitMps,
    binBase64: await fileToBase64(frame.file),
  };
  if (frame.labelFile)
    body.rellisLabelBase64 = await fileToBase64(frame.labelFile);
  const res = await fetch(`${backendUrl()}/track`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal: opts.signal,
  });
  const json = (await res
    .json()
    .catch(() => ({}))) as Partial<TrackResponse> & {
    error?: { message?: string };
  };
  if (!res.ok)
    throw Error(
      json.error?.message || `Backend tracking failed: ${res.status}`,
    );
  return json as TrackResponse;
}
