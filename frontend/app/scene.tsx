'use client';

import { useEffect, useRef, useState } from 'react';
import { LocateFixed, Minus, Plus } from 'lucide-react';
import { type Cell, type Config, type Point } from '@/lib/fovea/engine';
import type { TrackObject } from '@/lib/fovea/tracking';
import {
  createWebGpuRenderer,
  type WebGpuRenderer,
} from '@/lib/fovea/webgpuRenderer';

const motionColours = {
  moving: '#ff8a67',
  stationary: '#65dbb7',
  unknown: '#a6b4c8',
};

type HitTarget = {
  x: number;
  y: number;
  c?: Cell;
  o?: TrackObject;
};

type SceneProps = {
  points: Point[];
  cells: Cell[];
  config: Config;
  mode: string;
  visible: boolean[];
  top: boolean;
  labels: boolean;
  rings: boolean;
  objects?: TrackObject[];
  showBoxes?: boolean;
  showObjectLabels?: boolean;
  showTrajectories?: boolean;
  showMotion?: boolean;
  onSelectObject?: (object: TrackObject | null) => void;
};

function projectPoint(
  x: number,
  y: number,
  z: number,
  width: number,
  height: number,
  yaw: number,
  zoom: number,
  top: boolean,
) {
  const scale = Math.min(width / 145, height / 73) * zoom;
  const rotatedX = x * Math.cos(yaw) - y * Math.sin(yaw);
  const rotatedY = x * Math.sin(yaw) + y * Math.cos(yaw);

  return {
    x: width * 0.5 + rotatedX * scale,
    y: height * 0.58 + rotatedY * scale * (top ? 1 : 0.56) - z * scale * 0.95,
  };
}

export default function Scene({
  points,
  cells,
  config,
  mode,
  visible,
  top,
  labels,
  rings,
  objects = [],
  showBoxes = true,
  showObjectLabels = true,
  showTrajectories = true,
  showMotion = true,
  onSelectObject,
}: SceneProps) {
  const gpuCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<WebGpuRenderer | null>(null);
  const drag = useRef<{ x: number; y: number; move: boolean } | null>(null);
  const hit = useRef<HitTarget[]>([]);

  const [yaw, setYaw] = useState(-0.34);
  const [zoom, setZoom] = useState(1);
  const [selected, setSelected] = useState<Cell | null>(null);
  const [webGpuReady, setWebGpuReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const canvas = gpuCanvasRef.current;

    if (!canvas) return;

    createWebGpuRenderer(canvas).then((renderer) => {
      if (cancelled) {
        renderer?.dispose();
        return;
      }

      rendererRef.current = renderer;
      setWebGpuReady(Boolean(renderer));
    });

    return () => {
      cancelled = true;
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, []);

  useEffect(() => {
    const gpuCanvas = gpuCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;

    if (!gpuCanvas || !overlayCanvas) return;

    const draw = () => {
      const width = gpuCanvas.clientWidth;
      const height = gpuCanvas.clientHeight;
      const dpr = Math.min(devicePixelRatio, 2);

      gpuCanvas.width = width * dpr;
      gpuCanvas.height = height * dpr;
      overlayCanvas.width = width * dpr;
      overlayCanvas.height = height * dpr;

      const overlay = overlayCanvas.getContext('2d');
      if (!overlay) return;

      overlay.setTransform(dpr, 0, 0, dpr, 0, 0);
      overlay.clearRect(0, 0, width, height);

      const project = (x: number, y: number, z = 0) =>
        projectPoint(x, y, z, width, height, yaw, zoom, top);

      const line = (coords: number[][], colour: string, lineWidth = 1) => {
        overlay.strokeStyle = colour;
        overlay.lineWidth = lineWidth;
        overlay.beginPath();

        coords.forEach((coord, index) => {
          const p = project(coord[0], coord[1], coord[2] || 0);

          if (index) {
            overlay.lineTo(p.x, p.y);
          } else {
            overlay.moveTo(p.x, p.y);
          }
        });

        overlay.stroke();
      };

      const renderer = rendererRef.current;

      if (renderer) {
        renderer.render({
          points,
          cells,
          mode,
          labels,
          visible,
          config,
          yaw,
          zoom,
          top,
          width,
          height,
        });
      } else {
        overlay.fillStyle = '#0b1b23';
        overlay.fillRect(0, 0, width, height);
        overlay.fillStyle = '#d8f3ef';
        overlay.font = '13px Arial';
        overlay.fillText(
          'WebGPU unavailable. Use recent Chrome/Edge with hardware acceleration.',
          18,
          38,
        );
      }

      for (let x = -100; x <= 100; x += 10) {
        line([[x, -100], [x, 100]], '#16303a', 0.7);
        line([[-100, x], [100, x]], '#16303a', 0.7);
      }

      if (rings) {
        [config.near, config.mid, config.far].forEach((r, index) => {
          const coords = Array.from({ length: 129 }, (_, k) => [
            r * Math.cos((k / 128) * Math.PI * 2),
            r * Math.sin((k / 128) * Math.PI * 2),
          ]);

          overlay.setLineDash([4, 5]);
          line(coords, ['#4bc6ad60', '#779dcb65', '#cabc8060'][index]);
          overlay.setLineDash([]);

          const p = project(r * 0.707, r * 0.707);
          overlay.fillStyle = '#7fa2a9';
          overlay.font = '11px monospace';
          overlay.fillText(`${r} m`, p.x + 5, p.y);
        });
      }

      hit.current = [];

      if (mode !== 'points') {
        for (const cell of cells) {
          if (!visible[cell.label] || Math.hypot(cell.x, cell.y) > config.far) {
            continue;
          }

          const p = project(cell.x, cell.y, cell.mean);
          hit.current.push({ x: p.x, y: p.y, c: cell });
        }
      }

      if (showMotion) {
        const visibleObjects = objects.slice(0, showObjectLabels ? 8 : 80);

        for (const obj of visibleObjects) {
          const colour =
            motionColours[obj.motionStatus] || motionColours.unknown;
          const min = obj.min;
          const max = obj.max;
          const corners = [
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
            [max[0], max[1], min[2]],
            [min[0], max[1], min[2]],
            [min[0], min[1], max[2]],
            [max[0], min[1], max[2]],
            [max[0], max[1], max[2]],
            [min[0], max[1], max[2]],
          ];

          if (showTrajectories && obj.trackHistory.length > 1) {
            overlay.strokeStyle = colour;
            overlay.globalAlpha = 0.75;
            overlay.lineWidth = 1.4;
            overlay.beginPath();

            obj.trackHistory.forEach((point, index) => {
              const p = project(point[0], point[1], point[2]);

              if (index) {
                overlay.lineTo(p.x, p.y);
              } else {
                overlay.moveTo(p.x, p.y);
              }
            });

            overlay.stroke();
            overlay.globalAlpha = 1;
          }

          if (showBoxes) {
            const edges = [
              [0, 1],
              [1, 2],
              [2, 3],
              [3, 0],
              [4, 5],
              [5, 6],
              [6, 7],
              [7, 4],
              [0, 4],
              [1, 5],
              [2, 6],
              [3, 7],
            ];

            for (const [a, b] of edges) {
              line([corners[a], corners[b]], colour, 1.6);
            }
          }

          const centre = project(
            obj.centroid[0],
            obj.centroid[1],
            obj.centroid[2],
          );

          hit.current.push({ x: centre.x, y: centre.y, o: obj });
          overlay.fillStyle = colour;
          overlay.beginPath();
          overlay.arc(centre.x, centre.y, 4, 0, Math.PI * 2);
          overlay.fill();

          if (showObjectLabels) {
            const name = obj.className || `obstacle #${obj.trackId}`;
            const speed =
              obj.speedMps === null ? '' : ` · ${obj.speedMps.toFixed(1)} m/s`;
            const label = `#${obj.trackId} ${name}${speed}`;
            const textWidth = overlay.measureText(label).width;

            overlay.font = '12px Arial';
            overlay.fillStyle = '#0b1b23e6';
            overlay.fillRect(
              centre.x + 7,
              centre.y - 22,
              Math.min(textWidth + 10, 180),
              18,
            );
            overlay.fillStyle = colour;
            overlay.fillText(label, centre.x + 12, centre.y - 9);
          }
        }
      }

      const car = [
        [-2, -0.95, 0.2],
        [2, -0.95, 0.2],
        [2, 0.95, 0.2],
        [-2, 0.95, 0.2],
        [-2, -0.95, 0.2],
      ];

      line(car, '#e3fff3', 1.8);
      line(
        [
          [-1.4, -0.8, 1.3],
          [0.9, -0.8, 1.3],
          [1.7, 0, 1.3],
          [0.9, 0.8, 1.3],
          [-1.4, 0.8, 1.3],
          [-1.4, -0.8, 1.3],
        ],
        '#d9fff3',
        1.6,
      );

      const origin = project(0, 0);
      overlay.strokeStyle = '#88e8d38c';
      overlay.beginPath();
      overlay.arc(origin.x, origin.y, 11, 0, Math.PI * 2);
      overlay.stroke();

      overlay.fillStyle = '#b7e5d5';
      overlay.font = '10px monospace';
      overlay.fillText('EGO', origin.x - 11, origin.y + 25);

      if (selected) {
        const selectedPoint = project(selected.x, selected.y, selected.mean);
        overlay.strokeStyle = '#fff';
        overlay.strokeRect(
          selectedPoint.x - 5,
          selectedPoint.y - 5,
          10,
          10,
        );
      }

      const axisX = 62;
      const axisY = height - 55;
      overlay.lineWidth = 1.2;

      for (const [dx, dy, label, colour] of [
        [26, 0, 'X', '#bf8292'],
        [-12, -18, 'Y', '#77b49d'],
        [0, -32, 'Z', '#799db6'],
      ] as const) {
        overlay.strokeStyle = colour;
        overlay.beginPath();
        overlay.moveTo(axisX, axisY);
        overlay.lineTo(axisX + dx, axisY + dy);
        overlay.stroke();

        overlay.fillStyle = colour;
        overlay.fillText(label, axisX + dx + 3, axisY + dy);
      }

      overlay.fillStyle = '#7fa2a9';
      overlay.font = '11px monospace';
      overlay.fillText(
        webGpuReady ? 'WEBGPU' : 'CPU FALLBACK',
        width - 82,
        height - 18,
      );
    };

    draw();

    const resizeObserver = new ResizeObserver(draw);
    resizeObserver.observe(gpuCanvas);

    return () => resizeObserver.disconnect();
  }, [
    points,
    cells,
    config,
    mode,
    visible,
    top,
    labels,
    rings,
    yaw,
    zoom,
    selected,
    objects,
    showBoxes,
    showObjectLabels,
    showTrajectories,
    showMotion,
    webGpuReady,
  ]);

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    drag.current = {
      x: event.clientX,
      y: event.clientY,
      move: false,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drag.current) return;

    const dx = event.clientX - drag.current.x;

    if (Math.abs(dx) > 2) {
      drag.current.move = true;
    }

    setYaw((value) => value + dx * 0.006);
    drag.current.x = event.clientX;
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (drag.current && !drag.current.move) {
      const rect = event.currentTarget.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      let best: HitTarget | undefined;
      let distance = 100;

      for (const point of hit.current) {
        const d = (point.x - x) ** 2 + (point.y - y) ** 2;

        if (d < distance) {
          distance = d;
          best = point;
        }
      }

      setSelected(best?.c || null);
      onSelectObject?.(best?.o || null);
    }

    drag.current = null;
  };

  const resetView = () => {
    setZoom(1);
    setYaw(-0.34);
    setSelected(null);
    onSelectObject?.(null);
  };

  return (
    <div className="scene-wrap">
      <canvas ref={gpuCanvasRef} className="scene-gpu-canvas" />

      <canvas
        ref={overlayCanvasRef}
        className="scene-overlay-canvas"
        aria-label="Interactive LiDAR map. Drag to orbit, use zoom buttons, and select an object or cell."
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={() => {
          drag.current = null;
        }}
      />

      <div className="scene-label">
        <span className="dot" />{' '}
        {mode === 'points'
          ? 'Raw point cloud'
          : mode === 'elevation'
            ? 'Elevation map'
            : 'Semantic elevation map'}
      </div>

      <span className="scene-corner">
        {top ? 'TOP-DOWN' : 'PERSPECTIVE'} · SENSOR FRAME
      </span>

      {selected && (
        <div className="selected-cell">
          CELL {selected.key}
          <br />
          Height {selected.min.toFixed(2)} / {selected.mean.toFixed(2)} /{' '}
          {selected.max.toFixed(2)} m
          <br />
          {selected.count} points · roughness {selected.roughness.toFixed(3)} m
          <br />
          Label agreement {(selected.confidence * 100).toFixed(0)}%
        </div>
      )}

      <div className="scene-tools">
        <button
          title="Zoom in"
          aria-label="Zoom in"
          onClick={() => setZoom((value) => Math.min(4, value * 1.2))}
        >
          <Plus />
        </button>
        <button
          title="Zoom out"
          aria-label="Zoom out"
          onClick={() => setZoom((value) => Math.max(0.4, value / 1.2))}
        >
          <Minus />
        </button>
        <button title="Reset view" aria-label="Reset view" onClick={resetView}>
          <LocateFixed />
        </button>
      </div>

      <div className="scene-help">
        Drag to orbit <span> · </span> Click an object or grid cell
      </div>
    </div>
  );
}
