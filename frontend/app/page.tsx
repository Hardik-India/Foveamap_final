'use client';

/* eslint-disable react-hooks/set-state-in-effect */
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  ArrowDownToLine,
  Box,
  Database,
  FileText,
  Focus,
  Gauge,
  Layers3,
  Map,
  Pause,
  Play,
  Radar,
  RotateCcw,
  Settings2,
  SkipBack,
  SkipForward,
  SlidersHorizontal,
  Upload,
  Waypoints,
} from 'lucide-react';

import {
  SidebarProvider,
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarTrigger,
} from '@/components/ui/sidebar';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Toaster, toast } from 'sonner';

import {
  DEFAULT,
  COLOURS,
  simulate,
  project,
  parseCSV,
  parseBin,
  type Point,
  type Config,
} from '@/lib/fovea/engine';
import {
  FrameCache,
  attachLabelsToSequences,
  groupSequences,
  timestampFor,
  type LoadedFrame,
  type Sequence,
} from '@/lib/fovea/sequence';
import {
  LocalTracker,
  extractObjects,
  TRACKING_DEFAULTS,
  type TrackObject,
  type TrackingConfig,
} from '@/lib/fovea/tracking';
import { health, trackFrame } from '@/lib/fovea/api';
import Scene from './scene';

const nav = [
  ['workspace', 'Perception workspace', Map],
  ['metrics', 'Performance metrics', Gauge],
  ['data', 'Point cloud library', Database],
  ['guide', 'Project guide', FileText],
  ['settings', 'Pipeline settings', SlidersHorizontal],
] as const;

const scenes: Record<string, string> = {
  urban: 'Urban intersection',
  hazards: 'Road surface hazards',
  rugged: 'Rugged terrain',
};

const cache = new FrameCache(5);
const tracker = new LocalTracker();

const basename = (value: string) => value.replace(/^.*[\\/]/, '');

const download = (
  name: string,
  data: string,
  type = 'application/json',
) => {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const anchor = document.createElement('a');

  anchor.href = url;
  anchor.download = name;
  anchor.click();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

function Picker({
  value,
  onChange,
  items,
  label,
}: {
  value: string;
  onChange: (value: string) => void;
  items: Record<string, string>;
  label: string;
}) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="select-wide" aria-label={label}>
        <SelectValue />
      </SelectTrigger>

      <SelectContent>
        {Object.entries(items).map(([key, itemLabel]) => (
          <SelectItem value={key} key={key}>
            {itemLabel}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export default function Home() {
  const [mounted, setMounted] = useState(false);

  const [page, setPage] = useState('workspace');
  const [scene, setScene] = useState('urban');
  const [frame, setFrame] = useState(0);
  const [running, setRunning] = useState(true);
  const [mode, setMode] = useState('semantic');
  const [top, setTop] = useState(false);
  const [rings] = useState(true);
  const [visible, setVisible] = useState([true, true, true, true]);
  const [config, setConfig] = useState<Config>(DEFAULT);
  const [source, setSource] = useState<'labels' | 'geometry' | 'neural'>(
    'neural',
  );

  const [single, setSingle] = useState<Point[] | null>(null);
  const [fileName, setFileName] = useState('');
  const [sequences, setSequences] = useState<Sequence[]>([]);
  const [sequenceId, setSequenceId] = useState('');
  const [loaded, setLoaded] = useState<LoadedFrame | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const [captureHz, setCaptureHz] = useState(10);
  const [speed, setSpeed] = useState(1);
  const [loop, setLoop] = useState(true);
  const [renderFps, setRenderFps] = useState(0);
  const [history, setHistory] = useState<number[]>([]);

  const [objects, setObjects] = useState<TrackObject[]>([]);
  const [selectedObject, setSelectedObject] = useState<TrackObject | null>(
    null,
  );
  const [backend, setBackend] = useState<'checking' | 'online' | 'offline'>(
    'checking',
  );
  const [backendSession, setBackendSession] = useState<string | undefined>();
  const [tracking, setTracking] =
    useState<TrackingConfig>(TRACKING_DEFAULTS);

  const [showBoxes, setShowBoxes] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showTraj, setShowTraj] = useState(true);
  const [showMotion, setShowMotion] = useState(true);

  const input = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const labelFolderInput = useRef<HTMLInputElement>(null);
  const lastTick = useRef(0);
  const lastFrame = useRef(-1);
  const requestToken = useRef(0);

  const sequence = sequences.find((item) => item.id === sequenceId) || null;
  const sequenceMode = !!sequence;
  const maxFrame = sequence ? sequence.frames.length - 1 : single ? 0 : 299;
  const timingAssumed = sequence?.timingAssumed ?? true;

  const points = useMemo(
    () => loaded?.points || single || simulate(frame, scene),
    [loaded, single, frame, scene],
  );

  const result = useMemo(
    () => project(points, config, source),
    [points, config, source],
  );

  const displayPoints = useMemo(() => {
    if (source !== 'neural') {
      return points;
    }

    return points.map((point, index) => ({
      ...point,
      label: result.pointLabels[index],
    }));
  }, [points, result, source]);

  const counts = useMemo(
    () =>
      COLOURS.map((_, index) =>
        result.cells.filter((cell) => cell.label === index).length,
      ),
    [result],
  );

  const currentTitle =
    nav.find((item) => item[0] === page)?.[1] || 'Perception workspace';

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const abortController = new AbortController();

    health(abortController.signal)
      .then(() => setBackend('online'))
      .catch(() => setBackend('offline'));

    return () => abortController.abort();
  }, []);

  useEffect(() => {
    if (!running || single) {
      return;
    }

    const delay = Math.max(20, 1000 / (captureHz * speed));

    const id = window.setTimeout(() => {
      setFrame((currentFrame) => {
        if (currentFrame >= maxFrame) {
          return loop ? 0 : currentFrame;
        }

        return currentFrame + 1;
      });
    }, delay);

    return () => window.clearTimeout(id);
  }, [running, single, frame, maxFrame, loop, captureHz, speed]);

  useEffect(() => {
    const now = performance.now();
    const previous = lastTick.current || now;
    const nextFps = 1000 / Math.max(1, now - previous);

    lastTick.current = now;

    setRenderFps((current) =>
      Math.abs(current - nextFps) < 0.25 ? current : nextFps,
    );

    setHistory((current) => {
      const previousLatency = current[current.length - 1];

      if (
        previousLatency !== undefined &&
        Math.abs(previousLatency - result.latency) < 0.01
      ) {
        return current;
      }

      return [...current.slice(-39), result.latency];
    });
  }, [frame, result.latency]);

  useEffect(() => {
    if (!sequence) {
      return;
    }

    let stale = false;
    const token = ++requestToken.current;
    const current = sequence.frames[frame];

    setBusy(true);
    setError('');

    cache
      .load(current)
      .then((loadedFrame) => {
        if (stale || token !== requestToken.current) {
          return;
        }

        setLoaded(loadedFrame);
        cache.prefetch(sequence.frames, frame);
        setSource(loadedFrame.semanticLabels ? 'labels' : 'geometry');
      })
      .catch((caught) => {
        if (!stale) {
          setError(
            caught instanceof Error
              ? caught.message
              : 'Could not read sequence frame.',
          );
        }
      })
      .finally(() => {
        if (!stale) {
          setBusy(false);
        }
      });

    return () => {
      stale = true;
    };
  }, [sequence, frame]);

  useEffect(() => {
    let cancelled = false;
    const jumped =
      lastFrame.current >= 0 && frame !== lastFrame.current + 1;

    lastFrame.current = frame;

    if (!points.length) {
      setObjects([]);
      return;
    }

    const timestamp =
      sequence && loaded ? timestampFor(loaded, captureHz) : frame / captureHz;

    const semantic = loaded?.semanticLabels;
    const names = loaded?.semanticNames;

    if (!sequenceMode) {
      const detected = extractObjects(points, semantic, names, tracking);
      setObjects(tracker.update(detected, timestamp, frame, tracking, jumped));
      return;
    }

    if (backend === 'online' && loaded) {
      const abortController = new AbortController();
      const token = ++requestToken.current;

      trackFrame(loaded, {
        sessionId: backendSession,
        timestamp,
        frameIndex: frame,
        jumped,
        config: tracking,
        signal: abortController.signal,
      })
        .then((response) => {
          if (cancelled || token !== requestToken.current) {
            return;
          }

          setBackendSession(response.sessionId);
          setObjects(response.objects);
        })
        .catch(() => {
          if (cancelled) {
            return;
          }

          setBackend('offline');

          const detected = extractObjects(points, semantic, names, tracking);
          setObjects(
            tracker.update(detected, timestamp, frame, tracking, jumped),
          );
        });

      return () => {
        cancelled = true;
        abortController.abort();
      };
    }

    const detected = extractObjects(points, semantic, names, tracking);
    setObjects(tracker.update(detected, timestamp, frame, tracking, jumped));
  }, [
    points,
    loaded,
    backend,
    backendSession,
    frame,
    captureHz,
    tracking,
    sequenceMode,
  ]);

  const resetTracking = () => {
    tracker.reset();
    setBackendSession(undefined);
    lastFrame.current = -1;
    setObjects([]);
  };

  const clearToSynthetic = () => {
    cache.clear();
    resetTracking();
    setSequences([]);
    setSequenceId('');
    setLoaded(null);
    setSingle(null);
    setFileName('');
    setFrame(0);
    setRunning(true);
    setSource('neural');
  };

  const loadSingle = async (file?: File) => {
    if (!file) {
      return;
    }

    setBusy(true);
    setError('');

    try {
      if (file.size > 8 * 1024 * 1024) {
        throw Error('Choose a scan smaller than 8 MB.');
      }

      const ext = file.name.split('.').pop()?.toLowerCase();

      if (!['csv', 'bin'].includes(ext || '')) {
        throw Error('Choose a CSV or KITTI .bin point cloud.');
      }

      const parsed =
        ext === 'bin'
          ? parseBin(await file.arrayBuffer())
          : parseCSV(await file.text());

      clearToSynthetic();
      setSingle(parsed);
      setFileName(file.name);
      setRunning(false);
      setSource(parsed.every((point) => point.label !== undefined) ? 'labels' : 'geometry');
      setPage('workspace');

      toast.success(`Loaded ${parsed.length.toLocaleString()} points`);
    } catch (caught) {
      const message =
        caught instanceof Error ? caught.message : 'Unable to read scan.';

      setError(message);
      toast.error(message);
    } finally {
      setBusy(false);

      if (input.current) {
        input.current.value = '';
      }
    }
  };

  const loadSelection = (files: FileList | null) => {
    if (!files?.length) {
      return;
    }

    const chosen = [...files];
    const hasLabels = chosen.some((file) =>
      file.name.toLowerCase().endsWith('.label'),
    );

    if (chosen.length > 1 || hasLabels) {
      loadFolder(files);
      return;
    }

    void loadSingle(chosen[0]);
  };

  const loadFolder = (files: FileList | null) => {
    if (!files?.length) {
      return;
    }

    try {
      const selected = [...files];
      const hasBins = selected.some((file) =>
        file.name.toLowerCase().endsWith('.bin'),
      );
      const hasLabels = selected.some((file) =>
        file.name.toLowerCase().endsWith('.label'),
      );

      if (!hasBins && hasLabels) {
        if (!sequences.length) {
          throw Error(
            'Load the matching .bin sequence folder before adding a label-only folder.',
          );
        }

        const attached = attachLabelsToSequences(sequences, selected);

        if (!attached.matched) {
          throw Error('No matching .label files found for the loaded sequence frames.');
        }

        cache.clear();
        resetTracking();
        setSequences(attached.sequences);
        setLoaded(null);
        setRunning(false);
        setSource('labels');
        setPage('workspace');

        toast.success(
          `Attached ${attached.matched.toLocaleString()} label file(s) to the loaded sequence.`,
        );

        return;
      }

      const nextSequences = groupSequences(selected, captureHz);

      if (!nextSequences.length) {
        throw Error('No KITTI .bin files found in the selected folder.');
      }

      cache.clear();
      resetTracking();
      setSequences(nextSequences);
      setSequenceId(nextSequences[0].id);
      setSingle(null);
      setLoaded(null);
      setFileName(nextSequences[0].name);
      setFrame(0);
      setRunning(false);
      setSource(hasLabels ? 'labels' : 'geometry');
      setPage('workspace');

      toast.success(
        `Loaded ${nextSequences.length} sequence(s), ${nextSequences
          .reduce((total, item) => total + item.frames.length, 0)
          .toLocaleString()} frame(s)`,
      );
    } catch (caught) {
      const message =
        caught instanceof Error ? caught.message : 'Unable to load folder.';

      setError(message);
      toast.error(message);
    } finally {
      if (input.current) {
        input.current.value = '';
      }

      if (folderInput.current) {
        folderInput.current.value = '';
      }

      if (labelFolderInput.current) {
        labelFolderInput.current.value = '';
      }
    }
  };

  const switchSequence = (id: string) => {
    resetTracking();
    setSequenceId(id);
    setLoaded(null);
    setFrame(0);
    setRunning(false);
    setFileName(sequences.find((item) => item.id === id)?.name || '');
  };

  const report = () => {
    download(
      'foveamap-report.json',
      JSON.stringify(
        {
          source: fileName || `synthetic:${scene}`,
          frame,
          captureHz,
          playbackSpeed: speed,
          timingAssumed,
          backend,
          tracking,
          objects,
          result: {
            accepted: result.accepted,
            dropped: result.dropped,
            cells: result.cells.length,
            latencyMs: result.latency,
          },
        },
        null,
        2,
      ),
    );
  };

  if (!mounted) {
    return (
      <div className="shell flex">
        <main className="appmain">
          <div className="page">
            <div className="heading">
              <div>
                <h1>FoveaMap</h1>
                <p>Starting perception workspace...</p>
              </div>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <SidebarProvider
      style={{ '--sidebar-width': '223px' } as React.CSSProperties}
    >
      <div className="shell flex">
        <Sidebar>
          <SidebarHeader className="p-0">
            <div className="side-brand">
              <span className="logo">
                <Radar size={24} />
              </span>
              FoveaMap
              <span style={{ color: '#43d7b0', fontSize: 21 }}>•</span>
            </div>
          </SidebarHeader>

          <SidebarContent className="px-3">
            <div className="side-caption">PERCEPTION PLATFORM</div>

            <SidebarMenu>
              {nav.map(([id, label, Icon]) => (
                <SidebarMenuItem key={id}>
                  <SidebarMenuButton
                    className="nav-item"
                    isActive={page === id}
                    onClick={() => setPage(id)}
                  >
                    <Icon size={18} />
                    <span>{label}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarContent>

          <SidebarFooter className="p-0">
            <div className="side-foot">
              <Radar size={20} style={{ color: '#6ccba9', marginBottom: 8 }} />
              <b>Sequence-aware LiDAR.</b>
              <span>Geometry, tracking and honest motion states.</span>
            </div>
          </SidebarFooter>
        </Sidebar>

        <main className="appmain">
          <header className="topbar">
            <div className="breadcrumb">
              <SidebarTrigger className="mobile-trigger" />
              <span>FoveaMap</span>
              <strong>{currentTitle}</strong>
            </div>

            <div className="top-right">
              <span className="badge">
                <span className="dot" />{' '}
                {sequenceMode ? 'Sequence' : single ? 'Single scan' : 'Simulation'}
              </span>

              <span className={`badge ${backend === 'offline' ? 'offline' : ''}`}>
                Backend {backend}
              </span>

              <div className="avatar">FM</div>
            </div>
          </header>

          <div className="page">
            <div className="heading">
              <div>
                <h1>{currentTitle}</h1>
                <p>
                  {sequenceMode
                    ? 'RELLIS/KITTI sequence playback with lazy loading and object tracking.'
                    : 'Synthetic simulator and single-frame inspection remain available.'}
                </p>
              </div>

              <div className="actions">
                <button className="btn" onClick={report}>
                  <ArrowDownToLine />
                  Export report
                </button>

                {page === 'workspace' && (
                  <button
                    className="btn primary"
                    disabled={busy || !!single}
                    onClick={() => setRunning(!running)}
                  >
                    {running ? <Pause /> : <Play />}
                    {running ? 'Pause' : 'Play'}
                  </button>
                )}
              </div>
            </div>

            <input
              ref={input}
              type="file"
              accept=".csv,.bin,.label"
              className="file-input"
              multiple
              onChange={(event) => loadSelection(event.target.files)}
            />

            <input
              ref={folderInput}
              type="file"
              className="file-input"
              multiple
              {...{ webkitdirectory: '' }}
              onChange={(event) => loadFolder(event.target.files)}
            />

            <input
              ref={labelFolderInput}
              type="file"
              className="file-input"
              multiple
              {...{ webkitdirectory: '' }}
              onChange={(event) => loadFolder(event.target.files)}
            />

            {error && (
              <div className="error" role="alert" style={{ marginBottom: 15 }}>
                {error}
              </div>
            )}

            {backend === 'offline' && (
              <div className="notice">
                Backend is offline or unreachable. Basic visualisation and
                deterministic local tracking remain available; API session
                isolation and future model inference require the Python backend.
              </div>
            )}

            {(page === 'workspace' || page === 'metrics') && (
              <div className="stats">
                <div className="stat">
                  <div className="stat-top">
                    Rendered playback FPS
                    <Gauge />
                  </div>
                  <div className="stat-value">
                    {Number.isFinite(renderFps) ? renderFps.toFixed(1) : '—'}
                    <small>FPS</small>
                  </div>
                  <div className="stat-note">Canvas/UI rendering only</div>
                </div>

                <div className="stat">
                  <div className="stat-top">
                    Processing latency
                    <Activity />
                  </div>
                  <div className="stat-value">
                    {result.latency.toFixed(1)}
                    <small>ms</small>
                  </div>
                  <div className="stat-note">Grid projection, not draw FPS</div>
                </div>

                <div className="stat">
                  <div className="stat-top">
                    Current points
                    <Waypoints />
                  </div>
                  <div className="stat-value">{points.length.toLocaleString()}</div>
                  <div className="stat-note">
                    {result.cells.length.toLocaleString()} occupied cells
                  </div>
                </div>

                <div className="stat">
                  <div className="stat-top">
                    Tracked objects
                    <Box />
                  </div>
                  <div className="stat-value">{objects.length}</div>
                  <div className="stat-note">
                    {tracking.stationarySensor ||
                    objects.some((item) => item.motionFrame === 'world-compensated')
                      ? 'Speed valid'
                      : 'Relative / uncompensated'}
                  </div>
                </div>
              </div>
            )}

            {page === 'workspace' && (
              <>
                <div className="workspace-grid">
                  <section className="panel">
                    <div className="panel-head">
                      <h2>
                        <Box />
                        Live perception
                      </h2>

                      <Tabs
                        className="mode-tabs"
                        value={mode}
                        onValueChange={setMode}
                      >
                        <TabsList>
                          <TabsTrigger value="semantic">2.5D grid</TabsTrigger>
                          <TabsTrigger value="points">Point cloud</TabsTrigger>
                          <TabsTrigger value="elevation">Elevation</TabsTrigger>
                        </TabsList>
                      </Tabs>

                      <button
                        className="btn"
                        style={{ padding: '4px 8px', minHeight: 28, fontSize: 12 }}
                        onClick={() => setTop(!top)}
                      >
                        {top ? 'Perspective' : 'Top-down'}
                        <Focus size={13} />
                      </button>
                    </div>

                    <Scene
                      points={displayPoints}
                      cells={result.cells}
                      config={config}
                      mode={mode}
                      visible={visible}
                      top={top}
                      labels={source !== 'geometry'}
                      rings={rings}
                      objects={showMotion ? objects : []}
                      showBoxes={showBoxes}
                      showObjectLabels={showLabels}
                      showTrajectories={showTraj}
                      showMotion={showMotion}
                      onSelectObject={setSelectedObject}
                    />

                    <div className="scene-bottom">
                      <div className="legend">
                        {['Drivable', 'Non-drivable', 'Static obstacle', 'Dynamic object'].map(
                          (label, index) => (
                            <span key={label}>
                              <i style={{ background: COLOURS[index] }} />
                              {label}
                            </span>
                          ),
                        )}
                        <span>
                          <i style={{ background: '#ff8a67' }} />
                          Moving
                        </span>
                        <span>
                          <i style={{ background: '#65dbb7' }} />
                          Stationary
                        </span>
                        <span>
                          <i style={{ background: '#a6b4c8' }} />
                          Unknown
                        </span>
                      </div>

                      <span>
                        {loaded?.name || fileName || `${scenes[scene]} frame ${frame}`}
                      </span>
                    </div>

                    <div className="playback">
                      <button
                        aria-label="Previous frame"
                        disabled={!!single}
                        onClick={() => {
                          setRunning(false);
                          setFrame((current) => Math.max(0, current - 1));
                        }}
                      >
                        <SkipBack />
                      </button>

                      <button
                        aria-label={running ? 'Pause playback' : 'Play playback'}
                        disabled={!!single}
                        onClick={() => setRunning(!running)}
                      >
                        {running ? <Pause /> : <Play />}
                      </button>

                      <button
                        aria-label="Next frame"
                        disabled={!!single}
                        onClick={() => {
                          setRunning(false);
                          setFrame((current) => Math.min(maxFrame, current + 1));
                        }}
                      >
                        <SkipForward />
                      </button>

                      <span className="time">
                        {sequenceMode
                          ? basename(sequence!.frames[frame]?.name || '')
                          : single
                            ? 'SINGLE SCAN'
                            : `${(frame / captureHz).toFixed(1)}s`}
                      </span>

                      <Slider
                        min={0}
                        max={maxFrame}
                        step={1}
                        value={[frame]}
                        disabled={!!single}
                        onValueChange={(value) => {
                          setRunning(false);
                          setFrame(value[0]);
                        }}
                      />

                      <span className="time">
                        {frame + 1} / {maxFrame + 1}
                      </span>

                      <button
                        aria-label="Restart sequence"
                        disabled={!!single}
                        onClick={() => {
                          resetTracking();
                          setFrame(0);
                        }}
                      >
                        <RotateCcw />
                      </button>
                    </div>
                  </section>

                  <aside className="rail">
                    <section className="panel">
                      <div className="panel-head">
                        <h2>
                          <Database />
                          Data source
                        </h2>
                        <span className="badge">
                          {sequenceMode ? 'Sequence' : single ? 'File' : 'Synthetic'}
                        </span>
                      </div>

                      <div className="panel-body">
                        <div className="field-label">SOURCE</div>

                        {sequenceMode ? (
                          <>
                            <Picker
                              label="Dataset sequence"
                              value={sequenceId}
                              onChange={switchSequence}
                              items={Object.fromEntries(
                                sequences.map((item) => [
                                  item.id,
                                  `${item.name} (${item.frames.length})`,
                                ]),
                              )}
                            />
                            <p className="source-description">
                              Frames are sorted numerically by filename and kept
                              separate by sequence. Timing:{' '}
                              {timingAssumed
                                ? `assumed ${captureHz} Hz`
                                : 'timestamps provided'}
                              .
                            </p>
                          </>
                        ) : single ? (
                          <div style={{ fontSize: 14, overflowWrap: 'anywhere' }}>
                            {fileName}
                            <button
                              className="btn"
                              style={{ marginTop: 10 }}
                              onClick={clearToSynthetic}
                            >
                              Return to simulator
                            </button>
                          </div>
                        ) : (
                          <Picker
                            label="Simulation scene"
                            value={scene}
                            onChange={(value) => {
                              clearToSynthetic();
                              setScene(value);
                            }}
                            items={scenes}
                          />
                        )}

                        <div className="source-description">
                          {loaded?.semanticSource === 'rellis-ground-truth'
                            ? 'RELLIS ground-truth semantic labels loaded.'
                            : source === 'neural'
                              ? 'Synthetic-trained demo MLP, not a RELLIS model.'
                              : 'Geometry-only estimates; labels are unclassified obstacles.'}
                        </div>
                      </div>
                    </section>

                    <section className="panel">
                      <div className="panel-head">
                        <h2>
                          <Focus />
                          Adaptive resolution
                        </h2>
                        <button
                          aria-label="Edit grid settings"
                          onClick={() => setPage('settings')}
                        >
                          <Settings2 size={15} color="#aac1b8" />
                        </button>
                      </div>

                      <div className="panel-body" style={{ paddingTop: 5 }}>
                        {[
                          ['Near field', `0–${config.near} m`, config.nearSize, ''],
                          [
                            'Mid field',
                            `${config.near}–${config.mid} m`,
                            config.midSize,
                            'mid',
                          ],
                          [
                            'Far field',
                            `${config.mid}–${config.far} m`,
                            config.farSize,
                            'far',
                          ],
                        ].map(([label, range, size, className]) => (
                          <div className="resolution" key={label as string}>
                            <div className={`res-icon ${className}`}>
                              <Focus size={17} />
                            </div>
                            <div>
                              <strong>{label as string}</strong>
                              <span>{range as string}</span>
                            </div>
                            <b>
                              {Math.round(Number(size) * 100)} <span>cm</span>
                            </b>
                          </div>
                        ))}

                        <div className="res-note">Exclusive radial cell assignment</div>
                      </div>
                    </section>

                    <section className="panel">
                      <div className="panel-head">
                        <h2>
                          <Layers3 />
                          Semantic layers
                        </h2>
                      </div>

                      <div
                        className="panel-body"
                        style={{ paddingTop: 8, paddingBottom: 10 }}
                      >
                        {['Drivable', 'Non-drivable', 'Static obstacles', 'Dynamic objects'].map(
                          (label, index) => (
                            <div className="class-row" key={label}>
                              <i style={{ background: COLOURS[index] }} />
                              <span>{label}</span>
                              <small>
                                {(
                                  (100 * counts[index]) /
                                  Math.max(1, result.cells.length)
                                ).toFixed(0)}
                                %
                              </small>
                              <Switch
                                aria-label={`Show ${label}`}
                                checked={visible[index]}
                                onCheckedChange={(checked) =>
                                  setVisible((current) =>
                                    current.map((item, itemIndex) =>
                                      itemIndex === index ? checked : item,
                                    ),
                                  )
                                }
                              />
                            </div>
                          ),
                        )}
                      </div>
                    </section>

                    <section className="panel">
                      <div className="panel-head">
                        <h2>
                          <Layers3 />
                          Object overlays
                        </h2>
                      </div>

                      <div className="panel-body">
                        {[
                          ['Boxes', showBoxes, setShowBoxes],
                          ['Labels', showLabels, setShowLabels],
                          ['Trajectories', showTraj, setShowTraj],
                          ['Motion overlay', showMotion, setShowMotion],
                        ].map(([label, value, setter]) => (
                          <div className="class-row" key={label as string}>
                            <span>{label as string}</span>
                            <Switch
                              checked={value as boolean}
                              onCheckedChange={setter as (value: boolean) => void}
                            />
                          </div>
                        ))}
                      </div>
                    </section>

                    <section className="panel">
                      <div className="panel-head">
                        <h2>
                          <Box />
                          Selected object
                        </h2>
                      </div>

                      <div className="panel-body prose">
                        {selectedObject ? (
                          <>
                            <p>
                              <b>#{selectedObject.trackId}</b>{' '}
                              {selectedObject.className ||
                                `Unclassified obstacle #${selectedObject.trackId}`}
                            </p>
                            <p>
                              Motion: {selectedObject.motionStatus} ·{' '}
                              {selectedObject.speedMps === null
                                ? 'speed uncompensated'
                                : `${selectedObject.speedMps.toFixed(2)} m/s`}
                            </p>
                            <p>
                              Position:{' '}
                              {selectedObject.centroid
                                .map((value) => value.toFixed(2))
                                .join(', ')}{' '}
                              m
                            </p>
                            <p>
                              Dimensions:{' '}
                              {selectedObject.dimensions
                                .map((value) => value.toFixed(2))
                                .join(' x ')}{' '}
                              m
                            </p>
                            <p>Source: {selectedObject.source}</p>
                            <p>
                              Track history: {selectedObject.trackHistory.length}{' '}
                              samples · {selectedObject.trackingStatus}
                            </p>
                          </>
                        ) : (
                          <p>
                            Click a box or centroid to inspect dimensions, position,
                            class source and track state.
                          </p>
                        )}
                      </div>
                    </section>
                  </aside>

                  <div className="bottom-grid">
                    <section className="panel">
                      <div className="panel-head">
                        <h2>
                          <Activity />
                          Object tracks
                        </h2>
                        <small>{objects.length} active</small>
                      </div>

                      <div className="panel-body table-scroll">
                        <table className="metric-table">
                          <tbody>
                            {objects.slice(0, 12).map((object) => (
                              <tr key={object.trackId}>
                                <td>#{object.trackId}</td>
                                <td>
                                  {object.className ||
                                    `Unclassified obstacle #${object.trackId}`}
                                </td>
                                <td>{object.motionStatus}</td>
                                <td>
                                  {object.speedMps === null
                                    ? 'uncomp.'
                                    : `${object.speedMps.toFixed(1)} m/s`}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </section>

                    <section className="panel">
                      <div className="panel-head">
                        <h2>
                          <Settings2 />
                          Playback timing
                        </h2>
                      </div>

                      <div className="panel-body prose">
                        <p>
                          Capture frequency and playback speed are separate. Velocity
                          uses timestamps from capture frequency or supplied
                          poses/timestamps, never the UI speed multiplier.
                        </p>

                        <div className="settings-field">
                          <label>
                            Capture frequency
                            <strong>{captureHz} Hz</strong>
                          </label>
                          <Slider
                            min={1}
                            max={20}
                            step={1}
                            value={[captureHz]}
                            onValueChange={(value) => setCaptureHz(value[0])}
                          />
                        </div>

                        <div className="settings-field">
                          <label>
                            Playback speed
                            <strong>{speed.toFixed(1)}x</strong>
                          </label>
                          <Slider
                            min={0.25}
                            max={4}
                            step={0.25}
                            value={[speed]}
                            onValueChange={(value) => setSpeed(value[0])}
                          />
                        </div>

                        <label className="class-row">
                          Loop sequence
                          <Switch checked={loop} onCheckedChange={setLoop} />
                        </label>
                      </div>
                    </section>
                  </div>
                </div>
              </>
            )}

            {page === 'data' && (
              <section className="panel large-panel">
                <div
                  className="dropzone"
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => {
                    event.preventDefault();

                    const files = [...event.dataTransfer.files];

                    if (files.length > 1) {
                      loadFolder(event.dataTransfer.files);
                    } else {
                      loadSingle(files[0]);
                    }
                  }}
                >
                  <Upload />
                  <h2>Load a scan or sequence</h2>
                  <p>
                    Use a single CSV/KITTI `.bin`, a matching `.bin + .label`
                    pair, a RELLIS parent folder, or load the `.bin` folder first
                    and then add the matching `.label` folder.
                  </p>

                  <div className="actions">
                    <button
                      className="btn primary"
                      disabled={busy}
                      onClick={() => input.current?.click()}
                    >
                      Choose single scan
                    </button>

                    <button
                      className="btn"
                      disabled={busy}
                      onClick={() => folderInput.current?.click()}
                    >
                      Choose bin or parent folder
                    </button>

                    <button
                      className="btn"
                      disabled={busy || !sequences.length}
                      onClick={() => labelFolderInput.current?.click()}
                    >
                      Add label folder
                    </button>
                  </div>
                </div>
              </section>
            )}

            {page === 'metrics' && (
              <div className="doc-grid">
                <section className="panel large-panel prose">
                  <h2>Processing</h2>
                  <p>
                    Grid latency is measured separately from rendered playback FPS.
                    Current grid has {result.cells.length.toLocaleString()} cells
                    from {result.accepted.toLocaleString()} accepted points.
                  </p>

                  <svg viewBox="0 0 500 150" style={{ width: '100%', height: 150 }}>
                    <path
                      d="M0 125H500 M0 65H500 M0 5H500"
                      stroke="#2b424c"
                      strokeDasharray="4 4"
                    />
                    <polyline
                      fill="none"
                      stroke="#42d5ad"
                      strokeWidth="2.5"
                      points={history
                        .map(
                          (value, index) =>
                            `${(index / Math.max(1, history.length - 1)) * 490 + 5},${
                              135 -
                              (value / Math.max(1, ...history)) * 120
                            }`,
                        )
                        .join(' ')}
                    />
                  </svg>
                </section>

                <section className="panel large-panel prose">
                  <h2>Motion honesty</h2>
                  <p>
                    {tracking.stationarySensor
                      ? 'Stationary-sensor assumption enabled. Speeds are treated as valid in the sensor frame.'
                      : 'No pose compensation is active. Moving/stationary states from real sequences are shown as unknown or sensor-relative/uncompensated.'}
                  </p>
                </section>
              </div>
            )}

            {page === 'settings' && (
              <div className="doc-grid">
                <section className="panel large-panel prose">
                  <h2>Spatial representation</h2>

                  {[
                    ['nearSize', 'Near-field cell size', 0.05, 0.15, 0.01],
                    ['midSize', 'Mid-field cell size', 0.15, 0.5, 0.05],
                    ['farSize', 'Far-field cell size', 0.5, 1, 0.05],
                  ].map(([key, label, min, max, step]) => (
                    <div className="settings-field" key={key as string}>
                      <label>
                        {label as string}
                        <strong>
                          {Math.round(config[key as keyof Config] * 100)} cm
                        </strong>
                      </label>
                      <Slider
                        value={[config[key as keyof Config]]}
                        min={min as number}
                        max={max as number}
                        step={step as number}
                        onValueChange={(value) =>
                          setConfig((current) => ({
                            ...current,
                            [key as string]: value[0],
                          }))
                        }
                      />
                    </div>
                  ))}

                  <button className="btn" onClick={() => setConfig(DEFAULT)}>
                    <RotateCcw />
                    Restore defaults
                  </button>
                </section>

                <section className="panel large-panel prose">
                  <h2>Tracking</h2>

                  <label className="class-row">
                    Stationary sensor assumption
                    <Switch
                      checked={tracking.stationarySensor}
                      onCheckedChange={(checked) => {
                        setTracking((current) => ({
                          ...current,
                          stationarySensor: checked,
                        }));
                        resetTracking();
                      }}
                    />
                  </label>

                  <div className="settings-field">
                    <label>
                      Ground margin
                      <strong>{tracking.groundMargin.toFixed(2)} m</strong>
                    </label>
                    <Slider
                      min={0.1}
                      max={0.8}
                      step={0.05}
                      value={[tracking.groundMargin]}
                      onValueChange={(value) =>
                        setTracking((current) => ({
                          ...current,
                          groundMargin: value[0],
                        }))
                      }
                    />
                  </div>

                  <div className="settings-field">
                    <label>
                      Moving threshold
                      <strong>{tracking.movingEnterMps.toFixed(1)} m/s</strong>
                    </label>
                    <Slider
                      min={0.2}
                      max={5}
                      step={0.1}
                      value={[tracking.movingEnterMps]}
                      onValueChange={(value) =>
                        setTracking((current) => ({
                          ...current,
                          movingEnterMps: value[0],
                        }))
                      }
                    />
                  </div>

                  <div className="settings-field">
                    <label>Semantic input</label>
                    <Picker
                      label="Semantic input"
                      value={source}
                      onChange={(value) =>
                        setSource(value as 'labels' | 'geometry' | 'neural')
                      }
                      items={{
                        neural: 'Neural demo · synthetic only',
                        labels: 'Input labels / annotations',
                        geometry: 'Geometric baseline',
                      }}
                    />
                  </div>
                </section>
              </div>
            )}

            {page === 'guide' && (
              <div className="doc-grid">
                <section className="panel large-panel prose">
                  <h2>RELLIS sequence workflow</h2>
                  <p>
                    Select a folder that contains one or more five-digit RELLIS
                    sequence folders. FoveaMap sorts frame filenames numerically
                    and does not concatenate different sequences. `.label` files
                    are optional and must match the exact point count.
                  </p>
                  <pre>
                    {
                      '00000/os1_cloud_node_kitti_bin/000000.bin\n00000/os1_cloud_node_kitti_bin/000001.bin\n00001/os1_cloud_node_kitti_bin/000000.bin'
                    }
                  </pre>
                </section>

                <section className="panel large-panel prose">
                  <h2>Future model adapter</h2>
                  <p>
                    Imported predictions must preserve point order and use metres
                    with feature order x,y,z,intensity. Model predictions require
                    modelId and checkpointHash metadata. The current bundled MLP is
                    synthetic-only and is not presented as a trained RELLIS model.
                  </p>
                  <p>
                    Pose JSON support is documented in the backend README. Without
                    validated world-from-LiDAR poses, real sequence motion is
                    uncompensated.
                  </p>
                </section>
              </div>
            )}

            <footer className="footer">
              <span>
                <span className="dot" style={{ marginRight: 6 }} />
                {sequenceMode
                  ? 'Sequence playback'
                  : single
                    ? 'Single scan'
                    : 'Synthetic simulation'}{' '}
                ·{' '}
                {source === 'neural'
                  ? 'Synthetic demo semantics'
                  : source === 'labels'
                    ? 'Input/annotation semantics'
                    : 'Geometry semantics'}
              </span>
            </footer>
          </div>
        </main>

        <Toaster position="bottom-right" theme="dark" richColors />
      </div>
    </SidebarProvider>
  );
}