import { COLOURS, geometry, type Cell, type Config, type Point } from './engine.ts';

export type WebGpuRenderInput = {
  points: Point[];
  cells: Cell[];
  mode: string;
  labels: boolean;
  visible: boolean[];
  config: Config;
  yaw: number;
  zoom: number;
  top: boolean;
  width: number;
  height: number;
};

export type WebGpuRenderer = {
  render: (input: WebGpuRenderInput) => void;
  dispose: () => void;
};

type RendererState = {
  device: GPUDevice;
  context: GPUCanvasContext;
  pipeline: GPURenderPipeline;
  uniformBuffer: GPUBuffer;
  bindGroupLayout: GPUBindGroupLayout;
  pointBuffer?: GPUBuffer;
  pointCapacity: number;
  disposed: boolean;
};

const shaderCode = `
struct Uniforms {
  yaw: f32,
  zoom: f32,
  width: f32,
  height: f32,
  top: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

struct RenderPoint {
  position_size: vec4<f32>,
  color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> points: array<RenderPoint>;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

fn corner(vertex_index: u32) -> vec2<f32> {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
  );

  return corners[vertex_index];
}

@vertex
fn vs_main(
  @builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32,
) -> VertexOut {
  let item = points[instance_index];
  let world = item.position_size.xyz;
  let point_size = item.position_size.w;

  let c = cos(uniforms.yaw);
  let s = sin(uniforms.yaw);

  let rotated_x = world.x * c - world.y * s;
  let rotated_y = world.x * s + world.y * c;

  let base_scale = min(uniforms.width / 145.0, uniforms.height / 73.0) * uniforms.zoom;
  let y_scale = select(0.56, 1.0, uniforms.top > 0.5);

  let screen_x = uniforms.width * 0.5 + rotated_x * base_scale;
  let screen_y = uniforms.height * 0.58 + rotated_y * base_scale * y_scale - world.z * base_scale * 0.95;
  let offset = corner(vertex_index) * point_size;
  let px = screen_x + offset.x;
  let py = screen_y + offset.y;

  var out: VertexOut;
  out.position = vec4<f32>(
    (px / uniforms.width) * 2.0 - 1.0,
    1.0 - (py / uniforms.height) * 2.0,
    0.0,
    1.0
  );
  out.color = item.color;
  return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
  return input.color;
}
`;

function hexToRgb(hex: string) {
  const value = hex.replace('#', '');
  const r = Number.parseInt(value.slice(0, 2), 16) / 255;
  const g = Number.parseInt(value.slice(2, 4), 16) / 255;
  const b = Number.parseInt(value.slice(4, 6), 16) / 255;

  return [r, g, b] as const;
}

function buildRenderPoints({
  points,
  cells,
  mode,
  labels,
  visible,
  config,
  zoom,
}: WebGpuRenderInput) {
  const data =
    mode === 'points'
      ? points.map((point) => {
          const label =
            labels && point.label !== undefined ? point.label : geometry(point);

          return {
            x: point.x,
            y: point.y,
            z: point.z,
            label,
            size: label > 1 ? 0.8 : 0.55,
          };
        })
      : cells.map((cell) => ({
          x: cell.x,
          y: cell.y,
          z: cell.mean,
          label: cell.label,
          size: Math.max(0.7, cell.size * 2.2 * zoom),
        }));

  const filtered = data.filter(
    (item) => visible[item.label] && Math.hypot(item.x, item.y) <= config.far,
  );
  const packed = new Float32Array(filtered.length * 8);

  filtered.forEach((item, index) => {
    const offset = index * 8;
    const [r, g, b] =
      mode === 'elevation'
        ? [0.35, 0.8, 0.75]
        : hexToRgb(COLOURS[item.label] || '#a6b4c8');

    packed[offset + 0] = item.x;
    packed[offset + 1] = item.y;
    packed[offset + 2] = item.z;
    packed[offset + 3] = item.size;
    packed[offset + 4] = r;
    packed[offset + 5] = g;
    packed[offset + 6] = b;
    packed[offset + 7] = item.label === 0 ? 0.65 : 0.9;
  });

  return {
    packed,
    count: filtered.length,
  };
}

function renderWebGpu(state: RendererState, input: WebGpuRenderInput) {
  if (state.disposed) return;

  const { packed, count } = buildRenderPoints(input);
  const requiredBytes = Math.max(32, packed.byteLength);

  if (!state.pointBuffer || requiredBytes > state.pointCapacity) {
    state.pointBuffer?.destroy();
    state.pointCapacity = requiredBytes * 2;
    state.pointBuffer = state.device.createBuffer({
      size: state.pointCapacity,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  state.device.queue.writeBuffer(state.pointBuffer, 0, packed);
  state.device.queue.writeBuffer(
    state.uniformBuffer,
    0,
    new Float32Array([
      input.yaw,
      input.zoom,
      input.width,
      input.height,
      input.top ? 1 : 0,
      0,
      0,
      0,
    ]),
  );

  const bindGroup = state.device.createBindGroup({
    layout: state.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: state.uniformBuffer } },
      { binding: 1, resource: { buffer: state.pointBuffer } },
    ],
  });

  const encoder = state.device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: state.context.getCurrentTexture().createView(),
        clearValue: { r: 0.043, g: 0.106, b: 0.137, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  });

  pass.setPipeline(state.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(6, count);
  pass.end();

  state.device.queue.submit([encoder.finish()]);
}

export async function createWebGpuRenderer(
  canvas: HTMLCanvasElement,
): Promise<WebGpuRenderer | null> {
  if (!('gpu' in navigator) || !navigator.gpu) {
    return null;
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });

  if (!adapter) return null;

  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu');

  if (!context) return null;

  const format = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format,
    alphaMode: 'opaque',
  });

  const shaderModule = device.createShaderModule({ code: shaderCode });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' },
      },
    ],
  });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [
        {
          format,
          blend: {
            color: {
              srcFactor: 'src-alpha',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-list' },
  });

  const state: RendererState = {
    device,
    context,
    pipeline,
    bindGroupLayout,
    uniformBuffer: device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    pointCapacity: 0,
    disposed: false,
  };

  return {
    render: (input) => renderWebGpu(state, input),
    dispose: () => {
      state.disposed = true;
      state.pointBuffer?.destroy();
      state.uniformBuffer.destroy();
      state.device.destroy();
    },
  };
}
