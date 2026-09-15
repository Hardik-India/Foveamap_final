    type GPUTextureFormat = string;

type GPUBufferBindingType = 'uniform' | 'storage' | 'read-only-storage';
type GPUTextureView = unknown;

declare const GPUBufferUsage: {
  readonly COPY_DST: number;
  readonly STORAGE: number;
  readonly UNIFORM: number;
};

declare const GPUShaderStage: {
  readonly FRAGMENT: number;
  readonly VERTEX: number;
};

type GPUBuffer = {
  destroy(): void;
};

type GPUQueue = {
  submit(commandBuffers: unknown[]): void;
  writeBuffer(
    buffer: GPUBuffer,
    bufferOffset: number,
    data: BufferSource,
    dataOffset?: number,
    size?: number,
  ): void;
};

type GPUDevice = {
  queue: GPUQueue;
  createBindGroup(descriptor: unknown): unknown;
  createBindGroupLayout(descriptor: unknown): GPUBindGroupLayout;
  createBuffer(descriptor: unknown): GPUBuffer;
  createCommandEncoder(): {
    beginRenderPass(descriptor: unknown): {
      setPipeline(pipeline: GPURenderPipeline): void;
      setBindGroup(index: number, bindGroup: unknown): void;
      draw(vertexCount: number, instanceCount?: number): void;
      end(): void;
    };
    finish(): unknown;
  };
  createPipelineLayout(descriptor: unknown): unknown;
  createRenderPipeline(descriptor: unknown): GPURenderPipeline;
  createShaderModule(descriptor: unknown): unknown;
  destroy(): void;
};

type GPUAdapter = {
  requestDevice(): Promise<GPUDevice>;
};

type GPU = {
  getPreferredCanvasFormat(): GPUTextureFormat;
  requestAdapter(options?: unknown): Promise<GPUAdapter | null>;
};

type GPUCanvasContext = {
  configure(configuration: unknown): void;
  getCurrentTexture(): {
    createView(): GPUTextureView;
  };
};

type GPUBindGroupLayout = unknown;
type GPURenderPipeline = unknown;

interface Navigator {
  gpu?: GPU;
}

interface HTMLCanvasElement {
  getContext(contextId: 'webgpu'): GPUCanvasContext | null;
}
