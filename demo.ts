import * as THREE from 'three/webgpu';
import { GUI } from 'lil-gui';
import StatsGL from 'stats-gl';
import StatsJS from 'stats.js';
import { createGround, setupLights, InputHandler } from './src/world.js';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FirstPersonControls } from 'three/examples/jsm/controls/FirstPersonControls.js';
import { AnimatedOctahedralImpostor } from './animated-octahedral.js';
import { InstancedAnimatedOctahedralImpostor } from './instanced-animated-octahedral.js';
import RAPIER from '@dimforge/rapier3d-compat';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// Initialize scene, camera, and renderer
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100000);

const renderer = new THREE.WebGPURenderer({ 
  antialias: true,
  requiredLimits: {
    maxStorageBufferBindingSize: 2147483644, // Request maximum supported size
    maxBufferSize: 2147483644
  }
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Physics and control state
let physics: { world: RAPIER.World; rigidBodies: Map<THREE.Object3D, RAPIER.RigidBody> };
let firstPersonControls: FirstPersonControls | null = null;
let inputHandler: InputHandler;
let lastTime = 0;

let groundMesh: THREE.Mesh | null = null;
let ambientLightRef: THREE.AmbientLight | null = null;
let dirLightRef: THREE.DirectionalLight | null = null;
let orbitControls: OrbitControls | null = null;
type ControlMode = 'FPS' | 'Orbit';
let controlMode: ControlMode = 'Orbit';
let controlModeController: any | null = null;
let controlModeKeyListenerAdded = false;
let onControlModeKeyDownRef: ((e: KeyboardEvent) => void) | null = null;
let skyMesh: THREE.Mesh | null = null;
let lastOrbitRadius = 15;
let fpsBaseMoveSpeed = 0;
let fpsSpeedMultiplier = 0;
const fpsAccelTime = 3.0; // seconds to reach full speed
const fpsDecelTime = 0.0; // instant stop when keys released
let fpsMoveKeyListenerAdded = false;
const fpsMoveKeys = new Set<string>();

function isFpsMoveKey(code: string): boolean {
  switch (code) {
    case 'KeyW': case 'KeyA': case 'KeyS': case 'KeyD':
    case 'ArrowUp': case 'ArrowLeft': case 'ArrowDown': case 'ArrowRight':
      return true;
    default:
      return false;
  }
}

function applyControlMode(next: ControlMode) {
  if (next === 'Orbit') {
    if (orbitControls) {
      // Compute target along current forward vector at the previous orbit radius
      const forward = new (THREE as any).Vector3();
      (camera as any).getWorldDirection(forward);
      // If we just came from Orbit, lastOrbitRadius is set. Otherwise, clamp to min/max.
      const minD = (orbitControls as any).minDistance ?? 0.1;
      const maxD = (orbitControls as any).maxDistance ?? 1e9;
      const radius = Math.min(Math.max(lastOrbitRadius || 15, minD), maxD);
      const newTarget = (camera as any).position.clone().add(forward.multiplyScalar(radius));
      orbitControls.target.copy(newTarget);
      orbitControls.enabled = true;
      // Keep camera position/orientation unchanged, just align look target
      orbitControls.update();
    }
    scene.add(camera);
  } else {
    // Switching to FPS: capture current orbit radius and seed FPS orientation to avoid snap
    if (orbitControls) {
      lastOrbitRadius = (camera as any).position.distanceTo(orbitControls.target);
      orbitControls.enabled = false;
    }
    if (firstPersonControls) {
      const forward = new (THREE as any).Vector3();
      (camera as any).getWorldDirection(forward);
      const theta = Math.atan2(forward.z, forward.x); // [-pi, pi]
      const phi = Math.acos(Math.min(1, Math.max(-1, forward.y))); // [0, pi]
      const lon = (THREE as any).MathUtils.radToDeg(theta);
      const lat = 90 - (THREE as any).MathUtils.radToDeg(phi);
      try {
        (firstPersonControls as any).lon = lon;
        (firstPersonControls as any).lat = lat;
        (firstPersonControls as any).update?.(0);
      } catch {}
    }
    // Reset movement ramp so forward starts calm
    fpsSpeedMultiplier = 0;
  }
  controlMode = next;
}

// Performance HUD state
let statsGL: any;
let statsJS: any;
let perfHud: HTMLDivElement | null = null;
let renderHud: HTMLDivElement | null = null;

// Per-frame render stats (fallback when renderer.info doesn't auto-reset)
// (removed) last frame totals; replaced by pre/post delta logic in animate()

// WebGPU draw counter state
let gpuCountersInstalled = false;
let wgpuFrameCalls = 0;
let wgpuFrameTriangles = 0;

function installWebGPUCountersOnce() {
  if (gpuCountersInstalled) return;
  try {
    const g: any = (globalThis as any);
    if (!g.GPUDevice || !g.GPUCommandEncoder) return;

    const pipelineTopologyMap = new (g.WeakMap || WeakMap)();
    const passLastPipeline = new (g.WeakMap || WeakMap)();

    const trianglesFromCount = (count: number, topology: string): number => {
      if (!count || count <= 0) return 0;
      switch (topology) {
        case 'triangle-list': return Math.floor(count / 3);
        case 'triangle-strip': return Math.max(0, count - 2);
        default: return 0; // points/lines
      }
    };

    // Wrap device.createRenderPipeline to capture primitive topology per pipeline
    const devProto = g.GPUDevice.prototype;
    const origCreateRP = devProto.createRenderPipeline;
    if (typeof origCreateRP === 'function') {
      devProto.createRenderPipeline = function(descriptor: any) {
        const pipeline = origCreateRP.call(this, descriptor);
        try {
          const topo = descriptor?.primitive?.topology || 'triangle-list';
          pipelineTopologyMap.set(pipeline, topo);
        } catch {}
        return pipeline;
      };
    }

    // Async variant
    const origCreateRPAsync = devProto.createRenderPipelineAsync;
    if (typeof origCreateRPAsync === 'function') {
      devProto.createRenderPipelineAsync = async function(descriptor: any) {
        const pipeline = await origCreateRPAsync.call(this, descriptor);
        try {
          const topo = descriptor?.primitive?.topology || 'triangle-list';
          pipelineTopologyMap.set(pipeline, topo);
        } catch {}
        return pipeline;
      };
    }

    // Wrap beginRenderPass to patch pass encoder draw methods
    const encProto = g.GPUCommandEncoder.prototype;
    const origBeginRP = encProto.beginRenderPass;
    if (typeof origBeginRP === 'function') {
      encProto.beginRenderPass = function(desc: any) {
        const pass: any = origBeginRP.call(this, desc);
        try {
          const origSetPipeline = pass.setPipeline?.bind(pass);
          if (typeof origSetPipeline === 'function') {
            pass.setPipeline = function(pipeline: any) {
              try { passLastPipeline.set(pass, pipeline); } catch {}
              return origSetPipeline(pipeline);
            };
          }

          const origDraw = pass.draw?.bind(pass);
          if (typeof origDraw === 'function') {
            pass.draw = function(vertexCount = 0, instanceCount = 1, firstVertex?: number, firstInstance?: number) {
              try {
                wgpuFrameCalls += 1;
                const pipeline = passLastPipeline.get(pass);
                const topo = pipelineTopologyMap.get(pipeline) || 'triangle-list';
                wgpuFrameTriangles += trianglesFromCount(vertexCount | 0, topo) * Math.max(1, instanceCount | 0);
              } catch {}
              return origDraw(vertexCount, instanceCount, firstVertex, firstInstance);
            };
          }

          const origDrawIndexed = pass.drawIndexed?.bind(pass);
          if (typeof origDrawIndexed === 'function') {
            pass.drawIndexed = function(indexCount = 0, instanceCount = 1, firstIndex?: number, baseVertex?: number, firstInstance?: number) {
              try {
                wgpuFrameCalls += 1;
                const pipeline = passLastPipeline.get(pass);
                const topo = pipelineTopologyMap.get(pipeline) || 'triangle-list';
                wgpuFrameTriangles += trianglesFromCount(indexCount | 0, topo) * Math.max(1, instanceCount | 0);
              } catch {}
              return origDrawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
            };
          }

          const origDrawIndirect = pass.drawIndirect?.bind(pass);
          if (typeof origDrawIndirect === 'function') {
            pass.drawIndirect = function(...args: any[]) {
              try { wgpuFrameCalls += 1; } catch {}
              return origDrawIndirect(...args);
            };
          }

          const origDrawIndexedIndirect = pass.drawIndexedIndirect?.bind(pass);
          if (typeof origDrawIndexedIndirect === 'function') {
            pass.drawIndexedIndirect = function(...args: any[]) {
              try { wgpuFrameCalls += 1; } catch {}
              return origDrawIndexedIndirect(...args);
            };
          }
        } catch {}
        return pass;
      };
    }

    gpuCountersInstalled = true;
  } catch {}
}

// Animated crowd scene state
let animatedImpostor: AnimatedOctahedralImpostor | null = null;
let instancedAnimated: InstancedAnimatedOctahedralImpostor | null = null;
let spawnedCrowdGroups: InstancedAnimatedOctahedralImpostor[] = [];
let areaPool: InstancedAnimatedOctahedralImpostor | null = null;
let areaPoolNextFree = 0;
let areaPoolCapacity = 200000; // prewarm capacity; adjustable in GUI
let crowdTexture: THREE.Texture | null = null; // albedo array
let crowdNormalTexture: THREE.Texture | null = null; // normals array (optional)
let animatedPlaying = true;
let animatedFPS = 24;
let animatedLastFrameTime = 0;
let animatedFrameCount = 1;
let animatedCurrentFrame = 0; // UI-visible value (modulo depends on variant)
let animatedFrameTicker = 0;   // monotonically increasing ticker for robust looping
let animatedKeyListenerAdded = false;
let guiRef: GUI | null = null;
let animatedFolder: any = null;
let animatedFrameController: any = null;
// Walking params (GPU compute) state
const walkingParams = {
  baseSpeed: 0.2, // 5x slower overall
  turnRate: 0.7,
  randomness: 1.0,
  yawSpriteOffsetDeg: 90,
  cycleAmp: 0.2, // stride-synced speed modulation amplitude (0..1)
  // Crowd avoidance defaults
  avoidanceRadius: 2.5, // scaled down 100x
  avoidanceStrength: 2.0,
  neighborSamples: 16,
  pushStrength: 0.02 // stronger push-out
};
let onAnimatedKeyDownRef: ((e: KeyboardEvent) => void) | null = null;
let animatedScale = 1;
let animatedYOffset = 0.24;
let animatedFiltering: 'nearest' | 'linear' = 'nearest';
let animatedSpritesPerSide = 12;
let paletteTexture: THREE.Texture | null = null;
let paletteSize = 32; // default
let paletteRows = 1;  // default
let paletteDataLinear: Float32Array | null = null; // LUT for fast palette reads
// Mapping from variant index (0..4) to valid palette row indices for that person
// Rows are 0-based in the image (top row = 0). Adjust if your PNG packs differently.
// Based on provided JSON description of rows 1..15, we map:
// 0 (ogre1): base=0, variants=[0,1,2,3]
// 1 (blueguy2): base=4, variants=[4,5]
// 2 (ninja3): base=6, variants=[6,7,8,9]
// 3 (whiteguy4): base=10, variants=[10,11]
// 4 (knight5): base=12, variants=[12,13,14] (note: width differs; palette rows still valid)
const VARIANT_TO_PALETTE_ROWS: number[][] = [
  [0, 1, 2, 3],
  [4, 5],
  [6, 7, 8, 9],
  [10, 11],
  [12, 13, 14]
];
let pendingCrowdTexture: THREE.Texture | null = null;
let pendingCrowdInfo: { width: number; height: number; layers: number } | null = null;
let pendingCrowdNormalTexture: THREE.Texture | null = null;
let pendingCrowdNormalInfo: { width: number; height: number; layers: number } | null = null;
let fileInputAlbedoEl: HTMLInputElement | null = null;
let fileInputNormalEl: HTMLInputElement | null = null;
let animatedFlipY = true;
let animatedAtlasFlipX = true;
let animatedAtlasFlipY = false;
let animatedAtlasSwapAxes = false;
// Attractor tool state
let attractorToolEnabled = false;
let attractorActive = false;
let attractorPos = { x: 0, z: 0 };
let attractorHoldTimer: number | null = null;
let attractorMouseDown = false;
let attractorPointerMoved = false;
let attractorDown = { x: 0, y: 0 };
const attractorLongPressMs = 1000;
const attractorMinMovePx = 6;
const attractorOptions = { radius: 200, turnBoost: 3.0, falloff: 2.0 };

// Area Spawn tool state
let areaToolEnabled = false;
let areaSelecting = false;
let areaStart = { x: 0, y: 0 };
let areaEnd = { x: 0, y: 0 };
let areaOverlay: HTMLDivElement | null = null;
let areaDensity = 0.02; // instances per square world unit
let areaYawDeg = 0; // uniform orientation for all spawned instances (deg)
let areaIncludeVariants = [true, true, true, true, true]; // 5 variants
let areaMaxPerSpawn = 100000;
let areaFreezePrevOrbit = false;

// Attractor banner (GLB) state
let warBannerRoot: THREE.Object3D | null = null;
let warBannerMixer: THREE.AnimationMixer | null = null;
let warBannerClips: THREE.AnimationClip[] = [];
let warBannerLoaded = false;
let warBannerDesiredHeight = 2.0;
let warBannerScaleFactor = 0.5; // halve normalized scale
let warBannerBaseYOffset = 0.0; // computed from bbox min.y
const gltfLoader = new GLTFLoader();

// Terrain height sampling to align objects (match GPU/ground function)
function sampleTerrainHeight(x: number, z: number, amp: number = 3.0, freq: number = 0.02): number {
  const f1 = freq;
  const f2 = f1 * 2.0;
  const f3 = f1 * 4.0;
  const f4 = f1 * 8.0;
  const a1 = 0.6;
  const a2 = 0.3;
  const a3 = 0.15;
  const a4 = 0.075;
  const y1 = Math.sin(x * f1) * 0.5 + Math.cos(z * f1) * 0.5;
  const y2 = Math.sin(x * f2) * 0.5 + Math.cos(z * f2) * 0.5;
  const y3 = Math.sin(x * f3) * 0.5 + Math.cos(z * f3) * 0.5;
  const y4 = Math.sin(x * f4) * 0.5 + Math.cos(z * f4) * 0.5;
  const sum = y1 * a1 + y2 * a2 + y3 * a3 + y4 * a4;
  const diag = Math.sin((x + z) * (f3 * 0.7071)) * 0.1;
  return (sum + diag) * amp;
}

async function loadWarBannerOnce(): Promise<void> {
  if (warBannerLoaded) return;
  const gltf = await gltfLoader.loadAsync('./textures/warbanner.glb');
  const root = gltf.scene || gltf.scenes?.[0];
  if (!root) return;
  // Compute bbox to normalize height
  const box = new (THREE as any).Box3().setFromObject(root);
  const size = new (THREE as any).Vector3();
  box.getSize(size);
  const currentHeight = Math.max(1e-6, size.y);
  const scaleMul = (warBannerDesiredHeight / currentHeight) * warBannerScaleFactor;
  root.scale.setScalar(scaleMul);
  // Recompute bbox post-scale to get base offset
  const box2 = new (THREE as any).Box3().setFromObject(root);
  warBannerBaseYOffset = -box2.min.y; // move so base rests on y=0
  // Ensure proper depth/occlusion on all materials
  root.traverse((obj: any) => {
    if (!obj || obj.isMesh !== true) return;
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const m of mats) {
      if (!m) continue;
      try {
        m.depthWrite = true;
        m.depthTest = true;
        if (m.transparent) {
          m.alphaTest = Math.max(m.alphaTest || 0, 0.5);
          m.transparent = false;
        }
        m.needsUpdate = true;
      } catch {}
    }
  });
  // Collect animations
  warBannerClips = gltf.animations || [];
  warBannerRoot = root;
  warBannerLoaded = true;
}

function destroyWarBanner(): void {
  if (warBannerMixer) { try { (warBannerMixer as any).stopAllAction?.(); } catch {} warBannerMixer = null; }
  if (warBannerRoot) { try { scene.remove(warBannerRoot); } catch {} warBannerRoot = null; }
}

async function spawnWarBannerAt(x: number, z: number): Promise<void> {
  await loadWarBannerOnce();
  if (!warBannerRoot) return;
  // If already added, just reposition; else add
  if (!scene.children.includes(warBannerRoot)) scene.add(warBannerRoot);
  const h = sampleTerrainHeight(x, z, 3.0, 0.02);
  warBannerRoot.position.set(x, warBannerBaseYOffset + h, z);
  // Setup/refresh mixer and play all clips looping
  if (!warBannerMixer) warBannerMixer = new (THREE as any).AnimationMixer(warBannerRoot);
  try { (warBannerMixer as any).stopAllAction?.(); } catch {}
  for (const clip of warBannerClips) {
    const action = (warBannerMixer as any).clipAction?.(clip, warBannerRoot);
    if (action) {
      try { action.reset(); action.setLoop((THREE as any).LoopRepeat, Infinity); action.play(); } catch {}
    }
  }
}

function coerceToDataArrayTexture(tex: THREE.Texture): THREE.Texture {
  try {
    const anyTex = tex as any;
    const img = anyTex.image;
    const width = (img?.width || 0) | 0;
    const height = (img?.height || 0) | 0;
    const data = img?.data as Uint8Array | Uint8ClampedArray | Float32Array | undefined;
    let layers = (img?.depth || img?.layers || anyTex.layers || 0) | 0;
    if (!layers && data && width && height) {
      // Try to infer layers for 4bpp or 1bpp payloads
      const denom4 = width * height * 4;
      const denom1 = width * height;
      const est4 = denom4 > 0 ? Math.floor((data.length || 0) / denom4) : 0;
      const est1 = denom1 > 0 ? Math.floor((data.length || 0) / denom1) : 0;
      if (est4 > 0 && (est4 * denom4) === (data.length || 0)) {
        layers = est4 | 0;
      } else if (est1 > 0 && (est1 * denom1) === (data.length || 0)) {
        layers = est1 | 0;
      }
    }
    if (layers > 1 && !(anyTex.isDataArrayTexture === true) && data && width && height) {
      // Detect channels per pixel to set proper format
      const denom = width * height * Math.max(1, layers);
      const cpp = denom > 0 ? Math.max(1, Math.floor((data.length || 0) / denom)) : 4;
      const isR8 = cpp === 1;
      const dtex = new (THREE as any).DataArrayTexture(data, width, height, layers);
      dtex.format = isR8 ? (THREE as any).RedFormat : (THREE as any).RGBAFormat;
      dtex.type = (THREE as any).UnsignedByteType;
      dtex.minFilter = (THREE as any).NearestFilter;
      dtex.magFilter = (THREE as any).NearestFilter;
      dtex.generateMipmaps = false;
      dtex.wrapS = (THREE as any).ClampToEdgeWrapping;
      dtex.wrapT = (THREE as any).ClampToEdgeWrapping;
      dtex.colorSpace = (THREE as any).NoColorSpace;
      dtex.needsUpdate = true;
      return dtex as THREE.Texture;
    }
  } catch {}
  return tex;
}

function toR8IndexDataArray(tex: THREE.Texture): THREE.Texture {
  try {
    const anyTex = tex as any;
    const img = anyTex.image;
    const width = (img?.width || 0) | 0;
    const height = (img?.height || 0) | 0;
    const data = img?.data as Uint8Array | undefined;
    let layers = (img?.depth || img?.layers || anyTex.layers || 0) | 0;
    if (!data || !width || !height) return tex;
    if (!layers) {
      // Infer layers for 1bpp or 4bpp
      const denom4 = width * height * 4;
      const denom1 = width * height;
      const est4 = denom4 > 0 ? Math.floor((data.length || 0) / denom4) : 0;
      const est1 = denom1 > 0 ? Math.floor((data.length || 0) / denom1) : 0;
      if (est4 > 0 && (est4 * denom4) === (data.length || 0)) {
        layers = est4 | 0;
      } else if (est1 > 0 && (est1 * denom1) === (data.length || 0)) {
        layers = est1 | 0;
      }
    }
    if (layers <= 1) return tex;

    const denom = width * height * Math.max(1, layers);
    const cpp = denom > 0 ? Math.max(1, Math.floor((data.length || 0) / denom)) : 4;

    if (cpp === 1) {
      // Already R8 array; just ensure proper flags and return
      try { anyTex.isDataArrayTexture = true; } catch {}
      try { delete (anyTex as any).isCompressedArrayTexture; } catch {}
      try {
        anyTex.format = (THREE as any).RedFormat;
        anyTex.type = (THREE as any).UnsignedByteType;
        anyTex.minFilter = (THREE as any).NearestFilter;
        anyTex.magFilter = (THREE as any).NearestFilter;
        anyTex.generateMipmaps = false;
        anyTex.wrapS = (THREE as any).ClampToEdgeWrapping;
        anyTex.wrapT = (THREE as any).ClampToEdgeWrapping;
        anyTex.colorSpace = (THREE as any).NoColorSpace;
        anyTex.needsUpdate = true;
      } catch {}
      return tex;
    }

    // Convert RGBA (or >=4 cpp) to R8 by extracting red
    const rOnly = new Uint8Array(width * height * layers);
    const step = Math.max(4, cpp);
    for (let i = 0, j = 0; i < data.length && j < rOnly.length; i += step, j++) rOnly[j] = data[i];
    const dtex = new (THREE as any).DataArrayTexture(rOnly, width, height, layers);
    dtex.format = (THREE as any).RedFormat;
    dtex.type = (THREE as any).UnsignedByteType;
    dtex.minFilter = (THREE as any).NearestFilter;
    dtex.magFilter = (THREE as any).NearestFilter;
    dtex.generateMipmaps = false;
    dtex.wrapS = (THREE as any).ClampToEdgeWrapping;
    dtex.wrapT = (THREE as any).ClampToEdgeWrapping;
    dtex.colorSpace = (THREE as any).NoColorSpace;
    dtex.needsUpdate = true;
    return dtex as THREE.Texture;
  } catch {
    return tex;
  }
}

function setupPerformanceHUD(rendererRef: any) {
  const hud = document.createElement('div');
  hud.className = 'perf-hud';
  hud.style.setProperty('--perf-top', '0px');
  hud.style.setProperty('--perf-left', '0px');
  hud.style.setProperty('--perf-scale', '1');
  hud.style.setProperty('--statsgl-left', '0px');
  hud.style.setProperty('--statsjs-left', '0px');
  hud.style.setProperty('--statsjs-top', '48px');
  document.body.appendChild(hud);
  perfHud = hud;

  // stats-gl
  statsGL = new (StatsGL as any)({
    trackGPU: true,
    trackHz: true,
    trackCPT: false,
    logsPerSecond: 4,
    graphsPerSecond: 30,
    samplesLog: 40,
    samplesGraph: 10,
    precision: 2,
    horizontal: true,
    minimal: false,
    mode: 0
  });
  statsGL.dom.removeAttribute('style');
  statsGL.dom.classList.add('perf-panel', 'statsgl-panel', 'stats-panel');
  perfHud.appendChild(statsGL.dom);
  statsGL.init(rendererRef);

  // stats.js memory panel
  statsJS = new (StatsJS as any)();
  statsJS.showPanel(2);
  statsJS.dom.removeAttribute('style');
  statsJS.dom.classList.add('perf-panel', 'statsjs-panel', 'stats-panel');
  perfHud.appendChild(statsJS.dom);

  // small custom draw/tri HUD
  const rh = document.createElement('div');
  rh.style.cssText = 'position:absolute;top:0;left:80px;z-index:1;background:rgba(0,0,0,0.8);color:#0f0;font:11px/1.2 monospace;padding:4px 6px;width:80px;pointer-events:none;white-space:pre;';
  rh.textContent = 'calls: 0\ntris: 0';
  perfHud.appendChild(rh);
  renderHud = rh;

  // Position draw/tris HUD flush with bottom of panels (max of StatsGL/StatsJS heights)
  const positionRenderHud = () => {
    if (!renderHud) return;
    const glH = (statsGL?.dom as HTMLElement)?.offsetHeight || 0;
    const jsH = (statsJS?.dom as HTMLElement)?.offsetHeight || 0;
    const h = Math.max(glH, jsH);
    renderHud.style.top = `${h}px`;
  };
  positionRenderHud();
  try {
    const RO: any = (window as any).ResizeObserver;
    if (RO) {
      const ro = new RO(positionRenderHud);
      if (statsGL?.dom) ro.observe(statsGL.dom);
      if (statsJS?.dom) ro.observe(statsJS.dom);
    }
  } catch {}
  window.addEventListener('resize', positionRenderHud);
}

function formatTris(n: number): string {
  const allowedDigits = 4;
  if (n < 10000) return n.toLocaleString();

  const formatWithUnit = (value: number, unit: 'k' | 'm' | 'b'): string => {
    const intDigits = value >= 1 ? Math.floor(Math.log10(value)) + 1 : 1;
    const maxFraction = Math.max(0, allowedDigits - intDigits);
    let s = value.toFixed(maxFraction);
    if (s.indexOf('.') !== -1) {
      s = s.replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
    }
    const intPart = s.split('.')[0];
    if (intPart.length > allowedDigits) {
      if (unit === 'k') return formatWithUnit(value / 1000, 'm');
      if (unit === 'm') return formatWithUnit(value / 1000, 'b');
    }
    return `${s}${unit}`;
  };

  if (n >= 1000000) {
    return formatWithUnit(n / 1000000, 'm');
  }
  return formatWithUnit(n / 1000, 'k');
}

function updateRenderHUD(calls: number, triangles: number) {
  if (!renderHud) return;
  renderHud.textContent = `calls: ${calls}\ntris: ${formatTris(triangles)}`;
}

function estimateTrianglesThisFrame(root: THREE.Object3D): number {
  let total = 0;
  root.traverseVisible((obj: any) => {
    const isMesh = obj && (obj.isMesh === true || obj.isInstancedMesh === true);
    if (!isMesh) return;
    const geometry = obj.geometry as any;
    if (!geometry) return;
    let trisPerDraw = 0;
    try {
      const indexCount = geometry.index?.count | 0;
      if (indexCount > 0) {
        trisPerDraw = indexCount / 3;
      } else {
        const posCount = geometry.attributes?.position?.count | 0;
        if (posCount > 0) trisPerDraw = posCount / 3;
      }
    } catch {}
    const instances = obj.isInstancedMesh === true ? (obj.count | 0) : 1;
    if (trisPerDraw > 0 && instances > 0) total += trisPerDraw * instances;
  });
  return total | 0;
}

function rebuildAnimatedImpostorFromCurrentTexture() {
  if (!crowdTexture) return;
  // Dispose previous impostor
  if (animatedImpostor) {
    scene.remove(animatedImpostor);
    animatedImpostor.geometry.dispose();
    (animatedImpostor.material as any).dispose?.();
    animatedImpostor = null;
  }
  // Ensure we have a true array texture if possible
  crowdTexture = coerceToDataArrayTexture(crowdTexture);
  crowdTexture = toR8IndexDataArray(crowdTexture);
  const texAny = crowdTexture as any;
  // Use max frames for variant 0 (person 1)
  animatedFrameCount = 48;
  animatedCurrentFrame = 0;
  animatedLastFrameTime = performance.now();
  // Ensure NodeMaterial treats this as a 2D array texture for .depth() sampling
  try {
    (texAny as any).isDataArrayTexture = true;
    try { delete (texAny as any).isCompressedArrayTexture; } catch {}
  } catch {}
  // Verify normals array compatibility if present
  if (crowdNormalTexture) {
    const nAny = crowdNormalTexture as any;
    const nLayers = nAny.image?.depth || nAny.image?.layers || nAny.layers || 1;
    if (nLayers !== animatedFrameCount) {
      console.warn('Normals array layer count differs from albedo array:', nLayers, 'vs', animatedFrameCount);
    }
    if ((nAny as any).isDataArrayTexture === true) {
      try { delete (nAny as any).isDataArrayTexture; } catch {}
    }
  }
  animatedImpostor = new AnimatedOctahedralImpostor(crowdTexture, {
    frameCount: animatedFrameCount,
    spritesPerSide: animatedSpritesPerSide,
    transparent: true,
    alphaClamp: 0.02,
    useHemiOctahedron: true,
    scale: animatedScale,
    flipY: animatedFlipY,
    flipSpriteX: animatedAtlasFlipX,
    flipSpriteY: animatedAtlasFlipY,
    swapSpriteAxes: animatedAtlasSwapAxes
  }, crowdNormalTexture);
  animatedImpostor.position.set(0, animatedYOffset * animatedScale, 0);
  scene.add(animatedImpostor);
  animatedImpostor.setFrame(0);
  applyCrowdFiltering(animatedFiltering);
  if (animatedFrameController) {
    try {
      animatedFrameController.min(0).max(Math.max(0, animatedFrameCount - 1)).setValue(0);
    } catch {}
  }
}

// Simplified configuration for Animated Crowd only
const config = {
  instanceCount: 500000,
  terrainSize: 300,
  cameraStartPos: { x: 0, y: 2.5, z: 7 },
  moveSpeed: 50,
  jumpVelocity: 50
};

// Performance HUD will be initialized in init()

// Texture loading state (skybox only)
let skyboxTexture: THREE.Texture | null = null;

async function init() {
  // Initialize WebGPU renderer
  await renderer.init();
  
  console.log('WebGPU renderer initialized with high storage buffer limits for 10M+ instances');

  // Setup combined StatsGL + StatsJS HUD and custom counters
  setupPerformanceHUD(renderer);
  installWebGPUCountersOnce();
  
  await RAPIER.init();
  
  physics = {
    world: new RAPIER.World({ x: 0, y: -9.81, z: 0 }),
    rigidBodies: new Map()
  };

  groundMesh = createGround(physics, config.terrainSize, { hillAmp: 3.0, hillFreq: 0.02 });
  scene.add(groundMesh);
  const lights = setupLights(scene) as any;
  ambientLightRef = lights?.ambientLight ?? null;
  dirLightRef = lights?.directionalLight ?? null;

  camera.position.set(config.cameraStartPos.x, config.cameraStartPos.y, config.cameraStartPos.z);
  firstPersonControls = new FirstPersonControls(camera as any, renderer.domElement);
  firstPersonControls.movementSpeed = config.moveSpeed * 0.4;
  firstPersonControls.lookSpeed = 0.1;
  fpsBaseMoveSpeed = firstPersonControls.movementSpeed;
  fpsSpeedMultiplier = 0;
  
  // Initialize OrbitControls but keep disabled by default
  orbitControls = new OrbitControls(camera as any, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.dampingFactor = 0.08;
  orbitControls.enabled = false;
  orbitControls.target.set(0, 2, 0);
  orbitControls.enablePan = true;
  orbitControls.enableZoom = true;
  orbitControls.zoomSpeed = 1.0;
  orbitControls.rotateSpeed = 0.8;
  orbitControls.screenSpacePanning = true;
  (orbitControls as any).minDistance = 0.1;
  (orbitControls as any).maxDistance = 200000;
  // More intuitive close-range orbiting
  try { (orbitControls as any).enableDollyToCursor = true; } catch {}
  try { (orbitControls as any).dollyToCursor = true; } catch {}
  try { (orbitControls as any).minPolarAngle = 0.01; } catch {}
  try { (orbitControls as any).maxPolarAngle = Math.PI - 0.01; } catch {}
  try { (orbitControls as any).panSpeed = 0.6; } catch {}

  inputHandler = new InputHandler();

  // If starting in Orbit mode, enable orbit controls immediately
  if (controlMode === 'Orbit' && orbitControls) {
    scene.add(camera);
    orbitControls.enabled = true;
    const target = (camera as any).position.clone();
    target.y = Math.max(0, target.y - 2);
    orbitControls.target.copy(target);
    const offset = new (THREE as any).Vector3(0, 5, 15);
    (camera as any).position.copy(target.clone().add(offset));
    (camera as any).lookAt(target);
    orbitControls.update();
  }
  
  // Optional skybox/environment
  const textureLoader = new THREE.TextureLoader();
  skyboxTexture = await new Promise<THREE.Texture>((resolve, reject) => 
    textureLoader.load('./textures/skybox.png', resolve, undefined, reject)
  );
  
  // Setup LDR skybox/environment from PNG
  skyboxTexture.colorSpace = (THREE as any).SRGBColorSpace;
  skyboxTexture.mapping = THREE.EquirectangularReflectionMapping;
  const pmremGenerator = new (THREE as any).PMREMGenerator(renderer);
  const envRT = pmremGenerator.fromEquirectangular(skyboxTexture as any);
  const envMap = envRT.texture;
  scene.environment = envMap; // prefiltered IBL
  scene.background = skyboxTexture; // crisp background
  pmremGenerator.dispose();
  // Warm-load banner early to avoid spawn hitch
  try { await loadWarBannerOnce(); } catch {}
  await createAnimatedScene();
  
  window.addEventListener('resize', onWindowResize);
  setupGUI();
  // Toggle camera mode with Shift+F
  if (!controlModeKeyListenerAdded) {
    onControlModeKeyDownRef = (e: KeyboardEvent) => {
      // Support both code and key for robustness across browsers
      const isF = e.code === 'KeyF' || e.key === 'f' || e.key === 'F';
      const isShift = e.shiftKey || (e.getModifierState && e.getModifierState('Shift'));
      if (isF && isShift) {
        const newMode: ControlMode = controlMode === 'Orbit' ? 'FPS' : 'Orbit';
        applyControlMode(newMode);
        // Reflect in GUI if present
        try { controlModeController?.setValue?.(newMode as any); } catch {}
        e.preventDefault();
      }
    };
    document.addEventListener('keydown', onControlModeKeyDownRef);
    controlModeKeyListenerAdded = true;
  }
  // Track movement keys for FPS ramp
  if (!fpsMoveKeyListenerAdded) {
    const onDown = (e: KeyboardEvent) => { if (isFpsMoveKey(e.code)) fpsMoveKeys.add(e.code); };
    const onUp = (e: KeyboardEvent) => { if (isFpsMoveKey(e.code)) fpsMoveKeys.delete(e.code); };
    document.addEventListener('keydown', onDown);
    document.addEventListener('keyup', onUp);
    fpsMoveKeyListenerAdded = true;
  }
  setupOrbitControlsEnhancements();
  setupAttractorInputHandlers();
  setupAreaSpawnInputHandlers();
  requestAnimationFrame(animate);
}



// (Removed) tree impostor scene and related functions

function recreateGround(sizeOverride?: number) {
  if (groundMesh) {
    // Remove old ground from scene and physics world
    scene.remove(groundMesh);
    const rigidBody = physics.rigidBodies.get(groundMesh);
    if (rigidBody) {
      physics.world.removeRigidBody(rigidBody);
      physics.rigidBodies.delete(groundMesh);
    }
    groundMesh.geometry.dispose();
    (groundMesh.material as THREE.Material).dispose();
  }

  // Create new ground with updated size
  const size = sizeOverride ?? config.terrainSize;
  groundMesh = createGround(physics, size, { hillAmp: 3.0, hillFreq: 0.02 });
  scene.add(groundMesh);
}

async function createAnimatedScene() {
  // Load KTX2 merged texture array (5 variants x 31 layers). Normals optional via GUI
  if (!crowdTexture) {
    const loader = new KTX2Loader()
      .setTranscoderPath('https://unpkg.com/three@0.180.0/examples/jsm/libs/basis/')
      .detectSupport(renderer);
    crowdTexture = await loader.loadAsync('./textures/merged.ktx2');
    loader.dispose();
  }
  // Coerce KTX2 to DataArrayTexture if the payload is an uncompressed RGBA8 array
  crowdTexture = coerceToDataArrayTexture(crowdTexture);
  crowdTexture = toR8IndexDataArray(crowdTexture);
  // Disable normals logic for now per request
  crowdNormalTexture = null;

  const texAny = crowdTexture as any;
  // Use max frames for variant 0 (person 1) since merged.ktx2 has varying counts
  animatedFrameCount = 48;
  animatedFPS = 30;
  animatedPlaying = true;
  animatedCurrentFrame = 0;
  animatedLastFrameTime = performance.now();

  // Ensure NodeMaterial treats this as a 2D array texture for .depth() sampling
  try {
    (texAny as any).isDataArrayTexture = true;
    try { delete (texAny as any).isCompressedArrayTexture; } catch {}
  } catch {}

  // Force index preservation: nearest, no mipmaps, no color space transform
  try {
    (crowdTexture as any).minFilter = (THREE as any).NearestMipmapNearestFilter;
    (crowdTexture as any).magFilter = (THREE as any).NearestFilter;
    (crowdTexture as any).generateMipmaps = false;
    (crowdTexture as any).colorSpace = (THREE as any).NoColorSpace;
    (crowdTexture as any).needsUpdate = true;
  } catch {}

  // Load shared palette (1px height, width = number of colors). Support grid (rows) too.
  if (!paletteTexture) {
    const tl = new THREE.TextureLoader();
    paletteTexture = await new Promise<THREE.Texture>((resolve, reject) =>
      tl.load('./textures/palette.png', resolve, undefined, reject)
    );
    const pAny = paletteTexture as any;
    pAny.minFilter = (THREE as any).NearestFilter;
    pAny.magFilter = (THREE as any).NearestFilter;
    pAny.generateMipmaps = false;
    pAny.wrapS = (THREE as any).ClampToEdgeWrapping;
    pAny.wrapT = (THREE as any).ClampToEdgeWrapping;
    pAny.colorSpace = (THREE as any).SRGBColorSpace; // authored palette colors
    paletteSize = (pAny.image?.width || 32) | 0;
    paletteRows = (pAny.image?.height || 1) | 0;
    pAny.needsUpdate = true;
    // Extract linear palette data for LUT buffer
    try {
      const img = pAny.image as HTMLImageElement | HTMLCanvasElement;
      const w = img?.width | 0;
      const h = img?.height | 0;
      if (w && h) {
        const canvas = document.createElement('canvas');
        canvas.width = w; canvas.height = h;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(img as any, 0, 0);
          const src = ctx.getImageData(0, 0, w, h).data;
          const out = new Float32Array(w * h * 4);
          const srgbToLinear = (c: number) => c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
          for (let i = 0, j = 0; i < src.length; i += 4, j += 4) {
            out[j + 0] = srgbToLinear(src[i + 0] / 255);
            out[j + 1] = srgbToLinear(src[i + 1] / 255);
            out[j + 2] = srgbToLinear(src[i + 2] / 255);
            out[j + 3] = src[i + 3] / 255;
          }
          paletteDataLinear = out;
        }
      }
    } catch {}
  }

  // Create billboard impostor
  animatedImpostor = new AnimatedOctahedralImpostor(crowdTexture, {
    frameCount: animatedFrameCount,
    spritesPerSide: animatedSpritesPerSide,
    transparent: true,
    alphaClamp: 0.02,
    useHemiOctahedron: true,
    scale: animatedScale,
    flipY: animatedFlipY,
    flipSpriteX: animatedAtlasFlipX,
    flipSpriteY: animatedAtlasFlipY,
    swapSpriteAxes: animatedAtlasSwapAxes,
    paletteTexture: paletteTexture!,
    paletteSize,
    paletteRows,
    paletteRowIndex: 0,
    paletteData: paletteDataLinear || undefined
  }, null);
  animatedImpostor.position.set(0, animatedYOffset * animatedScale, 0);
  scene.add(animatedImpostor);
  animatedImpostor.setFrame(0);

  // Apply initial filtering (nearest by default)
  applyCrowdFiltering(animatedFiltering);

  // Setup GUI controls (create now if GUI exists, else will be created after setupGUI)
  ensureAnimatedCrowdGUI();

  // Ensure instancing defaults are applied (enabled with 100k and density 1.0)
  try {
    // Create if not created by GUI yet
    if (!instancedAnimated) {
      instancedAnimated = new InstancedAnimatedOctahedralImpostor(crowdTexture, {
        instanceCount: Math.max(1, config.instanceCount | 0),
        useHemiOctahedron: true,
        spritesPerSide: animatedSpritesPerSide,
        transparent: true,
        alphaClamp: 0.02,
        scale: animatedScale,
        flipY: animatedFlipY,
        flipSpriteX: animatedAtlasFlipX,
        flipSpriteY: animatedAtlasFlipY,
        swapSpriteAxes: animatedAtlasSwapAxes,
        paletteTexture: paletteTexture!,
        paletteSize,
        paletteRows,
        paletteData: paletteDataLinear || undefined
      }, null, animatedFrameCount);
      scene.add(instancedAnimated);
    }
    // set total frame count for wrap logic and randomize per-instance offsets
    instancedAnimated.setFrameCount(animatedFrameCount);
    const terrainBounds = config.terrainSize * 1.0;
    instancedAnimated.generateRandomPositions(instancedAnimated.count, {
      minX: -terrainBounds * 0.5,
      maxX: terrainBounds * 0.5,
      minZ: -terrainBounds * 0.5,
      maxZ: terrainBounds * 0.5,
      y: animatedYOffset * animatedScale
    });
    try { (instancedAnimated as any).randomizeFrameOffsetsPerVariant?.(); } catch {}
    try {
      // Weight: person 1 is 10x less frequent than each of others
      (instancedAnimated as any).assignVariantsWeighted?.([1, 10, 10, 10, 10]);
    } catch {}
    try { (instancedAnimated as any).randomizePaletteRowsPerVariant?.(VARIANT_TO_PALETTE_ROWS, paletteRows); } catch {}
    // Start GPU walking update
    instancedAnimated.startWalking({
      minX: -terrainBounds * 0.5,
      maxX: terrainBounds * 0.5,
      minZ: -terrainBounds * 0.5,
      maxZ: terrainBounds * 0.5,
      y: animatedYOffset * animatedScale,
      baseSpeed: walkingParams.baseSpeed,
      turnRate: walkingParams.turnRate,
      randomness: walkingParams.randomness,
      cycleAmp: walkingParams.cycleAmp,
      avoidanceRadius: walkingParams.avoidanceRadius,
      avoidanceStrength: walkingParams.avoidanceStrength,
      neighborSamples: walkingParams.neighborSamples,
      pushStrength: walkingParams.pushStrength,
      terrainHillAmp: 3.0,
      terrainHillFreq: 0.02,
      gridMaxPerCell: 16,
      binPasses: 3
    });
    // Apply default yaw offset (deg)
    try { (instancedAnimated as any).setYawSpriteOffsetRadians?.(walkingParams.yawSpriteOffsetDeg * Math.PI / 180); } catch {}
  } catch {}

  // Space to toggle
  if (!animatedKeyListenerAdded) {
    onAnimatedKeyDownRef = (e: KeyboardEvent) => {
      // Only one scene now; Space always toggles
      if (e.code === 'Space') {
        animatedPlaying = !animatedPlaying;
        e.preventDefault();
      }
    };
    document.addEventListener('keydown', onAnimatedKeyDownRef);
    animatedKeyListenerAdded = true;
  }
}

function destroyAnimatedScene() {
  if (animatedImpostor) {
    scene.remove(animatedImpostor);
    animatedImpostor.geometry.dispose();
    (animatedImpostor.material as any).dispose?.();
    animatedImpostor = null;
  }
  if (instancedAnimated) {
    scene.remove(instancedAnimated);
    instancedAnimated.geometry.dispose();
    (instancedAnimated.material as any).dispose?.();
    instancedAnimated = null;
  }
  if (animatedFolder) { animatedFolder.destroy(); animatedFolder = null; animatedFrameController = null; }
  if (animatedKeyListenerAdded && onAnimatedKeyDownRef) {
    document.removeEventListener('keydown', onAnimatedKeyDownRef);
    animatedKeyListenerAdded = false;
    onAnimatedKeyDownRef = null;
  }
}

function applyCrowdFiltering(mode: 'nearest' | 'linear') {
  if (crowdTexture) {
    const tex = crowdTexture as any;
    if (mode === 'nearest') {
      tex.minFilter = (THREE as any).NearestFilter;
      tex.magFilter = (THREE as any).NearestFilter;
    } else {
      tex.minFilter = (THREE as any).LinearMipmapLinearFilter;
      tex.magFilter = (THREE as any).LinearFilter;
    }
    tex.needsUpdate = true;
  }
  if (crowdNormalTexture) {
    const ntex = crowdNormalTexture as any;
    if (mode === 'nearest') {
      ntex.minFilter = (THREE as any).NearestFilter;
      ntex.magFilter = (THREE as any).NearestFilter;
    } else {
      ntex.minFilter = (THREE as any).LinearMipmapLinearFilter;
      ntex.magFilter = (THREE as any).LinearFilter;
    }
    ntex.needsUpdate = true;
  }
}

function ensureAnimatedCrowdGUI() {
  if (!guiRef) return;
  if (animatedFolder) return;
  animatedFolder = guiRef.addFolder('Animated Crowd');
  animatedFolder.add({ playPause: () => { animatedPlaying = !animatedPlaying; } }, 'playPause').name('Play/Pause (Space)');
  animatedFolder.add({ fps: animatedFPS }, 'fps', 1, 120, 1).name('FPS').onChange((v: number) => { animatedFPS = v | 0; });
  animatedFrameController = animatedFolder.add({ frame: 0 }, 'frame', 0, Math.max(0, animatedFrameCount - 1), 1)
    .name('Frame')
    .onChange((v: number) => { animatedCurrentFrame = v | 0; animatedImpostor && animatedImpostor.setFrame(animatedCurrentFrame); });
  animatedFolder.add({ y: animatedYOffset }, 'y', 0, 20, 0.01).name('Y Offset').onChange((v: number) => {
    animatedYOffset = v;
    if (animatedImpostor) animatedImpostor.position.y = animatedYOffset * animatedScale;
  });
  animatedFolder.add({ scale: animatedScale }, 'scale', 0.1, 20000, 0.1).name('Scale').onChange((v: number) => {
    animatedScale = v;
    if (animatedImpostor) {
      animatedImpostor.setScale(animatedScale);
      animatedImpostor.position.y = animatedYOffset * animatedScale;
    }
  });
  animatedFolder.add({ filtering: animatedFiltering }, 'filtering', ['nearest', 'linear']).name('Filtering').onChange((v: 'nearest' | 'linear') => {
    animatedFiltering = v;
    applyCrowdFiltering(animatedFiltering);
  });

  // Instancing controls
  const instancingFolder = animatedFolder.addFolder('Instancing');
  const instancingState = {
    enableInstancing: true,
    instances: 1000,
    maxInstances: 1000000,
    regenerate: () => {},
    density: 1.0,
    showInstanced: true
  };
  instancingFolder.add(instancingState, 'enableInstancing').name('Enable Instancing').onChange((v: boolean) => {
    if (!crowdTexture) return;
    if (v) {
      // create instanced if not present
      if (!instancedAnimated) {
      instancedAnimated = new InstancedAnimatedOctahedralImpostor(crowdTexture!, {
          instanceCount: Math.max(1, Math.min(instancingState.instances | 0, instancingState.maxInstances)),
          useHemiOctahedron: true,
          spritesPerSide: animatedSpritesPerSide,
          transparent: true,
          alphaClamp: 0.02,
          scale: animatedScale,
          flipY: animatedFlipY,
          flipSpriteX: animatedAtlasFlipX,
          flipSpriteY: animatedAtlasFlipY,
          swapSpriteAxes: animatedAtlasSwapAxes,
          paletteTexture: paletteTexture!,
          paletteSize,
          paletteRows,
          paletteData: paletteDataLinear || undefined
        }, crowdNormalTexture);
        scene.add(instancedAnimated);
        // spread across terrain using density
        const terrainBounds = config.terrainSize * Math.max(0.01, instancingState.density);
        instancedAnimated.generateRandomPositions(instancingState.instances | 0, {
          minX: -terrainBounds * 0.5,
          maxX: terrainBounds * 0.5,
          minZ: -terrainBounds * 0.5,
          maxZ: terrainBounds * 0.5,
          y: animatedYOffset * animatedScale
        });
        try { (instancedAnimated as any).assignVariantsEqually?.(5); } catch {}
        try { (instancedAnimated as any).randomizePaletteRowsPerVariant?.(VARIANT_TO_PALETTE_ROWS, paletteRows); } catch {}
        instancedAnimated.startWalking({
          minX: -terrainBounds * 0.5,
          maxX: terrainBounds * 0.5,
          minZ: -terrainBounds * 0.5,
          maxZ: terrainBounds * 0.5,
          y: animatedYOffset * animatedScale,
          baseSpeed: walkingParams.baseSpeed,
          turnRate: walkingParams.turnRate,
          randomness: walkingParams.randomness,
          cycleAmp: walkingParams.cycleAmp,
          avoidanceRadius: walkingParams.avoidanceRadius,
          avoidanceStrength: walkingParams.avoidanceStrength,
          neighborSamples: walkingParams.neighborSamples,
          pushStrength: walkingParams.pushStrength,
          gridMaxPerCell: 16,
          binPasses: 3
        });
        try { (instancedAnimated as any).setYawSpriteOffsetRadians?.(walkingParams.yawSpriteOffsetDeg * Math.PI / 180); } catch {}
      }
    } else if (instancedAnimated) {
      scene.remove(instancedAnimated);
      instancedAnimated.geometry.dispose();
      (instancedAnimated.material as any).dispose?.();
      instancedAnimated = null;
    }
  });
  const instancesCtrl = instancingFolder.add(instancingState, 'instances', 1, 1000000, 1).name('Instances');
  instancesCtrl.onChange((v: number) => {
    if (instancedAnimated) {
      // Recreate with new count
      scene.remove(instancedAnimated);
      instancedAnimated.geometry.dispose();
      (instancedAnimated.material as any).dispose?.();
      instancedAnimated = new InstancedAnimatedOctahedralImpostor(crowdTexture!, {
        instanceCount: Math.max(1, Math.min(v | 0, instancingState.maxInstances)),
        useHemiOctahedron: true,
        spritesPerSide: animatedSpritesPerSide,
        transparent: true,
        alphaClamp: 0.02,
        scale: animatedScale,
        flipY: animatedFlipY,
        flipSpriteX: animatedAtlasFlipX,
        flipSpriteY: animatedAtlasFlipY,
        swapSpriteAxes: animatedAtlasSwapAxes,
        paletteTexture: paletteTexture!,
        paletteSize,
        paletteRows,
        paletteData: paletteDataLinear || undefined
      }, crowdNormalTexture, animatedFrameCount);
      scene.add(instancedAnimated);
      instancedAnimated.setFrameCount(animatedFrameCount);
      instancingState.regenerate();
      try { (instancedAnimated as any).randomizeFrameOffsetsPerVariant?.(); } catch {}
      try { (instancedAnimated as any).assignVariantsWeighted?.([1, 10, 10, 10, 10]); } catch {}
      const t2 = config.terrainSize * Math.max(0.01, instancingState.density);
      instancedAnimated.startWalking({
        minX: -t2 * 0.5,
        maxX: t2 * 0.5,
        minZ: -t2 * 0.5,
        maxZ: t2 * 0.5,
        y: animatedYOffset * animatedScale,
        baseSpeed: walkingParams.baseSpeed,
        turnRate: walkingParams.turnRate,
        randomness: walkingParams.randomness,
        cycleAmp: walkingParams.cycleAmp,
        avoidanceRadius: walkingParams.avoidanceRadius,
        avoidanceStrength: walkingParams.avoidanceStrength,
        neighborSamples: walkingParams.neighborSamples,
        pushStrength: walkingParams.pushStrength,
        terrainHillAmp: 3.0,
        terrainHillFreq: 0.02,
        gridMaxPerCell: 16,
        binPasses: 3
      });
      try { (instancedAnimated as any).setYawSpriteOffsetRadians?.(walkingParams.yawSpriteOffsetDeg * Math.PI / 180); } catch {}
    }
  });
  instancingFolder.add(instancingState, 'density', 0.01, 1.0, 0.01).name('Density').onChange(() => instancingState.regenerate());
  instancingFolder.add(instancingState, 'showInstanced').name('Show Instanced').onChange((v: boolean) => {
    if (instancedAnimated) instancedAnimated.visible = v;
  });
  instancingState.regenerate = () => {
    if (!instancedAnimated) return;
    const terrainBounds = config.terrainSize * Math.max(0.01, instancingState.density);
    instancedAnimated.generateRandomPositions(Math.max(1, instancingState.instances | 0), {
      minX: -terrainBounds * 0.5,
      maxX: terrainBounds * 0.5,
      minZ: -terrainBounds * 0.5,
      maxZ: terrainBounds * 0.5,
      y: animatedYOffset * animatedScale
    });
    try {
      instancedAnimated.startWalking({
        minX: -terrainBounds * 0.5,
        maxX: terrainBounds * 0.5,
        minZ: -terrainBounds * 0.5,
        maxZ: terrainBounds * 0.5,
        y: animatedYOffset * animatedScale,
        baseSpeed: walkingParams.baseSpeed,
        turnRate: walkingParams.turnRate,
        randomness: walkingParams.randomness,
        cycleAmp: walkingParams.cycleAmp,
        avoidanceRadius: walkingParams.avoidanceRadius,
        avoidanceStrength: walkingParams.avoidanceStrength,
        neighborSamples: walkingParams.neighborSamples,
        pushStrength: walkingParams.pushStrength,
        terrainHillAmp: 0.5,
        terrainHillFreq: 0.015,
        gridMaxPerCell: 16,
        binPasses: 3
      });
    } catch {}
  };

  // Walking controls
  const walkingFolder = animatedFolder.addFolder('Walking');
  walkingFolder.add(walkingParams, 'baseSpeed', 0.0, 1000.0, 0.1).name('Speed').onChange((v: number) => {
    walkingParams.baseSpeed = v;
    if (!instancedAnimated) return;
    const terrainBounds = config.terrainSize * Math.max(0.01, instancingState.density);
    instancedAnimated.startWalking({
      minX: -terrainBounds * 0.5,
      maxX: terrainBounds * 0.5,
      minZ: -terrainBounds * 0.5,
      maxZ: terrainBounds * 0.5,
      y: animatedYOffset * animatedScale,
      baseSpeed: walkingParams.baseSpeed,
      turnRate: walkingParams.turnRate,
      randomness: walkingParams.randomness,
      cycleAmp: walkingParams.cycleAmp,
      avoidanceRadius: walkingParams.avoidanceRadius,
      avoidanceStrength: walkingParams.avoidanceStrength,
      neighborSamples: walkingParams.neighborSamples,
      pushStrength: walkingParams.pushStrength
    });
  });
  walkingFolder.add(walkingParams, 'cycleAmp', 0.0, 1.0, 0.01).name('Cycle Amp').onChange((v: number) => {
    walkingParams.cycleAmp = v;
    if (!instancedAnimated) return;
    const terrainBounds = config.terrainSize * Math.max(0.01, instancingState.density);
    instancedAnimated.startWalking({
      minX: -terrainBounds * 0.5,
      maxX: terrainBounds * 0.5,
      minZ: -terrainBounds * 0.5,
      maxZ: terrainBounds * 0.5,
      y: animatedYOffset * animatedScale,
      baseSpeed: walkingParams.baseSpeed,
      turnRate: walkingParams.turnRate,
      randomness: walkingParams.randomness,
      cycleAmp: walkingParams.cycleAmp,
      avoidanceRadius: walkingParams.avoidanceRadius,
      avoidanceStrength: walkingParams.avoidanceStrength,
      neighborSamples: walkingParams.neighborSamples,
      pushStrength: walkingParams.pushStrength
    });
  });
  walkingFolder.add(walkingParams, 'turnRate', 0.0, 10.0, 0.01).name('Turn Rate').onChange((v: number) => {
    walkingParams.turnRate = v;
    if (!instancedAnimated) return;
    const terrainBounds = config.terrainSize * Math.max(0.01, instancingState.density);
    instancedAnimated.startWalking({
      minX: -terrainBounds * 0.5,
      maxX: terrainBounds * 0.5,
      minZ: -terrainBounds * 0.5,
      maxZ: terrainBounds * 0.5,
      y: animatedYOffset * animatedScale,
      baseSpeed: walkingParams.baseSpeed,
      turnRate: walkingParams.turnRate,
      randomness: walkingParams.randomness,
      cycleAmp: walkingParams.cycleAmp,
      avoidanceRadius: walkingParams.avoidanceRadius,
      avoidanceStrength: walkingParams.avoidanceStrength,
      neighborSamples: walkingParams.neighborSamples,
      pushStrength: walkingParams.pushStrength
    });
  });
  walkingFolder.add(walkingParams, 'randomness', 0.0, 2.0, 0.01).name('Randomness').onChange((v: number) => {
    walkingParams.randomness = v;
    if (!instancedAnimated) return;
    const terrainBounds = config.terrainSize * Math.max(0.01, instancingState.density);
    instancedAnimated.startWalking({
      minX: -terrainBounds * 0.5,
      maxX: terrainBounds * 0.5,
      minZ: -terrainBounds * 0.5,
      maxZ: terrainBounds * 0.5,
      y: animatedYOffset * animatedScale,
      baseSpeed: walkingParams.baseSpeed,
      turnRate: walkingParams.turnRate,
      randomness: walkingParams.randomness,
      cycleAmp: walkingParams.cycleAmp,
      avoidanceRadius: walkingParams.avoidanceRadius,
      avoidanceStrength: walkingParams.avoidanceStrength,
      neighborSamples: walkingParams.neighborSamples,
      pushStrength: walkingParams.pushStrength
    });
  });

  // Crowd avoidance GUI
  const avoidanceFolder = walkingFolder.addFolder('Crowd Avoidance');
  avoidanceFolder.add(walkingParams, 'avoidanceRadius', 0, 2000, 1).name('Radius').onChange((v: number) => {
    walkingParams.avoidanceRadius = v;
    if (!instancedAnimated) return;
    const terrainBounds = config.terrainSize * Math.max(0.01, instancingState.density);
    instancedAnimated.startWalking({
      minX: -terrainBounds * 0.5,
      maxX: terrainBounds * 0.5,
      minZ: -terrainBounds * 0.5,
      maxZ: terrainBounds * 0.5,
      y: animatedYOffset * animatedScale,
      baseSpeed: walkingParams.baseSpeed,
      turnRate: walkingParams.turnRate,
      randomness: walkingParams.randomness,
      cycleAmp: walkingParams.cycleAmp,
      avoidanceRadius: walkingParams.avoidanceRadius,
      avoidanceStrength: walkingParams.avoidanceStrength,
      neighborSamples: walkingParams.neighborSamples,
      pushStrength: walkingParams.pushStrength
    });
  });
  avoidanceFolder.add(walkingParams, 'avoidanceStrength', 0, 5, 0.01).name('Strength').onChange((v: number) => {
    walkingParams.avoidanceStrength = v;
    if (!instancedAnimated) return;
    const terrainBounds = config.terrainSize * Math.max(0.01, instancingState.density);
    instancedAnimated.startWalking({
      minX: -terrainBounds * 0.5,
      maxX: terrainBounds * 0.5,
      minZ: -terrainBounds * 0.5,
      maxZ: terrainBounds * 0.5,
      y: animatedYOffset * animatedScale,
      baseSpeed: walkingParams.baseSpeed,
      turnRate: walkingParams.turnRate,
      randomness: walkingParams.randomness,
      cycleAmp: walkingParams.cycleAmp,
      avoidanceRadius: walkingParams.avoidanceRadius,
      avoidanceStrength: walkingParams.avoidanceStrength,
      neighborSamples: walkingParams.neighborSamples,
      pushStrength: walkingParams.pushStrength
    });
  });
  avoidanceFolder.add(walkingParams, 'neighborSamples', 1, 64, 1).name('Samples').onChange((v: number) => {
    walkingParams.neighborSamples = v | 0;
    if (!instancedAnimated) return;
    const terrainBounds = config.terrainSize * Math.max(0.01, instancingState.density);
    instancedAnimated.startWalking({
      minX: -terrainBounds * 0.5,
      maxX: terrainBounds * 0.5,
      minZ: -terrainBounds * 0.5,
      maxZ: terrainBounds * 0.5,
      y: animatedYOffset * animatedScale,
      baseSpeed: walkingParams.baseSpeed,
      turnRate: walkingParams.turnRate,
      randomness: walkingParams.randomness,
      avoidanceRadius: walkingParams.avoidanceRadius,
      avoidanceStrength: walkingParams.avoidanceStrength,
      neighborSamples: walkingParams.neighborSamples,
      pushStrength: walkingParams.pushStrength
    });
  });
  avoidanceFolder.add(walkingParams, 'pushStrength', 0, 5, 0.01).name('Push-out').onChange((v: number) => {
    walkingParams.pushStrength = v;
    if (!instancedAnimated) return;
    const terrainBounds = config.terrainSize * Math.max(0.01, instancingState.density);
    instancedAnimated.startWalking({
      minX: -terrainBounds * 0.5,
      maxX: terrainBounds * 0.5,
      minZ: -terrainBounds * 0.5,
      maxZ: terrainBounds * 0.5,
      y: animatedYOffset * animatedScale,
      baseSpeed: walkingParams.baseSpeed,
      turnRate: walkingParams.turnRate,
      randomness: walkingParams.randomness,
      avoidanceRadius: walkingParams.avoidanceRadius,
      avoidanceStrength: walkingParams.avoidanceStrength,
      neighborSamples: walkingParams.neighborSamples,
      pushStrength: walkingParams.pushStrength
    });
  });
  walkingFolder.add(walkingParams, 'yawSpriteOffsetDeg', -180, 180, 1).name('Yaw Offset (deg)').onChange((v: number) => {
    walkingParams.yawSpriteOffsetDeg = v | 0;
    if (instancedAnimated) (instancedAnimated as any).setYawSpriteOffsetRadians?.(v * Math.PI / 180);
  });
  // Apply default yaw offset immediately
  try { if (instancedAnimated) (instancedAnimated as any).setYawSpriteOffsetRadians?.(walkingParams.yawSpriteOffsetDeg * Math.PI / 180); } catch {}

  // Import KTX2 subfolders
  const importFolder = animatedFolder.addFolder('Import KTX2 (Albedo)');
  const importState = {
    chooseFile: () => {},
    apply: () => {},
    resolution: 'nearest' as 'nearest' | 'linear',
    framesPerSide: animatedSpritesPerSide,
    flipY: animatedFlipY,
    flipAtlasX: animatedAtlasFlipX,
    flipAtlasY: animatedAtlasFlipY,
    swapAtlasAxes: animatedAtlasSwapAxes,
    info: 'No file loaded'
  };

  // Create (or reuse) hidden file input for albedo
  if (!fileInputAlbedoEl) {
    fileInputAlbedoEl = document.createElement('input');
    fileInputAlbedoEl.type = 'file';
    fileInputAlbedoEl.accept = '.ktx2';
    fileInputAlbedoEl.style.display = 'none';
    document.body.appendChild(fileInputAlbedoEl);
    fileInputAlbedoEl.addEventListener('change', async () => {
      const file = fileInputAlbedoEl!.files?.[0];
      if (!file) return;
      const objectURL = URL.createObjectURL(file);
      try {
        const loader = new KTX2Loader()
          .setTranscoderPath('https://unpkg.com/three@0.180.0/examples/jsm/libs/basis/')
          .detectSupport(renderer);
      const tex = await loader.loadAsync(objectURL);
        loader.dispose();
        URL.revokeObjectURL(objectURL);
        // Ensure WebGPU array semantics if needed
        const texAny = tex as any;
        const layers = texAny.image?.depth || texAny.image?.layers || texAny.layers || 1;
        try {
          (texAny as any).isDataArrayTexture = true;
          try { delete (texAny as any).isCompressedArrayTexture; } catch {}
        } catch {}
        pendingCrowdTexture = coerceToDataArrayTexture(tex as THREE.Texture);
        pendingCrowdTexture = toR8IndexDataArray(pendingCrowdTexture);
        pendingCrowdInfo = { width: texAny.image?.width || 0, height: texAny.image?.height || 0, layers: layers | 0 };
        importState.info = `${pendingCrowdInfo.width}x${pendingCrowdInfo.height}, layers: ${pendingCrowdInfo.layers}`;
        infoCtrl.updateDisplay();
      } catch (e) {
        console.error('Failed to load KTX2', e);
        importState.info = 'Failed to load file';
        infoCtrl.updateDisplay();
      }
    });
  }

  importState.chooseFile = () => fileInputAlbedoEl!.click();
  importState.apply = () => {
    if (!pendingCrowdTexture) return;
    // Dispose old texture if any
    if (crowdTexture && crowdTexture !== pendingCrowdTexture) {
      (crowdTexture as any).dispose?.();
    }
    crowdTexture = pendingCrowdTexture;
    pendingCrowdTexture = null;
    // Apply UI-selected params
    animatedSpritesPerSide = importState.framesPerSide | 0;
    animatedFiltering = importState.resolution;
    applyCrowdFiltering(animatedFiltering);
    // Rebuild impostor with new texture
    rebuildAnimatedImpostorFromCurrentTexture();
    // Recreate instanced with new array and randomized offsets
    if (instancedAnimated) {
      scene.remove(instancedAnimated);
      instancedAnimated.geometry.dispose();
      (instancedAnimated.material as any).dispose?.();
      instancedAnimated = new InstancedAnimatedOctahedralImpostor(crowdTexture!, {
        instanceCount: instancedAnimated.count,
        useHemiOctahedron: true,
        spritesPerSide: animatedSpritesPerSide,
        transparent: true,
        alphaClamp: 0.02,
        scale: animatedScale,
        flipY: animatedFlipY,
        flipSpriteX: animatedAtlasFlipX,
        flipSpriteY: animatedAtlasFlipY,
        swapSpriteAxes: animatedAtlasSwapAxes,
        paletteTexture: paletteTexture!,
        paletteSize,
        paletteRows,
        paletteData: paletteDataLinear || undefined
      }, crowdNormalTexture, animatedFrameCount);
      scene.add(instancedAnimated);
      instancedAnimated.setFrameCount(animatedFrameCount);
      try { (instancedAnimated as any).randomizeFrameOffsetsPerVariant?.(); } catch {}
      try { (instancedAnimated as any).assignVariantsWeighted?.([1, 10, 10, 10, 10]); } catch {}
      try { (instancedAnimated as any).randomizePaletteRowsPerVariant?.(VARIANT_TO_PALETTE_ROWS, paletteRows); } catch {}
    const terrainBounds = config.terrainSize * Math.max(0.01, 1.0);
      instancedAnimated.generateRandomPositions(instancedAnimated.count, {
        minX: -terrainBounds * 0.5,
        maxX: terrainBounds * 0.5,
        minZ: -terrainBounds * 0.5,
        maxZ: terrainBounds * 0.5,
        y: animatedYOffset * animatedScale
      });
      instancedAnimated.startWalking({
        minX: -terrainBounds * 0.5,
        maxX: terrainBounds * 0.5,
        minZ: -terrainBounds * 0.5,
        maxZ: terrainBounds * 0.5,
        y: animatedYOffset * animatedScale,
        baseSpeed: walkingParams.baseSpeed,
        turnRate: walkingParams.turnRate,
        randomness: walkingParams.randomness,
        avoidanceRadius: walkingParams.avoidanceRadius,
        avoidanceStrength: walkingParams.avoidanceStrength,
        neighborSamples: walkingParams.neighborSamples,
        pushStrength: walkingParams.pushStrength,
        terrainHillAmp: 0.5,
        terrainHillFreq: 0.015
      });
      try { (instancedAnimated as any).setYawSpriteOffsetRadians?.(walkingParams.yawSpriteOffsetDeg * Math.PI / 180); } catch {}
    }
  };

  importFolder.add(importState, 'chooseFile').name('Choose .ktx2 file');
  const infoCtrl = importFolder.add(importState, 'info').name('Detected');
  importFolder.add(importState, 'resolution', { Nearest: 'nearest', Linear: 'linear' }).name('Resolution');
  importFolder.add(importState, 'framesPerSide', [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]).name('Frames per side').onChange((v: number) => {
    animatedSpritesPerSide = v | 0;
    if (animatedImpostor) animatedImpostor.setSpritesPerSide(animatedSpritesPerSide);
  });
  importFolder.add(importState, 'flipY').name('Flip Y').onChange((v: boolean) => {
    animatedFlipY = !!v;
    if (animatedImpostor) (animatedImpostor as any).setFlipY(animatedFlipY);
  });
  importFolder.add(importState, 'flipAtlasX').name('Flip Sprite X').onChange((v: boolean) => {
    animatedAtlasFlipX = !!v;
    if (animatedImpostor) (animatedImpostor as any).setAtlasFlipAndSwap({ flipX: animatedAtlasFlipX });
  });
  importFolder.add(importState, 'flipAtlasY').name('Flip Sprite Y').onChange((v: boolean) => {
    animatedAtlasFlipY = !!v;
    if (animatedImpostor) (animatedImpostor as any).setAtlasFlipAndSwap({ flipY: animatedAtlasFlipY });
  });
  importFolder.add(importState, 'swapAtlasAxes').name('Swap Sprite Axes').onChange((v: boolean) => {
    animatedAtlasSwapAxes = !!v;
    if (animatedImpostor) (animatedImpostor as any).setAtlasFlipAndSwap({ swapAxes: animatedAtlasSwapAxes });
  });
  importFolder.add(importState, 'apply').name('Apply');

  // Normals import subfolder
  const normalsFolder = animatedFolder.addFolder('Import KTX2 (Normals)');
  const normalsState = {
    chooseFile: () => {},
    apply: () => {},
    info: 'No normals file loaded'
  };

  if (!fileInputNormalEl) {
    fileInputNormalEl = document.createElement('input');
    fileInputNormalEl.type = 'file';
    fileInputNormalEl.accept = '.ktx2';
    fileInputNormalEl.style.display = 'none';
    document.body.appendChild(fileInputNormalEl);
    fileInputNormalEl.addEventListener('change', async () => {
      const file = fileInputNormalEl!.files?.[0];
      if (!file) return;
      const objectURL = URL.createObjectURL(file);
      try {
        const loader = new KTX2Loader()
          .setTranscoderPath('https://unpkg.com/three@0.180.0/examples/jsm/libs/basis/')
          .detectSupport(renderer);
        const tex = await loader.loadAsync(objectURL);
        loader.dispose();
        URL.revokeObjectURL(objectURL);
        const texAny = tex as any;
        const layers = texAny.image?.depth || texAny.image?.layers || texAny.layers || 1;
        if ((texAny as any).isDataArrayTexture === true) {
          try { delete (texAny as any).isDataArrayTexture; } catch {}
        }
        pendingCrowdNormalTexture = tex as THREE.Texture;
        pendingCrowdNormalInfo = { width: texAny.image?.width || 0, height: texAny.image?.height || 0, layers: layers | 0 };
        normalsState.info = `${pendingCrowdNormalInfo.width}x${pendingCrowdNormalInfo.height}, layers: ${pendingCrowdNormalInfo.layers}`;
        normalsInfoCtrl.updateDisplay();
      } catch (e) {
        console.error('Failed to load KTX2', e);
        normalsState.info = 'Failed to load file';
        normalsInfoCtrl.updateDisplay();
      }
    });
  }

  normalsState.chooseFile = () => fileInputNormalEl!.click();
  normalsState.apply = () => {
    if (!pendingCrowdNormalTexture) return;
    if (crowdNormalTexture && crowdNormalTexture !== pendingCrowdNormalTexture) {
      (crowdNormalTexture as any).dispose?.();
    }
    crowdNormalTexture = pendingCrowdNormalTexture;
    pendingCrowdNormalTexture = null;
    applyCrowdFiltering(animatedFiltering);
    rebuildAnimatedImpostorFromCurrentTexture();
    if (instancedAnimated) {
      // recreate material to inject normals
      scene.remove(instancedAnimated);
      instancedAnimated.geometry.dispose();
      (instancedAnimated.material as any).dispose?.();
      instancedAnimated = new InstancedAnimatedOctahedralImpostor(crowdTexture!, {
        instanceCount: instancedAnimated.count,
        useHemiOctahedron: true,
        spritesPerSide: animatedSpritesPerSide,
        transparent: true,
        alphaClamp: 0.02,
        scale: animatedScale,
        flipY: animatedFlipY,
        flipSpriteX: animatedAtlasFlipX,
        flipSpriteY: animatedAtlasFlipY,
        swapSpriteAxes: animatedAtlasSwapAxes,
        paletteTexture: paletteTexture!,
        paletteSize,
        paletteRows,
        paletteData: paletteDataLinear || undefined
      }, crowdNormalTexture, animatedFrameCount);
      scene.add(instancedAnimated);
      instancedAnimated.setFrameCount(animatedFrameCount);
      try { (instancedAnimated as any).randomizeFrameOffsetsPerVariant?.(); } catch {}
      try { (instancedAnimated as any).assignVariantsWeighted?.([1, 10, 10, 10, 10]); } catch {}
      try { (instancedAnimated as any).randomizePaletteRowsPerVariant?.(VARIANT_TO_PALETTE_ROWS, paletteRows); } catch {}
      const terrainBounds = config.terrainSize * 1.0;
      instancedAnimated.generateRandomPositions(instancedAnimated.count, {
        minX: -terrainBounds * 0.5,
        maxX: terrainBounds * 0.5,
        minZ: -terrainBounds * 0.5,
        maxZ: terrainBounds * 0.5,
        y: animatedYOffset * animatedScale
      });
      instancedAnimated.startWalking({
        minX: -terrainBounds * 0.5,
        maxX: terrainBounds * 0.5,
        minZ: -terrainBounds * 0.5,
        maxZ: terrainBounds * 0.5,
        y: animatedYOffset * animatedScale,
        baseSpeed: walkingParams.baseSpeed,
        turnRate: walkingParams.turnRate,
        randomness: walkingParams.randomness,
        avoidanceRadius: walkingParams.avoidanceRadius,
        avoidanceStrength: walkingParams.avoidanceStrength,
        neighborSamples: walkingParams.neighborSamples,
        pushStrength: walkingParams.pushStrength,
        terrainHillAmp: 3.0,
        terrainHillFreq: 0.02
      });
      try { (instancedAnimated as any).setYawSpriteOffsetRadians?.(walkingParams.yawSpriteOffsetDeg * Math.PI / 180); } catch {}
    }
  };

  normalsFolder.add(normalsState, 'chooseFile').name('Choose normals .ktx2 file');
  const normalsInfoCtrl = normalsFolder.add(normalsState, 'info').name('Detected');
  normalsFolder.add(normalsState, 'apply').name('Apply Normals');

  // Turn on instancing by default in UI sense (will be created already in scene
  try {
    // If GUI exists, reflect defaults
    const instancingFolder = animatedFolder.folders?.find?.((f: any) => f._title === 'Instancing');
    if (instancingFolder) {
      // Best-effort: if controller exists, set values
      // Not relying on private internals; the actual scene already created instances above
    }
  } catch {}
  animatedFolder.open();

  // Palette controls
  const paletteFolder = animatedFolder.addFolder('Palette');
  paletteFolder.add({ distribute: () => { try { (instancedAnimated as any)?.assignVariantsWeighted?.([1, 10, 10, 10, 10]); } catch {} } }, 'distribute').name('Distribute Weighted (1:10:10:10:10)');
  paletteFolder.add({ setRow0: () => {
    if (!instancedAnimated) return;
    for (let i = 0; i < instancedAnimated.count; i++) (instancedAnimated as any).setPaletteRow?.(i, 0, paletteRows);
  } }, 'setRow0').name('Set All Row 0');
  paletteFolder.add({ setRow1: () => {
    if (!instancedAnimated) return;
    for (let i = 0; i < instancedAnimated.count; i++) (instancedAnimated as any).setPaletteRow?.(i, Math.min(1, paletteRows - 1), paletteRows);
  } }, 'setRow1').name('Set All Row 1');

  // Area Spawn controls (drag rectangle to spawn instanced people)
  const areaFolder = animatedFolder.addFolder('Area Spawn');
  const areaGUI = {
    enable: false,
    density: areaDensity,
    yawDeg: areaYawDeg,
    person1: areaIncludeVariants[0],
    person2: areaIncludeVariants[1],
    person3: areaIncludeVariants[2],
    person4: areaIncludeVariants[3],
    person5: areaIncludeVariants[4],
    maxPerSpawn: areaMaxPerSpawn,
    clearLast: () => {
      const g = spawnedCrowdGroups.pop();
      if (!g) return;
      try { scene.remove(g); g.geometry.dispose(); (g.material as any).dispose?.(); } catch {}
    },
    clearAll: () => {
      for (const g of spawnedCrowdGroups) { try { scene.remove(g); g.geometry.dispose(); (g.material as any).dispose?.(); } catch {} }
      spawnedCrowdGroups.length = 0;
    }
  };
  areaFolder.add(areaGUI, 'enable').name('Enable Area Tool').onChange((v: boolean) => {
    areaToolEnabled = !!v;
    // Freeze orbit camera when toggled
    if (orbitControls) {
      if (v) {
        areaFreezePrevOrbit = orbitControls.enabled;
        orbitControls.enabled = false;
        prewarmAreaPool();
      } else {
        orbitControls.enabled = areaFreezePrevOrbit;
      }
    }
  });
  areaFolder.add(areaGUI, 'density', 0.001, 10.0, 0.001).name('Density (/u^2)').onChange((v: number) => { areaDensity = Math.max(0.001, v); });
  areaFolder.add(areaGUI, 'yawDeg', -180, 180, 1).name('Yaw (deg)').onChange((v: number) => { areaYawDeg = v | 0; }).disable();
  areaFolder.add(areaGUI, 'person1').name('Include Person 1').onChange((v: boolean) => { areaIncludeVariants[0] = !!v; });
  areaFolder.add(areaGUI, 'person2').name('Include Person 2').onChange((v: boolean) => { areaIncludeVariants[1] = !!v; });
  areaFolder.add(areaGUI, 'person3').name('Include Person 3').onChange((v: boolean) => { areaIncludeVariants[2] = !!v; });
  areaFolder.add(areaGUI, 'person4').name('Include Person 4').onChange((v: boolean) => { areaIncludeVariants[3] = !!v; });
  areaFolder.add(areaGUI, 'person5').name('Include Person 5').onChange((v: boolean) => { areaIncludeVariants[4] = !!v; });
  areaFolder.add(areaGUI, 'maxPerSpawn', 1, 1000000, 1).name('Max per Spawn').onChange((v: number) => { areaMaxPerSpawn = Math.max(1, v | 0); });
  areaFolder.add(areaGUI, 'clearLast').name('Clear Last Group');
  areaFolder.add(areaGUI, 'clearAll').name('Clear All Groups');
  areaFolder.add({ capacity: areaPoolCapacity }, 'capacity', 1000, 2000000, 1000).name('Pool Capacity').onChange((v: number) => { areaPoolCapacity = Math.max(1, v | 0); if (areaToolEnabled) prewarmAreaPool(true); });
}

function setupGUI() {
  const gui = new GUI();
  guiRef = gui;
  
  // Align GUI to right edge with no gap
  gui.domElement.style.position = 'fixed';
  gui.domElement.style.right = '0';
  gui.domElement.style.top = '0';
  
  // Terrain Controls
  const terrainFolder = gui.addFolder('Terrain');
  const terrainSizeController = terrainFolder.add(config, 'terrainSize', 1, 200000, 1).name('Terrain Size').onChange((v: number) => {
    // Ensure config is in sync and ground rebuild uses current value
    config.terrainSize = v;
    recreateGround(v);
  });
  
  // Ensure the controller reflects the current value
  terrainSizeController.min(1).max(200000).step(1);
  terrainSizeController.setValue(config.terrainSize);
  ensureAnimatedCrowdGUI();
  
  // Display Controls
  const displayConfig = {
    showInstancedTrees: false
  };
  
  const displayFolder = gui.addFolder('Display');
  // (trees removed)

  // Attractor tool GUI (animated crowd only)
  const attractorFolder = gui.addFolder('Attractor');
  const attractorGUI = {
    enable: false,
    radius: attractorOptions.radius,
    strength: attractorOptions.turnBoost,
    falloff: attractorOptions.falloff,
    yeet: false,
    arrivalDist: 0.65,
    yeetSpeed: 14.0,
    horizFrac: 0.35,
    life: 3.0,
    gravity: 9.81,
    spin: 12.0,
    clear: () => {
      if (instancedAnimated) (instancedAnimated as any).setAttractor?.({ enabled: false, x: 0, z: 0 });
      attractorActive = false;
      if (instancedAnimated) {
        try { (instancedAnimated as any).setYeetEnabled?.(false); (instancedAnimated as any).resetYeetStates?.(); } catch {}
      }
      destroyWarBanner();
    }
  };
  attractorFolder.add(attractorGUI, 'enable').name('Enable (hold LMB 1s)').onChange((v: boolean) => {
    attractorToolEnabled = !!v;
    if (!v && instancedAnimated) {
      (instancedAnimated as any).setAttractor?.({ enabled: false, x: 0, z: 0 });
      attractorActive = false;
      try { (instancedAnimated as any).setYeetEnabled?.(false); (instancedAnimated as any).resetYeetStates?.(); } catch {}
      destroyWarBanner();
    }
  });
  attractorFolder.add(attractorGUI, 'radius', 1, 5000, 1).name('Radius').onChange((v: number) => {
    attractorOptions.radius = v;
    if (instancedAnimated && attractorActive) (instancedAnimated as any).setAttractor?.({ enabled: true, x: attractorPos.x, z: attractorPos.z, radius: attractorOptions.radius, turnBoost: attractorOptions.turnBoost, falloff: attractorOptions.falloff });
  });
  attractorFolder.add(attractorGUI, 'strength', 0, 10, 0.01).name('Turn Boost').onChange((v: number) => {
    attractorOptions.turnBoost = v;
    if (instancedAnimated && attractorActive) (instancedAnimated as any).setAttractor?.({ enabled: true, x: attractorPos.x, z: attractorPos.z, radius: attractorOptions.radius, turnBoost: attractorOptions.turnBoost, falloff: attractorOptions.falloff });
  });
  attractorFolder.add(attractorGUI, 'falloff', 0.1, 4.0, 0.1).name('Falloff').onChange((v: number) => {
    attractorOptions.falloff = v;
    if (instancedAnimated && attractorActive) (instancedAnimated as any).setAttractor?.({ enabled: true, x: attractorPos.x, z: attractorPos.z, radius: attractorOptions.radius, turnBoost: attractorOptions.turnBoost, falloff: attractorOptions.falloff });
  });
  attractorFolder.add(attractorGUI, 'yeet').name('Yeet Arrivals').onChange((v: boolean) => {
    if (instancedAnimated) (instancedAnimated as any).setYeetEnabled?.(!!v);
  });
  attractorFolder.add(attractorGUI, 'arrivalDist', 0.1, 5.0, 0.01).name('Arrival Dist').onChange((v: number) => {
    if (instancedAnimated) (instancedAnimated as any).setYeetParams?.({ arrivalDist: v });
  });
  attractorFolder.add(attractorGUI, 'yeetSpeed', 0.0, 100.0, 0.1).name('Yeet Speed').onChange((v: number) => {
    if (instancedAnimated) (instancedAnimated as any).setYeetParams?.({ speed: v });
  });
  attractorFolder.add(attractorGUI, 'horizFrac', 0.0, 1.0, 0.01).name('Horiz Fraction').onChange((v: number) => {
    if (instancedAnimated) (instancedAnimated as any).setYeetParams?.({ horizFrac: v });
  });
  attractorFolder.add(attractorGUI, 'life', 0.0, 10.0, 0.01).name('Life (s)').onChange((v: number) => {
    if (instancedAnimated) (instancedAnimated as any).setYeetParams?.({ life: v });
  });
  attractorFolder.add(attractorGUI, 'gravity', 0.0, 50.0, 0.01).name('Gravity').onChange((v: number) => {
    if (instancedAnimated) (instancedAnimated as any).setYeetParams?.({ gravity: v });
  });
  attractorFolder.add(attractorGUI, 'spin', -50.0, 50.0, 0.1).name('Spin (rad/s)').onChange((v: number) => {
    if (instancedAnimated) (instancedAnimated as any).setYeetParams?.({ spin: v });
  });
  attractorFolder.add(attractorGUI, 'clear').name('Clear Attractor');
  
  // Controls
  const controlsFolder = gui.addFolder('Controls');
  const controlsState = {
    mode: controlMode as ControlMode
  };
  controlModeController = controlsFolder.add(controlsState, 'mode', ['FPS', 'Orbit']).name('Control Mode').onChange((v: ControlMode) => {
    applyControlMode(v);
  });

  // Lighting controls
  const lightingFolder = gui.addFolder('Lighting');
  const lightingState = {
    ambientIntensity: ambientLightRef?.intensity ?? 1.0,
    ambientColor: `#${(ambientLightRef?.color?.getHex() ?? 0xFFFFFF).toString(16).padStart(6, '0')}`,
    dirIntensity: dirLightRef?.intensity ?? 1.0,
    dirColor: `#${(dirLightRef?.color?.getHex() ?? 0xFFFFFF).toString(16).padStart(6, '0')}`,
    dirX: dirLightRef?.position.x ?? 5,
    dirY: dirLightRef?.position.y ?? 10,
    dirZ: dirLightRef?.position.z ?? 7.5,
    exposure: (renderer as any).toneMappingExposure ?? 1.0,
    backgroundIntensity: (scene as any).backgroundIntensity ?? 1.0,
    backgroundBlurriness: (scene as any).backgroundBlurriness ?? 0.0,
    environmentIntensity: (scene as any).environmentIntensity ?? 1.0,
    envMapIntensity: 1.0,
    showBackground: !!scene.background
  };
  lightingFolder.add(lightingState, 'ambientIntensity', 0, 5, 0.01).name('Ambient Intensity').onChange((v: number) => {
    if (ambientLightRef) ambientLightRef.intensity = v;
  });
  lightingFolder.addColor(lightingState, 'ambientColor').name('Ambient Color').onChange((v: string) => {
    if (ambientLightRef) ambientLightRef.color.set(v);
  });
  lightingFolder.add(lightingState, 'dirIntensity', 0, 10, 0.01).name('Directional Intensity').onChange((v: number) => {
    if (dirLightRef) dirLightRef.intensity = v;
  });
  lightingFolder.addColor(lightingState, 'dirColor').name('Directional Color').onChange((v: string) => {
    if (dirLightRef) dirLightRef.color.set(v);
  });
  const dirPosFolder = lightingFolder.addFolder('Directional Position');
  dirPosFolder.add(lightingState, 'dirX', -100, 100, 0.1).name('X').onChange((v: number) => {
    if (dirLightRef) dirLightRef.position.x = v;
  });
  dirPosFolder.add(lightingState, 'dirY', -100, 100, 0.1).name('Y').onChange((v: number) => {
    if (dirLightRef) dirLightRef.position.y = v;
  });
  dirPosFolder.add(lightingState, 'dirZ', -100, 100, 0.1).name('Z').onChange((v: number) => {
    if (dirLightRef) dirLightRef.position.z = v;
  });

  // Tone mapping and environment/background controls
  const tmOptions: Record<string, number> = {
    None: (THREE as any).NoToneMapping,
    ACESFilmic: (THREE as any).ACESFilmicToneMapping,
    Reinhard: (THREE as any).ReinhardToneMapping,
    Cineon: (THREE as any).CineonToneMapping
  };
  const currentTMEntry = Object.entries(tmOptions).find(([, v]) => v === (renderer as any).toneMapping);
  const tmState = { toneMapping: currentTMEntry?.[0] || 'ACESFilmic' };
  lightingFolder.add(tmState, 'toneMapping', Object.keys(tmOptions)).name('Tone Mapping').onChange((name: string) => {
    (renderer as any).toneMapping = tmOptions[name] ?? (THREE as any).ACESFilmicToneMapping;
  });
  lightingFolder.add(lightingState, 'exposure', 0.0, 5.0, 0.01).name('Exposure').onChange((v: number) => {
    (renderer as any).toneMappingExposure = v;
  });
  lightingFolder.add(lightingState, 'showBackground').name('Show Background').onChange((v: boolean) => {
    scene.background = v ? (skyboxTexture || scene.background) : null;
  });
  if ('backgroundIntensity' in (scene as any)) {
    lightingFolder.add(lightingState, 'backgroundIntensity', 0.0, 5.0, 0.01).name('Background Intensity').onChange((v: number) => {
      (scene as any).backgroundIntensity = v;
    });
  }
  if ('backgroundBlurriness' in (scene as any)) {
    lightingFolder.add(lightingState, 'backgroundBlurriness', 0.0, 1.0, 0.001).name('Background Blur').onChange((v: number) => {
      (scene as any).backgroundBlurriness = v;
    });
  }
  if ('environmentIntensity' in (scene as any)) {
    lightingFolder.add(lightingState, 'environmentIntensity', 0.0, 5.0, 0.01).name('Environment Intensity').onChange((v: number) => {
      (scene as any).environmentIntensity = v;
    });
  } else {
    // Fallback: apply envMapIntensity to all materials that support it
    lightingFolder.add(lightingState, 'envMapIntensity', 0.0, 5.0, 0.01).name('EnvMap Intensity (All)').onChange((v: number) => {
      scene.traverse((obj: any) => {
        const mat = obj?.material;
        if (!mat) return;
        if (Array.isArray(mat)) {
          for (const m of mat) if ('envMapIntensity' in m) m.envMapIntensity = v;
        } else if ('envMapIntensity' in mat) {
          mat.envMapIntensity = v;
        }
      });
    });
  }

  // (Forest Settings removed)

  // FPS Controls
  const fpsFolder = gui.addFolder('FPS Controls');
  fpsFolder.add({ get movementSpeed() { return firstPersonControls?.movementSpeed ?? 0; }, set movementSpeed(v: number) { if (firstPersonControls) firstPersonControls.movementSpeed = v; } }, 'movementSpeed', 1, 100, 0.5).name('Move Speed');
  fpsFolder.add({ get lookSpeed() { return firstPersonControls?.lookSpeed ?? 0; }, set lookSpeed(v: number) { if (firstPersonControls) firstPersonControls.lookSpeed = v; } }, 'lookSpeed', 0.01, 1.0, 0.01).name('Look Speed');
}

function setupAttractorInputHandlers() {
  const canvas = renderer.domElement as HTMLCanvasElement;
  const isInsideGUI = (target: EventTarget | null) => !!(guiRef && guiRef.domElement.contains(target as Node));
  const onPointerDown = (e: PointerEvent) => {
    if (!attractorToolEnabled) return;
    if (e.button !== 0) return;
    if (isInsideGUI(e.target)) return;
    attractorMouseDown = true;
    attractorPointerMoved = false;
    attractorDown.x = e.clientX;
    attractorDown.y = e.clientY;
    if (attractorHoldTimer) { clearTimeout(attractorHoldTimer); attractorHoldTimer = null; }
    attractorHoldTimer = window.setTimeout(() => {
      if (!attractorMouseDown || attractorPointerMoved) return;
      const hit = screenToGroundXZ(e.clientX, e.clientY);
      if (!hit) return;
      attractorPos = hit;
      if (instancedAnimated) {
        (instancedAnimated as any).setAttractor?.({
          enabled: true,
          x: attractorPos.x,
          z: attractorPos.z,
          radius: attractorOptions.radius,
          turnBoost: attractorOptions.turnBoost,
          falloff: attractorOptions.falloff
        });
        attractorActive = true;
        // spawn or move banner
        void spawnWarBannerAt(attractorPos.x, attractorPos.z);
        // apply current yeet params defaults on placement
        try {
          (instancedAnimated as any).setYeetParams?.({ arrivalDist: 0.65, speed: 14, horizFrac: 0.35, life: 3.0, gravity: 9.81, spin: 12.0 });
        } catch {}
      }
    }, attractorLongPressMs);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!attractorToolEnabled || !attractorMouseDown) return;
    const dx = e.clientX - attractorDown.x;
    const dy = e.clientY - attractorDown.y;
    if ((dx * dx + dy * dy) > (attractorMinMovePx * attractorMinMovePx)) {
      attractorPointerMoved = true;
      if (attractorHoldTimer) { clearTimeout(attractorHoldTimer); attractorHoldTimer = null; }
    }
  };
  const clearHold = () => {
    attractorMouseDown = false;
    if (attractorHoldTimer) { clearTimeout(attractorHoldTimer); attractorHoldTimer = null; }
  };
  const onPointerUp = () => clearHold();
  const onPointerCancel = () => clearHold();
  canvas.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointercancel', onPointerCancel);
}

function setupOrbitControlsEnhancements() {
  const canvas = renderer.domElement as HTMLCanvasElement;
  const isInsideGUI = (target: EventTarget | null) => !!(guiRef && guiRef.domElement.contains(target as Node));

  const focusOrbitAt = (clientX: number, clientY: number) => {
    if (!orbitControls) return;
    const hit = screenToGroundXZ(clientX, clientY);
    if (!hit) return;
    const y = sampleTerrainHeight(hit.x, hit.z, 3.0, 0.02);
    const newTarget = new (THREE as any).Vector3(hit.x, y, hit.z);
    const camPos = (camera as any).position.clone();
    const prevTarget = orbitControls.target.clone();
    const offset = camPos.clone().sub(prevTarget);
    const dist = Math.max(0.001, offset.length());
    orbitControls.target.copy(newTarget);
    const newPos = newTarget.clone().add(offset.normalize().multiplyScalar(dist));
    (camera as any).position.copy(newPos);
    orbitControls.update();
  };

  canvas.addEventListener('dblclick', (e) => {
    if (controlMode !== 'Orbit') return;
    if (isInsideGUI(e.target)) return;
    focusOrbitAt(e.clientX, e.clientY);
  });
}

function setupAreaSpawnInputHandlers() {
  const canvas = renderer.domElement as HTMLCanvasElement;
  const isInsideGUI = (target: EventTarget | null) => !!(guiRef && guiRef.domElement.contains(target as Node));

  const ensureOverlay = () => {
    if (areaOverlay) return;
    const d = document.createElement('div');
    d.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;z-index:9998;';
    document.body.appendChild(d);
    areaOverlay = d;
  };
  const drawOverlay = () => {
    if (!areaOverlay) return;
    const x0 = Math.min(areaStart.x, areaEnd.x);
    const y0 = Math.min(areaStart.y, areaEnd.y);
    const w = Math.abs(areaEnd.x - areaStart.x);
    const h = Math.abs(areaEnd.y - areaStart.y);
    areaOverlay.innerHTML = '';
    const box = document.createElement('div');
    box.style.cssText = 'position:absolute;border:1px dashed #0f0;background:rgba(0,255,0,0.1);pointer-events:none;';
    box.style.left = `${x0}px`;
    box.style.top = `${y0}px`;
    box.style.width = `${w}px`;
    box.style.height = `${h}px`;
    areaOverlay.appendChild(box);
  };
  const clearOverlay = () => { if (areaOverlay) areaOverlay.innerHTML = ''; };

  const onPointerDown = (e: PointerEvent) => {
    if (!areaToolEnabled) return;
    if (e.button !== 0) return;
    if (isInsideGUI(e.target)) return;
    ensureOverlay();
    areaSelecting = true;
    areaStart.x = e.clientX; areaStart.y = e.clientY;
    areaEnd.x = e.clientX; areaEnd.y = e.clientY;
    drawOverlay();
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!areaToolEnabled || !areaSelecting) return;
    areaEnd.x = e.clientX; areaEnd.y = e.clientY;
    drawOverlay();
  };
  const onPointerUp = () => {
    if (!areaToolEnabled || !areaSelecting) return;
    areaSelecting = false;
    clearOverlay();
    // Convert drag rectangle to world-space AABB on ground plane (y=0)
    const x0 = Math.min(areaStart.x, areaEnd.x);
    const y0 = Math.min(areaStart.y, areaEnd.y);
    const x1 = Math.max(areaStart.x, areaEnd.x);
    const y1 = Math.max(areaStart.y, areaEnd.y);
    // Sample four corners to ground
    const p00 = screenToGroundXZ(x0, y0);
    const p01 = screenToGroundXZ(x0, y1);
    const p10 = screenToGroundXZ(x1, y0);
    const p11 = screenToGroundXZ(x1, y1);
    const pts = [p00, p01, p10, p11].filter(Boolean) as Array<{ x: number; z: number }>;
    if (pts.length < 2) return;
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const p of pts) { if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x; if (p.z < minZ) minZ = p.z; if (p.z > maxZ) maxZ = p.z; }
    if (!isFinite(minX) || !isFinite(maxX) || !isFinite(minZ) || !isFinite(maxZ)) return;
    // Spawn crowd group within this rectangle
    spawnCrowdInArea(minX, minZ, maxX, maxZ);
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
}

function spawnCrowdInArea(minX: number, minZ: number, maxX: number, maxZ: number) {
  if (!crowdTexture) return;
  // Compute area and number of instances based on density, clamp to maxPerSpawn
  const width = Math.max(0, maxX - minX);
  const depth = Math.max(0, maxZ - minZ);
  const area = width * depth;
  let count = Math.max(1, Math.floor(area * areaDensity)) | 0;
  count = Math.min(count, areaMaxPerSpawn);
  // Build uniform grid close to square
  const nx = Math.max(1, Math.floor(Math.sqrt(count * (width / Math.max(1e-6, depth)))));
  const nz = Math.max(1, Math.ceil(count / nx));
  const stepX = width / Math.max(1, nx);
  const stepZ = depth / Math.max(1, nz);
  const want = nx * nz;

  // Fast path: allocate from prewarmed pool if available and sufficient
  if (areaPool && (areaPoolNextFree + want) <= areaPool.count) {
    const start = areaPoolNextFree;
    const positions: Array<{ x: number; z: number; yaw: number }> = [];
    for (let j = 0; j < nz; j++) {
      for (let i = 0; i < nx; i++) {
        const x = minX + (i + 0.5) * stepX;
        const z = minZ + (j + 0.5) * stepZ;
        const yaw = Math.random() * Math.PI * 2; // 360° disperse
        positions.push({ x, z, yaw });
        if (positions.length >= want) break;
      }
      if (positions.length >= want) break;
    }
    areaPool.writePositionsXZYaw(start, positions, animatedYOffset * animatedScale);
    areaPool.markRangeAlive(start, want);
    // Assign variants (range)
    const allowed: number[] = []; for (let v = 0; v < 5; v++) if (areaIncludeVariants[v]) allowed.push(v);
    if (allowed.length === 0) allowed.push(0);
    const vTemp = new Float32Array(want);
    for (let i = 0; i < want; i++) vTemp[i] = allowed[i % allowed.length];
    areaPool.setVariantIndicesRange(start, vTemp, 5);
    try { (areaPool as any).randomizePaletteRowsPerVariantRange?.(start, want, VARIANT_TO_PALETTE_ROWS, paletteRows); } catch {}
    // Diversify animation layers for the newly spawned range
    try { areaPool.randomizeFrameOffsetsPerVariantRange?.(start, want); } catch {}
    // Start/refresh walking with bounds around selection for pooled agent region
    const pad = Math.max(stepX, stepZ) * 0.5;
    const minXb = minX - pad, maxXb = maxX + pad, minZb = minZ - pad, maxZb = maxZ + pad;
    const prevBounds = (areaPool as any).getWalkBounds?.();
    areaPool.startWalking({
      minX: Math.min(prevBounds?.minX ?? minXb, minXb),
      maxX: Math.max(prevBounds?.maxX ?? maxXb, maxXb),
      minZ: Math.min(prevBounds?.minZ ?? minZb, minZb),
      maxZ: Math.max(prevBounds?.maxZ ?? maxZb, maxZb),
      y: animatedYOffset * animatedScale,
      baseSpeed: walkingParams.baseSpeed,
      turnRate: walkingParams.turnRate,
      randomness: walkingParams.randomness,
      avoidanceRadius: walkingParams.avoidanceRadius,
      avoidanceStrength: walkingParams.avoidanceStrength,
      neighborSamples: walkingParams.neighborSamples,
      pushStrength: walkingParams.pushStrength,
      terrainHillAmp: 3.0,
      terrainHillFreq: 0.02,
      gridMaxPerCell: 16,
      binPasses: 3
    });
    areaPoolNextFree += want;
    return;
  }

  // Fallback: create a dedicated group (slower path)
  const group = new InstancedAnimatedOctahedralImpostor(crowdTexture, {
    instanceCount: want,
    useHemiOctahedron: true,
    spritesPerSide: animatedSpritesPerSide,
    transparent: true,
    alphaClamp: 0.02,
    scale: animatedScale,
    flipY: animatedFlipY,
    flipSpriteX: animatedAtlasFlipX,
    flipSpriteY: animatedAtlasFlipY,
    swapSpriteAxes: animatedAtlasSwapAxes,
    paletteTexture: paletteTexture!,
    paletteSize,
    paletteRows,
    paletteData: paletteDataLinear || undefined
  }, crowdNormalTexture, animatedFrameCount);
  scene.add(group);
  group.setFrameCount(animatedFrameCount);
  try { group.setYawSpriteOffsetRadians?.(walkingParams.yawSpriteOffsetDeg * Math.PI / 180); } catch {}
  const positions: Array<{ x: number; z: number; yaw: number }> = [];
  for (let j = 0; j < nz; j++) {
    for (let i = 0; i < nx; i++) {
      const x = minX + (i + 0.5) * stepX;
      const z = minZ + (j + 0.5) * stepZ;
      const yaw = Math.random() * Math.PI * 2; // 360° disperse
      positions.push({ x, z, yaw });
      if (positions.length >= group.count) break;
    }
    if (positions.length >= group.count) break;
  }
  group.setPositionsXZYaw(positions, animatedYOffset * animatedScale);
  const allowed: number[] = []; for (let v = 0; v < 5; v++) if (areaIncludeVariants[v]) allowed.push(v);
  if (allowed.length === 0) allowed.push(0);
  const variantArr = new Float32Array(group.count); for (let i = 0; i < group.count; i++) variantArr[i] = allowed[i % allowed.length];
  group.setVariantIndices(variantArr, 5);
  try { (group as any).randomizePaletteRowsPerVariant?.(VARIANT_TO_PALETTE_ROWS, paletteRows); } catch {}
  try { group.randomizeFrameOffsetsPerVariant?.(); } catch {}
  const pad = Math.max(stepX, stepZ) * 0.5;
  const minXb = minX - pad, maxXb = maxX + pad, minZb = minZ - pad, maxZb = maxZ + pad;
  group.startWalking({
    minX: minXb, maxX: maxXb, minZ: minZb, maxZ: maxZb, y: animatedYOffset * animatedScale,
    baseSpeed: walkingParams.baseSpeed,
    turnRate: walkingParams.turnRate,
    randomness: walkingParams.randomness,
    avoidanceRadius: walkingParams.avoidanceRadius,
    avoidanceStrength: walkingParams.avoidanceStrength,
    neighborSamples: walkingParams.neighborSamples,
    pushStrength: walkingParams.pushStrength,
    terrainHillAmp: 3.0,
    terrainHillFreq: 0.02,
    gridMaxPerCell: 16,
    binPasses: 3
  });
  spawnedCrowdGroups.push(group);
}

function prewarmAreaPool(forceRebuild = false) {
  if (!crowdTexture) return;
  if (areaPool && !forceRebuild && areaPool.count >= areaPoolCapacity) return;
  // Dispose old pool
  if (areaPool) {
    try { scene.remove(areaPool); areaPool.geometry.dispose(); (areaPool.material as any).dispose?.(); } catch {}
    areaPool = null;
  }
  // Create pooled instanced mesh
  areaPool = new InstancedAnimatedOctahedralImpostor(crowdTexture, {
    instanceCount: areaPoolCapacity,
    useHemiOctahedron: true,
    spritesPerSide: animatedSpritesPerSide,
    transparent: true,
    alphaClamp: 0.02,
    scale: animatedScale,
    flipY: animatedFlipY,
    flipSpriteX: animatedAtlasFlipX,
    flipSpriteY: animatedAtlasFlipY,
    swapSpriteAxes: animatedAtlasSwapAxes,
    paletteTexture: paletteTexture!,
    paletteSize,
    paletteRows,
    paletteData: paletteDataLinear || undefined
  }, crowdNormalTexture, animatedFrameCount);
  scene.add(areaPool);
  areaPool.setFrameCount(animatedFrameCount);
  try { areaPool.setYawSpriteOffsetRadians?.(walkingParams.yawSpriteOffsetDeg * Math.PI / 180); } catch {}
  // Hide all initially
  areaPool.markAllDead();
  areaPoolNextFree = 0;
}

function screenToGroundXZ(clientX: number, clientY: number): { x: number; z: number } | null {
  const rect = (renderer.domElement as HTMLCanvasElement).getBoundingClientRect();
  const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
  const ndcY = -(((clientY - rect.top) / rect.height) * 2 - 1);
  const origin = (camera as any).position.clone();
  const dir = new (THREE as any).Vector3(ndcX, ndcY, 0.5);
  dir.unproject(camera);
  dir.sub(origin).normalize();
  const dy = dir.y;
  if (Math.abs(dy) < 1e-6) return null;
  const t = -origin.y / dy;
  if (t <= 0) return null;
  const hit = origin.clone().add(dir.multiplyScalar(t));
  return { x: hit.x, z: hit.z };
}

function animate(time: number) {
  statsJS && statsJS.begin();
  
  const deltaTime = (time - lastTime) / 1000;
  lastTime = time;

  physics.world.step();
  inputHandler.update();
  if (controlMode === 'FPS') {
    if (firstPersonControls) {
      // 3s ease-in to full speed while any movement key is pressed; instant stop on release
      const anyMove = controlMode === 'FPS' && fpsMoveKeys.size > 0;
      const accelRate = fpsAccelTime > 0 ? (1 / fpsAccelTime) : 1000;
      const decelRate = fpsDecelTime > 0 ? (1 / fpsDecelTime) : 1000;
      if (anyMove) {
        fpsSpeedMultiplier = Math.min(1, fpsSpeedMultiplier + accelRate * deltaTime);
      } else {
        fpsSpeedMultiplier = Math.max(0, fpsSpeedMultiplier - decelRate * deltaTime);
      }
      firstPersonControls.movementSpeed = fpsBaseMoveSpeed * fpsSpeedMultiplier;
      firstPersonControls.update(deltaTime);
    }
  } else if (orbitControls) {
    orbitControls.update();
  }
  
  if (animatedImpostor && crowdTexture) {
    if (animatedPlaying) {
      const now = performance.now();
      const frameMs = 1000 / Math.max(1, Math.min(240, animatedFPS));
      if (now - animatedLastFrameTime >= frameMs) {
        animatedLastFrameTime = now;
        animatedFrameTicker += 1;
        animatedImpostor.setFrame(animatedFrameTicker);
        if (instancedAnimated) instancedAnimated.setFrame(animatedFrameTicker);
        for (const g of spawnedCrowdGroups) { try { g.setFrame(animatedFrameTicker); } catch {} }
        if (areaPool) { try { areaPool.setFrame(animatedFrameTicker); } catch {} }
        // UI: wrap to max visible count for display (48 for person 1)
        animatedCurrentFrame = animatedFrameTicker % Math.max(1, animatedFrameCount);
        if (animatedFrameController) animatedFrameController.setValue(animatedCurrentFrame);
      }
    }
    // GPU walking integration: advance per-instance positions/yaw on GPU each frame
    if (instancedAnimated) { try { (instancedAnimated as any).updateWalking?.(renderer, deltaTime, time / 1000); } catch {} }
    for (const g of spawnedCrowdGroups) { try { (g as any).updateWalking?.(renderer, deltaTime, time / 1000); } catch {} }
    if (areaPool) { try { (areaPool as any).updateWalking?.(renderer, deltaTime, time / 1000); } catch {} }
    // Advance war banner animation
    if (warBannerMixer) {
      try { (warBannerMixer as any).update?.(deltaTime); } catch {}
    }
  }
  // Keep sky sphere centered at camera to eliminate translation-induced jitter
  if (skyMesh) {
    (camera as any).getWorldPosition?.(skyMesh.position);
  }
  // Reset WebGPU counters for this frame
  wgpuFrameCalls = 0;
  wgpuFrameTriangles = 0;

  // Measure renderer.info pre/post to get per-frame deltas robustly
  const preInfo = (renderer as any).info || {};
  const preCalls = preInfo?.render?.calls ?? 0;
  const preTris = preInfo?.render?.triangles ?? 0;

  // Render this frame
  renderer.render(scene, camera);

  const postInfo = (renderer as any).info || {};
  const postCalls = postInfo?.render?.calls ?? 0;
  const postTris = postInfo?.render?.triangles ?? 0;

  // Compute deltas; if counters reset within render, fall back to post values
  // Prefer exact WebGPU counters if available
  let frameCalls = (gpuCountersInstalled && (wgpuFrameCalls | 0) > 0)
    ? wgpuFrameCalls | 0
    : (postCalls >= preCalls ? (postCalls - preCalls) : postCalls) | 0;

  let frameTriangles = (gpuCountersInstalled && (wgpuFrameTriangles | 0) > 0)
    ? wgpuFrameTriangles | 0
    : (postTris >= preTris ? (postTris - preTris) : postTris) | 0;

  // If triangles are still 0, estimate from scene geometry
  if ((frameTriangles | 0) === 0 && (frameCalls | 0) > 0) {
    frameTriangles = estimateTrianglesThisFrame(scene);
  }

  updateRenderHUD(frameCalls | 0, frameTriangles | 0);

  // Update GPU/CPU/fps panels
  if (statsGL && typeof statsGL.update === 'function') statsGL.update();

  // Occasionally resolve timestamp queries to avoid blocking
  try {
    if (Math.random() < 0.1) {
      (renderer as any).resolveTimestampsAsync?.((THREE as any).TimestampQuery?.RENDER)?.catch?.(() => {});
      (renderer as any).resolveTimestampsAsync?.((THREE as any).TimestampQuery?.COMPUTE)?.catch?.(() => {});
    }
  } catch {}

  statsJS && statsJS.end();
  requestAnimationFrame(animate);
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  if (firstPersonControls) firstPersonControls.handleResize();
}

init();


