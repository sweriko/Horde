import {
  InstancedMesh,
  Matrix4,
  PlaneGeometry,
  Texture,
  Vector3,
  Euler,
  Quaternion,
  StorageInstancedBufferAttribute
} from 'three/webgpu';
import * as THREE from 'three/webgpu';
import {
  texture as textureNode,
  uniform,
  Fn,
  positionLocal,
  uv,
  cameraPosition,
  vec3,
  vec2,
  vec4,
  float,
  dot,
  normalize,
  cross,
  mix,
  step,
  floor,
  abs,
  sign,
  min,
  round,
  clamp,
  Discard,
  If,
  varying,
  length,
  storage,
  instanceIndex,
  sin,
  cos,
  fract,
  compute
} from 'three/tsl';

interface InstancedAnimatedConfig {
  useHemiOctahedron?: boolean;
  spritesPerSide?: number;
  transparent?: boolean;
  alphaClamp?: number;
  instanceCount: number;
  scale?: number;
  flipY?: boolean;
  flipSpriteX?: boolean;
  flipSpriteY?: boolean;
  swapSpriteAxes?: boolean;
  // Palette indexing
  paletteTexture?: Texture;
  paletteSize?: number; // width in pixels
  paletteRows?: number; // height in pixels (rows)
  paletteData?: Float32Array; // optional: linear RGBA palette LUT
}

function createInstancedAnimatedMaterial(
  arrayTexture: Texture,
  parameters: InstancedAnimatedConfig,
  instancedMesh: InstancedMesh<any, any>,
  normalArrayTexture?: Texture | null
) {
  const material = new (THREE as any).MeshStandardNodeMaterial();
  material.transparent = parameters.transparent ?? true;
  material.metalness = 0.0;
  material.roughness = 0.7;

  // Uniforms
  const spritesPerSide = uniform(parameters.spritesPerSide ?? 16);
  const alphaClamp = uniform(parameters.alphaClamp ?? 0.05);
  const useHemiOctahedron = uniform(parameters.useHemiOctahedron ? 1 : 0);
  const frameIndex = uniform(0);
  const frameCountUniform = uniform(1);
  const globalScale = uniform(parameters.scale ?? 1);
  const flipYFlag = uniform(parameters.flipY ? 1 : 0);
  const flipSpriteXFlag = uniform(parameters.flipSpriteX ? 1 : 0);
  const flipSpriteYFlag = uniform(parameters.flipSpriteY ? 1 : 0);
  const swapSpriteAxesFlag = uniform(parameters.swapSpriteAxes ? 1 : 0);
  // Optional facing calibration offset (radians) to align atlas forward axis
  const yawSpriteOffset = uniform(0.0);

  // Palette uniforms
  const hasPalette = !!(parameters.paletteTexture || parameters.paletteData);
  const paletteSize = uniform(parameters.paletteSize ?? 32);
  const paletteRows = uniform(parameters.paletteRows ?? 1);
  // Optional palette LUT storage buffer (vec4 per entry)
  const paletteBuffer = (parameters.paletteData && (THREE as any).StorageBufferAttribute)
    ? storage(new (THREE as any).StorageBufferAttribute(parameters.paletteData, 4))
    : null;

  // Per-variant animation layout (layer counts and base offsets in merged.ktx2), inline constants
  const c0 = float(48.0); const c1 = float(38.0); const c2 = float(29.0); const c3 = float(31.0); const c4 = float(33.0);
  const b0 = float(0.0);  const b1 = float(48.0); const b2 = float(86.0); const b3 = float(115.0); const b4 = float(146.0);

  // Storage buffer for per-instance state: xyz = world position, w = yaw (radians)
  const instanceStateStorage = storage(new StorageInstancedBufferAttribute(instancedMesh.count, 4));
  // Storage buffer for per-instance animation offsets (float frames in [0, frameCount))
  const instanceOffsetStorage = storage(new StorageInstancedBufferAttribute(new Float32Array(instancedMesh.count), 1));
  // Storage buffer for per-instance palette row (0..paletteRows-1)
  const instancePaletteRowStorage = storage(new StorageInstancedBufferAttribute(new Float32Array(instancedMesh.count), 1));
  // Storage buffer for per-instance variant index (0..variantCount-1)
  const instanceVariantStorage = storage(new StorageInstancedBufferAttribute(new Float32Array(instancedMesh.count), 1));
  // Additional storages for yeet behavior
  // stateFlags: 0 = walking, 1 = yeeting, 2 = dead/removed
  const instanceStateFlagsStorage = storage(new StorageInstancedBufferAttribute(new Float32Array(instancedMesh.count), 1));
  // velocities: xyz velocity, w = angular velocity (yaw rate)
  const instanceVelocityStorage = storage(new StorageInstancedBufferAttribute(new Float32Array(instancedMesh.count * 4), 4));
  // lifetime: time remaining for yeet (seconds)
  const instanceLifeStorage = storage(new StorageInstancedBufferAttribute(new Float32Array(instancedMesh.count), 1));

  // Varyings
  const vSprite = varying(vec2(), 'vSprite');
  const vSpriteUV = varying(vec2(), 'vSpriteUV');

  // Vertex node: billboarding + octahedral sprite selection per instance
  material.positionNode = Fn(() => {
    const spritesMinusOne = vec2(spritesPerSide.sub(1.0));

    // Read per-instance state (position.xyz, yaw)
    const state = instanceStateStorage.element(instanceIndex);
    const instanceCenter = state.xyz;
    const yaw = state.w.add(yawSpriteOffset);
    const cameraPosWorldSpace = cameraPosition.sub(instanceCenter);
    // Transform camera vector into instance local space using inverse yaw (for sprite selection)
    const cosYaw = cos(yaw);
    const sinYaw = sin(yaw);
    // World→local with -yaw: [ [cos, sin], [-sin, cos] ] * [wx, wz]
    const camLocalX = cosYaw.mul(cameraPosWorldSpace.x).add(sinYaw.mul(cameraPosWorldSpace.z));
    const camLocalZ = sinYaw.mul(-1.0).mul(cameraPosWorldSpace.x).add(cosYaw.mul(cameraPosWorldSpace.z));
    const cameraPosLocal = vec3(camLocalX, cameraPosWorldSpace.y, camLocalZ);

    // No Y dampening: always face true camera direction in local space
    const cameraDir = normalize(vec3(cameraPosLocal.x, cameraPosLocal.y, cameraPosLocal.z));

    let up = vec3(0.0, 1.0, 0.0).toVar();
    If(useHemiOctahedron, () => {
      up.assign(mix(up, vec3(-1.0, 0.0, 0.0), step(0.999, cameraDir.y)));
    }).Else(() => {
      up.assign(mix(up, vec3(-1.0, 0.0, 0.0), step(0.999, cameraDir.y)));
      up.assign(mix(up, vec3(1.0, 0.0, 0.0), step(cameraDir.y, -0.999)));
    });

    // Billboard orientation should face camera in world space without Y dampening
    const cameraDirWorld = normalize(vec3(
      cameraPosWorldSpace.x,
      cameraPosWorldSpace.y,
      cameraPosWorldSpace.z
    ));
    const tangent = normalize(cross(up, cameraDirWorld));
    const bitangent = cross(cameraDirWorld, tangent);

    // Variant-based scale: 5 persons (variant 0..4), factors relative to person 1
    const vIdxF = clamp(instanceVariantStorage.element(instanceIndex).x, float(0.0), float(4.0));
    const s0 = float(1.0);       // person 1: biggest (base)
    const s1 = float(0.65);      // person 2
    const s2 = float(0.50);      // person 3
    const s3 = float(0.50);      // person 4
    const s4 = float(0.50);      // person 5
    const is1b = abs(vIdxF.sub(1.0)).lessThan(0.5);
    const is2 = abs(vIdxF.sub(2.0)).lessThan(0.5);
    const is3 = abs(vIdxF.sub(3.0)).lessThan(0.5);
    const is4 = abs(vIdxF.sub(4.0)).lessThan(0.5);
    const scaleVar = float(1.0).toVar();
    scaleVar.assign(s0);
    If(is1b, () => { scaleVar.assign(s1); });
    If(is2, () => { scaleVar.assign(s2); });
    If(is3, () => { scaleVar.assign(s3); });
    If(is4, () => { scaleVar.assign(s4); });

    const finalScale = scaleVar.mul(globalScale);
    // Billboard quad in local XY (tangent/bitangent), then translate by instance position (GPU-only transform)
    const projectedVertex = tangent.mul(positionLocal.x.mul(finalScale)).add(bitangent.mul(positionLocal.y.mul(finalScale))).add(instanceCenter);

    // Octahedral grid
    const grid = vec2().toVar();
    If(useHemiOctahedron, () => {
      const octahedron = cameraDir.div(dot(cameraDir, sign(cameraDir)));
      grid.assign(vec2(octahedron.x.add(octahedron.z), octahedron.z.sub(octahedron.x)).add(1.0).mul(0.5));
    }).Else(() => {
      const dir = cameraDir.div(dot(abs(cameraDir), vec3(1.0))).toVar();
      If(dir.y.lessThan(0.0), () => {
        const signNotZero = mix(vec2(1.0), sign(dir.xz), step(0.0, dir.xz));
        const oldX = dir.x;
        dir.x.assign(float(1.0).sub(abs(dir.z)).mul(signNotZero.x));
        dir.z.assign(float(1.0).sub(abs(oldX)).mul(signNotZero.y));
      });
      grid.assign(dir.xz.mul(0.5).add(0.5));
    });

    // Atlas orientation adjustments
    const gridX = mix(grid.x, float(1.0).sub(grid.x), flipSpriteXFlag);
    const gridY = mix(grid.y, float(1.0).sub(grid.y), flipSpriteYFlag);
    const gridFlip = vec2(gridX, gridY);
    const gridFinal = vec2().toVar();
    If(swapSpriteAxesFlag, () => {
      gridFinal.assign(vec2(gridFlip.y, gridFlip.x));
    }).Else(() => {
      gridFinal.assign(gridFlip);
    });

    const spriteGrid = gridFinal.mul(spritesMinusOne);
    vSprite.assign(min(round(spriteGrid), spritesMinusOne));
    vSpriteUV.assign(uv());

    return vec4(projectedVertex, 1.0);
  })();

  // Color sampling
  material.colorNode = Fn(() => {
    // Hide if per-instance state >= 2 (dead)
    const sf = instanceStateFlagsStorage.element(instanceIndex).x;
    If(sf.greaterThanEqual(2.0), () => { Discard(); });

    const frameSize = float(1.0).div(spritesPerSide);
    const uvY = mix(vSpriteUV.y, float(1.0).sub(vSpriteUV.y), flipYFlag);
    const uv2 = vec2(vSpriteUV.x, uvY);
    const spriteUV = frameSize.mul(vSprite.add(clamp(uv2, vec2(0), vec2(1))));
    // Per-variant wrap and base offset (robust modulo)
    const instOff = instanceOffsetStorage.element(instanceIndex).x;
    // Ensure integer frame before modulo to avoid float drift
    const baseIdx = floor(frameIndex.add(instOff));
    const variantIdx = clamp(instanceVariantStorage.element(instanceIndex).x, float(0.0), float(4.0)).toInt();
    const vCountSel = float(0.0).toVar();
    vCountSel.assign(c0);
    If(abs(float(1.0).sub(variantIdx)).lessThan(0.5), () => { vCountSel.assign(c1); });
    If(abs(float(2.0).sub(variantIdx)).lessThan(0.5), () => { vCountSel.assign(c2); });
    If(abs(float(3.0).sub(variantIdx)).lessThan(0.5), () => { vCountSel.assign(c3); });
    If(abs(float(4.0).sub(variantIdx)).lessThan(0.5), () => { vCountSel.assign(c4); });
    const divF = baseIdx.div(vCountSel);
    const fracF = divF.sub(floor(divF));
    const localIdx = floor(fracF.mul(vCountSel)).toInt();
    const vBaseSel = float(0.0).toVar();
    vBaseSel.assign(b0);
    If(abs(float(1.0).sub(variantIdx)).lessThan(0.5), () => { vBaseSel.assign(b1); });
    If(abs(float(2.0).sub(variantIdx)).lessThan(0.5), () => { vBaseSel.assign(b2); });
    If(abs(float(3.0).sub(variantIdx)).lessThan(0.5), () => { vBaseSel.assign(b3); });
    If(abs(float(4.0).sub(variantIdx)).lessThan(0.5), () => { vBaseSel.assign(b4); });
    const variantBase = vBaseSel.toInt();
    const finalIdx = variantBase.add(localIdx);
    // Force array semantics by hinting depth usage on the node
    const sampleNode = (textureNode as any)(arrayTexture, spriteUV).depth((finalIdx as any).toInt());
    if (hasPalette) {
      const idx255 = floor(sampleNode.x.mul(255.0).add(0.5));
      const maxIdx = paletteSize.sub(1.0);
      const clampedIdx = clamp(idx255, float(0.0), maxIdx);
      const row = clamp(instancePaletteRowStorage.element(instanceIndex).x, float(0.0), paletteRows.sub(1.0));
      if (paletteBuffer) {
        const flatIndex = row.mul(paletteSize).add(clampedIdx).toInt();
        const palColor = (paletteBuffer as any).element(flatIndex);
        const alphaFromId = step(float(0.5), clampedIdx);
        const outColor = vec4(palColor.xyz, alphaFromId);
        If(outColor.a.lessThanEqual(alphaClamp), () => { Discard(); });
        return outColor;
      } else {
        const uCoord = clampedIdx.add(0.5).div(paletteSize);
        const vCoord = row.add(0.5).div(paletteRows);
        const palColor = textureNode(parameters.paletteTexture as any, vec2(uCoord, vCoord));
        const alphaFromId = step(float(0.5), clampedIdx);
        const outColor = vec4(palColor.rgb, alphaFromId);
        If(outColor.a.lessThanEqual(alphaClamp), () => { Discard(); });
        return outColor;
      }
    }
    const spriteColor = sampleNode;
    If(spriteColor.a.lessThanEqual(alphaClamp), () => { Discard(); });
    return spriteColor;
  })();

  // Normal sampling
  if (normalArrayTexture) {
    material.normalNode = Fn(() => {
      const frameSize = float(1.0).div(spritesPerSide);
      const uvY = mix(vSpriteUV.y, float(1.0).sub(vSpriteUV.y), flipYFlag);
      const uv2 = vec2(vSpriteUV.x, uvY);
      const spriteUV = frameSize.mul(vSprite.add(clamp(uv2, vec2(0), vec2(1))));
      const instOff = instanceOffsetStorage.element(instanceIndex).x;
      const baseIdx = floor(frameIndex.add(instOff));
      const variantIdx = clamp(instanceVariantStorage.element(instanceIndex).x, float(0.0), float(4.0)).toInt();
      const vCountSel = float(0.0).toVar();
      vCountSel.assign(c0);
      If(abs(float(1.0).sub(variantIdx)).lessThan(0.5), () => { vCountSel.assign(c1); });
      If(abs(float(2.0).sub(variantIdx)).lessThan(0.5), () => { vCountSel.assign(c2); });
      If(abs(float(3.0).sub(variantIdx)).lessThan(0.5), () => { vCountSel.assign(c3); });
      If(abs(float(4.0).sub(variantIdx)).lessThan(0.5), () => { vCountSel.assign(c4); });
      const divF = baseIdx.div(vCountSel);
      const fracF = divF.sub(floor(divF));
      const localIdx = floor(fracF.mul(vCountSel)).toInt();
      const vBaseSel = float(0.0).toVar();
      vBaseSel.assign(b0);
      If(abs(float(1.0).sub(variantIdx)).lessThan(0.5), () => { vBaseSel.assign(b1); });
      If(abs(float(2.0).sub(variantIdx)).lessThan(0.5), () => { vBaseSel.assign(b2); });
      If(abs(float(3.0).sub(variantIdx)).lessThan(0.5), () => { vBaseSel.assign(b3); });
      If(abs(float(4.0).sub(variantIdx)).lessThan(0.5), () => { vBaseSel.assign(b4); });
      const variantBase = vBaseSel.toInt();
      const finalIdx = variantBase.add(localIdx);
      const normalSample = (textureNode as any)(normalArrayTexture, spriteUV).depth((finalIdx as any).toInt());
      return normalize(normalSample.xyz);
    })();
  }

  material.instancedAnimatedUniforms = {
    spritesPerSide,
    alphaClamp,
    frameIndex,
    frameCountUniform,
    globalScale,
    flipYFlag,
    flipSpriteXFlag,
    flipSpriteYFlag,
    swapSpriteAxesFlag,
    paletteSize,
    paletteRows,
    yawSpriteOffset
  };
  (material as any).instanceStateStorage = instanceStateStorage;
  material.instanceOffsetStorage = instanceOffsetStorage;
  (material as any).instancePaletteRowStorage = instancePaletteRowStorage;
  (material as any).instanceVariantStorage = instanceVariantStorage;
  // expose yeet storages
  (material as any).instanceStateFlagsStorage = instanceStateFlagsStorage;
  (material as any).instanceVelocityStorage = instanceVelocityStorage;
  (material as any).instanceLifeStorage = instanceLifeStorage;

  return material;
}

const INST_PLANE_GEOMETRY = new THREE.PlaneGeometry();

export class InstancedAnimatedOctahedralImpostor extends InstancedMesh<PlaneGeometry, any> {
  private _instanceStateStorage: any;
  private _instanceOffsetStorage: any;
  private _instancePaletteRowStorage: any;
  private _instanceVariantStorage: any;
  private _instanceStateFlagsStorage: any;
  private _instanceVelocityStorage: any;
  private _instanceLifeStorage: any;
  private _walkCompute: any;
  private _walkUniforms: any;
  private _gridClearCompute: any;
  private _gridBinCompute: any;
  private _gridIndicesStorage: any;
  private _lastFrameIndex: number = 0;

  constructor(
    arrayTexture: Texture,
    config: InstancedAnimatedConfig,
    normalArrayTexture?: Texture | null,
    frameCount?: number
  ) {
    super(INST_PLANE_GEOMETRY, new (THREE as any).MeshStandardNodeMaterial(), config.instanceCount);
    const mat = createInstancedAnimatedMaterial(arrayTexture, config, this, normalArrayTexture);
    this.material = mat;
    this._instanceStateStorage = (mat as any).instanceStateStorage;
    this._instanceOffsetStorage = (mat as any).instanceOffsetStorage;
    this._instancePaletteRowStorage = (mat as any).instancePaletteRowStorage;
    this._instanceVariantStorage = (mat as any).instanceVariantStorage;
    this._instanceStateFlagsStorage = (mat as any).instanceStateFlagsStorage;
    this._instanceVelocityStorage = (mat as any).instanceVelocityStorage;
    this._instanceLifeStorage = (mat as any).instanceLifeStorage;
    this.frustumCulled = false;
    if (frameCount !== undefined) {
      this.setFrameCount(frameCount);
    }

    // Initialize walking compute (disabled until startWalking is called)
    this._walkCompute = null;
    this._walkUniforms = null;
    this._gridClearCompute = null;
    this._gridBinCompute = null;
    this._gridIndicesStorage = null;
  }

  // Public read-only accessor for current walk bounds (used to merge new area selections)
  getWalkBounds(): { minX: number; maxX: number; minZ: number; maxZ: number } | null {
    const u = this._walkUniforms;
    if (!u) return null;
    return {
      minX: Number(u.minX?.value ?? NaN),
      maxX: Number(u.maxX?.value ?? NaN),
      minZ: Number(u.minZ?.value ?? NaN),
      maxZ: Number(u.maxZ?.value ?? NaN)
    };
  }

  setInstanceTransform(index: number, position: Vector3, rotation?: Euler | Quaternion, scale?: Vector3): void {
    const matrix = new Matrix4();
    if (rotation instanceof Euler) {
      matrix.makeRotationFromEuler(rotation);
    } else if (rotation instanceof Quaternion) {
      matrix.makeRotationFromQuaternion(rotation);
    }
    if (scale) matrix.scale(scale);
    matrix.setPosition(position);
    this.setMatrixAt(index, matrix);
  }

  private setAllInstanceMatricesIdentity(): void {
    const identity = new Matrix4();
    for (let i = 0; i < this.count; i++) this.setMatrixAt(i, identity);
    this.instanceMatrix.needsUpdate = true;
  }

  generateRandomPositions(count: number, terrainBounds: { minX: number; maxX: number; minZ: number; maxZ: number; y?: number }): void {
    // Seed GPU state buffer with random positions and yaw; keep instanced matrices identity (GPU will translate)
    const arr = this._instanceStateStorage?.value?.array as Float32Array | undefined;
    const flags = this._instanceStateFlagsStorage?.value?.array as Float32Array | undefined;
    const vels = this._instanceVelocityStorage?.value?.array as Float32Array | undefined;
    const lifes = this._instanceLifeStorage?.value?.array as Float32Array | undefined;
    if (arr && arr.length >= this.count * 4) {
      for (let i = 0; i < count && i < this.count; i++) {
        const x = Math.random() * (terrainBounds.maxX - terrainBounds.minX) + terrainBounds.minX;
        const z = Math.random() * (terrainBounds.maxZ - terrainBounds.minZ) + terrainBounds.minZ;
        const y = terrainBounds.y ?? 0;
        const yaw = Math.random() * Math.PI * 2;
        const idx = i * 4;
        arr[idx + 0] = x;
        arr[idx + 1] = y;
        arr[idx + 2] = z;
        arr[idx + 3] = yaw;
        if (flags) flags[i] = 0; // walking
        if (lifes) lifes[i] = 0;
        if (vels) {
          const vi = i * 4;
          vels[vi + 0] = 0;
          vels[vi + 1] = 0;
          vels[vi + 2] = 0;
          vels[vi + 3] = 0;
        }
      }
      this._instanceStateStorage.value.needsUpdate = true;
      if (this._instanceStateFlagsStorage) this._instanceStateFlagsStorage.value.needsUpdate = true;
      if (this._instanceVelocityStorage) this._instanceVelocityStorage.value.needsUpdate = true;
      if (this._instanceLifeStorage) this._instanceLifeStorage.value.needsUpdate = true;
    }
    this.setAllInstanceMatricesIdentity();
  }

  setFrame(value: number): void {
    const uniforms = (this.material as any).instancedAnimatedUniforms;
    if (uniforms) uniforms.frameIndex.value = value | 0;
    this._lastFrameIndex = value | 0;
    if (this._walkUniforms) this._walkUniforms.frameRaw.value = this._lastFrameIndex;
  }

  setFrameCount(value: number): void {
    const uniforms = (this.material as any).instancedAnimatedUniforms;
    if (uniforms) uniforms.frameCountUniform.value = Math.max(1, value | 0);
  }

  setSpritesPerSide(value: number): void {
    const uniforms = (this.material as any).instancedAnimatedUniforms;
    if (uniforms) uniforms.spritesPerSide.value = value | 0;
  }

  setAlphaClamp(value: number): void {
    const uniforms = (this.material as any).instancedAnimatedUniforms;
    if (uniforms) uniforms.alphaClamp.value = value;
  }

  setScale(value: number): void {
    const uniforms = (this.material as any).instancedAnimatedUniforms;
    if (uniforms) uniforms.globalScale.value = value;
  }

  setYawSpriteOffsetRadians(value: number): void {
    const uniforms = (this.material as any).instancedAnimatedUniforms;
    if (uniforms) uniforms.yawSpriteOffset.value = value;
  }

  /**
   * Set per-instance world positions (x,z) and a uniform Y height plus yaw.
   * This seeds the GPU state buffers directly and resets instance matrices to identity
   * because GPU transforms are applied in the shader from the state storage.
   */
  setPositionsXZYaw(positions: Array<{ x: number; z: number; yaw: number }>, y: number): void {
    if (!this._instanceStateStorage) return;
    const arr = this._instanceStateStorage.value.array as Float32Array;
    const n = Math.min(this.count | 0, positions.length | 0);
    const flags = this._instanceStateFlagsStorage?.value?.array as Float32Array | undefined;
    const vels = this._instanceVelocityStorage?.value?.array as Float32Array | undefined;
    const lifes = this._instanceLifeStorage?.value?.array as Float32Array | undefined;
    for (let i = 0; i < n; i++) {
      const p = positions[i];
      const idx = i * 4;
      arr[idx + 0] = p.x;
      arr[idx + 1] = y;
      arr[idx + 2] = p.z;
      arr[idx + 3] = p.yaw;
      if (flags) flags[i] = 0; // walking
      if (lifes) lifes[i] = 0;
      if (vels) {
        const vi = i * 4;
        vels[vi + 0] = 0;
        vels[vi + 1] = 0;
        vels[vi + 2] = 0;
        vels[vi + 3] = 0;
      }
    }
    this._instanceStateStorage.value.needsUpdate = true;
    if (this._instanceStateFlagsStorage) this._instanceStateFlagsStorage.value.needsUpdate = true;
    if (this._instanceVelocityStorage) this._instanceVelocityStorage.value.needsUpdate = true;
    if (this._instanceLifeStorage) this._instanceLifeStorage.value.needsUpdate = true;
    // Reset instance matrices to identity (GPU state drives transforms)
    const identity = new Matrix4();
    for (let i = 0; i < this.count; i++) this.setMatrixAt(i, identity);
    this.instanceMatrix.needsUpdate = true;
  }

  /** Write positions/yaw into a subrange [startIndex, startIndex+positions.length). */
  writePositionsXZYaw(startIndex: number, positions: Array<{ x: number; z: number; yaw: number }>, y: number): void {
    if (!this._instanceStateStorage) return;
    const arr = this._instanceStateStorage.value.array as Float32Array;
    const n = Math.min(positions.length | 0, Math.max(0, this.count - (startIndex | 0)));
    const base = (startIndex | 0) * 4;
    const flags = this._instanceStateFlagsStorage?.value?.array as Float32Array | undefined;
    const vels = this._instanceVelocityStorage?.value?.array as Float32Array | undefined;
    const lifes = this._instanceLifeStorage?.value?.array as Float32Array | undefined;
    for (let i = 0; i < n; i++) {
      const p = positions[i];
      const idx = base + i * 4;
      arr[idx + 0] = p.x;
      arr[idx + 1] = y;
      arr[idx + 2] = p.z;
      arr[idx + 3] = p.yaw;
      if (flags) flags[(startIndex | 0) + i] = 0; // mark alive/walking
      if (lifes) lifes[(startIndex | 0) + i] = 0;
      if (vels) {
        const vi = ((startIndex | 0) + i) * 4;
        vels[vi + 0] = 0; vels[vi + 1] = 0; vels[vi + 2] = 0; vels[vi + 3] = 0;
      }
    }
    this._instanceStateStorage.value.needsUpdate = true;
    if (this._instanceStateFlagsStorage) this._instanceStateFlagsStorage.value.needsUpdate = true;
    if (this._instanceVelocityStorage) this._instanceVelocityStorage.value.needsUpdate = true;
    if (this._instanceLifeStorage) this._instanceLifeStorage.value.needsUpdate = true;
  }

  /** Mark all instances as dead/hidden (state flag = 2). */
  markAllDead(): void {
    if (!this._instanceStateFlagsStorage) return;
    const flags = this._instanceStateFlagsStorage.value.array as Float32Array;
    for (let i = 0; i < this.count; i++) flags[i] = 2;
    this._instanceStateFlagsStorage.value.needsUpdate = true;
  }

  /** Mark a subrange [start, start+count) as alive (state flag = 0). */
  markRangeAlive(startIndex: number, count: number): void {
    if (!this._instanceStateFlagsStorage) return;
    const flags = this._instanceStateFlagsStorage.value.array as Float32Array;
    const s = Math.max(0, startIndex | 0);
    const e = Math.min(this.count, s + (count | 0));
    for (let i = s; i < e; i++) flags[i] = 0;
    this._instanceStateFlagsStorage.value.needsUpdate = true;
  }

  /** Set variant indices for a range quickly from an array-like source. */
  setVariantIndicesRange(startIndex: number, variants: ArrayLike<number>, variantCount = 5): void {
    if (!this._instanceVariantStorage || !this._instancePaletteRowStorage) return;
    const vArr = this._instanceVariantStorage.value.array as Float32Array;
    const pArr = this._instancePaletteRowStorage.value.array as Float32Array;
    const rows = Math.max(1, variantCount | 0);
    const s = Math.max(0, startIndex | 0);
    const n = Math.min(variants.length | 0, this.count - s);
    for (let i = 0; i < n; i++) {
      const vi = Math.max(0, Math.min(rows - 1, (variants[i] | 0)));
      vArr[s + i] = vi;
      pArr[s + i] = vi;
    }
    this._instanceVariantStorage.value.needsUpdate = true;
    this._instancePaletteRowStorage.value.needsUpdate = true;
  }

  setFlipY(value: boolean): void {
    const uniforms = (this.material as any).instancedAnimatedUniforms;
    if (uniforms) uniforms.flipYFlag.value = value ? 1 : 0;
  }

  setAtlasFlipAndSwap(options: { flipX?: boolean; flipY?: boolean; swapAxes?: boolean }): void {
    const uniforms = (this.material as any).instancedAnimatedUniforms;
    if (!uniforms) return;
    if (options.flipX !== undefined) uniforms.flipSpriteXFlag.value = options.flipX ? 1 : 0;
    if (options.flipY !== undefined) uniforms.flipSpriteYFlag.value = options.flipY ? 1 : 0;
    if (options.swapAxes !== undefined) uniforms.swapSpriteAxesFlag.value = options.swapAxes ? 1 : 0;
  }

  randomizeFrameOffsets(frameCount: number): void {
    if (!this._instanceOffsetStorage) return;
    const arr = this._instanceOffsetStorage.value.array as Float32Array;
    const max = Math.max(1, frameCount | 0);
    for (let i = 0; i < this.count; i++) {
      arr[i] = Math.floor(Math.random() * max);
    }
    this._instanceOffsetStorage.value.needsUpdate = true;
  }

  setFrameOffsets(offsets: Float32Array, frameCount: number): void {
    if (!this._instanceOffsetStorage) return;
    const arr = this._instanceOffsetStorage.value.array as Float32Array;
    const max = Math.max(1, frameCount | 0);
    const n = Math.min(arr.length, offsets.length);
    for (let i = 0; i < n; i++) {
      const v = offsets[i] | 0;
      arr[i] = ((v % max) + max) % max;
    }
    this._instanceOffsetStorage.value.needsUpdate = true;
  }

  /** Assign pseudo-random frame offsets to a subrange using stable hashing. */
  randomizeFrameOffsetsPerVariantRange(startIndex: number, count: number): void {
    if (!this._instanceOffsetStorage || !this._instanceVariantStorage) return;
    const offsets = this._instanceOffsetStorage.value.array as Float32Array;
    const variants = this._instanceVariantStorage.value.array as Float32Array;
    const s = Math.max(0, startIndex | 0);
    const e = Math.min(this.count, s + (count | 0));
    const countsPerVariant = [48, 38, 29, 31, 33];
    for (let i = s; i < e; i++) {
      const v = Math.max(0, Math.min(4, (variants[i] | 0)));
      const c = countsPerVariant[v] | 0;
      const h = Math.abs(Math.sin(i * 12.9898) * 43758.5453);
      offsets[i] = c > 0 ? (h % c) : 0;
    }
    this._instanceOffsetStorage.value.needsUpdate = true;
  }

  randomizeFrameOffsetsPerVariant(): void {
    if (!this._instanceOffsetStorage || !this._instanceVariantStorage) return;
    const offsets = this._instanceOffsetStorage.value.array as Float32Array;
    const variants = this._instanceVariantStorage.value.array as Float32Array;
    const counts = [48, 38, 29, 31, 33];
    for (let i = 0; i < this.count; i++) {
      const v = Math.max(0, Math.min(4, (variants[i] | 0)));
      const c = counts[v] | 0;
      const h = Math.abs(Math.sin(i * 12.9898) * 43758.5453);
      offsets[i] = c > 0 ? (h % c) : 0;
    }
    this._instanceOffsetStorage.value.needsUpdate = true;
  }

  setPaletteRow(index: number, row: number, totalRows?: number): void {
    if (!this._instancePaletteRowStorage) return;
    const arr = this._instancePaletteRowStorage.value.array as Float32Array;
    const rows = Math.max(1, (totalRows ?? 1) | 0);
    arr[index | 0] = Math.max(0, Math.min(rows - 1, row | 0));
    this._instancePaletteRowStorage.value.needsUpdate = true;
  }

  randomizePaletteRows(totalRows: number): void {
    if (!this._instancePaletteRowStorage) return;
    const arr = this._instancePaletteRowStorage.value.array as Float32Array;
    const rows = Math.max(1, totalRows | 0);
    for (let i = 0; i < this.count; i++) {
      arr[i] = rows > 1 ? (Math.random() * rows) | 0 : 0;
    }
    this._instancePaletteRowStorage.value.needsUpdate = true;
  }

  setVariantIndex(index: number, variant: number, variantCount = 5): void {
    if (!this._instanceVariantStorage) return;
    const vArr = this._instanceVariantStorage.value.array as Float32Array;
    const clamped = Math.max(0, Math.min((variantCount | 0) - 1, variant | 0));
    vArr[index | 0] = clamped;
    this._instanceVariantStorage.value.needsUpdate = true;
    // Keep palette row in sync: row == variant
    this.setPaletteRow(index, clamped, variantCount);
  }

  setVariantIndices(variants: ArrayLike<number>, variantCount = 5): void {
    if (!this._instanceVariantStorage || !this._instancePaletteRowStorage) return;
    const vArr = this._instanceVariantStorage.value.array as Float32Array;
    const pArr = this._instancePaletteRowStorage.value.array as Float32Array;
    const rows = Math.max(1, variantCount | 0);
    const n = Math.min(this.count, variants.length | 0);
    for (let i = 0; i < n; i++) {
      const v = Math.max(0, Math.min(rows - 1, (variants[i] | 0)));
      vArr[i] = v;
      pArr[i] = v; // palette row == variant index
    }
    this._instanceVariantStorage.value.needsUpdate = true;
    this._instancePaletteRowStorage.value.needsUpdate = true;
  }

  /** Randomize palette rows per instance based on its assigned variant. */
  randomizePaletteRowsPerVariant(variantToRows: number[][], totalRows?: number): void {
    if (!this._instanceVariantStorage || !this._instancePaletteRowStorage) return;
    const vArr = this._instanceVariantStorage.value.array as Float32Array;
    const pArr = this._instancePaletteRowStorage.value.array as Float32Array;
    const rowLimit = totalRows !== undefined ? Math.max(1, (totalRows | 0)) : Number.MAX_SAFE_INTEGER;
    for (let i = 0; i < this.count; i++) {
      const v = Math.max(0, Math.min(variantToRows.length - 1, (vArr[i] | 0)));
      const choices = variantToRows[v] && variantToRows[v].length > 0 ? variantToRows[v] : [v];
      const pick = choices[(Math.random() * choices.length) | 0] | 0;
      const clamped = Math.max(0, Math.min(rowLimit - 1, pick));
      pArr[i] = clamped;
    }
    this._instancePaletteRowStorage.value.needsUpdate = true;
  }

  /** Randomize palette rows for a subrange [startIndex, startIndex+count) based on assigned variants. */
  randomizePaletteRowsPerVariantRange(startIndex: number, count: number, variantToRows: number[][], totalRows?: number): void {
    if (!this._instanceVariantStorage || !this._instancePaletteRowStorage) return;
    const vArr = this._instanceVariantStorage.value.array as Float32Array;
    const pArr = this._instancePaletteRowStorage.value.array as Float32Array;
    const s = Math.max(0, startIndex | 0);
    const e = Math.min(this.count, s + (count | 0));
    const rowLimit = totalRows !== undefined ? Math.max(1, (totalRows | 0)) : Number.MAX_SAFE_INTEGER;
    for (let i = s; i < e; i++) {
      const v = Math.max(0, Math.min(variantToRows.length - 1, (vArr[i] | 0)));
      const choices = variantToRows[v] && variantToRows[v].length > 0 ? variantToRows[v] : [v];
      const pick = choices[(Math.random() * choices.length) | 0] | 0;
      const clamped = Math.max(0, Math.min(rowLimit - 1, pick));
      pArr[i] = clamped;
    }
    this._instancePaletteRowStorage.value.needsUpdate = true;
  }

  assignVariantsEqually(variantCount = 5): void {
    if (!this._instanceVariantStorage || !this._instancePaletteRowStorage) return;
    const vArr = this._instanceVariantStorage.value.array as Float32Array;
    const pArr = this._instancePaletteRowStorage.value.array as Float32Array;
    const rows = Math.max(1, variantCount | 0);
    for (let i = 0; i < this.count; i++) {
      const v = i % rows;
      vArr[i] = v;
      pArr[i] = v; // palette row == variant index
    }
    this._instanceVariantStorage.value.needsUpdate = true;
    this._instancePaletteRowStorage.value.needsUpdate = true;
  }

  assignVariantsWeighted(weights: number[]): void {
    if (!this._instanceVariantStorage || !this._instancePaletteRowStorage) return;
    const vArr = this._instanceVariantStorage.value.array as Float32Array;
    const pArr = this._instancePaletteRowStorage.value.array as Float32Array;
    const n = this.count | 0;
    const rows = Math.max(1, (weights?.length | 0));
    if (!weights || rows <= 0) return this.assignVariantsEqually(5);
    // Normalize weights (ensure non-negative) and compute cumulative sum
    const safeW = new Array(rows).fill(0);
    let total = 0;
    for (let i = 0; i < rows; i++) {
      const w = Math.max(0, Number(weights[i] ?? 0));
      safeW[i] = w;
      total += w;
    }
    if (total <= 0) return this.assignVariantsEqually(rows);
    // Deterministic weighted assignment using cumulative thresholds over repeating window [0, total)
    const cum = new Array(rows);
    let acc = 0;
    for (let i = 0; i < rows; i++) { acc += safeW[i]; cum[i] = acc; }
    for (let i = 0; i < n; i++) {
      const t = i % total; // round-robin across weighted window
      // Find first cum[j] > t
      let j = 0;
      while (j < rows && cum[j] <= t) j++;
      const v = j < rows ? j : (rows - 1);
      vArr[i] = v;
      pArr[i] = v;
    }
    this._instanceVariantStorage.value.needsUpdate = true;
    this._instancePaletteRowStorage.value.needsUpdate = true;
  }

  startWalking(options: {
    minX: number; maxX: number; minZ: number; maxZ: number; y: number;
    baseSpeed?: number; // world units per second
    turnRate?: number;  // radians per second (max)
    randomness?: number; // 0..1 multiplier for random turning
    // Crowd/avoidance parameters (all optional)
    avoidanceRadius?: number;      // separation radius in world units
    avoidanceStrength?: number;    // scalar applied to separation steering
    neighborSamples?: number;      // how many random neighbors to sample (1..8 recommended)
    pushStrength?: number;         // soft positional push after move to resolve overlaps
    gridMaxPerCell?: number;       // max agents stored per grid cell (sampling set)
    binPasses?: number;            // how many times to bin per frame (>=1)
    // Terrain hills (optional)
    terrainHillAmp?: number;
    terrainHillFreq?: number;
    // Walk-cycle coupling (optional)
    cycleAmp?: number;             // 0..1 sinusoidal modulation of forward speed synced to anim frame
  }): void {
    // Create uniforms and compute kernel if not created
    if (!this._walkUniforms) {
      const u = {
        deltaTime: uniform(0.016),
        time: uniform(0.0),
        minX: uniform(options.minX),
        maxX: uniform(options.maxX),
        minZ: uniform(options.minZ),
        maxZ: uniform(options.maxZ),
        yHeight: uniform(options.y),
        baseSpeed: uniform(options.baseSpeed ?? 2.0),
        turnRate: uniform(options.turnRate ?? 0.7),
        randomness: uniform(options.randomness ?? 1.0),
        agentCount: uniform(this.count | 0),
        avoidRadius: uniform(options.avoidanceRadius ?? 150.0),
        avoidStrength: uniform(options.avoidanceStrength ?? 1.0),
        neighborSamples: uniform(Math.max(1, Math.min(64, (options.neighborSamples ?? 6) | 0))),
        pushStrength: uniform(options.pushStrength ?? 0.5),
        // Sync with sprite animation
        frameRaw: uniform(this._lastFrameIndex | 0),
        cycleAmp: uniform(Math.max(0, Math.min(1, options.cycleAmp ?? 0.15))),
        // Attractor controls
        attractorEnabled: uniform(0),
        attractorX: uniform(0.0),
        attractorZ: uniform(0.0),
        attractorRadius: uniform(1000.0),
        attractorTurnBoost: uniform(2.0),
        attractorFalloff: uniform(1.0), // 1 = linear, 2 = quadratic, etc.
        // Grid params (initialized below and kept in sync)
        gridMinX: uniform(options.minX),
        gridMinZ: uniform(options.minZ),
        cellSize: uniform((options.avoidanceRadius ?? 150.0) * 1.0),
        gridResX: uniform(1),
        gridResZ: uniform(1),
        maxPerCell: uniform(Math.max(1, (options.gridMaxPerCell ?? 16) | 0)),
        binPasses: uniform(Math.max(1, (options.binPasses ?? 2) | 0))
      };
      // Yeet controls
      (u as any).yeetEnabled = uniform(0);
      (u as any).yeetArrivalDist = uniform(1.0);
      (u as any).yeetSpeed = uniform(12.0);
      (u as any).yeetHorizFrac = uniform(0.35);
      (u as any).yeetLife = uniform(3.0);
      (u as any).yeetGravity = uniform(9.81);
      (u as any).yeetSpin = uniform(8.0);
      // Terrain hill uniforms
      (u as any).terrainHillAmp = uniform(Math.max(0, options.terrainHillAmp ?? 3.0));
      (u as any).terrainHillFreq = uniform(Math.max(0.00001, options.terrainHillFreq ?? 0.02));
      this._walkUniforms = u;

      const state = (this.material as any).instanceStateStorage;
      const variant = (this.material as any).instanceVariantStorage;
      const offsets = (this.material as any).instanceOffsetStorage;
      const flags = (this.material as any).instanceStateFlagsStorage;
      const vels = (this.material as any).instanceVelocityStorage;
      const lifes = (this.material as any).instanceLifeStorage;

      // Helper to ensure grid storages exist and sized
      // no-op placeholder removed

      // Build grid kernels: clear and bin
      const gridIndices = storage(new StorageInstancedBufferAttribute(new Float32Array(1), 1));
      (this as any)._gridIndicesStorageRef = gridIndices; // hold reference for JS-side rebind

      const clearKernel = Fn(() => {
        const idx = instanceIndex; // index into gridIndices flat buffer
        const e = gridIndices.element(idx);
        e.x.assign(float(-1.0));
      })();

      const binKernel = Fn(() => {
        const s = state.element(instanceIndex);
        const sf = flags.element(instanceIndex).x;
        If(sf.lessThan(0.5), () => {
          const idxf = float(instanceIndex);
          // Compute grid cell
          const gx = floor(s.x.sub(u.gridMinX).div(u.cellSize)).toVar();
          const gz = floor(s.z.sub(u.gridMinZ).div(u.cellSize)).toVar();
          const rx = clamp(gx, float(0.0), u.gridResX.sub(1.0));
          const rz = clamp(gz, float(0.0), u.gridResZ.sub(1.0));
          const cell = rz.mul(u.gridResX).add(rx).toInt();
          const base = cell.mul(u.maxPerCell.toInt());

          // Try up to 4 slots with different seeds (best-effort without atomics)
          const tryPlace = (seedA: number, seedB: number) => {
            const h = fract(sin(idxf.mul(seedA).add(u.time.mul(seedB))).mul(43758.5453));
            const slot = h.mul(u.maxPerCell).toInt();
            const ptr = gridIndices.element(base.add(slot));
            const cur = ptr.x;
            If(cur.lessThan(0.0), () => { ptr.x.assign(idxf); });
          };
          tryPlace(12.5, 0.13);
          tryPlace(71.7, 0.73);
          tryPlace(39.3, 1.37);
          tryPlace(93.9, 2.11);
        });
      })();

      this._gridClearCompute = compute(clearKernel, 1, [256]); // placeholder, will resize below
      this._gridBinCompute = compute(binKernel, this.count, [256]);
      const walkKernel = Fn(() => {
        const s = state.element(instanceIndex); // vec4: pos.xyz, yaw
        const sf = flags.element(instanceIndex).x.toVar();
        const v = vels.element(instanceIndex); // vec4: vel.xyz, angVel
        const life = lifes.element(instanceIndex).x.toVar();

        // Current forward from yaw
        const yaw0 = s.w.toVar();
        const fcx = cos(yaw0);
        const fsz = sin(yaw0);

        // Separation via uniform grid lookup (3x3 cells, fixed MAX_PER_CELL)
        const steerX = float(0.0).toVar();
        const steerZ = float(0.0).toVar();
        const neighborCount = float(0.0).toVar();
        const sampleCount = float(0.0).toVar();
        const eps = float(0.0001);
        const eps2 = eps.mul(eps);
        const r = u.avoidRadius;
        const r2 = r.mul(r);

        // Compute our cell
        const gx = floor(s.x.sub(u.gridMinX).div(u.cellSize)).toVar();
        const gz = floor(s.z.sub(u.gridMinZ).div(u.cellSize)).toVar();
        const rx = clamp(gx, float(0.0), u.gridResX.sub(1.0));
        const rz = clamp(gz, float(0.0), u.gridResZ.sub(1.0));
        const ix = rx.toInt();
        const iz = rz.toInt();

        // Helper to sample one cell
        const sampleCell = (cx: any, cz: any) => {
          const insideX = cx.greaterThanEqual(0).and(cx.lessThan(u.gridResX.toInt()));
          const insideZ = cz.greaterThanEqual(0).and(cz.lessThan(u.gridResZ.toInt()));
          If(insideX.and(insideZ), () => {
            const cell = cz.mul(u.gridResX.toInt()).add(cx);
            const base = cell.mul(u.maxPerCell.toInt());
            const slot0 = gridIndices.element(base.add(0)).x;
            const slot1 = gridIndices.element(base.add(1)).x;
            const slot2 = gridIndices.element(base.add(2)).x;
            const slot3 = gridIndices.element(base.add(3)).x;
            const slot4 = gridIndices.element(base.add(4)).x;
            const slot5 = gridIndices.element(base.add(5)).x;
            const slot6 = gridIndices.element(base.add(6)).x;
            const slot7 = gridIndices.element(base.add(7)).x;
            const acc = (val: any) => {
              const cond = val.greaterThanEqual(0.0);
              If(cond.and(sampleCount.lessThan(u.neighborSamples)), () => {
                const j = val.toInt();
                const n = state.element(j);
                const dxn = s.x.sub(n.x);
                const dzn = s.z.sub(n.z);
                const dist2 = dxn.mul(dxn).add(dzn.mul(dzn));
                If(dist2.greaterThan(eps2).and(dist2.lessThan(r2)), () => {
                  const dist = length(vec3(dxn, float(0.0), dzn));
                  const inv = float(1.0).div(dist.add(eps));
                  const strength = r.sub(dist).mul(inv);
                  steerX.assign(steerX.add(dxn.mul(inv).mul(strength)));
                  steerZ.assign(steerZ.add(dzn.mul(inv).mul(strength)));
                  neighborCount.assign(neighborCount.add(1.0));
                });
                sampleCount.assign(sampleCount.add(1.0));
              });
            };
            acc(slot0); acc(slot1); acc(slot2); acc(slot3);
            acc(slot4); acc(slot5); acc(slot6); acc(slot7);
          });
        };

        // 3x3 neighborhood
        sampleCell(ix.sub(1), iz.sub(1));
        sampleCell(ix, iz.sub(1));
        sampleCell(ix.add(1), iz.sub(1));
        sampleCell(ix.sub(1), iz);
        sampleCell(ix, iz);
        sampleCell(ix.add(1), iz);
        sampleCell(ix.sub(1), iz.add(1));
        sampleCell(ix, iz.add(1));
        sampleCell(ix.add(1), iz.add(1));

        // Scale separation steering by configured strength and neighbor count (prevent over-steer)
        const invCount = float(1.0).div(neighborCount.add(1.0));
        const sepScale = u.avoidStrength.mul(invCount);

        // Attractor desired direction
        const toAx = u.attractorX.sub(s.x);
        const toAz = u.attractorZ.sub(s.z);
        // const distA2 = toAx.mul(toAx).add(toAz.mul(toAz)); // unused
        const distA = length(vec3(toAx, float(0.0), toAz));
        const inRange = distA.lessThanEqual(u.attractorRadius);
        const normAx = toAx.div(distA.add(eps));
        const normAz = toAz.div(distA.add(eps));
        // Falloff: closer -> stronger boost; clamp 0..1
        const fall = float(1.0).sub(clamp(distA.div(u.attractorRadius), float(0.0), float(1.0)));
        // Approximate pow with quadratic when falloff >= 1, else linear
        const useQuad = step(float(1.0), u.attractorFalloff);
        const fallPow = mix(fall, fall.mul(fall), useQuad);
        const attractStrength = fallPow.mul(u.attractorTurnBoost);
        const attrX = normAx.mul(attractStrength);
        const attrZ = normAz.mul(attractStrength);

        // Branch: yeeting vs walking
        // dead: do nothing (skip below branches)

        If(sf.greaterThan(0.5), () => {
          // YEETING: integrate ballistic motion and spin
          const nx = s.x.add(v.x.mul(u.deltaTime)).toVar();
          const ny = s.y.add(v.y.mul(u.deltaTime)).toVar();
          const nz = s.z.add(v.z.mul(u.deltaTime)).toVar();
          const yaw = s.w.add(v.w.mul(u.deltaTime)).toVar();
          const vy = v.y.sub(((u as any).yeetGravity).mul(u.deltaTime)).toVar();
          v.y.assign(vy);
          s.x.assign(nx);
          s.y.assign(ny);
          s.z.assign(nz);
          s.w.assign(yaw);
          const newLife = life.sub(u.deltaTime).toVar();
          lifes.element(instanceIndex).x.assign(newLife);
          If(newLife.lessThanEqual(0.0), () => { flags.element(instanceIndex).x.assign(float(2.0)); });
        }).Else(() => {
          // WALKING
          // Combine forward + separation + attractor (if enabled & in range)
          const baseDesX = fcx.add(steerX.mul(sepScale));
          const baseDesZ = fsz.add(steerZ.mul(sepScale));
          const desX = baseDesX.toVar();
          const desZ = baseDesZ.toVar();
          If(u.attractorEnabled.greaterThan(0).and(inRange), () => {
            desX.assign(desX.add(attrX));
            desZ.assign(desZ.add(attrZ));
          });
          // Normalize desired
          const desLen = length(vec3(desX, float(0.0), desZ));
          const ndx = desX.div(desLen.add(eps));
          const ndz = desZ.div(desLen.add(eps));

        // Turn toward desired using sign of 2D cross and magnitude factor (1 - dot)
          const dotfd = clamp(fcx.mul(ndx).add(fsz.mul(ndz)), float(-1.0), float(1.0));
          const crossfd = fcx.mul(ndz).sub(fsz.mul(ndx));
          const turnDir = sign(crossfd);
          // Boost turning when attractor influence is strong; do NOT change base speed
          const baseMaxTurn = u.turnRate.mul(u.deltaTime);
          const boost = float(1.0).toVar();
          If(u.attractorEnabled.greaterThan(0).and(inRange), () => {
            boost.assign(float(1.0).add(attractStrength));
          });
          const maxTurn = baseMaxTurn.mul(boost);
          const angleScale = float(1.0).sub(dotfd).clamp(float(0.0), float(1.0));
          const yaw = yaw0.add(turnDir.mul(maxTurn).mul(angleScale)).toVar();
          // Inject light per-instance yaw noise to break symmetry (uses u.randomness)
          const idxfN = float(instanceIndex);
          const noiseRaw = fract(sin(idxfN.mul(91.7).add(u.time.mul(0.61))).mul(43758.5453)).mul(2.0).sub(1.0);
          const yawJitter = noiseRaw.mul(u.randomness).mul(baseMaxTurn.mul(0.5));
          yaw.assign(yaw.add(yawJitter));

        // Forward step with base speed, modulated by walk-cycle phase
          const cx = cos(yaw);
          const sz = sin(yaw);

          // Per-variant walking speed (slight differences)
          const vIdxF2 = clamp(variant.element(instanceIndex).x, float(0.0), float(4.0));
          const f0 = float(0.90); // person 1: slowest
          const f1 = float(0.95);
          const f2 = float(1.00);
          const f3 = float(1.03);
          const f4 = float(1.06);
          const is1w = abs(vIdxF2.sub(1.0)).lessThan(0.5);
          const is2w = abs(vIdxF2.sub(2.0)).lessThan(0.5);
          const is3w = abs(vIdxF2.sub(3.0)).lessThan(0.5);
          const is4w = abs(vIdxF2.sub(4.0)).lessThan(0.5);
          const spMul = float(1.0).toVar();
          spMul.assign(f0);
          If(is1w, () => { spMul.assign(f1); });
          If(is2w, () => { spMul.assign(f2); });
          If(is3w, () => { spMul.assign(f3); });
          If(is4w, () => { spMul.assign(f4); });

        // Compute per-instance animation phase using shared frameRaw and per-instance instOff
        const instOff = (offsets as any).element(instanceIndex).x;
        const vCnt = float(0.0).toVar();
        vCnt.assign(float(48.0));
        If(abs(float(1.0).sub(vIdxF2)).lessThan(0.5), () => { vCnt.assign(float(38.0)); });
        If(abs(float(2.0).sub(vIdxF2)).lessThan(0.5), () => { vCnt.assign(float(29.0)); });
        If(abs(float(3.0).sub(vIdxF2)).lessThan(0.5), () => { vCnt.assign(float(31.0)); });
        If(abs(float(4.0).sub(vIdxF2)).lessThan(0.5), () => { vCnt.assign(float(33.0)); });
        const baseIdx = floor((u as any).frameRaw.add(instOff));
        const phase = fract(baseIdx.div(vCnt));
        const twoPi = float(6.283185307179586);
        const wave = sin(phase.mul(twoPi));
        const spPhase = float(1.0).add(((u as any).cycleAmp).mul(wave)).clamp(float(0.1), float(3.0));

        const dx = cx.mul(u.baseSpeed.mul(spMul).mul(spPhase)).mul(u.deltaTime);
        const dz = sz.mul(u.baseSpeed.mul(spMul).mul(spPhase)).mul(u.deltaTime);

          const minX = u.minX;
          const maxX = u.maxX;
          const minZ = u.minZ;
          const maxZ = u.maxZ;
          const nx = s.x.add(dx).toVar();
          const nz = s.z.add(dz).toVar();

          // Soft push-out after movement using grid lookup again
          const pushX = float(0.0).toVar();
          const pushZ = float(0.0).toVar();
          const minSep = r; // reuse avoid radius as minimum separation
          const minSep2 = minSep.mul(minSep);
          const pushCount = float(0.0).toVar();
          const pushCell = (cx: any, cz: any) => {
            const insideX = cx.greaterThanEqual(0).and(cx.lessThan(u.gridResX.toInt()));
            const insideZ = cz.greaterThanEqual(0).and(cz.lessThan(u.gridResZ.toInt()));
            If(insideX.and(insideZ), () => {
              const cell = cz.mul(u.gridResX.toInt()).add(cx);
              const base = cell.mul(u.maxPerCell.toInt());
              const accP = (slotIdx: number) => {
                const val = gridIndices.element(base.add(slotIdx)).x;
                If(val.greaterThanEqual(0.0).and(pushCount.lessThan(u.neighborSamples)), () => {
                  const j = val.toInt();
                  const n = state.element(j);
                  const dxp = nx.sub(n.x);
                  const dzp = nz.sub(n.z);
                  const d2 = dxp.mul(dxp).add(dzp.mul(dzp));
                  If(d2.greaterThan(eps.mul(eps)).and(d2.lessThan(minSep2)), () => {
                    const d = length(vec3(dxp, float(0.0), dzp));
                    const over = minSep.sub(d);
                    const invd = float(1.0).div(d.add(eps));
                    pushX.assign(pushX.add(dxp.mul(invd).mul(over)));
                    pushZ.assign(pushZ.add(dzp.mul(invd).mul(over)));
                    pushCount.assign(pushCount.add(1.0));
                  });
                });
              };
              accP(0); accP(1); accP(2); accP(3); accP(4); accP(5); accP(6); accP(7);
            });
          };
          const nix = floor(nx.sub(u.gridMinX).div(u.cellSize)).toInt();
          const niz = floor(nz.sub(u.gridMinZ).div(u.cellSize)).toInt();
          pushCell(nix.sub(1), niz.sub(1));
          pushCell(nix, niz.sub(1));
          pushCell(nix.add(1), niz.sub(1));
          pushCell(nix.sub(1), niz);
          pushCell(nix, niz);
          pushCell(nix.add(1), niz);
          pushCell(nix.sub(1), niz.add(1));
          pushCell(nix, niz.add(1));
          pushCell(nix.add(1), niz.add(1));

          // Scale push by deltaTime to avoid over-correction and frame dependency
          const nnx = nx.add(pushX.mul(u.pushStrength).mul(u.deltaTime)).toVar();
          const nnz = nz.add(pushZ.mul(u.pushStrength).mul(u.deltaTime)).toVar();

          // World bounds with bounce (mirror yaw)
          const pi = float(3.141592653589793);
          If(nnx.lessThan(minX), () => { nnx.assign(minX); yaw.assign(pi.sub(yaw)); });
          If(nnx.greaterThan(maxX), () => { nnx.assign(maxX); yaw.assign(pi.sub(yaw)); });
          If(nnz.lessThan(minZ), () => { nnz.assign(minZ); yaw.assign(yaw.mul(-1.0)); });
          If(nnz.greaterThan(maxZ), () => { nnz.assign(maxZ); yaw.assign(yaw.mul(-1.0)); });

          s.x.assign(nnx);
          // Per-variant vertical offset so each size rests on ground
          const vIdxFh = clamp(variant.element(instanceIndex).x, float(0.0), float(4.0));
          const h0 = float(1.0);
          const h1 = float(0.65);
          const h2 = float(0.50);
          const h3 = float(0.50);
          const h4 = float(0.50);
          const is1h = abs(vIdxFh.sub(1.0)).lessThan(0.5);
          const is2h = abs(vIdxFh.sub(2.0)).lessThan(0.5);
          const is3h = abs(vIdxFh.sub(3.0)).lessThan(0.5);
          const is4h = abs(vIdxFh.sub(4.0)).lessThan(0.5);
          const hMul = float(1.0).toVar();
          hMul.assign(h0);
          If(is1h, () => { hMul.assign(h1); });
          If(is2h, () => { hMul.assign(h2); });
          If(is3h, () => { hMul.assign(h3); });
          If(is4h, () => { hMul.assign(h4); });
          // Sample terrain height at final (nnx, nnz)
          // Match fBm-ish ground function for consistent foot placement
          const f1h = (u as any).terrainHillFreq;
          const f2h = f1h.mul(2.0);
          const f3h = f1h.mul(4.0);
          const f4h = f1h.mul(8.0);
          const a1 = float(0.6);
          const a2 = float(0.3);
          const a3 = float(0.15);
          const a4 = float(0.075);
          const y1 = sin(nnx.mul(f1h)).mul(0.5).add(cos(nnz.mul(f1h)).mul(0.5));
          const y2 = sin(nnx.mul(f2h)).mul(0.5).add(cos(nnz.mul(f2h)).mul(0.5));
          const y3 = sin(nnx.mul(f3h)).mul(0.5).add(cos(nnz.mul(f3h)).mul(0.5));
          const y4 = sin(nnx.mul(f4h)).mul(0.5).add(cos(nnz.mul(f4h)).mul(0.5));
          const sum = y1.mul(a1).add(y2.mul(a2)).add(y3.mul(a3)).add(y4.mul(a4));
          const diag = sin(nnx.add(nnz).mul(f3h.mul(0.7071))).mul(0.1);
          const h = sum.add(diag).mul(((u as any).terrainHillAmp));
          s.y.assign(u.yHeight.mul(hMul).add(h));
          s.z.assign(nnz);
          s.w.assign(yaw);

          // Arrival → trigger YEET
          If(((u as any).yeetEnabled).greaterThan(0).and(u.attractorEnabled.greaterThan(0)).and(inRange).and(distA.lessThanEqual((u as any).yeetArrivalDist)), () => {
            flags.element(instanceIndex).x.assign(float(1.0));
            // Randomize direction and spin
            const idxf = float(instanceIndex);
            const h1 = fract(sin(idxf.mul(12.9898).add(u.time.mul(78.233))).mul(43758.5453));
            const h2 = fract(sin(idxf.mul(93.9898).add(u.time.mul(11.233))).mul(24634.6345));
            const ang = h1.mul(float(6.283185307179586));
            const sp = (u as any).yeetSpeed;
            const hf = (u as any).yeetHorizFrac;
            const vx = cos(ang).mul(sp).mul(hf);
            const vz = sin(ang).mul(sp).mul(hf);
            const vy = sp; // main upward impulse
            v.x.assign(vx);
            v.y.assign(vy);
            v.z.assign(vz);
            v.w.assign(h2.mul(float(2.0)).sub(float(1.0)).mul((u as any).yeetSpin));
            lifes.element(instanceIndex).x.assign((u as any).yeetLife);
          });
        });
      })();

      this._walkCompute = compute(walkKernel, this.count, [256]);
    }

    // Update bounds/y params now
    this._walkUniforms.minX.value = options.minX;
    this._walkUniforms.maxX.value = options.maxX;
    this._walkUniforms.minZ.value = options.minZ;
    this._walkUniforms.maxZ.value = options.maxZ;
    this._walkUniforms.yHeight.value = options.y;
    if (options.baseSpeed !== undefined) this._walkUniforms.baseSpeed.value = options.baseSpeed;
    if (options.turnRate !== undefined) this._walkUniforms.turnRate.value = options.turnRate;
    if (options.randomness !== undefined) this._walkUniforms.randomness.value = options.randomness;
    if (options.avoidanceRadius !== undefined) this._walkUniforms.avoidRadius.value = options.avoidanceRadius;
    if (options.avoidanceStrength !== undefined) this._walkUniforms.avoidStrength.value = options.avoidanceStrength;
    if (options.neighborSamples !== undefined) this._walkUniforms.neighborSamples.value = Math.max(1, Math.min(8, options.neighborSamples | 0));
    if (options.pushStrength !== undefined) this._walkUniforms.pushStrength.value = options.pushStrength;
    // Grid setup: pick cell size from avoidanceRadius, compute resolution, (re)allocate storage
    const cellSize = (options.avoidanceRadius ?? this._walkUniforms.avoidRadius.value) * 1.0;
    const worldX = Math.max(1, options.maxX - options.minX);
    const worldZ = Math.max(1, options.maxZ - options.minZ);
    const resX = Math.max(1, Math.ceil(worldX / cellSize));
    const resZ = Math.max(1, Math.ceil(worldZ / cellSize));
    const maxPerCell = Math.max(1, (options.gridMaxPerCell ?? this._walkUniforms.maxPerCell.value) | 0);
    const binPasses = Math.max(1, (options.binPasses ?? this._walkUniforms.binPasses.value) | 0);
    this._walkUniforms.gridMinX.value = options.minX;
    this._walkUniforms.gridMinZ.value = options.minZ;
    this._walkUniforms.cellSize.value = cellSize;
    this._walkUniforms.gridResX.value = resX;
    this._walkUniforms.gridResZ.value = resZ;
    this._walkUniforms.maxPerCell.value = maxPerCell;
    this._walkUniforms.binPasses.value = binPasses;

    // (Re)allocate gridIndices buffer if size changed
    const totalSlots = resX * resZ * maxPerCell;
    const gridIndicesNode = (this as any)._gridIndicesStorageRef;
    if (!this._gridIndicesStorage || (this._gridIndicesStorage?.value?.array?.length | 0) !== totalSlots) {
      const arr = new Float32Array(totalSlots);
      for (let i = 0; i < totalSlots; i++) arr[i] = -1;
      this._gridIndicesStorage = storage(new StorageInstancedBufferAttribute(arr, 1));
      // Rebind the internal node to the new storage attribute
      (gridIndicesNode as any).value = this._gridIndicesStorage.value;
      // Recreate clear compute sized to grid slots
      const clearKernel = Fn(() => {
        const idx = instanceIndex; const e = (gridIndicesNode as any).element(idx); e.x.assign(float(-1.0));
      })();
      this._gridClearCompute = compute(clearKernel, totalSlots, [256]);
    } else {
      // Only resize clear dispatch size if needed
      const total = totalSlots;
      const clearKernel = Fn(() => {
        const idx = instanceIndex; const e = (gridIndicesNode as any).element(idx); e.x.assign(float(-1.0));
      })();
      this._gridClearCompute = compute(clearKernel, total, [256]);
    }
  }

  setAttractor(options: { enabled: boolean; x: number; z: number; radius?: number; turnBoost?: number; falloff?: number }): void {
    if (!this._walkUniforms) return;
    this._walkUniforms.attractorEnabled.value = options.enabled ? 1 : 0;
    this._walkUniforms.attractorX.value = options.x;
    this._walkUniforms.attractorZ.value = options.z;
    if (options.radius !== undefined) this._walkUniforms.attractorRadius.value = Math.max(0.0001, options.radius);
    if (options.turnBoost !== undefined) this._walkUniforms.attractorTurnBoost.value = Math.max(0.0, options.turnBoost);
    if (options.falloff !== undefined) this._walkUniforms.attractorFalloff.value = Math.max(0.1, options.falloff);
  }

  setYeetEnabled(enabled: boolean): void {
    if (!this._walkUniforms) return;
    this._walkUniforms.yeetEnabled.value = enabled ? 1 : 0;
  }

  setYeetParams(params: { arrivalDist?: number; speed?: number; horizFrac?: number; life?: number; gravity?: number; spin?: number }): void {
    if (!this._walkUniforms) return;
    if (params.arrivalDist !== undefined) this._walkUniforms.yeetArrivalDist.value = Math.max(0, params.arrivalDist);
    if (params.speed !== undefined) this._walkUniforms.yeetSpeed.value = Math.max(0, params.speed);
    if (params.horizFrac !== undefined) this._walkUniforms.yeetHorizFrac.value = Math.min(1, Math.max(0, params.horizFrac));
    if (params.life !== undefined) this._walkUniforms.yeetLife.value = Math.max(0, params.life);
    if (params.gravity !== undefined) this._walkUniforms.yeetGravity.value = Math.max(0, params.gravity);
    if (params.spin !== undefined) this._walkUniforms.yeetSpin.value = params.spin;
  }

  resetYeetStates(): void {
    if (!this._instanceStateFlagsStorage || !this._instanceLifeStorage || !this._instanceVelocityStorage) return;
    const n = this.count | 0;
    const flags = this._instanceStateFlagsStorage.value.array as Float32Array;
    const lifes = this._instanceLifeStorage.value.array as Float32Array;
    const vels = this._instanceVelocityStorage.value.array as Float32Array;
    for (let i = 0; i < n; i++) {
      flags[i] = 0;
      lifes[i] = 0;
      const vi = i * 4;
      vels[vi + 0] = 0; vels[vi + 1] = 0; vels[vi + 2] = 0; vels[vi + 3] = 0;
    }
    this._instanceStateFlagsStorage.value.needsUpdate = true;
    this._instanceLifeStorage.value.needsUpdate = true;
    this._instanceVelocityStorage.value.needsUpdate = true;
  }

  updateWalking(renderer: any, deltaTime: number, timeSeconds: number): void {
    if (!this._walkCompute || !this._walkUniforms) return;
    this._walkUniforms.deltaTime.value = Math.max(0, deltaTime);
    this._walkUniforms.time.value = Math.max(0, timeSeconds);
    // 1) Clear grid indices to -1
    if (this._gridClearCompute) (renderer as any).compute?.(this._gridClearCompute);
    // 2) Bin all agents into grid
    if (this._gridBinCompute) (renderer as any).compute?.(this._gridBinCompute);
    // (Optional) second bin pass to reduce slot collisions
    if ((this._walkUniforms.binPasses.value | 0) > 1 && this._gridBinCompute) (renderer as any).compute?.(this._gridBinCompute);
    // 3) Steer/move using grid
    (renderer as any).compute?.(this._walkCompute);
    // Resolve compute timestamp queries to avoid exhausting the query pool
    try {
      (renderer as any).resolveTimestampsAsync?.((THREE as any).TimestampQuery?.COMPUTE)?.catch?.(() => {});
    } catch {}
  }
}





