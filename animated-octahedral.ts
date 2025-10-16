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
  storage
} from 'three/tsl';

interface AnimatedOctahedralConfig {
  spritesPerSide?: number;
  transparent?: boolean;
  alphaClamp?: number;
  useHemiOctahedron?: boolean;
  scale?: number;
  flipY?: boolean;
  flipSpriteX?: boolean;
  flipSpriteY?: boolean;
  swapSpriteAxes?: boolean;
  // Palette indexing
  paletteTexture?: THREE.Texture;
  paletteSize?: number; // number of colors (palette width)
  paletteRows?: number; // number of palette rows (height)
  paletteRowIndex?: number; // which row to use (non-instanced variant)
  paletteData?: Float32Array; // optional: linear RGBA palette data as LUT
}

function createAnimatedArrayOctahedralMaterial(
  arrayTexture: THREE.Texture,
  parameters: AnimatedOctahedralConfig,
  normalArrayTexture?: THREE.Texture | null
): any {
  // Use a lit material so normals contribute to lighting
  const material = new (THREE as any).MeshStandardNodeMaterial();
  material.transparent = parameters.transparent ?? true;
  material.metalness = 0.0;
  material.roughness = 0.7;

  // Uniforms
  const spritesPerSide = uniform(parameters.spritesPerSide ?? 16);
  const alphaClamp = uniform(parameters.alphaClamp ?? 0.05);
  const useHemiOctahedron = uniform(parameters.useHemiOctahedron ? 1 : 0);
  const frameIndex = uniform(0);
  // Single impostor: allow selecting a variant and sampling within its base/count
  const selectedVariant = uniform(0); // 0..4
  // Inline constants for counts and bases
  const c0 = float(48.0); const c1 = float(38.0); const c2 = float(29.0); const c3 = float(31.0); const c4 = float(33.0);
  const b0 = float(0.0);  const b1 = float(48.0); const b2 = float(86.0); const b3 = float(115.0); const b4 = float(146.0);
  const globalScale = uniform(parameters.scale ?? 1);
  const flipYFlag = uniform(parameters.flipY ? 1 : 0);
  const flipSpriteXFlag = uniform(parameters.flipSpriteX ? 1 : 0);
  const flipSpriteYFlag = uniform(parameters.flipSpriteY ? 1 : 0);
  const swapSpriteAxesFlag = uniform(parameters.swapSpriteAxes ? 1 : 0);

  // Palette uniforms
  const hasPalette = !!(parameters.paletteTexture || parameters.paletteData);
  const paletteSize = uniform(parameters.paletteSize ?? 32);
  const paletteRows = uniform(parameters.paletteRows ?? 1);
  const paletteRowIndex = uniform(parameters.paletteRowIndex ?? 0);
  // Optional palette LUT buffer (vec4 per entry)
  const paletteBuffer = (parameters.paletteData && (THREE as any).StorageBufferAttribute)
    ? storage(new (THREE as any).StorageBufferAttribute(parameters.paletteData, 4))
    : null;

  // Varyings
  const vSprite = varying(vec2(), 'vSprite');
  const vSpriteUV = varying(vec2(), 'vSpriteUV');

  // Vertex: billboarding + octahedral sprite selection
  material.positionNode = Fn(() => {
    const spritesMinusOne = vec2(spritesPerSide.sub(1.0));

    const cameraPosLocal = cameraPosition; // impostor at origin in local

    // No Y dampening: always face true camera direction
    const cameraDir = normalize(vec3(cameraPosLocal.x, cameraPosLocal.y, cameraPosLocal.z));

    let up = vec3(0.0, 1.0, 0.0).toVar();
    If(useHemiOctahedron, () => {
      up.assign(mix(up, vec3(-1.0, 0.0, 0.0), step(0.999, cameraDir.y)));
    }).Else(() => {
      up.assign(mix(up, vec3(-1.0, 0.0, 0.0), step(0.999, cameraDir.y)));
      up.assign(mix(up, vec3(1.0, 0.0, 0.0), step(cameraDir.y, -0.999)));
    });

    const tangent = normalize(cross(up, cameraDir));
    const bitangent = cross(cameraDir, tangent);
    const projectedVertex = tangent.mul(positionLocal.x.mul(globalScale)).add(bitangent.mul(positionLocal.y.mul(globalScale)));

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

    // Apply atlas orientation adjustments (flip X/Y, optional axis swap)
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

  // Fragment: sample array layer by frameIndex (flip Y) and remap via palette if provided
  material.colorNode = Fn(() => {
    const frameSize = float(1.0).div(spritesPerSide);
    const uvY = mix(vSpriteUV.y, float(1.0).sub(vSpriteUV.y), flipYFlag);
    const uv = vec2(vSpriteUV.x, uvY);
    const spriteUV = frameSize.mul(vSprite.add(clamp(uv, vec2(0), vec2(1))));
    // Per-variant addressing for single impostor
    const vIdx = clamp(selectedVariant, float(0.0), float(4.0)).toInt();
    const vCountSel = float(0.0).toVar();
    vCountSel.assign(c0);
    If(abs(float(1.0).sub(vIdx)).lessThan(0.5), () => { vCountSel.assign(c1); });
    If(abs(float(2.0).sub(vIdx)).lessThan(0.5), () => { vCountSel.assign(c2); });
    If(abs(float(3.0).sub(vIdx)).lessThan(0.5), () => { vCountSel.assign(c3); });
    If(abs(float(4.0).sub(vIdx)).lessThan(0.5), () => { vCountSel.assign(c4); });
    const vBaseSel = float(0.0).toVar();
    vBaseSel.assign(b0);
    If(abs(float(1.0).sub(vIdx)).lessThan(0.5), () => { vBaseSel.assign(b1); });
    If(abs(float(2.0).sub(vIdx)).lessThan(0.5), () => { vBaseSel.assign(b2); });
    If(abs(float(3.0).sub(vIdx)).lessThan(0.5), () => { vBaseSel.assign(b3); });
    If(abs(float(4.0).sub(vIdx)).lessThan(0.5), () => { vBaseSel.assign(b4); });
    const baseIdx = floor(frameIndex);
    const divF = baseIdx.div(vCountSel);
    const fracF = divF.sub(floor(divF));
    const localIdx = floor(fracF.mul(vCountSel)).toInt();
    const finalIdx = vBaseSel.toInt().add(localIdx);
    const sampleNode = (textureNode as any)(arrayTexture, spriteUV).depth((finalIdx as any).toInt());

    // If palette is provided, treat red channel as 8-bit index and look up in palette
    if (hasPalette) {
      // Convert normalized channel to 0..255 index, then clamp to paletteSize-1
      const idx255 = floor(sampleNode.x.mul(255.0).add(0.5));
      const maxIdx = paletteSize.sub(1.0);
      const clampedIdx = clamp(idx255, float(0.0), maxIdx);
      const row = clamp(paletteRowIndex, float(0.0), paletteRows.sub(1.0));
      // Prefer LUT buffer if available; fallback to palette texture
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

    // Fallback: direct color
    const spriteColor = sampleNode;
    If(spriteColor.a.lessThanEqual(alphaClamp), () => { Discard(); });
    return spriteColor;
  })();

  // Normals: optional normal array texture per frame with identical atlas layout
  if (normalArrayTexture) {
    material.normalNode = Fn(() => {
      const frameSize = float(1.0).div(spritesPerSide);
      const uvY = mix(vSpriteUV.y, float(1.0).sub(vSpriteUV.y), flipYFlag);
      const uv = vec2(vSpriteUV.x, uvY);
      const spriteUV = frameSize.mul(vSprite.add(clamp(uv, vec2(0), vec2(1))));
      const vIdx = clamp(selectedVariant, float(0.0), float(4.0)).toInt();
      const vCountSel = float(0.0).toVar();
      vCountSel.assign(c0);
      If(abs(float(1.0).sub(vIdx)).lessThan(0.5), () => { vCountSel.assign(c1); });
      If(abs(float(2.0).sub(vIdx)).lessThan(0.5), () => { vCountSel.assign(c2); });
      If(abs(float(3.0).sub(vIdx)).lessThan(0.5), () => { vCountSel.assign(c3); });
      If(abs(float(4.0).sub(vIdx)).lessThan(0.5), () => { vCountSel.assign(c4); });
      const vBaseSel = float(0.0).toVar();
      vBaseSel.assign(b0);
      If(abs(float(1.0).sub(vIdx)).lessThan(0.5), () => { vBaseSel.assign(b1); });
      If(abs(float(2.0).sub(vIdx)).lessThan(0.5), () => { vBaseSel.assign(b2); });
      If(abs(float(3.0).sub(vIdx)).lessThan(0.5), () => { vBaseSel.assign(b3); });
      If(abs(float(4.0).sub(vIdx)).lessThan(0.5), () => { vBaseSel.assign(b4); });
      const baseIdx = floor(frameIndex);
      const divF = baseIdx.div(vCountSel);
      const fracF = divF.sub(floor(divF));
      const localIdx = floor(fracF.mul(vCountSel)).toInt();
      const finalIdx = vBaseSel.toInt().add(localIdx);
      const normalSample = (textureNode as any)(normalArrayTexture, spriteUV).depth((finalIdx as any).toInt());
      return normalize(normalSample.xyz);
    })();
  }

  material.animatedImpostorUniforms = {
    spritesPerSide,
    alphaClamp,
    frameIndex,
    globalScale,
    flipYFlag,
    flipSpriteXFlag,
    flipSpriteYFlag,
    swapSpriteAxesFlag,
    paletteSize,
    paletteRows,
    paletteRowIndex,
    selectedVariant
  };

  return material;
}

const PLANE_GEOMETRY = new THREE.PlaneGeometry();

export class AnimatedOctahedralImpostor extends THREE.Mesh<any, any> {
  private _frameCount: number;

  constructor(
    arrayTexture: THREE.Texture,
    parameters: AnimatedOctahedralConfig & { frameCount: number },
    normalArrayTexture?: THREE.Texture | null
  ) {
    const mat = createAnimatedArrayOctahedralMaterial(arrayTexture, parameters, normalArrayTexture);
    super(PLANE_GEOMETRY, mat);
    this.frustumCulled = false;
    this._frameCount = Math.max(1, parameters.frameCount);
  }

  setFrame(index: number): void {
    const uniforms = (this.material as any).animatedImpostorUniforms;
    if (uniforms) {
      uniforms.frameIndex.value = Math.max(0, Math.floor(index));
    }
  }

  setSpritesPerSide(value: number): void {
    const uniforms = (this.material as any).animatedImpostorUniforms;
    if (uniforms) {
      uniforms.spritesPerSide.value = value;
    }
  }

  setAlphaClamp(value: number): void {
    const uniforms = (this.material as any).animatedImpostorUniforms;
    if (uniforms) {
      uniforms.alphaClamp.value = value;
    }
  }

  setScale(value: number): void {
    const uniforms = (this.material as any).animatedImpostorUniforms;
    if (uniforms) {
      uniforms.globalScale.value = value;
    }
  }

  setFlipY(value: boolean): void {
    const uniforms = (this.material as any).animatedImpostorUniforms;
    if (uniforms) {
      uniforms.flipYFlag.value = value ? 1 : 0;
    }
  }

  setPaletteRowIndex(value: number): void {
    const uniforms = (this.material as any).animatedImpostorUniforms;
    if (uniforms) {
      uniforms.paletteRowIndex.value = Math.max(0, value | 0);
    }
  }

  setAtlasFlipAndSwap(options: { flipX?: boolean; flipY?: boolean; swapAxes?: boolean }): void {
    const uniforms = (this.material as any).animatedImpostorUniforms;
    if (!uniforms) return;
    if (options.flipX !== undefined) uniforms.flipSpriteXFlag.value = options.flipX ? 1 : 0;
    if (options.flipY !== undefined) uniforms.flipSpriteYFlag.value = options.flipY ? 1 : 0;
    if (options.swapAxes !== undefined) uniforms.swapSpriteAxesFlag.value = options.swapAxes ? 1 : 0;
  }

  get frameCount(): number {
    return this._frameCount;
  }
}



