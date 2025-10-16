import * as THREE from 'three/webgpu';
import RAPIER from '@dimforge/rapier3d-compat';
import { uv, texture as texNode, uniform, Fn, vec2, vec4, float, floor, fract, sin, cos, dot, positionLocal } from 'three/tsl';

// ============================================================================
// INTERFACES & TYPES
// ============================================================================

interface PhysicsWorld {
  world: RAPIER.World;
  rigidBodies: Map<THREE.Object3D, RAPIER.RigidBody>;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Setup scene lighting
export function setupLights(scene: THREE.Scene) {
  const ambientLight = new THREE.AmbientLight(0xFFFFFF, 1.0);
  scene.add(ambientLight);
  
  const dirLight = new THREE.DirectionalLight(0xFFFFFF, 10.0);
  dirLight.position.set(5, 10, 7.5);
  scene.add(dirLight);

  return { ambientLight, directionalLight: dirLight };
}

// Create ground plane with physics
export function createGround(physics: PhysicsWorld, size: number = 1200, options?: { hillAmp?: number; hillFreq?: number }) {
  const groundGeometry = new THREE.PlaneGeometry(size, size, 512, 512);
  groundGeometry.rotateX(-Math.PI / 2);
  
  // Tileable albedo for ground
  const tileWorldSize = 0.8; // meters per tile (5x more tiling vs 4)
  const repeat = Math.max(1, Math.floor(size / tileWorldSize));
  const tex = new THREE.TextureLoader().load('./textures/ground.png');
  tex.wrapS = (THREE as any).RepeatWrapping;
  tex.wrapT = (THREE as any).RepeatWrapping;
  tex.minFilter = (THREE as any).LinearMipmapLinearFilter;
  tex.magFilter = (THREE as any).LinearFilter;
  tex.colorSpace = (THREE as any).SRGBColorSpace;
  tex.generateMipmaps = true;
  tex.anisotropy = 16 as any;
  tex.needsUpdate = true;

  // Texture bombing (random per-tile rotation + jitter) using NodeMaterial (WebGPU-friendly)
  const repeatU = uniform(repeat);
  const groundMaterial = new (THREE as any).MeshStandardNodeMaterial();
  groundMaterial.metalness = 0.2;
  groundMaterial.roughness = 0.8;
  const hillAmpU = uniform(Math.max(0, options?.hillAmp ?? 1.2));
  const hillFreqU = uniform(Math.max(0.00001, options?.hillFreq ?? 0.08));
  groundMaterial.positionNode = Fn(() => {
    const p = positionLocal;
    // 4-octave fBm: finer detail hills
    const f1 = hillFreqU;
    const f2 = hillFreqU.mul(2.0);
    const f3 = hillFreqU.mul(4.0);
    const f4 = hillFreqU.mul(8.0);
    const a1 = float(0.6);
    const a2 = float(0.3);
    const a3 = float(0.15);
    const a4 = float(0.075);
    const y1 = sin(p.x.mul(f1)).mul(0.5).add(cos(p.z.mul(f1)).mul(0.5));
    const y2 = sin(p.x.mul(f2)).mul(0.5).add(cos(p.z.mul(f2)).mul(0.5));
    const y3 = sin(p.x.mul(f3)).mul(0.5).add(cos(p.z.mul(f3)).mul(0.5));
    const y4 = sin(p.x.mul(f4)).mul(0.5).add(cos(p.z.mul(f4)).mul(0.5));
    const sum = y1.mul(a1).add(y2.mul(a2)).add(y3.mul(a3)).add(y4.mul(a4));
    // slight diagonal component for realism
    const diag = sin(p.x.add(p.z).mul(f3.mul(0.7071))).mul(0.1);
    const yDisp = sum.add(diag).mul(hillAmpU);
    return vec4(p.x, p.y.add(yDisp), p.z, 1.0);
  })();
  groundMaterial.colorNode = Fn(() => {
    const baseUV = uv();
    const uvScaled = baseUV.mul(repeatU);
    const cell = floor(uvScaled).toVar();
    const st = fract(uvScaled).sub(vec2(0.5)).toVar();

    // Hash-based random angle per tile
    const n = dot(cell, vec2(127.1, 311.7));
    const rnd = fract(sin(n).mul(43758.5453123));
    const angle = rnd.mul(float(6.283185307179586)); // 2*PI
    const s = sin(angle);
    const c = cos(angle);

    // Rotate local UV around tile center
    const rx = st.x.mul(c).sub(st.y.mul(s));
    const ry = st.x.mul(s).add(st.y.mul(c));
    st.assign(vec2(rx, ry).add(vec2(0.5)));

    // 2D jitter per tile
    const n1 = dot(cell.add(vec2(1.0, 3.0)), vec2(12.9898, 78.233));
    const n2 = dot(cell.add(vec2(5.0, 7.0)), vec2(39.3467, 11.135));
    const jx = fract(sin(n1).mul(43758.5453)).sub(0.5);
    const jy = fract(sin(n2).mul(24634.6345)).sub(0.5);
    const jitterStrength = float(0.08);
    const stJittered = fract(st.add(vec2(jx, jy).mul(jitterStrength)));

    // Sample with per-tile UV offset so tiles repeat across the plane
    const uvFinal = stJittered.add(cell);
    const sample = texNode(tex, uvFinal);
    return sample.rgb; // feed into standard material baseColor
  })();
  
  const groundMesh = new THREE.Mesh(groundGeometry, groundMaterial);
  groundMesh.receiveShadow = true;
  
  const groundBodyDesc = RAPIER.RigidBodyDesc.fixed();
  const groundBody = physics.world.createRigidBody(groundBodyDesc);
  
  const groundColliderDesc = RAPIER.ColliderDesc.cuboid(size / 2, 0.1, size / 2);
  // Make the collider conform to a flat ground but keep visual displacement only for now
  physics.world.createCollider(groundColliderDesc, groundBody);
  
  physics.rigidBodies.set(groundMesh, groundBody);
  return groundMesh;
}

// ============================================================================
// INPUT HANDLER
// ============================================================================

export class InputHandler {
  public update(): void {
    // Minimal input handler - functionality moved to FPSController
  }
}

// ============================================================================
// FPS CONTROLLER
// ============================================================================

export class FPSController {
  object: THREE.Object3D;
  camera: THREE.Camera;
  physics: PhysicsWorld;
  rigidBody: RAPIER.RigidBody;
  collider: RAPIER.Collider;
  characterController: RAPIER.KinematicCharacterController;
  domElement: HTMLElement;
  pitchObject: THREE.Object3D;
  yawObject: THREE.Object3D;
  isLocked: boolean;
  position: THREE.Vector3;
  scene: THREE.Scene | null;
  
  // Movement state
  moveForward = false;
  moveBackward = false;
  moveLeft = false;
  moveRight = false;
  verticalVelocity = 0;
  
  // Movement parameters
  moveSpeed = 50.0;
  jumpVelocity = 50.0;
  gravityForce = 20.0;

  constructor(camera: THREE.Camera, physics: PhysicsWorld, domElement: HTMLElement) {
    this.camera = camera;
    this.physics = physics;
    this.domElement = domElement;
    this.isLocked = false;
    this.scene = null;

    // Create kinematic rigid body for player
    const position = new RAPIER.Vector3(0, 5, 10);
    const bodyDesc = RAPIER.RigidBodyDesc.kinematicPositionBased()
      .setTranslation(position.x, position.y, position.z);

    this.rigidBody = physics.world.createRigidBody(bodyDesc);
    
    // Create capsule collider
    const colliderDesc = RAPIER.ColliderDesc.capsule(0.9, 0.3);
    this.collider = physics.world.createCollider(colliderDesc, this.rigidBody);

    // Create character controller
    this.characterController = physics.world.createCharacterController(0.01);
    this.characterController.enableAutostep(0.5, 0.3, true);
    this.characterController.enableSnapToGround(0.3);

    // Create 3D objects for camera control
    this.pitchObject = new THREE.Object3D();
    this.pitchObject.add(camera);

    this.yawObject = new THREE.Object3D();
    this.yawObject.position.set(position.x, position.y, position.z);
    this.yawObject.add(this.pitchObject);

    this.object = this.yawObject;
    this.position = this.yawObject.position;

    // Setup controls
    this.setupPointerLock();
    document.addEventListener('keydown', this.onKeyDown.bind(this));
    document.addEventListener('keyup', this.onKeyUp.bind(this));
  }

  setScene(scene: THREE.Scene) {
    this.scene = scene;
  }

  setupPointerLock() {
    const lockChangeEvent = () => {
      this.isLocked = document.pointerLockElement === this.domElement;
    };

    const moveCallback = (event: MouseEvent) => {
      if (!this.isLocked) return;

      this.yawObject.rotation.y -= event.movementX * 0.002;
      this.pitchObject.rotation.x -= event.movementY * 0.002;
      this.pitchObject.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.pitchObject.rotation.x));
    };

    document.addEventListener('pointerlockchange', lockChangeEvent);
    document.addEventListener('mousemove', moveCallback);
  }

  onKeyDown(event: KeyboardEvent) {
    switch (event.code) {
      case 'KeyW': case 'ArrowUp': this.moveForward = true; break;
      case 'KeyS': case 'ArrowDown': this.moveBackward = true; break;
      case 'KeyA': case 'ArrowLeft': this.moveLeft = true; break;
      case 'KeyD': case 'ArrowRight': this.moveRight = true; break;
      case 'Space': 
        if (this.characterController.computedGrounded()) {
          this.verticalVelocity = this.jumpVelocity;
        }
        break;
    }
  }

  onKeyUp(event: KeyboardEvent) {
    switch (event.code) {
      case 'KeyW': case 'ArrowUp': this.moveForward = false; break;
      case 'KeyS': case 'ArrowDown': this.moveBackward = false; break;
      case 'KeyA': case 'ArrowLeft': this.moveLeft = false; break;
      case 'KeyD': case 'ArrowRight': this.moveRight = false; break;
    }
  }

  update(deltaTime: number) {
    if (!this.rigidBody || !this.characterController) return;

    // Calculate movement direction
    const direction = new THREE.Vector3();
    if (this.moveForward) direction.z = -1;
    if (this.moveBackward) direction.z = 1;
    if (this.moveLeft) direction.x = -1;
    if (this.moveRight) direction.x = 1;
    
    if (direction.lengthSq() > 0) {
      direction.normalize();
      direction.applyAxisAngle(new THREE.Vector3(0, 1, 0), this.yawObject.rotation.y);
    }

    // Apply gravity
    if (!this.characterController.computedGrounded()) {
      this.verticalVelocity -= this.gravityForce * deltaTime;
    } else if (this.verticalVelocity < 0) {
      this.verticalVelocity = 0;
    }

    // Create movement vector
    const movementVector = {
      x: direction.x * this.moveSpeed * deltaTime,
      y: this.verticalVelocity * deltaTime,
      z: direction.z * this.moveSpeed * deltaTime
    };
    
    // Compute and apply movement
    this.characterController.computeColliderMovement(this.collider, movementVector);
    const correctedMovement = this.characterController.computedMovement();
    
    const currentPos = this.rigidBody.translation();
    const newPos = {
      x: currentPos.x + correctedMovement.x,
      y: currentPos.y + correctedMovement.y,
      z: currentPos.z + correctedMovement.z
    };
    
    this.rigidBody.setNextKinematicTranslation(newPos);
    this.position.set(newPos.x, newPos.y, newPos.z);
  }
}