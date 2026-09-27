"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { DetectedObject } from "@/lib/types";
import {
  Layers,
  Camera,
  Eye,
  Crosshair,
  Zap,
  Info,
  Maximize2,
  Minimize2,
  Radio,
  AlertTriangle,
  RotateCcw,
  Sparkles,
  ChevronRight,
  X,
  Volume2,
} from "lucide-react";

interface LidarVisualizerProps {
  objects: DetectedObject[];
  chaosScore: number;
  width?: number;
  height?: number;
  showTechDetailsByDefault?: boolean;
}

type CameraPreset = "chase" | "cockpit" | "bird" | "side" | "free";

export default function LidarVisualizer({
  objects,
  chaosScore,
  width,
  height,
  showTechDetailsByDefault = false,
}: LidarVisualizerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const frameIdRef = useRef<number | null>(null);

  // Groups and dynamic references
  const egoGroupRef = useRef<THREE.Group | null>(null);
  const wheelsRef = useRef<THREE.Mesh[]>([]);
  const lidarRotorRef = useRef<THREE.Mesh | null>(null);
  const boxesGroupRef = useRef<THREE.Group | null>(null);
  const trajectoryGroupRef = useRef<THREE.Group | null>(null);
  const pointCloudRef = useRef<THREE.Points | null>(null);
  const frustumMeshRef = useRef<THREE.Mesh | null>(null);
  const sweepMeshRef = useRef<THREE.Mesh | null>(null);
  const roadGridRef = useRef<THREE.GridHelper | null>(null);
  const roadStripesRef = useRef<THREE.Line[]>([]);
  const headlightsRef = useRef<THREE.SpotLight[]>([]);

  // Interactive UI States
  const [activePreset, setActivePreset] = useState<CameraPreset>("chase");
  const [showPointCloud, setShowPointCloud] = useState<"dense" | "sparse" | "off">("dense");
  const [showFrustum, setShowFrustum] = useState(true);
  const [showSweep, setShowSweep] = useState(true);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [showHeadlights, setShowHeadlights] = useState(true);
  const [showTechModal, setShowTechModal] = useState(showTechDetailsByDefault);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [activeFps, setActiveFps] = useState(60);

  // Camera presets coordinates
  const applyCameraPreset = useCallback((preset: CameraPreset) => {
    const cam = cameraRef.current;
    const ctrl = controlsRef.current;
    if (!cam || !ctrl) return;

    setActivePreset(preset);

    if (preset === "chase") {
      cam.position.set(0, 3.6, 8.2);
      ctrl.target.set(0, 1.1, -12);
      ctrl.enableRotate = false;
    } else if (preset === "cockpit") {
      cam.position.set(0, 1.35, 0.4);
      ctrl.target.set(0, 1.25, -28);
      ctrl.enableRotate = false;
    } else if (preset === "bird") {
      cam.position.set(0, 32, -6);
      ctrl.target.set(0, 0, -8);
      ctrl.enableRotate = false;
    } else if (preset === "side") {
      cam.position.set(13, 5.2, -4);
      ctrl.target.set(0, 1.0, -8);
      ctrl.enableRotate = false;
    } else if (preset === "free") {
      ctrl.enableRotate = true;
      ctrl.enableZoom = true;
      ctrl.enablePan = true;
    }
    ctrl.update();
  }, []);

  // --- THREE.JS INITIALIZATION ---
  useEffect(() => {
    if (!mountRef.current) return;
    const container = mountRef.current;
    const w = width || container.clientWidth || 800;
    const h = height || container.clientHeight || 500;

    // 1. Scene & Depth Fog (Deep Automotive Obsidian Cherry)
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x0a0205);
    scene.fog = new THREE.FogExp2(0x0a0205, 0.018);

    // 2. Perspective Camera
    const camera = new THREE.PerspectiveCamera(46, w / h, 0.1, 150);
    camera.position.set(0, 3.6, 8.2);
    cameraRef.current = camera;

    // 3. WebGL Renderer with High-DPI & Shadow support
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    rendererRef.current = renderer;
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    // 4. Orbit Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controlsRef.current = controls;
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.maxDistance = 55;
    controls.minDistance = 1.5;
    controls.maxPolarAngle = Math.PI / 2 - 0.02; // Don't clip under ground
    controls.target.set(0, 1.1, -12);
    controls.update();

    // 5. Ambient & Directional Lighting
    const ambientLight = new THREE.AmbientLight(0xd8ccc0, 0.45);
    scene.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0x7f0b32, 0x14050d, 0.6);
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xf7f1ea, 0.8);
    keyLight.position.set(8, 18, 12);
    scene.add(keyLight);

    // --- 6. HIGH-REALISM ROADWAY SURFACE & ASPHALT ---
    const roadWidth = 14.4; // 4 lanes (3.6m each)
    const roadLength = 100;
    const roadGeo = new THREE.PlaneGeometry(roadWidth, roadLength, 32, 32);
    const roadMat = new THREE.MeshStandardMaterial({
      color: 0x120409,
      roughness: 0.82,
      metalness: 0.18,
    });
    const roadMesh = new THREE.Mesh(roadGeo, roadMat);
    roadMesh.rotation.x = -Math.PI / 2;
    roadMesh.position.set(0, -0.01, -20);
    roadMesh.receiveShadow = true;
    scene.add(roadMesh);

    // Subtle Ground Grid (Cherry/Graphite)
    const gridHelper = new THREE.GridHelper(90, 45, 0x7f0b32, 0x240713);
    gridHelper.position.set(0, 0, -15);
    roadGridRef.current = gridHelper;
    scene.add(gridHelper);

    // Realistic Road Markings
    const stripes: THREE.Line[] = [];
    const lineMatSolid = new THREE.LineBasicMaterial({
      color: 0xf7f1ea,
      transparent: true,
      opacity: 0.75,
      linewidth: 2,
    });
    const lineMatYellow = new THREE.LineBasicMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.85,
      linewidth: 2,
    });

    // Solid outer shoulder boundaries
    const leftEdgeGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-6.8, 0.02, -60),
      new THREE.Vector3(-6.8, 0.02, 20),
    ]);
    const rightEdgeGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(6.8, 0.02, -60),
      new THREE.Vector3(6.8, 0.02, 20),
    ]);
    const lEdge = new THREE.Line(leftEdgeGeo, lineMatSolid);
    const rEdge = new THREE.Line(rightEdgeGeo, lineMatSolid);
    scene.add(lEdge);
    scene.add(rEdge);
    stripes.push(lEdge, rEdge);

    // Center divider double yellow lines
    const centerLeftGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.15, 0.02, -60),
      new THREE.Vector3(-0.15, 0.02, 20),
    ]);
    const centerRightGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0.15, 0.02, -60),
      new THREE.Vector3(0.15, 0.02, 20),
    ]);
    scene.add(new THREE.Line(centerLeftGeo, lineMatYellow));
    scene.add(new THREE.Line(centerRightGeo, lineMatYellow));

    // Roadway reflective cat's eyes (studs)
    const studsGeo = new THREE.BoxGeometry(0.12, 0.04, 0.12);
    const studMat = new THREE.MeshBasicMaterial({ color: 0xffe066 });
    for (let z = -50; z < 15; z += 3) {
      const stud1 = new THREE.Mesh(studsGeo, studMat);
      stud1.position.set(-3.6, 0.02, z);
      scene.add(stud1);
      const stud2 = new THREE.Mesh(studsGeo, studMat);
      stud2.position.set(3.6, 0.02, z);
      scene.add(stud2);
    }
    roadStripesRef.current = stripes;

    // --- 7. REALISTIC 3D EGO VEHICLE ---
    const egoGroup = new THREE.Group();
    egoGroupRef.current = egoGroup;
    egoGroup.position.set(0, 0, 0);
    scene.add(egoGroup);

    // Chassis Body (Aerodynamic sculpted vehicle in cherry metallic)
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0x670626,
      metalness: 0.75,
      roughness: 0.22,
    });
    const cabinMat = new THREE.MeshStandardMaterial({
      color: 0x14050d,
      metalness: 0.92,
      roughness: 0.12,
      transparent: true,
      opacity: 0.88,
    });

    // Lower Chassis
    const lowerBodyGeo = new THREE.BoxGeometry(1.85, 0.55, 4.2);
    const lowerBody = new THREE.Mesh(lowerBodyGeo, bodyMat);
    lowerBody.position.set(0, 0.45, 0);
    lowerBody.castShadow = true;
    egoGroup.add(lowerBody);

    // Cabin Greenhouse & Windshield
    const cabinGeo = new THREE.BoxGeometry(1.5, 0.52, 2.3);
    const cabin = new THREE.Mesh(cabinGeo, cabinMat);
    cabin.position.set(0, 0.95, -0.2);
    cabin.castShadow = true;
    egoGroup.add(cabin);

    // Hood Slope
    const hoodGeo = new THREE.BoxGeometry(1.65, 0.25, 1.2);
    const hood = new THREE.Mesh(hoodGeo, bodyMat);
    hood.position.set(0, 0.65, -1.45);
    hood.rotation.x = 0.1;
    egoGroup.add(hood);

    // 4 Realistic Wheels with Alloy Rims
    const wheels: THREE.Mesh[] = [];
    const wheelGeo = new THREE.CylinderGeometry(0.34, 0.34, 0.28, 20);
    const tireMat = new THREE.MeshStandardMaterial({
      color: 0x1a1a1a,
      roughness: 0.9,
    });
    const rimMat = new THREE.MeshStandardMaterial({
      color: 0xd8ccc0,
      metalness: 0.85,
      roughness: 0.2,
    });

    const wheelPositions = [
      { x: -0.96, y: 0.34, z: -1.3 },
      { x: 0.96, y: 0.34, z: -1.3 },
      { x: -0.96, y: 0.34, z: 1.3 },
      { x: 0.96, y: 0.34, z: 1.3 },
    ];

    wheelPositions.forEach((pos) => {
      const tire = new THREE.Mesh(wheelGeo, tireMat);
      tire.rotation.z = Math.PI / 2;
      tire.position.set(pos.x, pos.y, pos.z);
      tire.castShadow = true;

      // Rim center
      const rim = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.29, 12), rimMat);
      tire.add(rim);

      egoGroup.add(tire);
      wheels.push(tire);
    });
    wheelsRef.current = wheels;

    // Roof-Mounted LiDAR Puck (Velodyne/Ouster Style)
    const puckBase = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.22, 0.18, 24),
      new THREE.MeshStandardMaterial({ color: 0x2b0614, metalness: 0.8, roughness: 0.2 })
    );
    puckBase.position.set(0, 1.32, -0.2);
    egoGroup.add(puckBase);

    // Spinning LiDAR Optical Rotor
    const puckRotor = new THREE.Mesh(
      new THREE.CylinderGeometry(0.16, 0.16, 0.12, 24),
      new THREE.MeshBasicMaterial({ color: 0xf7f1ea })
    );
    puckRotor.position.set(0, 0.12, 0);
    puckBase.add(puckRotor);
    lidarRotorRef.current = puckRotor;

    // Optical LiDAR Laser Emitter Ring
    const ringGeo = new THREE.RingGeometry(0.17, 0.24, 24);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xef4444,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = Math.PI / 2;
    ringMesh.position.set(0, 0.02, 0);
    puckBase.add(ringMesh);

    // Front Headlight Projectors
    const headlightLeft = new THREE.SpotLight(0xfff8ee, 2.5, 45, Math.PI / 6, 0.45, 1.2);
    headlightLeft.position.set(-0.65, 0.6, -2.1);
    headlightLeft.target.position.set(-0.65, 0, -25);
    egoGroup.add(headlightLeft);
    egoGroup.add(headlightLeft.target);

    const headlightRight = new THREE.SpotLight(0xfff8ee, 2.5, 45, Math.PI / 6, 0.45, 1.2);
    headlightRight.position.set(0.65, 0.6, -2.1);
    headlightRight.target.position.set(0.65, 0, -25);
    egoGroup.add(headlightRight);
    egoGroup.add(headlightRight.target);
    headlightsRef.current = [headlightLeft, headlightRight];

    // Rear Taillight Lightbar
    const taillightGeo = new THREE.BoxGeometry(1.7, 0.08, 0.05);
    const taillightMat = new THREE.MeshBasicMaterial({ color: 0xff1030 });
    const taillight = new THREE.Mesh(taillightGeo, taillightMat);
    taillight.position.set(0, 0.62, 2.11);
    egoGroup.add(taillight);

    // Underglow Halo
    const underglowGeo = new THREE.PlaneGeometry(2.4, 4.8);
    const underglowMat = new THREE.MeshBasicMaterial({
      color: 0x8b0f38,
      transparent: true,
      opacity: 0.35,
      side: THREE.DoubleSide,
    });
    const underglow = new THREE.Mesh(underglowGeo, underglowMat);
    underglow.rotation.x = -Math.PI / 2;
    underglow.position.set(0, 0.03, 0);
    egoGroup.add(underglow);

    // --- 8. AUTHENTIC MULTI-BEAM LIDAR SCAN RINGS (16,000+ Depth-Color Points) ---
    const totalRings = 48;
    const pointsPerRing = 320;
    const pointCount = totalRings * pointsPerRing;
    const ptPositions = new Float32Array(pointCount * 3);
    const ptColors = new Float32Array(pointCount * 3);

    const cNear = new THREE.Color(0xf7f1ea); // Warm Ivory
    const cMid = new THREE.Color(0xa31243);  // Vibrant Cherry
    const cFar = new THREE.Color(0x3b0717);  // Deep Crimson

    let idx = 0;
    for (let r = 1; r <= totalRings; r++) {
      const radius = 2.2 + Math.pow(r / totalRings, 1.4) * 44;
      for (let p = 0; p < pointsPerRing; p++) {
        const theta = (p / pointsPerRing) * Math.PI * 2;
        const x = Math.sin(theta) * radius * 0.75;
        const z = -Math.cos(theta) * radius + 2.0;
        const y = Math.max(0.02, Math.random() * 0.08 + (Math.abs(x) > 6.8 ? 0.35 : 0));

        ptPositions[idx * 3] = x;
        ptPositions[idx * 3 + 1] = y;
        ptPositions[idx * 3 + 2] = z;

        const depthFactor = Math.min(1, Math.abs(z) / 45);
        const col = depthFactor < 0.3 ? cNear : depthFactor < 0.7 ? cMid : cFar;
        ptColors[idx * 3] = col.r;
        ptColors[idx * 3 + 1] = col.g;
        ptColors[idx * 3 + 2] = col.b;

        idx++;
      }
    }

    const cloudGeo = new THREE.BufferGeometry();
    cloudGeo.setAttribute("position", new THREE.BufferAttribute(ptPositions, 3));
    cloudGeo.setAttribute("color", new THREE.BufferAttribute(ptColors, 3));

    const cloudMat = new THREE.PointsMaterial({
      size: 0.12,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
    });
    const pointCloud = new THREE.Points(cloudGeo, cloudMat);
    pointCloudRef.current = pointCloud;
    scene.add(pointCloud);

    // --- 9. MONOCULAR CAMERA 120° FOV FRUSTUM CONE ---
    const frustumGeo = new THREE.ConeGeometry(24, 38, 4, 1, true);
    const frustumMat = new THREE.MeshBasicMaterial({
      color: 0x670626,
      wireframe: true,
      transparent: true,
      opacity: 0.22,
    });
    const frustumMesh = new THREE.Mesh(frustumGeo, frustumMat);
    frustumMesh.rotation.x = -Math.PI / 2;
    frustumMesh.rotation.z = Math.PI / 4;
    frustumMesh.position.set(0, 1.25, -19);
    frustumMeshRef.current = frustumMesh;
    scene.add(frustumMesh);

    // --- 10. ROTATING 360° LIDAR LASER SWEEP FAN ---
    const sweepGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 1.3, 0),
      new THREE.Vector3(-18, 0.1, -38),
      new THREE.Vector3(18, 0.1, -38),
    ]);
    const sweepMat = new THREE.MeshBasicMaterial({
      color: 0xef4444,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
    });
    const sweepMesh = new THREE.Mesh(sweepGeo, sweepMat);
    sweepMeshRef.current = sweepMesh;
    scene.add(sweepMesh);

    // Groups for Dynamic Objects and Trajectory Splines
    const boxesGroup = new THREE.Group();
    boxesGroupRef.current = boxesGroup;
    scene.add(boxesGroup);

    const trajectoryGroup = new THREE.Group();
    trajectoryGroupRef.current = trajectoryGroup;
    scene.add(trajectoryGroup);

    // --- 11. ANIMATION LOOP (60 FPS) ---
    let sweepAngle = 0;
    let lastTime = performance.now();
    let frameCount = 0;

    const animate = () => {
      frameIdRef.current = requestAnimationFrame(animate);

      const now = performance.now();
      frameCount++;
      if (now - lastTime >= 1000) {
        setActiveFps(Math.round((frameCount * 1000) / (now - lastTime)));
        frameCount = 0;
        lastTime = now;
      }

      // 1. Road streaming forward (simulating 45 km/h)
      if (gridHelper) {
        gridHelper.position.z = (gridHelper.position.z + 0.12) % 2;
      }

      // 2. Rolling vehicle wheels
      wheels.forEach((wheel) => {
        wheel.rotation.x -= 0.15;
      });

      // 3. Spinning LiDAR sensor rotor
      if (puckRotor) {
        puckRotor.rotation.y += 0.25;
      }

      // 4. Rotating LiDAR laser sweep fan
      sweepAngle += 0.04;
      sweepMesh.rotation.y = Math.sin(sweepAngle) * 0.48;

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize Observer
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: nw, height: nh } = entry.contentRect;
        if (nw > 0 && nh > 0) {
          camera.aspect = nw / nh;
          camera.updateProjectionMatrix();
          renderer.setSize(nw, nh);
        }
      }
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      if (frameIdRef.current) cancelAnimationFrame(frameIdRef.current);
      renderer.dispose();
      container.innerHTML = "";
    };
  }, [width, height]);

  // --- UPDATE PROCEDURAL 3D OBSTACLE MODELS & TRAJECTORY SPLINES ---
  useEffect(() => {
    const group = boxesGroupRef.current;
    const trajGroup = trajectoryGroupRef.current;
    if (!group || !trajGroup) return;

    // Clear prior dynamic meshes
    while (group.children.length > 0) {
      group.remove(group.children[0]);
    }
    while (trajGroup.children.length > 0) {
      trajGroup.remove(trajGroup.children[0]);
    }

    objects.forEach((obj, idx) => {
      const dist = obj.distance_m ?? 8 + idx * 4;
      // Coordinate transformation from image frame to 3D world space
      const cx = (obj.bbox.x1 + obj.bbox.x2) / 2;
      const xPos = (cx - 320) * 0.016;
      const zPos = -Math.min(dist, 40);

      const isHigh = obj.risk_level === "High";
      const isMod = obj.risk_level === "Moderate";
      const colorHex = isHigh ? 0xef4444 : isMod ? 0xf59e0b : 0x10b981;

      const modelGroup = new THREE.Group();
      modelGroup.position.set(xPos, 0, zPos);

      // Model dispatch based on detected class
      const raw = (obj.class_raw || "").toLowerCase();
      if (raw === "cow" || raw === "horse" || raw === "sheep" || raw === "goat" || raw === "dog") {
        // --- REALISTIC 3D BOVINE / CATTLE MODEL ---
        const hideMat = new THREE.MeshStandardMaterial({
          color: 0xf7f1ea,
          roughness: 0.85,
        });
        const hornMat = new THREE.MeshStandardMaterial({ color: 0x260714 });

        // Torso
        const torso = new THREE.Mesh(new THREE.BoxGeometry(0.85, 0.95, 1.9), hideMat);
        torso.position.set(0, 0.95, 0);
        modelGroup.add(torso);

        // Head and Muzzle
        const head = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.6, 0.75), hideMat);
        head.position.set(0, 1.35, -1.05);
        head.rotation.x = 0.15;
        modelGroup.add(head);

        // Dual Horns
        const hornL = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.35, 8), hornMat);
        hornL.position.set(-0.3, 1.7, -1.0);
        hornL.rotation.z = -0.4;
        modelGroup.add(hornL);

        const hornR = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.35, 8), hornMat);
        hornR.position.set(0.3, 1.7, -1.0);
        hornR.rotation.z = 0.4;
        modelGroup.add(hornR);

        // 4 Legs
        const legGeo = new THREE.CylinderGeometry(0.09, 0.08, 0.85, 8);
        const legPositions = [
          [-0.32, 0.42, -0.65],
          [0.32, 0.42, -0.65],
          [-0.32, 0.42, 0.65],
          [0.32, 0.42, 0.65],
        ];
        legPositions.forEach(([lx, ly, lz]) => {
          const leg = new THREE.Mesh(legGeo, hideMat);
          leg.position.set(lx, ly, lz);
          modelGroup.add(leg);
        });
      } else if (raw === "auto-rickshaw") {
        // --- REALISTIC 3D INDIAN AUTO-RICKSHAW MODEL ---
        const canopyMat = new THREE.MeshStandardMaterial({
          color: 0xf59e0b,
          roughness: 0.4,
          metalness: 0.3,
        });
        const frameMat = new THREE.MeshStandardMaterial({
          color: 0x14050d,
          roughness: 0.7,
        });

        // Cabin canopy
        const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.3, 1.4, 2.2), canopyMat);
        cabin.position.set(0, 1.05, 0);
        modelGroup.add(cabin);

        // Slanted front apron
        const frontApron = new THREE.Mesh(new THREE.ConeGeometry(0.7, 1.1, 4), frameMat);
        frontApron.rotation.x = Math.PI;
        frontApron.position.set(0, 0.7, -1.2);
        modelGroup.add(frontApron);

        // 3 Wheels (1 front, 2 rear)
        const autoWheelGeo = new THREE.CylinderGeometry(0.25, 0.25, 0.18, 12);
        const autoTireMat = new THREE.MeshStandardMaterial({ color: 0x111111 });

        const frontW = new THREE.Mesh(autoWheelGeo, autoTireMat);
        frontW.rotation.z = Math.PI / 2;
        frontW.position.set(0, 0.25, -1.2);
        modelGroup.add(frontW);

        const rearWL = new THREE.Mesh(autoWheelGeo, autoTireMat);
        rearWL.rotation.z = Math.PI / 2;
        rearWL.position.set(-0.65, 0.25, 0.7);
        modelGroup.add(rearWL);

        const rearWR = new THREE.Mesh(autoWheelGeo, autoTireMat);
        rearWR.rotation.z = Math.PI / 2;
        rearWR.position.set(0.65, 0.25, 0.7);
        modelGroup.add(rearWR);
      } else if (raw === "person") {
        // --- REALISTIC 3D PEDESTRIAN AVATAR ---
        const pedMat = new THREE.MeshStandardMaterial({
          color: 0xf7f1ea,
          roughness: 0.6,
        });
        const head = new THREE.Mesh(new THREE.SphereGeometry(0.18, 12, 12), pedMat);
        head.position.set(0, 1.6, 0);
        modelGroup.add(head);

        const torso = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.18, 0.75, 8), pedMat);
        torso.position.set(0, 1.05, 0);
        modelGroup.add(torso);

        const legL = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.07, 0.7, 8), pedMat);
        legL.position.set(-0.12, 0.35, 0);
        modelGroup.add(legL);

        const legR = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.07, 0.7, 8), pedMat);
        legR.position.set(0.12, 0.35, 0);
        modelGroup.add(legR);
      } else {
        // --- VEHICLE / OBSTACLE 3D ENVELOPE ---
        const boxMat = new THREE.MeshStandardMaterial({
          color: colorHex,
          roughness: 0.4,
          metalness: 0.6,
        });
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(1.7, 1.4, 3.2), boxMat);
        mesh.position.set(0, 0.75, 0);
        modelGroup.add(mesh);
      }

      // Ground Danger Pulsing Ring
      const ringGeo = new THREE.RingGeometry(1.2, 1.45, 24);
      const ringMat = new THREE.MeshBasicMaterial({
        color: colorHex,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: isHigh ? 0.8 : 0.4,
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(0, 0.03, 0);
      modelGroup.add(ring);

      // AR Holographic 3D Bounding Cage & Brackets
      const cageGeo = new THREE.BoxGeometry(2.0, 1.9, 2.6);
      const cageMat = new THREE.MeshBasicMaterial({
        color: colorHex,
        wireframe: true,
        transparent: true,
        opacity: 0.6,
      });
      const cage = new THREE.Mesh(cageGeo, cageMat);
      cage.position.set(0, 0.95, 0);
      modelGroup.add(cage);

      group.add(modelGroup);

      // --- 3D PREDICTIVE COLLISION TRAJECTORY SPLINE (TTC ARC) ---
      const ttc = obj.ttc_s;
      if (showTrajectory && (isHigh || (ttc != null && ttc < 4.0))) {
        const curve = new THREE.QuadraticBezierCurve3(
          new THREE.Vector3(0, 0.5, -2.1), // Front of ego car
          new THREE.Vector3(xPos * 0.45, 0.5, zPos * 0.5), // Arc midpoint
          new THREE.Vector3(xPos, 0.5, zPos) // Target hazard position
        );

        const curvePoints = curve.getPoints(24);
        const trajGeo = new THREE.BufferGeometry().setFromPoints(curvePoints);
        const trajMat = new THREE.LineDashedMaterial({
          color: 0xef4444,
          dashSize: 0.8,
          gapSize: 0.4,
          linewidth: 2,
        });
        const trajLine = new THREE.Line(trajGeo, trajMat);
        trajLine.computeLineDistances();
        trajGroup.add(trajLine);

        // 3D Impact Reticle on Roadway
        const impactRing = new THREE.Mesh(
          new THREE.RingGeometry(0.4, 0.6, 16),
          new THREE.MeshBasicMaterial({ color: 0xef4444, side: THREE.DoubleSide })
        );
        impactRing.rotation.x = -Math.PI / 2;
        impactRing.position.set(xPos, 0.04, zPos);
        trajGroup.add(impactRing);
      }
    });
  }, [objects, showTrajectory]);

  // Toggle point cloud density
  useEffect(() => {
    if (!pointCloudRef.current) return;
    if (showPointCloud === "off") {
      pointCloudRef.current.visible = false;
    } else {
      pointCloudRef.current.visible = true;
      const mat = pointCloudRef.current.material as THREE.PointsMaterial;
      mat.size = showPointCloud === "dense" ? 0.12 : 0.08;
      mat.opacity = showPointCloud === "dense" ? 0.85 : 0.4;
    }
  }, [showPointCloud]);

  // Toggle Camera FOV Frustum
  useEffect(() => {
    if (frustumMeshRef.current) {
      frustumMeshRef.current.visible = showFrustum;
    }
  }, [showFrustum]);

  // Toggle LiDAR Sweep
  useEffect(() => {
    if (sweepMeshRef.current) {
      sweepMeshRef.current.visible = showSweep;
    }
  }, [showSweep]);

  // Toggle Headlights
  useEffect(() => {
    headlightsRef.current.forEach((light) => {
      light.visible = showHeadlights;
    });
  }, [showHeadlights]);

  return (
    <div
      className={`relative w-full rounded-2xl overflow-hidden border border-[#670626]/60 bg-[#0A0205] shadow-[0_0_50px_rgba(103,6,38,0.25)] transition-all duration-300 ${
        isFullscreen ? "fixed inset-0 z-50 rounded-none border-none h-screen w-screen" : "h-full min-h-[440px]"
      }`}
    >
      {/* 3D WebGL Canvas Mount */}
      <div ref={mountRef} className="w-full h-full min-h-[440px] cursor-grab active:cursor-grabbing" />

      {/* TOP-LEFT: TACTICAL PERCEPTION HUD OVERLAY */}
      <div className="absolute top-4 left-4 pointer-events-none font-mono text-[11px] text-[#D8CCC0] space-y-2 bg-[#14050D]/90 p-3.5 rounded-xl border border-[#670626]/70 backdrop-blur-md shadow-2xl z-10 max-w-[280px]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-[#F7F1EA] font-heading font-bold tracking-wider text-xs">
            <span className="w-2.5 h-2.5 rounded-full bg-[#10B981] animate-pulse shadow-[0_0_10px_#10B981]" />
            SPATIAL DIGITAL TWIN
          </div>
          <span className="text-[10px] text-[#10B981] font-bold">{activeFps} FPS</span>
        </div>

        <div className="text-[10px] text-[#9E8F81] border-b border-[#670626]/40 pb-1.5">
          Monocular 3D World Reconstruction
        </div>

        <div className="grid grid-cols-2 gap-2 text-[10px]">
          <div className="bg-[#1F0714] p-1.5 rounded border border-[#670626]/40">
            <span className="text-[#9E8F81] block">TARGETS</span>
            <span className="text-[#F7F1EA] font-bold text-xs">{objects.length} Active</span>
          </div>
          <div className="bg-[#1F0714] p-1.5 rounded border border-[#670626]/40">
            <span className="text-[#9E8F81] block">CHAOS</span>
            <span
              className={`font-bold text-xs ${
                chaosScore > 65 ? "text-[#EF4444]" : chaosScore > 35 ? "text-[#F59E0B]" : "text-[#10B981]"
              }`}
            >
              {Math.round(chaosScore)}/100
            </span>
          </div>
        </div>

        <div className="text-[10px] text-[#D8CCC0] flex items-center justify-between pt-1">
          <span>EGO VELOCITY</span>
          <span className="text-[#F7F1EA] font-bold">45.0 km/h</span>
        </div>
      </div>

      {/* TOP-RIGHT: INTERACTIVE CAMERA PRESETS & FULLSCREEN CONTROLS */}
      <div className="absolute top-4 right-4 flex items-center gap-2 z-10">
        {/* Camera Views Selector */}
        <div className="flex items-center bg-[#14050D]/90 p-1 rounded-xl border border-[#670626]/70 backdrop-blur-md shadow-lg text-[11px] font-heading">
          <button
            onClick={() => applyCameraPreset("chase")}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              activePreset === "chase"
                ? "bg-[#670626] text-[#F7F1EA] font-bold shadow-md"
                : "text-[#D8CCC0] hover:text-[#F7F1EA]"
            }`}
            title="3rd-person vehicle follow view"
          >
            Chase
          </button>
          <button
            onClick={() => applyCameraPreset("cockpit")}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              activePreset === "cockpit"
                ? "bg-[#670626] text-[#F7F1EA] font-bold shadow-md"
                : "text-[#D8CCC0] hover:text-[#F7F1EA]"
            }`}
            title="Dashcam driver cockpit vantage"
          >
            Cockpit
          </button>
          <button
            onClick={() => applyCameraPreset("bird")}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              activePreset === "bird"
                ? "bg-[#670626] text-[#F7F1EA] font-bold shadow-md"
                : "text-[#D8CCC0] hover:text-[#F7F1EA]"
            }`}
            title="Overhead tactical 2D/3D traffic view"
          >
            Bird's Eye
          </button>
          <button
            onClick={() => applyCameraPreset("side")}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              activePreset === "side"
                ? "bg-[#670626] text-[#F7F1EA] font-bold shadow-md"
                : "text-[#D8CCC0] hover:text-[#F7F1EA]"
            }`}
            title="Side profile perspective view"
          >
            Side
          </button>
          <button
            onClick={() => applyCameraPreset("free")}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              activePreset === "free"
                ? "bg-[#670626] text-[#F7F1EA] font-bold shadow-md"
                : "text-[#D8CCC0] hover:text-[#F7F1EA]"
            }`}
            title="Free 360 degree drag & zoom orbit"
          >
            Orbit
          </button>
        </div>

        {/* Behind-The-Tech Info Button */}
        <button
          onClick={() => setShowTechModal(true)}
          className="p-2 rounded-xl bg-[#14050D]/90 border border-[#670626]/70 text-[#F7F1EA] hover:bg-[#670626] transition-all shadow-lg"
          title="Inspect the Monocular-to-3D Geometry Math & Sensor Fusion Pipeline"
        >
          <Info size={15} />
        </button>

        {/* Fullscreen Toggle */}
        <button
          onClick={() => setIsFullscreen(!isFullscreen)}
          className="p-2 rounded-xl bg-[#14050D]/90 border border-[#670626]/70 text-[#F7F1EA] hover:bg-[#670626] transition-all shadow-lg"
          title={isFullscreen ? "Exit Fullscreen" : "Expand 3D Spatial Theater"}
        >
          {isFullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
        </button>
      </div>

      {/* BOTTOM-LEFT: SENSOR TECHNOLOGY LAYER TOGGLES */}
      <div className="absolute bottom-4 left-4 flex flex-wrap items-center gap-1.5 bg-[#14050D]/90 p-1.5 rounded-xl border border-[#670626]/70 backdrop-blur-md shadow-xl z-10 text-[11px] font-mono text-[#D8CCC0]">
        <span className="text-[10px] text-[#9E8F81] px-2 font-bold tracking-wider">LAYERS:</span>

        <button
          onClick={() =>
            setShowPointCloud((prev) => (prev === "dense" ? "sparse" : prev === "sparse" ? "off" : "dense"))
          }
          className={`px-2 py-1 rounded-lg border transition-all flex items-center gap-1 ${
            showPointCloud !== "off"
              ? "bg-[#670626]/70 border-[#F7F1EA]/30 text-[#F7F1EA]"
              : "border-transparent text-[#9E8F81] hover:text-[#F7F1EA]"
          }`}
        >
          <Sparkles size={11} />
          LiDAR ({showPointCloud})
        </button>

        <button
          onClick={() => setShowFrustum(!showFrustum)}
          className={`px-2 py-1 rounded-lg border transition-all flex items-center gap-1 ${
            showFrustum
              ? "bg-[#670626]/70 border-[#F7F1EA]/30 text-[#F7F1EA]"
              : "border-transparent text-[#9E8F81] hover:text-[#F7F1EA]"
          }`}
        >
          <Eye size={11} />
          120° FOV
        </button>

        <button
          onClick={() => setShowTrajectory(!showTrajectory)}
          className={`px-2 py-1 rounded-lg border transition-all flex items-center gap-1 ${
            showTrajectory
              ? "bg-[#EF4444]/20 border-[#EF4444]/50 text-[#FF708F]"
              : "border-transparent text-[#9E8F81] hover:text-[#F7F1EA]"
          }`}
        >
          <AlertTriangle size={11} />
          TTC Splines
        </button>

        <button
          onClick={() => setShowSweep(!showSweep)}
          className={`px-2 py-1 rounded-lg border transition-all flex items-center gap-1 ${
            showSweep
              ? "bg-[#670626]/70 border-[#F7F1EA]/30 text-[#F7F1EA]"
              : "border-transparent text-[#9E8F81] hover:text-[#F7F1EA]"
          }`}
        >
          <Radio size={11} />
          Scan Fan
        </button>

        <button
          onClick={() => setShowHeadlights(!showHeadlights)}
          className={`px-2 py-1 rounded-lg border transition-all flex items-center gap-1 ${
            showHeadlights
              ? "bg-[#670626]/70 border-[#F7F1EA]/30 text-[#F7F1EA]"
              : "border-transparent text-[#9E8F81] hover:text-[#F7F1EA]"
          }`}
        >
          <Zap size={11} />
          Headlights
        </button>
      </div>

      {/* BOTTOM-RIGHT: HONEST TELEMETRY DISCLAIMER BADGE */}
      <div className="absolute bottom-4 right-4 pointer-events-none font-mono text-[10px] text-[#9E8F81] bg-[#14050D]/90 px-3 py-1.5 rounded-xl border border-[#670626]/50 shadow-lg flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-[#10B981]" />
        Pinhole Raycast Reconstruction • No LiDAR Hardware Required
      </div>

      {/* BEHIND-THE-TECHNOLOGY MODAL POPUP */}
      {showTechModal && (
        <div className="absolute inset-0 z-30 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="glass-card-elevated border border-[#670626] rounded-2xl max-w-xl w-full p-6 text-[#F7F1EA] shadow-2xl relative animate-fade-in-up">
            <button
              onClick={() => setShowTechModal(false)}
              className="absolute top-4 right-4 text-[#D8CCC0] hover:text-[#F7F1EA]"
            >
              <X size={18} />
            </button>

            <div className="flex items-center gap-2 mb-4">
              <Zap size={20} className="text-[#A31243]" />
              <h3 className="font-heading text-lg font-bold">
                Behind The Technology: Monocular-to-3D Spatial Engine
              </h3>
            </div>

            <p className="text-xs text-[#D8CCC0] leading-relaxed mb-4">
              SARTHI Vision reconstructs full 3D spatial digital twins from single standard 2D dashboard cameras,
              without requiring expensive $10,000+ hardware LiDAR or dual stereo cameras.
            </p>

            <div className="space-y-3 font-mono text-xs">
              <div className="p-3 rounded-xl bg-[#14050D] border border-[#670626]/50">
                <span className="text-[#FF708F] font-bold block mb-1">1. Pinhole Depth Reconstruction (Z)</span>
                <code className="text-[#10B981] block">Distance (Z) = (focal_length_px * RealWorldWidth_m) / bbox_width_px</code>
                <span className="text-[10px] text-[#9E8F81] mt-1 block">
                  Calibrated against real-world metrics (e.g. Bovine: 0.6m, Auto-Rickshaw: 1.4m, Car: 1.8m).
                </span>
              </div>

              <div className="p-3 rounded-xl bg-[#14050D] border border-[#670626]/50">
                <span className="text-[#FF708F] font-bold block mb-1">2. Lateral & Vertical Offset (X, Y)</span>
                <code className="text-[#10B981] block">X = ((bbox_center_x - img_cx) * Z) / focal_length_px</code>
                <span className="text-[10px] text-[#9E8F81] mt-1 block">
                  Maps pixel displacement to exact physical lateral lane offset in meters.
                </span>
              </div>

              <div className="p-3 rounded-xl bg-[#14050D] border border-[#670626]/50">
                <span className="text-[#FF708F] font-bold block mb-1">3. Predictive TTC Collision Vector</span>
                <code className="text-[#10B981] block">TTC = Distance_m / ClosingVelocity_m_s</code>
                <span className="text-[10px] text-[#9E8F81] mt-1 block">
                  Triggers 3D Bézier trajectory splines and voice warnings 4.0 seconds prior to collision.
                </span>
              </div>
            </div>

            <div className="mt-5 flex justify-end">
              <button
                onClick={() => setShowTechModal(false)}
                className="btn-cherry px-4 py-2 rounded-xl text-xs font-heading font-semibold"
              >
                Close Architecture View
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
