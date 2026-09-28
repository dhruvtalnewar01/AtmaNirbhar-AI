"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import Image from "next/image";
import {
  Upload,
  Shield,
  Eye,
  Activity,
  Zap,
  AlertTriangle,
  FileVideo,
  FileImage,
  X,
  Loader2,
  Download,
  SlidersHorizontal,
  Search,
  Box,
  Layers,
  Sparkles,
  ArrowRight,
  Play,
  Pause,
  Maximize2,
  Minimize2,
  SkipBack,
  SkipForward,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  ChevronRight,
  Volume2,
  VolumeX,
  Clock,
  Target,
  Radio,
} from "lucide-react";
import { analyzeMedia, isSuccess, isError, checkHealth } from "@/lib/api";
import type {
  AnalysisResponse,
  DetectedObject,
  NearMissEvent,
  AudioEvent,
} from "@/lib/types";
import { SIGNAL_COLORS } from "@/lib/types";
import { renderFrame, hitTestDetection } from "@/lib/canvas-renderer";
import LidarVisualizer from "@/components/viewer/LidarVisualizer";
import { alertEngine, unlockAudio } from "@/lib/audio-alert";


const MAX_FILE_SIZE_MB = 50;
const MAX_DURATION_S = 45;
const SUPPORTED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/bmp",
  "image/webp",
  "video/mp4",
  "video/quicktime",
  "video/x-msvideo",
  "video/webm",
];

const SAMPLE_FILES = [
  {
    id: "animal",
    title: "Bovine Hazard on Highway",
    desc: "Unpredictable animal obstruction on Indian corridor",
    path: "/samples/sample_animal.jpg",
    badge: "High Risk Class",
    badgeColor: "#EF4444",
  },
  {
    id: "auto",
    title: "Urban Auto-Rickshaw Traffic",
    desc: "Dense mixed traffic with 3-wheelers and pedestrians",
    path: "/samples/sample_auto.jpg",
    badge: "Fine-Tuned Model",
    badgeColor: "#F59E0B",
  },
  {
    id: "obstacle",
    title: "Unstructured Road Debris",
    desc: "Road construction rubble and barricade obstructions",
    path: "/samples/sample_obstacle.jpg",
    badge: "Obstacle Detection",
    badgeColor: "#10B981",
  },
];

const DEMO_SCENARIOS: Record<
  "bovine" | "traffic" | "night",
  {
    title: string;
    badge: string;
    desc: string;
    chaosScore: number;
    objects: DetectedObject[];
  }
> = {
  bovine: {
    title: "Highway Bovine Hazard (Predictive TTC 3.2s)",
    badge: "ANIMAL HAZARD • IMMINENT WARNING",
    desc: "Stationary cattle in primary ego corridor triggering 3D Bézier trajectory spline & voice alert",
    chaosScore: 68.5,
    objects: [
      {
        track_id: 1,
        class_name: "Bovine (Cow)",
        class_raw: "cow",
        confidence: 0.96,
        bbox: { x1: 220, y1: 210, x2: 360, y2: 360 },
        status: "Stationary",
        distance_m: 6.8,
        distance_available: true,
        distance_method: "pinhole" as const,
        risk_level: "High",
        risk_score: 0.92,
        risk_explanation: "High risk: stationary animal in primary vehicle corridor",
        rel_dx: 0,
        rel_dy: 0,
        closing_speed: -12.5,
        ttc_s: 3.2,
      },
      {
        track_id: 2,
        class_name: "Auto-Rickshaw",
        class_raw: "auto-rickshaw",
        confidence: 0.89,
        bbox: { x1: 410, y1: 190, x2: 520, y2: 330 },
        status: "Moving",
        distance_m: 15.2,
        distance_available: true,
        distance_method: "pinhole" as const,
        risk_level: "Moderate",
        risk_score: 0.52,
        risk_explanation: "Moderate risk: vehicle lateral transit",
        rel_dx: -0.8,
        rel_dy: 0.1,
        closing_speed: -2.1,
        ttc_s: 5.1,
      },
      {
        track_id: 3,
        class_name: "Motorcycle",
        class_raw: "motorcycle",
        confidence: 0.84,
        bbox: { x1: 140, y1: 240, x2: 210, y2: 340 },
        status: "Moving",
        distance_m: 22.0,
        distance_available: true,
        distance_method: "pinhole" as const,
        risk_level: "Low",
        risk_score: 0.28,
        risk_explanation: "Low risk: shoulder corridor transit",
        rel_dx: 0.2,
        rel_dy: 0.0,
        closing_speed: 0.4,
        ttc_s: null,
      },
    ],
  },
  traffic: {
    title: "Dense Mixed Traffic (Auto-Rickshaw Cut-In)",
    badge: "DENSE URBAN • HIGH CHAOS 82/100",
    desc: "Aggressive multi-agent lateral convergence with auto-rickshaws, pedestrians, and transport trucks",
    chaosScore: 82.0,
    objects: [
      {
        track_id: 4,
        class_name: "Auto-Rickshaw",
        class_raw: "auto-rickshaw",
        confidence: 0.93,
        bbox: { x1: 270, y1: 220, x2: 400, y2: 370 },
        status: "Moving",
        distance_m: 8.4,
        distance_available: true,
        distance_method: "pinhole" as const,
        risk_level: "High",
        risk_score: 0.88,
        risk_explanation: "High risk: auto-rickshaw swerving across lane markings",
        rel_dx: -1.6,
        rel_dy: 0.3,
        closing_speed: -8.2,
        ttc_s: 2.8,
      },
      {
        track_id: 5,
        class_name: "Pedestrian",
        class_raw: "person",
        confidence: 0.91,
        bbox: { x1: 110, y1: 220, x2: 170, y2: 350 },
        status: "Moving",
        distance_m: 12.5,
        distance_available: true,
        distance_method: "pinhole" as const,
        risk_level: "Moderate",
        risk_score: 0.62,
        risk_explanation: "Moderate risk: roadside pedestrian path projection",
        rel_dx: 0.5,
        rel_dy: 0.0,
        closing_speed: -1.2,
        ttc_s: 6.4,
      },
      {
        track_id: 6,
        class_name: "Commercial Truck",
        class_raw: "truck",
        confidence: 0.95,
        bbox: { x1: 450, y1: 160, x2: 580, y2: 340 },
        status: "Moving",
        distance_m: 24.5,
        distance_available: true,
        distance_method: "pinhole" as const,
        risk_level: "Low",
        risk_score: 0.31,
        risk_explanation: "Low risk: adjacent transport corridor",
        rel_dx: 0.1,
        rel_dy: 0.0,
        closing_speed: 1.5,
        ttc_s: null,
      },
    ],
  },
  night: {
    title: "Nighttime Unlit Roadway (Cattle Herd + Canine)",
    badge: "LOW-LIGHT CLAHE • THERMAL INFRARED",
    desc: "Unlit dark-colored animals on night asphalt detected via adaptive CLAHE enhancement and pinhole math",
    chaosScore: 74.0,
    objects: [
      {
        track_id: 7,
        class_name: "Bovine (Cow)",
        class_raw: "cow",
        confidence: 0.95,
        bbox: { x1: 240, y1: 220, x2: 370, y2: 360 },
        status: "Stationary",
        distance_m: 7.6,
        distance_available: true,
        distance_method: "pinhole" as const,
        risk_level: "High",
        risk_score: 0.94,
        risk_explanation: "High risk: unlit bovine cattle standing on dark asphalt",
        rel_dx: 0,
        rel_dy: 0,
        closing_speed: -11.0,
        ttc_s: 3.5,
      },
      {
        track_id: 8,
        class_name: "Stray Dog",
        class_raw: "dog",
        confidence: 0.86,
        bbox: { x1: 420, y1: 270, x2: 480, y2: 340 },
        status: "Moving",
        distance_m: 14.8,
        distance_available: true,
        distance_method: "pinhole" as const,
        risk_level: "Moderate",
        risk_score: 0.58,
        risk_explanation: "Moderate risk: fast-moving canine near wheel trajectory",
        rel_dx: -0.9,
        rel_dy: 0.1,
        closing_speed: -2.4,
        ttc_s: 5.8,
      },
    ],
  },
};


export default function Home() {
  const [phase, setPhase] = useState<
    "upload" | "processing" | "results" | "error"
  >("upload");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingStage, setProcessingStage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [confidence, setConfidence] = useState(0.35);
  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [selectedObject, setSelectedObject] = useState<DetectedObject | null>(
    null
  );
  const [isDragging, setIsDragging] = useState(false);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [viewMode, setViewMode] = useState<"split" | "feed" | "spatial">("split");
  const [demoScenario, setDemoScenario] = useState<"bovine" | "traffic" | "night">("bovine");

  // Video playback states
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [showOverlays, setShowOverlays] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Full-size image modal state
  const [isImageModalOpen, setIsImageModalOpen] = useState(false);
  const [imageZoom, setImageZoom] = useState(1);

  // Voice alert state
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [speakingPhrase, setSpeakingPhrase] = useState<string | null>(null);

  // Near-miss timeline panel
  const [showTimeline, setShowTimeline] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modalCanvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);

  // Health check
  useEffect(() => {
    checkHealth().then(setBackendOnline);
    const interval = setInterval(() => {
      checkHealth().then(setBackendOnline);
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  // Unlock audio on initial user gesture (click or keypress)
  useEffect(() => {
    const handleUnlock = () => unlockAudio();
    window.addEventListener("click", handleUnlock, { once: true });
    window.addEventListener("keydown", handleUnlock, { once: true });
    return () => {
      window.removeEventListener("click", handleUnlock);
      window.removeEventListener("keydown", handleUnlock);
    };
  }, []);

  // Subscribe to speaking phrase state updates from alertEngine
  useEffect(() => {
    alertEngine.setSpeakingCallback((phrase) => {
      setSpeakingPhrase(phrase);
    });
  }, []);

  // Sync voice toggle with alert engine
  useEffect(() => {
    if (!voiceEnabled) {
      alertEngine.clear();
    }
  }, [voiceEnabled]);

  // Queue voice alerts when frame changes - ONLY while video is actively playing!
  useEffect(() => {
    if (!result || !voiceEnabled) return;
    // For videos: if paused or stopped, do NOT play or queue voice warnings!
    if (result.media_type === "video" && !isPlaying) return;

    const frame = result.frames[currentFrameIdx];
    if (!frame?.voice_alerts?.length) return;

    for (const alert of frame.voice_alerts) {
      if (alert.phrase) {
        const hazardKey = (alert.fact as Record<string, unknown>)?.class as string | undefined;
        alertEngine.enqueue(alert.phrase, (alert.urgency as "imminent" | "high") || "high", hazardKey);
      }
    }
  }, [currentFrameIdx, result, voiceEnabled, isPlaying]);



  const handleFileSelect = useCallback(
    (selectedFile: File) => {
      if (
        !SUPPORTED_TYPES.includes(selectedFile.type) &&
        !selectedFile.name.match(/\.(jpg|jpeg|png|bmp|webp|mp4|mov|avi|mkv|webm)$/i)
      ) {
        setErrorMessage(
          `Unsupported format: "${selectedFile.type || selectedFile.name.split(".").pop()}". Please upload a JPG, PNG, MP4, or MOV file.`
        );
        setPhase("error");
        return;
      }

      if (selectedFile.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
        setErrorMessage(
          `File size (${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB) exceeds the ${MAX_FILE_SIZE_MB} MB limit.`
        );
        setPhase("error");
        return;
      }

      setFile(selectedFile);
      setErrorMessage("");

      const url = URL.createObjectURL(selectedFile);
      setPreview(url);
    },
    []
  );

  const handleLoadSample = useCallback(
    async (samplePath: string, sampleTitle: string) => {
      try {
        setProcessingStage(`Loading ${sampleTitle}...`);
        setPhase("processing");
        setUploadProgress(20);

        const res = await fetch(samplePath);
        const blob = await res.blob();
        const sampleFile = new File([blob], samplePath.split("/").pop() || "sample.jpg", {
          type: "image/jpeg",
        });

        setFile(sampleFile);
        setPreview(samplePath);
        setUploadProgress(50);
        setProcessingStage("Running multi-model perception analysis...");

        const apiRes = await analyzeMedia(sampleFile, confidence, (pct) => {
          setUploadProgress(Math.max(50, pct));
        });

        if (isSuccess(apiRes)) {
          setResult(apiRes);
          setCurrentFrameIdx(0);
          setPhase("results");
        } else if (isError(apiRes)) {
          setErrorMessage(apiRes.error);
          setPhase("error");
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Failed to load sample.";
        setErrorMessage(msg);
        setPhase("error");
      }
    },
    [confidence]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile) handleFileSelect(droppedFile);
    },
    [handleFileSelect]
  );

  const handleAnalyze = useCallback(async () => {
    if (!file) return;

    setPhase("processing");
    setUploadProgress(0);
    setProcessingStage("Uploading media to perception engine...");

    try {
      const response = await analyzeMedia(file, confidence, (pct) => {
        setUploadProgress(pct);
        if (pct < 100) {
          setProcessingStage("Uploading video stream...");
        } else {
          setProcessingStage("Fusing YOLO11, ByteTrack, TTC, ego-motion and depth metrics...");
        }
      });

      if (isSuccess(response)) {
        setResult(response);
        setCurrentFrameIdx(0);
        setPhase("results");
      } else if (isError(response)) {
        setErrorMessage(response.error);
        setPhase("error");
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "An unexpected error occurred.";
      setErrorMessage(message);
      setPhase("error");
    }
  }, [file, confidence]);

  const handleReset = useCallback(() => {
    setPhase("upload");
    setFile(null);
    setPreview(null);
    setResult(null);
    setSelectedObject(null);
    setErrorMessage("");
    setUploadProgress(0);
    setCurrentFrameIdx(0);
    setIsPlaying(false);
    setShowTimeline(false);
    alertEngine.clear();
  }, []);


  // Video playback controls
  const togglePlay = useCallback(() => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      alertEngine.resume();
      unlockAudio();
      videoRef.current.play();
      setIsPlaying(true);
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
      alertEngine.stop();
    }
  }, []);

  const handleSeek = useCallback(
    (time: number) => {
      if (!videoRef.current || !result) return;
      alertEngine.stop();
      videoRef.current.currentTime = time;
      setCurrentTime(time);

      let closest = 0;
      let minDiff = Infinity;
      for (let i = 0; i < result.frames.length; i++) {
        const diff = Math.abs(result.frames[i].timestamp_s - time);
        if (diff < minDiff) {
          minDiff = diff;
          closest = i;
        }
      }
      setCurrentFrameIdx(closest);
    },
    [result]
  );

  const stepFrame = useCallback(
    (direction: -1 | 1) => {
      if (!result) return;
      alertEngine.stop();
      const nextIdx = Math.max(
        0,
        Math.min(result.frames.length - 1, currentFrameIdx + direction)
      );
      setCurrentFrameIdx(nextIdx);
      if (videoRef.current) {
        const nextTime = result.frames[nextIdx].timestamp_s;
        videoRef.current.currentTime = nextTime;
        setCurrentTime(nextTime);
      }
    },
    [result, currentFrameIdx]
  );

  const changePlaybackRate = useCallback((rate: number) => {
    if (!videoRef.current) return;
    videoRef.current.playbackRate = rate;
    setPlaybackRate(rate);
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (!videoContainerRef.current) return;
    if (!document.fullscreenElement) {
      videoContainerRef.current.requestFullscreen().then(() => setIsFullscreen(true));
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false));
    }
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  // Video synchronization
  useEffect(() => {
    if (!result || result.media_type !== "video" || !videoRef.current) return;
    const video = videoRef.current;

    const onLoadedMetadata = () => {
      setDuration(video.duration || result.duration_s);
    };

    const onTimeUpdate = () => {
      const time = video.currentTime;
      setCurrentTime(time);

      let closest = 0;
      let minDiff = Infinity;
      for (let i = 0; i < result.frames.length; i++) {
        const diff = Math.abs(result.frames[i].timestamp_s - time);
        if (diff < minDiff) {
          minDiff = diff;
          closest = i;
        }
      }
      setCurrentFrameIdx(closest);
    };

    const onPause = () => {
      setIsPlaying(false);
      alertEngine.stop();
    };

    const onEnded = () => {
      setIsPlaying(false);
      alertEngine.stop();
    };

    video.addEventListener("loadedmetadata", onLoadedMetadata);
    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("pause", onPause);
    video.addEventListener("ended", onEnded);

    return () => {
      video.removeEventListener("loadedmetadata", onLoadedMetadata);
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("ended", onEnded);
    };
  }, [result]);


  // Main Canvas Rendering
  useEffect(() => {
    if (!result || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (!showOverlays) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const frame = result.frames[currentFrameIdx];
    if (!frame) return;

    const scaleX = canvas.width / result.frame_width;
    const scaleY = canvas.height / result.frame_height;

    renderFrame(
      ctx,
      frame,
      scaleX,
      scaleY,
      confidence,
      selectedObject?.track_id ?? null
    );
  }, [result, currentFrameIdx, confidence, selectedObject, showOverlays]);

  // Modal Canvas Rendering
  useEffect(() => {
    if (!isImageModalOpen || !result || !modalCanvasRef.current) return;
    const canvas = modalCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const frame = result.frames[currentFrameIdx];
    if (!frame) return;

    const scaleX = canvas.width / result.frame_width;
    const scaleY = canvas.height / result.frame_height;

    renderFrame(
      ctx,
      frame,
      scaleX,
      scaleY,
      confidence,
      selectedObject?.track_id ?? null
    );
  }, [isImageModalOpen, result, currentFrameIdx, confidence, selectedObject, imageZoom]);

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!result || !canvasRef.current) return;
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      const frame = result.frames[currentFrameIdx];
      if (!frame) return;

      const scaleX = canvas.width / result.frame_width;
      const scaleY = canvas.height / result.frame_height;

      const hit = hitTestDetection(frame, clickX, clickY, scaleX, scaleY, confidence);
      if (hit) {
        setSelectedObject(hit);
      } else if (result.media_type === "video") {
        togglePlay();
      }
    },
    [result, currentFrameIdx, confidence, togglePlay]
  );

  const handleExportJSON = useCallback(() => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `atmanirbhar-telemetry-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result]);

  const jumpToNearMiss = useCallback(
    (event: NearMissEvent) => {
      if (!result) return;
      setCurrentFrameIdx(event.frame_index);
      if (videoRef.current) {
        videoRef.current.currentTime = event.timestamp_s;
        setCurrentTime(event.timestamp_s);
      }
    },
    [result]
  );

  const currentFrame = result?.frames[currentFrameIdx];
  const isTargetVisible = useCallback(
    (o: DetectedObject) => {
      const isAnimalOrObstacle =
        ["cow", "dog", "horse", "sheep", "goat", "cat", "elephant", "bear", "road-obstacle"].includes(
          o.class_raw?.toLowerCase() || ""
        ) ||
        o.class_name?.toLowerCase().includes("cow") ||
        o.class_name?.toLowerCase().includes("obstacle") ||
        o.risk_level === "High";

      const effConf = isAnimalOrObstacle
        ? Math.max(0.08, confidence * 0.45)
        : confidence;
      return o.confidence >= effConf;
    },
    [confidence]
  );
  const activeObjects = currentFrame?.objects.filter(isTargetVisible) || [];


  return (
    <div className="min-h-screen bg-[#0C0306] text-[#F7F1EA] flex flex-col justify-between selection:bg-[#670626] selection:text-[#F7F1EA]">
      {/* --- Symmetrical Luxury Navbar --- */}
      <header className="sticky top-0 z-50 bg-[#16050C]/90 backdrop-blur-xl border-b border-[#670626]/40 shadow-xl">
        <div className="max-w-[1560px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">

          {/* Logo & Brand */}
          <div className="flex items-center gap-3.5">
            <div className="relative w-10 h-10 rounded-full overflow-hidden border border-[#F7F1EA]/30 shadow-[0_0_15px_rgba(103,6,38,0.6)]">
              <Image
                src="/logo.jpg"
                alt="SARTHI Vision Logo"
                fill
                sizes="40px"
                className="object-cover"
                priority
              />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-heading text-base font-bold tracking-wider text-[#F7F1EA]">
                  ATMANIRBHAR AI
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#670626] text-[#F7F1EA] font-semibold border border-[#F7F1EA]/20">
                  v3.0 Production
                </span>
              </div>
              <p className="text-[11px] font-body text-[#D8CCC0]">
                India-first autonomous driving Multi-Agents AI Intelligence for unstructured roads
              </p>
            </div>
          </div>

          {/* Right Status / Action Controls */}
          <div className="flex items-center gap-4">
            {backendOnline !== null && (
              <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#220814] border border-[#670626]/60">
                <span
                  className={`w-2 h-2 rounded-full ${
                    backendOnline ? "bg-[#10B981] shadow-[0_0_8px_#10B981]" : "bg-[#EF4444] shadow-[0_0_8px_#EF4444]"
                  }`}
                />
                <span className="text-xs font-mono text-[#D8CCC0]">
                  {backendOnline ? "Core Online : 8000" : "Core Offline"}
                </span>
              </div>
            )}

            {phase === "results" && (
              <button
                onClick={handleReset}
                className="btn-cherry flex items-center gap-2 text-xs"
              >
                <RotateCcw size={14} /> New Inspection
              </button>
            )}
          </div>
        </div>
      </header>

      {/* --- HERO & UPLOAD SECTION (CENTERED & SPACIOUS) --- */}
      {phase === "upload" && (
        <main className="flex-1 max-w-[1560px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 flex flex-col items-center justify-center text-center animate-fade-in-up">

          {/* Floating Logo Badge */}
          <div className="relative mb-6 group">
            <div className="absolute -inset-2 bg-gradient-to-r from-[#670626] via-[#A31243] to-[#D11D56] rounded-full blur-xl opacity-60 group-hover:opacity-90 transition duration-1000" />
            <div className="relative w-24 h-24 rounded-full overflow-hidden border-2 border-[#F7F1EA]/40 shadow-2xl animate-float">
              <Image
                src="/logo.jpg"
                alt="AtmaNirbhar AI Logo Orb"
                fill
                sizes="96px"
                className="object-cover"
              />
            </div>
          </div>

          {/* Centered Main Typography */}
          <div className="max-w-3xl mx-auto space-y-4 mb-10">
            <h1 className="font-heading text-4xl sm:text-5xl md:text-6xl font-extrabold tracking-tight text-[#F7F1EA] leading-tight">
              India-First Autonomous Driving{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#F7F1EA] via-[#E8D0BD] to-[#A31243]">
                Multi-Agents AI Intelligence
              </span>
            </h1>
            <p className="font-body text-[#D8CCC0] text-base sm:text-lg max-w-2xl mx-auto leading-relaxed">
              Engineered specifically for unstructured Indian roadways: Real-time multi-agent vision, acoustic emergency siren detection, 3D LiDAR point clouds, and dynamic sub-100ms collision replanning.
            </p>
          </div>

          {/* 3D Spatial Visualizer Showcase Banner (EXPANSIVE COMMANDING VIEW) */}
          <div className="w-full max-w-6xl mx-auto mb-14 rounded-3xl overflow-hidden border border-[#670626]/70 shadow-[0_0_60px_rgba(103,6,38,0.35)] bg-[#12030A]/95 backdrop-blur-xl">
            {/* Showcase Header with Scenario Switcher */}
            <div className="flex flex-col md:flex-row md:items-center justify-between px-6 py-4 border-b border-[#670626]/50 bg-[#1A0510]/95 gap-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-xl bg-[#670626] flex items-center justify-center border border-[#F7F1EA]/20 shadow-md">
                  <Box size={18} className="text-[#F7F1EA]" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-[#10B981] animate-pulse shadow-[0_0_8px_#10B981]" />
                    <span className="font-heading text-xs sm:text-sm font-bold tracking-wider text-[#F7F1EA]">
                      3D AUTONOMOUS SPATIAL DIGITAL TWIN
                    </span>
                  </div>
                  <span className="font-mono text-[11px] text-[#9E8F81] block">
                    Real-Time Monocular Pinhole Reconstruction • 64-Beam LiDAR Simulation
                  </span>
                </div>
              </div>

              {/* Scenario Switcher Pills */}
              <div className="flex items-center gap-1.5 bg-[#14050D] p-1 rounded-xl border border-[#670626]/50 font-heading text-xs">
                <button
                  onClick={() => setDemoScenario("bovine")}
                  className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                    demoScenario === "bovine"
                      ? "bg-[#670626] text-[#F7F1EA] font-bold shadow-md"
                      : "text-[#D8CCC0] hover:text-[#F7F1EA]"
                  }`}
                >
                  <AlertTriangle size={12} className={demoScenario === "bovine" ? "text-[#FF708F]" : "text-[#EF4444]"} />
                  Bovine (TTC 3.2s)
                </button>
                <button
                  onClick={() => setDemoScenario("traffic")}
                  className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                    demoScenario === "traffic"
                      ? "bg-[#670626] text-[#F7F1EA] font-bold shadow-md"
                      : "text-[#D8CCC0] hover:text-[#F7F1EA]"
                  }`}
                >
                  <Layers size={12} />
                  Urban Cut-In
                </button>
                <button
                  onClick={() => setDemoScenario("night")}
                  className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                    demoScenario === "night"
                      ? "bg-[#670626] text-[#F7F1EA] font-bold shadow-md"
                      : "text-[#D8CCC0] hover:text-[#F7F1EA]"
                  }`}
                >
                  <Eye size={12} />
                  Night Cattle
                </button>
              </div>
            </div>

            {/* Scenario Description Sub-Bar */}
            <div className="px-6 py-2 bg-[#14040D] border-b border-[#670626]/30 flex flex-wrap items-center justify-between text-xs font-mono text-[#D8CCC0]">
              <span className="text-[#FF708F] font-bold">
                {DEMO_SCENARIOS[demoScenario].title}
              </span>
              <span className="text-[#9E8F81] text-[11px]">
                {DEMO_SCENARIOS[demoScenario].desc}
              </span>
            </div>

            {/* Expansive 3D Canvas Viewport */}
            <div className="h-[520px] sm:h-[580px] w-full relative">
              <LidarVisualizer
                objects={DEMO_SCENARIOS[demoScenario].objects}
                chaosScore={DEMO_SCENARIOS[demoScenario].chaosScore}
              />
            </div>
          </div>

          {/* Quick Benchmark 1-Click Samples */}
          <div className="w-full max-w-4xl mx-auto mb-10 text-left">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Sparkles size={16} className="text-[#A31243]" />
                <h3 className="font-heading text-sm font-semibold tracking-wider text-[#F7F1EA]">
                  Quick Benchmark Samples
                </h3>
              </div>
              <span className="text-xs font-mono text-[#D8CCC0]">
                1-Click Execution
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {SAMPLE_FILES.map((sample) => (
                <div
                  key={sample.id}
                  onClick={() => handleLoadSample(sample.path, sample.title)}
                  className="glass-card p-5 rounded-xl cursor-pointer hover:border-[#F7F1EA]/40 hover:bg-[#2A0B1B]/70 transition-all group flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span
                        className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full"
                        style={{
                          background: `${sample.badgeColor}22`,
                          color: sample.badgeColor,
                          border: `1px solid ${sample.badgeColor}55`,
                        }}
                      >
                        {sample.badge}
                      </span>
                      <ArrowRight
                        size={14}
                        className="text-[#D8CCC0] group-hover:text-[#F7F1EA] group-hover:translate-x-1 transition-all"
                      />
                    </div>
                    <h4 className="font-heading text-sm font-bold text-[#F7F1EA] mb-1">
                      {sample.title}
                    </h4>
                    <p className="text-xs text-[#D8CCC0] leading-relaxed">
                      {sample.desc}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Centered Upload Drop Zone */}
          <div className="w-full max-w-3xl mx-auto mb-8">
            <div
              className={`upload-zone-luxury ${isDragging ? "dragover" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  fileInputRef.current?.click();
                }
              }}
              aria-label="Upload footage"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.bmp,.webp,.mp4,.mov,.avi,.mkv,.webm"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFileSelect(f);
                }}
              />

              {file ? (
                <div className="flex flex-col items-center gap-3">
                  {file.type.startsWith("image/") ? (
                    <FileImage size={48} className="text-[#F7F1EA]" />
                  ) : (
                    <FileVideo size={48} className="text-[#F7F1EA]" />
                  )}
                  <div>
                    <p className="font-heading text-base font-bold text-[#F7F1EA]">
                      {file.name}
                    </p>
                    <p className="text-xs font-mono text-[#D8CCC0] mt-1">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB | Ready for analysis
                    </p>
                  </div>
                  <button
                    className="btn-glass text-xs mt-2"
                    onClick={(e) => {
                      e.stopPropagation();
                      setFile(null);
                      setPreview(null);
                    }}
                  >
                    <X size={12} /> Remove Selection
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3">
                  <div className="w-16 h-16 rounded-2xl flex items-center justify-center bg-[#670626]/40 border border-[#F7F1EA]/20 shadow-lg">
                    <Upload size={28} className="text-[#F7F1EA]" />
                  </div>
                  <p className="font-heading text-lg font-bold text-[#F7F1EA]">
                    Upload Dashcam Video or High-Resolution Frame
                  </p>
                  <p className="text-xs text-[#D8CCC0] max-w-sm">
                    Drag and drop MP4, MOV, PNG, or JPG. Maximum {MAX_DURATION_S}s duration and {MAX_FILE_SIZE_MB}MB size.
                  </p>
                </div>
              )}
            </div>

            {/* Confidence Slider */}
            {file && (
              <div className="glass-card p-5 mt-4 rounded-xl text-left">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <SlidersHorizontal size={14} className="text-[#D8CCC0]" />
                    <span className="font-heading text-xs font-semibold text-[#F7F1EA]">
                      Perception Confidence Threshold
                    </span>
                  </div>
                  <span className="font-mono text-sm font-bold text-[#F7F1EA]">
                    {(confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  className="confidence-slider-cherry"
                  min="0.1"
                  max="0.95"
                  step="0.05"
                  value={confidence}
                  onChange={(e) => setConfidence(parseFloat(e.target.value))}
                />
                <div className="flex justify-between text-[11px] font-mono text-[#D8CCC0] mt-1">
                  <span>10% (High Recall / Distant)</span>
                  <span>95% (High Precision Only)</span>
                </div>
              </div>
            )}

            {/* Run Pipeline CTA Button */}
            {file && (
              <div className="mt-6 text-center">
                <button
                  onClick={handleAnalyze}
                  disabled={backendOnline === false}
                  className="btn-cherry text-sm py-4 px-10 rounded-xl font-bold tracking-wider"
                >
                  <Search size={18} /> RUN PERCEPTION PIPELINE
                </button>
                {backendOnline === false && (
                  <p className="text-xs mt-2 text-[#EF4444]">
                    Backend API offline. Ensure port 8000 server is active.
                  </p>
                )}
              </div>
            )}
          </div>
        </main>
      )}

      {/* --- PROCESSING SCREEN --- */}
      {phase === "processing" && (
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
          <div className="glass-card-elevated p-10 max-w-md w-full rounded-2xl border border-[#670626] shadow-2xl">
            <Loader2 size={48} className="mx-auto mb-5 text-[#A31243] animate-spin" />
            <h3 className="font-heading text-lg font-bold text-[#F7F1EA] mb-2">
              {processingStage}
            </h3>
            <div className="w-full h-2 bg-[#220814] rounded-full overflow-hidden my-4 border border-[#670626]/40">
              <div
                className="h-full bg-gradient-to-r from-[#670626] via-[#A31243] to-[#F7F1EA] transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <p className="font-mono text-xs text-[#D8CCC0]">
              {uploadProgress < 100
                ? `Transmitting Data: ${uploadProgress}%`
                : "Fusing YOLO11, ByteTrack, TTC, Ego-Motion, Distance and Chaos calculation..."}
            </p>
          </div>
        </div>
      )}

      {/* --- ERROR SCREEN --- */}
      {phase === "error" && (
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
          <div className="glass-card p-10 max-w-md w-full rounded-2xl border border-[#EF4444]/60 shadow-2xl">
            <AlertTriangle size={48} className="mx-auto mb-4 text-[#EF4444]" />
            <h3 className="font-heading text-lg font-bold text-[#FF708F] mb-2">
              Perception Error
            </h3>
            <p className="text-xs font-body text-[#D8CCC0] mb-6 leading-relaxed">
              {errorMessage}
            </p>
            <button onClick={handleReset} className="btn-cherry">
              <RotateCcw size={14} /> Try Another Footage
            </button>
          </div>
        </div>
      )}

      {/* --- RESULTS DASHBOARD (CENTERED & BALANCED) --- */}
      {phase === "results" && result && currentFrame && (
        <main className="flex-1 max-w-[1560px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6 animate-fade-in-up">

          {/* Top Analytics Metrics Banner */}
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
            {/* Chaos Score Gauge */}
            <div className="glass-card p-4 rounded-xl flex items-center gap-4 col-span-2 md:col-span-1 border border-[#670626]/50">
              <div className="relative w-14 h-14 shrink-0">
                <svg viewBox="0 0 100 100" className="w-full h-full">
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    fill="none"
                    stroke="#220814"
                    strokeWidth="8"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    fill="none"
                    stroke={
                      currentFrame.chaos_score > 65
                        ? "#EF4444"
                        : currentFrame.chaos_score > 35
                        ? "#F59E0B"
                        : "#10B981"
                    }
                    strokeWidth="8"
                    strokeLinecap="round"
                    style={{
                      strokeDasharray: 251,
                      strokeDashoffset:
                        251 - (251 * currentFrame.chaos_score) / 100,
                      transform: "rotate(-90deg)",
                      transformOrigin: "50% 50%",
                      transition: "stroke-dashoffset 0.6s ease",
                    }}
                  />
                  <text
                    x="50"
                    y="50"
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="#F7F1EA"
                    fontFamily="var(--font-heading)"
                    fontSize="22"
                    fontWeight="700"
                  >
                    {Math.round(currentFrame.chaos_score)}
                  </text>
                </svg>
              </div>
              <div>
                <p className="font-heading text-xs font-bold text-[#D8CCC0] uppercase tracking-wider">
                  Scene Chaos
                </p>
                <p className="font-mono text-xs text-[#F7F1EA] font-semibold">
                  {currentFrame.chaos_score > 65
                    ? "Severe Hazard"
                    : currentFrame.chaos_score > 35
                    ? "Moderate Traffic"
                    : "Stable Flow"}
                </p>
              </div>
            </div>

            {/* Objects count */}
            <div className="glass-card p-4 rounded-xl border border-[#670626]/40 flex flex-col justify-center text-center">
              <span className="font-heading text-[11px] font-bold text-[#D8CCC0] uppercase tracking-wider">
                Active Targets
              </span>
              <span className="font-mono text-2xl font-bold text-[#F7F1EA]">
                {activeObjects.length}
              </span>
              <span className="text-[10px] font-mono text-[#D8CCC0]">
                {result.total_unique_tracks} total tracked
              </span>
            </div>

            {/* Classes count */}
            <div className="glass-card p-4 rounded-xl border border-[#670626]/40 flex flex-col justify-center text-center">
              <span className="font-heading text-[11px] font-bold text-[#D8CCC0] uppercase tracking-wider">
                Classes
              </span>
              <span className="font-mono text-2xl font-bold text-[#F7F1EA]">
                {result.unique_classes.length}
              </span>
              <span className="text-[10px] font-mono text-[#D8CCC0] truncate">
                {result.unique_classes.join(", ") || "None"}
              </span>
            </div>

            {/* Frames */}
            <div className="glass-card p-4 rounded-xl border border-[#670626]/40 flex flex-col justify-center text-center">
              <span className="font-heading text-[11px] font-bold text-[#D8CCC0] uppercase tracking-wider">
                Frame Samples
              </span>
              <span className="font-mono text-2xl font-bold text-[#F7F1EA]">
                {result.total_frames}
              </span>
              <span className="text-[10px] font-mono text-[#D8CCC0]">
                {result.media_type === "video"
                  ? `${result.duration_s}s @ ${result.sample_fps} FPS`
                  : "Static Frame"}
              </span>
            </div>

            {/* Near-miss count */}
            <div className="glass-card p-4 rounded-xl border border-[#670626]/40 flex flex-col justify-center text-center">
              <span className="font-heading text-[11px] font-bold text-[#D8CCC0] uppercase tracking-wider">
                Near-Miss Events
              </span>
              <span className={`font-mono text-2xl font-bold ${(result.near_miss_events?.length || 0) > 0 ? "text-[#EF4444]" : "text-[#10B981]"}`}>
                {result.near_miss_events?.length || 0}
              </span>
              <span className="text-[10px] font-mono text-[#D8CCC0]">
                TTC violations
              </span>
            </div>

            {/* Processing time */}
            <div className="glass-card p-4 rounded-xl border border-[#670626]/40 flex flex-col justify-center text-center">
              <span className="font-heading text-[11px] font-bold text-[#D8CCC0] uppercase tracking-wider">
                Latency
              </span>
              <span className="font-mono text-2xl font-bold text-[#10B981]">
                {result.processing_time_s}s
              </span>
              <span className="text-[10px] font-mono text-[#D8CCC0]">
                Wall-clock inference
              </span>
            </div>
          </div>

          {/* Viewport Mode Switcher & Filter Bar */}
          <div className="glass-card p-3 rounded-xl border border-[#670626]/50 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setViewMode("split")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-heading font-semibold transition-all ${
                  viewMode === "split"
                    ? "bg-[#F7F1EA] text-[#3F0317] shadow-md"
                    : "text-[#D8CCC0] hover:text-[#F7F1EA]"
                }`}
              >
                <Layers size={14} /> Split View
              </button>
              <button
                onClick={() => setViewMode("feed")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-heading font-semibold transition-all ${
                  viewMode === "feed"
                    ? "bg-[#F7F1EA] text-[#3F0317] shadow-md"
                    : "text-[#D8CCC0] hover:text-[#F7F1EA]"
                }`}
              >
                <Eye size={14} /> Camera Feed
              </button>
              <button
                onClick={() => setViewMode("spatial")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-heading font-semibold transition-all ${
                  viewMode === "spatial"
                    ? "bg-[#F7F1EA] text-[#3F0317] shadow-md"
                    : "text-[#D8CCC0] hover:text-[#F7F1EA]"
                }`}
              >
                <Box size={14} /> 3D Spatial View
              </button>
            </div>

            <div className="flex items-center gap-4">
              {/* Confidence threshold slider */}
              <div className="flex items-center gap-2">
                <SlidersHorizontal size={13} className="text-[#D8CCC0]" />
                <input
                  type="range"
                  className="confidence-slider-cherry w-28"
                  min="0.1"
                  max="0.95"
                  step="0.05"
                  value={confidence}
                  onChange={(e) => setConfidence(parseFloat(e.target.value))}
                />
                <span className="font-mono text-xs text-[#F7F1EA] font-bold">
                  {(confidence * 100).toFixed(0)}%
                </span>
              </div>

              {/* Voice toggle */}
              <button
                onClick={() => setVoiceEnabled(!voiceEnabled)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-heading border transition-all ${
                  voiceEnabled
                    ? "bg-[#670626] text-[#F7F1EA] border-[#F7F1EA]/30"
                    : "bg-transparent text-[#D8CCC0] border-[#670626]/60"
                }`}
                title={voiceEnabled ? "Voice alerts on" : "Voice alerts off"}
              >
                {voiceEnabled ? <Volume2 size={13} /> : <VolumeX size={13} />}
                Voice
              </button>

              {/* Test Voice Warning button */}
              <button
                onClick={() => {
                  unlockAudio();
                  alertEngine.enqueue(
                    "Collision predicted in 3.5 seconds. Bovine hazard ahead, apply emergency brakes.",
                    "imminent"
                  );
                }}
                className="btn-cherry flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-heading shadow-md hover:brightness-110 active:scale-95 transition-all"
                title="Test live predictive collision voice warning prior to impact"
              >
                <Radio size={13} className="animate-pulse text-[#F7F1EA]" />
                Test Voice Alert
              </button>


              {/* Near-miss timeline toggle */}
              {result.near_miss_events && result.near_miss_events.length > 0 && (
                <button
                  onClick={() => setShowTimeline(!showTimeline)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-heading border transition-all ${
                    showTimeline
                      ? "bg-[#EF4444]/20 text-[#FF708F] border-[#EF4444]/40"
                      : "bg-transparent text-[#D8CCC0] border-[#670626]/60"
                  }`}
                >
                  <Clock size={13} /> Timeline
                </button>
              )}

              {/* View full size image button (when image) */}
              {result.media_type === "image" && (
                <button
                  onClick={() => setIsImageModalOpen(true)}
                  className="btn-glass text-xs flex items-center gap-1.5"
                >
                  <Maximize2 size={13} /> View Full Size
                </button>
              )}

              {/* JSON export */}
              <button
                onClick={handleExportJSON}
                className="btn-glass text-xs flex items-center gap-1.5"
              >
                <Download size={13} /> JSON
              </button>
            </div>
          </div>

          {/* Near-Miss Timeline Panel */}
          {showTimeline && result.near_miss_events && result.near_miss_events.length > 0 && (
            <div className="glass-card-elevated rounded-xl border border-[#EF4444]/30 overflow-hidden">
              <div className="px-4 py-3 bg-[#1C0612] border-b border-[#670626]/50 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle size={14} className="text-[#EF4444]" />
                  <span className="font-heading text-xs font-bold text-[#F7F1EA]">
                    PREDICTIVE COLLISION TIMELINE (PRIOR ADVANCE ADVISORIES)
                  </span>
                  <span className="font-mono text-[10px] text-[#D8CCC0]">
                    {result.near_miss_events.length} advance predictions
                  </span>
                </div>
                <button onClick={() => setShowTimeline(false)} className="text-[#D8CCC0] hover:text-[#F7F1EA]">
                  <X size={14} />
                </button>
              </div>
              <div className="max-h-48 overflow-y-auto p-3 space-y-2">
                {result.near_miss_events.map((event, idx) => (
                  <div
                    key={idx}
                    onClick={() => jumpToNearMiss(event)}
                    className="flex items-center justify-between px-3 py-2 rounded-lg bg-[#18050E] border border-[#670626]/40 cursor-pointer hover:border-[#EF4444]/60 transition-all text-xs"
                  >
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-[#EF4444] font-bold w-16">
                        {event.timestamp_s.toFixed(1)}s
                      </span>
                      <span className="font-heading text-[#F7F1EA] font-semibold">
                        {event.class_name}
                      </span>
                      <span className="text-[#D8CCC0]">
                        {event.event_type === "object_object" ? "Pair Convergence" : "Prior Advance Warning"}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[#EF4444] font-bold bg-[#EF4444]/15 px-2 py-0.5 rounded border border-[#EF4444]/30">
                        PREDICTED {event.ttc_s.toFixed(1)}s PRIOR
                      </span>
                      <ChevronRight size={12} className="text-[#D8CCC0]" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Main Inspection Grid (Video/Image Viewport + Side Roster) */}
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
            {/* Viewport Area */}
            <div className="xl:col-span-8 space-y-4">
              {(viewMode === "feed" || viewMode === "split") && (
                <div
                  ref={videoContainerRef}
                  className={`glass-card-elevated rounded-2xl overflow-hidden border border-[#670626]/70 relative flex flex-col ${
                    isFullscreen ? "fixed inset-0 z-50 bg-black max-h-screen" : ""
                  }`}
                >
                  {/* Top Tactical Banner */}
                  <div className="flex items-center justify-between px-4 py-2 bg-[#1C0612]/90 border-b border-[#670626]/50 text-xs font-mono text-[#D8CCC0]">
                    <span className="flex items-center gap-2 text-[#F7F1EA] font-semibold">
                      <span className="w-2 h-2 rounded-full bg-[#10B981] animate-pulse shadow-[0_0_6px_#10B981]" />
                      FORWARD CAMERA STREAM [HUD ACTIVE]
                    </span>
                    <div className="flex items-center gap-3">
                      {currentFrame.low_light_enhanced && (
                        <span className="text-[#F59E0B] font-semibold flex items-center gap-1">
                          <Zap size={10} /> CLAHE Enhanced
                        </span>
                      )}
                      <span>
                        {result.frame_width}x{result.frame_height}
                      </span>
                    </div>
                  </div>

                  {/* Live Voice Alert Speaking Toast Banner */}
                  {speakingPhrase && (
                    <div className="mx-3 my-2 p-2.5 rounded-xl bg-[#2A0818]/95 border border-[#EF4444]/60 shadow-[0_0_20px_rgba(239,68,68,0.3)] flex items-center justify-between gap-3 animate-fade-in-up z-20">
                      <div className="flex items-center gap-2.5 overflow-hidden">
                        <div className="w-6 h-6 rounded-full bg-[#EF4444] flex items-center justify-center shrink-0 shadow-[0_0_10px_#EF4444]">
                          <Volume2 size={13} className="text-[#F7F1EA] animate-pulse" />
                        </div>
                        <div className="truncate">
                          <span className="text-[10px] font-mono uppercase tracking-wider text-[#FF708F] font-bold block">
                            LIVE COLLISION VOICE WARNING
                          </span>
                          <span className="text-xs font-heading font-semibold text-[#F7F1EA] truncate block">
                            {speakingPhrase}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-1 shrink-0 pr-1">
                        <span className="w-1 h-3 bg-[#EF4444] rounded-full animate-pulse" />
                        <span className="w-1 h-5 bg-[#FF708F] rounded-full animate-pulse [animation-delay:150ms]" />
                        <span className="w-1 h-2 bg-[#EF4444] rounded-full animate-pulse [animation-delay:300ms]" />
                        <span className="w-1 h-4 bg-[#FF708F] rounded-full animate-pulse [animation-delay:450ms]" />
                      </div>
                    </div>
                  )}


                  {/* Centered Media Container */}
                  <div className="relative bg-black flex items-center justify-center w-full overflow-hidden flex-1 p-1">
                    <div
                      className="relative max-h-[58vh] max-w-full flex items-center justify-center rounded-xl overflow-hidden shadow-2xl"
                      style={{
                        aspectRatio: `${result.frame_width} / ${result.frame_height}`,
                        width: `min(100%, calc(58vh * (${result.frame_width} / ${result.frame_height})))`,
                      }}
                    >
                      {result.media_type === "video" && preview ? (
                        <video
                          ref={videoRef}
                          src={preview}
                          playsInline
                          preload="auto"
                          className="w-full h-full object-fill block"
                          onClick={togglePlay}
                        />
                      ) : preview ? (
                        <img
                          src={preview}
                          alt="Analyzed target"
                          className="w-full h-full object-fill block"
                        />
                      ) : null}

                      {/* Pixel-perfect Overlay Canvas */}
                      <canvas
                        ref={canvasRef}
                        className="absolute inset-0 w-full h-full cursor-crosshair block"
                        width={result.frame_width}
                        height={result.frame_height}
                        onClick={handleCanvasClick}
                      />
                    </div>
                  </div>

                  {/* Integrated Tactical Video Controller */}
                  {result.media_type === "video" && (
                    <div className="px-4 py-3 bg-[#1C0612]/95 border-t border-[#670626]/50 space-y-2">
                      {/* Timeline Scrubber */}
                      <div className="flex items-center gap-3">
                        <span className="font-mono text-xs text-[#D8CCC0] w-12 text-right">
                          {Math.floor(currentTime / 60)}:
                          {Math.floor(currentTime % 60)
                            .toString()
                            .padStart(2, "0")}
                        </span>
                        <input
                          type="range"
                          min="0"
                          max={duration || result.duration_s || 1}
                          step="0.05"
                          value={currentTime}
                          onChange={(e) => handleSeek(parseFloat(e.target.value))}
                          className="confidence-slider-cherry flex-1"
                        />
                        <span className="font-mono text-xs text-[#D8CCC0] w-12">
                          {Math.floor((duration || result.duration_s) / 60)}:
                          {Math.floor((duration || result.duration_s) % 60)
                            .toString()
                            .padStart(2, "0")}
                        </span>
                      </div>

                      {/* Action Buttons Row */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={togglePlay}
                            className="btn-cherry p-2 rounded-lg"
                            aria-label={isPlaying ? "Pause" : "Play"}
                          >
                            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
                          </button>
                          <button
                            onClick={() => stepFrame(-1)}
                            className="btn-glass p-2 rounded-lg text-xs"
                            title="Step -1 Frame"
                          >
                            <SkipBack size={14} />
                          </button>
                          <button
                            onClick={() => stepFrame(1)}
                            className="btn-glass p-2 rounded-lg text-xs"
                            title="Step +1 Frame"
                          >
                            <SkipForward size={14} />
                          </button>
                        </div>

                        {/* Speed & Overlays & Fullscreen */}
                        <div className="flex items-center gap-3">
                          <div className="flex items-center gap-1 bg-[#2D0C1B] p-1 rounded-lg border border-[#670626]/60">
                            {[0.5, 1, 2].map((rate) => (
                              <button
                                key={rate}
                                onClick={() => changePlaybackRate(rate)}
                                className={`px-2 py-0.5 text-[10px] font-mono rounded ${
                                  playbackRate === rate
                                    ? "bg-[#F7F1EA] text-[#3F0317] font-bold"
                                    : "text-[#D8CCC0]"
                                }`}
                              >
                                {rate}x
                              </button>
                            ))}
                          </div>

                          <button
                            onClick={() => setShowOverlays(!showOverlays)}
                            className={`text-xs px-3 py-1.5 rounded-lg font-heading border transition-all ${
                              showOverlays
                                ? "bg-[#670626] text-[#F7F1EA] border-[#F7F1EA]/30"
                                : "bg-transparent text-[#D8CCC0] border-[#670626]/60"
                            }`}
                          >
                            HUD {showOverlays ? "ON" : "OFF"}
                          </button>

                          <button
                            onClick={toggleFullscreen}
                            className="btn-glass p-2 rounded-lg"
                            title="Toggle Fullscreen"
                          >
                            {isFullscreen ? (
                              <Minimize2 size={16} />
                            ) : (
                              <Maximize2 size={16} />
                            )}
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* 3D Spatial Visualizer (EXPANSIVE DIGITAL TWIN) */}
              {(viewMode === "spatial" || viewMode === "split") && (
                <div
                  className={`w-full transition-all duration-300 ${
                    viewMode === "spatial" ? "h-[700px] xl:h-[740px]" : "h-[480px]"
                  }`}
                >
                  <LidarVisualizer
                    objects={activeObjects}
                    chaosScore={currentFrame.chaos_score}
                  />
                </div>
              )}
            </div>

            {/* Side Obstacle Roster (Tactical Card Inspector) */}
            <div className="xl:col-span-4">
              <div className="glass-card-elevated rounded-2xl border border-[#670626]/60 overflow-hidden flex flex-col h-full max-h-[740px]">

                <div className="p-4 bg-[#1C0612] border-b border-[#670626]/50 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Shield size={16} className="text-[#A31243]" />
                    <h3 className="font-heading text-sm font-bold text-[#F7F1EA]">
                      PERCEPTION ROSTER
                    </h3>
                  </div>
                  <span className="font-mono text-xs text-[#D8CCC0]">
                    Frame {currentFrameIdx + 1} / {result.total_frames}
                  </span>
                </div>

                <div className="p-4 overflow-y-auto space-y-3 flex-1">
                  {activeObjects.length === 0 ? (
                    <div className="p-8 text-center text-[#D8CCC0]">
                      <Search size={32} className="mx-auto mb-3 opacity-40" />
                      <p className="font-heading text-sm text-[#F7F1EA]">
                        No Targets Detected
                      </p>
                      <p className="text-xs mt-1">
                        Try lowering the perception confidence slider.
                      </p>
                    </div>
                  ) : (
                    activeObjects.map((obj) => {
                      const isSelected = selectedObject?.track_id === obj.track_id;
                      return (
                        <div
                          key={obj.track_id}
                          onClick={() =>
                            setSelectedObject(isSelected ? null : obj)
                          }
                          className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                            isSelected
                              ? "bg-[#670626]/30 border-[#F7F1EA]/60 shadow-lg shadow-[#670626]/40"
                              : "bg-[#18050E]/80 border-[#670626]/40 hover:border-[#F7F1EA]/30"
                          }`}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-heading text-sm font-bold text-[#F7F1EA]">
                              #{obj.track_id} {obj.class_name}
                            </span>
                            <span
                              className={`risk-badge risk-badge-${obj.risk_level.toLowerCase()}`}
                            >
                              {obj.risk_level}
                            </span>
                          </div>

                          <div className="flex items-center gap-3 text-xs font-mono text-[#D8CCC0] flex-wrap">
                            <span
                              className={
                                obj.status === "Moving"
                                  ? "status-moving"
                                  : "status-stationary"
                              }
                            >
                              {obj.status}
                            </span>
                            {obj.distance_available && obj.distance_m !== null ? (
                              <span className="text-[#F7F1EA] font-semibold">
                                ~{obj.distance_m}m
                                <span className="text-[10px] text-[#9E8F81] ml-0.5">
                                  {obj.distance_method === "depth_model" ? "(depth)" : "(est.)"}
                                </span>
                              </span>
                            ) : (
                              <span className="text-[#9E8F81] italic">distance unavailable</span>
                            )}
                            <span>{(obj.confidence * 100).toFixed(0)}% conf</span>
                            {obj.ttc_s !== null && obj.ttc_s !== undefined && (
                              <span className={`font-bold ${obj.ttc_s < 2 ? "text-[#EF4444]" : obj.ttc_s < 5 ? "text-[#F59E0B]" : "text-[#D8CCC0]"}`}>
                                TTC {obj.ttc_s.toFixed(1)}s
                              </span>
                            )}
                          </div>

                          {/* Expanded Risk Rationale Card */}
                          {isSelected && (
                            <div className="mt-3 p-3 rounded-lg bg-[#0C0206] border border-[#670626]/60 text-xs font-mono text-[#D8CCC0] space-y-1.5 animate-fade-in-up">
                              <div className="text-[#F7F1EA] font-bold">
                                Risk Analysis:
                              </div>
                              <p className="text-[#D8CCC0] leading-relaxed">
                                {obj.risk_explanation}
                              </p>
                              <div className="flex justify-between text-[10px] text-[#9E8F81] pt-1.5 border-t border-[#240713]">
                                <span>Closing: {obj.closing_speed} m/f</span>
                                <span>Lateral dx: {obj.rel_dx.toFixed(1)} px</span>
                                {obj.ttc_s !== null && obj.ttc_s !== undefined && (
                                  <span className="text-[#EF4444]">TTC: {obj.ttc_s.toFixed(1)}s</span>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })
                  )}
                </div>

                {/* Collision Warnings Section */}
                {currentFrame.collision_warnings && currentFrame.collision_warnings.length > 0 && (
                  <div className="p-4 border-t border-[#670626]/50 bg-[#1C0612]/50">
                    <div className="flex items-center gap-2 mb-3">
                      <Radio size={14} className="text-[#EF4444]" />
                      <span className="font-heading text-xs font-bold text-[#EF4444]">
                        COLLISION WARNINGS
                      </span>
                    </div>
                    <div className="space-y-2">
                      {currentFrame.collision_warnings.map((cw, idx) => (
                        <div
                          key={idx}
                          className={`p-2.5 rounded-lg border text-xs font-mono ${
                            cw.urgency === "imminent"
                              ? "bg-[#EF4444]/10 border-[#EF4444]/40 text-[#FF708F]"
                              : "bg-[#F59E0B]/10 border-[#F59E0B]/40 text-[#FCD34D]"
                          }`}
                        >
                          <div className="font-bold mb-1">
                            {cw.urgency.toUpperCase()}: #{cw.track_a} + #{cw.track_b}
                          </div>
                          <div className="text-[#D8CCC0]">
                            {cw.class_a} and {cw.class_b} converging, TTC {cw.ttc_s.toFixed(1)}s
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* v3: System Status Strip */}
          {result.system_status && (
            <div className="glass-card p-3 rounded-xl border border-[#670626]/40">
              <div className="flex items-center gap-2 mb-2">
                <Shield size={14} className="text-[#F7F1EA]" />
                <span className="font-heading text-xs font-bold text-[#F7F1EA] tracking-wider">
                  SYSTEM STATUS
                </span>
              </div>
              <div className="grid grid-cols-3 gap-3">
                {/* Camera Signal */}
                <div className="flex items-center gap-2 p-2 rounded-lg bg-[#0C0306]/60 border border-[#670626]/30">
                  <div
                    className="w-2 h-2 rounded-full animate-pulse"
                    style={{ backgroundColor: SIGNAL_COLORS[result.system_status.camera?.status || 'active'] }}
                  />
                  <div>
                    <div className="text-[11px] font-semibold text-[#F7F1EA]">
                      {result.system_status.camera?.label || 'Camera active'}
                    </div>
                    {result.system_status.camera?.detail && (
                      <div className="text-[9px] text-[#9E8F81]">
                        {result.system_status.camera.detail}
                      </div>
                    )}
                  </div>
                </div>
                {/* Depth Signal */}
                <div className="flex items-center gap-2 p-2 rounded-lg bg-[#0C0306]/60 border border-[#670626]/30">
                  <div
                    className="w-2 h-2 rounded-full animate-pulse"
                    style={{ backgroundColor: SIGNAL_COLORS[result.system_status.depth?.status || 'active'] }}
                  />
                  <div>
                    <div className="text-[11px] font-semibold text-[#F7F1EA]">
                      {result.system_status.depth?.label || 'Depth active'}
                    </div>
                    {result.system_status.depth?.detail && (
                      <div className="text-[9px] text-[#9E8F81]">
                        {result.system_status.depth.detail}
                      </div>
                    )}
                  </div>
                </div>
                {/* Audio Signal */}
                <div className="flex items-center gap-2 p-2 rounded-lg bg-[#0C0306]/60 border border-[#670626]/30">
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{
                      backgroundColor: SIGNAL_COLORS[result.system_status.audio?.status || 'absent'],
                      animation: result.system_status.audio?.status === 'active' ? 'pulse 2s infinite' : 'none',
                    }}
                  />
                  <div>
                    <div className="text-[11px] font-semibold text-[#F7F1EA]">
                      {result.system_status.audio?.label || 'No audio input'}
                    </div>
                    {result.system_status.audio?.detail && (
                      <div className="text-[9px] text-[#9E8F81]">
                        {result.system_status.audio.detail}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* v3: Siren Alert Banner */}
          {currentFrame?.siren_active && (
            <div className="glass-card p-3 rounded-xl border border-[#EF4444]/40 bg-[#EF4444]/5 animate-pulse">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-full bg-[#EF4444]/20">
                  <Volume2 size={18} className="text-[#EF4444]" />
                </div>
                <div>
                  <div className="font-heading text-sm font-bold text-[#EF4444]">
                    EMERGENCY SIREN DETECTED
                  </div>
                  <div className="text-xs text-[#D8CCC0] font-mono">
                    {currentFrame.siren_class || 'Siren'} detected nearby, exact location unknown.
                    Confidence: {(currentFrame.siren_confidence * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* v3: Audio Events Timeline (for videos with sirens) */}
          {result.audio_events && result.audio_events.length > 0 && (
            <div className="glass-card p-3 rounded-xl border border-[#670626]/40">
              <div className="flex items-center gap-2 mb-2">
                <Volume2 size={14} className="text-[#EF4444]" />
                <span className="font-heading text-xs font-bold text-[#F7F1EA] tracking-wider">
                  AUDIO EVENTS ({result.audio_events.length})
                </span>
              </div>
              <div className="space-y-1.5">
                {result.audio_events.map((ae: AudioEvent, idx: number) => (
                  <div key={idx} className="flex items-center gap-2 p-1.5 rounded bg-[#0C0206]/60 text-xs font-mono text-[#D8CCC0]">
                    <span className="text-[#EF4444] font-bold">{ae.class_name}</span>
                    <span>{ae.start_s.toFixed(1)}s - {ae.end_s.toFixed(1)}s</span>
                    <span className="text-[#9E8F81]">({(ae.confidence * 100).toFixed(0)}%)</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* v3: Pipeline Timing Breakdown */}
          {result.pipeline_timings && result.pipeline_timings.per_frame_avg_ms > 0 && (
            <div className="glass-card p-3 rounded-xl border border-[#670626]/40">
              <div className="flex items-center gap-2 mb-2">
                <Clock size={14} className="text-[#F59E0B]" />
                <span className="font-heading text-xs font-bold text-[#F7F1EA] tracking-wider">
                  PIPELINE LATENCY (measured, not estimated)
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono text-[#D8CCC0]">
                <div className="p-2 rounded bg-[#0C0306]/60">
                  <div className="text-[#9E8F81] text-[10px]">Per-Frame Avg</div>
                  <div className="text-[#F7F1EA] font-bold">
                    {result.pipeline_timings.per_frame_avg_ms.toFixed(1)} ms
                  </div>
                </div>
                <div className="p-2 rounded bg-[#0C0306]/60">
                  <div className="text-[#9E8F81] text-[10px]">Per-Frame Range</div>
                  <div className="text-[#F7F1EA] font-bold">
                    {result.pipeline_timings.per_frame_min_ms.toFixed(1)} - {result.pipeline_timings.per_frame_max_ms.toFixed(1)} ms
                  </div>
                </div>
              </div>
              {result.pipeline_timings.per_stage_avg_ms && Object.keys(result.pipeline_timings.per_stage_avg_ms).length > 0 && (
                <div className="mt-2 space-y-1">
                  {Object.entries(result.pipeline_timings.per_stage_avg_ms)
                    .sort(([, a], [, b]) => (b as number) - (a as number))
                    .map(([stage, ms]) => (
                      <div key={stage} className="flex items-center gap-2">
                        <span className="text-[10px] text-[#9E8F81] w-16 text-right">{stage}</span>
                        <div className="flex-1 h-1.5 bg-[#0C0306] rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-[#670626] to-[#A31243]"
                            style={{
                              width: `${Math.min(100, ((ms as number) / result.pipeline_timings.per_frame_avg_ms) * 100)}%`,
                            }}
                          />
                        </div>
                        <span className="text-[10px] text-[#D8CCC0] w-14 text-right">
                          {(ms as number).toFixed(1)} ms
                        </span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}

          {/* Pipeline Stats Footer */}
          <div className="glass-card p-4 rounded-xl border border-[#670626]/40 flex flex-wrap items-center justify-center gap-6 text-xs font-mono text-[#D8CCC0]">
            {result.frames_enhanced > 0 && (
              <span className="flex items-center gap-1.5">
                <Zap size={12} className="text-[#F59E0B]" />
                {result.frames_enhanced} frames CLAHE enhanced
              </span>
            )}
            {result.vlm_verifications > 0 && (
              <span className="flex items-center gap-1.5">
                <Eye size={12} className="text-[#A31243]" />
                {result.vlm_verifications} VLM verifications
              </span>
            )}
            <span className="flex items-center gap-1.5">
              <Target size={12} className="text-[#10B981]" />
              {result.total_unique_tracks} objects tracked
            </span>
            <span className="flex items-center gap-1.5">
              <Activity size={12} className="text-[#F7F1EA]" />
              {result.processing_time_s}s total inference
            </span>
            {result.has_audio_track && (
              <span className="flex items-center gap-1.5">
                <Volume2 size={12} className={result.audio_events?.length > 0 ? "text-[#EF4444]" : "text-[#10B981]"} />
                {result.audio_events?.length > 0
                  ? `${result.audio_events.length} siren event(s)`
                  : 'Audio analyzed, no sirens'}
              </span>
            )}
          </div>
        </main>
      )}

      {/* --- FULL-SIZE IMAGE INSPECTOR LIGHTBOX MODAL --- */}
      {isImageModalOpen && result && preview && (
        <div className="fixed inset-0 z-50 bg-black/95 backdrop-blur-2xl flex flex-col justify-between p-6 animate-fade-in-up">
          {/* Modal Header */}
          <div className="flex items-center justify-between border-b border-[#670626]/60 pb-4 max-w-7xl mx-auto w-full">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full overflow-hidden border border-[#F7F1EA]/30">
                <Image src="/logo.jpg" alt="Logo" width={32} height={32} />
              </div>
              <h3 className="font-heading text-base font-bold text-[#F7F1EA]">
                High-Resolution Perception Inspector
              </h3>
            </div>

            {/* Controls */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 bg-[#1C0612] px-3 py-1.5 rounded-lg border border-[#670626]/60">
                <button
                  onClick={() => setImageZoom((z) => Math.max(0.5, z - 0.25))}
                  className="text-[#D8CCC0] hover:text-[#F7F1EA]"
                  title="Zoom Out"
                >
                  <ZoomOut size={16} />
                </button>
                <span className="font-mono text-xs text-[#F7F1EA] w-12 text-center">
                  {(imageZoom * 100).toFixed(0)}%
                </span>
                <button
                  onClick={() => setImageZoom((z) => Math.min(3, z + 0.25))}
                  className="text-[#D8CCC0] hover:text-[#F7F1EA]"
                  title="Zoom In"
                >
                  <ZoomIn size={16} />
                </button>
                <button
                  onClick={() => setImageZoom(1)}
                  className="text-xs font-mono text-[#A31243] ml-1"
                >
                  Reset
                </button>
              </div>

              <button
                onClick={() => setIsImageModalOpen(false)}
                className="btn-cherry p-2 rounded-lg"
                title="Close Inspector"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {/* Modal Centered Canvas Viewport */}
          <div className="flex-1 flex items-center justify-center overflow-auto p-4">
            <div
              className="relative transition-transform duration-200 rounded-lg overflow-hidden shadow-2xl"
              style={{
                transform: `scale(${imageZoom})`,
                aspectRatio: `${result.frame_width} / ${result.frame_height}`,
                width: `min(90vw, calc(80vh * (${result.frame_width} / ${result.frame_height})))`,
                maxHeight: "80vh",
              }}
            >
              <img
                src={preview}
                alt="Target Frame Full Size"
                className="w-full h-full object-fill block"
              />
              <canvas
                ref={modalCanvasRef}
                className="absolute inset-0 w-full h-full block"
                width={result.frame_width}
                height={result.frame_height}
              />
            </div>
          </div>

          {/* Modal Footer Info */}
          <div className="max-w-[1560px] w-full mx-auto px-4 sm:px-6 lg:px-8 pt-3 border-t border-[#670626]/60 flex items-center justify-between text-xs font-mono text-[#D8CCC0]">
            <span>
              RESOLUTION: {result.frame_width} x {result.frame_height}
            </span>
            <span>TARGETS VISIBLE: {activeObjects.length}</span>
          </div>
        </div>
      )}

      {/* --- Symmetrical Luxury Footer --- */}
      <footer className="border-t border-[#670626]/40 bg-[#0C0306] py-6 px-4 sm:px-6 lg:px-8 text-center">
        <div className="max-w-[1560px] w-full mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 font-mono text-xs text-[#D8CCC0]">

          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-full overflow-hidden border border-[#F7F1EA]/20">
              <Image src="/logo.jpg" alt="Logo" width={20} height={20} />
            </div>
            <span className="text-[#F7F1EA] font-semibold">ATMANIRBHAR AI v3.0</span>
            <span>India-First Autonomous Driving Multi-Agents AI Intelligence</span>
          </div>
          <div>Warm Ivory and Cherry Design System</div>
        </div>
      </footer>
    </div>
  );
}
