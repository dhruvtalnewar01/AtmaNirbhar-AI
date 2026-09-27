/**
 * SARTHI Vision - Canvas Overlay Renderer
 * Draws detection bounding boxes, label cards, TTC countdown,
 * collision trajectory lines, and status indicators
 * in the Warm Ivory & Cherry design palette.
 */

import type { DetectedObject, FrameResult, CollisionWarning } from "./types";
import { RISK_COLORS, RISK_BORDER_COLORS } from "./types";

const LABEL_FONT = "500 11px 'JetBrains Mono', monospace";
const LABEL_FONT_BOLD = "bold 11px 'JetBrains Mono', monospace";
const TTC_FONT = "bold 13px 'JetBrains Mono', monospace";
const BOX_LINE_WIDTH = 2;
const LABEL_PADDING = 6;
const LABEL_GAP = 3;
const CORNER_RADIUS = 4;

/**
 * Render all detections for a single frame onto a canvas.
 */
export function renderFrame(
  ctx: CanvasRenderingContext2D,
  frame: FrameResult,
  scaleX: number,
  scaleY: number,
  confidenceThreshold: number = 0.35,
  selectedTrackId: number | null = null
): void {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  const isAnimalOrObstacle = (obj: DetectedObject) =>
    ["cow", "dog", "horse", "sheep", "goat", "cat", "elephant", "bear", "road-obstacle"].includes(
      obj.class_raw?.toLowerCase() || ""
    ) ||
    obj.class_name?.toLowerCase().includes("cow") ||
    obj.class_name?.toLowerCase().includes("obstacle") ||
    obj.risk_level === "High";

  const filteredObjects = frame.objects.filter((obj) => {
    const effConf = isAnimalOrObstacle(obj)
      ? Math.max(0.08, confidenceThreshold * 0.45)
      : confidenceThreshold;
    return obj.confidence >= effConf;
  });


  // Draw collision trajectory lines first (behind boxes)
  if (frame.collision_warnings && frame.collision_warnings.length > 0) {
    drawCollisionWarnings(ctx, frame.collision_warnings, filteredObjects, scaleX, scaleY);
  }

  // Draw boxes (back to front, so closer boxes appear on top)
  const sorted = [...filteredObjects].sort(
    (a, b) => (b.bbox.y2 - b.bbox.y1) - (a.bbox.y2 - a.bbox.y1)
  );

  for (const obj of sorted) {
    const isSelected = selectedTrackId === obj.track_id;
    const isDimmed = selectedTrackId !== null && !isSelected;
    drawDetection(ctx, obj, scaleX, scaleY, isSelected, isDimmed);
  }
}

/**
 * Draw collision warning lines between converging objects.
 */
function drawCollisionWarnings(
  ctx: CanvasRenderingContext2D,
  warnings: CollisionWarning[],
  objects: DetectedObject[],
  scaleX: number,
  scaleY: number
): void {
  const objectMap = new Map<number, DetectedObject>();
  for (const obj of objects) {
    objectMap.set(obj.track_id, obj);
  }

  for (const warning of warnings) {
    const objA = objectMap.get(warning.track_a);
    const objB = objectMap.get(warning.track_b);
    if (!objA || !objB) continue;

    const cxA = ((objA.bbox.x1 + objA.bbox.x2) / 2) * scaleX;
    const cyA = ((objA.bbox.y1 + objA.bbox.y2) / 2) * scaleY;
    const cxB = ((objB.bbox.x1 + objB.bbox.x2) / 2) * scaleX;
    const cyB = ((objB.bbox.y1 + objB.bbox.y2) / 2) * scaleY;

    ctx.save();

    // Collision trajectory line
    const isImminent = warning.urgency === "imminent";
    const lineColor = isImminent ? "#EF4444" : "#F59E0B";

    ctx.strokeStyle = lineColor;
    ctx.lineWidth = isImminent ? 3 : 2;
    ctx.setLineDash(isImminent ? [] : [6, 4]);
    ctx.globalAlpha = isImminent ? 0.9 : 0.6;

    ctx.beginPath();
    ctx.moveTo(cxA, cyA);
    ctx.lineTo(cxB, cyB);
    ctx.stroke();

    // Midpoint TTC label
    const midX = (cxA + cxB) / 2;
    const midY = (cyA + cyB) / 2;

    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    const ttcText = `TTC ${warning.ttc_s.toFixed(1)}s`;
    ctx.font = TTC_FONT;
    const textWidth = ctx.measureText(ttcText).width;
    const labelW = textWidth + 12;
    const labelH = 20;

    ctx.fillStyle = isImminent ? "rgba(239, 68, 68, 0.85)" : "rgba(245, 158, 11, 0.85)";
    roundRect(ctx, midX - labelW / 2, midY - labelH / 2, labelW, labelH, 4);
    ctx.fill();

    ctx.fillStyle = "#FFFFFF";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(ttcText, midX, midY);
    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";

    // Pulsing danger zone for imminent collisions
    if (isImminent) {
      const pulsePhase = (Date.now() % 1000) / 1000;
      const pulseAlpha = 0.1 + Math.sin(pulsePhase * Math.PI * 2) * 0.08;

      ctx.fillStyle = `rgba(239, 68, 68, ${pulseAlpha})`;
      const radius = Math.max(
        Math.abs(cxB - cxA) / 2,
        Math.abs(cyB - cyA) / 2,
        40
      );
      ctx.beginPath();
      ctx.ellipse(midX, midY, radius, radius * 0.7, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }
}

/**
 * Draw a single detection: bounding box + tactical corner brackets + floating label card + TTC.
 */
function drawDetection(
  ctx: CanvasRenderingContext2D,
  obj: DetectedObject,
  scaleX: number,
  scaleY: number,
  isSelected: boolean,
  isDimmed: boolean
): void {
  const x1 = obj.bbox.x1 * scaleX;
  const y1 = obj.bbox.y1 * scaleY;
  const x2 = obj.bbox.x2 * scaleX;
  const y2 = obj.bbox.y2 * scaleY;
  const w = x2 - x1;
  const h = y2 - y1;

  if (w <= 0 || h <= 0) return;

  const color = RISK_COLORS[obj.risk_level] || "#F7F1EA";
  const borderColor = RISK_BORDER_COLORS[obj.risk_level] || "rgba(247, 241, 234, 0.9)";
  const alpha = isDimmed ? 0.25 : 1.0;

  ctx.save();
  ctx.globalAlpha = alpha;

  // --- Semi-transparent fill ---
  ctx.fillStyle = isSelected ? `${color}28` : `${color}0D`;
  roundRect(ctx, x1, y1, w, h, CORNER_RADIUS);
  ctx.fill();

  // --- Thin bounding box outline ---
  ctx.strokeStyle = borderColor;
  ctx.lineWidth = isSelected ? BOX_LINE_WIDTH * 1.5 : BOX_LINE_WIDTH;
  roundRect(ctx, x1, y1, w, h, CORNER_RADIUS);
  ctx.stroke();

  // --- Tactical corner accents ---
  const cornerLen = Math.min(16, w * 0.25, h * 0.25);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;

  // Top-left
  ctx.beginPath();
  ctx.moveTo(x1, y1 + cornerLen);
  ctx.lineTo(x1, y1);
  ctx.lineTo(x1 + cornerLen, y1);
  ctx.stroke();

  // Top-right
  ctx.beginPath();
  ctx.moveTo(x2 - cornerLen, y1);
  ctx.lineTo(x2, y1);
  ctx.lineTo(x2, y1 + cornerLen);
  ctx.stroke();

  // Bottom-left
  ctx.beginPath();
  ctx.moveTo(x1, y2 - cornerLen);
  ctx.lineTo(x1, y2);
  ctx.lineTo(x1 + cornerLen, y2);
  ctx.stroke();

  // Bottom-right
  ctx.beginPath();
  ctx.moveTo(x2 - cornerLen, y2);
  ctx.lineTo(x2, y2);
  ctx.lineTo(x2, y2 - cornerLen);
  ctx.stroke();

  // --- Floating Label Card ---
  drawLabelCard(ctx, obj, x1, y1, color);

  // --- TTC Countdown Badge (when TTC < 5s) ---
  if (obj.ttc_s !== null && obj.ttc_s !== undefined && obj.ttc_s < 5) {
    drawTTCBadge(ctx, obj.ttc_s, x2, y1, color);
  }

  // --- Motion vector indicator ---
  if (obj.status === "Moving" && (Math.abs(obj.rel_dx) > 0.4 || Math.abs(obj.rel_dy) > 0.4)) {
    const arrowX = (x1 + x2) / 2;
    const arrowY = (y1 + y2) / 2;
    const angle = Math.atan2(obj.rel_dy, obj.rel_dx);
    const arrowLen = 22;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(
      arrowX + Math.cos(angle) * arrowLen,
      arrowY + Math.sin(angle) * arrowLen
    );
    ctx.stroke();

    // Arrowhead
    const headLen = 6;
    ctx.beginPath();
    ctx.moveTo(
      arrowX + Math.cos(angle) * arrowLen,
      arrowY + Math.sin(angle) * arrowLen
    );
    ctx.lineTo(
      arrowX + Math.cos(angle - 0.4) * (arrowLen - headLen),
      arrowY + Math.sin(angle - 0.4) * (arrowLen - headLen)
    );
    ctx.moveTo(
      arrowX + Math.cos(angle) * arrowLen,
      arrowY + Math.sin(angle) * arrowLen
    );
    ctx.lineTo(
      arrowX + Math.cos(angle + 0.4) * (arrowLen - headLen),
      arrowY + Math.sin(angle + 0.4) * (arrowLen - headLen)
    );
    ctx.stroke();
  }

  ctx.restore();
}

/**
 * Draw TTC countdown badge at top-right of bounding box.
 */
function drawTTCBadge(
  ctx: CanvasRenderingContext2D,
  ttc: number,
  x: number,
  y: number,
  color: string
): void {
  const isImminent = ttc < 2;
  const text = `${ttc.toFixed(1)}s`;
  ctx.font = TTC_FONT;
  const tw = ctx.measureText(text).width;
  const badgeW = tw + 14;
  const badgeH = 22;
  const bx = x - badgeW - 2;
  const by = y - badgeH - 4;

  // Background
  ctx.fillStyle = isImminent ? "rgba(239, 68, 68, 0.92)" : "rgba(245, 158, 11, 0.88)";
  roundRect(ctx, bx, by, badgeW, badgeH, 4);
  ctx.fill();

  // Border
  ctx.strokeStyle = isImminent ? "#FF8888" : "#FFD700";
  ctx.lineWidth = 1;
  roundRect(ctx, bx, by, badgeW, badgeH, 4);
  ctx.stroke();

  // Text
  ctx.fillStyle = "#FFFFFF";
  ctx.font = TTC_FONT;
  ctx.fillText(text, bx + 7, by + 16);
}

/**
 * Draw the floating label card above a bounding box.
 */
function drawLabelCard(
  ctx: CanvasRenderingContext2D,
  obj: DetectedObject,
  boxX: number,
  boxY: number,
  color: string
): void {
  ctx.font = LABEL_FONT;

  const line1 = `${obj.class_name}`;
  const line2Parts: string[] = [obj.status];
  if (obj.distance_available && obj.distance_m !== null) {
    line2Parts.push(`~${obj.distance_m}m`);
  }
  if (obj.ttc_s !== null && obj.ttc_s !== undefined && obj.ttc_s < 10) {
    line2Parts.push(`TTC ${obj.ttc_s.toFixed(1)}s`);
  }
  line2Parts.push(obj.risk_level);
  const line2 = line2Parts.join(" | ");

  const line1Width = ctx.measureText(line1).width;
  const line2Width = ctx.measureText(line2).width;
  const cardWidth = Math.max(line1Width, line2Width) + LABEL_PADDING * 2 + 6;
  const lineHeight = 14;
  const cardHeight = lineHeight * 2 + LABEL_PADDING * 2;

  // Position card above the box, or inside if near top edge
  const cardX = Math.max(4, boxX);
  const cardY = boxY - cardHeight - LABEL_GAP > 4
    ? boxY - cardHeight - LABEL_GAP
    : boxY + 4;

  // Dark obsidian cherry background
  ctx.fillStyle = "rgba(16, 4, 9, 0.92)";
  roundRect(ctx, cardX, cardY, cardWidth, cardHeight, CORNER_RADIUS);
  ctx.fill();

  // Border with subtle cherry glow
  ctx.strokeStyle = `${color}88`;
  ctx.lineWidth = 1;
  roundRect(ctx, cardX, cardY, cardWidth, cardHeight, CORNER_RADIUS);
  ctx.stroke();

  // Left status bar
  ctx.fillStyle = color;
  ctx.fillRect(cardX, cardY + 3, 3, cardHeight - 6);

  // Line 1: Class Name in Warm Ivory
  ctx.font = LABEL_FONT_BOLD;
  ctx.fillStyle = "#F7F1EA";
  ctx.fillText(line1, cardX + LABEL_PADDING + 4, cardY + LABEL_PADDING + 10);

  // Line 2: Details in Muted Ivory
  ctx.font = LABEL_FONT;
  ctx.fillStyle = "#D8CCC0";
  ctx.fillText(line2, cardX + LABEL_PADDING + 4, cardY + LABEL_PADDING + 10 + lineHeight);
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

/**
 * Hit-test: find which detected object (if any) contains a click point.
 */
export function hitTestDetection(
  frame: FrameResult,
  clickX: number,
  clickY: number,
  scaleX: number,
  scaleY: number,
  confidenceThreshold: number = 0.35
): DetectedObject | null {
  const isAnimalOrObstacle = (obj: DetectedObject) =>
    ["cow", "dog", "horse", "sheep", "goat", "cat", "elephant", "bear", "road-obstacle"].includes(
      obj.class_raw?.toLowerCase() || ""
    ) ||
    obj.class_name?.toLowerCase().includes("cow") ||
    obj.class_name?.toLowerCase().includes("obstacle") ||
    obj.risk_level === "High";

  const filtered = frame.objects.filter((obj) => {
    const effConf = isAnimalOrObstacle(obj)
      ? Math.max(0.08, confidenceThreshold * 0.45)
      : confidenceThreshold;
    return obj.confidence >= effConf;
  });


  for (let i = filtered.length - 1; i >= 0; i--) {
    const obj = filtered[i];
    const x1 = obj.bbox.x1 * scaleX;
    const y1 = obj.bbox.y1 * scaleY;
    const x2 = obj.bbox.x2 * scaleX;
    const y2 = obj.bbox.y2 * scaleY;

    if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
      return obj;
    }
  }

  return null;
}
