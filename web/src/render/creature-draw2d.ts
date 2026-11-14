/** Canvas fallback that paints the same part layout as the lit meshes. */

import { creatureParts, type CreaturePart, type CreaturePose } from "./creature-layout";

export function paintCreature2d(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  heading: number,
  pose: CreaturePose,
): void {
  const ordered = creatureParts(pose).slice().sort((a, b) => a.z - b.z || a.x - b.x);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(heading);
  for (const piece of ordered) {
    paintPart(ctx, piece, radius);
  }
  ctx.restore();
}

function paintPart(ctx: CanvasRenderingContext2D, piece: CreaturePart, radius: number): void {
  ctx.save();
  ctx.translate(piece.x * radius, piece.y * radius);
  ctx.rotate(piece.yaw);
  const fill = `rgb(${Math.round(piece.color[0] * 255)},${Math.round(piece.color[1] * 255)},${Math.round(piece.color[2] * 255)})`;
  if (piece.halo) {
    ctx.globalAlpha = 0.35;
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.ellipse(0, 0, piece.sx * radius * 2.4, piece.sy * radius * 2.4, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  ctx.fillStyle = fill;
  const sx = Math.max(0.6, piece.sx * radius);
  const sy = Math.max(0.6, piece.sy * radius);
  ctx.beginPath();
  if (piece.mesh === "cone") {
    ctx.moveTo(sx, 0);
    ctx.lineTo(-sx, sy);
    ctx.lineTo(-sx, -sy);
    ctx.closePath();
    ctx.fill();
  } else if (piece.mesh === "fin") {
    ctx.moveTo(sx * 0.2, 0);
    ctx.lineTo(-sx * 0.2, sy);
    ctx.lineTo(-sx, sy * 0.35);
    ctx.lineTo(-sx, -sy * 0.35);
    ctx.lineTo(-sx * 0.2, -sy);
    ctx.closePath();
    ctx.fill();
  } else if (piece.mesh === "claw") {
    ctx.moveTo(-sx, 0);
    ctx.quadraticCurveTo(0, sy * 0.2, sx * 0.5, sy * 0.6);
    ctx.quadraticCurveTo(sx, sy, sx * 0.55, sy * 1.05);
    ctx.quadraticCurveTo(sx * 0.2, sy * 0.55, -sx, sy * 0.15);
    ctx.closePath();
    ctx.fill();
  } else if (piece.mesh === "plate") {
    ctx.moveTo(-sx, -sy * 0.85);
    ctx.lineTo(-sx * 0.2, -sy);
    ctx.lineTo(sx, -sy * 0.7);
    ctx.lineTo(sx, sy * 0.7);
    ctx.lineTo(-sx * 0.2, sy);
    ctx.lineTo(-sx, sy * 0.85);
    ctx.closePath();
    ctx.fill();
  } else if (piece.mesh === "ring") {
    ctx.ellipse(0, 0, sx, sy, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = "destination-out";
    ctx.beginPath();
    ctx.ellipse(0, 0, sx * 0.55, sy * 0.55, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";
  } else if (piece.mesh === "dome") {
    ctx.ellipse(0, 0, sx, sy, 0, Math.PI, Math.PI * 2);
    ctx.closePath();
    ctx.fill();
  } else {
    ctx.ellipse(0, 0, sx, piece.mesh === "capsule" || piece.mesh === "tube" ? sy * 0.72 : sy, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}
