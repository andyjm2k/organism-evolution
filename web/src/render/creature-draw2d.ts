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
  ctx.beginPath();
  if (piece.mesh === "cone") {
    const length = piece.sx * radius;
    const half = Math.max(1, piece.sy * radius);
    ctx.moveTo(length, 0);
    ctx.lineTo(-length, half);
    ctx.lineTo(-length, -half);
    ctx.closePath();
  } else if (piece.mesh === "box") {
    ctx.rect(-piece.sx * radius, -piece.sy * radius, piece.sx * radius * 2, piece.sy * radius * 2);
  } else {
    ctx.ellipse(0, 0, Math.max(0.6, piece.sx * radius), Math.max(0.6, piece.sy * radius), 0, 0, Math.PI * 2);
  }
  ctx.fill();
  ctx.restore();
}
