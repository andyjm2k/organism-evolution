/** Top-down body plan. Parts stick out in the plane so each organ reads from above. */

import { PART_VISIBLE, type PartLevels } from "../sim/morphology";

export interface CreaturePose {
  color: [number, number, number];
  parts: PartLevels;
  jaw: number;
  claw: number;
  horn: number;
  flagella: number;
  selected: boolean;
}

export type MeshKind = "sphere" | "cone" | "box";

export interface CreaturePart {
  name: string;
  mesh: MeshKind;
  x: number;
  y: number;
  z: number;
  sx: number;
  sy: number;
  sz: number;
  yaw: number;
  color: [number, number, number];
  emissive: number;
  roughness: number;
  camo: number;
  wiggle: number;
  halo: boolean;
}

type RGB = [number, number, number];

const BONE: RGB = [0.94, 0.9, 0.8];
const PUPIL: RGB = [0.04, 0.03, 0.04];
const EYE: RGB = [0.93, 0.96, 0.94];
const CLAW: RGB = [0.28, 0.13, 0.1];
const SPINE: RGB = [0.98, 0.58, 0.16];
const VENOM: RGB = [0.72, 0.18, 0.9];
const GLOW: RGB = [1, 0.84, 0.32];
const PHOTO: RGB = [0.62, 1, 0.18];
const SIPHON: RGB = [0.9, 0.52, 0.38];
const CYAN: RGB = [0.3, 0.95, 1];
const PLATE: RGB = [0.16, 0.14, 0.12];
const INK: RGB = [0.04, 0.04, 0.05];
const RING: RGB = [0.95, 0.92, 0.72];

/** Screen size of the torso. Physics radius stays smaller than this drawing. */
export function visualRadius(radius: number): number {
  return Math.max(22, Math.min(40, 18 + radius));
}

/** Local parts before heading is applied. +X is forward and +Z is up. */
export function creatureParts(pose: CreaturePose): CreaturePart[] {
  const gene = pose.parts;
  const body = unitColor(pose.color);
  const camo = gene.camo;
  const fin = Math.min(1, gene.fin + Math.min(0.12, pose.flagella * 0.03));
  const jaw = Math.max(gene.jaw, pose.jaw >= 0.25 ? Math.min(0.38, pose.jaw / 5) : 0);
  const clawLevel = Math.max(gene.claw, pose.claw >= 0.25 ? Math.min(0.45, pose.claw / 4) : 0);
  const hornLevel = Math.max(gene.horn, pose.horn >= 0.25 ? Math.min(0.34, pose.horn / 10) : 0);
  const list: CreaturePart[] = [];
  const skin = pose.selected ? mix(body, [1, 0.82, 0.28], 0.28) : body;
  list.push(part("torso", "sphere", 0, 0, 0.42, 0.9, 0.58, 0.36, 0, skin, { roughness: 0.28, camo, emissive: pose.selected ? 0.22 : 0 }));
  list.push(part("stripe", "sphere", 0.08, 0, 0.64, 0.46, 0.09, 0.1, 0, mix(body, [1, 1, 1], 0.5), { roughness: 0.2 }));
  list.push(part("head", "sphere", 0.86, 0, 0.5, 0.4, 0.34, 0.28, 0, skin, { roughness: 0.3, camo }));
  const eyeScale = 0.13 + gene.eye * 0.16;
  for (const side of [-1, 1]) {
    list.push(part("eye", "sphere", 1.2, side * 0.2, 0.66, eyeScale, eyeScale, eyeScale * 0.86, 0, EYE, { roughness: 0.08 }));
    list.push(part("pupil", "sphere", 1.3, side * 0.2, 0.72, eyeScale * 0.42, eyeScale * 0.42, eyeScale * 0.36, 0, PUPIL, { roughness: 0.2 }));
  }
  list.push(
    part("tail", "sphere", -1.28, 0, 0.36, 0.42 + fin * 0.62, 0.34 + fin * 0.48, 0.08, 0, mix(body, [1, 1, 1], 0.22), {
      roughness: 0.38,
      wiggle: 0.55,
    }),
  );
  if (fin >= 0.3) {
    for (const side of [-1, 1]) {
      list.push(part("fin", "sphere", 0.12, side * 0.9, 0.28, 0.24, 0.32 + fin * 0.22, 0.06, 0, mix(body, [1, 1, 1], 0.15), { wiggle: 0.28 }));
    }
  }
  if (jaw >= 0.12) {
    list.push(part("mouth", "sphere", 1.08, 0, 0.4, 0.12, 0.1 + jaw * 0.14, 0.06, 0, INK, { roughness: 0.6 }));
    for (const side of [-1, 1]) {
      list.push(part("tooth", "cone", 1.18, side * 0.08, 0.44, 0.1 + jaw * 0.32, 0.055, 0.055, side * 0.22, BONE, { roughness: 0.25 }));
    }
  }
  if (gene.antenna >= PART_VISIBLE) {
    for (const side of [-1, 1]) {
      list.push(
        part("antenna", "cone", 0.78, side * 0.3, 0.8, 0.22 + gene.antenna * 0.9, 0.055, 0.055, side * 0.95, mix(body, INK, 0.65), {
          wiggle: 0.3,
          roughness: 0.4,
        }),
      );
    }
  }
  if (gene.lateral >= PART_VISIBLE) {
    for (const side of [-1, 1]) {
      for (let index = 0; index < 4; index += 1) {
        list.push(part("lateral", "sphere", -0.35 + index * 0.28, side * 0.64, 0.5, 0.07, 0.07, 0.07, 0, CYAN, { emissive: 0.9, roughness: 0.12 }));
      }
    }
  }
  if (clawLevel >= PART_VISIBLE) {
    for (const side of [-1, 1]) {
      list.push(part("claw", "cone", 0.42, side * 1.05, 0.3, 0.24 + clawLevel * 0.58, 0.12, 0.09, side * 1.15, CLAW, { roughness: 0.35 }));
      list.push(part("claw", "cone", 0.62, side * 1.22, 0.34, 0.16 + clawLevel * 0.36, 0.08, 0.07, side * 0.5, CLAW, { roughness: 0.35 }));
    }
  }
  if (gene.horn >= 0.55) {
    for (const side of [-1, 1]) {
      list.push(part("horn", "cone", 0.92, side * 0.18, 0.76, 0.28 + gene.horn * 0.58, 0.09, 0.09, side * 0.2, BONE, { roughness: 0.22 }));
    }
  } else if (hornLevel >= PART_VISIBLE) {
    list.push(part("horn", "cone", 0.98, 0, 0.74, 0.18 + hornLevel * 0.7, 0.09, 0.09, 0, BONE, { roughness: 0.22 }));
  }
  if (gene.spine >= PART_VISIBLE) {
    const angles = [2.35, 2.75, 3.14, 3.55, 3.95, 4.4];
    for (const yaw of angles) {
      list.push(
        part("spine", "cone", Math.cos(yaw) * 0.72, Math.sin(yaw) * 0.5, 0.5, 0.16 + gene.spine * 0.52, 0.08, 0.08, yaw, SPINE, { roughness: 0.3 }),
      );
    }
  }
  if (gene.venom >= PART_VISIBLE) {
    const sac = 0.12 + gene.venom * 0.16;
    list.push(part("venom", "sphere", -0.22, 0.66, 0.55, sac, sac, sac, 0, VENOM, { emissive: 0.45, roughness: 0.18 }));
    list.push(part("venom", "cone", 1.28, 0.1, 0.4, 0.14 + gene.venom * 0.22, 0.05, 0.05, 0.15, VENOM, { emissive: 0.35 }));
  }
  if (gene.shell >= PART_VISIBLE) {
    list.push(
      part("shell", "sphere", -0.02, 0, 0.7, 0.84 + gene.shell * 0.16, 0.7 + gene.shell * 0.08, 0.2 + gene.shell * 0.1, 0, mix(body, BONE, 0.8), {
        roughness: 0.7,
      }),
    );
  }
  if (gene.plate >= PART_VISIBLE) {
    for (const x of [-0.42, 0.05, 0.48]) {
      list.push(part("plate", "box", x, 0, 0.58, 0.18, 0.92 + gene.plate * 0.2, 0.08, 0, PLATE, { roughness: 0.75 }));
    }
  }
  if (gene.camo >= PART_VISIBLE) {
    for (const side of [-1, 1]) {
      list.push(part("eyespot", "sphere", -0.35, side * 0.95, 0.46, 0.18, 0.18, 0.08, 0, RING, { roughness: 0.4 }));
      list.push(part("eyespot", "sphere", -0.28, side * 0.95, 0.54, 0.08, 0.08, 0.05, 0, INK, { roughness: 0.3 }));
    }
  }
  if (gene.glow >= PART_VISIBLE) {
    const spots: [number, number, number][] = [
      [-0.4, 0.38, 0.58],
      [0.15, -0.42, 0.6],
      [0.45, 0.22, 0.64],
    ];
    const scale = 0.09 + gene.glow * 0.08;
    for (const [x, y, z] of spots) {
      list.push(part("glow", "sphere", x, y, z, scale, scale, scale, 0, GLOW, { emissive: 1, roughness: 0.08, halo: true }));
    }
  }
  if (gene.photo >= PART_VISIBLE) {
    const spots: [number, number][] = [
      [-0.28, 0.16],
      [0.02, -0.18],
      [0.28, 0.14],
      [0.12, 0],
    ];
    const scale = 0.09 + gene.photo * 0.07;
    for (const [x, y] of spots) {
      list.push(part("photo", "sphere", x, y, 0.72, scale, scale, scale * 0.7, 0, PHOTO, { emissive: 0.4, roughness: 0.2 }));
    }
  }
  if (gene.siphon >= PART_VISIBLE) {
    list.push(part("siphon", "cone", 1.4, 0, 0.28, 0.26 + gene.siphon * 0.5, 0.11, 0.09, 0.04, SIPHON, { roughness: 0.32 }));
  }
  return list;
}

function part(
  name: string,
  mesh: MeshKind,
  x: number,
  y: number,
  z: number,
  sx: number,
  sy: number,
  sz: number,
  yaw: number,
  color: RGB,
  material: { emissive?: number; roughness?: number; camo?: number; wiggle?: number; halo?: boolean } = {},
): CreaturePart {
  return {
    name,
    mesh,
    x,
    y,
    z,
    sx,
    sy,
    sz,
    yaw,
    color,
    emissive: material.emissive ?? 0,
    roughness: material.roughness ?? 0.45,
    camo: material.camo ?? 0,
    wiggle: material.wiggle ?? 0,
    halo: material.halo ?? false,
  };
}

function unitColor(color: [number, number, number]): RGB {
  return [color[0] / 255, color[1] / 255, color[2] / 255];
}

function mix(a: RGB, b: RGB, amount: number): RGB {
  return [a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount, a[2] + (b[2] - a[2]) * amount];
}
