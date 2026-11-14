/** Top-down body plan. Parts stick out so each organ keeps a clear silhouette. */

import { PART_VISIBLE, type PartLevels } from "../sim/morphology";
import type { MeshKind } from "./creature-mesh";

export interface CreaturePose {
  color: [number, number, number];
  parts: PartLevels;
  jaw: number;
  claw: number;
  horn: number;
  flagella: number;
  selected: boolean;
}

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
  pitch: number;
  color: [number, number, number];
  emissive: number;
  roughness: number;
  camo: number;
  wiggle: number;
  mark: number;
  halo: boolean;
}

type RGB = [number, number, number];

/** Surface marks read by the creature shader. */
export const MARK = {
  smooth: 0,
  skin: 1,
  scales: 2,
  ridges: 3,
  enamel: 4,
  glow: 5,
  leaf: 6,
  shell: 7,
  plate: 8,
} as const;

const BONE: RGB = [0.96, 0.92, 0.82];
const PUPIL: RGB = [0.03, 0.02, 0.03];
const SCLERA: RGB = [0.95, 0.97, 0.96];
const IRIS: RGB = [0.18, 0.42, 0.55];
const CLAW: RGB = [0.22, 0.1, 0.08];
const SPINE: RGB = [0.98, 0.52, 0.12];
const SPINE_ROOT: RGB = [0.35, 0.16, 0.08];
const VENOM: RGB = [0.78, 0.16, 0.95];
const GLOW: RGB = [1, 0.86, 0.28];
const PHOTO: RGB = [0.45, 0.95, 0.18];
const SIPHON: RGB = [0.95, 0.55, 0.36];
const CYAN: RGB = [0.25, 0.95, 1];
const PLATE: RGB = [0.18, 0.16, 0.14];
const INK: RGB = [0.05, 0.04, 0.05];
const RING: RGB = [0.98, 0.94, 0.72];
const GUM: RGB = [0.55, 0.18, 0.22];
const ANTENNA: RGB = [0.12, 0.1, 0.12];

/** Screen size of the torso. Physics radius stays smaller than this drawing. */
export function visualRadius(radius: number): number {
  return Math.max(24, Math.min(44, 20 + radius));
}

/** Local parts before heading is applied. +X is forward and +Z is up. */
export function creatureParts(pose: CreaturePose): CreaturePart[] {
  const gene = pose.parts;
  const body = unitColor(pose.color);
  const camo = gene.camo;
  const fin = Math.min(1, gene.fin + Math.min(0.12, pose.flagella * 0.03));
  const jaw = Math.max(gene.jaw, pose.jaw >= 0.25 ? Math.min(0.42, pose.jaw / 5) : 0);
  const clawLevel = Math.max(gene.claw, pose.claw >= 0.25 ? Math.min(0.5, pose.claw / 4) : 0);
  const hornLevel = Math.max(gene.horn, pose.horn >= 0.25 ? Math.min(0.4, pose.horn / 10) : 0);
  const list: CreaturePart[] = [];
  const skin = pose.selected ? mix(body, [1, 0.82, 0.28], 0.28) : body;
  const belly = mix(body, [1, 1, 1], 0.28);

  list.push(part("torso", "sphere", 0.02, 0, 0.4, 0.78, 0.62, 0.42, 0, 0, skin, { roughness: 0.4, camo, mark: MARK.skin, emissive: pose.selected ? 0.18 : 0 }));
  list.push(part("belly", "sphere", 0.0, 0, 0.22, 0.58, 0.46, 0.22, 0, 0, belly, { roughness: 0.32, mark: MARK.smooth }));
  list.push(part("rump", "sphere", -0.55, 0, 0.36, 0.42, 0.4, 0.3, 0, 0, skin, { roughness: 0.4, camo, mark: MARK.skin }));
  list.push(part("head", "sphere", 0.82, 0, 0.48, 0.48, 0.4, 0.34, 0, 0, skin, { roughness: 0.34, camo, mark: MARK.skin }));
  list.push(part("snout", "capsule", 1.2, 0, 0.4, 0.28, 0.2, 0.16, 0, 0.1, mix(skin, INK, 0.1), { roughness: 0.42, mark: MARK.smooth }));

  const eyeScale = 0.16 + gene.eye * 0.2;
  for (const side of [-1, 1]) {
    list.push(part("socket", "sphere", 1.05, side * 0.28, 0.62, eyeScale * 1.2, eyeScale * 1.1, eyeScale * 0.75, 0, 0, mix(skin, INK, 0.3), { roughness: 0.55 }));
    list.push(part("eye", "sphere", 1.16, side * 0.28, 0.68, eyeScale, eyeScale, eyeScale * 0.92, 0, 0, SCLERA, { roughness: 0.08, mark: MARK.smooth }));
    list.push(part("iris", "ring", 1.28, side * 0.28, 0.74, eyeScale * 0.78, eyeScale * 0.78, eyeScale * 0.22, 0, 1.25, IRIS, { roughness: 0.15, mark: MARK.enamel }));
    list.push(part("pupil", "sphere", 1.32, side * 0.28, 0.76, eyeScale * 0.36, eyeScale * 0.36, eyeScale * 0.3, 0, 0, PUPIL, { roughness: 0.2 }));
  }

  const tailLen = 0.55 + fin * 0.85;
  list.push(part("tailbase", "capsule", -0.95, 0, 0.36, 0.34, 0.28, 0.2, 0, 0, skin, { roughness: 0.4, mark: MARK.skin }));
  list.push(
    part("tail", "fin", -1.15 - fin * 0.35, 0, 0.38, tailLen, 0.42 + fin * 0.55, 0.08, 0, 0, mix(body, [1, 1, 1], 0.18), {
      roughness: 0.32,
      wiggle: 0.45,
      mark: MARK.scales,
    }),
  );
  if (fin >= 0.28) {
    for (const side of [-1, 1]) {
      list.push(
        part("fin", "fin", 0.05, side * 0.95, 0.3, 0.34 + fin * 0.2, 0.42 + fin * 0.28, 0.06, side * 1.35, 0.35, mix(body, [1, 1, 1], 0.12), {
          wiggle: 0.22,
          mark: MARK.scales,
        }),
      );
    }
  }

  if (jaw >= 0.12) {
    list.push(part("mouth", "sphere", 1.22, 0, 0.34, 0.14, 0.12 + jaw * 0.16, 0.08, 0, 0, INK, { roughness: 0.7 }));
    list.push(part("gum", "capsule", 1.3, 0, 0.36, 0.1 + jaw * 0.12, 0.14 + jaw * 0.14, 0.06, 0, 0.4, GUM, { roughness: 0.55 }));
    for (const side of [-1, 1]) {
      list.push(part("tooth", "cone", 1.38, side * (0.06 + jaw * 0.05), 0.4, 0.14 + jaw * 0.34, 0.05, 0.05, side * 0.18, 0.35, BONE, { roughness: 0.18, mark: MARK.enamel }));
      list.push(part("tooth", "cone", 1.32, side * (0.12 + jaw * 0.08), 0.34, 0.1 + jaw * 0.22, 0.04, 0.04, side * 0.55, 0.55, BONE, { roughness: 0.18, mark: MARK.enamel }));
    }
  }

  if (gene.antenna >= PART_VISIBLE) {
    for (const side of [-1, 1]) {
      list.push(
        part("antenna", "cone", 0.7, side * 0.32, 0.82, 0.34 + gene.antenna * 0.75, 0.06, 0.06, side * 1.05, 0.65, ANTENNA, {
          wiggle: 0.28,
          roughness: 0.45,
          mark: MARK.smooth,
        }),
      );
      list.push(
        part("antenna", "sphere", 0.85 + gene.antenna * 0.55, side * (0.55 + gene.antenna * 0.35), 1.05, 0.1 + gene.antenna * 0.04, 0.1, 0.1, 0, 0, CYAN, {
          emissive: 0.7,
          roughness: 0.18,
          mark: MARK.glow,
        }),
      );
    }
  }

  if (gene.lateral >= PART_VISIBLE) {
    for (const side of [-1, 1]) {
      for (let index = 0; index < 4; index += 1) {
        const x = -0.48 + index * 0.3;
        list.push(part("lateral", "sphere", x, side * 0.7, 0.5, 0.07, 0.07, 0.07, 0, 0, CYAN, { emissive: 1, roughness: 0.12, mark: MARK.glow }));
        if (index < 3) {
          list.push(part("lateral", "capsule", x + 0.15, side * 0.7, 0.5, 0.12, 0.035, 0.035, side * 1.57, 0, CYAN, { emissive: 0.55, roughness: 0.2, mark: MARK.glow }));
        }
      }
    }
  }

  if (clawLevel >= PART_VISIBLE) {
    for (const side of [-1, 1]) {
      list.push(part("claw", "capsule", 0.22, side * 0.85, 0.32, 0.26, 0.12, 0.1, side * 1.15, 0.15, mix(skin, CLAW, 0.35), { roughness: 0.4, mark: MARK.smooth }));
      list.push(part("claw", "claw", 0.58, side * 1.2, 0.34, 0.36 + clawLevel * 0.5, 0.2 + clawLevel * 0.14, 0.12, side * 0.2, 0.1, CLAW, { roughness: 0.25, mark: MARK.enamel }));
      list.push(part("claw", "claw", 0.78, side * 1.32, 0.38, 0.22 + clawLevel * 0.32, 0.12, 0.09, side * -0.7, -0.05, CLAW, { roughness: 0.25, mark: MARK.enamel }));
    }
  }

  if (gene.horn >= 0.55) {
    for (const side of [-1, 1]) {
      list.push(part("horn", "cone", 0.95, side * 0.2, 0.86, 0.34 + gene.horn * 0.55, 0.1, 0.1, side * 0.18, 1.05, BONE, { roughness: 0.2, mark: MARK.ridges }));
      list.push(part("horn", "ring", 0.78, side * 0.16, 0.72, 0.12, 0.12, 0.05, 0, 1.1, mix(BONE, INK, 0.25), { roughness: 0.4 }));
    }
  } else if (hornLevel >= PART_VISIBLE) {
    list.push(part("horn", "cone", 1.02, 0, 0.86, 0.24 + hornLevel * 0.7, 0.1, 0.1, 0, 1.15, BONE, { roughness: 0.2, mark: MARK.ridges }));
    list.push(part("horn", "ring", 0.86, 0, 0.72, 0.13, 0.13, 0.05, 0, 1.1, mix(BONE, INK, 0.2), { roughness: 0.4 }));
  }

  if (gene.spine >= PART_VISIBLE) {
    const angles = [2.2, 2.55, 2.95, 3.35, 3.75, 4.15, 4.55];
    for (const yaw of angles) {
      list.push(part("spine", "cone", Math.cos(yaw) * 0.55, Math.sin(yaw) * 0.42, 0.62, 0.08, 0.08, 0.08, yaw, 0.2, SPINE_ROOT, { roughness: 0.45 }));
      list.push(
        part("spine", "cone", Math.cos(yaw) * 0.78, Math.sin(yaw) * 0.58, 0.7, 0.22 + gene.spine * 0.55, 0.07, 0.07, yaw, 0.55, SPINE, {
          roughness: 0.25,
          mark: MARK.ridges,
        }),
      );
    }
  }

  if (gene.venom >= PART_VISIBLE) {
    const sac = 0.14 + gene.venom * 0.16;
    list.push(part("venom", "sphere", -0.18, 0.7, 0.55, sac, sac, sac, 0, 0, VENOM, { emissive: 0.55, roughness: 0.16, mark: MARK.glow, halo: true }));
    list.push(part("venom", "tube", 1.42, 0.08, 0.34, 0.16 + gene.venom * 0.18, 0.04, 0.04, 0.12, 0.25, VENOM, { emissive: 0.4, mark: MARK.glow }));
    list.push(part("venom", "cone", 1.58, 0.1, 0.32, 0.1 + gene.venom * 0.12, 0.035, 0.035, 0.12, 0.2, mix(VENOM, INK, 0.3), { emissive: 0.25 }));
  }

  if (gene.shell >= PART_VISIBLE) {
    list.push(
      part("shell", "dome", -0.05, 0, 0.62, 0.78 + gene.shell * 0.18, 0.68 + gene.shell * 0.12, 0.42 + gene.shell * 0.18, 0, 0, mix(body, BONE, 0.72), {
        roughness: 0.62,
        mark: MARK.shell,
      }),
    );
    list.push(part("shell", "ring", -0.05, 0, 0.48, 0.72 + gene.shell * 0.12, 0.62 + gene.shell * 0.08, 0.05, 0, 0, mix(BONE, INK, 0.35), { roughness: 0.5 }));
  }

  if (gene.plate >= PART_VISIBLE) {
    for (const [x, yaw] of [
      [-0.35, 0.15],
      [0.05, 0],
      [0.42, -0.12],
    ] as [number, number][]) {
      list.push(part("plate", "plate", x, 0, 0.58, 0.22, 0.7 + gene.plate * 0.28, 0.08, yaw, 0.15, PLATE, { roughness: 0.78, mark: MARK.plate }));
    }
  }

  if (gene.camo >= PART_VISIBLE) {
    for (const side of [-1, 1]) {
      list.push(part("eyespot", "ring", -0.32, side * 0.95, 0.48, 0.22, 0.22, 0.05, 0, 0, RING, { roughness: 0.35, mark: MARK.enamel }));
      list.push(part("eyespot", "sphere", -0.28, side * 0.95, 0.52, 0.1, 0.1, 0.06, 0, 0, INK, { roughness: 0.25 }));
      list.push(part("eyespot", "sphere", -0.24, side * 0.95, 0.55, 0.045, 0.045, 0.03, 0, 0, RING, { emissive: 0.2 }));
    }
  }

  if (gene.glow >= PART_VISIBLE) {
    const spots: [number, number, number][] = [
      [-0.4, 0.34, 0.6],
      [0.12, -0.38, 0.62],
      [0.42, 0.2, 0.66],
      [-0.1, 0, 0.7],
    ];
    const scale = 0.09 + gene.glow * 0.09;
    for (const [x, y, z] of spots) {
      list.push(part("glow", "sphere", x, y, z, scale, scale, scale, 0, 0, GLOW, { emissive: 1, roughness: 0.08, mark: MARK.glow, halo: true }));
    }
  }

  if (gene.photo >= PART_VISIBLE) {
    const spots: [number, number, number][] = [
      [-0.3, 0.18, 0.74],
      [0.05, -0.2, 0.76],
      [0.3, 0.14, 0.74],
      [0.12, 0.02, 0.78],
    ];
    for (const [x, y, z] of spots) {
      list.push(part("photo", "plate", x, y, z, 0.1 + gene.photo * 0.06, 0.12 + gene.photo * 0.05, 0.04, x * 0.4, 0.2, PHOTO, { emissive: 0.35, roughness: 0.28, mark: MARK.leaf }));
    }
  }

  if (gene.siphon >= PART_VISIBLE) {
    list.push(part("siphon", "tube", 1.48, 0, 0.28, 0.3 + gene.siphon * 0.45, 0.1, 0.09, 0.05, 0.25, SIPHON, { roughness: 0.3, mark: MARK.smooth }));
    list.push(part("siphon", "ring", 1.78 + gene.siphon * 0.2, 0, 0.26, 0.12, 0.12, 0.04, 0, 1.2, mix(SIPHON, INK, 0.25), { roughness: 0.35 }));
    list.push(part("siphon", "sphere", 1.86 + gene.siphon * 0.22, 0, 0.26, 0.07, 0.07, 0.07, 0, 0, mix(SIPHON, [1, 1, 1], 0.25), { roughness: 0.2, emissive: 0.15 }));
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
  pitch: number,
  color: RGB,
  material: { emissive?: number; roughness?: number; camo?: number; wiggle?: number; mark?: number; halo?: boolean } = {},
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
    pitch,
    color,
    emissive: material.emissive ?? 0,
    roughness: material.roughness ?? 0.45,
    camo: material.camo ?? 0,
    wiggle: material.wiggle ?? 0,
    mark: material.mark ?? MARK.smooth,
    halo: material.halo ?? false,
  };
}

function unitColor(color: [number, number, number]): RGB {
  return [color[0] / 255, color[1] / 255, color[2] / 255];
}

function mix(a: RGB, b: RGB, amount: number): RGB {
  return [a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount, a[2] + (b[2] - a[2]) * amount];
}
