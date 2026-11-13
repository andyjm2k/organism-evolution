/** Physical organs. Founder values leave speed, hunger, and senses unchanged. */

import type { PythonRandom } from "../rng/python-random";

export const PART_KEYS = [
  "eye",
  "antenna",
  "lateral",
  "jaw",
  "claw",
  "horn",
  "spine",
  "venom",
  "shell",
  "plate",
  "camo",
  "fin",
  "glow",
  "photo",
  "siphon",
] as const;

export type PartKey = (typeof PART_KEYS)[number];
export type PartLevels = Record<PartKey, number>;

/** Organs render once a gene clears this. A bud from zero is meant to clear it. */
export const PART_VISIBLE = 0.16;

/** Highest signature a body can reach. Sensing queries use it only after a sense grows. */
export const SIGNATURE_MAX = 1.33;

const EYE_BASE = 0.42;
const FIN_BASE = 0.48;
const BUD_CHANCE = 0.22;
const BUD_MIN = 0.18;
const BUD_MAX = 0.34;

const BASELINE: PartLevels = {
  eye: EYE_BASE,
  antenna: 0,
  lateral: 0,
  jaw: 0,
  claw: 0,
  horn: 0,
  spine: 0,
  venom: 0,
  shell: 0,
  plate: 0,
  camo: 0,
  fin: FIN_BASE,
  glow: 0,
  photo: 0,
  siphon: 0,
};

const SUMMARY: Record<PartKey, string> = {
  eye: "eyes",
  antenna: "antennae",
  lateral: "lateral line",
  jaw: "jaw",
  claw: "claws",
  horn: "horn",
  spine: "spines",
  venom: "venom gland",
  shell: "shell",
  plate: "plates",
  camo: "camouflage",
  fin: "tail fin",
  glow: "glow",
  photo: "light patches",
  siphon: "siphon",
};

const GAIN: Record<PartKey, string> = {
  eye: "Larger eyes see farther and cost more energy.",
  antenna: "Antennae find plants farther away and add drag.",
  lateral: "A lateral line feels threats farther away and costs energy.",
  jaw: "A heavier jaw bites harder and costs energy.",
  claw: "Claws bite harder and cost more energy to swing.",
  horn: "A horn adds a hit and some armor, and it slows the body.",
  spine: "Spines injure anything that bites, and they slow the body.",
  venom: "A venom gland adds poison that ignores armor, and it costs energy.",
  shell: "A shell blocks a large share of bites, and the body is slow and hungry.",
  plate: "Armor plates block some damage and slow the body.",
  camo: "Camouflage is harder to notice and shortens the animal's own sight.",
  fin: "A larger tail is faster and burns more energy while moving.",
  glow: "Bioluminescence helps mates find you, and hunters notice you too.",
  photo: "Green patches make energy inside nutrient clouds and weaken bites.",
  siphon: "A siphon pulls more from plants and weakens bites.",
};

const LOSS: Partial<Record<PartKey, string>> = {
  eye: "Smaller eyes see less.",
  fin: "A smaller tail is slower.",
};

export interface MorphStats {
  speed: number;
  metabolism: number;
  moveCost: number;
  armor: number;
  weapon: number;
  venom: number;
  spine: number;
  vision: number;
  foodSense: number;
  threatSense: number;
  breedSense: number;
  signature: number;
  photo: number;
  siphon: number;
  huntScale: number;
}

export interface SenseRadii {
  detectionRadius: number;
  foodDetectionRadius: number;
  threatDetectionRadius: number;
  breedingDetectionRadius: number;
}

export interface AnatomyReport {
  summary: string;
  trades: string[];
}

/** Exact founding body. Stats from this are a no-op on the living-world digest. */
export function founderParts(): PartLevels {
  return { ...BASELINE };
}

/** Missing or old saves read as the founding body. */
export function readParts(genes: { parts?: Partial<PartLevels> } | null | undefined): PartLevels {
  const next = founderParts();
  if (!genes?.parts) {
    return next;
  }
  for (const key of PART_KEYS) {
    const value = genes.parts[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      next[key] = clamp(value, 0, 1);
    }
  }
  return next;
}

/** Drift every organ, and sometimes bud one new organ far enough to see. */
export function mutateParts(parts: PartLevels, rng: PythonRandom, variance = 0.04): PartLevels {
  const next = { ...parts };
  for (const key of PART_KEYS) {
    next[key] = clamp(next[key] + rng.gauss(0, variance), 0, 1);
  }
  if (rng.random() < BUD_CHANCE) {
    const key = rng.choice(PART_KEYS);
    next[key] = clamp(next[key] + rng.uniform(BUD_MIN, BUD_MAX), 0, 1);
  }
  return next;
}

/** Multipliers are 1 and additives are 0 when every gene sits on its baseline. */
export function morphStats(parts: PartLevels): MorphStats {
  const eye = Math.max(0, parts.eye - EYE_BASE);
  const eyeLost = Math.max(0, EYE_BASE - parts.eye);
  const fin = Math.max(0, parts.fin - FIN_BASE);
  const finLost = Math.max(0, FIN_BASE - parts.fin);
  let speed = 1;
  speed *= 1 + 0.48 * fin - 0.55 * finLost;
  speed *= 1 - 0.32 * parts.shell;
  speed *= 1 - 0.14 * parts.plate;
  speed *= 1 - 0.12 * parts.horn;
  speed *= 1 - 0.1 * parts.spine;
  speed *= 1 - 0.16 * parts.antenna;
  let metabolism = 1;
  metabolism += 0.22 * eye;
  metabolism += 0.12 * parts.lateral;
  metabolism += 0.14 * parts.jaw;
  metabolism += 0.2 * parts.venom;
  metabolism += 0.24 * parts.shell;
  metabolism += 0.1 * parts.glow;
  const moveCost = 1 + 0.28 * fin + 0.18 * parts.claw;
  const armor = Math.min(0.62, 0.58 * parts.shell + 0.32 * parts.plate + 0.16 * parts.horn);
  const soften = Math.max(0, 1 - 0.4 * parts.photo - 0.3 * parts.siphon);
  const weapon = (1.15 * parts.jaw + 1.05 * parts.claw + 0.8 * parts.horn) * soften;
  const vision = clamp(1 + 0.8 * eye - 0.85 * eyeLost - 0.18 * parts.camo, 0.4, 2);
  const foodSense = clamp(1 + 0.6 * parts.antenna + 0.25 * parts.siphon, 1, 2);
  const threatSense = clamp(1 + 0.7 * parts.lateral, 1, 1.9);
  const breedSense = clamp(1 + 0.6 * parts.glow, 1, 1.8);
  const signature = clamp(1 - 0.45 * parts.camo + 0.33 * parts.glow, 0.5, SIGNATURE_MAX);
  const huntScale = clamp(1 - 0.5 * parts.photo - 0.28 * parts.siphon, 0.28, 1);
  return {
    speed: clamp(speed, 0.34, 1.85),
    metabolism: clamp(metabolism, 1, 2.6),
    moveCost,
    armor,
    weapon,
    venom: 18 * parts.venom,
    spine: 0.5 * parts.spine,
    vision,
    foodSense,
    threatSense,
    breedSense,
    signature,
    photo: 0.055 * parts.photo,
    siphon: 1 + 0.55 * parts.siphon,
    huntScale,
  };
}

/** How far this body can see that kind of target before camouflage or glow. */
export function senseRadius(base: number, sense: number, vision: number): number {
  return base * sense * vision;
}

/**
 * Neighbor query large enough for the body's own senses.
 * A founding body stays on the original detection radius, so cell-walk order does not change.
 */
export function neighborQueryRadius(env: SenseRadii, stats: MorphStats): number {
  const room = stats.vision > 1 || stats.threatSense > 1 || stats.breedSense > 1 ? SIGNATURE_MAX : 1;
  return Math.max(
    env.detectionRadius * stats.vision * room,
    senseRadius(env.threatDetectionRadius, stats.threatSense, stats.vision) * room,
    senseRadius(env.breedingDetectionRadius, stats.breedSense, stats.vision) * room,
  );
}

/** Radii passed into the neural sensor. Founder stats reproduce the environment radii exactly. */
export function sensedRadii(env: SenseRadii, stats: MorphStats): SenseRadii {
  return {
    detectionRadius: senseRadius(env.detectionRadius, 1, stats.vision),
    foodDetectionRadius: senseRadius(env.foodDetectionRadius, stats.foodSense, stats.vision),
    threatDetectionRadius: senseRadius(env.threatDetectionRadius, stats.threatSense, stats.vision),
    breedingDetectionRadius: senseRadius(env.breedingDetectionRadius, stats.breedSense, stats.vision),
  };
}

/** Energy gained while standing in a nutrient cloud. */
export function absorbLight(
  energy: number,
  maxEnergy: number,
  position: [number, number],
  clouds: { position: [number, number]; radius: number }[],
  rate: number,
): number {
  if (rate <= 0) {
    return energy;
  }
  for (const cloud of clouds) {
    const dx = position[0] - cloud.position[0];
    const dy = position[1] - cloud.position[1];
    if (dx * dx + dy * dy <= cloud.radius * cloud.radius) {
      return Math.min(maxEnergy, energy + rate);
    }
  }
  return energy;
}

/** Names and tradeoffs for the inspection panel. */
export function anatomyReport(
  parts: PartLevels,
  neural: { jaw: number; claw: number; horn: number; flagella: number },
): AnatomyReport {
  const names = ["eyes", "tail fin"];
  if (neural.jaw >= 0.25 && parts.jaw < PART_VISIBLE) {
    names.push("beak");
  }
  if (neural.horn >= 0.25 && parts.horn < PART_VISIBLE) {
    names.push("short horn");
  }
  if (neural.claw >= 0.25 && parts.claw < PART_VISIBLE) {
    names.push("claws");
  }
  for (const key of PART_KEYS) {
    if (key === "eye" || key === "fin") {
      continue;
    }
    if (parts[key] >= PART_VISIBLE) {
      names.push(SUMMARY[key]);
    }
  }
  const trades: string[] = [];
  for (const key of PART_KEYS) {
    const above = parts[key] - BASELINE[key];
    const below = BASELINE[key] - parts[key];
    if (above >= 0.08) {
      trades.push(GAIN[key]);
    } else if (below >= 0.08 && LOSS[key]) {
      trades.push(LOSS[key]);
    }
  }
  if (trades.length === 0) {
    trades.push("Founding body. Offspring can grow organs, and each organ helps one job and costs another.");
  }
  return { summary: `Anatomy: ${names.join(", ")}`, trades };
}

export function partSummary(key: PartKey): string {
  return SUMMARY[key];
}

export function partGain(key: PartKey): string {
  return GAIN[key];
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}
