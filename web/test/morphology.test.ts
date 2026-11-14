/** Organ genes stay neutral for founders and change a real stat once they grow. */

import { describe, expect, it } from "vitest";
import { PythonRandom } from "../src/rng/python-random";
import { inheritBodyGenes, makeFounderGenes } from "../src/sim/body-genes";
import { biteDamage, applyBite } from "../src/sim/combat";
import { graze, Food } from "../src/sim/food-ecology";
import {
  absorbLight,
  anatomyReport,
  founderParts,
  morphStats,
  mutateParts,
  neighborQueryRadius,
  PART_KEYS,
  sensedRadii,
} from "../src/sim/morphology";
import { PlantStrain } from "../src/sim/plant";
import { buildNetworkInputs } from "../src/sim/network-inputs";
import { creatureParts } from "../src/render/creature-layout";
import type { Organism } from "../src/sim/organism";

const RADII = { detectionRadius: 200, foodDetectionRadius: 1400, threatDetectionRadius: 100, breedingDetectionRadius: 150 };

describe("morphology", () => {
  it("leaves a founding body with the original speed, hunger, and senses", () => {
    const stats = morphStats(founderParts());
    expect(stats.speed).toBe(1);
    expect(stats.metabolism).toBe(1);
    expect(stats.moveCost).toBe(1);
    expect(stats.armor).toBe(0);
    expect(stats.weapon).toBe(0);
    expect(stats.venom).toBe(0);
    expect(stats.spine).toBe(0);
    expect(stats.vision).toBe(1);
    expect(stats.foodSense).toBe(1);
    expect(stats.threatSense).toBe(1);
    expect(stats.breedSense).toBe(1);
    expect(stats.signature).toBe(1);
    expect(stats.photo).toBe(0);
    expect(stats.siphon).toBe(1);
    expect(stats.huntScale).toBe(1);
    expect(neighborQueryRadius(RADII, stats)).toBe(200);
    expect(sensedRadii(RADII, stats)).toEqual(RADII);
    expect(makeFounderGenes(new PythonRandom(8)).parts).toEqual(founderParts());
  });

  it("trades a benefit for a cost on each developed organ", () => {
    const shell = morphStats({ ...founderParts(), shell: 1 });
    expect(shell.armor).toBeGreaterThan(0.4);
    expect(shell.speed).toBeLessThan(0.8);
    expect(shell.metabolism).toBeGreaterThan(1);
    const fin = morphStats({ ...founderParts(), fin: 1 });
    expect(fin.speed).toBeGreaterThan(1);
    expect(fin.moveCost).toBeGreaterThan(1);
    const dim = morphStats({ ...founderParts(), eye: 0 });
    expect(dim.vision).toBeLessThan(1);
    expect(dim.metabolism).toBe(1);
    const glow = morphStats({ ...founderParts(), glow: 1 });
    expect(glow.signature).toBeGreaterThan(1);
    expect(glow.breedSense).toBeGreaterThan(1);
    const camo = morphStats({ ...founderParts(), camo: 1 });
    expect(camo.signature).toBeLessThan(1);
    expect(camo.vision).toBeLessThan(1);
    const photo = morphStats({ ...founderParts(), photo: 1 });
    expect(photo.photo).toBeGreaterThan(0);
    expect(photo.huntScale).toBeLessThan(1);
    const siphon = morphStats({ ...founderParts(), siphon: 1 });
    expect(siphon.siphon).toBeGreaterThan(1);
    expect(siphon.huntScale).toBeLessThan(1);
  });

  it("keeps mutated organs inside 0 to 1 and can change one", () => {
    const next = mutateParts(founderParts(), new PythonRandom(1), 0.2);
    for (const key of PART_KEYS) {
      expect(next[key]).toBeGreaterThanOrEqual(0);
      expect(next[key]).toBeLessThanOrEqual(1);
    }
    expect(PART_KEYS.some((key) => next[key] !== founderParts()[key])).toBe(true);
  });

  it("copies a parent's shell onto a child and keeps it in range", () => {
    const parent = makeFounderGenes(new PythonRandom(1));
    parent.parts = { ...founderParts(), shell: 0.4 };
    const child = { bodyGenes: null as typeof parent | null };
    inheritBodyGenes(child, { bodyGenes: parent }, { bodyGenes: null }, new PythonRandom(2));
    const shell = child.bodyGenes?.parts?.shell ?? -1;
    expect(shell).toBeGreaterThan(0);
    expect(shell).toBeLessThanOrEqual(1);
  });

  it("names a shell and explains its cost once the gene is visible", () => {
    const report = anatomyReport({ ...founderParts(), shell: 0.8 }, { jaw: 0, claw: 0, horn: 0, flagella: 0 });
    expect(report.summary).toContain("shell");
    expect(report.trades.some((line) => line.includes("slow"))).toBe(true);
    const founder = anatomyReport(founderParts(), { jaw: 0, claw: 0, horn: 0, flagella: 0 });
    expect(founder.summary).toContain("eyes");
    expect(founder.trades[0]).toContain("Founding body");
  });

  it("lets a shell shrink a bite and spines hurt the attacker", () => {
    const hunter = animal(1, 1, 80, 80);
    const prey = animal(0, 1, 40, 80);
    const open = biteDamage(hunter, prey);
    prey.morph = morphStats({ ...founderParts(), shell: 1 });
    const blocked = biteDamage(hunter, prey);
    expect(blocked).toBeLessThan(open * 0.55);
    hunter.morph = morphStats({ ...founderParts(), venom: 1 });
    expect(biteDamage(hunter, prey)).toBeGreaterThan(blocked);
    const attacker = animal(1, 1, 80, 50);
    const spiny = animal(0, 1, 40, 80);
    spiny.morph = morphStats({ ...founderParts(), spine: 1 });
    applyBite(attacker, spiny);
    expect(attacker.health).toBeLessThan(50);
  });

  it("gives a siphon a larger plant meal", () => {
    const plain = feeder(0.2, 50);
    graze(plain, [pellet()]);
    const tube = feeder(0.2, 50);
    tube.morph = morphStats({ ...founderParts(), siphon: 1 });
    graze(tube, [pellet()]);
    expect(tube.energy).toBeGreaterThan(plain.energy);
  });

  it("adds light only inside a nutrient cloud", () => {
    expect(absorbLight(10, 100, [0, 0], [{ position: [10, 0], radius: 20 }], 0.5)).toBe(10.5);
    expect(absorbLight(10, 100, [0, 0], [{ position: [100, 0], radius: 5 }], 0.5)).toBe(10);
  });

  it("hides a camouflaged neighbor that a plain one would still sense", () => {
    const environment = {
      width: 400,
      height: 400,
      foodDetectionRadius: 100,
      threatDetectionRadius: 100,
      detectionRadius: 100,
      breedingDetectionRadius: 100,
    };
    const hunter = sensor(1);
    const seen = sensor(0);
    seen.position = [90, 0];
    const visible = buildNetworkInputs(hunter, [], [seen], [], [], environment);
    const hidden = sensor(0);
    hidden.position = [90, 0];
    hidden.morph = { signature: 0.5 };
    const missed = buildNetworkInputs(hunter, [], [hidden], [], [], environment);
    expect(visible[18]).toBeGreaterThan(0);
    expect(missed[16]).toBe(0);
    expect(missed[17]).toBe(1);
  });
});

describe("creature layout", () => {
  it("puts eyes in front of the torso and keeps a horn off the founding body", () => {
    const founder = creatureParts(pose(founderParts(), 1.8));
    const eye = founder.find((part) => part.name === "eye");
    const torso = founder.find((part) => part.name === "torso");
    expect(eye && torso && eye.x > torso.x).toBe(true);
    expect(founder.some((part) => part.name === "tail")).toBe(true);
    expect(founder.some((part) => part.name === "horn")).toBe(false);
    const wired = creatureParts({ ...pose(founderParts(), 0), horn: 3.6 });
    expect(wired.some((part) => part.name === "horn" && part.mesh === "cone")).toBe(true);
    expect(founder.some((part) => part.name === "tooth")).toBe(true);
    expect(new Set(founder.map((part) => part.mesh)).size).toBeGreaterThan(3);
  });

  it("draws a wider shell, eyespots, and a glow halo when those genes grow", () => {
    const armored = creatureParts(pose({ ...founderParts(), shell: 0.8, camo: 0.7, glow: 0.8, horn: 0.8 }, 0));
    const shell = armored.find((part) => part.name === "shell");
    const torso = armored.find((part) => part.name === "torso");
    expect(shell && torso && shell.sy > torso.sy).toBe(true);
    expect(armored.some((part) => part.name === "horn" && part.mesh === "cone")).toBe(true);
    expect(armored.some((part) => part.name === "eyespot" && part.mesh === "ring")).toBe(true);
    expect(armored.some((part) => part.halo)).toBe(true);
    expect(armored.some((part) => part.mesh === "dome")).toBe(true);
  });

  it("builds a distinct mesh for each major organ family", () => {
    const loaded = creatureParts(
      pose(
        {
          ...founderParts(),
          antenna: 0.8,
          lateral: 0.8,
          claw: 0.8,
          spine: 0.8,
          venom: 0.8,
          shell: 0.8,
          plate: 0.8,
          camo: 0.8,
          glow: 0.8,
          photo: 0.8,
          siphon: 0.8,
          jaw: 0.8,
          horn: 0.8,
          fin: 0.9,
          eye: 0.9,
        },
        1.8,
      ),
    );
    const meshes = new Set(loaded.map((part) => part.mesh));
    for (const kind of ["sphere", "capsule", "cone", "fin", "dome", "tube", "claw", "plate", "ring"]) {
      expect(meshes.has(kind as never)).toBe(true);
    }
  });
});

function pose(parts: ReturnType<typeof founderParts>, jaw: number) {
  return { color: [90, 160, 80] as [number, number, number], parts, jaw, claw: 0, horn: 0, flagella: 0, selected: false };
}

function feeder(diet: number, energy: number): Organism {
  return {
    diet,
    energy,
    maxEnergy: 400,
    energyEfficiency: 1,
    position: [0, 0],
    isCarnivore: diet >= 0.5,
    foodConsumed: 0,
    stepsSinceLastFood: 10,
    avgStepsBetweenFood: [],
    fitnessBonus: 0,
    getRadius: () => 6,
    environment: { foodEnergyValue: 40 },
  } as unknown as Organism;
}

function pellet(): Food {
  return new Food(0, 0, new PlantStrain(0, 1, 0.4, 1));
}

function animal(diet: number, size: number, energy: number, health: number): Organism {
  return {
    diet,
    size,
    energy,
    maxEnergy: 500,
    position: [0, 0],
    health,
    genome: null,
    organismsConsumed: 0,
    stepsSinceLastHunt: 4,
    avgStepsBetweenHunts: [],
    fitnessBonus: 0,
    getRadius: () => 6,
  } as unknown as Organism;
}

function sensor(diet: number): {
  position: [number, number];
  lastPosition: [number, number];
  speed: number;
  size: number;
  energy: number;
  maxEnergy: number;
  isCarnivore: boolean;
  diet: number;
  canBreed: () => boolean;
  morph?: { signature: number };
} {
  return {
    position: [0, 0],
    lastPosition: [0, 0],
    speed: 1,
    size: 1,
    energy: 100,
    maxEnergy: 100,
    isCarnivore: diet >= 0.5,
    diet,
    canBreed: () => false,
  };
}
