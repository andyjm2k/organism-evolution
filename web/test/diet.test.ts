/** Diet is a preference: plants fade out, hunting fades in, and both still score. */

import { describe, expect, it } from "vitest";
import { PythonRandom } from "../src/rng/python-random";
import {
  dietLabel,
  huntIntake,
  huntPower,
  huntSizeGate,
  plantIntake,
} from "../src/sim/body-genes";
import { BodyTraits } from "../src/sim/body-traits";
import { huntPrey } from "../src/sim/combat";
import { calculateFitness } from "../src/sim/fitness";
import { FitnessTracker } from "../src/sim/fitness-tracker";
import { Food, graze } from "../src/sim/food-ecology";
import { buildNetworkInputs } from "../src/sim/network-inputs";
import { PlantStrain } from "../src/sim/plant";
import type { Organism } from "../src/sim/organism";

describe("diet preference", () => {
  it("keeps founder diets on plants and specialists on meat", () => {
    expect(plantIntake(0)).toBe(1);
    expect(plantIntake(0.2)).toBe(1);
    expect(huntIntake(0.2)).toBe(0);
    expect(huntPower(0.2)).toBe(0);
    expect(plantIntake(0.6)).toBeCloseTo(0.5);
    expect(huntIntake(0.6)).toBeCloseTo(0.5);
    expect(plantIntake(1)).toBe(0);
    expect(huntIntake(1)).toBe(1);
    expect(huntPower(1)).toBe(1);
    expect(huntSizeGate(0.25)).toBeGreaterThan(huntSizeGate(1));
    expect(huntSizeGate(1)).toBeCloseTo(0.8);
    expect(dietLabel(0.2)).toBe("Herbivore");
    expect(dietLabel(0.5)).toBe("Omnivore");
    expect(dietLabel(0.9)).toBe("Carnivore");
  });

  it("pays a shrinking plant meal and leaves a pellet alone at diet 1", () => {
    const full = feeder(0.2, 50);
    graze(full, [pellet()]);
    expect(full.energy).toBeCloseTo(90);
    expect(full.foodConsumed).toBe(1);
    const mixed = feeder(0.6, 50);
    graze(mixed, [pellet()]);
    expect(mixed.energy).toBeCloseTo(70);
    const obligate = feeder(1, 50);
    const food = pellet();
    graze(obligate, [food]);
    expect(obligate.energy).toBe(50);
    expect(food.position).not.toBeNull();
  });

  it("lets a hunter bite a lower diet and ignore its own diet and larger prey", () => {
    const hunter = animal(0.5, 2, 40, 80);
    const prey = animal(0.1, 1, 100, 10);
    huntPrey(hunter, [prey]);
    expect(prey.energy).toBe(0);
    expect(hunter.organismsConsumed).toBe(1);
    expect(hunter.energy).toBeCloseTo(40 + 100 * 0.7 * huntIntake(0.5));
    const peer = animal(0.5, 2, 100, 20);
    huntPrey(hunter, [peer]);
    expect(peer.health).toBe(20);
    const nip = animal(0.3, 1, 40, 80);
    const bigger = animal(0.1, 1.4, 100, 80);
    huntPrey(nip, [bigger]);
    expect(bigger.health).toBe(80);
    const grazer = animal(0.2, 2, 40, 80);
    const meal = animal(0.0, 1, 100, 10);
    huntPrey(grazer, [meal]);
    expect(meal.health).toBe(10);
  });

  it("keeps plant direction on a hunter and threat direction on a grazer", () => {
    const environment = {
      width: 400,
      height: 400,
      foodDetectionRadius: 200,
      threatDetectionRadius: 50,
      detectionRadius: 50,
      breedingDetectionRadius: 40,
    };
    const hunter = sensor(0.8, true);
    const food = { position: [10, 0] as [number, number] };
    const prey = sensor(0.1, false);
    prey.position = [0, 10];
    const hunting = buildNetworkInputs(hunter, [food], [prey], [], [], environment);
    expect(hunting[11]).toBeCloseTo(1);
    expect(hunting[19]).toBeGreaterThan(0);
    const grazer = sensor(0.1, false);
    const neighbor = sensor(0, false);
    neighbor.position = [0, 30];
    const threat = sensor(0.8, true);
    threat.position = [40, 0];
    const grazing = buildNetworkInputs(grazer, [food], [neighbor], [threat], [], environment);
    expect(grazing[18]).toBeCloseTo(0.8);
    expect(grazing[19]).toBeCloseTo(0);
  });

  it("scores both a meal and a kill for a mixed diet", () => {
    const tracker = new FitnessTracker(150, 0.5, 100, 100);
    const mixed = { genomeId: 3, diet: 0.6, isCarnivore: true, foodConsumed: 1, organismsConsumed: 1 } as Organism;
    tracker.tickAlive(mixed);
    expect(tracker.get(3)).toBeCloseTo(0.5 + 50 + 50);
    tracker.tickAlive(mixed);
    expect(tracker.get(3)).toBeCloseTo(0.5 + 50 + 50 + 0.5);
  });

  it("keeps a mixed feeder's display score when it has eaten both foods", () => {
    const organism = {
      diet: 0.8,
      stepsTaken: 10,
      energy: 100,
      energyEfficiency: 1,
      position: [100, 100] as [number, number],
      lastPositions: [] as [number, number][],
      fitnessBonus: 0,
      highestFitness: 0,
      organismsConsumed: 1,
      foodConsumed: 1,
      avgStepsBetweenHunts: [10],
      avgStepsBetweenFood: [10],
      stepsSinceLastHunt: 0,
      stepsSinceLastFood: 0,
      speed: 1,
      size: 1,
    } as unknown as Organism;
    const rng = new PythonRandom(1);
    rng.uniform = () => 0;
    expect(calculateFitness(organism, 400, 400, rng)).toBeGreaterThan(100);
  });

  it("splits a prey wire into claw and horn through the middle of the diet range", () => {
    const genome = {
      connections: new Map([
        ["food", { inNode: -12, outNode: 0, weight: 1.8, enabled: true, innovation: 1 }],
        ["prey", { inNode: -19, outNode: 1, weight: 2, enabled: true, innovation: 2 }],
      ]),
    };
    const mid = BodyTraits.fromGenome(genome as never, 0.6);
    expect(mid.jaw).toBeCloseTo(1.8);
    expect(mid.claw).toBeCloseTo(1);
    expect(mid.horn).toBeCloseTo(1);
    const hunter = BodyTraits.fromGenome(genome as never, 1);
    expect(hunter.claw).toBeCloseTo(2);
    expect(hunter.horn).toBeCloseTo(0);
  });
});

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

function sensor(diet: number, carnivore: boolean) {
  return {
    position: [0, 0] as [number, number],
    lastPosition: [0, 0] as [number, number],
    speed: 1,
    size: 1,
    energy: 100,
    maxEnergy: 100,
    isCarnivore: carnivore,
    diet,
    canBreed: () => false,
  };
}
