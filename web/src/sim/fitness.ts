/** Display fitness. The tiny uniform noise is part of the shared random stream. */

import type { PythonRandom } from "../rng/python-random";
import { huntIntake, plantIntake } from "./body-genes";
import type { Organism } from "./organism";

export function calculateFitness(
  organism: Organism,
  width: number,
  height: number,
  rng: PythonRandom,
): number {
  const survival = (organism.stepsTaken * 0.4 + organism.energy * 0.3) * organism.energyEfficiency;
  const role = herbivoreScore(organism) * plantIntake(organism.diet) + carnivoreScore(organism) * huntIntake(organism.diet);
  const centerX = width / 2;
  const centerY = height / 2;
  const distCenter = Math.hypot(organism.position![0] - centerX, organism.position![1] - centerY);
  const maxDist = Math.hypot(width / 2, height / 2) || 1;
  const centerReward = (1 - distCenter / maxDist) * 100;
  const exploration = organism.lastPositions.length
    ? Math.min(200, uniqueCells(organism.lastPositions) * 2)
    : 0;
  const boundary = Math.min(
    organism.position![0],
    organism.position![1],
    width - organism.position![0],
    height - organism.position![1],
  );
  const safe = Math.min(width, height) * 0.1;
  const boundaryMultiplier = boundary >= safe ? 1.2 : 0.5 + (boundary / safe) * 0.5;
  let fitness = (survival * 0.3 + role * 0.4 + centerReward * 0.15 + exploration * 0.15) * boundaryMultiplier;
  fitness += organism.fitnessBonus;
  fitness += rng.uniform(-0.05, 0.05);
  if (fitness > organism.highestFitness) {
    organism.highestFitness = fitness;
  }
  return fitness;
}

function carnivoreScore(organism: Organism): number {
  const efficiency = consumptionEfficiency(organism.avgStepsBetweenHunts, 100);
  const sizeBonus = organism.size < 1 ? (organism.speed > 2 ? 1.5 : 1) : organism.size > 2 ? 1.3 : 1;
  const speedBonus = organism.speed > 2 ? (organism.size < 1.5 ? 1.4 : 1.1) : 1;
  const territory = Math.min(1.5, organism.stepsTaken / 1000);
  const early = organism.organismsConsumed > 0 && organism.stepsTaken < 500 ? 1.5 : 1;
  let role =
    organism.organismsConsumed * 400 +
    organism.organismsConsumed * 250 * efficiency +
    organism.organismsConsumed * 150 * sizeBonus +
    organism.organismsConsumed * 150 * speedBonus +
    organism.organismsConsumed * 100 * early +
    organism.stepsTaken * 0.6 * organism.speed * territory;
  if (organism.stepsSinceLastHunt > 150) {
    role *= Math.exp(-organism.stepsSinceLastHunt / 350);
  }
  return role;
}

function herbivoreScore(organism: Organism): number {
  const efficiency = consumptionEfficiency(organism.avgStepsBetweenFood, 50);
  const speedBonus = organism.speed > 2 ? 1.5 : organism.speed > 1.5 ? 1.2 : 1;
  const sizeBonus = organism.size > 1.5 ? 1.3 : organism.size < 0.8 ? 1.2 : 1;
  const efficiencyBonus = organism.energyEfficiency * 1.2;
  let role =
    organism.foodConsumed * 350 +
    organism.foodConsumed * 250 * efficiency +
    organism.foodConsumed * 100 * speedBonus +
    organism.foodConsumed * 100 * sizeBonus +
    organism.foodConsumed * 50 * efficiencyBonus +
    organism.stepsTaken * 0.5 * organism.speed;
  if (organism.stepsSinceLastFood > 100) {
    role *= Math.exp(-organism.stepsSinceLastFood / 250);
  }
  return role;
}

function consumptionEfficiency(steps: number[], optimal: number): number {
  if (steps.length === 0) {
    return 0;
  }
  const avg = steps.reduce((sum, value) => sum + value, 0) / steps.length;
  return Math.exp(-avg / optimal);
}

function uniqueCells(cells: [number, number][]): number {
  const seen = new Set<string>();
  for (const [x, y] of cells) {
    seen.add(`${x},${y}`);
  }
  return seen.size;
}
