/** Color, diet, and size genes, kept apart from the NEAT brain. */

import type { PythonRandom } from "../rng/python-random";
import type { BodyGeneState } from "../neat/genome";

const COLOR_MIN = 0;
const COLOR_MAX = 1;
const SIZE_MIN = 0.75;
const SIZE_MAX = 1.35;
export const CARNIVORE_DIET = 0.5;
const DEFAULT_VARIANCE = 0.06;
const FOUNDER_RED = 0.42;
const FOUNDER_GREEN = 0.7;
const FOUNDER_BLUE = 0.28;
const FOUNDER_DIET = 0.08;

export function copyGenes(genes: BodyGeneState): BodyGeneState {
  return { ...genes };
}

/** Nudge every gene by a gaussian and clamp it. */
export function mutateGenes(genes: BodyGeneState, rng: PythonRandom, variance = DEFAULT_VARIANCE): BodyGeneState {
  const spread = variance;
  genes.red = clamp(genes.red + rng.gauss(0, spread), COLOR_MIN, COLOR_MAX);
  genes.green = clamp(genes.green + rng.gauss(0, spread), COLOR_MIN, COLOR_MAX);
  genes.blue = clamp(genes.blue + rng.gauss(0, spread), COLOR_MIN, COLOR_MAX);
  genes.diet = clamp(genes.diet + rng.gauss(0, spread * 0.45), COLOR_MIN, COLOR_MAX);
  genes.size = clamp(genes.size + rng.gauss(0, spread * 0.5), SIZE_MIN, SIZE_MAX);
  return genes;
}

/** Green grazer founders. Diet is capped so the nudge cannot make a hunter. */
export function makeFounderGenes(rng: PythonRandom): BodyGeneState {
  const genes = mutateGenes(
    { red: FOUNDER_RED, green: FOUNDER_GREEN, blue: FOUNDER_BLUE, diet: FOUNDER_DIET, size: 1 },
    rng,
    0.03,
  );
  genes.diet = Math.min(0.2, Math.max(0, genes.diet));
  genes.size = 1;
  return genes;
}

/** Copy the fitter parent's genes onto the child and nudge them. */
export function inheritBodyGenes(
  child: { bodyGenes: BodyGeneState | null },
  parentA: { bodyGenes: BodyGeneState | null },
  parentB: { bodyGenes: BodyGeneState | null },
  rng: PythonRandom,
): BodyGeneState | null {
  const source = parentA.bodyGenes ?? parentB.bodyGenes;
  if (!source) {
    return null;
  }
  const childGenes = mutateGenes(copyGenes(source), rng);
  child.bodyGenes = childGenes;
  return childGenes;
}

export function geneColor(genes: BodyGeneState): [number, number, number] {
  return [
    Math.round(genes.red * 255),
    Math.round(genes.green * 255),
    Math.round(genes.blue * 255),
  ];
}

export function isCarnivoreDiet(diet: number): boolean {
  return diet >= CARNIVORE_DIET;
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}
