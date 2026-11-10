/** A living-world birth is filled, laid, then hatched. */

import type { PythonRandom } from "../rng/python-random";
import { Genome } from "../neat/genome";
import { inheritBodyGenes } from "./body-genes";
import { applyMaturity } from "./growth";
import { Organism } from "./organism";

export class Egg {
  stored = 0;
  position: [number, number] | null = null;
  laid = false;
  stepsLaid = 0;

  constructor(
    readonly parent: Organism,
    readonly partner: Organism,
    readonly child: Organism,
    readonly childGenome: Genome,
    readonly energyGoal: number,
    readonly hatchDelay: number,
  ) {}

  /** Pull energy from the carrier while the breed output stays high. */
  gestate(parent: Organism, desire: number): boolean {
    if (this.laid || this.stored >= this.energyGoal) {
      return true;
    }
    if (desire <= 0.5) {
      return false;
    }
    const take = Math.min(2, parent.energy - 40, this.energyGoal - this.stored);
    if (take <= 0) {
      return false;
    }
    parent.energy -= take;
    this.stored += take;
    return this.stored >= this.energyGoal;
  }

  lay(position: [number, number]): void {
    this.laid = true;
    this.position = [position[0], position[1]];
    this.stepsLaid = 0;
  }

  /** Returns true when the laid egg should hatch. */
  tick(): boolean {
    if (!this.laid) {
      return false;
    }
    this.stepsLaid += 1;
    return this.stepsLaid >= this.hatchDelay;
  }
}

export function fertilize(parent: Organism, partner: Organism): Egg {
  const { child, childGenome } = buildOffspring(parent, partner);
  child.maturity = parent.environment.hatchMaturity;
  applyMaturity(child);
  child.energy = Math.min(child.energy, child.maxEnergy);
  const egg = new Egg(
    parent,
    partner,
    child,
    childGenome,
    parent.environment.eggEnergyGoal,
    parent.environment.eggHatchDelay,
  );
  parent.carriedEgg = egg;
  parent.stepsSinceBreeding = 0;
  partner.stepsSinceBreeding = 0;
  parent.energy = Math.max(0, parent.energy - 10);
  partner.energy = Math.max(0, partner.energy - 10);
  return egg;
}

export function tendEgg(parent: Organism, desire: number): void {
  const egg = parent.carriedEgg;
  if (!egg || !parent.position) {
    return;
  }
  if (!egg.gestate(parent, desire)) {
    return;
  }
  egg.lay(parent.position);
  parent.carriedEgg = null;
  parent.world?.receiveEgg(egg);
}

function buildOffspring(parent: Organism, partner: Organism): { child: Organism; childGenome: Genome } {
  const rng = parent.rng;
  let spawnX = (parent.position![0] + partner.position![0]) / 2 + rng.uniform(-5, 5);
  let spawnY = (parent.position![1] + partner.position![1]) / 2 + rng.uniform(-5, 5);
  spawnX = Math.max(0, Math.min(spawnX, parent.environment.width));
  spawnY = Math.max(0, Math.min(spawnY, parent.environment.height));
  parent.genome.fitness = parent.calculateFitness();
  partner.genome.fitness = partner.calculateFitness();
  const parent1 = parent.genome.fitness >= partner.genome.fitness ? parent.genome : partner.genome;
  const parent2 = parent1 === parent.genome ? partner.genome : parent.genome;
  const childGenome = new Genome(0);
  childGenome.configureCrossover(parent1, parent2, parent.config, rng);
  childGenome.mutate(parent.config, rng);
  inheritBodyGenes(childGenome, parent1, parent2, rng);
  const child = new Organism(
    childGenome,
    parent.config,
    [spawnX, spawnY],
    parent.environment,
    rng,
    parent.speciesId,
    parent.dietKey,
    );
  child.energy = Math.min(parent.environment.offspringStartingEnergy, child.maxEnergy);
  child.stepsSinceBreeding = 0;
  return { child, childGenome };
}
