/** Steady-state cull, immigration, and species-id restoration. */

import type { PythonRandom } from "../rng/python-random";
import type { GenomeConfig } from "../neat/config";
import { Genome } from "../neat/genome";
import type { Population } from "../neat/population";
import { Species } from "../neat/species";
import { inheritBodyGenes } from "./body-genes";
import { bootstrapImmigrant, type GenesisConfig } from "./bootstrap";
import type { Organism } from "./organism";
import type { PopulationRegistry } from "./registry";
import type { FitnessTracker } from "./fitness-tracker";
import type { LivingWorld } from "./world";

const SEARCH_SPAN = 8;

export class SelectionScheduler {
  constructor(
    readonly interval: number,
    readonly maxPopulation: number,
    readonly cullFraction: number,
    readonly immigrationRate: number,
    readonly minAgeSteps: number,
    private rng: PythonRandom,
  ) {}

  shouldRun(step: number): boolean {
    return step > 0 && step % this.interval === 0;
  }

  run(
    population: Population,
    registry: PopulationRegistry,
    fitness: FitnessTracker,
    world: LivingWorld,
    sim: SpeciesBand,
    genesis: GenesisConfig,
    log: (line: string) => void,
  ): void {
    const alive = registry.alive();
    if (alive.length === 0) {
      log("Selection: population extinct — reseeding immigrants");
      this.inject(population, registry, fitness, world, genesis, 5, log);
      this.refresh(population, registry, sim, log);
      return;
    }
    fitness.apply(population.genomes);
    const young = new Set(
      alive.filter((organism) => organism.stepsTaken < this.minAgeSteps).map((organism) => organism.genomeId),
    );
    const eligible = alive.length - young.size;
    if (eligible > 0) {
      const cullCount = Math.max(1, Math.trunc(eligible * this.cullFraction));
      const age = new Map(alive.map((organism) => [organism.genomeId, organism.stepsTaken]));
      const culled = registry.cullWorst((id) => fitness.perStep(id, age.get(id) ?? 1), cullCount, young, true);
      log(
        `Selection: removed ${culled.length} lowest-fitness-per-step organisms older than ${this.minAgeSteps} steps (${registry.count()} remain)`,
      );
    }
    const deficit = this.maxPopulation - registry.count();
    if (deficit > 0 && this.rng.random() < this.immigrationRate) {
      const inject = Math.min(deficit, Math.max(1, Math.trunc(deficit * 0.3)));
      this.inject(population, registry, fitness, world, genesis, inject, log);
    }
    this.refresh(population, registry, sim, log);
  }

  private inject(
    population: Population,
    registry: PopulationRegistry,
    fitness: FitnessTracker,
    world: LivingWorld,
    genesis: GenesisConfig,
    count: number,
    log: (line: string) => void,
  ): void {
    let created = 0;
    const top = fitness.top(2);
    for (let i = 0; i < count; i += 1) {
      if (registry.atCapacity()) {
        break;
      }
      const genome = createImmigrant(population, this.rng, genesis, top);
      if (!genome) {
        break;
      }
      const living = new Set(registry.alive().map((organism) => organism.genomeId));
      assignSpecies(population, genome, living);
      fitness.init(genome.key);
      const organism = registry.spawn(genome, population, world);
      if (organism) {
        created += 1;
      }
    }
    if (created) {
      log(`Selection: injected ${created} immigrant organisms`);
    }
  }

  private refresh(population: Population, registry: PopulationRegistry, sim: SpeciesBand, log: (line: string) => void): void {
    const previous = population.species.threshold;
    const count = refreshLivingSpecies(population, registry.alive(), sim);
    const updated = population.species.threshold;
    if (Math.abs(updated - previous) >= 0.05) {
      log(`Selection: ${count} living species (compatibility ${previous.toFixed(2)} -> ${updated.toFixed(2)})`);
    } else {
      log(`Selection: ${count} living species`);
    }
  }
}

export interface SpeciesBand {
  targetSpecies: number;
  minSpecies: number;
  maxSpecies: number;
}

function createImmigrant(population: Population, rng: PythonRandom, genesis: GenesisConfig, topIds: number[]): Genome {
  const maxKey = Math.max(0, ...population.genomes.keys());
  const genomeId = maxKey + 1;
  if (topIds.length >= 2 && topIds.every((id) => population.genomes.has(id))) {
    const genome = new Genome(genomeId);
    const parentA = population.genomes.get(topIds[0])!;
    const parentB = population.genomes.get(topIds[1])!;
    parentA.fitness = parentA.fitness ?? 0;
    parentB.fitness = parentB.fitness ?? 0;
    genome.configureCrossover(parentA, parentB, population.config, rng);
    genome.mutate(population.config, rng);
    for (const connection of genome.connections.values()) {
      connection.enabled = true;
    }
    inheritBodyGenes(genome, parentA, parentB, rng);
    genome.key = genomeId;
    genome.dietKey = parentA.dietKey ?? parentB.dietKey ?? 1;
    population.genomes.set(genomeId, genome);
    return genome;
  }
  const genome = bootstrapImmigrant(population.config, rng, genesis);
  genome.key = genomeId;
  population.genomes.set(genomeId, genome);
  return genome;
}

/** Nearest living species, or a new one when the genome is past the threshold. */
export function assignSpecies(population: Population, genome: Genome, livingIds: Set<number> | null): number {
  if (livingIds) {
    dropAbsent(population, livingIds);
  }
  const threshold = population.species.threshold;
  let bestSid: number | null = null;
  let bestDistance: number | null = null;
  for (const [sid, species] of population.species.species) {
    for (const member of species.members.values()) {
      const distance = genome.distance(member, population.config);
      if (bestDistance === null || distance < bestDistance) {
        bestDistance = distance;
        bestSid = sid;
      }
    }
  }
  if (bestSid === null || bestDistance! >= threshold) {
    const sid = population.species.nextSpeciesId;
    population.species.nextSpeciesId += 1;
    return foundSpecies(population, genome, sid);
  }
  const species = population.species.species.get(bestSid)!;
  species.members.set(genome.key, genome);
  population.species.genomeToSpecies.set(genome.key, bestSid);
  const repKey = species.representative?.key;
  if (!species.representative || !species.members.has(repKey!)) {
    species.representative = genome;
  }
  return bestSid;
}

function foundSpecies(population: Population, genome: Genome, sid: number): number {
  const created = new Species(sid);
  created.representative = genome;
  created.members.set(genome.key, genome);
  population.species.species.set(sid, created);
  population.species.genomeToSpecies.set(genome.key, sid);
  return sid;
}

function dropAbsent(population: Population, livingIds: Set<number>): void {
  const empty: number[] = [];
  for (const [sid, species] of population.species.species) {
    for (const memberId of [...species.members.keys()]) {
      if (!livingIds.has(memberId)) {
        species.members.delete(memberId);
        if (population.species.genomeToSpecies.get(memberId) === sid) {
          population.species.genomeToSpecies.delete(memberId);
        }
      }
    }
    const repKey = species.representative?.key;
    if (species.members.size && (species.representative === null || !species.members.has(repKey!))) {
      species.representative = species.members.values().next().value ?? null;
    }
    if (species.members.size === 0) {
      empty.push(sid);
    }
  }
  for (const sid of empty) {
    population.species.species.delete(sid);
  }
}

export function refreshLivingSpecies(population: Population, organisms: Organism[], sim: SpeciesBand): number {
  const living = new Map<number, Genome>();
  const previous = new Map<number, number>();
  for (const organism of organisms) {
    if (organism.energy <= 0) {
      continue;
    }
    organism.genome.key = organism.genomeId;
    living.set(organism.genomeId, organism.genome);
    if (organism.speciesId !== null) {
      previous.set(organism.genomeId, organism.speciesId);
    }
  }
  if (living.size === 0) {
    return 0;
  }
  tuneThreshold(population, living, sim);
  population.species.species.clear();
  population.species.genomeToSpecies.clear();
  population.species.speciate(population.config, living, population.generation);
  restoreSpeciesIds(population, previous);
  for (const organism of organisms) {
    const sid = population.species.genomeToSpecies.get(organism.genomeId);
    if (sid !== undefined) {
      organism.speciesId = sid;
    }
  }
  return population.species.species.size;
}

function tuneThreshold(population: Population, genomes: Map<number, Genome>, sim: SpeciesBand): number {
  if (population.species.baseline === null) {
    population.species.baseline = population.species.threshold;
  }
  const baseline = population.species.baseline;
  let current = Math.max(population.species.threshold, baseline);
  if (genomes.size <= 1) {
    applyThreshold(population, current);
    return current;
  }
  const { minimum, target, ceiling } = speciesSettings(sim, genomes.size);
  const { keys, distances } = pairDistances(genomes, population.config);
  const currentCount = clusterCount(keys, distances, current);
  if (currentCount >= minimum && currentCount <= ceiling) {
    applyThreshold(population, current);
    return current;
  }
  let bestThreshold = current;
  let bestRank: number[] | null = null;
  const candidates = new Set<number>();
  for (let step = 0; step < 9; step += 1) {
    candidates.add(round4(baseline + (SEARCH_SPAN * step) / 8));
  }
  candidates.add(round4(current));
  for (const threshold of [...candidates].sort((a, b) => a - b)) {
    if (threshold + 1e-9 < baseline) {
      continue;
    }
    const count = clusterCount(keys, distances, threshold);
    const rank = [Math.max(0, count - ceiling), Math.abs(count - target), Math.abs(threshold - current), threshold];
    if (!bestRank || compareRank(rank, bestRank) < 0) {
      bestRank = rank;
      bestThreshold = threshold;
    }
  }
  applyThreshold(population, bestThreshold);
  return bestThreshold;
}

function speciesSettings(sim: SpeciesBand, populationSize: number): { minimum: number; target: number; ceiling: number } {
  const packed = Math.max(1, Math.trunc(populationSize / 4));
  let ceiling = Math.max(1, Math.min(Math.max(1, sim.maxSpecies), packed));
  const minimum = Math.max(1, Math.min(sim.minSpecies, ceiling));
  ceiling = Math.max(1, ceiling);
  const target = Math.max(minimum, Math.min(Math.max(1, sim.targetSpecies), ceiling));
  return { minimum, target, ceiling };
}

function pairDistances(genomes: Map<number, Genome>, config: GenomeConfig): { keys: number[]; distances: Map<string, number> } {
  const keys = [...genomes.keys()].sort((a, b) => a - b);
  const distances = new Map<string, number>();
  for (let i = 0; i < keys.length; i += 1) {
    for (let j = i + 1; j < keys.length; j += 1) {
      const distance = genomes.get(keys[i])!.distance(genomes.get(keys[j])!, config);
      distances.set(`${keys[i]}|${keys[j]}`, distance);
      distances.set(`${keys[j]}|${keys[i]}`, distance);
    }
  }
  return { keys, distances };
}

function clusterCount(keys: number[], distances: Map<string, number>, threshold: number): number {
  const representatives: number[] = [];
  for (const genomeId of keys) {
    let nearest: number | null = null;
    let nearestDistance: number | null = null;
    for (const representative of representatives) {
      const distance = distances.get(`${genomeId}|${representative}`)!;
      if (distance < threshold && (nearestDistance === null || distance < nearestDistance)) {
        nearest = representative;
        nearestDistance = distance;
      }
    }
    if (nearest === null) {
      representatives.push(genomeId);
    }
  }
  return representatives.length;
}

function applyThreshold(population: Population, threshold: number): void {
  population.species.threshold = threshold;
  population.config.compatibilityThreshold = threshold;
}

function restoreSpeciesIds(population: Population, previous: Map<number, number>): void {
  if (previous.size === 0 || population.species.species.size === 0) {
    return;
  }
  const ranked: { votes: number; sid: number; bestId: number | null }[] = [];
  for (const [sid, species] of population.species.species) {
    const tally = new Map<number, number>();
    for (const genomeId of species.members.keys()) {
      const oldId = previous.get(genomeId);
      if (oldId === undefined) {
        continue;
      }
      tally.set(oldId, (tally.get(oldId) ?? 0) + 1);
    }
    let bestId: number | null = null;
    let bestVotes = 0;
    for (const [oldId, count] of tally) {
      const better = count > bestVotes;
      const tied = count === bestVotes && (bestId === null || oldId < bestId);
      if (better || tied) {
        bestId = oldId;
        bestVotes = count;
      }
    }
    ranked.push({ votes: bestVotes, sid, bestId });
  }
  ranked.sort((a, b) => b.votes - a.votes || a.sid - b.sid);
  const used = new Set<number>();
  const assignment = new Map<number, number>();
  for (const item of ranked) {
    let chosen: number;
    if (item.bestId !== null && !used.has(item.bestId)) {
      chosen = item.bestId;
    } else if (!used.has(item.sid)) {
      chosen = item.sid;
    } else {
      chosen = population.species.nextSpeciesId;
      population.species.nextSpeciesId += 1;
      while (used.has(chosen)) {
        chosen = population.species.nextSpeciesId;
        population.species.nextSpeciesId += 1;
      }
    }
    used.add(chosen);
    assignment.set(item.sid, chosen);
  }
  if ([...assignment].every(([oldId, newId]) => oldId === newId)) {
    return;
  }
  const rebuilt = new Map<number, import("../neat/species").Species>();
  const rebuiltMap = new Map<number, number>();
  for (const [sid, species] of population.species.species) {
    const stable = assignment.get(sid) ?? sid;
    species.key = stable;
    rebuilt.set(stable, species);
    for (const genomeId of species.members.keys()) {
      rebuiltMap.set(genomeId, stable);
    }
  }
  population.species.species = rebuilt;
  population.species.genomeToSpecies = rebuiltMap;
}

function round4(value: number): number {
  return Math.round(value * 10000) / 10000;
}

function compareRank(a: number[], b: number[]): number {
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return a[i] - b[i];
    }
  }
  return 0;
}
