/** Replace the random initial population with the shared founding genome. */

import type { PythonRandom } from "../rng/python-random";
import type { GenomeConfig } from "../neat/config";
import { Genome } from "../neat/genome";
import type { Population } from "../neat/population";
import { makeFounderGenes } from "./body-genes";

const FOOD_DX = -12;
const FOOD_DY = -13;
const PREY_DX = -19;
const PREY_DY = -20;
const VX = 0;
const VY = 1;
const BREED = 2;
const REST = 3;
const WIRES: [number, number, number][] = [
  [FOOD_DX, VX, 1.8],
  [FOOD_DY, VY, 1.8],
  [PREY_DX, VX, 1.8],
  [PREY_DY, VY, 1.8],
];
const BIASES: Record<number, number> = { [VX]: 0.2, [VY]: 0.05, [BREED]: 1, [REST]: -1 };

export interface GenesisConfig {
  archetypeCount: number;
  extraHidden: number;
  extraConnections: number;
  weightJitter: number;
  foragingBias: number;
}

/** Clone archetypes onto the existing genome ids, then speciate again. */
export function bootstrapPopulation(
  population: Population,
  rng: PythonRandom,
  genesis: GenesisConfig,
): void {
  const archetypes = createArchetypes(population.config, rng, genesis);
  const bootstrapped = new Map<number, Genome>();
  for (const genomeId of population.genomes.keys()) {
    const template = archetypes[genomeId % genesis.archetypeCount];
    const genome = cloneGenome(population.config, genomeId, template, rng);
    jitter(genome, rng, genesis.weightJitter);
    enableAll(genome);
    genome.dietKey = template.dietKey;
    genome.bodyGenes = makeFounderGenes(rng);
    bootstrapped.set(genomeId, genome);
  }
  population.genomes = bootstrapped;
  population.species.species.clear();
  population.species.genomeToSpecies.clear();
  population.species.speciate(population.config, population.genomes, population.generation);
}

export function bootstrapImmigrant(config: GenomeConfig, rng: PythonRandom, genesis: GenesisConfig): Genome {
  const genome = new Genome(0);
  genome.configureNew(config, rng);
  for (let i = 0; i < genesis.extraHidden; i += 1) {
    genome.mutateAddNode(config, rng);
  }
  for (let i = 0; i < genesis.extraConnections; i += 1) {
    genome.mutateAddConnection(config, rng);
  }
  enableAll(genome);
  genome.fitness = 0;
  genome.dietKey = 1;
  installMotif(genome, config, rng);
  if (genesis.foragingBias > 0) {
    for (const [src, dst] of WIRES) {
      const connection = genome.connections.get(`${src},${dst}`);
      if (connection) {
        connection.weight += rng.uniform(genesis.foragingBias * 0.5, genesis.foragingBias * 1.5);
        connection.enabled = true;
      }
    }
  }
  enableAll(genome);
  genome.bodyGenes = makeFounderGenes(rng);
  return genome;
}

function createArchetypes(config: GenomeConfig, rng: PythonRandom, genesis: GenesisConfig): Genome[] {
  const archetypes: Genome[] = [];
  for (let index = 0; index < genesis.archetypeCount; index += 1) {
    const genome = new Genome(100_000 + index);
    genome.configureNew(config, rng);
    for (let i = 0; i < genesis.extraHidden; i += 1) {
      genome.mutateAddNode(config, rng);
    }
    for (let i = 0; i < genesis.extraConnections; i += 1) {
      genome.mutateAddConnection(config, rng);
    }
    enableAll(genome);
    genome.fitness = 0;
    genome.dietKey = 1;
    installMotif(genome, config, rng);
    archetypes.push(genome);
  }
  return archetypes;
}

function cloneGenome(config: GenomeConfig, id: number, template: Genome, rng: PythonRandom): Genome {
  const clone = new Genome(id);
  clone.fitness = 0;
  template.fitness = 0;
  clone.configureCrossover(template, template, config, rng);
  return clone;
}

function installMotif(genome: Genome, config: GenomeConfig, rng: PythonRandom): void {
  for (const [nodeId, bias] of Object.entries(BIASES)) {
    const node = genome.nodes.get(Number(nodeId));
    if (node) {
      node.bias = bias;
    }
  }
  for (const [src, dst, weight] of WIRES) {
    const key = `${src},${dst}`;
    let existing = genome.connections.get(key);
    if (!existing) {
      const innovation = config.innovation.getInnovationNumber(src, dst, "founding_motif");
      genome.addConnection(config, rng, src, dst, weight, true, innovation);
      existing = genome.connections.get(key)!;
    }
    existing.weight = weight;
    existing.enabled = true;
  }
}

function jitter(genome: Genome, rng: PythonRandom, stdev: number): void {
  if (stdev <= 0) {
    return;
  }
  for (const connection of genome.connections.values()) {
    if (connection.enabled) {
      connection.weight += rng.gauss(0, stdev);
    }
  }
}

function enableAll(genome: Genome): void {
  for (const connection of genome.connections.values()) {
    connection.enabled = true;
  }
}
