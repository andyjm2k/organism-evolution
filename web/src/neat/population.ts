/** Initial population. create_new draws node attributes even though bootstrap replaces them. */

import type { PythonRandom } from "../rng/python-random";
import type { GenomeConfig } from "./config";
import { Genome } from "./genome";
import { SpeciesSet } from "./species";

export interface Population {
  genomes: Map<number, Genome>;
  species: SpeciesSet;
  generation: number;
  config: GenomeConfig;
}

/** Build pop_size unconnected genomes and speciate them once. */
export function createPopulation(config: GenomeConfig, rng: PythonRandom): Population {
  const genomes = new Map<number, Genome>();
  for (let key = 1; key <= config.popSize; key += 1) {
    const genome = new Genome(key);
    genome.configureNew(config, rng);
    genomes.set(key, genome);
  }
  const species = new SpeciesSet(config.compatibilityThreshold);
  species.speciate(config, genomes, 0);
  return { genomes, species, generation: 0, config };
}
