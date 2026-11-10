/** Live bodies and the carrying-capacity cap. */

import type { PythonRandom } from "../rng/python-random";
import type { Genome } from "../neat/genome";
import type { GenomeConfig } from "../neat/config";
import { CompiledNetwork } from "../neat/network";
import { Organism, type EnvironmentConfig } from "./organism";
import type { Population } from "../neat/population";
import type { LivingWorld } from "./world";

export class PopulationRegistry {
  private organisms = new Map<number, Organism>();
  private nextId = 1;

  constructor(
    readonly maxPopulation: number,
    private environment: EnvironmentConfig,
    private config: GenomeConfig,
    private rng: PythonRandom,
  ) {}

  seed(genomes: Iterable<[number, Genome]>, population: Population, world: LivingWorld): void {
    for (const [genomeId, genome] of genomes) {
      this.nextId = Math.max(this.nextId, genomeId + 1);
      const position: [number, number] = [
        this.rng.randint(10, Math.max(11, this.environment.width - 10)),
        this.rng.randint(10, Math.max(11, this.environment.height - 10)),
      ];
      const speciesId = population.species.getSpeciesId(genomeId);
      const organism = this.make(genome, position, speciesId, genomeId, genome.dietKey, world);
      this.organisms.set(genomeId, organism);
    }
  }

  alive(): Organism[] {
    return [...this.organisms.values()].filter((organism) => organism.energy > 0 && organism.position);
  }

  all(): Organism[] {
    return [...this.organisms.values()];
  }

  count(): number {
    return this.alive().length;
  }

  atCapacity(): boolean {
    return this.count() >= this.maxPopulation;
  }

  has(genomeId: number): boolean {
    return this.organisms.has(genomeId);
  }

  /** Next id the registry will hand out. */
  peekNextId(): number {
    return this.nextId;
  }

  noteNextId(nextId: number): void {
    this.nextId = nextId;
  }

  replaceAll(organisms: Map<number, Organism>, nextId: number): void {
    this.organisms = organisms;
    this.nextId = nextId;
  }

  /** Register a birth. Ids come from this counter so they stay unique in the world. */
  addBirth(childGenome: Genome, child: Organism): number {
    let genomeId = this.nextId;
    while (this.organisms.has(genomeId)) {
      genomeId += 1;
    }
    this.nextId = genomeId + 1;
    childGenome.key = genomeId;
    child.genomeId = genomeId;
    const preserved = child.energy;
    child.recomputeAttributes();
    child.energy = Math.min(preserved, child.maxEnergy);
    child.network = CompiledNetwork.fromGenome(childGenome, this.config);
    this.organisms.set(genomeId, child);
    return genomeId;
  }

  removeDead(): number[] {
    const dead = [...this.organisms.entries()].filter(([, organism]) => organism.energy <= 0).map(([id]) => id);
    for (const id of dead) {
      this.organisms.delete(id);
    }
    return dead;
  }

  /** Lowest selection fitness first. */
  cullWorst(score: (genomeId: number) => number, count: number): number[] {
    const ranked = this.alive().sort((a, b) => score(a.genomeId) - score(b.genomeId));
    const removed: number[] = [];
    for (const organism of ranked.slice(0, Math.max(0, count))) {
      organism.energy = 0;
      removed.push(organism.genomeId);
      this.organisms.delete(organism.genomeId);
    }
    return removed;
  }

  spawn(genome: Genome, population: Population, world: LivingWorld): Organism | null {
    if (this.atCapacity()) {
      return null;
    }
    let genomeId = genome.key;
    if (this.organisms.has(genomeId)) {
      genomeId = this.nextId;
      this.nextId += 1;
      genome.key = genomeId;
    } else {
      this.nextId = Math.max(this.nextId, genomeId + 1);
    }
    const position: [number, number] = [
      this.rng.randint(10, Math.max(11, this.environment.width - 10)),
      this.rng.randint(10, Math.max(11, this.environment.height - 10)),
    ];
    const speciesId = population.species.genomeToSpecies.get(genomeId) ?? population.species.getSpeciesId(genomeId);
    const organism = this.make(genome, position, speciesId, genomeId, genome.dietKey, world);
    this.organisms.set(genomeId, organism);
    return organism;
  }

  private make(
    genome: Genome,
    position: [number, number],
    speciesId: number,
    genomeId: number,
    dietKey: number,
    world: LivingWorld,
  ): Organism {
    const organism = new Organism(genome, this.config, position, this.environment, this.rng, speciesId, dietKey);
    organism.genomeId = genomeId;
    organism.world = world;
    return organism;
  }
}
