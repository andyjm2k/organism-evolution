/** Persistent living world. One tick matches LivingWorldSimulation._simulation_step. */

import { PythonRandom } from "../rng/python-random";
import { parseNeatConfig, type GenomeConfig } from "../neat/config";
import { createPopulation, type Population } from "../neat/population";
import { FoodEcology, type Food } from "./food-ecology";
import { SpatialGrid } from "./spatial";
import { withinRadius } from "./distance";
import { Organism, type EnvironmentConfig } from "./organism";
import { PopulationRegistry } from "./registry";
import { FitnessTracker } from "./fitness-tracker";
import { assignSpecies, SelectionScheduler, type SpeciesBand } from "./selection";
import { bootstrapPopulation, type GenesisConfig } from "./bootstrap";
import { scientificName } from "./names";
import type { Egg } from "./egg";
import { BodyTraits } from "./body-traits";
import type { Genome } from "../neat/genome";

export interface SimConfig {
  [key: string]: number | string | boolean | null;
}

export interface BodySnapshot {
  id: number;
  x: number;
  y: number;
  radius: number;
  energy: number;
  maxEnergy: number;
  color: [number, number, number];
  jaw: number;
  claw: number;
  horn: number;
  flagella: number;
  trail: [number, number][];
  speciesId: number;
  carnivore: boolean;
  age: number;
  food: number;
  hunts: number;
  fitness: number;
  size: number;
  speed: number;
  diet: number;
  nodes: number;
  connections: number;
  maturity: number;
  edges: { from: number; to: number; weight: number }[];
}

export interface WorldSnapshot {
  step: number;
  epoch: number;
  foodCount: number;
  scarcity: number;
  speciesCount: number;
  births: number;
  alive: number;
  capacity: number;
  bodies: BodySnapshot[];
  foods: { x: number; y: number; color: [number, number, number]; regrowing: boolean }[];
  eggs: { x: number; y: number }[];
  clouds: { x: number; y: number; radius: number; intensity: number }[];
  scoreboard: { id: string; name: string; fitness: number; carnivore: boolean }[];
  log: string[];
  worldWidth: number;
  worldHeight: number;
}

interface ScoreRecord {
  name: string;
  fitness: number;
  carnivore: boolean;
}

export class LivingWorld {
  readonly seed: number;
  readonly rng: PythonRandom;
  readonly config: GenomeConfig;
  readonly environment: EnvironmentConfig;
  ecology: FoodEcology;
  registry: PopulationRegistry;
  fitness: FitnessTracker;
  selection: SelectionScheduler;
  population: Population;
  step = 0;
  epoch = 0;
  epochLength: number;
  maxSteps: number | null;
  births = 0;
  log: string[] = [];
  eggs: Egg[] = [];
  fitnessCullMinSteps = 1000;
  private pending: Organism[] = [];
  scores = new Map<string, ScoreRecord>();
  private foodGrid: SpatialGrid<Food>;
  private orgGrid: SpatialGrid<Organism>;
  genesis: GenesisConfig;
  band: SpeciesBand;
  sim: SimConfig;

  private constructor(sim: SimConfig, neatText: string, seed: number) {
    this.sim = sim;
    this.seed = seed;
    this.rng = new PythonRandom(seed);
    this.config = parseNeatConfig(neatText);
    this.environment = environmentFrom(sim);
    this.epochLength = Math.max(1, num(sim, "simulation_steps", 2000));
    this.maxSteps = sim.max_world_steps === null || sim.max_world_steps === undefined ? null : Number(sim.max_world_steps);
    this.ecology = new FoodEcology(ecologyFrom(sim), this.rng);
    this.registry = new PopulationRegistry(num(sim, "max_population", 120), this.environment, this.config, this.rng);
    this.fitness = new FitnessTracker(
      num(sim, "fitness_offspring_weight", 150),
      num(sim, "fitness_survival_weight", 0.5),
      num(sim, "fitness_food_weight", 100),
      num(sim, "fitness_kill_weight", 100),
    );
    this.fitnessCullMinSteps = Math.max(0, num(sim, "fitness_cull_min_steps", 1000));
    this.selection = new SelectionScheduler(
      num(sim, "selection_interval_steps", 4000),
      num(sim, "max_population", 120),
      num(sim, "cull_fraction", 0.1),
      num(sim, "immigration_rate", 0.03),
      this.fitnessCullMinSteps,
      this.rng,
    );
    this.band = {
      targetSpecies: num(sim, "target_species_count", 6),
      minSpecies: num(sim, "min_species_count", 4),
      maxSpecies: num(sim, "max_species_count", 9),
    };
    this.genesis = {
      archetypeCount: num(sim, "genesis_archetype_count", 8),
      extraHidden: num(sim, "genesis_extra_hidden_nodes", 0),
      extraConnections: num(sim, "genesis_extra_connections", 0),
      weightJitter: num(sim, "genesis_weight_jitter", 0.05),
      foragingBias: num(sim, "genesis_foraging_bias", 0),
    };
    const cell = Math.max(50, Math.trunc(this.environment.detectionRadius / 2));
    this.foodGrid = new SpatialGrid(cell);
    this.orgGrid = new SpatialGrid(cell);
    this.population = createPopulation(this.config, this.rng);
    bootstrapPopulation(this.population, this.rng, this.genesis);
    this.registry.seed(this.population.genomes, this.population, this);
    for (const organism of this.registry.all()) {
      this.fitness.init(organism.genomeId);
    }
    this.log.push(
      `Living world started: ${this.registry.count()} organisms, world ${this.environment.width}x${this.environment.height}`,
    );
  }

  static create(sim: SimConfig, neatText: string, seed: number): LivingWorld {
    return new LivingWorld(sim, neatText, seed);
  }

  get alive(): boolean {
    return this.maxSteps === null || this.step < this.maxSteps;
  }

  receiveEgg(egg: Egg): void {
    this.eggs.push(egg);
  }

  registerBirth(child: Organism, childGenome: Genome, parentA: Organism, parentB: Organism): void {
    if (this.registry.atCapacity()) {
      const protect = new Set([parentA.genomeId, parentB.genomeId]);
      for (const organism of this.registry.alive()) {
        if (organism.stepsTaken < this.fitnessCullMinSteps) {
          protect.add(organism.genomeId);
        }
      }
      const age = new Map(this.registry.alive().map((organism) => [organism.genomeId, organism.stepsTaken]));
      this.registry.cullWorst((id) => this.fitness.perStep(id, age.get(id) ?? 1), 1, protect);
    }
    const genomeId = this.registry.addBirth(childGenome, child);
    const living = new Set(
      this.registry.alive().map((organism) => organism.genomeId).filter((id) => id !== childGenome.key),
    );
    child.speciesId = assignSpecies(this.population, childGenome, living);
    child.world = this;
    this.fitness.init(genomeId);
    this.fitness.recordOffspring([parentA.genomeId, parentB.genomeId]);
    child.parentIds = [parentA.genomeId, parentB.genomeId];
    this.pending.push(child);
    this.births += 1;
    parentA.fitnessBonus += 100;
    parentB.fitnessBonus += 100;
  }

  /** Advance one tick. Returns false when max_world_steps is reached. */
  stepOnce(): boolean {
    if (!this.alive) {
      return false;
    }
    const organisms = this.registry.alive();
    this.foodGrid.clear();
    this.orgGrid.clear();
    for (const food of this.ecology.foodItems) {
      if (food.position && food.regrowIn <= 0) {
        this.foodGrid.insert(food, food.position);
      }
    }
    for (const organism of organisms) {
      if (organism.position && organism.energy > 0) {
        this.orgGrid.insert(organism, organism.position);
      }
    }
    const sensed = new Map<Organism, [Food[], Organism[], Organism[], Organism[]]>();
    for (const organism of organisms) {
      if (organism.energy <= 0 || !organism.position) {
        continue;
      }
      sensed.set(organism, this.nearby(organism));
    }
    for (const organism of organisms) {
      if (organism.energy <= 0) {
        continue;
      }
      const [food, others, threats, partners] = sensed.get(organism)!;
      organism.takeAction(food, others, threats, partners);
      organism.update(food, others);
      this.fitness.tickAlive(organism);
      if (organism.energy <= 0) {
        this.fitness.finalizeDeath(organism, organism.calculateFitness());
      }
    }
    this.tickEggs();
    this.pending = [];
    this.registry.removeDead();
    this.ecology.tick();
    this.step += 1;
    if (this.selection.shouldRun(this.step)) {
      for (const organism of this.registry.all()) {
        organism.world = this;
      }
      this.selection.run(this.population, this.registry, this.fitness, this, this.band, this.genesis, (line) => {
        this.log.push(line);
      });
    }
    if (this.step % this.epochLength === 0) {
      this.epoch += 1;
      this.recordScoreboard();
      this.log.push(
        `Epoch ${this.epoch} step ${this.step}: ${this.registry.count()} alive, food ${this.ecology.activeFoodCount()}, births ${this.births}`,
      );
    }
    if (this.log.length > 40) {
      this.log.splice(0, this.log.length - 40);
    }
    return this.alive;
  }

  steps(count: number): void {
    for (let i = 0; i < count && this.alive; i += 1) {
      this.stepOnce();
    }
  }

  snapshot(): WorldSnapshot {
    const bodies = this.registry.alive().map((organism) => bodySnapshot(organism));
    const foods = this.ecology.foodItems
      .filter((food) => food.position)
      .map((food) => ({
        x: food.position![0],
        y: food.position![1],
        color: food.strain ? food.strain.pigment() : ([80, 220, 90] as [number, number, number]),
        regrowing: food.regrowIn > 0,
      }));
    const eggs = this.eggs
      .filter((egg) => egg.position)
      .map((egg) => ({ x: egg.position![0], y: egg.position![1] }));
    return {
      step: this.step,
      epoch: this.epoch,
      foodCount: this.ecology.activeFoodCount(),
      scarcity: this.ecology.scarcityRatio(),
      speciesCount: new Set(bodies.map((body) => body.speciesId)).size,
      births: this.births,
      alive: bodies.length,
      capacity: this.registry.maxPopulation,
      bodies,
      foods,
      eggs,
      clouds: this.ecology.clouds.map((cloud) => ({
        x: cloud.position[0],
        y: cloud.position[1],
        radius: cloud.radius,
        intensity: cloud.intensity,
      })),
      scoreboard: [...this.scores.entries()]
        .sort((a, b) => b[1].fitness - a[1].fitness)
        .slice(0, 8)
        .map(([id, record]) => ({ id, name: record.name, fitness: record.fitness, carnivore: record.carnivore })),
      log: this.log.slice(-8),
      worldWidth: this.environment.width,
      worldHeight: this.environment.height,
    };
  }

  /** Compact digest for the golden comparison. */
  digest(): { step: number; food: number; births: number; bodies: { id: number; x: number; y: number; energy: number; species: number }[] } {
    return {
      step: this.step,
      food: this.ecology.activeFoodCount(),
      births: this.births,
      bodies: this.registry
        .alive()
        .map((organism) => ({
          id: organism.genomeId,
          x: round6(organism.position![0]),
          y: round6(organism.position![1]),
          energy: round6(organism.energy),
          species: organism.speciesId ?? -1,
        }))
        .sort((a, b) => a.id - b.id),
    };
  }

  private nearby(organism: Organism): [Food[], Organism[], Organism[], Organism[]] {
    const food = this.foodGrid.query(organism.position!, this.environment.foodDetectionRadius);
    const others = this.orgGrid.query(organism.position!, this.environment.detectionRadius);
    const threats: Organism[] = [];
    const partners: Organism[] = [];
    for (const other of others) {
      if (other === organism || !other.position) {
        continue;
      }
      if (other.isCarnivore && !organism.isCarnivore) {
        if (withinRadius(organism.position!, other.position, this.environment.threatDetectionRadius)) {
          threats.push(other);
        }
      } else if (other.speciesId === organism.speciesId) {
        if (withinRadius(organism.position!, other.position, this.environment.breedingDetectionRadius)) {
          partners.push(other);
        }
      }
    }
    return [food, others, threats, partners];
  }

  private tickEggs(): void {
    const waiting: Egg[] = [];
    for (const egg of this.eggs) {
      if (egg.tick()) {
        if (egg.position) {
          egg.child.position = [egg.position[0], egg.position[1]];
          egg.child.lastPosition = [egg.position[0], egg.position[1]];
        }
        this.registerBirth(egg.child, egg.childGenome, egg.parent, egg.partner);
      } else {
        waiting.push(egg);
      }
    }
    this.eggs = waiting;
  }

  private recordScoreboard(): void {
    const survivors = this.registry.alive();
    if (survivors.length === 0) {
      return;
    }
    const best = survivors.reduce((a, b) => (a.highestFitness >= b.highestFitness ? a : b));
    const id = String(best.speciesId);
    const existing = this.scores.get(id);
    const record: ScoreRecord = {
      name: existing?.name ?? scientificName(this.rng),
      fitness: best.highestFitness,
      carnivore: best.isCarnivore,
    };
    if (!existing || best.highestFitness > existing.fitness) {
      this.scores.set(id, record);
    } else {
      existing.carnivore = best.isCarnivore;
    }
  }
}

function bodySnapshot(organism: Organism): BodySnapshot {
  const traits = BodyTraits.fromGenome(organism.genome, organism.isCarnivore);
  return {
    id: organism.genomeId,
    x: organism.position![0],
    y: organism.position![1],
    radius: organism.getRadius(),
    energy: organism.energy,
    maxEnergy: organism.maxEnergy,
    color: organism.color(),
    jaw: traits.jaw,
    claw: traits.claw,
    horn: traits.horn,
    flagella: traits.flagella,
    trail: organism.movementTrail.map((point) => [point[0], point[1]]),
    speciesId: organism.speciesId ?? 0,
    carnivore: organism.isCarnivore,
    age: organism.stepsTaken,
    food: organism.foodConsumed,
    hunts: organism.organismsConsumed,
    fitness: organism.highestFitness,
    size: organism.size,
    speed: organism.speed,
    diet: organism.genome.bodyGenes?.diet ?? (organism.isCarnivore ? 1 : 0),
    nodes: organism.nodeCount,
    connections: organism.connectionCount,
    maturity: organism.maturity,
    edges: [...organism.genome.connections.values()]
      .filter((connection) => connection.enabled)
      .map((connection) => ({ from: connection.inNode, to: connection.outNode, weight: connection.weight })),
  };
}

function environmentFrom(sim: SimConfig): EnvironmentConfig {
  const detection = num(sim, "detection_radius", 200);
  return {
    width: num(sim, "environment_width", 4000),
    height: num(sim, "environment_height", 4000),
    boundaryPenalty: num(sim, "boundary_penalty", 0.5),
    detectionRadius: detection,
    foodDetectionRadius: num(sim, "food_detection_radius", detection),
    threatDetectionRadius: num(sim, "threat_detection_radius", detection),
    breedingDetectionRadius: num(sim, "breeding_detection_radius", detection),
    startingEnergy: num(sim, "starting_energy", 350),
    foodEnergyValue: num(sim, "food_energy_value", 40),
    movementCost: num(sim, "movement_cost", 0.01),
    breedingCooldown: num(sim, "breeding_cooldown", 200),
    offspringStartingEnergy: num(sim, "offspring_starting_energy", 150),
    eggsEnabled: sim.eggs_enabled !== false,
    eggEnergyGoal: num(sim, "egg_energy_goal", 80),
    eggHatchDelay: num(sim, "egg_hatch_delay", 150),
    hatchMaturity: num(sim, "hatch_maturity", 0.35),
    growthEnergyCost: num(sim, "growth_energy_cost", 0.15),
    growthStep: num(sim, "growth_step", 0.004),
  };
}

function ecologyFrom(sim: SimConfig) {
  return {
    width: num(sim, "environment_width", 4000),
    height: num(sim, "environment_height", 4000),
    foodTarget: num(sim, "food_target_density", 200),
    numFood: num(sim, "num_food_items", 200),
    spawnRate: num(sim, "nutrient_cloud_spawn_rate", 0.04),
    scarcityThreshold: num(sim, "food_scarcity_threshold", 0.35),
    scarcityMultiplier: num(sim, "food_regrowth_scarcity_multiplier", 2.5),
    cloudCount: num(sim, "nutrient_cloud_count", 14),
    cloudMin: num(sim, "nutrient_cloud_min_radius", 180),
    cloudMax: num(sim, "nutrient_cloud_max_radius", 420),
  };
}

function num(sim: SimConfig, key: string, fallback: number): number {
  const value = sim[key];
  return typeof value === "number" ? value : fallback;
}

function round6(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}
