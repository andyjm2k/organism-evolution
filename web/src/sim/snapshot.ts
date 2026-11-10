/** JSON save of a running world. Restore does not replay ticks. */

import type { ConnectionGene, Genome, NodeGene } from "../neat/genome";
import { Genome as GenomeClass } from "../neat/genome";
import type { LivingWorld, SimConfig } from "./world";
import { LivingWorld as WorldClass } from "./world";
import { Organism } from "./organism";
import { Food, NutrientCloud } from "./food-ecology";
import { PlantStrain } from "./plant";
import { Egg } from "./egg";
import { applyMaturity } from "./growth";

const VERSION = 1;

export interface SaveFile {
  version: number;
  seed: number;
  sim: SimConfig;
  state: Record<string, unknown>;
}

export function captureWorld(world: LivingWorld): SaveFile {
  const genomes = new Map<number, Genome>();
  for (const genome of world.population.genomes.values()) {
    genomes.set(genome.key, genome);
  }
  for (const organism of world.registry.all()) {
    genomes.set(organism.genome.key, organism.genome);
  }
  for (const egg of world.eggs) {
    genomes.set(egg.childGenome.key, egg.childGenome);
  }
  return {
    version: VERSION,
    seed: world.seed,
    sim: world.sim,
    state: {
      rng: world.rng.exportState(),
      step: world.step,
      epoch: world.epoch,
      births: world.births,
      log: world.log,
      scores: [...world.scores.entries()],
      nextNodeId: world.config.nextNodeId,
      innovation: world.config.innovation.exportState(),
      nextSpeciesId: world.population.species.nextSpeciesId,
      threshold: world.population.species.threshold,
      baseline: world.population.species.baseline,
      generation: world.population.generation,
      species: [...world.population.species.species.entries()].map(([id, species]) => ({
        id,
        representative: species.representative?.key ?? null,
        members: [...species.members.keys()],
      })),
      genomeToSpecies: [...world.population.species.genomeToSpecies.entries()],
      populationIds: [...world.population.genomes.keys()],
      genomes: [...genomes.values()].map(serializeGenome),
      organisms: world.registry.all().map(serializeOrganism),
      nextId: world.registry.peekNextId(),
      fitness: world.fitness.exportState(),
      clouds: world.ecology.clouds.map((cloud) => ({
        x: cloud.position[0],
        y: cloud.position[1],
        radius: cloud.radius,
        intensity: cloud.intensity,
        driftAngle: cloud.driftAngle,
        driftSpeed: cloud.driftSpeed,
        strains: cloud.strains.map(serializeStrain),
      })),
      foods: world.ecology.foodItems.map((food) => ({
        x: food.position?.[0] ?? null,
        y: food.position?.[1] ?? null,
        regrowIn: food.regrowIn,
        cloud: food.cloud ? world.ecology.clouds.indexOf(food.cloud) : -1,
        strain: food.strain ? serializeStrain(food.strain) : null,
      })),
      eggs: world.eggs.map((egg) => ({
        parentId: egg.parent.genomeId,
        partnerId: egg.partner.genomeId,
        childId: egg.child.genomeId || egg.childGenome.key,
        stored: egg.stored,
        laid: egg.laid,
        stepsLaid: egg.stepsLaid,
        x: egg.position?.[0] ?? null,
        y: egg.position?.[1] ?? null,
        energyGoal: egg.energyGoal,
        hatchDelay: egg.hatchDelay,
        child: serializeOrganism(egg.child),
      })),
    },
  };
}

export function restoreWorld(file: SaveFile, neatText: string): LivingWorld {
  if (file.version !== VERSION || !file.state || typeof file.sim !== "object") {
    throw new Error("Unrecognized save");
  }
  const world = WorldClass.create(sanitizeSim(file.sim), neatText, finite(file.seed, 1, 1, 1_000_000_000));
  const state = file.state;
  const genomes = new Map<number, Genome>();
  for (const raw of state.genomes as ReturnType<typeof serializeGenome>[]) {
    genomes.set(raw.key, deserializeGenome(raw));
  }
  world.population.genomes = new Map();
  for (const id of state.populationIds as number[]) {
    const genome = genomes.get(id);
    if (genome) {
      world.population.genomes.set(id, genome);
    }
  }
  world.config.nextNodeId = (state.nextNodeId as number | null) ?? null;
  world.config.innovation.importState(state.innovation as { globalCounter: number; generation: [string, number][] });
  world.population.species.genomeToSpecies = new Map(state.genomeToSpecies as [number, number][]);
  world.population.species.nextSpeciesId = finite(state.nextSpeciesId, 1, 1, 1_000_000);
  world.population.species.threshold = Number(state.threshold) || world.population.species.threshold;
  world.population.species.baseline = state.baseline === null ? null : Number(state.baseline);
  world.population.generation = finite(state.generation, 0, 0, 1_000_000);
  rebuildSpecies(world, state.species as { id: number; representative: number | null; members: number[] }[], genomes);
  const organisms = new Map<number, Organism>();
  for (const raw of state.organisms as ReturnType<typeof serializeOrganism>[]) {
    const genome = genomes.get(raw.genomeKey);
    if (!genome) {
      continue;
    }
    organisms.set(raw.genomeId, rebuildOrganism(world, genome, raw));
  }
  world.registry.replaceAll(organisms, finite(state.nextId, organisms.size + 1, 1, 10_000_000));
  world.fitness.importState(state.fitness as ReturnType<LivingWorld["fitness"]["exportState"]>);
  world.ecology.clouds = (state.clouds as CloudSave[]).map(
    (cloud) =>
      new NutrientCloud(
        cloud.x,
        cloud.y,
        cloud.radius,
        cloud.intensity,
        world.environment.width,
        world.environment.height,
        world.rng,
      ),
  );
  // NutrientCloud constructor draws RNG. Re-apply saved drift and then restore the RNG.
  world.ecology.clouds.forEach((cloud, index) => {
    const saved = (state.clouds as CloudSave[])[index];
    cloud.position = [saved.x, saved.y];
    cloud.driftAngle = saved.driftAngle;
    cloud.driftSpeed = saved.driftSpeed;
    cloud.strains = saved.strains.map(deserializeStrain);
  });
  world.ecology.foodItems = (state.foods as FoodSave[]).map((food) => {
    const item = new Food(food.x ?? 0, food.y ?? 0, food.strain ? deserializeStrain(food.strain) : null);
    item.position = food.x === null ? null : [food.x, food.y ?? 0];
    item.regrowIn = food.regrowIn;
    item.cloud = food.cloud >= 0 ? world.ecology.clouds[food.cloud] ?? null : null;
    item.ecology = world.ecology;
    return item;
  });
  const byId = new Map(world.registry.all().map((organism) => [organism.genomeId, organism]));
  world.eggs = [];
  for (const raw of (state.eggs as EggSave[]) ?? []) {
    const parent = byId.get(raw.parentId);
    const partner = byId.get(raw.partnerId);
    const childGenome = genomes.get(raw.child.genomeKey);
    if (!parent || !partner || !childGenome) {
      continue;
    }
    const child = rebuildOrganism(world, childGenome, raw.child);
    const egg = new Egg(parent, partner, child, childGenome, raw.energyGoal, raw.hatchDelay);
    egg.stored = raw.stored;
    egg.laid = raw.laid;
    egg.stepsLaid = raw.stepsLaid;
    egg.position = raw.x === null ? null : [raw.x, raw.y ?? 0];
    if (!egg.laid) {
      parent.carriedEgg = egg;
    }
    world.eggs.push(egg);
  }
  world.step = finite(state.step, 0, 0, 50_000_000);
  world.epoch = finite(state.epoch, 0, 0, 50_000_000);
  world.births = finite(state.births, 0, 0, 50_000_000);
  world.log = Array.isArray(state.log) ? (state.log as string[]).slice(-40).map((line) => String(line).slice(0, 240)) : [];
  world.scores = new Map(state.scores as [string, { name: string; fitness: number; carnivore: boolean }][]);
  world.rng.importState(state.rng as { mt: number[]; index: number; gaussNext: number | null });
  return world;
}

function rebuildSpecies(
  world: LivingWorld,
  rows: { id: number; representative: number | null; members: number[] }[],
  genomes: Map<number, Genome>,
): void {
  world.population.species.species.clear();
  const { Species } = speciesCtor();
  for (const row of rows) {
    const species = new Species(row.id);
    const members = new Map<number, Genome>();
    for (const id of row.members) {
      const genome = genomes.get(id);
      if (genome) {
        members.set(id, genome);
      }
    }
    species.members = members;
    species.representative = row.representative === null ? null : genomes.get(row.representative) ?? null;
    world.population.species.species.set(row.id, species);
  }
}

function speciesCtor(): { Species: typeof import("../neat/species").Species } {
  return { Species: SpeciesRef };
}

import { Species as SpeciesRef } from "../neat/species";

function rebuildOrganism(world: LivingWorld, genome: Genome, raw: ReturnType<typeof serializeOrganism>): Organism {
  const organism = new Organism(
    genome,
    world.config,
    [raw.x, raw.y],
    world.environment,
    world.rng,
    raw.speciesId,
    raw.dietKey,
  );
  organism.genomeId = raw.genomeId;
  organism.world = world;
  organism.energy = raw.energy;
  organism.health = raw.health;
  organism.maturity = raw.maturity;
  organism.stepsTaken = raw.stepsTaken;
  organism.foodConsumed = raw.foodConsumed;
  organism.organismsConsumed = raw.organismsConsumed;
  organism.stepsSinceBreeding = raw.stepsSinceBreeding;
  organism.stepsSinceLastFood = raw.stepsSinceLastFood;
  organism.stepsSinceLastHunt = raw.stepsSinceLastHunt;
  organism.fitnessBonus = raw.fitnessBonus;
  organism.highestFitness = raw.highestFitness;
  organism.movementTrail = raw.trail.map((point) => [point[0], point[1]]);
  organism.lastPosition = [raw.lastX, raw.lastY];
  applyMaturity(organism);
  organism.energy = raw.energy;
  organism.health = raw.health;
  return organism;
}

function serializeGenome(genome: Genome) {
  return {
    key: genome.key,
    fitness: genome.fitness,
    dietKey: genome.dietKey,
    bodyGenes: genome.bodyGenes,
    nodes: [...genome.nodes.values()],
    connections: [...genome.connections.values()],
  };
}

function deserializeGenome(raw: ReturnType<typeof serializeGenome>): Genome {
  const genome = new GenomeClass(raw.key);
  genome.fitness = raw.fitness;
  genome.dietKey = raw.dietKey;
  genome.bodyGenes = raw.bodyGenes;
  for (const node of raw.nodes as NodeGene[]) {
    genome.nodes.set(node.key, { ...node });
  }
  for (const connection of raw.connections as ConnectionGene[]) {
    genome.connections.set(`${connection.inNode},${connection.outNode}`, { ...connection });
  }
  return genome;
}

function serializeOrganism(organism: Organism) {
  return {
    genomeId: organism.genomeId,
    genomeKey: organism.genome.key,
    speciesId: organism.speciesId,
    dietKey: organism.dietKey,
    x: organism.position?.[0] ?? 0,
    y: organism.position?.[1] ?? 0,
    lastX: organism.lastPosition[0],
    lastY: organism.lastPosition[1],
    energy: organism.energy,
    health: organism.health,
    maturity: organism.maturity,
    stepsTaken: organism.stepsTaken,
    foodConsumed: organism.foodConsumed,
    organismsConsumed: organism.organismsConsumed,
    stepsSinceBreeding: organism.stepsSinceBreeding,
    stepsSinceLastFood: organism.stepsSinceLastFood,
    stepsSinceLastHunt: organism.stepsSinceLastHunt,
    fitnessBonus: organism.fitnessBonus,
    highestFitness: organism.highestFitness,
    trail: organism.movementTrail,
  };
}

function serializeStrain(strain: PlantStrain) {
  return {
    relation: strain.relation,
    nutrition: strain.nutrition,
    strike: strain.strike,
    regrowth: strain.regrowth,
    success: strain.success,
  };
}

function deserializeStrain(raw: ReturnType<typeof serializeStrain>): PlantStrain {
  const strain = new PlantStrain(raw.relation, raw.nutrition, raw.strike, raw.regrowth);
  strain.success = raw.success;
  return strain;
}

interface CloudSave {
  x: number;
  y: number;
  radius: number;
  intensity: number;
  driftAngle: number;
  driftSpeed: number;
  strains: ReturnType<typeof serializeStrain>[];
}

interface FoodSave {
  x: number | null;
  y: number | null;
  regrowIn: number;
  cloud: number;
  strain: ReturnType<typeof serializeStrain> | null;
}

interface EggSave {
  parentId: number;
  partnerId: number;
  stored: number;
  laid: boolean;
  stepsLaid: number;
  x: number | null;
  y: number | null;
  energyGoal: number;
  hatchDelay: number;
  child: ReturnType<typeof serializeOrganism>;
}

function sanitizeSim(sim: SimConfig): SimConfig {
  return {
    ...sim,
    max_population: finite(sim.max_population, 120, 4, 400),
    num_food_items: finite(sim.num_food_items, 200, 1, 2000),
    food_target_density: finite(sim.food_target_density, 200, 1, 2000),
    nutrient_cloud_count: finite(sim.nutrient_cloud_count, 14, 1, 40),
    environment_width: finite(sim.environment_width, 4000, 200, 8000),
    environment_height: finite(sim.environment_height, 4000, 200, 8000),
    breeding_cooldown: finite(sim.breeding_cooldown, 200, 1, 100_000),
  };
}

function finite(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}
