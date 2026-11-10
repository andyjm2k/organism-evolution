/** One body: network, movement, meals, and breeding gates. */

import type { PythonRandom } from "../rng/python-random";
import { Genome } from "../neat/genome";
import type { GenomeConfig } from "../neat/config";
import { CompiledNetwork } from "../neat/network";
import { geneColor, isCarnivoreDiet } from "./body-genes";
import { buildNetworkInputs } from "./network-inputs";
import { withinRadius } from "./distance";
import { calculateFitness } from "./fitness";
import { fertilize, tendEgg, type Egg } from "./egg";
import { graze, type Food } from "./food-ecology";
import { huntPrey } from "./combat";
import { bindAdultBody, growStep } from "./growth";
import type { LivingWorld } from "./world";

export interface EnvironmentConfig {
  width: number;
  height: number;
  boundaryPenalty: number;
  detectionRadius: number;
  foodDetectionRadius: number;
  threatDetectionRadius: number;
  breedingDetectionRadius: number;
  startingEnergy: number;
  foodEnergyValue: number;
  movementCost: number;
  breedingCooldown: number;
  offspringStartingEnergy: number;
  eggsEnabled: boolean;
  eggEnergyGoal: number;
  eggHatchDelay: number;
  hatchMaturity: number;
  growthEnergyCost: number;
  growthStep: number;
}

export class Organism {
  genome: Genome;
  network: CompiledNetwork;
  position: [number, number] | null;
  lastPosition: [number, number];
  speciesId: number | null;
  fitnessBonus = 0;
  highestFitness = 0;
  world: LivingWorld | null = null;
  stepsTaken = 0;
  foodConsumed = 0;
  organismsConsumed = 0;
  wasMoving = false;
  stepsSinceBreeding = 0;
  stepsSinceLastFood = 0;
  stepsSinceLastHunt = 0;
  avgStepsBetweenFood: number[] = [];
  avgStepsBetweenHunts: number[] = [];
  lastPositions: [number, number][] = [];
  movementTrail: [number, number][] = [];
  carriedEgg: Egg | null = null;
  isCarnivore: boolean;
  dietKey: number;
  nodeCount = 0;
  connectionCount = 0;
  hiddenCount = 0;
  speed = 1;
  size = 1;
  maxEnergy = 1;
  energy = 1;
  energyEfficiency = 1;
  movementEnergyCost = 0;
  baseRadius = 4;
  baseRadiusScale = 1;
  adultSize = 1;
  adultMaxEnergy = 1;
  adultMaxHealth = 90;
  maturity = 1;
  health = 90;
  maxHealth = 90;
  healthBound = false;
  genomeId = 0;
  parentIds: [number | null, number | null] | null = null;
  private startingEnergy = 1;

  constructor(
    genome: Genome,
    readonly config: GenomeConfig,
    position: [number, number],
    readonly environment: EnvironmentConfig,
    readonly rng: PythonRandom,
    speciesId: number | null,
    dietKey: number | null,
  ) {
    this.genome = genome;
    this.network = CompiledNetwork.fromGenome(genome, config);
    this.position = position;
    this.lastPosition = [position[0], position[1]];
    this.speciesId = speciesId;
    this.dietKey = dietKey ?? speciesId ?? 1;
    this.isCarnivore = dietFromKey(this.dietKey);
    const genes = genome.bodyGenes;
    if (genes) {
      this.isCarnivore = isCarnivoreDiet(genes.diet);
      this.dietKey = this.isCarnivore ? 3 : 1;
    }
    this.recomputeAttributes();
    this.energy = this.startingEnergy;
  }

  /** Rebuild speed, size, and the adult body from the current genome. */
  recomputeAttributes(): void {
    this.calculateAttributes();
  }

  color(): [number, number, number] {
    if (this.genome.bodyGenes) {
      return geneColor(this.genome.bodyGenes);
    }
    return this.isCarnivore ? [180, 80, 60] : [70, 160, 90];
  }

  getRadius(): number {
    return Math.max(2, Math.min(20, this.baseRadius));
  }

  canBreed(): boolean {
    if (this.energy < 100 || this.maturity < 0.99 || this.carriedEgg || !this.position) {
      return false;
    }
    if (this.foodConsumed <= 0 && this.organismsConsumed <= 0) {
      return false;
    }
    if (this.stepsSinceBreeding < this.environment.breedingCooldown) {
      return false;
    }
    const minDistance = Math.min(this.environment.width, this.environment.height) * 0.02;
    const edge = Math.min(
      this.position[0],
      this.position[1],
      this.environment.width - this.position[0],
      this.environment.height - this.position[1],
    );
    return edge >= minDistance;
  }

  takeAction(nearbyFood: Food[], nearbyOrganisms: Organism[], threats: Organism[], partners: Organism[]): void {
    if (this.energy <= 0 || !this.position) {
      return;
    }
    this.lastPosition = [this.position[0], this.position[1]];
    const inputs = buildNetworkInputs(this, nearbyFood, nearbyOrganisms, threats, partners, this.environment);
    const output = this.network.activate(inputs);
    const vx = output[0];
    const vy = output[1];
    const breed = output[2];
    const rest = output.length > 3 ? output[3] : -1;
    let moveX = 0;
    let moveY = 0;
    if (rest <= 0.5) {
      moveX = this.speed * vx + this.rng.uniform(-0.01, 0.01) * this.speed;
      moveY = this.speed * vy + this.rng.uniform(-0.01, 0.01) * this.speed;
    }
    const newX = Math.max(0, Math.min(this.position[0] + moveX, this.environment.width));
    const newY = Math.max(0, Math.min(this.position[1] + moveY, this.environment.height));
    if (moveX !== 0 || moveY !== 0) {
      this.position = [newX, newY];
      this.wasMoving = true;
      this.pushTrail(this.position);
      this.energy = Math.max(0, this.energy - this.movementEnergyCost);
    } else {
      this.wasMoving = false;
    }
    if (this.carriedEgg) {
      tendEgg(this, breed);
    } else if (breed > 0.5 && this.canBreed()) {
      this.tryBreed(partners.length ? partners : nearbyOrganisms);
    }
  }

  update(food: Food[], organisms: Organism[]): void {
    this.stepsTaken += 1;
    this.stepsSinceBreeding += 1;
    this.stepsSinceLastFood += 1;
    this.stepsSinceLastHunt += 1;
    this.energy = Math.min(this.energy, this.maxEnergy);
    if (!this.position) {
      return;
    }
    this.fitnessBonus += 0.05;
    this.energy -= 0.01 * this.size;
    growStep(this);
    if (this.health <= 0) {
      this.energy = 0;
    }
    const cell: [number, number] = [Math.round(this.position[0] / 10) * 10, Math.round(this.position[1] / 10) * 10];
    this.lastPositions.push(cell);
    if (this.lastPositions.length > 100) {
      this.lastPositions.shift();
    }
    this.applyBoundary();
    graze(this, food);
    if (this.isCarnivore) {
      huntPrey(this, organisms);
    }
  }

  calculateFitness(): number {
    return calculateFitness(this, this.environment.width, this.environment.height, this.rng);
  }

  private tryBreed(candidates: Organism[]): void {
    const partners: Organism[] = [];
    for (const other of candidates) {
      if (other === this || other.speciesId !== this.speciesId || !other.canBreed() || !other.position || !this.position) {
        continue;
      }
      if (!withinRadius(this.position, other.position, this.environment.breedingDetectionRadius)) {
        continue;
      }
      partners.push(other);
    }
    if (partners.length === 0) {
      return;
    }
    fertilize(this, partners[0]);
  }

  private pushTrail(position: [number, number]): void {
    const last = this.movementTrail[this.movementTrail.length - 1];
    if (!last || last[0] !== position[0] || last[1] !== position[1]) {
      this.movementTrail.push([position[0], position[1]]);
    }
    if (this.movementTrail.length > 24) {
      this.movementTrail.shift();
    }
  }

  private applyBoundary(): void {
    if (!this.position) {
      return;
    }
    let [x, y] = this.position;
    let wrapped = false;
    if (x < 0) {
      x = this.environment.width;
      wrapped = true;
    } else if (x > this.environment.width) {
      x = 0;
      wrapped = true;
    }
    if (y < 0) {
      y = this.environment.height;
      wrapped = true;
    } else if (y > this.environment.height) {
      y = 0;
      wrapped = true;
    }
    if (wrapped) {
      this.energy *= 1 - this.environment.boundaryPenalty;
    }
    this.position = [x, y];
  }

  private calculateAttributes(): void {
    this.nodeCount = this.genome.nodes.size;
    this.connectionCount = this.genome.connections.size;
    this.hiddenCount = [...this.genome.nodes.keys()].filter((key) => key > 0).length;
    const baseSpeed = this.isCarnivore ? 1.2 : 1.3;
    const baseSize = this.isCarnivore ? 1.1 : 1;
    const baseEnergy = this.isCarnivore ? 300 : 250;
    const baseEfficiency = this.isCarnivore ? 1.5 : 1.8;
    this.speed = baseSpeed * Math.min(4, Math.max(0.25, 0.25 + this.nodeCount / 10));
    this.size = baseSize * Math.min(4, Math.max(0.25, 0.25 + this.connectionCount / 10));
    const genes = this.genome.bodyGenes;
    if (genes) {
      this.size *= genes.size;
      this.baseRadiusScale = genes.size;
    } else {
      this.baseRadiusScale = 1;
    }
    this.maxEnergy = baseEnergy * this.size * 3;
    this.startingEnergy = Math.min(this.environment.startingEnergy, this.maxEnergy);
    this.energyEfficiency = baseEfficiency * (2 - this.size * 0.1);
    this.movementEnergyCost = this.environment.movementCost * this.speed ** 1.1 * this.size ** 0.5;
    this.baseRadius = (4 + Math.min(8, this.nodeCount / 10)) * this.baseRadiusScale;
    bindAdultBody(this);
  }
}

function dietFromKey(speciesId: number): boolean {
  return Math.abs(Math.trunc(speciesId)) % 4 >= 2;
}
