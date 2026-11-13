/** Nutrient clouds, pellets, and the bites that change a strain. */

import type { PythonRandom } from "../rng/python-random";
import { plantIntake } from "./body-genes";
import { colliding } from "./distance";
import { makeFounderStrain, PlantStrain } from "./plant";
import type { Organism } from "./organism";

export class Food {
  position: [number, number] | null;
  regrowIn = 0;
  ecology: FoodEcology | null = null;
  cloud: NutrientCloud | null = null;

  constructor(
    x: number,
    y: number,
    readonly strain: PlantStrain | null,
  ) {
    this.position = [x, y];
  }
}

export class NutrientCloud {
  position: [number, number];
  driftAngle: number;
  driftSpeed: number;
  strains: PlantStrain[] = [];

  constructor(
    x: number,
    y: number,
    readonly radius: number,
    readonly intensity: number,
    private worldWidth: number,
    private worldHeight: number,
    rng: PythonRandom,
  ) {
    this.position = [x, y];
    this.driftAngle = rng.uniform(0, 2 * Math.PI);
    this.driftSpeed = rng.uniform(0.15, 0.45);
  }

  tick(rng: PythonRandom): void {
    let x = this.position[0] + Math.cos(this.driftAngle) * this.driftSpeed;
    let y = this.position[1] + Math.sin(this.driftAngle) * this.driftSpeed;
    if (x < 0) {
      x += this.worldWidth;
    } else if (x >= this.worldWidth) {
      x -= this.worldWidth;
    }
    if (y < 0) {
      y += this.worldHeight;
    } else if (y >= this.worldHeight) {
      y -= this.worldHeight;
    }
    this.position = [x, y];
    if (rng.random() < 0.01) {
      this.driftAngle = rng.uniform(0, 2 * Math.PI);
    }
  }
}

export interface EcologyConfig {
  width: number;
  height: number;
  foodTarget: number;
  numFood: number;
  spawnRate: number;
  scarcityThreshold: number;
  scarcityMultiplier: number;
  cloudCount: number;
  cloudMin: number;
  cloudMax: number;
}

export class FoodEcology {
  foodItems: Food[] = [];
  clouds: NutrientCloud[] = [];

  constructor(
    private config: EcologyConfig,
    private rng: PythonRandom,
  ) {
    this.seedClouds();
    this.seedFood();
  }

  activeFoodCount(): number {
    return this.foodItems.filter((food) => food.position).length;
  }

  foodCap(): number {
    return Math.max(1, Math.trunc(this.config.foodTarget * 1.6));
  }

  hasRoom(): boolean {
    return this.activeFoodCount() < this.foodCap();
  }

  scarcityRatio(): number {
    if (this.config.foodTarget <= 0) {
      return 1;
    }
    return this.activeFoodCount() / this.config.foodTarget;
  }

  tick(): void {
    for (const cloud of this.clouds) {
      cloud.tick(this.rng);
    }
    for (const food of this.foodItems) {
      if (food.regrowIn > 0) {
        food.regrowIn -= 1;
      }
    }
    this.prune();
    this.trimToCap();
    if (!this.hasRoom()) {
      this.releaseCrowdedPockets();
    }
    if (!this.hasRoom()) {
      return;
    }
    let boost = 1;
    if (this.scarcityRatio() < this.config.scarcityThreshold) {
      boost = this.config.scarcityMultiplier;
    }
    for (const cloud of this.clouds) {
      const rate = this.config.spawnRate * cloud.intensity * boost;
      if (this.rng.random() < rate && this.hasRoom() && !this.pocketFull(cloud.position)) {
        this.foodItems.push(this.spawnNear(cloud));
      }
    }
  }

  notePlantEvent(food: Food, amount: number): void {
    const strain = food.strain;
    if (!strain || !strain.addSuccess(amount)) {
      return;
    }
    const child = strain.mutate(this.rng);
    if (food.cloud) {
      this.adopt(food.cloud, child);
    }
    if (!this.hasRoom()) {
      return;
    }
    const home = this.homeForSeed(food.cloud);
    if (home && this.pocketFull(home.position)) {
      return;
    }
    this.foodItems.push(this.pellet(home, child));
  }

  prune(): void {
    if (this.foodItems.some((food) => food.position === null)) {
      this.foodItems = this.foodItems.filter((food) => food.position !== null);
    }
  }

  private seedClouds(): void {
    const margin = this.config.cloudMax + 20;
    for (let i = 0; i < this.config.cloudCount; i += 1) {
      const x = this.rng.uniform(margin, this.config.width - margin);
      const y = this.rng.uniform(margin, this.config.height - margin);
      const radius = this.rng.uniform(this.config.cloudMin, this.config.cloudMax);
      const intensity = this.rng.uniform(0.55, 1);
      const cloud = new NutrientCloud(x, y, radius, intensity, this.config.width, this.config.height, this.rng);
      cloud.strains = [makeFounderStrain(this.rng), makeFounderStrain(this.rng)];
      this.clouds.push(cloud);
    }
    if (this.clouds.length) {
      this.clouds[0].strains[0].relation = 0.15;
      this.clouds[this.clouds.length - 1].strains[0].relation = -0.15;
    }
  }

  private seedFood(): void {
    for (let i = 0; i < this.config.numFood; i += 1) {
      if (this.clouds.length && this.rng.random() < 0.7) {
        this.foodItems.push(this.spawnNear(this.rng.choice(this.clouds)));
      } else {
        this.foodItems.push(this.pellet(null, makeFounderStrain(this.rng, 0)));
      }
    }
  }

  private spawnNear(cloud: NutrientCloud, strain?: PlantStrain): Food {
    return this.pellet(cloud, strain ?? this.pickStrain(cloud));
  }

  private pickStrain(cloud: NutrientCloud): PlantStrain {
    const strains = cloud.strains.length ? cloud.strains : [makeFounderStrain(this.rng)];
    const weights = strains.map((strain) => Math.max(0.2, 1 + strain.success));
    return this.rng.choices(strains, weights, 1)[0];
  }

  private pellet(cloud: NutrientCloud | null, strain: PlantStrain): Food {
    let x: number;
    let y: number;
    if (!cloud) {
      const margin = 10;
      x = this.rng.randint(margin, Math.max(margin + 1, this.config.width - margin));
      y = this.rng.randint(margin, Math.max(margin + 1, this.config.height - margin));
    } else {
      const angle = this.rng.uniform(0, 2 * Math.PI);
      const dist = this.rng.uniform(0, cloud.radius * 0.85);
      x = Math.trunc(cloud.position[0] + Math.cos(angle) * dist);
      y = Math.trunc(cloud.position[1] + Math.sin(angle) * dist);
      x = Math.max(5, Math.min(x, this.config.width - 5));
      y = Math.max(5, Math.min(y, this.config.height - 5));
    }
    const food = new Food(x, y, strain);
    food.cloud = cloud;
    food.ecology = this;
    return food;
  }

  private adopt(cloud: NutrientCloud, child: PlantStrain): void {
    if (cloud.strains.length < 3) {
      cloud.strains.push(child);
      return;
    }
    let worst = 0;
    for (let i = 1; i < cloud.strains.length; i += 1) {
      if (cloud.strains[i].success < cloud.strains[worst].success) {
        worst = i;
      }
    }
    cloud.strains.splice(worst, 1);
    cloud.strains.push(child);
  }

  /** A fast-regrowing patch cannot keep the whole food cap. */
  private releaseCrowdedPockets(): void {
    const share = this.pocketShare();
    const groups = new Map<string, Food[]>();
    for (const food of this.foodItems) {
      if (!food.position) {
        continue;
      }
      const key = this.pocketKey(food.position[0], food.position[1]);
      const list = groups.get(key);
      if (list) {
        list.push(food);
      } else {
        groups.set(key, [food]);
      }
    }
    let overflow = false;
    for (const list of groups.values()) {
      if (list.length <= share) {
        continue;
      }
      for (const food of list.slice(share)) {
        food.position = null;
        overflow = true;
      }
    }
    if (overflow) {
      this.prune();
    }
  }

  private pocketFull(position: [number, number]): boolean {
    if (this.activeFoodCount() < this.foodCap() * 0.85) {
      return false;
    }
    return this.countInPocket(position) >= this.pocketShare();
  }

  private homeForSeed(cloud: NutrientCloud | null): NutrientCloud | null {
    if (!cloud || !this.pocketFull(cloud.position)) {
      return cloud;
    }
    const other = this.emptiestCloud();
    if (other && this.countInPocket(other.position) < this.pocketShare()) {
      return other;
    }
    return cloud;
  }

  private emptiestCloud(): NutrientCloud | null {
    let best: NutrientCloud | null = null;
    let bestCount = Infinity;
    for (const cloud of this.clouds) {
      const count = this.countInPocket(cloud.position);
      if (count < bestCount) {
        best = cloud;
        bestCount = count;
      }
    }
    return best;
  }

  private countInPocket(position: [number, number]): number {
    const key = this.pocketKey(position[0], position[1]);
    let count = 0;
    for (const food of this.foodItems) {
      if (!food.position) {
        continue;
      }
      if (this.pocketKey(food.position[0], food.position[1]) === key) {
        count += 1;
      }
    }
    return count;
  }

  private pocketShare(): number {
    return Math.max(4, Math.floor(this.foodCap() / this.pocketCount()));
  }

  private pocketCount(): number {
    const size = this.pocketSize();
    const cols = Math.max(1, Math.ceil(this.config.width / size));
    const rows = Math.max(1, Math.ceil(this.config.height / size));
    return cols * rows;
  }

  private pocketSize(): number {
    return 800;
  }

  private pocketKey(x: number, y: number): string {
    const size = this.pocketSize();
    return `${Math.floor(x / size)},${Math.floor(y / size)}`;
  }

  private trimToCap(): void {
    const cap = this.foodCap();
    let kept = 0;
    let overflow = false;
    for (const food of this.foodItems) {
      if (!food.position) {
        continue;
      }
      kept += 1;
      if (kept > cap) {
        food.position = null;
        overflow = true;
      }
    }
    if (overflow) {
      this.prune();
    }
  }
}

/** One bite of the first ripe pellet. Plant calories fall as diet rises, and stop at diet 1. */
export function graze(organism: Organism, foodItems: Food[]): void {
  if (organism.energy <= 0 || !organism.position) {
    return;
  }
  const intake = plantIntake(organism.diet);
  if (intake <= 0) {
    return;
  }
  const foodEnergy = organism.environment.foodEnergyValue * intake * (organism.morph?.siphon ?? 1);
  for (const food of foodItems) {
    if (!food.position || food.regrowIn > 0) {
      continue;
    }
    if (!colliding(organism.position, organism.getRadius(), food.position, 5)) {
      continue;
    }
    const strain = food.strain;
    const band = strain ? strain.band() : "indifferent";
    if (band === "predatory" && strain) {
      predatoryBite(organism, food, strain, foodEnergy);
    } else if (band === "symbiotic" && strain) {
      symbioticBite(organism, food, strain, foodEnergy);
    } else {
      indifferentBite(organism, food, strain, foodEnergy);
    }
    break;
  }
}

function creditMeal(organism: Organism): void {
  organism.foodConsumed += 1;
  if (organism.stepsSinceLastFood > 0) {
    organism.avgStepsBetweenFood.push(organism.stepsSinceLastFood);
    if (organism.avgStepsBetweenFood.length > 10) {
      organism.avgStepsBetweenFood.shift();
    }
  }
  organism.stepsSinceLastFood = 0;
  organism.fitnessBonus += 10;
}

function pay(organism: Organism, amount: number): void {
  organism.energy = Math.min(organism.maxEnergy, organism.energy + amount);
}

function indifferentBite(organism: Organism, food: Food, strain: PlantStrain | null, foodEnergy: number): void {
  const nutrition = strain ? strain.nutrition : 1;
  pay(organism, foodEnergy * organism.energyEfficiency * nutrition);
  creditMeal(organism);
  food.position = null;
  food.ecology?.notePlantEvent(food, 0.15);
}

function symbioticBite(organism: Organism, food: Food, strain: PlantStrain, foodEnergy: number): void {
  pay(organism, foodEnergy * organism.energyEfficiency * strain.nutrition);
  creditMeal(organism);
  food.regrowIn = Math.trunc(36 / Math.max(strain.regrowth, 0.2));
  food.ecology?.notePlantEvent(food, 1);
}

function predatoryBite(organism: Organism, food: Food, strain: PlantStrain, foodEnergy: number): void {
  organism.health -= 10 * Math.max(strain.strike, 0.3);
  pay(organism, foodEnergy * 0.25 * strain.nutrition);
  creditMeal(organism);
  food.position = null;
  if (organism.health <= 0) {
    organism.health = 0;
    organism.energy = 0;
    food.ecology?.notePlantEvent(food, 2);
  } else {
    food.ecology?.notePlantEvent(food, 0.5);
  }
}
