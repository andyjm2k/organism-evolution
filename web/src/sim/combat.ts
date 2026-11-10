/** A bite spends health. Energy stays fuel until a killing bite. */

import { colliding } from "./distance";
import { BodyTraits } from "./body-traits";
import type { Organism } from "./organism";

export function biteDamage(attacker: Organism, defender: Organism): number {
  const traits = BodyTraits.fromGenome(attacker.genome, attacker.isCarnivore);
  const weapon = Math.max(traits.claw, traits.jaw * 0.5, 0.8);
  const scale = attacker.size / Math.max(defender.size, 0.15);
  return 18 * weapon * scale;
}

export function applyBite(attacker: Organism, defender: Organism): boolean {
  const damage = biteDamage(attacker, defender);
  defender.health -= damage;
  if (defender.health > 0) {
    return false;
  }
  defender.health = 0;
  defender.energy = 0;
  return true;
}

/** First smaller-enough grazer the carnivore is touching. */
export function huntPrey(organism: Organism, organisms: Organism[]): void {
  if (organism.energy <= 0 || !organism.isCarnivore || !organism.position) {
    return;
  }
  for (const other of organisms) {
    if (other === organism || other.isCarnivore || other.energy <= 0 || !other.position) {
      continue;
    }
    if (!colliding(organism.position, organism.getRadius(), other.position, other.getRadius())) {
      continue;
    }
    if (organism.size < other.size * 0.8) {
      continue;
    }
    const stored = other.energy;
    const killed = applyBite(organism, other);
    if (!killed) {
      break;
    }
    organism.energy = Math.min(organism.maxEnergy, organism.energy + stored * 0.7);
    organism.organismsConsumed += 1;
    if (organism.stepsSinceLastHunt > 0) {
      organism.avgStepsBetweenHunts.push(organism.stepsSinceLastHunt);
      if (organism.avgStepsBetweenHunts.length > 10) {
        organism.avgStepsBetweenHunts.shift();
      }
    }
    organism.stepsSinceLastHunt = 0;
    organism.fitnessBonus += 25;
    break;
  }
}
