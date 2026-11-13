/** A bite spends health. Energy stays fuel until a killing bite. */

import { huntIntake, huntPower, huntSizeGate } from "./body-genes";
import { colliding } from "./distance";
import { BodyTraits } from "./body-traits";
import type { Organism } from "./organism";

export function biteDamage(attacker: Organism, defender: Organism): number {
  const traits = BodyTraits.fromGenome(attacker.genome, attacker.diet);
  const weapon = Math.max(traits.claw, traits.jaw * 0.5, 0.8);
  const scale = attacker.size / Math.max(defender.size, 0.15);
  return 18 * weapon * scale * huntPower(attacker.diet);
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

/** First touched body with a lower diet that this hunter is large enough to bite. */
export function huntPrey(organism: Organism, organisms: Organism[]): void {
  if (organism.energy <= 0 || huntPower(organism.diet) <= 0 || !organism.position) {
    return;
  }
  const gate = huntSizeGate(organism.diet);
  for (const other of organisms) {
    if (other === organism || other.diet >= organism.diet || other.energy <= 0 || !other.position) {
      continue;
    }
    if (!colliding(organism.position, organism.getRadius(), other.position, other.getRadius())) {
      continue;
    }
    if (organism.size < other.size * gate) {
      continue;
    }
    const stored = other.energy;
    const killed = applyBite(organism, other);
    if (!killed) {
      break;
    }
    const gained = stored * 0.7 * huntIntake(organism.diet);
    organism.energy = Math.min(organism.maxEnergy, organism.energy + gained);
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
