/** Hatchlings start small and spend spare energy to reach the adult body. */

import type { Organism } from "./organism";

export function bindAdultBody(organism: Organism): void {
  organism.adultSize = organism.size;
  organism.adultMaxEnergy = organism.maxEnergy;
  const armor = organism.morph?.armor ?? 0;
  organism.adultMaxHealth = 90 * Math.max(organism.adultSize, 0.25) * (1 + armor * 0.8);
  const firstBody = !organism.healthBound;
  applyMaturity(organism);
  if (firstBody) {
    organism.health = organism.maxHealth;
    organism.healthBound = true;
  }
}

export function applyMaturity(organism: Organism): void {
  const maturity = Math.min(1, Math.max(0.2, organism.maturity));
  organism.maturity = maturity;
  organism.size = organism.adultSize * maturity;
  organism.maxEnergy = organism.adultMaxEnergy * maturity;
  organism.maxHealth = organism.adultMaxHealth * maturity;
  organism.energy = Math.min(organism.energy, organism.maxEnergy);
  if (organism.health !== undefined) {
    organism.health = Math.min(organism.health, organism.maxHealth);
  }
  organism.baseRadius = (4 + Math.min(8, organism.nodeCount / 10)) * organism.baseRadiusScale * maturity;
}

export function growStep(organism: Organism): void {
  if (organism.energy <= 0 || organism.maturity >= 1) {
    return;
  }
  const ratio = organism.energy / Math.max(organism.maxEnergy, 1);
  if (ratio < 0.55) {
    return;
  }
  const cost = organism.environment.growthEnergyCost;
  if (organism.energy <= cost + 20) {
    return;
  }
  organism.energy -= cost;
  organism.maturity = Math.min(1, organism.maturity + organism.environment.growthStep);
  applyMaturity(organism);
}
