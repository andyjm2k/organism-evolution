/** Lifetime selection fitness. Meals and kills score once, not every step. */

import type { Genome } from "../neat/genome";
import type { Organism } from "./organism";

export class FitnessTracker {
  private scores = new Map<number, number>();
  private offspring = new Map<number, number>();
  private meals = new Map<number, number>();
  private kills = new Map<number, number>();

  constructor(
    private offspringWeight: number,
    private survivalWeight: number,
    private foodWeight: number,
    private killWeight: number,
  ) {}

  init(genomeId: number): void {
    if (!this.scores.has(genomeId)) {
      this.scores.set(genomeId, 0);
    }
    if (!this.offspring.has(genomeId)) {
      this.offspring.set(genomeId, 0);
    }
  }

  get(genomeId: number): number {
    return this.scores.get(genomeId) ?? 0;
  }

  /** Score divided by age, so a long life of merely surviving does not outrank a short successful one. */
  perStep(genomeId: number, stepsTaken: number): number {
    return this.get(genomeId) / Math.max(1, stepsTaken);
  }

  recordOffspring(parentIds: (number | null)[]): void {
    for (const genomeId of parentIds) {
      if (genomeId === null) {
        continue;
      }
      this.init(genomeId);
      this.offspring.set(genomeId, (this.offspring.get(genomeId) ?? 0) + 1);
      this.scores.set(genomeId, this.get(genomeId) + this.offspringWeight);
    }
  }

  tickAlive(organism: Organism): void {
    const genomeId = organism.genomeId;
    this.init(genomeId);
    this.scores.set(genomeId, this.get(genomeId) + this.survivalWeight);
    if (organism.isCarnivore) {
      const total = organism.organismsConsumed;
      const gained = Math.max(0, total - (this.kills.get(genomeId) ?? 0));
      this.kills.set(genomeId, total);
      this.scores.set(genomeId, this.get(genomeId) + this.killWeight * gained);
    } else {
      const total = organism.foodConsumed;
      const gained = Math.max(0, total - (this.meals.get(genomeId) ?? 0));
      this.meals.set(genomeId, total);
      this.scores.set(genomeId, this.get(genomeId) + this.foodWeight * gained);
    }
  }

  finalizeDeath(organism: Organism, displayFitness: number): void {
    this.init(organism.genomeId);
    this.scores.set(organism.genomeId, this.get(organism.genomeId) + displayFitness * 0.05);
  }

  apply(genomes: Map<number, Genome>): void {
    for (const [genomeId, genome] of genomes) {
      genome.fitness = this.get(genomeId);
    }
  }

  exportState(): { scores: [number, number][]; offspring: [number, number][]; meals: [number, number][]; kills: [number, number][] } {
    return {
      scores: [...this.scores.entries()],
      offspring: [...this.offspring.entries()],
      meals: [...this.meals.entries()],
      kills: [...this.kills.entries()],
    };
  }

  importState(state: { scores: [number, number][]; offspring: [number, number][]; meals: [number, number][]; kills: [number, number][] }): void {
    this.scores = new Map(state.scores);
    this.offspring = new Map(state.offspring);
    this.meals = new Map(state.meals);
    this.kills = new Map(state.kills);
  }

  top(limit: number): number[] {
    return [...this.scores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([genomeId]) => genomeId);
  }
}
