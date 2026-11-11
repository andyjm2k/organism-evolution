/** Speciation for the living genomes, matching neat.DefaultSpeciesSet.speciate. */

import type { GenomeConfig } from "./config";
import type { Genome } from "./genome";

export class Species {
  key: number;
  representative: Genome | null = null;
  members = new Map<number, Genome>();

  constructor(key: number) {
    this.key = key;
  }
}

export class SpeciesSet {
  species = new Map<number, Species>();
  genomeToSpecies = new Map<number, number>();
  /** count(1) in neat. Advances during the discarded initial speciate and stays there. */
  nextSpeciesId = 1;
  /** Ini threshold remembered so selection can raise it and later relax. */
  baseline: number | null = null;
  threshold: number;

  constructor(threshold: number) {
    this.threshold = threshold;
  }

  getSpeciesId(genomeId: number): number {
    const id = this.genomeToSpecies.get(genomeId);
    if (id === undefined) {
      throw new Error(`genome ${genomeId} has no species`);
    }
    return id;
  }

  /** Partition genomes. Existing representatives are replaced from the new set. */
  speciate(config: GenomeConfig, population: Map<number, Genome>, _generation: number): void {
    const threshold = this.threshold;
    const unspeciated = [...population.keys()].sort((a, b) => a - b);
    const distances = new DistanceCache(config);
    const newReps = new Map<number, number>();
    const newMembers = new Map<number, number[]>();
    const speciesIds = [...this.species.keys()].sort((a, b) => a - b);
    for (const sid of speciesIds) {
      const species = this.species.get(sid)!;
      let bestDistance = Infinity;
      let best: Genome | null = null;
      for (const gid of unspeciated) {
        const genome = population.get(gid)!;
        const distance = distances.get(species.representative!, genome);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = genome;
        }
      }
      if (!best) {
        continue;
      }
      newReps.set(sid, best.key);
      newMembers.set(sid, [best.key]);
      const index = unspeciated.indexOf(best.key);
      unspeciated.splice(index, 1);
    }
    while (unspeciated.length) {
      const gid = unspeciated.shift()!;
      const genome = population.get(gid)!;
      let bestDistance = Infinity;
      let bestSid: number | null = null;
      for (const [sid, rid] of newReps) {
        const distance = distances.get(population.get(rid)!, genome);
        if (distance < threshold && distance < bestDistance) {
          bestDistance = distance;
          bestSid = sid;
        }
      }
      if (bestSid !== null) {
        newMembers.get(bestSid)!.push(gid);
      } else {
        const sid = this.nextSpeciesId;
        this.nextSpeciesId += 1;
        newReps.set(sid, gid);
        newMembers.set(sid, [gid]);
      }
    }
    this.genomeToSpecies = new Map();
    const rebuilt = new Map<number, Species>();
    const ordered = [...newReps.keys()].sort((a, b) => a - b);
    for (const sid of ordered) {
      const rid = newReps.get(sid)!;
      let species = this.species.get(sid) ?? null;
      if (!species) {
        species = new Species(sid);
      }
      const memberIds = newMembers.get(sid)!;
      const members = new Map<number, Genome>();
      for (const gid of memberIds) {
        this.genomeToSpecies.set(gid, sid);
        members.set(gid, population.get(gid)!);
      }
      species.representative = population.get(rid)!;
      species.members = members;
      species.key = sid;
      rebuilt.set(sid, species);
    }
    this.species = rebuilt;
  }
}

class DistanceCache {
  private distances = new Map<string, number>();

  constructor(private config: GenomeConfig) {}

  /** Cache both directions with the first computed value, as neat does. */
  get(a: Genome, b: Genome): number {
    const key = `${a.key}|${b.key}`;
    const cached = this.distances.get(key);
    if (cached !== undefined) {
      return cached;
    }
    const distance = a.distance(b, this.config);
    this.distances.set(key, distance);
    this.distances.set(`${b.key}|${a.key}`, distance);
    return distance;
  }
}
