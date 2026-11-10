/** Plant strains. Relation decides whether a bite feeds, spreads, or bites back. */

import type { PythonRandom } from "../rng/python-random";

export const SYMBIOTIC_AT = 0.35;
export const PREDATORY_AT = -0.35;
const SEED_EVERY = 8;
const MUTATION_VARIANCE = 0.04;

export class PlantStrain {
  success = 0;

  constructor(
    public relation = 0,
    public nutrition = 1,
    public strike = 0.4,
    public regrowth = 1,
  ) {}

  band(): "symbiotic" | "predatory" | "indifferent" {
    if (this.relation >= SYMBIOTIC_AT) {
      return "symbiotic";
    }
    if (this.relation <= PREDATORY_AT) {
      return "predatory";
    }
    return "indifferent";
  }

  pigment(): [number, number, number] {
    if (this.relation >= 0) {
      const blend = Math.min(1, this.relation / Math.max(SYMBIOTIC_AT, 0.01));
      return rgb(90 * (1 - blend) + 40 * blend, 190 + 40 * blend, 80 * (1 - blend) + 50 * blend);
    }
    const blend = Math.min(1, -this.relation / Math.max(-PREDATORY_AT, 0.01));
    return rgb(90 + 140 * blend, 190 * (1 - blend) + 50 * blend, 70 * (1 - blend) + 40 * blend);
  }

  mutate(rng: PythonRandom, variance = MUTATION_VARIANCE): PlantStrain {
    const spread = variance;
    return new PlantStrain(
      clamp(this.relation + rng.gauss(0, spread), -1, 1),
      clamp(this.nutrition + rng.gauss(0, spread), 0.4, 1.6),
      clamp(this.strike + rng.gauss(0, spread), 0.1, 2.5),
      clamp(this.regrowth + rng.gauss(0, spread * 0.5), 0.4, 2),
    );
  }

  /** Returns true when the strain has earned a seed. */
  addSuccess(amount: number): boolean {
    this.success += amount;
    if (this.success < SEED_EVERY) {
      return false;
    }
    this.success = 0;
    return true;
  }
}

export function makeFounderStrain(rng: PythonRandom, bias = 0): PlantStrain {
  const relation = clamp(bias + rng.gauss(0, 0.05), -0.3, 0.3);
  return new PlantStrain(relation, 1, 0.4, 1);
}

function rgb(red: number, green: number, blue: number): [number, number, number] {
  return [clamp(red, 0, 255), clamp(green, 0, 255), clamp(blue, 0, 255)].map((channel) => Math.trunc(channel)) as [
    number,
    number,
    number,
  ];
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}
