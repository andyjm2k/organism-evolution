/** Organs read off enabled sensor wires. */

import type { Genome } from "../neat/genome";
import { huntIntake, plantIntake } from "./body-genes";

const VISIBLE = 0.25;
const VELOCITY = new Set([0, 1]);

export class BodyTraits {
  constructor(
    readonly jaw: number,
    readonly claw: number,
    readonly horn: number,
    readonly flagella: number,
  ) {}

  static fromGenome(genome: Genome | null, diet: number): BodyTraits {
    if (!genome) {
      return new BodyTraits(0, 0, 0, 0);
    }
    let jaw = 0;
    let claw = 0;
    let horn = 0;
    let flagella = 0;
    for (const connection of genome.connections.values()) {
      if (!connection.enabled) {
        continue;
      }
      const kind = sensorClass(connection.inNode);
      if (!kind || !drivesBody(connection.outNode)) {
        continue;
      }
      const strength = Math.abs(connection.weight);
      if (kind === "food") {
        jaw += strength;
      } else if (kind === "prey") {
        claw += strength * huntIntake(diet);
        horn += strength * plantIntake(diet);
      } else if (kind === "core") {
        flagella += strength;
      }
    }
    return new BodyTraits(jaw, claw, horn, flagella);
  }

  present(): [string, number][] {
    return (
      [
        ["Jaw", this.jaw],
        ["Claw", this.claw],
        ["Horn", this.horn],
        ["Flagella", this.flagella],
      ] as [string, number][]
    ).filter(([, strength]) => strength >= VISIBLE);
  }
}

export function sensorChannel(nodeKey: number): number | null {
  if (nodeKey >= 0) {
    return null;
  }
  return -nodeKey - 1;
}

function sensorClass(nodeKey: number): "food" | "prey" | "core" | null {
  const channel = sensorChannel(nodeKey);
  if (channel === null) {
    return null;
  }
  if (channel >= 9 && channel <= 15) {
    return "food";
  }
  if (channel >= 16 && channel <= 22) {
    return "prey";
  }
  if (channel >= 0 && channel <= 8) {
    return "core";
  }
  return null;
}

function drivesBody(target: number): boolean {
  return target >= 0 && (VELOCITY.has(target) || target > 3);
}

export const VISIBLE_STRENGTH = VISIBLE;
