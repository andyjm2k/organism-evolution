/** Uniform grid. Query order matches the Python nested cell loops. */

import { withinRadius } from "./distance";

export interface Positioned {
  position: [number, number] | null;
}

export class SpatialGrid<T extends Positioned> {
  private cells = new Map<string, T[]>();
  private cellSize: number;

  constructor(cellSize: number) {
    this.cellSize = Math.max(1, Math.trunc(cellSize));
  }

  clear(): void {
    this.cells.clear();
  }

  insert(entity: T, position: readonly [number, number]): void {
    const key = this.cellKey(position[0], position[1]);
    const bucket = this.cells.get(key);
    if (bucket) {
      bucket.push(entity);
    } else {
      this.cells.set(key, [entity]);
    }
  }

  /** Entities within radius, in cell-walk order, each entity once. */
  query(position: readonly [number, number], radius: number): T[] {
    if (radius <= 0) {
      return [];
    }
    const span = Math.trunc(Math.ceil(radius / this.cellSize)) + 1;
    const [cx, cy] = this.cellIndex(position[0], position[1]);
    const seen = new Set<T>();
    const found: T[] = [];
    for (let dx = -span; dx <= span; dx += 1) {
      for (let dy = -span; dy <= span; dy += 1) {
        const bucket = this.cells.get(`${cx + dx},${cy + dy}`);
        if (!bucket) {
          continue;
        }
        for (const entity of bucket) {
          if (seen.has(entity) || entity.position === null) {
            continue;
          }
          seen.add(entity);
          if (withinRadius(position as [number, number], entity.position, radius)) {
            found.push(entity);
          }
        }
      }
    }
    return found;
  }

  private cellIndex(x: number, y: number): [number, number] {
    return [Math.trunc(x / this.cellSize), Math.trunc(y / this.cellSize)];
  }

  private cellKey(x: number, y: number): string {
    const [cx, cy] = this.cellIndex(x, y);
    return `${cx},${cy}`;
  }
}
