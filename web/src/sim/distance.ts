/** Squared-distance helpers. Comparisons stay in squared units. */

export function squaredDistance(a: readonly [number, number], b: readonly [number, number]): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

export function linearDistance(a: readonly [number, number], b: readonly [number, number]): number {
  return Math.sqrt(squaredDistance(a, b));
}

export function withinRadius(
  a: readonly [number, number],
  b: readonly [number, number],
  radius: number,
): boolean {
  return squaredDistance(a, b) <= radius * radius;
}

export function colliding(
  a: readonly [number, number],
  radiusA: number,
  b: readonly [number, number],
  radiusB: number,
): boolean {
  const reach = radiusA + radiusB;
  return squaredDistance(a, b) <= reach * reach;
}
