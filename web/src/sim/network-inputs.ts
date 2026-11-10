/** Fixed 31-channel sensor vector. Channel order matches the desktop harness. */

import { squaredDistance } from "./distance";

export const NUM_NETWORK_INPUTS = 31;
const EMPTY_SECOND: [number, number, number] = [1, 0, 0];

interface Sensed {
  position: [number, number] | null;
}

/** Fields the sensor builder reads. Kept here so organism.ts can import this module. */
export interface SensorBody extends Sensed {
  lastPosition: [number, number];
  speed: number;
  size: number;
  energy: number;
  maxEnergy: number;
  isCarnivore: boolean;
  canBreed(): boolean;
}

export function buildNetworkInputs(
  organism: SensorBody,
  nearbyFood: Sensed[],
  nearbyOrganisms: SensorBody[],
  nearbyThreats: SensorBody[],
  nearbyPartners: SensorBody[],
  environment: { width: number; height: number; foodDetectionRadius: number; threatDetectionRadius: number; detectionRadius: number; breedingDetectionRadius: number },
): number[] {
  const { width, height } = environment;
  const [x, y] = organism.position!;
  const energyFrac = organism.energy / Math.max(organism.maxEnergy, 1);
  const centerX = width / 2;
  const centerY = height / 2;
  const distCenter = Math.hypot(x - centerX, y - centerY);
  const maxDist = Math.hypot(centerX, centerY) || 1;
  let mdx = 0;
  let mdy = 0;
  if (organism.lastPosition[0] !== x || organism.lastPosition[1] !== y) {
    if (organism.speed > 0) {
      mdx = (x - organism.lastPosition[0]) / organism.speed;
      mdy = (y - organism.lastPosition[1]) / organism.speed;
      const mag = Math.hypot(mdx, mdy);
      if (mag > 0) {
        mdx /= mag;
        mdy /= mag;
      }
    }
  }
  const inputs = [
    energyFrac,
    distCenter / maxDist,
    width ? (x / width) * 2 - 1 : 0,
    height ? (y / height) * 2 - 1 : 0,
    mdx,
    mdy,
    Math.min(1, organism.size / 4),
    width ? Math.min(x / width, (width - x) / width) : 0,
    height ? Math.min(y / height, (height - y) / height) : 0,
  ];
  if (organism.isCarnivore) {
    inputs.push(0, 1, 0, 0, ...EMPTY_SECOND);
  } else {
    const [foodNear, foodSecond] = closestEntities(organism, nearbyFood, environment.foodDetectionRadius, true);
    inputs.push(...foodNear, ...foodSecond);
  }
  if (organism.isCarnivore) {
    const prey = nearbyOrganisms.filter((other) => !other.isCarnivore);
    const [preyNear, preySecond] = closestEntities(organism, prey, environment.detectionRadius, false);
    inputs.push(...preyNear, ...preySecond);
  } else {
    const [threatNear, threatSecond] = closestEntities(organism, nearbyThreats, environment.threatDetectionRadius, false);
    inputs.push(...threatNear, ...threatSecond);
  }
  const [partnerNear, partnerSecond] = closestEntities(organism, nearbyPartners, environment.breedingDetectionRadius, false);
  inputs.push(...partnerNear, organism.canBreed() ? 1 : 0, ...partnerSecond);
  if (inputs.length !== NUM_NETWORK_INPUTS) {
    throw new Error(`Expected ${NUM_NETWORK_INPUTS} inputs, built ${inputs.length}`);
  }
  return inputs;
}

function closestEntities(
  organism: SensorBody,
  entities: Sensed[],
  radius: number,
  unitDirection: boolean,
): [[number, number, number, number], [number, number, number]] {
  const ranked = entitiesInRadius(organism, entities, radius);
  if (ranked.length === 0) {
    return [[0, 1, 0, 0], EMPTY_SECOND];
  }
  const countNorm = Math.min(1, ranked.length / 10);
  const nearest = pack(organism, ranked[0].entity, ranked[0].distSq, radius, unitDirection);
  const nearestBlock: [number, number, number, number] = [countNorm, nearest[0], nearest[1], nearest[2]];
  if (ranked.length < 2) {
    return [nearestBlock, EMPTY_SECOND];
  }
  return [nearestBlock, pack(organism, ranked[1].entity, ranked[1].distSq, radius, unitDirection)];
}

function entitiesInRadius(organism: SensorBody, entities: Sensed[], radius: number): { distSq: number; entity: Sensed }[] {
  const radiusSq = radius > 0 ? radius * radius : 0;
  const ranked: { distSq: number; entity: Sensed }[] = [];
  for (const entity of entities) {
    if (!entity.position || !organism.position) {
      continue;
    }
    const distSq = squaredDistance(organism.position, entity.position);
    if (distSq <= radiusSq) {
      ranked.push({ distSq, entity });
    }
  }
  ranked.sort((a, b) => a.distSq - b.distSq);
  return ranked;
}

function pack(
  organism: SensorBody,
  entity: Sensed,
  distSq: number,
  radius: number,
  unitDirection: boolean,
): [number, number, number] {
  const best = Math.sqrt(distSq);
  const distNorm = Math.min(1, best / Math.max(radius, 1));
  const dx = entity.position![0] - organism.position![0];
  const dy = entity.position![1] - organism.position![1];
  const scale = unitDirection ? best : radius;
  if (scale <= 0) {
    return [distNorm, 0, 0];
  }
  return [distNorm, dx / scale, dy / scale];
}
