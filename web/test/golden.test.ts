import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { LivingWorld } from "../src/sim/world";
import { captureWorld, restoreWorld } from "../src/sim/snapshot";

const root = new URL("../../", import.meta.url);
const sim = JSON.parse(readFileSync(new URL("config/living-world-config.json", root), "utf8"));
const neat = readFileSync(new URL("config/neat-config.ini", root), "utf8");

describe("python golden digest", () => {
  it("matches the desktop world after 30 steps from seed 1", () => {
    const world = LivingWorld.create({ ...sim, render: false, max_world_steps: 30 }, neat, 1);
    world.steps(30);
    const digest = world.digest();
    expect(digest.step).toBe(30);
    expect(digest.food).toBe(212);
    expect(digest.births).toBe(0);
    expect(digest.bodies).toHaveLength(60);
    expect(digest.bodies[0]).toEqual({ id: 1, x: 3196.32681, y: 3139.999669, energy: 349.604035, species: 2 });
    expect(digest.bodies[1]).toEqual({ id: 2, x: 889.301636, y: 1788.641428, energy: 349.604035, species: 2 });
    expect(new Set(digest.bodies.map((body) => body.species))).toEqual(new Set([2]));
  });

  it("restores a save to the same bodies and the next step", () => {
    const original = LivingWorld.create({ ...sim, render: false, max_world_steps: null }, neat, 4);
    original.steps(20);
    const saved = captureWorld(original);
    const restored = restoreWorld(saved, neat);
    expect(restored.digest()).toEqual(original.digest());
    original.stepOnce();
    restored.stepOnce();
    expect(restored.digest()).toEqual(original.digest());
  });

  it("removes the lowest fitness bodies only after they pass the age threshold", () => {
    const early = LivingWorld.create(
      { ...sim, render: false, max_world_steps: null, selection_interval_steps: 30, fitness_cull_min_steps: 10_000, cull_fraction: 0.5 },
      neat,
      1,
    );
    early.steps(30);
    expect(early.registry.count()).toBe(60);
    const aged = LivingWorld.create(
      { ...sim, render: false, max_world_steps: null, selection_interval_steps: 30, fitness_cull_min_steps: 1, cull_fraction: 0.5 },
      neat,
      1,
    );
    aged.steps(30);
    expect(aged.registry.count()).toBeLessThan(60);
    expect(aged.log.some((line) => line.includes("lowest-fitness"))).toBe(true);
  });
});
