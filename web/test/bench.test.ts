import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { LivingWorld } from "../src/sim/world";

describe("step throughput", () => {
  it("runs at least 120 steps per second, the 60 fps stride-2 budget", () => {
    const root = new URL("../../", import.meta.url);
    const sim = JSON.parse(readFileSync(new URL("config/living-world-config.json", root), "utf8"));
    const neat = readFileSync(new URL("config/neat-config.ini", root), "utf8");
    const world = LivingWorld.create({ ...sim, render: false, max_world_steps: null }, neat, 1);
    const steps = 240;
    const started = performance.now();
    world.steps(steps);
    const elapsed = performance.now() - started;
    const rate = (steps / elapsed) * 1000;
    console.log(`steps/sec ${rate.toFixed(1)} over ${steps} steps (${elapsed.toFixed(0)} ms)`);
    expect(rate).toBeGreaterThan(120);
  });

  it("survives the first selection boundary", () => {
    const root = new URL("../../", import.meta.url);
    const sim = JSON.parse(readFileSync(new URL("config/living-world-config.json", root), "utf8"));
    const neat = readFileSync(new URL("config/neat-config.ini", root), "utf8");
    const world = LivingWorld.create({ ...sim, render: false, max_world_steps: null }, neat, 1);
    world.steps(4000);
    expect(world.step).toBe(4000);
    expect(world.digest().bodies.length).toBeGreaterThan(0);
    expect(world.log.some((line) => line.startsWith("Selection:"))).toBe(true);
  }, 60_000);
});
