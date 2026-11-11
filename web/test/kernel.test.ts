import { describe, expect, it } from "vitest";
import { PythonRandom } from "../src/rng/python-random";
import { SpatialGrid } from "../src/sim/spatial";
import { colliding, withinRadius } from "../src/sim/distance";
import { makeFounderGenes, mutateGenes, isCarnivoreDiet } from "../src/sim/body-genes";
import { BodyTraits } from "../src/sim/body-traits";
import { PlantStrain, makeFounderStrain } from "../src/sim/plant";
import { Camera } from "../src/render/camera";
import { FitnessTracker } from "../src/sim/fitness-tracker";
import { SelectionScheduler } from "../src/sim/selection";
import { Egg } from "../src/sim/egg";
import { FoodEcology } from "../src/sim/food-ecology";
import { PopulationRegistry } from "../src/sim/registry";
import { parseNeatConfig } from "../src/neat/config";
import { readFileSync } from "node:fs";

describe("PythonRandom", () => {
  it("matches CPython random, gauss, randint, and choice", () => {
    const random = new PythonRandom(1);
    expect(random.random()).toBeCloseTo(0.13436424411240122, 12);
    const gauss = new PythonRandom(1);
    expect(gauss.gauss(0, 0.35)).toBeCloseTo(0.45086466360441196, 10);
    const ints = new PythonRandom(1);
    expect([ints.randint(10, 3990), ints.randint(10, 3990)]).toEqual([560, 2341]);
    const picker = new PythonRandom(1);
    expect(picker.choice(["a", "b", "c", "d"])).toBe("b");
    const state = picker.exportState();
    const copy = new PythonRandom(2);
    copy.importState(state);
    expect(copy.random()).toBe(picker.random());
  });
});

describe("SpatialGrid", () => {
  it("returns only entities inside the radius", () => {
    const grid = new SpatialGrid<{ position: [number, number] | null }>(50);
    const near = { position: [10, 10] as [number, number] };
    const far = { position: [400, 400] as [number, number] };
    grid.insert(near, near.position);
    grid.insert(far, far.position);
    expect(grid.query([0, 0], 30)).toEqual([near]);
    grid.clear();
    expect(grid.query([0, 0], 30)).toEqual([]);
    expect(withinRadius([0, 0], [3, 4], 5)).toBe(true);
    expect(colliding([0, 0], 2, [10, 0], 2)).toBe(false);
  });
});

describe("body genes and plants", () => {
  it("keeps founders grazing and clamps mutated genes", () => {
    const rng = new PythonRandom(3);
    const genes = makeFounderGenes(rng);
    expect(isCarnivoreDiet(genes.diet)).toBe(false);
    expect(genes.size).toBe(1);
    genes.diet = 0.99;
    mutateGenes(genes, rng, 0);
    expect(genes.diet).toBeLessThanOrEqual(1);
    const strain = makeFounderStrain(new PythonRandom(4));
    expect(strain.band()).toBe("indifferent");
    strain.relation = 0.5;
    expect(strain.band()).toBe("symbiotic");
    strain.relation = -0.5;
    expect(strain.band()).toBe("predatory");
    expect(strain.pigment()).toHaveLength(3);
    const child = strain.mutate(new PythonRandom(5));
    expect(child).not.toBe(strain);
    expect(strain.addSuccess(3)).toBe(false);
    expect(strain.addSuccess(5)).toBe(true);
  });
});

describe("BodyTraits", () => {
  it("reads a jaw from an enabled food wire and ignores a disabled one", () => {
    const genome = {
      connections: new Map([
        ["a", { inNode: -12, outNode: 0, weight: 1.8, enabled: true, innovation: 1 }],
        ["b", { inNode: -19, outNode: 1, weight: 2, enabled: false, innovation: 2 }],
      ]),
    };
    const traits = BodyTraits.fromGenome(genome as never, false);
    expect(traits.jaw).toBeCloseTo(1.8);
    expect(traits.horn).toBe(0);
    expect(traits.present().map(([label]) => label)).toEqual(["Jaw"]);
  });
});

describe("Camera", () => {
  it("clamps panning and converts coordinates", () => {
    const camera = new Camera(4000, 4000, 900, 900, 0.15);
    camera.beginDrag(0, 0);
    camera.dragTo(-50, -20);
    camera.endDrag();
    expect(camera.offsetX).toBe(50);
    expect(camera.offsetY).toBe(20);
    expect(camera.screenToWorld(0, 0)).toEqual([50, 20]);
    expect(camera.worldToScreen(50, 20)).toEqual([0, 0]);
    camera.centerOn(4000, 4000);
    expect(camera.offsetX).toBe(3100);
    expect(camera.isDragging()).toBe(false);
    camera.setTrack([100, 100]);
    camera.updateTracking();
    camera.setTrack(null);
    expect(camera.viewportRect()[2]).toBeGreaterThan(camera.viewportRect()[0]);
  });
});

describe("FitnessTracker and selection gate", () => {
  it("scores a meal once and gates selection on the interval", () => {
    const tracker = new FitnessTracker(150, 0.5, 100, 100);
    tracker.init(1);
    const organism = { genomeId: 1, isCarnivore: false, organismsConsumed: 0, foodConsumed: 2 } as never;
    tracker.tickAlive(organism);
    tracker.tickAlive(organism);
    expect(tracker.get(1)).toBeCloseTo(0.5 * 2 + 200);
    tracker.recordOffspring([1, null]);
    expect(tracker.top(1)).toEqual([1]);
    tracker.finalizeDeath(organism, 10);
    const genomes = new Map([[1, { fitness: null as number | null }]]);
    tracker.apply(genomes as never);
    expect(genomes.get(1)?.fitness).toBeGreaterThan(200);
    const scheduler = new SelectionScheduler(4000, 120, 0.1, 0.03, 1000, new PythonRandom(1));
    expect(scheduler.shouldRun(0)).toBe(false);
    expect(scheduler.shouldRun(4000)).toBe(true);
  });
});

describe("Egg", () => {
  it("fills only while the parent is willing and then hatches on its clock", () => {
    const parent = { energy: 200, position: [10, 10] as [number, number] };
    const egg = new Egg(parent as never, parent as never, parent as never, {} as never, 80, 2);
    expect(egg.gestate(parent as never, 0.1)).toBe(false);
    while (parent.energy > 40 && egg.stored < 80) {
      egg.gestate(parent as never, 1);
    }
    expect(egg.stored).toBeGreaterThanOrEqual(80);
    egg.lay([3, 4]);
    expect(egg.tick()).toBe(false);
    expect(egg.tick()).toBe(true);
  });
});

describe("PopulationRegistry", () => {
  it("culls someone else before a protected parent", () => {
    const text = readFileSync(new URL("../../config/neat-config.ini", import.meta.url), "utf8");
    const registry = new PopulationRegistry(2, {} as never, parseNeatConfig(text), new PythonRandom(1));
    const bodies = new Map<number, { energy: number; position: [number, number]; genomeId: number }>();
    bodies.set(1, { energy: 10, position: [0, 0], genomeId: 1 });
    bodies.set(2, { energy: 10, position: [1, 1], genomeId: 2 });
    registry.replaceAll(bodies as never, 3);
    const removed = registry.cullWorst((id) => id, 1, new Set([1]));
    expect(removed).toEqual([2]);
    expect(registry.has(1)).toBe(true);
    const spared = registry.cullWorst((id) => id, 1, new Set([1]), true);
    expect(spared).toEqual([]);
    expect(registry.has(1)).toBe(true);
  });
});

describe("FoodEcology", () => {
  it("seeds the configured pellet count and reports scarcity", () => {
    const text = readFileSync(new URL("../../config/neat-config.ini", import.meta.url), "utf8");
    expect(parseNeatConfig(text).numInputs).toBe(31);
    const ecology = new FoodEcology(
      {
        width: 4000,
        height: 4000,
        foodTarget: 20,
        numFood: 20,
        spawnRate: 0,
        scarcityThreshold: 0.35,
        scarcityMultiplier: 2.5,
        cloudCount: 2,
        cloudMin: 50,
        cloudMax: 80,
      },
      new PythonRandom(1),
    );
    expect(ecology.activeFoodCount()).toBe(20);
    expect(ecology.foodCap()).toBe(32);
    expect(ecology.hasRoom()).toBe(true);
    ecology.foodItems[0].position = null;
    ecology.prune();
    expect(ecology.activeFoodCount()).toBe(19);
    expect(ecology.scarcityRatio()).toBeCloseTo(19 / 20);
    ecology.tick();
    expect(ecology.clouds[0].position[0]).toBeGreaterThanOrEqual(0);
  });
});
