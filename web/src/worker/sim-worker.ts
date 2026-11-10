/** Simulation off the UI thread. The page only receives plain snapshots. */

import { LivingWorld, type SimConfig, type WorldSnapshot } from "../sim/world";
import { captureWorld, restoreWorld, type SaveFile } from "../sim/snapshot";

let world: LivingWorld | null = null;
let neatText = "";
let simConfig: SimConfig = {};

self.onmessage = (event: MessageEvent) => {
  const message = event.data as { type: string; sim?: SimConfig; neat?: string; seed?: number; count?: number; save?: SaveFile };
  if (message.type === "init" && message.sim && message.neat && message.seed !== undefined) {
    simConfig = message.sim;
    neatText = message.neat;
    world = LivingWorld.create(simConfig, neatText, message.seed);
    publish();
    return;
  }
  if (!world) {
    return;
  }
  if (message.type === "steps") {
    world.steps(message.count ?? 1);
    publish();
    return;
  }
  if (message.type === "save") {
    self.postMessage({ type: "saved", file: captureWorld(world) });
    return;
  }
  if (message.type === "load" && message.save) {
    try {
      simConfig = message.save.sim;
      world = restoreWorld(message.save, neatText);
      publish();
    } catch {
      self.postMessage({ type: "restore-error" });
    }
  }
};

function publish(): void {
  if (!world) {
    return;
  }
  const snapshot: WorldSnapshot = world.snapshot();
  self.postMessage({ type: "snapshot", snapshot });
}
