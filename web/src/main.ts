/** Page shell: camera, controls, and a worker-owned living world. */

import { Camera } from "./render/camera";
import { minimapRect, WorldView } from "./render/view";
import type { SimConfig, WorldSnapshot } from "./sim/world";

const app = document.querySelector("#app") as HTMLElement;
const stage = document.querySelector("#stage") as HTMLElement;
const canvas = document.querySelector("#world") as HTMLCanvasElement;
const hud = document.querySelector("#hud") as HTMLElement;
const panel = document.querySelector("#panel") as HTMLElement;
const speciesEl = document.querySelector("#species") as HTMLElement;
const logEl = document.querySelector("#log") as HTMLElement;
const statusEl = document.querySelector("#status") as HTMLElement;
const board = document.querySelector("#board") as HTMLElement;
const dialog = document.querySelector("#settings-dialog") as HTMLDialogElement;
const form = document.querySelector("#settings-form") as HTMLFormElement;

const view = new WorldView(canvas);
if (view.mode === "webgl2") {
  stage.appendChild(view.ensureOverlay());
}
const camera = new Camera(4000, 4000, stage.clientWidth, stage.clientHeight, 0.15);
let snapshot: WorldSnapshot | null = null;
let selectedId: number | null = null;
let tracking = false;
let showRings = true;
let paused = false;
let stride = 2;
let restoring = false;
let busy = false;
let frames = 0;
let fpsStamp = performance.now();
let fps = 0;
let sim: SimConfig = {};
let neat = "";
let seed = 1;

const worker = new Worker(new URL("./worker/sim-worker.ts", import.meta.url), { type: "module" });

worker.onmessage = (event: MessageEvent) => {
  const message = event.data as { type: string; snapshot?: WorldSnapshot; file?: SaveFile; step?: number; target?: number };
  if (message.type === "snapshot" && message.snapshot) {
    snapshot = message.snapshot;
    busy = false;
    restoring = false;
    statusEl.textContent = `${view.mode} · ${fps.toFixed(0)} fps`;
    renderHud();
  } else if (message.type === "restore") {
    restoring = true;
    statusEl.textContent = `Restoring ${message.step} / ${message.target}`;
  } else if (message.type === "restore-error") {
    restoring = false;
    statusEl.textContent = "Could not load that save";
  } else if (message.type === "saved" && message.file) {
    void storeSave(message.file);
    statusEl.textContent = `Saved step ${message.file.steps}`;
  }
};

const configUrl = new URL("./config/living-world-config.json", document.baseURI);
const neatUrl = new URL("./config/neat-config.ini", document.baseURI);

Promise.all([fetch(configUrl).then((response) => response.json()), fetch(neatUrl).then((response) => response.text())]).then(
  ([loadedSim, loadedNeat]: [SimConfig, string]) => {
    sim = loadedSim;
    neat = loadedNeat;
    const params = new URLSearchParams(location.search);
    if (params.get("seed")) {
      seed = clampInt(params.get("seed"), 1, 1, 1_000_000_000);
      sim.seed = seed;
    }
    if (params.get("max_world_steps")) {
      sim.max_world_steps = clampInt(params.get("max_world_steps"), 1, 1, 50_000_000);
    }
    seed = typeof sim.seed === "number" ? sim.seed : 1;
    fillSettings();
    worker.postMessage({ type: "init", sim, neat, seed });
  },
);

function loop(): void {
  const width = stage.clientWidth;
  const height = stage.clientHeight;
  camera.viewWidth = width;
  camera.viewHeight = height;
  view.resize(width, height);
  if (snapshot && tracking && selectedId !== null) {
    const body = snapshot.bodies.find((item) => item.id === selectedId);
    camera.setTrack(body ? [body.x, body.y] : null);
  }
  camera.updateTracking();
  if (snapshot) {
    view.draw(snapshot, camera, {
      showRings,
      selectedId,
      foodRadius: numberConfig("food_detection_radius", 1400),
      threatRadius: numberConfig("threat_detection_radius", 100),
      breedRadius: numberConfig("breeding_detection_radius", 150),
    });
  }
  frames += 1;
  const now = performance.now();
  if (now - fpsStamp >= 500) {
    fps = (frames * 1000) / (now - fpsStamp);
    frames = 0;
    fpsStamp = now;
  }
  const hidden = document.visibilityState !== "visible";
  if (!paused && !hidden && !restoring && !busy && snapshot) {
    busy = true;
    worker.postMessage({ type: "steps", count: stride });
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

canvas.addEventListener("pointerdown", (event) => {
  const point = localPoint(event);
  if (inMinimap(point.x, point.y)) {
    jumpMinimap(point.x, point.y);
    tracking = false;
    return;
  }
  const picked = snapshot ? view.pick(point.x, point.y, snapshot, camera) : null;
  if (picked !== null) {
    selectedId = picked;
    tracking = true;
    renderPanel();
  } else {
    camera.beginDrag(point.x, point.y);
    tracking = false;
  }
});
canvas.addEventListener("pointermove", (event) => {
  if (camera.isDragging()) {
    const point = localPoint(event);
    camera.dragTo(point.x, point.y);
  }
});
canvas.addEventListener("pointerup", () => camera.endDrag());
canvas.addEventListener("pointerleave", () => camera.endDrag());

window.addEventListener("keydown", (event) => {
  if (event.key === "s" || event.key === "S") {
    showRings = !showRings;
  } else if (event.key === "t" || event.key === "T") {
    tracking = !tracking;
  } else if (event.key === "Escape") {
    selectedId = null;
    tracking = false;
    panel.hidden = true;
  } else if (event.key === " ") {
    event.preventDefault();
    togglePause();
  }
});

document.querySelector("#pause")!.addEventListener("click", togglePause);
document.querySelector("#sense")!.addEventListener("click", () => {
  showRings = !showRings;
});
document.querySelector("#track")!.addEventListener("click", () => {
  tracking = !tracking;
});
document.querySelector("#clear")!.addEventListener("click", () => {
  selectedId = null;
  tracking = false;
  panel.hidden = true;
});
document.querySelector("#fast")!.addEventListener("click", () => {
  const levels = [1, 2, 8, 32];
  stride = levels[(levels.indexOf(stride) + 1) % levels.length];
  (document.querySelector("#fast") as HTMLButtonElement).textContent = `Fast ${stride}×`;
});
document.querySelector("#save")!.addEventListener("click", () => worker.postMessage({ type: "save" }));
document.querySelector("#load")!.addEventListener("click", () => void loadSave());
document.querySelector("#settings")!.addEventListener("click", () => dialog.showModal());
document.querySelector("#species-toggle")!.addEventListener("click", () => {
  if (window.matchMedia("(max-width: 800px)").matches) {
    app.classList.toggle("species-open");
  } else {
    app.classList.toggle("species-closed");
  }
});
document.querySelector("#board-close")!.addEventListener("click", () => {
  app.classList.remove("species-open");
  app.classList.add("species-closed");
});
form.addEventListener("submit", () => {
  if ((document.activeElement as HTMLButtonElement | null)?.value !== "restart" && form.returnValue !== "restart") {
    return;
  }
});
document.querySelector("#restart")!.addEventListener("click", () => {
  const data = new FormData(form);
  seed = clampInt(String(data.get("seed") ?? seed), seed, 1, 1_000_000_000);
  sim.max_population = clampInt(String(data.get("max_population") ?? ""), numberConfig("max_population", 120), 4, 400);
  sim.fitness_cull_min_steps = clampInt(String(data.get("fitness_cull_min_steps") ?? ""), numberConfig("fitness_cull_min_steps", 1000), 0, 100_000);
  sim.breeding_cooldown = clampInt(String(data.get("breeding_cooldown") ?? ""), numberConfig("breeding_cooldown", 200), 1, 100000);
  sim.food_target_density = clampInt(String(data.get("food_target_density") ?? ""), numberConfig("food_target_density", 200), 1, 5000);
  sim.num_food_items = sim.food_target_density;
  const maxSteps = String(data.get("max_world_steps") ?? "");
  sim.max_world_steps = maxSteps === "" ? null : clampInt(maxSteps, 1, 1, 50_000_000);
  selectedId = null;
  tracking = false;
  worker.postMessage({ type: "init", sim, neat, seed });
});

function togglePause(): void {
  paused = !paused;
  (document.querySelector("#pause") as HTMLButtonElement).textContent = paused ? "Resume" : "Pause";
}

function renderHud(): void {
  if (!snapshot) {
    return;
  }
  const track = tracking && selectedId !== null ? "ON" : "OFF";
  hud.textContent = `Step ${snapshot.step} | Alive ${snapshot.alive}/${snapshot.capacity} | Food ${snapshot.foodCount} (${Math.round(snapshot.scarcity * 100)}%) | Species ${snapshot.speciesCount} | Track ${track}`;
  speciesEl.replaceChildren();
  if (snapshot.scoreboard.length === 0) {
    const empty = document.createElement("p");
    empty.textContent = "Names appear at the first epoch.";
    speciesEl.appendChild(empty);
  }
  for (const row of snapshot.scoreboard) {
    const card = document.createElement("div");
    card.className = "card";
    const name = document.createElement("strong");
    name.textContent = row.name;
    const detail = document.createElement("span");
    detail.textContent = `${row.carnivore ? "Carnivore" : "Herbivore"} · fitness ${Math.round(row.fitness)}`;
    card.append(name, detail);
    speciesEl.appendChild(card);
  }
  logEl.textContent = snapshot.log.join("\n");
  renderPanel();
}

function renderPanel(): void {
  const body = snapshot?.bodies.find((item) => item.id === selectedId);
  if (!body) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  panel.replaceChildren();
  const header = document.createElement("div");
  header.id = "panel-head";
  const title = document.createElement("h3");
  title.textContent = `Species ${body.speciesId}`;
  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Close";
  close.addEventListener("click", () => {
    selectedId = null;
    tracking = false;
    panel.hidden = true;
  });
  header.append(title, close);
  panel.appendChild(header);
  const lines = [
    `${body.carnivore ? "Carnivore" : "Herbivore"} ${body.diet.toFixed(2)} | Energy ${Math.round(body.energy)}/${Math.round(body.maxEnergy)}`,
    `Age ${body.age} steps`,
    `Food ${body.food} | Hunts ${body.hunts}`,
    `Fitness ${Math.round(body.fitness)}`,
    `Genome ${body.id}`,
    `Size ${body.size.toFixed(2)} | Speed ${body.speed.toFixed(2)}`,
    organLine(body),
  ];
  for (const line of lines) {
    const paragraph = document.createElement("p");
    paragraph.textContent = line;
    panel.appendChild(paragraph);
  }
  panel.appendChild(genomeGraph(body));
}

function organLine(body: NonNullable<WorldSnapshot["bodies"][number]>): string {
  const parts = [
    ["Jaw", body.jaw],
    ["Claw", body.claw],
    ["Horn", body.horn],
    ["Flagella", body.flagella],
  ].filter(([, strength]) => Number(strength) >= 0.25);
  if (parts.length === 0) {
    return "Body: plain cell";
  }
  return `Body: ${parts.map(([label, strength]) => `${label} ${Number(strength).toFixed(1)}`).join(" | ")}`;
}

function genomeGraph(body: WorldSnapshot["bodies"][number]): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.id = "graph";
  svg.setAttribute("viewBox", "0 0 280 130");
  const inputs = Array.from({ length: 31 }, (_, index) => -index - 1);
  const hidden = [...new Set(body.edges.flatMap((edge) => [edge.from, edge.to]))].filter((key) => key > 3).sort((a, b) => a - b).slice(0, 8);
  const outputs = [0, 1, 2, 3];
  const columns = [inputs, hidden, outputs].filter((column) => column.length > 0);
  const positions = new Map<number, [number, number]>();
  columns.forEach((column, columnIndex) => {
    column.forEach((key, row) => {
      const x = 28 + ((280 - 56) * columnIndex) / Math.max(1, columns.length - 1);
      const y = 16 + ((130 - 32) * (row + 1)) / (column.length + 1);
      positions.set(key, [x, y]);
    });
  });
  for (const edge of body.edges) {
    const from = positions.get(edge.from);
    const to = positions.get(edge.to);
    if (!from || !to) {
      continue;
    }
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(from[0]));
    line.setAttribute("y1", String(from[1]));
    line.setAttribute("x2", String(to[0]));
    line.setAttribute("y2", String(to[1]));
    const clamped = Math.max(-1, Math.min(1, edge.weight / 30));
    line.setAttribute("stroke", clamped >= 0 ? `rgb(80,${120 + 100 * clamped},220)` : `rgb(${120 + 100 * Math.abs(clamped)},100,100)`);
    svg.appendChild(line);
  }
  for (const [key, position] of positions) {
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", String(position[0]));
    dot.setAttribute("cy", String(position[1]));
    dot.setAttribute("r", "3");
    dot.setAttribute("fill", key >= 0 ? (key <= 3 ? "#dc7850" : "#508cdc") : "#78b478");
    svg.appendChild(dot);
  }
  return svg;
}

function localPoint(event: PointerEvent): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function inMinimap(x: number, y: number): boolean {
  const rect = minimapRect(canvas.clientWidth);
  return x >= rect.x && x <= rect.x + rect.size && y >= rect.y && y <= rect.y + rect.size;
}

function jumpMinimap(x: number, y: number): void {
  const rect = minimapRect(canvas.clientWidth);
  const worldX = ((x - rect.x) / rect.size) * camera.worldWidth;
  const worldY = ((y - rect.y) / rect.size) * camera.worldHeight;
  camera.setTrack(null);
  camera.centerOn(worldX, worldY, false);
}

function numberConfig(key: string, fallback: number): number {
  const value = sim[key];
  return typeof value === "number" ? value : fallback;
}

function clampInt(raw: string | null, fallback: number, min: number, max: number): number {
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, Math.trunc(parsed)));
}

function fillSettings(): void {
  (form.elements.namedItem("seed") as HTMLInputElement).value = String(seed);
  (form.elements.namedItem("max_population") as HTMLInputElement).value = String(numberConfig("max_population", 120));
  (form.elements.namedItem("fitness_cull_min_steps") as HTMLInputElement).value = String(numberConfig("fitness_cull_min_steps", 1000));
  (form.elements.namedItem("breeding_cooldown") as HTMLInputElement).value = String(numberConfig("breeding_cooldown", 200));
  (form.elements.namedItem("food_target_density") as HTMLInputElement).value = String(numberConfig("food_target_density", 200));
  const maxSteps = sim.max_world_steps;
  (form.elements.namedItem("max_world_steps") as HTMLInputElement).value = typeof maxSteps === "number" ? String(maxSteps) : "";
}

interface SaveFile {
  version: 1;
  seed: number;
  steps: number;
  sim: SimConfig;
}

function storeSave(file: SaveFile): Promise<void> {
  return openDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const tx = db.transaction("saves", "readwrite");
        tx.objectStore("saves").put(file, "world");
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      }),
  );
}

async function loadSave(): Promise<void> {
  const db = await openDb();
  const file = await new Promise<SaveFile | undefined>((resolve, reject) => {
    const tx = db.transaction("saves", "readonly");
    const request = tx.objectStore("saves").get("world");
    request.onsuccess = () => resolve(request.result as SaveFile | undefined);
    request.onerror = () => reject(request.error);
  });
  if (!file || file.version !== 1) {
    statusEl.textContent = "No saved world";
    return;
  }
  sim = file.sim;
  seed = file.seed;
  restoring = true;
  worker.postMessage({ type: "load", save: file });
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("living-world", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("saves");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
