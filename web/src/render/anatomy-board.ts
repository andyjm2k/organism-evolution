/** Reference sheet of the organs, drawn with the same meshes as the world. */

import { founderParts, partGain, type PartKey, type PartLevels } from "../sim/morphology";
import type { BodySnapshot } from "../sim/world";
import { CreaturePass } from "./creature-gl";
import { paintCreature2d } from "./creature-draw2d";
import { visualRadius, type CreaturePose } from "./creature-layout";

interface Sample {
  title: string;
  blurb: string;
  parts: PartLevels;
  color: [number, number, number];
  jaw: number;
  horn: number;
}

const SHOWN: PartKey[][] = [
  [],
  ["eye", "antenna", "lateral"],
  ["jaw", "claw", "horn", "spine", "venom"],
  ["shell", "plate", "camo"],
  ["fin", "glow", "photo", "siphon"],
];

export function mountAnatomyBoard(button: HTMLButtonElement, dialog: HTMLDialogElement): void {
  const canvas = dialog.querySelector("#anatomy") as HTMLCanvasElement;
  const notes = dialog.querySelector("#anatomy-notes") as HTMLElement;
  const samples = buildSamples();
  notes.replaceChildren();
  for (const sample of samples) {
    const block = document.createElement("section");
    const title = document.createElement("h3");
    title.textContent = sample.title;
    const text = document.createElement("p");
    text.textContent = sample.blurb;
    block.append(title, text);
    notes.appendChild(block);
  }
  let pass: CreaturePass | null = null;
  let gl: WebGL2RenderingContext | null = null;
  try {
    gl = canvas.getContext("webgl2", { alpha: false, antialias: true, depth: true });
    if (gl) {
      pass = new CreaturePass(gl);
    }
  } catch (error) {
    console.error(error);
    gl = null;
    pass = null;
  }
  let frame = 0;
  const draw = (now: number) => {
    if (!dialog.open) {
      return;
    }
    paintBoard(canvas, gl, pass, samples, now / 1000);
    frame = requestAnimationFrame(draw);
  };
  button.addEventListener("click", () => {
    dialog.showModal();
    frame = requestAnimationFrame(draw);
  });
  dialog.addEventListener("close", () => cancelAnimationFrame(frame));
}

function buildSamples(): Sample[] {
  const base = founderParts();
  const grown = (keys: PartKey[]): PartLevels => {
    const parts = { ...base };
    for (const key of keys) {
      parts[key] = key === "eye" || key === "fin" ? 0.95 : 0.85;
    }
    return parts;
  };
  const blurbs = [
    "Eyes and a tail fin. Every lineage starts here, before any extra organ.",
    SHOWN[1].map((key) => partGain(key)).join(" "),
    SHOWN[2].map((key) => partGain(key)).join(" "),
    SHOWN[3].map((key) => partGain(key)).join(" "),
    SHOWN[4].map((key) => partGain(key)).join(" "),
  ];
  const titles = ["Founder", "Senses", "Weapons", "Armor", "Functions"];
  const colors: [number, number, number][] = [
    [96, 168, 86],
    [78, 138, 196],
    [186, 78, 62],
    [168, 142, 86],
    [52, 158, 146],
  ];
  return titles.map((title, index) => ({
    title,
    blurb: blurbs[index],
    parts: grown(SHOWN[index]),
    color: colors[index],
    jaw: index === 0 ? 1.8 : 0,
    horn: index === 0 ? 3.6 : 0,
  }));
}

function paintBoard(
  canvas: HTMLCanvasElement,
  gl: WebGL2RenderingContext | null,
  pass: CreaturePass | null,
  samples: Sample[],
  time: number,
): void {
  const width = Math.max(320, canvas.clientWidth);
  const height = Math.max(160, canvas.clientHeight);
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const bufferWidth = Math.floor(width * ratio);
  const bufferHeight = Math.floor(height * ratio);
  if (canvas.width !== bufferWidth || canvas.height !== bufferHeight) {
    canvas.width = bufferWidth;
    canvas.height = bufferHeight;
  }
  const bodies = samples.map((sample, index) => fakeBody(sample, (width * (index + 0.5)) / samples.length, height * 0.46, index + 1));
  if (gl && pass) {
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.047, 0.086, 0.11, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    pass.draw(bodies, null, { offsetX: 0, offsetY: 0, worldToScreen: (x, y) => [x, y] }, width, height, time);
    return;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.fillStyle = "#0c161c";
  ctx.fillRect(0, 0, width, height);
  for (const body of bodies) {
    const pose: CreaturePose = { color: body.color, parts: body.parts, jaw: body.jaw, claw: 0, horn: body.horn, flagella: 0, selected: false };
    paintCreature2d(ctx, body.x, body.y, visualRadius(body.radius), 0, pose);
  }
}

function fakeBody(sample: Sample, x: number, y: number, id: number): BodySnapshot {
  return {
    id,
    x,
    y,
    radius: 20,
    energy: 1,
    maxEnergy: 1,
    color: sample.color,
    jaw: sample.jaw,
    claw: 0,
    horn: sample.horn,
    flagella: 0,
    trail: [],
    speciesId: 0,
    carnivore: false,
    age: 0,
    food: 0,
    hunts: 0,
    fitness: 0,
    size: 1,
    speed: 1,
    diet: 0.1,
    nodes: 0,
    connections: 0,
    maturity: 1,
    heading: 0,
    parts: sample.parts,
    edges: [],
  };
}
