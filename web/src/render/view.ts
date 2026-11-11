/** World drawing. WebGL2 discs when the context exists, Canvas 2D otherwise and for organs. */

import type { Camera } from "./camera";
import type { BodySnapshot, WorldSnapshot } from "../sim/world";
import { VISIBLE_STRENGTH } from "../sim/body-traits";

export interface ViewOptions {
  showRings: boolean;
  selectedId: number | null;
  foodRadius: number;
  threatRadius: number;
  breedRadius: number;
}

export class WorldView {
  private gl: WebGL2RenderingContext | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private program: WebGLProgram | null = null;
  private buffer: WebGLBuffer | null = null;
  private instance: WebGLBuffer | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private capacity = 0;
  mode: "webgl2" | "canvas" = "canvas";

  constructor(private canvas: HTMLCanvasElement) {
    try {
      const gl = canvas.getContext("webgl2", { alpha: false, antialias: true });
      if (gl) {
        this.gl = gl;
        this.mode = "webgl2";
        this.program = buildProgram(gl);
        this.buffer = gl.createBuffer();
        this.instance = gl.createBuffer();
        this.vao = gl.createVertexArray();
        this.bindQuad(gl);
        return;
      }
    } catch {
      this.gl = null;
    }
    this.mode = "canvas";
    this.ctx = canvas.getContext("2d");
  }

  resize(width: number, height: number): void {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const nextWidth = Math.max(1, Math.floor(width * ratio));
    const nextHeight = Math.max(1, Math.floor(height * ratio));
    if (this.canvas.width === nextWidth && this.canvas.height === nextHeight) {
      return;
    }
    this.canvas.width = nextWidth;
    this.canvas.height = nextHeight;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    if (this.gl) {
      this.bindQuad(this.gl);
    }
  }

  draw(snapshot: WorldSnapshot, camera: Camera, options: ViewOptions): void {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (this.gl && this.mode === "webgl2") {
      this.drawGl(snapshot, camera, options, width, height);
      return;
    }
    const ctx = this.ctx ?? this.canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    this.ctx = ctx;
    const ratio = this.canvas.width / Math.max(1, width);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawScene(ctx, snapshot, camera, options, width, height);
  }

  pick(screenX: number, screenY: number, snapshot: WorldSnapshot, camera: Camera): number | null {
    const world = camera.screenToWorld(screenX, screenY);
    let best: number | null = null;
    let bestDist = Infinity;
    for (const body of snapshot.bodies) {
      const dx = world[0] - body.x;
      const dy = world[1] - body.y;
      const hit = Math.max(12, body.radius + 8);
      const dist = dx * dx + dy * dy;
      if (dist <= hit * hit && dist < bestDist) {
        best = body.id;
        bestDist = dist;
      }
    }
    return best;
  }

  private drawGl(snapshot: WorldSnapshot, camera: Camera, options: ViewOptions, width: number, height: number): void {
    const gl = this.gl!;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(18 / 255, 22 / 255, 28 / 255, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const circles: number[] = [];
    const push = (x: number, y: number, radius: number, color: [number, number, number], alpha: number) => {
      circles.push(x, y, radius, color[0] / 255, color[1] / 255, color[2] / 255, alpha);
    };
    for (const cloud of snapshot.clouds) {
      const alpha = (40 + 60 * cloud.intensity) / 255;
      push(cloud.x, cloud.y, cloud.radius, [60, 200, 90], alpha);
      push(cloud.x, cloud.y, cloud.radius * 0.55, [120, 240, 140], alpha * 0.5);
    }
    for (const food of snapshot.foods) {
      push(food.x, food.y, food.regrowing ? 2 : 3, food.color, 1);
    }
    for (const egg of snapshot.eggs) {
      push(egg.x, egg.y, 4, [230, 210, 160], 1);
    }
    for (const body of snapshot.bodies.slice(0, 800)) {
      const [sx, sy] = camera.worldToScreen(body.x, body.y);
      if (sx < -40 || sy < -40 || sx > width + 40 || sy > height + 40) {
        continue;
      }
      const radius = Math.max(6, Math.min(16, Math.trunc(body.radius) + 2));
      push(body.x, body.y, radius, body.color, 1);
    }
    this.flushCircles(gl, circles, camera, width, height);
    const overlay = this.overlay();
    const ratio = this.canvas.width / Math.max(1, width);
    overlay.setTransform(ratio, 0, 0, ratio, 0, 0);
    overlay.clearRect(0, 0, width, height);
    drawOrgansAndChrome(overlay, snapshot, camera, width, height, options);
  }

  private overlayContext: CanvasRenderingContext2D | null = null;
  private overlayCanvas: HTMLCanvasElement | null = null;

  /** Transparent canvas stacked by the page. Created on demand and returned for mounting. */
  ensureOverlay(): HTMLCanvasElement {
    if (!this.overlayCanvas) {
      this.overlayCanvas = document.createElement("canvas");
      this.overlayCanvas.className = "overlay";
      this.overlayContext = this.overlayCanvas.getContext("2d");
    }
    return this.overlayCanvas;
  }

  private overlay(): CanvasRenderingContext2D {
    const canvas = this.ensureOverlay();
    canvas.width = this.canvas.width;
    canvas.height = this.canvas.height;
    canvas.style.width = this.canvas.style.width;
    canvas.style.height = this.canvas.style.height;
    return this.overlayContext ?? canvas.getContext("2d")!;
  }

  private bindQuad(gl: WebGL2RenderingContext): void {
    const quad = new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]);
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
    const program = this.program!;
    const corner = gl.getAttribLocation(program, "a_corner");
    gl.enableVertexAttribArray(corner);
    gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instance);
    const stride = 7 * 4;
    const names = ["a_center", "a_radius", "a_color", "a_alpha"];
    const sizes = [2, 1, 3, 1];
    let offset = 0;
    for (let i = 0; i < names.length; i += 1) {
      const location = gl.getAttribLocation(program, names[i]);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, sizes[i], gl.FLOAT, false, stride, offset);
      gl.vertexAttribDivisor(location, 1);
      offset += sizes[i] * 4;
    }
  }

  private flushCircles(gl: WebGL2RenderingContext, circles: number[], camera: Camera, width: number, height: number): void {
    const program = this.program!;
    gl.useProgram(program);
    gl.bindVertexArray(this.vao);
    const data = new Float32Array(circles);
    if (circles.length / 7 > this.capacity) {
      this.capacity = Math.max(256, circles.length / 7);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instance);
      gl.bufferData(gl.ARRAY_BUFFER, data.byteLength, gl.DYNAMIC_DRAW);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instance);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    gl.uniform2f(gl.getUniformLocation(program, "u_view"), width, height);
    gl.uniform2f(gl.getUniformLocation(program, "u_camera"), camera.offsetX, camera.offsetY);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, circles.length / 7);
  }
}

function drawScene(
  ctx: CanvasRenderingContext2D,
  snapshot: WorldSnapshot,
  camera: Camera,
  options: ViewOptions,
  width: number,
  height: number,
): void {
  ctx.fillStyle = "#12161c";
  ctx.fillRect(0, 0, width, height);
  for (const cloud of snapshot.clouds) {
    const [sx, sy] = camera.worldToScreen(cloud.x, cloud.y);
    const alpha = (40 + 60 * cloud.intensity) / 255;
    ctx.beginPath();
    ctx.fillStyle = `rgba(60,200,90,${alpha})`;
    ctx.arc(sx, sy, cloud.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.fillStyle = `rgba(120,240,140,${alpha / 2})`;
    ctx.arc(sx, sy, cloud.radius * 0.55, 0, Math.PI * 2);
    ctx.fill();
  }
  for (const food of snapshot.foods) {
    const [sx, sy] = camera.worldToScreen(food.x, food.y);
    if (sx < -6 || sy < -6 || sx > width + 6 || sy > height + 6) {
      continue;
    }
    ctx.beginPath();
    ctx.fillStyle = `rgb(${food.color[0]},${food.color[1]},${food.color[2]})`;
    ctx.arc(sx, sy, food.regrowing ? 2 : 3, 0, Math.PI * 2);
    ctx.fill();
  }
  drawOrgansAndChrome(ctx, snapshot, camera, width, height, options);
}

function drawOrgansAndChrome(
  ctx: CanvasRenderingContext2D,
  snapshot: WorldSnapshot,
  camera: Camera,
  width: number,
  height: number,
  options?: ViewOptions,
): void {
  for (const body of snapshot.bodies.slice(0, 800)) {
    drawTrail(ctx, body, camera);
  }
  for (const egg of snapshot.eggs) {
    const [sx, sy] = camera.worldToScreen(egg.x, egg.y);
    ctx.beginPath();
    ctx.fillStyle = "rgb(245,236,214)";
    ctx.strokeStyle = "rgb(90,70,40)";
    ctx.lineWidth = 1;
    ctx.arc(sx, sy, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  const selected = snapshot.bodies.find((body) => options && body.id === options.selectedId);
  if (selected && options?.showRings) {
    drawRings(ctx, selected, camera, options);
  }
  for (const body of snapshot.bodies.slice(0, 800)) {
    const [sx, sy] = camera.worldToScreen(body.x, body.y);
    if (sx < -40 || sy < -40 || sx > width + 40 || sy > height + 40) {
      continue;
    }
    drawBody(ctx, sx, sy, body);
    if (options && body.id === options.selectedId) {
      ctx.beginPath();
      ctx.strokeStyle = "rgb(255,210,60)";
      ctx.lineWidth = 2;
      ctx.arc(sx, sy, Math.max(6, Math.min(16, Math.trunc(body.radius) + 2)) + 8, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  drawMinimap(ctx, snapshot, camera, width);
}

function drawTrail(ctx: CanvasRenderingContext2D, body: BodySnapshot, camera: Camera): void {
  if (body.trail.length < 2) {
    return;
  }
  for (let i = 1; i < body.trail.length; i += 1) {
    const alpha = 50 + (150 * i) / body.trail.length;
    const [x0, y0] = camera.worldToScreen(body.trail[i - 1][0], body.trail[i - 1][1]);
    const [x1, y1] = camera.worldToScreen(body.trail[i][0], body.trail[i][1]);
    ctx.strokeStyle = `rgba(${body.color[0]},${body.color[1]},${body.color[2]},${alpha / 255})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }
}

function drawRings(ctx: CanvasRenderingContext2D, body: BodySnapshot, camera: Camera, options: ViewOptions): void {
  const [sx, sy] = camera.worldToScreen(body.x, body.y);
  const rings: [number, string][] = [];
  if (!body.carnivore) {
    rings.push([options.foodRadius, "rgba(80,180,100,0.35)"]);
  }
  rings.push([options.threatRadius, "rgba(220,90,70,0.35)"]);
  rings.push([options.breedRadius, "rgba(150,110,220,0.35)"]);
  for (const [radius, color] of rings) {
    if (radius <= 0) {
      continue;
    }
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.arc(sx, sy, radius, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function drawBody(ctx: CanvasRenderingContext2D, x: number, y: number, body: BodySnapshot): void {
  const radius = Math.max(6, Math.min(16, Math.trunc(body.radius) + 2));
  if (body.flagella >= VISIBLE_STRENGTH) {
    const length = radius + Math.min(18, Math.trunc(body.flagella * 2));
    ctx.strokeStyle = tint(body.color, [140, 220, 180], 0.55);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x - radius + 1, y);
    for (let index = 1; index <= 3; index += 1) {
      const along = radius + ((length - radius) * index) / 3;
      const wave = 3 * Math.sin(index * 1.4) * (index % 2 ? 1 : -1);
      ctx.lineTo(x - along, y + wave);
    }
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.fillStyle = `rgb(${body.color[0]},${body.color[1]},${body.color[2]})`;
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  if (body.jaw >= VISIBLE_STRENGTH) {
    const length = radius + Math.min(18, Math.trunc(body.jaw * 2));
    ctx.fillStyle = tint(body.color, [240, 190, 120], 0.55);
    ctx.beginPath();
    ctx.moveTo(x + radius - 1, y - Math.max(3, radius / 2));
    ctx.lineTo(x + length, y);
    ctx.lineTo(x + radius - 1, y + Math.max(3, radius / 2));
    ctx.fill();
  }
  if (body.horn >= VISIBLE_STRENGTH) {
    const length = radius + Math.min(18, Math.trunc(body.horn * 2));
    ctx.fillStyle = tint(body.color, [245, 245, 230], 0.65);
    ctx.beginPath();
    ctx.moveTo(x - 3, y - radius + 1);
    ctx.lineTo(x + 3, y - radius + 1);
    ctx.lineTo(x, y - length);
    ctx.fill();
  }
  if (body.claw >= VISIBLE_STRENGTH) {
    const length = radius + Math.min(18, Math.trunc(body.claw * 2));
    ctx.strokeStyle = tint(body.color, [230, 170, 130], 0.5);
    ctx.lineWidth = 2;
    const spread = Math.max(3, radius / 2);
    for (const sign of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(x + radius - 2, y + sign * spread);
      ctx.lineTo(x + length, y + sign * (spread + 3));
      ctx.stroke();
    }
  }
}

export function minimapRect(width: number): { x: number; y: number; size: number } {
  const size = width < 800 ? 112 : 160;
  const y = width < 800 ? 52 : 10;
  return { x: width - size - 10, y, size };
}

function drawMinimap(ctx: CanvasRenderingContext2D, snapshot: WorldSnapshot, camera: Camera, width: number): void {
  const { x, y, size } = minimapRect(width);
  ctx.fillStyle = "#0a0e12";
  ctx.fillRect(x, y, size, size);
  ctx.strokeStyle = "#465a6e";
  ctx.strokeRect(x, y, size, size);
  const scaleX = size / snapshot.worldWidth;
  const scaleY = size / snapshot.worldHeight;
  for (const cloud of snapshot.clouds) {
    ctx.beginPath();
    ctx.fillStyle = "#286432";
    ctx.arc(x + cloud.x * scaleX, y + cloud.y * scaleY, Math.max(2, cloud.radius * scaleX * 0.5), 0, Math.PI * 2);
    ctx.fill();
  }
  for (const food of snapshot.foods.slice(0, 400)) {
    ctx.fillStyle = `rgb(${food.color[0]},${food.color[1]},${food.color[2]})`;
    ctx.fillRect(x + food.x * scaleX, y + food.y * scaleY, 1, 1);
  }
  for (const body of snapshot.bodies.slice(0, 300)) {
    ctx.beginPath();
    ctx.fillStyle = `rgb(${body.color[0]},${body.color[1]},${body.color[2]})`;
    ctx.arc(x + body.x * scaleX, y + body.y * scaleY, 2, 0, Math.PI * 2);
    ctx.fill();
  }
  const [vx0, vy0, vx1, vy1] = camera.viewportRect();
  ctx.strokeStyle = "#ffdc50";
  ctx.lineWidth = 2;
  ctx.strokeRect(x + vx0 * scaleX, y + vy0 * scaleY, Math.max(4, (vx1 - vx0) * scaleX), Math.max(4, (vy1 - vy0) * scaleY));
  ctx.fillStyle = "#b4bec8";
  ctx.font = "12px sans-serif";
  ctx.fillText("Map", x + 6, y + size - 8);
}

function tint(color: [number, number, number], toward: number[], amount: number): string {
  const mixed = color.map((channel, index) => Math.max(0, Math.min(255, Math.trunc(channel * (1 - amount) + toward[index] * amount))));
  return `rgb(${mixed[0]},${mixed[1]},${mixed[2]})`;
}

function buildProgram(gl: WebGL2RenderingContext): WebGLProgram {
  const vertex = `#version 300 es
layout(location=0) in vec2 a_corner;
layout(location=1) in vec2 a_center;
layout(location=2) in float a_radius;
layout(location=3) in vec3 a_color;
layout(location=4) in float a_alpha;
uniform vec2 u_view;
uniform vec2 u_camera;
out vec3 v_color;
out float v_alpha;
out vec2 v_local;
void main() {
  vec2 screen = (a_center - u_camera) + a_corner * a_radius;
  vec2 ndc = (screen / u_view) * 2.0 - 1.0;
  ndc.y = -ndc.y;
  gl_Position = vec4(ndc, 0.0, 1.0);
  v_color = a_color;
  v_alpha = a_alpha;
  v_local = a_corner;
}`;
  const fragment = `#version 300 es
precision mediump float;
in vec3 v_color;
in float v_alpha;
in vec2 v_local;
out vec4 fragColor;
void main() {
  float dist = length(v_local);
  float alpha = (1.0 - smoothstep(0.92, 1.05, dist)) * v_alpha;
  if (alpha < 0.01) discard;
  fragColor = vec4(v_color, alpha);
}`;
  const program = gl.createProgram()!;
  const vs = compile(gl, gl.VERTEX_SHADER, vertex);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fragment);
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) ?? "shader link failed");
  }
  return program;
}

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) ?? "shader compile failed");
  }
  return shader;
}
