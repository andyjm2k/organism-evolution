/** Lit instanced creatures on the orthographic living-world camera. */

import { creatureParts, visualRadius, type CreaturePose } from "./creature-layout";
import { boxMesh, coneMesh, sphereMesh, type MeshData } from "./creature-mesh";
import type { BodySnapshot } from "../sim/world";

const STRIDE = 15;

const VERTEX = `#version 300 es
layout(location=0) in vec3 a_pos;
layout(location=1) in vec3 a_nrm;
layout(location=2) in vec3 i_origin;
layout(location=3) in float i_yaw;
layout(location=4) in vec3 i_scale;
layout(location=5) in float i_seed;
layout(location=6) in vec3 i_color;
layout(location=7) in vec4 i_mat;
uniform vec2 u_view;
uniform vec2 u_camera;
uniform float u_time;
out vec3 v_color;
out vec3 v_nrm;
out vec3 v_world;
out vec3 v_local;
out vec4 v_mat;
vec3 yawed(vec3 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}
void main() {
  vec3 shaped = a_pos;
  float wave = sin(u_time * 6.4 + i_seed) * i_mat.w;
  shaped.y += wave * shaped.x * shaped.x;
  vec3 local = yawed(shaped * i_scale, i_yaw);
  vec3 world = i_origin + local;
  vec2 screen = world.xy - u_camera;
  vec2 ndc = (screen / u_view) * 2.0 - 1.0;
  ndc.y = -ndc.y;
  gl_Position = vec4(ndc, 0.84 - world.z * 0.01, 1.0);
  v_nrm = normalize(yawed(a_nrm, i_yaw));
  v_color = i_color;
  v_world = world;
  v_local = shaped;
  v_mat = i_mat;
}`;

const FRAGMENT = `#version 300 es
precision mediump float;
in vec3 v_color;
in vec3 v_nrm;
in vec3 v_world;
in vec3 v_local;
in vec4 v_mat;
uniform float u_halo;
out vec4 fragColor;
void main() {
  vec3 n = normalize(v_nrm);
  vec3 key = normalize(vec3(-0.42, -0.28, 0.86));
  float wrap = n.z * 0.35 + max(dot(n, key), 0.0) * 0.65;
  vec3 hemi = mix(vec3(0.03, 0.06, 0.08), vec3(0.78, 0.86, 0.82), clamp(n.z * 0.5 + 0.5, 0.0, 1.0));
  vec3 halfV = normalize(key + vec3(0.0, 0.0, 1.0));
  float gloss = mix(64.0, 8.0, v_mat.y);
  float spec = pow(max(dot(n, halfV), 0.0), gloss) * (1.0 - v_mat.y);
  float rim = pow(1.0 - max(n.z, 0.0), 1.65);
  float stripe = 0.5 + 0.5 * sin(v_world.x * 0.85 + v_local.y * 18.0);
  vec3 albedo = mix(v_color, v_color * vec3(0.42, 0.55, 0.38), v_mat.z * stripe);
  vec3 lit = albedo * (hemi * 0.55 + vec3(1.0, 0.95, 0.88) * (0.16 + wrap));
  lit += vec3(1.0, 0.97, 0.9) * spec * 0.55;
  lit += vec3(0.45, 0.75, 0.7) * rim * 0.35;
  lit += albedo * v_mat.x;
  if (u_halo > 0.5) {
    fragColor = vec4(albedo * (0.45 + v_mat.x), 0.28);
    return;
  }
  fragColor = vec4(lit, 1.0);
}`;

interface MeshSlot {
  kind: "sphere" | "cone" | "box";
  vao: WebGLVertexArrayObject;
  buffer: WebGLBuffer;
  count: number;
}

export class CreaturePass {
  private program: WebGLProgram;
  private slots: MeshSlot[];

  constructor(private gl: WebGL2RenderingContext) {
    this.program = link(gl, VERTEX, FRAGMENT);
    this.slots = [sphereMesh(), coneMesh(), boxMesh()].map((mesh, index) => this.makeSlot(mesh, (["sphere", "cone", "box"] as const)[index]));
  }

  draw(
    bodies: BodySnapshot[],
    selectedId: number | null,
    camera: { offsetX: number; offsetY: number; worldToScreen(x: number, y: number): [number, number] },
    width: number,
    height: number,
    time: number,
  ): void {
    const buckets: Record<"sphere" | "cone" | "box", number[]> = { sphere: [], cone: [], box: [] };
    const halos: number[] = [];
    for (const body of bodies) {
      const [sx, sy] = camera.worldToScreen(body.x, body.y);
      if (sx < -120 || sy < -120 || sx > width + 120 || sy > height + 120) {
        continue;
      }
      const radius = visualRadius(body.radius);
      pushInstance(buckets.sphere, body.x + 3, body.y + 4, 0.4, 0, radius * 0.9, radius * 0.68, Math.max(0.6, radius * 0.05), body.id, [0.01, 0.02, 0.03], 0, 1, 0, 0);
      const pose: CreaturePose = {
        color: body.color,
        parts: body.parts,
        jaw: body.jaw,
        claw: body.claw,
        horn: body.horn,
        flagella: body.flagella,
        selected: body.id === selectedId,
      };
      const heading = body.heading || 0;
      const cos = Math.cos(heading);
      const sin = Math.sin(heading);
      creatureParts(pose).forEach((piece, index) => {
        const x = body.x + (piece.x * cos - piece.y * sin) * radius;
        const y = body.y + (piece.x * sin + piece.y * cos) * radius;
        const z = piece.z * radius;
        pushInstance(
          buckets[piece.mesh],
          x,
          y,
          z,
          heading + piece.yaw,
          piece.sx * radius,
          piece.sy * radius,
          piece.sz * radius,
          body.id * 0.17 + index,
          piece.color,
          piece.emissive,
          piece.roughness,
          piece.camo,
          piece.wiggle,
        );
        if (piece.halo) {
          pushInstance(halos, x, y, z, heading + piece.yaw, piece.sx * radius * 2.6, piece.sy * radius * 2.6, piece.sz * radius * 1.4, body.id, piece.color, 1, 0.2, 0, 0);
        }
      });
    }
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.uniform2f(gl.getUniformLocation(this.program, "u_view"), width, height);
    gl.uniform2f(gl.getUniformLocation(this.program, "u_camera"), camera.offsetX, camera.offsetY);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_time"), time);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_halo"), 0);
    for (const slot of this.slots) {
      this.flush(slot, buckets[slot.kind]);
    }
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.depthMask(false);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_halo"), 1);
    this.flush(this.slots[0], halos);
    gl.depthMask(true);
    gl.disable(gl.DEPTH_TEST);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  private makeSlot(mesh: MeshData, kind: MeshSlot["kind"]): MeshSlot {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    const geometry = gl.createBuffer()!;
    const buffer = gl.createBuffer()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, geometry);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.vertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    const layout: [number, number][] = [
      [2, 3],
      [3, 1],
      [4, 3],
      [5, 1],
      [6, 3],
      [7, 4],
    ];
    let offset = 0;
    for (const [location, size] of layout) {
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, STRIDE * 4, offset);
      gl.vertexAttribDivisor(location, 1);
      offset += size * 4;
    }
    gl.bindVertexArray(null);
    return { kind, vao, buffer, count: mesh.count };
  }

  private flush(slot: MeshSlot, values: number[]): void {
    if (values.length === 0) {
      return;
    }
    const gl = this.gl;
    gl.bindVertexArray(slot.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, slot.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(values), gl.DYNAMIC_DRAW);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, slot.count, values.length / STRIDE);
  }
}

function pushInstance(
  target: number[],
  x: number,
  y: number,
  z: number,
  yaw: number,
  sx: number,
  sy: number,
  sz: number,
  seed: number,
  color: [number, number, number],
  emissive: number,
  roughness: number,
  camo: number,
  wiggle: number,
): void {
  target.push(x, y, z, yaw, sx, sy, sz, seed, color[0], color[1], color[2], emissive, roughness, camo, wiggle);
}

function link(gl: WebGL2RenderingContext, vertex: string, fragment: string): WebGLProgram {
  const program = gl.createProgram()!;
  const vs = compile(gl, gl.VERTEX_SHADER, vertex);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fragment);
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) ?? "creature shader link failed");
  }
  return program;
}

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) ?? "creature shader compile failed");
  }
  return shader;
}
