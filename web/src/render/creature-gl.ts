/** Lit instanced creatures on the orthographic living-world camera. */

import { creatureParts, visualRadius, type CreaturePose } from "./creature-layout";
import { MESH_KINDS, meshFor, type MeshKind } from "./creature-mesh";
import type { BodySnapshot } from "../sim/world";

const STRIDE = 17;

const VERTEX = `#version 300 es
layout(location=0) in vec3 a_pos;
layout(location=1) in vec3 a_nrm;
layout(location=2) in vec3 i_origin;
layout(location=3) in vec2 i_orient;
layout(location=4) in vec3 i_scale;
layout(location=5) in float i_seed;
layout(location=6) in vec3 i_color;
layout(location=7) in vec4 i_mat;
layout(location=8) in float i_mark;
uniform vec2 u_view;
uniform vec2 u_camera;
uniform float u_time;
out vec3 v_color;
out vec3 v_nrm;
out vec3 v_world;
out vec3 v_local;
out vec4 v_mat;
out float v_mark;
vec3 pitched(vec3 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return vec3(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}
vec3 yawed(vec3 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}
void main() {
  vec3 shaped = a_pos;
  float wave = sin(u_time * 6.4 + i_seed) * i_mat.w;
  shaped.y += wave * shaped.x * shaped.x;
  vec3 local = yawed(pitched(shaped * i_scale, i_orient.y), i_orient.x);
  vec3 world = i_origin + local;
  vec2 screen = world.xy - u_camera;
  vec2 ndc = (screen / u_view) * 2.0 - 1.0;
  ndc.y = -ndc.y;
  gl_Position = vec4(ndc, 0.84 - world.z * 0.01, 1.0);
  v_nrm = normalize(yawed(pitched(a_nrm, i_orient.y), i_orient.x));
  v_color = i_color;
  v_world = world;
  v_local = shaped;
  v_mat = i_mat;
  v_mark = i_mark;
}`;

const FRAGMENT = `#version 300 es
precision mediump float;
in vec3 v_color;
in vec3 v_nrm;
in vec3 v_world;
in vec3 v_local;
in vec4 v_mat;
in float v_mark;
uniform float u_halo;
out vec4 fragColor;
void main() {
  vec3 n = normalize(v_nrm);
  vec3 key = normalize(vec3(-0.42, -0.28, 0.86));
  float wrap = n.z * 0.35 + max(dot(n, key), 0.0) * 0.65;
  vec3 hemi = mix(vec3(0.03, 0.06, 0.08), vec3(0.78, 0.86, 0.82), clamp(n.z * 0.5 + 0.5, 0.0, 1.0));
  vec3 halfV = normalize(key + vec3(0.0, 0.0, 1.0));
  float gloss = mix(72.0, 8.0, v_mat.y);
  float spec = pow(max(dot(n, halfV), 0.0), gloss) * (1.0 - v_mat.y);
  float rim = pow(1.0 - max(n.z, 0.0), 1.55);
  vec3 albedo = v_color;
  float mark = floor(v_mark + 0.1);
  if (mark < 0.5) {
  } else if (mark < 1.5) {
    float pore = 0.5 + 0.5 * sin(v_local.x * 10.0) * sin(v_local.y * 12.0);
    albedo *= 0.94 + 0.08 * pore;
  } else if (mark < 2.5) {
    float scale = 0.5 + 0.5 * sin(v_local.x * 9.0 + sin(v_local.y * 8.0));
    albedo = mix(albedo * 0.82, albedo * 1.06, scale);
  } else if (mark < 3.5) {
    float ridge = 0.55 + 0.45 * abs(sin(v_local.x * 14.0));
    albedo *= ridge;
    n = normalize(n + vec3(0.0, 0.0, (ridge - 0.7) * 0.4));
  } else if (mark < 4.5) {
    albedo = mix(albedo, vec3(1.0), 0.12);
    spec *= 1.5;
  } else if (mark < 5.5) {
    albedo *= 1.15;
    spec += 0.2;
  } else if (mark < 6.5) {
    float vein = abs(sin(v_local.x * 11.0 + v_local.y * 7.0));
    albedo = mix(albedo * 0.7, albedo * 1.15, vein);
  } else if (mark < 7.5) {
    float band = 0.5 + 0.5 * sin(atan(v_local.y, v_local.x) * 8.0);
    albedo = mix(albedo * 0.75, albedo * 1.12, band);
  } else {
    float bevel = smoothstep(0.75, 1.0, max(abs(v_local.x), abs(v_local.y)));
    albedo = mix(albedo, albedo * 0.55, bevel);
  }
  float camo = v_mat.z;
  float stripe = 0.5 + 0.5 * sin(v_world.x * 0.85 + v_local.y * 18.0);
  albedo = mix(albedo, albedo * vec3(0.42, 0.55, 0.38), camo * stripe);
  vec3 lit = albedo * (hemi * 0.55 + vec3(1.0, 0.95, 0.88) * (0.16 + wrap));
  lit += vec3(1.0, 0.97, 0.9) * spec * 0.6;
  lit += vec3(0.45, 0.75, 0.7) * rim * 0.35;
  lit += albedo * v_mat.x;
  if (u_halo > 0.5) {
    fragColor = vec4(albedo * (0.45 + v_mat.x), 0.28);
    return;
  }
  fragColor = vec4(lit, 1.0);
}`;

interface MeshSlot {
  kind: MeshKind;
  vao: WebGLVertexArrayObject;
  buffer: WebGLBuffer;
  count: number;
}

export class CreaturePass {
  private program: WebGLProgram;
  private slots: MeshSlot[];

  constructor(private gl: WebGL2RenderingContext) {
    this.program = link(gl, VERTEX, FRAGMENT);
    this.slots = MESH_KINDS.map((kind) => this.makeSlot(meshFor(kind), kind));
  }

  draw(
    bodies: BodySnapshot[],
    selectedId: number | null,
    camera: { offsetX: number; offsetY: number; worldToScreen(x: number, y: number): [number, number] },
    width: number,
    height: number,
    time: number,
  ): void {
    const buckets = Object.fromEntries(MESH_KINDS.map((kind) => [kind, [] as number[]])) as Record<MeshKind, number[]>;
    const halos: number[] = [];
    for (const body of bodies) {
      const [sx, sy] = camera.worldToScreen(body.x, body.y);
      if (sx < -120 || sy < -120 || sx > width + 120 || sy > height + 120) {
        continue;
      }
      const radius = visualRadius(body.radius);
      pushInstance(buckets.sphere, body.x + 3, body.y + 4, 0.35, 0, 0, radius * 0.95, radius * 0.7, Math.max(0.5, radius * 0.05), body.id, [0.01, 0.02, 0.03], 0, 1, 0, 0, 0);
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
          piece.pitch,
          piece.sx * radius,
          piece.sy * radius,
          piece.sz * radius,
          body.id * 0.17 + index,
          piece.color,
          piece.emissive,
          piece.roughness,
          piece.camo,
          piece.wiggle,
          piece.mark,
        );
        if (piece.halo) {
          pushInstance(halos, x, y, z, heading + piece.yaw, piece.pitch, piece.sx * radius * 2.5, piece.sy * radius * 2.5, piece.sz * radius * 1.5, body.id, piece.color, 1, 0.2, 0, 0, 5);
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

  private makeSlot(mesh: ReturnType<typeof meshFor>, kind: MeshKind): MeshSlot {
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
      [3, 2],
      [4, 3],
      [5, 1],
      [6, 3],
      [7, 4],
      [8, 1],
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
  pitch: number,
  sx: number,
  sy: number,
  sz: number,
  seed: number,
  color: [number, number, number],
  emissive: number,
  roughness: number,
  camo: number,
  wiggle: number,
  mark: number,
): void {
  target.push(x, y, z, yaw, pitch, sx, sy, sz, seed, color[0], color[1], color[2], emissive, roughness, camo, wiggle, mark);
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
