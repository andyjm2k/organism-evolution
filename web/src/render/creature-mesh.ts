/** Unit meshes for instanced creatures. Positions and normals are interleaved. */

export interface MeshData {
  vertices: Float32Array;
  count: number;
}

export type MeshKind = "sphere" | "capsule" | "cone" | "fin" | "dome" | "tube" | "claw" | "plate" | "ring";

export const MESH_KINDS: MeshKind[] = ["sphere", "capsule", "cone", "fin", "dome", "tube", "claw", "plate", "ring"];

export function meshFor(kind: MeshKind): MeshData {
  switch (kind) {
    case "capsule":
      return capsuleMesh();
    case "cone":
      return coneMesh(16);
    case "fin":
      return finMesh();
    case "dome":
      return domeMesh();
    case "tube":
      return tubeMesh(14);
    case "claw":
      return clawMesh();
    case "plate":
      return plateMesh();
    case "ring":
      return ringMesh();
    default:
      return sphereMesh(16, 20);
  }
}

export function sphereMesh(latBands = 12, lonBands = 16): MeshData {
  const data: number[] = [];
  for (let lat = 0; lat < latBands; lat += 1) {
    const t0 = (lat / latBands) * Math.PI;
    const t1 = ((lat + 1) / latBands) * Math.PI;
    for (let lon = 0; lon < lonBands; lon += 1) {
      const p0 = (lon / lonBands) * Math.PI * 2;
      const p1 = ((lon + 1) / lonBands) * Math.PI * 2;
      const a = spherePoint(t0, p0);
      const b = spherePoint(t1, p0);
      const c = spherePoint(t0, p1);
      const d = spherePoint(t1, p1);
      pushTriangle(data, a, b, c);
      pushTriangle(data, c, b, d);
    }
  }
  return pack(data);
}

/** Capsule along +X from -1 to 1 with round caps. */
export function capsuleMesh(sides = 12, rings = 12): MeshData {
  const data: number[] = [];
  for (let ring = 0; ring < rings; ring += 1) {
    const u0 = ring / rings;
    const u1 = (ring + 1) / rings;
    for (let side = 0; side < sides; side += 1) {
      const a0 = (side / sides) * Math.PI * 2;
      const a1 = ((side + 1) / sides) * Math.PI * 2;
      pushTriangle(data, capsulePoint(u0, a0), capsulePoint(u1, a0), capsulePoint(u0, a1));
      pushTriangle(data, capsulePoint(u0, a1), capsulePoint(u1, a0), capsulePoint(u1, a1));
    }
  }
  return pack(data);
}

/** Sharp cone along +X, tip at +1. */
export function coneMesh(sides = 16): MeshData {
  const data: number[] = [];
  for (let index = 0; index < sides; index += 1) {
    const a0 = (index / sides) * Math.PI * 2;
    const a1 = ((index + 1) / sides) * Math.PI * 2;
    const y0 = Math.cos(a0);
    const z0 = Math.sin(a0);
    const y1 = Math.cos(a1);
    const z1 = Math.sin(a1);
    const midY = (y0 + y1) * 0.5;
    const midZ = (z0 + z1) * 0.5;
    pushVertex(data, 1, 0, 0, 0.85, midY, midZ);
    pushVertex(data, -1, y0, z0, 0.2, y0, z0);
    pushVertex(data, -1, y1, z1, 0.2, y1, z1);
    pushVertex(data, -1, 0, 0, -1, 0, 0);
    pushVertex(data, -1, y1, z1, -1, 0, 0);
    pushVertex(data, -1, y0, z0, -1, 0, 0);
  }
  return pack(data);
}

/** Flat sail in the XY plane, tip toward -X. */
export function finMesh(): MeshData {
  const data: number[] = [];
  const ribs: [number, number][] = [
    [0.2, 0],
    [0.05, 0.35],
    [-0.25, 0.85],
    [-0.7, 1],
    [-1, 0.55],
    [-1, -0.55],
    [-0.7, -1],
    [-0.25, -0.85],
    [0.05, -0.35],
  ];
  const top = ribs.map(([x, y]) => [x, y, 0.05] as [number, number, number]);
  const bottom = ribs.map(([x, y]) => [x, y, -0.05] as [number, number, number]);
  for (let index = 1; index < ribs.length - 1; index += 1) {
    pushTriangle(data, top[0], top[index], top[index + 1]);
    pushTriangle(data, bottom[0], bottom[index + 1], bottom[index]);
  }
  for (let index = 0; index < ribs.length; index += 1) {
    const next = (index + 1) % ribs.length;
    pushTriangle(data, top[index], bottom[index], top[next]);
    pushTriangle(data, top[next], bottom[index], bottom[next]);
  }
  return pack(data);
}

/** Ridged dome, open on -Z, crest toward +Z. */
export function domeMesh(latBands = 8, lonBands = 18): MeshData {
  const data: number[] = [];
  for (let lat = 0; lat < latBands; lat += 1) {
    const t0 = (lat / latBands) * (Math.PI * 0.5);
    const t1 = ((lat + 1) / latBands) * (Math.PI * 0.5);
    for (let lon = 0; lon < lonBands; lon += 1) {
      const p0 = (lon / lonBands) * Math.PI * 2;
      const p1 = ((lon + 1) / lonBands) * Math.PI * 2;
      const ridge0 = 1 + 0.08 * Math.cos(lon * 3);
      const ridge1 = 1 + 0.08 * Math.cos((lon + 1) * 3);
      const a = domePoint(t0, p0, ridge0);
      const b = domePoint(t1, p0, ridge0);
      const c = domePoint(t0, p1, ridge1);
      const d = domePoint(t1, p1, ridge1);
      pushTriangle(data, a, b, c);
      pushTriangle(data, c, b, d);
    }
  }
  return pack(data);
}

/** Hollow tube along +X from -1 to 1. */
export function tubeMesh(sides = 14): MeshData {
  const data: number[] = [];
  const outer = 1;
  const inner = 0.55;
  for (let index = 0; index < sides; index += 1) {
    const a0 = (index / sides) * Math.PI * 2;
    const a1 = ((index + 1) / sides) * Math.PI * 2;
    const o0: [number, number, number] = [-1, Math.cos(a0) * outer, Math.sin(a0) * outer];
    const o1: [number, number, number] = [-1, Math.cos(a1) * outer, Math.sin(a1) * outer];
    const o2: [number, number, number] = [1, Math.cos(a0) * outer, Math.sin(a0) * outer];
    const o3: [number, number, number] = [1, Math.cos(a1) * outer, Math.sin(a1) * outer];
    pushLit(data, o0, o2, o1, [0, Math.cos(a0), Math.sin(a0)]);
    pushLit(data, o1, o2, o3, [0, Math.cos(a1), Math.sin(a1)]);
    const i0: [number, number, number] = [-1, Math.cos(a0) * inner, Math.sin(a0) * inner];
    const i1: [number, number, number] = [-1, Math.cos(a1) * inner, Math.sin(a1) * inner];
    const i2: [number, number, number] = [1, Math.cos(a0) * inner, Math.sin(a0) * inner];
    const i3: [number, number, number] = [1, Math.cos(a1) * inner, Math.sin(a1) * inner];
    pushLit(data, i0, i1, i2, [0, -Math.cos(a0), -Math.sin(a0)]);
    pushLit(data, i1, i3, i2, [0, -Math.cos(a1), -Math.sin(a1)]);
    pushLit(data, o2, i2, o3, [1, 0, 0]);
    pushLit(data, o3, i2, i3, [1, 0, 0]);
    pushLit(data, o0, o1, i0, [-1, 0, 0]);
    pushLit(data, o1, i1, i0, [-1, 0, 0]);
  }
  return pack(data);
}

/** Hooked claw, tip toward +X and curling in +Y. */
export function clawMesh(): MeshData {
  const data: number[] = [];
  const path: [number, number, number][] = [
    [-1, 0, 0],
    [-0.35, 0.05, 0],
    [0.2, 0.2, 0],
    [0.65, 0.45, 0],
    [0.95, 0.75, 0],
    [0.7, 1, 0],
  ];
  const radii = [0.42, 0.36, 0.3, 0.22, 0.14, 0.05];
  const sides = 8;
  for (let segment = 0; segment < path.length - 1; segment += 1) {
    for (let side = 0; side < sides; side += 1) {
      const a0 = (side / sides) * Math.PI * 2;
      const a1 = ((side + 1) / sides) * Math.PI * 2;
      const p0 = clawRing(path[segment], path[segment + 1], radii[segment], a0);
      const p1 = clawRing(path[segment], path[segment + 1], radii[segment], a1);
      const p2 = clawRing(path[segment + 1], path[Math.min(segment + 2, path.length - 1)], radii[segment + 1], a0);
      const p3 = clawRing(path[segment + 1], path[Math.min(segment + 2, path.length - 1)], radii[segment + 1], a1);
      pushTriangle(data, p0, p2, p1);
      pushTriangle(data, p1, p2, p3);
    }
  }
  return pack(data);
}

/** Beveled armor plate in the XY plane. */
export function plateMesh(): MeshData {
  const data: number[] = [];
  const inset = 0.72;
  const top = 0.18;
  const mid = 0.05;
  const outer: [number, number][] = [
    [-1, -0.85],
    [-0.2, -1],
    [1, -0.7],
    [1, 0.7],
    [-0.2, 1],
    [-1, 0.85],
  ];
  const inner = outer.map(([x, y]) => [x * inset, y * inset] as [number, number]);
  for (let index = 1; index < inner.length - 1; index += 1) {
    pushLit(data, [inner[0][0], inner[0][1], top], [inner[index][0], inner[index][1], top], [inner[index + 1][0], inner[index + 1][1], top], [0, 0, 1]);
  }
  for (let index = 0; index < outer.length; index += 1) {
    const next = (index + 1) % outer.length;
    const o0: [number, number, number] = [outer[index][0], outer[index][1], mid];
    const o1: [number, number, number] = [outer[next][0], outer[next][1], mid];
    const i0: [number, number, number] = [inner[index][0], inner[index][1], top];
    const i1: [number, number, number] = [inner[next][0], inner[next][1], top];
    pushTriangle(data, o0, i0, o1);
    pushTriangle(data, o1, i0, i1);
    const b0: [number, number, number] = [outer[index][0], outer[index][1], -mid];
    const b1: [number, number, number] = [outer[next][0], outer[next][1], -mid];
    pushLit(data, o0, o1, b0, [outer[index][0], outer[index][1], 0]);
    pushLit(data, o1, b1, b0, [outer[next][0], outer[next][1], 0]);
  }
  return pack(data);
}

/** Flat ring in the XY plane for irises and eyespots. */
export function ringMesh(sides = 20): MeshData {
  const data: number[] = [];
  const outer = 1;
  const inner = 0.55;
  for (let index = 0; index < sides; index += 1) {
    const a0 = (index / sides) * Math.PI * 2;
    const a1 = ((index + 1) / sides) * Math.PI * 2;
    const o0: [number, number, number] = [Math.cos(a0) * outer, Math.sin(a0) * outer, 0.04];
    const o1: [number, number, number] = [Math.cos(a1) * outer, Math.sin(a1) * outer, 0.04];
    const i0: [number, number, number] = [Math.cos(a0) * inner, Math.sin(a0) * inner, 0.04];
    const i1: [number, number, number] = [Math.cos(a1) * inner, Math.sin(a1) * inner, 0.04];
    pushLit(data, o0, i0, o1, [0, 0, 1]);
    pushLit(data, o1, i0, i1, [0, 0, 1]);
    const bo0: [number, number, number] = [o0[0], o0[1], -0.04];
    const bo1: [number, number, number] = [o1[0], o1[1], -0.04];
    const bi0: [number, number, number] = [i0[0], i0[1], -0.04];
    const bi1: [number, number, number] = [i1[0], i1[1], -0.04];
    pushLit(data, bo0, bo1, bi0, [0, 0, -1]);
    pushLit(data, bo1, bi1, bi0, [0, 0, -1]);
  }
  return pack(data);
}

function capsulePoint(u: number, angle: number): [number, number, number] {
  // Straight midsection with hemispherical caps, matching a classic capsule.
  const cap = 0.35;
  let x: number;
  let radius: number;
  if (u < cap) {
    const t = (u / cap) * Math.PI * 0.5;
    x = -1 + cap + Math.sin(t - Math.PI * 0.5) * cap;
    radius = Math.cos(t - Math.PI * 0.5);
  } else if (u > 1 - cap) {
    const t = ((u - (1 - cap)) / cap) * Math.PI * 0.5;
    x = 1 - cap + Math.sin(t) * cap;
    radius = Math.cos(t);
  } else {
    x = -1 + cap + ((u - cap) / (1 - 2 * cap)) * (2 - 2 * cap);
    radius = 1;
  }
  return [x, Math.cos(angle) * radius, Math.sin(angle) * radius];
}

function domePoint(theta: number, phi: number, ridge: number): [number, number, number] {
  const ring = Math.sin(theta) * ridge;
  return [ring * Math.cos(phi), ring * Math.sin(phi), Math.cos(theta) * ridge];
}

function clawRing(
  point: [number, number, number],
  next: [number, number, number],
  radius: number,
  angle: number,
): [number, number, number] {
  const tx = next[0] - point[0];
  const ty = next[1] - point[1];
  const tz = next[2] - point[2];
  const length = Math.hypot(tx, ty, tz) || 1;
  const dx = tx / length;
  const dy = ty / length;
  const dz = tz / length;
  let bx = -dy;
  let by = dx;
  let bz = 0;
  if (Math.hypot(bx, by, bz) < 0.001) {
    bx = 0;
    by = -dz;
    bz = dy;
  }
  const bLen = Math.hypot(bx, by, bz) || 1;
  bx /= bLen;
  by /= bLen;
  bz /= bLen;
  const cx = dy * bz - dz * by;
  const cy = dz * bx - dx * bz;
  const cz = dx * by - dy * bx;
  return [
    point[0] + (bx * Math.cos(angle) + cx * Math.sin(angle)) * radius,
    point[1] + (by * Math.cos(angle) + cy * Math.sin(angle)) * radius,
    point[2] + (bz * Math.cos(angle) + cz * Math.sin(angle)) * radius,
  ];
}

function spherePoint(theta: number, phi: number): [number, number, number] {
  const ring = Math.sin(theta);
  return [ring * Math.cos(phi), ring * Math.sin(phi), Math.cos(theta)];
}

function pushTriangle(data: number[], a: [number, number, number], b: [number, number, number], c: [number, number, number]): void {
  const nx = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]);
  const ny = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
  const nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const length = Math.hypot(nx, ny, nz) || 1;
  const normal: [number, number, number] = [nx / length, ny / length, nz / length];
  pushLit(data, a, b, c, normal);
}

function pushLit(data: number[], a: [number, number, number], b: [number, number, number], c: [number, number, number], normal: [number, number, number]): void {
  const length = Math.hypot(normal[0], normal[1], normal[2]) || 1;
  const nx = normal[0] / length;
  const ny = normal[1] / length;
  const nz = normal[2] / length;
  pushVertex(data, a[0], a[1], a[2], nx, ny, nz);
  pushVertex(data, b[0], b[1], b[2], nx, ny, nz);
  pushVertex(data, c[0], c[1], c[2], nx, ny, nz);
}

function pushVertex(data: number[], x: number, y: number, z: number, nx: number, ny: number, nz: number): void {
  const length = Math.hypot(nx, ny, nz) || 1;
  data.push(x, y, z, nx / length, ny / length, nz / length);
}

function pack(data: number[]): MeshData {
  return { vertices: new Float32Array(data), count: data.length / 6 };
}
