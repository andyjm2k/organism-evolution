/** Unit meshes for instanced creatures. Positions and normals are interleaved. */

export interface MeshData {
  vertices: Float32Array;
  count: number;
}

export function sphereMesh(latBands = 10, lonBands = 12): MeshData {
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

/** Cone along +X, from x=-1 (base) to x=1 (tip), radius 1. */
export function coneMesh(sides = 10): MeshData {
  const data: number[] = [];
  for (let index = 0; index < sides; index += 1) {
    const a0 = (index / sides) * Math.PI * 2;
    const a1 = ((index + 1) / sides) * Math.PI * 2;
    const y0 = Math.cos(a0);
    const z0 = Math.sin(a0);
    const y1 = Math.cos(a1);
    const z1 = Math.sin(a1);
    pushVertex(data, 1, 0, 0, 0.55, y0, z0);
    pushVertex(data, -1, y0, z0, 0.55, y0, z0);
    pushVertex(data, -1, y1, z1, 0.55, y1, z1);
    pushVertex(data, -1, 0, 0, -1, 0, 0);
    pushVertex(data, -1, y1, z1, -1, 0, 0);
    pushVertex(data, -1, y0, z0, -1, 0, 0);
  }
  return pack(data);
}

/** Box from -1 to 1 on each axis. */
export function boxMesh(): MeshData {
  const data: number[] = [];
  const faces: [number, number, number, number[][]][] = [
    [1, 0, 0, [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]]],
    [-1, 0, 0, [[-1, 1, -1], [-1, -1, -1], [-1, -1, 1], [-1, 1, 1]]],
    [0, 1, 0, [[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]]],
    [0, -1, 0, [[-1, -1, 1], [1, -1, 1], [1, -1, -1], [-1, -1, -1]]],
    [0, 0, 1, [[-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]]],
    [0, 0, -1, [[1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]]],
  ];
  for (const [nx, ny, nz, corners] of faces) {
    const [a, b, c, d] = corners;
    pushVertex(data, a[0], a[1], a[2], nx, ny, nz);
    pushVertex(data, b[0], b[1], b[2], nx, ny, nz);
    pushVertex(data, c[0], c[1], c[2], nx, ny, nz);
    pushVertex(data, a[0], a[1], a[2], nx, ny, nz);
    pushVertex(data, c[0], c[1], c[2], nx, ny, nz);
    pushVertex(data, d[0], d[1], d[2], nx, ny, nz);
  }
  return pack(data);
}

function spherePoint(theta: number, phi: number): [number, number, number] {
  const ring = Math.sin(theta);
  return [ring * Math.cos(phi), ring * Math.sin(phi), Math.cos(theta)];
}

function pushTriangle(data: number[], a: [number, number, number], b: [number, number, number], c: [number, number, number]): void {
  pushVertex(data, a[0], a[1], a[2], a[0], a[1], a[2]);
  pushVertex(data, b[0], b[1], b[2], b[0], b[1], b[2]);
  pushVertex(data, c[0], c[1], c[2], c[0], c[1], c[2]);
}

function pushVertex(data: number[], x: number, y: number, z: number, nx: number, ny: number, nz: number): void {
  const length = Math.hypot(nx, ny, nz) || 1;
  data.push(x, y, z, nx / length, ny / length, nz / length);
}

function pack(data: number[]): MeshData {
  return { vertices: new Float32Array(data), count: data.length / 6 };
}
