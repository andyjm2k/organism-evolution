/** CPython random.Random (MT19937) so a seeded world matches the desktop sim. */

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER = 0x80000000;
const LOWER = 0x7fffffff;

export class PythonRandom {
  /** MT state words. */
  private mt = new Uint32Array(N);
  /** Next unused word. */
  private index = N;
  /** Spare Box-Muller sample, matching random.Random.gauss_next. */
  private gaussNext: number | null = null;

  constructor(seed: number | null = null) {
    if (seed !== null) {
      this.seed(seed);
    }
  }

  /** Seed from a Python int. Negative seeds use the absolute value. */
  seed(value: number): void {
    const abs = Math.abs(Math.trunc(value));
    const key = abs === 0 ? [0] : splitUint32(abs);
    this.initByArray(key);
    this.gaussNext = null;
  }

  /** Snapshot the generator so a saved world can resume. */
  exportState(): { mt: number[]; index: number; gaussNext: number | null } {
    return { mt: Array.from(this.mt), index: this.index, gaussNext: this.gaussNext };
  }

  /** Restore a snapshot from exportState. */
  importState(state: { mt: number[]; index: number; gaussNext: number | null }): void {
    this.mt.set(state.mt);
    this.index = state.index;
    this.gaussNext = state.gaussNext;
  }

  /** Float in [0, 1), identical to random.random. */
  random(): number {
    const a = this.genrand() >>> 5;
    const b = this.genrand() >>> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
  }

  /** random.uniform(a, b). */
  uniform(a: number, b: number): number {
    return a + (b - a) * this.random();
  }

  /** random.gauss(mu, sigma), including the cached second sample. */
  gauss(mu: number, sigma: number): number {
    let z = this.gaussNext;
    this.gaussNext = null;
    if (z === null) {
      const x2pi = this.random() * (2 * Math.PI);
      const g2rad = Math.sqrt(-2.0 * Math.log(1.0 - this.random()));
      z = Math.cos(x2pi) * g2rad;
      this.gaussNext = Math.sin(x2pi) * g2rad;
    }
    return mu + z * sigma;
  }

  /** random.randrange(start, stop) with step 1. stop is exclusive. */
  randrange(start: number, stop: number): number {
    const width = stop - start;
    if (width <= 0) {
      throw new Error(`empty range ${start}..${stop}`);
    }
    return start + this.randbelow(width);
  }

  /** random.randint(a, b), inclusive. */
  randint(a: number, b: number): number {
    return this.randrange(a, b + 1);
  }

  /** random.choice. CPython 3.12 indexes with _randbelow, not random(). */
  choice<T>(seq: readonly T[]): T {
    if (seq.length === 0) {
      throw new Error("choice from empty sequence");
    }
    return seq[this.randbelow(seq.length)];
  }

  /** random.choices with weights, k samples with replacement. */
  choices<T>(population: readonly T[], weights: readonly number[], k: number): T[] {
    const cum: number[] = [];
    let total = 0;
    for (const weight of weights) {
      total += weight;
      cum.push(total);
    }
    const hi = cum.length - 1;
    const picked: T[] = [];
    for (let i = 0; i < k; i += 1) {
      picked.push(population[bisectRight(cum, this.random() * total, 0, hi)]);
    }
    return picked;
  }

  /** Unsigned 32-bit MT output. */
  private genrand(): number {
    if (this.index >= N) {
      this.twist();
      this.index = 0;
    }
    let y = this.mt[this.index];
    this.index += 1;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** Advance the twist when the buffer is exhausted. */
  private twist(): void {
    for (let kk = 0; kk < N; kk += 1) {
      const y = (this.mt[kk] & UPPER) | (this.mt[(kk + 1) % N] & LOWER);
      let next = this.mt[(kk + M) % N] ^ (y >>> 1);
      if (y & 1) {
        next ^= MATRIX_A;
      }
      this.mt[kk] = next >>> 0;
    }
  }

  /** init_by_array from CPython's _random. */
  private initByArray(key: number[]): void {
    this.initGenrand(19650218);
    let i = 1;
    let j = 0;
    const keyLength = key.length;
    let k = N > keyLength ? N : keyLength;
    for (; k > 0; k -= 1) {
      const prev = this.mt[i - 1];
      const mixed = Math.imul((prev ^ (prev >>> 30)) >>> 0, 1664525) >>> 0;
      this.mt[i] = (this.mt[i] ^ mixed) + key[j] + j >>> 0;
      // The addition above is (a + b + c) >>> 0 because + and >>> associate left.
      i += 1;
      j += 1;
      if (i >= N) {
        this.mt[0] = this.mt[N - 1];
        i = 1;
      }
      if (j >= keyLength) {
        j = 0;
      }
    }
    for (k = N - 1; k > 0; k -= 1) {
      const prev = this.mt[i - 1];
      const mixed = Math.imul((prev ^ (prev >>> 30)) >>> 0, 1566083941) >>> 0;
      this.mt[i] = ((this.mt[i] ^ mixed) - i) >>> 0;
      i += 1;
      if (i >= N) {
        this.mt[0] = this.mt[N - 1];
        i = 1;
      }
    }
    this.mt[0] = UPPER;
    this.index = N;
  }

  /** init_genrand. */
  private initGenrand(seed: number): void {
    this.mt[0] = seed >>> 0;
    for (let mti = 1; mti < N; mti += 1) {
      const prev = this.mt[mti - 1];
      this.mt[mti] = (Math.imul(1812433253, (prev ^ (prev >>> 30)) >>> 0) + mti) >>> 0;
    }
    this.index = N;
  }

  /** random._randbelow. */
  private randbelow(n: number): number {
    const k = bitLength(n);
    let r = this.getrandbits(k);
    while (r >= n) {
      r = this.getrandbits(k);
    }
    return r;
  }

  /** random.getrandbits for widths up to 53. */
  private getrandbits(k: number): number {
    if (k <= 0) {
      return 0;
    }
    if (k <= 32) {
      return this.genrand() >>> (32 - k);
    }
    const words = Math.ceil(k / 32);
    let result = 0;
    let shift = 0;
    for (let i = 0; i < words; i += 1) {
      const bits = Math.min(32, k - shift);
      const word = this.genrand() >>> (32 - bits);
      result += word * 2 ** shift;
      shift += 32;
    }
    return result;
  }
}

/** Little-endian 32-bit chunks of a non-negative integer, matching _PyLong_AsByteArray. */
function splitUint32(value: number): number[] {
  const chunks: number[] = [];
  let rest = value;
  while (rest > 0) {
    chunks.push(rest >>> 0);
    rest = Math.floor(rest / 4294967296);
  }
  return chunks.length === 0 ? [0] : chunks;
}

/** Number of bits in a positive integer. */
function bitLength(n: number): number {
  return Math.floor(Math.log2(n)) + 1;
}

/** bisect.bisect_right on a sorted array, hi exclusive of the last index used by choices. */
function bisectRight(cum: number[], x: number, lo: number, hi: number): number {
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (x < cum[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}
