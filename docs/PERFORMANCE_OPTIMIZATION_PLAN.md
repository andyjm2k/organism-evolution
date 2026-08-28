# Performance Optimization Plan

Plan for speeding up the organism-evolution simulation **without changing evolutionary outcomes** — same NEAT semantics, same environment physics, same fitness contract. Focus areas: **GPU-accelerated graphics** and **CPU overhead reduction**.

---

## 1. Workload baseline

Default configuration (`config/simulation-config.json`, `config/neat-config.ini`):

| Parameter | Default | Location |
|-----------|---------|----------|
| Population | 50 | `neat-config.ini` `pop_size` |
| Trials per genome | 3 (hardcoded) | `simulation.py` |
| Steps per trial | 2000 | `simulation_steps` |
| Generations | 2000 | `num_generations` |
| Arena | 1000×1000 | `environment_width/height` |
| Food pellets | 75 | `num_food_items` |
| Render | **true** | `render` |
| NN I/O | 22 in / 8 out | `num_inputs` / `num_outputs` |

**Per-generation eval:** 50 × 3 × 2000 = **300,000 organism-steps**.

With `render=true`, the sim also draws up to **6000 frames/generation** (2000 steps × 3 trials), capped at 60 FPS via `clock.tick(60)`.

---

## 2. CPU vs GPU boundary

### Must remain on CPU (correctness-critical)

These directly affect selection, speciation, and fitness — any change risks altering evolution:

| Component | Why |
|-----------|-----|
| `neat.Population.run`, speciation, stagnation | Genome/species identity and reproduction |
| `configure_crossover`, `mutate` | Topology and weight semantics |
| `FeedForwardNetwork.activate` per organism | Action → movement → survival → fitness |
| Consumption collision order (`break` on first hit) | First-match wins in food/hunt loops |
| Median of 3 trial fitnesses | Eval contract in `eval_genomes` |
| Movement noise (`random.uniform`) | Tie-breaking and exploration |
| Episode-local breeding during eval | Crossover must stay faithful |

### Safe to offload or optimize (display / infrastructure)

| Component | Notes |
|-----------|-------|
| Pygame drawing | Pure visualization; `render=false` already skips it |
| Font rendering, scoreboard cards | Display-only |
| Terminal dashboard / `log_always` output | I/O only |
| Spatial grid bookkeeping | Safe if query results and iteration order match |
| Object allocation patterns | Safe if semantics unchanged |

### GPU applicability summary

```
┌─────────────────────────────────────────────────────────────┐
│  NEAT evolution loop (CPU)                                  │
│  ├── genome eval / crossover / speciation                   │
│  ├── network.activate (CPU today; batchable with validation)│
│  ├── physics / consumption / breeding logic                 │
│  └── fitness aggregation                                    │
├─────────────────────────────────────────────────────────────┤
│  Visualization (GPU candidate)                              │
│  ├── arena background, food dots, organism sprites          │
│  ├── scoreboard panel                                       │
│  └── HUD / generation label                                 │
└─────────────────────────────────────────────────────────────┘
```

Pygame uses SDL software rendering — **all draw calls are CPU-bound**. GPU wins require a different rendering backend or instanced draw path.

---

## 3. Current hot paths

### 3.1 Simulation loop (`src/simulation.py`)

Per step, per trial:

1. **Rebuild two `SpatialGrid` instances** — new objects every step (lines ~190–191); `clear()` exists but is unused.
2. **Per-organism sensing** — grid query + `_nearby_entities` (O(n × local density)); lists allocated via `list()`.
3. **`take_action`** — `build_network_inputs` + `network.activate` for every living organism.
4. **`update`** — `check_for_food` scans **all food** (O(n × f)); `hunt_prey` scans **all organisms** (O(n²) among carnivores).
5. **Dead removal** — `organisms.remove()` inside loop (O(n²) worst case).
6. **`renderer.render()`** when enabled — once per sim step, not decoupled from physics.

### 3.2 Input building (`src/network_inputs.py`)

`_closest_entity` recomputes `math.sqrt` for every candidate in range, even though `within_radius` elsewhere uses squared distance.

### 3.3 Rendering (`src/renderer.py`)

Per frame:

- `pygame.event.get()` — duplicates polling in `simulation.py`.
- Up to 500 food circles + 500 organisms × (1 circle + up to 16 spike lines).
- Debug HUD `font.render` **every frame** (uncached).
- Arena organisms drawn fresh — no surface cache (scoreboard cards do cache).
- `clock.tick(60)` — caps sim throughput when rendering.

### 3.4 Startup overhead (`src/main.py`)

`pygame.init()` runs even in headless mode (`render=false`), initializing subsystems that are never used.

---

## 4. Optimization tiers

### Tier A — Safe CPU wins (no effectiveness impact)

Implement first; each preserves identical simulation semantics.

| ID | Optimization | Target | Expected impact | Effort |
|----|--------------|--------|-----------------|--------|
| A-1 | **Training mode default: `render=false`** | `simulation-config.json`, docs | Largest wall-clock win for evolution runs | Trivial |
| A-2 | **Skip `pygame.init()` when headless** | `main.py` L45 | Faster startup, less idle CPU | Low |
| A-3 | **Reuse spatial grids with `.clear()`** | `simulation.py` L190–191 | Cuts per-step allocation/GC | Low |
| A-4 | **Route `check_for_food` / `hunt_prey` through spatial grids** | `organism.py`, `simulation.py` | Removes O(n×f) and O(n²) scans; same first-hit semantics if iteration order preserved | Medium |
| A-5 | **Use squared distance in `_closest_entity`** | `network_inputs.py` L33–36 | Avoids sqrt in inner loop; identical comparisons | Low |
| A-6 | **Mark dead organisms instead of `list.remove`** | `simulation.py` L198–201 | O(n) per step vs O(n²) | Low |
| A-7 | **Avoid `list()` copies in `_nearby_entities`** | `simulation.py` L117–122 | Less allocation per organism | Low |
| A-8 | **Reuse `seen` set or entity-id bitset in grid query** | `spatial.py` | 50+ set allocations per step at default pop | Low |
| A-9 | **Cache debug HUD surface** | `renderer.py` L141–146 | Eliminates per-frame font render | Low |
| A-10 | **Deduplicate event polling** | `simulation.py` + `renderer.py` | Single `event.get()` per frame | Low |
| A-11 | **Gate `log_always` generation dashboard behind config** | `simulation.py`, `logging_util.py` | Reduces stdout I/O during long runs | Low |
| A-12 | **Make `num_trials` configurable (keep default 3)** | `simulation.py` L168 | Ops tuning only; default unchanged | Low |

**Validation:** Run existing 25 unit tests + A/B fitness comparison on 5 generations with fixed seed (`render=false`, same config).

---

### Tier B — GPU graphics rendering (display-only)

Goal: move **visualization** off the CPU hot path while keeping the sim loop on CPU (or optionally decoupled).

#### B-1. Decouple sim tick from render tick (prerequisite)

| Approach | Description |
|----------|-------------|
| **Sim/render split** | Advance N physics steps per frame, or render every K steps (`render_stride=10` → 200 frames/trial instead of 2000). |
| **Async render thread** | Sim pushes snapshot (positions, colors) to a queue; render thread consumes at 60 FPS. Sim never waits on `clock.tick`. |

**Effectiveness impact:** None on evolution when `render=false`. When rendering for observation, stride only affects visual smoothness, not fitness.

#### B-2. Pygame GPU-adjacent improvements (stay on pygame)

| Approach | Pros | Cons |
|----------|------|------|
| **Pre-render organism sprites to `Surface` cache** | Same API; mirrors scoreboard `get_species_visual` | Still CPU blit; moderate gain |
| **`pygame.transform.scale` batch blits** | Simple | Limited GPU use |
| **SDL2 `SDL_RENDERER_ACCELERATED`** | May use GPU compositing via SDL | Still per-sprite draw calls; pygame exposes this indirectly |

Best pygame-only path: **sprite atlas + cached surfaces + render stride** — low risk, ~2–5× render throughput, no new dependencies.

#### B-3. Modern GPU backend options

| Backend | Fit | Notes |
|---------|-----|-------|
| **ModernGL / PyOpenGL** | Instanced quads/circles for food + organisms | Full GPU control; custom shader for 500+ entities |
| **Vispy / Pyglet** | Batch rendering, GPU buffers | Heavier dependency; good for 1k+ entities |
| **Pygame + moderngl context** | Hybrid: keep pygame window, upload vertex buffer each frame | Practical migration path |
| **CUDA/OpenCL display** | Overkill | Not recommended |

**Recommended GPU architecture:**

```
Simulation (CPU)                    Renderer (GPU)
─────────────────                   ─────────────────
organisms[], food[]  ──snapshot──►  Vertex buffer:
  position, radius,                  [x, y, r, g, b, size, spikes...]
  color, species_id
                                    Instanced draw call:
                                      - 1 shader, 1 VBO update/frame
                                      - food: GL_POINTS
                                      - orgs: instanced triangles/circles
                                    Scoreboard: separate ortho pass (CPU or GPU text)
```

**Libraries to evaluate:** `moderngl`, `moderngl-window`, or `pygame + OpenGL` via `pygame.display.set_mode(..., OPENGL)`.

#### B-4. Scoreboard and text on GPU

Text rendering is expensive on CPU (`font.render` per card). Options:

- **Pre-rasterize species names** to textures when species first seen (cache like `species_surfaces`).
- **Bitmap font atlas** for digits/HUD (single draw for debug overlay).
- **Optional:** drop live scoreboard during fast training; use terminal dashboard only.

---

### Tier C — CPU simulation acceleration (validate carefully)

These can speed eval but need **seeded regression tests** to confirm fitness distributions match.

| ID | Optimization | Risk | Notes |
|----|--------------|------|-------|
| C-1 | **Batch NN inference** (numpy/torch/jax) | Medium | Stack 22×N inputs, run N networks; float order may differ slightly |
| C-2 | **Numba JIT for `build_network_inputs`** | Low–medium | Hot loop in `_closest_entity` |
| C-3 | **Multiprocessing: eval genomes in parallel** | Medium | neat-python is single-process; shard population across workers with deterministic RNG per genome |
| C-4 | **C extension / Rust module for spatial grid** | Low | Same algorithm, faster constant factors |
| C-5 | **Reduce Python attribute lookups in inner loop** | Low | Local variable binding in `eval_genomes` step loop |

**C-1 detail (batch inference):**

Export each `FeedForwardNetwork` to a matrix form (feed-forward only — already enforced by config). For each step:

```python
# Pseudocode — validate against sequential activate
input_matrix = np.zeros((len(alive), 22), dtype=np.float32)
for i, org in enumerate(alive):
    input_matrix[i] = build_network_inputs(org, ...)
output_matrix = batch_activate(networks, input_matrix)  # CPU BLAS or GPU
```

Start with **NumPy on CPU** (OpenBLAS/MKL). Move to **CuPy/PyTorch CUDA** only after correctness parity tests pass.

**Do not batch across trials or change organism processing order** without explicit parity validation.

---

### Tier D — Operational patterns (zero code risk)

| Pattern | When to use |
|---------|-------------|
| Headless training + periodic checkpoint renders | Production evolution runs |
| `dashboard=minimal`, `logging=normal` | Long runs |
| Lower `num_generations` during dev; full run overnight | Iteration vs production |
| `psutil` memory tracking (already in scoreboard) | Catch leaks across generations |
| Profile before optimizing | See §5 |

---

## 5. Measurement methodology

### 5.1 Baseline benchmarks

Record before any change:

```bash
# Headless throughput (primary metric)
python src/main.py render=false dashboard=minimal logging=normal

# With rendering (secondary — frames/sec + gen time)
python src/main.py render=true dashboard=minimal
```

Metrics to capture:

| Metric | How |
|--------|-----|
| ms / generation | `time.perf_counter()` around `eval_genomes` |
| ms / sim step | Sample every 100 steps |
| ms / render frame | Wrap `renderer.render` |
| RSS memory | `psutil` (scoreboard already supports) |
| Fitness parity | Same seed, 5 gens, compare best/median fitness |

### 5.2 Profiling

```python
# Temporary: profile one eval_genomes call
import cProfile, pstats
cProfile.runctx(
    "simulation.eval_genomes(genomes, config)",
    globals(), locals(),
    "eval.prof",
)
pstats.Stats("eval.prof").sort_stats("cumulative").print_stats(30)
```

Use reduced `simulation_steps=200`, `pop_size=20` for profiling iterations only.

### 5.3 Success criteria

| Tier | Target |
|------|--------|
| A (CPU overhead) | ≥30% faster headless ms/generation |
| B (GPU render) | ≥5× frame rate at pop=50, or sim not blocked by 60 FPS cap |
| C (batch NN) | ≥2× step loop with <0.1% fitness deviation vs baseline |
| All | 25/25 unit tests pass; fitness parity within tolerance |

---

## 6. Phased rollout

```
Phase 1 — Quick wins (1–2 days)
├── A-1  render=false for training
├── A-2  skip pygame init headless
├── A-3  reuse spatial grids
├── A-5  squared distance in _closest_entity
├── A-6  dead flags not list.remove
├── A-9  cache debug HUD
└── A-10 dedupe event polling

Phase 2 — Collision path (2–3 days)
├── A-4  spatial grid for food/hunt
├── A-7  reduce list allocations
└── A-8  reuse seen set in grid query

Phase 3 — Render decoupling (3–5 days)
├── B-1  render_stride + sim/render split
├── B-2  pygame sprite cache for arena organisms
└── Benchmark render vs headless

Phase 4 — GPU renderer (1–2 weeks)
├── B-3  moderngl instanced draw prototype
├── B-4  texture cache for scoreboard text
├── Feature flag: render_backend = pygame | moderngl
└── Visual parity checklist (not fitness — display only)

Phase 5 — Optional CPU batch NN (1–2 weeks, gated on parity)
├── C-1  NumPy batch activate
├── Parity test suite (fixed seeds, 10 generations)
└── Only then evaluate CUDA/CuPy
```

---

## 7. Risk matrix

| Change | Effectiveness risk | Performance gain | Recommend |
|--------|-------------------|------------------|-----------|
| `render=false` training | None | Very high | ✅ Now |
| Reuse spatial grids | None | Medium | ✅ Now |
| Grid-based food/hunt | Low (preserve order) | High | ✅ Phase 2 |
| Render stride | None (eval) | High (observation) | ✅ Phase 3 |
| GPU instanced draw | None (eval) | High (visual) | ✅ Phase 4 |
| Batch NN (GPU) | Medium | Very high | ⚠️ After parity tests |
| Reduce trials/steps/pop | **High** | High | ❌ Not in this plan |
| Remove movement noise | **High** | Low | ❌ Avoid |
| Change median → mean fitness | **High** | None | ❌ Avoid |

---

## 8. Dependency considerations

| Addition | Purpose | Required? |
|----------|---------|-----------|
| `moderngl` | GPU instanced rendering | Phase 4 only |
| `numpy` | Batch inference | Phase 5 (likely already transitive) |
| `numba` | JIT hot loops | Optional Phase C-2 |
| `torch` / `cupy` | GPU NN batch | Optional, after CPU batch proven |

Keep `requirements.txt` split:

```
requirements.txt          # core (neat, pygame, psutil)
requirements-gpu.txt      # moderngl, optional torch
```

Headless CI should not require GPU deps.

---

## 9. Summary

**Biggest wins with zero evolutionary impact:**

1. Run training **headless** (`render=false`) — pygame rendering dominates wall time today.
2. **Extend spatial indexing** to food consumption and predation (sensing already uses grids).
3. **Eliminate per-step allocations** (grid reuse, dead flags, iterator reuse).
4. **Decouple render from sim tick** so observation runs do not cap at 60 FPS.

**GPU role:** Graphics only in early phases. The NEAT loop stays on CPU until batch inference passes seeded parity tests. A ModernGL instanced-draw renderer is the recommended path for 500+ entities without changing simulation logic.

**Explicit non-goals:** Reducing trials, steps, or population; changing fitness aggregation; removing RNG noise; altering collision resolution order.

---

## 10. Execution status

| ID | Status | Notes |
|----|--------|-------|
| A-1 | Done | `simulation-config.json` defaults `"render": false`; CLI `render=true` for visuals |
| A-2 | Done | `main.py` skips `pygame.init()` when headless |
| A-3 | Done | Reused `_food_grid` / `_org_grid` with `.clear()` each step |
| A-4 | Done | `organism.update(nearby_food, nearby_organisms)` uses spatial candidates |
| A-5 | Done | `_closest_entity` uses `squared_distance`; one sqrt for winner |
| A-6 | Done | Rebuild `next_organisms` list instead of `list.remove` |
| A-7 | Partial | Nearby lists built once and reused for action + consumption |
| A-8 | Done | `SpatialGrid._seen` reused across queries |
| A-9 | Done | Renderer caches HUD surface keyed by food/gen/species counts |
| A-10 | Done | Event pump only in `Simulation`; renderer no longer drains queue |
| A-12 | Done | `num_trials` configurable (default 3) |
| B-1 | Done | `render_stride` (default 10) draws every Nth step |
| B-2 | Done | Arena organisms blit from cached `get_species_visual` surfaces |
| B-3 | Done | `ModernGLRenderer` + `render_backend=moderngl` (falls back to pygame) |
| B-4 | Done | Scoreboard/HUD rasterized to cached GL textures |
| C-1 | Done | `CompiledNetwork` + `BatchInferenceEngine`; `batch_inference` config flag |
| C-parity | Done | `tests/test_batch_parity.py` seeded fitness match |
