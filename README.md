# NEAT Simulation

This project simulates the genetic evolution of simple cell organisms using the NEAT algorithm. Organisms compete for survival by consuming food and each other, developing traits to gain advantages.

Two simulation harnesses are available:

- **Episodic** (default) — fixed-length trials with full world resets; optimized for batch NEAT training
- **Living world** — persistent ecology with nutrient-cloud food regrowth, steady-state evolution, and a large pannable viewport ([details](docs/LIVING_WORLD.md))

## Features

- Configurable simulation parameters
- Visual rendering of the simulation
- Genetic evolution using NEAT
- Terminal dashboard for headless mode
- 31-channel neural network inputs (food, prey/threat, breeding — nearest + second-nearest)
- Continuous steering control (angle + speed) with movement trails and click-to-inspect overlays
- **Living world mode**: 4000×4000 persistent arena, draggable camera, minimap, nutrient clouds, organism genome panel

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   For GPU rendering (`render_backend=moderngl`):
   ```bash
   pip install -r requirements-gpu.txt
   ```

2. Run headless episodic training (default):
   ```bash
   python src/main.py
   ```

3. Run the living world (visual):
   ```bash
   python src/main.py harness=living_world
   ```

4. Run episodic mode with visual rendering:
   ```bash
   python src/main.py render=true
   ```

5. Run with GPU renderer (requires OpenGL display):
   ```bash
   python src/main.py render=true render_backend=moderngl
   ```

## Running Options

CLI flags use `key=value` syntax and override the active config JSON (`simulation-config.json` or `living-world-config.json`):

| Flag | Values | Description |
|------|--------|-------------|
| `harness=` | `episodic` / `living_world` | Select simulation harness |
| `render=` | `true` / `false` | Enable or disable rendering |
| `render_backend=` | `pygame` / `moderngl` | Software pygame or GPU ModernGL backend (episodic only) |
| `logging=` | `normal` / `detailed` | Console log verbosity during evaluation |
| `dashboard=` | `minimal` / `normal` / `detailed` | Terminal scoreboard detail level |

**Default episodic training mode is headless** (`"render": false` in JSON) for maximum throughput. The living world defaults to rendering enabled.

```bash
python src/main.py render=true
python src/main.py harness=living_world render=false max_world_steps=8000
python src/main.py logging=detailed dashboard=detailed
```

When rendering episodic mode, `render_stride` in JSON controls how often frames are drawn (default `10` = every 10th sim step).

- **Episodic rendering**: organisms and food fill the window; scoreboard on the right. Click to inspect, **S** for sense rings, **Esc** to clear.
- **Living world rendering**: 900×900 viewport into a 4000×4000 world. Drag to pan, click organism to track, minimap top-right. See [docs/LIVING_WORLD.md](docs/LIVING_WORLD.md).
- **Headless mode**: no visual output; terminal dashboard after each generation/epoch.

## Configuration

- `config/neat-config.ini`: NEAT algorithm settings (**31 network inputs**, 4 outputs)
- `config/simulation-config.json`: Episodic harness (arena size, trials, steps, food count, …)
- `config/living-world-config.json`: Living world harness (world size, viewport, nutrient clouds, steady-state selection, …)

Performance-related JSON keys (episodic):

- `batch_inference` (default `true`): NumPy-compiled forward passes with parity to NEAT `activate`
- `render_backend`: `pygame` (default) or `moderngl` (GPU instanced draw; falls back to pygame)
- `render_stride`: draw every Nth sim step when rendering (default `10`)

Living-world-specific keys include `max_population`, `nutrient_cloud_count`, `food_target_density`, `selection_interval_steps`, and `viewport_width/height`. See [docs/LIVING_WORLD.md](docs/LIVING_WORLD.md) for the full reference.

Legacy JSON keys are still supported:

- `petridish_size` → `environment_width` / `environment_height`
- `episode_length` → `simulation_steps`

### Neural network input schema (31 channels)

| Channels | Content |
|----------|---------|
| 0–8 | Core body/environment (energy, position, movement, size, boundaries) |
| 9–12 | Nearest food (herbivores; zeros for carnivores) |
| 13–15 | Second-nearest food |
| 16–19 | Nearest prey (carnivores) or threats (herbivores) |
| 20–22 | Second-nearest prey/threat |
| 23–26 | Nearest breeding partner |
| 27 | Breeding readiness |
| 28–30 | Second-nearest breeding partner |

### Neural network outputs (4 channels)

| Output | Role |
|--------|------|
| 0 | Steering angle (tanh → full circle) |
| 1 | Speed fraction (tanh → 0..max speed) |
| 2 | Breeding desire (>0.5 triggers attempt) |
| 3 | Rest gate (>0.5 skips movement) |

## Testing

Run the full unit test suite from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

Living-world modules have focused test files:

| Test file | Covers |
|-----------|--------|
| `test_living_world.py` | Food ecology, rolling fitness, harness integration |
| `test_population_registry.py` | Seeding, births, culling, capacity |
| `test_selection_scheduler.py` | Steady-state cull and immigration |
| `test_camera.py` | Viewport pan, drag, coordinate transforms |
| `test_world_clock.py` | Step/epoch counting |
| `test_genome_viz.py` | Genome graph rendering |
| `test_main_harness.py` | Harness config loading |

See [docs/LIVING_WORLD.md](docs/LIVING_WORLD.md) for living-world architecture and config reference.

Optional GPU dependencies (for `render_backend=moderngl`):

```bash
pip install -r requirements-gpu.txt
```

### GPU smoke tests

`tests/test_renderer_gl.py` runs live ModernGL init/render tests **only when OpenGL is available** (display + `libGL`). In headless CI they are **skipped**, not failed.

Check whether GPU tests would run on your machine:

```bash
python3 -c "from tests.gl_support import opengl_available, skip_reason; print(opengl_available(), skip_reason())"
```

### Manual visual checklist (GPU renderer)

Use this when validating `render=true render_backend=moderngl` on a machine with a display:

1. Install GPU deps: `pip install -r requirements-gpu.txt`
2. Start visual run: `python src/main.py render=true render_backend=moderngl`
3. Confirm window title contains **GPU** and arena renders (food dots + colored organisms)
4. Confirm scoreboard panel on the right shows species cards (or “Waiting for species data…” early on)
5. Confirm HUD text top-left updates (`Food`, `Gen`, `Species`) as the sim runs
6. Let one generation complete; verify terminal dashboard still prints
7. Close the window — sim should exit cleanly without traceback
8. Compare with pygame backend: `python src/main.py render=true render_backend=pygame` (same layout, software draw)

If ModernGL fails to init, the factory falls back to pygame and logs  
`ModernGL renderer unavailable (...); falling back to pygame`.

## Project layout

```
config/          NEAT, episodic, and living-world JSON settings
docs/            Architecture docs (including LIVING_WORLD.md)
src/             Simulation runtime (main, organism, simulation, living_world, …)
tests/           Unit tests (distance, network inputs, organism, living world, …)
```

## License

MIT License
