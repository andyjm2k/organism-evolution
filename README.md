# NEAT Simulation

This project simulates the genetic evolution of simple cell organisms using the NEAT algorithm. Organisms compete for survival in a petri dish by consuming food and each other, developing traits to gain advantages.

## Features

- Configurable simulation parameters
- Visual rendering of the simulation
- Genetic evolution using NEAT
- Terminal dashboard for headless mode
- 31-channel neural network inputs (food, prey/threat, breeding — nearest + second-nearest)
- Continuous steering control (angle + speed) with movement trails and click-to-inspect overlays

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   For GPU rendering (`render_backend=moderngl`):
   ```bash
   pip install -r requirements-gpu.txt
   ```

2. Run headless training (default from JSON):
   ```bash
   python src/main.py
   ```

3. Run with visual rendering:
   ```bash
   python src/main.py render=true
   ```

4. Run with GPU renderer (requires OpenGL display):
   ```bash
   python src/main.py render=true render_backend=moderngl
   ```

## Running Options

CLI flags use `key=value` syntax and override `config/simulation-config.json`:

| Flag | Values | Description |
|------|--------|-------------|
| `render=` | `true` / `false` | Enable or disable rendering |
| `render_backend=` | `pygame` / `moderngl` | Software pygame or GPU ModernGL backend |
| `logging=` | `normal` / `detailed` | Console log verbosity during evaluation |
| `dashboard=` | `minimal` / `normal` / `detailed` | Terminal scoreboard detail level |

**Default training mode is headless** (`"render": false` in JSON) for maximum throughput. Enable visuals explicitly:

```bash
python src/main.py render=true
python src/main.py render=false dashboard=minimal
python src/main.py logging=detailed dashboard=detailed
```

When rendering, `render_stride` in JSON controls how often frames are drawn (default `10` = every 10th sim step), so observation does not force a draw on every physics tick.

- **With Rendering**: The simulation displays organisms and food in the environment, plus an in-window scoreboard of top species.
- **Without Rendering (Headless Mode)**: Runs without visual output and prints a terminal dashboard after each generation. Useful for faster runs or servers without a display.

## Configuration

- `config/neat-config.ini`: NEAT algorithm settings (**31 network inputs**, 4 outputs)
- `config/simulation-config.json`: Simulation parameters (arena size, food count, detection radii, energy economy, `batch_inference`, `render_backend`)

Performance-related JSON keys:

- `batch_inference` (default `true`): NumPy-compiled forward passes with parity to NEAT `activate`
- `render_backend`: `pygame` (default) or `moderngl` (GPU instanced draw; falls back to pygame)
- `render_stride`: draw every Nth sim step when rendering (default `10`)

Legacy JSON keys are still supported:

- `petridish_size` → `environment_width` / `environment_height`
- `episode_length` → `simulation_steps`

When rendering, use **click** to select an organism, **S** to toggle sense-radius rings, and **Esc** to clear selection. Movement trails fade behind each organism automatically.

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
config/          NEAT and simulation JSON settings
docs/            Review plans and bug sweep findings
src/             Simulation runtime (main, organism, simulation, renderer, …)
tests/           Unit tests (distance, network inputs, organism, breeding, …)
```

## License

MIT License
