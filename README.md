# NEAT Simulation

This project simulates the genetic evolution of simple cell organisms using the NEAT algorithm. Organisms compete for survival in a petri dish by consuming food and each other, developing traits to gain advantages.

## Features

- Configurable simulation parameters
- Visual rendering of the simulation
- Genetic evolution using NEAT
- Terminal dashboard for headless mode
- 22-channel neural network inputs (food, prey/threat, breeding)

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

- `config/neat-config.ini`: NEAT algorithm settings (**22 network inputs**, 8 outputs)
- `config/simulation-config.json`: Simulation parameters (arena size, food count, detection radii, energy economy, `batch_inference`, `render_backend`)

Performance-related JSON keys:

- `batch_inference` (default `true`): NumPy-compiled forward passes with parity to NEAT `activate`
- `render_backend`: `pygame` (default) or `moderngl` (GPU instanced draw; falls back to pygame)
- `render_stride`: draw every Nth sim step when rendering (default `10`)

Legacy JSON keys are still supported:

- `petridish_size` → `environment_width` / `environment_height`
- `episode_length` → `simulation_steps`

### Neural network input schema (22 channels)

| Channels | Content |
|----------|---------|
| 0–8 | Core body/environment (energy, position, movement, size, boundaries) |
| 9–12 | Food sensing (herbivores; zeros for carnivores) |
| 13–16 | Prey (carnivores) or threats (herbivores) |
| 17–21 | Breeding partners and readiness |

## Testing

Run the full unit test suite from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

## Project layout

```
config/          NEAT and simulation JSON settings
docs/            Review plans and bug sweep findings
src/             Simulation runtime (main, organism, simulation, renderer, …)
tests/           Unit tests (distance, network inputs, organism, breeding, …)
```

## License

MIT License
