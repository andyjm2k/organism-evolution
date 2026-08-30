# Living World Harness

The **living world** is a continuous-evolution simulation mode. Unlike the default episodic NEAT harness, it never resets the arena between trials or generations. Organisms live until they starve or are hunted, food depletes and regrows from **nutrient clouds**, and in-world breeding adds offspring directly to the live population.

Steady-state NEAT (**Approach B**) periodically culls low-fitness organisms and injects crossover offspring from top performers.

## Quick start

```bash
# Visual run (recommended)
python src/main.py harness=living_world

# Headless run with step limit
python src/main.py harness=living_world render=false max_world_steps=10000
```

Configuration defaults live in `config/living-world-config.json`.

## Architecture

| Module | Role |
|--------|------|
| `src/living_world.py` | Main harness loop — persistent stepping, selection, scoreboard |
| `src/food_ecology.py` | Nutrient clouds + food regrowth under scarcity |
| `src/population_registry.py` | Live organism/genome registry with carrying capacity |
| `src/rolling_fitness.py` | Lifetime selection fitness (survival, role, offspring) |
| `src/selection_scheduler.py` | Periodic cull + immigration (steady-state NEAT) |
| `src/world_clock.py` | Monotonic step counter and epoch grouping |
| `src/camera.py` | Viewport pan and world/screen coordinate transforms |
| `src/living_world_renderer.py` | Large-world pygame UI (minimap, tracking, genome panel) |
| `src/genome_viz.py` | NEAT genome graph for the inspection panel |

## Episodic vs living world

| | Episodic (`simulation-config.json`) | Living world (`living-world-config.json`) |
|---|-------------------------------------|-------------------------------------------|
| Time model | Fixed steps × trials per generation | Continuous until `max_world_steps` or quit |
| Food | Full respawn each trial | Nutrient clouds regrow pellets |
| Organisms | Reset each trial; recreated each generation | Persist until death |
| Breeding offspring | Episode-local only | Join live population |
| Selection | NEAT generational reproduction | Steady-state cull + immigration |
| Default world | 1000×1000 | 4000×4000 |
| Default viewport | Full window | 900×900 draggable region |

## Nutrient clouds

Nutrient clouds are large, slowly drifting regions that spawn food pellets. When overall food density drops below `food_scarcity_threshold`, regrowth accelerates by `food_regrowth_scarcity_multiplier`. Clouds are rendered as semi-transparent green fields; food appears as bright dots, often clustered inside clouds.

Key config keys:

- `nutrient_cloud_count` — number of clouds placed at startup
- `nutrient_cloud_min_radius` / `nutrient_cloud_max_radius` — cloud size range
- `nutrient_cloud_spawn_rate` — base per-tick spawn probability per cloud
- `food_target_density` — equilibrium food level the ecology targets

## Steady-state selection

Every `selection_interval_steps`:

1. Rolling fitness is synced to NEAT genomes
2. Bottom `cull_fraction` of living organisms are removed
3. If below `max_population`, immigrants may spawn (crossover of top performers, or fresh random genomes)

When the population goes extinct, the scheduler automatically reseeds immigrants.

## Rendering and controls

The living-world renderer uses a **viewport** into a larger world:

| Input | Action |
|-------|--------|
| **Mouse drag** | Pan the camera |
| **Click organism** | Select and auto-track |
| **Click minimap** | Jump viewport to that world location |
| `T` | Toggle auto-track on selected organism |
| `S` | Toggle sense-radius rings |
| `Esc` | Clear selection and stop tracking |

UI elements:

- **HUD** (top-left) — world step, food count, scarcity ratio, species count, track status
- **Minimap** (top-right) — organisms, food, clouds, gold viewport rectangle
- **Inspection panel** (bottom-left when selected) — energy, age, diet stats, NEAT genome graph
- **Scoreboard** (right panel) — top species (shared with episodic mode)

## Configuration reference

See `config/living-world-config.json` for defaults. Important keys:

| Key | Default | Description |
|-----|---------|-------------|
| `harness_mode` | `living_world` | Set automatically when using `harness=living_world` CLI |
| `environment_width/height` | 4000 | Full world size |
| `viewport_width/height` | 900 | Visible window region |
| `max_population` | 120 | Carrying capacity |
| `selection_interval_steps` | 4000 | Steady-state selection cadence |
| `cull_fraction` | 0.1 | Fraction culled each selection tick |
| `immigration_rate` | 0.03 | Probability of injecting immigrants when under cap |
| `max_world_steps` | `null` | Optional stop condition (`null` = run until quit) |
| `minimap_size` | 160 | Minimap pixel width/height |
| `camera_track_smoothing` | 0.15 | Smooth follow factor for tracked organisms |

### Genesis bootstrap

At startup the living world replaces vanilla NEAT genomes with **enhanced archetypes** so early organisms survive long enough to forage and breed:

| Key | Default | Description |
|-----|---------|-------------|
| `genesis_archetype_count` | 8 | Number of seed archetypes (forms distinct species clusters) |
| `genesis_extra_hidden_nodes` | 6 | Hidden nodes added to each archetype beyond `num_hidden` |
| `genesis_extra_connections` | 20 | Additional feed-forward connections per archetype |
| `genesis_weight_jitter` | 0.05 | Gaussian noise on cloned weights for within-species diversity |
| `genesis_foraging_bias` | 0.35 | Strength of food/prey→movement weight nudging |

CLI overrides use the same `key=value` syntax as the episodic harness, e.g. `render=false`, `max_world_steps=5000`.

## Testing

Living-world modules have dedicated unit tests:

```bash
python3 -m unittest discover -s tests -p 'test_living_world*.py' -v
python3 -m unittest discover -s tests -p 'test_population_registry.py' -v
python3 -m unittest discover -s tests -p 'test_selection_scheduler.py' -v
python3 -m unittest discover -s tests -p 'test_camera.py' -v
python3 -m unittest discover -s tests -p 'test_world_clock.py' -v
python3 -m unittest discover -s tests -p 'test_genome_viz.py' -v
python3 -m unittest discover -s tests -p 'test_main_harness.py' -v
```

Or run the full suite:

```bash
python3 -m unittest discover -s tests -v
```
