# Bug Sweep Findings

Executed per `docs/BUG_SWEEP_PLAN.md` on branch `cursor/execute-refactor-review-ac70`.

## Confirmed bugs (fixed in this pass)

| ID | Sev | Module | Issue | Fix |
|---|---|---|---|---|
| SW-1 | P1 | scoreboard | `last_seen` not updated when fitness did not beat record | Always bump `last_seen`; update attrs only on new high |
| SW-2 | P1 | scoreboard | `dashboard_level` read via stack introspection | `Scoreboard.set_dashboard_level` + explicit arg from Simulation |
| SW-3 | P1 | renderer | `scoreboard_height` undefined in VIDEORESIZE handler | Initialize `scoreboard_height`; remove broken handler |
| SW-4 | P1 | renderer | `get_species_visual` passed `spike_length` as `num_nodes` | Correct parameter + optional `num_nodes` |
| SW-5 | P1 | organism | `reset()` did not clear `highest_fitness` between trials | Reset peak fitness on episode reset |
| SW-6 | P1 | simulation | Legacy keys `episode_length`, `petridish_size` ignored | Map to `simulation_steps` / environment size |
| SW-7 | P2 | simulation | O(n²) proximity each step | `SpatialGrid` rebuilt once per step |
| SW-8 | P2 | renderer | Redundant `pygame.init`, aggressive `gc.collect`, double event drain | Guard init; lighter cache clears |
| SW-9 | P2 | renderer | Bare `except` on font load | Catch `FileNotFoundError` / `OSError`; use project-relative font path |

## Verified clean (no change needed)

- Distance helpers: linear radius semantics (covered by tests)
- Network inputs: 22 channels incl. herbivore threats
- Headless eval: pygame events gated on `display.get_init()`
- Breeding: episode-local children only
- Config economy: starting energy, food value, movement cost wired

## P3 hygiene (completed)

| ID | Item | Resolution |
|---|---|---|
| SW-10 | `scoreboard.py` / `renderer.py` >500 LOC | Already under limit (269 / 323 LOC); no split required |
| SW-11 | `test_genetics.py` placeholder | Replaced with `tests/test_breeding.py` (episode-local NEAT crossover) |
| SW-12 | README outdated (18 inputs) | README refreshed: 22-input schema, CLI flags, testing, layout |
| SW-13 | Legacy config alias coverage | Added `test_legacy_config_aliases_are_honored` in `test_simulation.py` |

## Test coverage after sweep

```bash
python3 -m unittest discover -s tests -v
```

25 tests: distance, network inputs, organism, breeding, simulation, spatial, scoreboard.
