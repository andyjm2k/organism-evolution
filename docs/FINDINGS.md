# Codebase Review Findings

Executed against `main` per `docs/CODEBASE_REVIEW_PLAN.md`. Severity: P0–P3.

## Phase A — Contracts

| Area | Intended contract | Observed |
|---|---|---|
| NEAT I/O | 18 inputs / 8 outputs (`neat-config.ini`) | Herbivore threat channels computed then **dropped**; diet channels swapped by role to keep length 18 |
| Config economy | JSON drives energy/food/movement | Keys like `starting_energy`, `food_energy_value`, `movement_cost` largely **ignored**; hardcoded in `organism.py` |
| Distance | Detection radii in world units | `distance_to` returns **squared** distance; compared to linear radii |
| Lifecycle | eval → fitness → scoreboard → next gen | Organisms `cleanup()`’d then `run()` looks for `genome.organism` survivors (usually empty) |
| Breeding | Same-species offspring | Child genome ID hard-coded `0`; mid-eval NEAT mutation + speciation KeyError recovery |

## Phase B — Confirmed defects

### P0-1 Squared vs linear distance (detection broken)
- **Where:** `Organism.distance_to` returns `dx²+dy²`; `Simulation.eval_genomes` compares to `food_detection_radius` (e.g. 300).
- **Evidence:** Food 50 units away has squared distance 2500 ≮ 300 → not detected. Effective radius ≈ √300 ≈ 17.3.
- **Also:** `math.sqrt(distance_to(...))` double-applies root incorrectly in intent; breeding compares squared distance to linear radius expression; `take_action` food uses `distance_to <= 0` (exact overlap only).

### P0-2 Dual consumption / counter paths
- **Where:** Each step calls `take_action` then `update`.
- **Impact:** Timers/`steps_taken` double-increment; food eaten in both `take_action` and `check_for_food`; predation rules differ (`take_action` size-by-radius vs `hunt_prey` size×0.8).

### P0-3 Herbivore threats never reach the network
- Threat vectors computed for herbivores but **not appended** to `inputs`; only food + breeding follow. Escape behavior cannot evolve via NN.

### P0-4 Breeding corrupts NEAT population
- `DefaultGenome(0)` + `add_organism` mutates population/species mid-evaluation.
- `run()` KeyError recovery rebuilds speciation — masks root cause.

### P0-5 Fitness / scoreboard linkage broken after eval
- `eval_genomes` clears organism refs then `run()` collects `genome.organism` for `evaluate_generation` → empty / warning path.
- `fitness_bonus` updated on breed/eat but **not included** in `calculate_fitness`.

### P1-1 Config not applied
- Hardcoded energy bases, food gain `75 * efficiency`, movement costs ignore JSON.

### P1-2 Breeding energy gate mismatch
- Desire path uses energy ≥ 50; `can_breed` requires ≥ 100.

### P1-3 Dead / inconsistent helpers
- `is_colliding_with` references undefined `_radius`; unused `get_closest_*` / prey/threat helpers; empty `utils.py` / stub `genetics`.

### P2 Performance
- O(n²) organism proximity every step; unconditional `[DEBUG]` prints in hot paths; `gc.collect()` every generation; pygame double `set_mode` (main + Renderer).

### P2/P3 Packaging / tests / deps
- Tests are `pass` stubs; `matplotlib` imported but not in requirements; `graphviz` imported unused; duplicate `neat-config copy*.ini`; missing `assets/fonts`; modules exceed 500 LOC.

## Phase C — Performance notes

Workload: pop≈50 × 3 trials × 2000 steps × O(n²) proximity dominates. Logging I/O amplifies cost. Spatial hashing deferred after correctness; logging gates included in this execution.

## Phase F — Execution order applied in this change set

1. Unified distance/collision API + single consumption path  
2. NN schema with food + threat/prey channels; NEAT `num_inputs` updated  
3. Breeding: episode-local children + parent fitness bonus; no mid-gen NEAT mutation  
4. Fitness includes bonus; scoreboard recorded before cleanup  
5. Config-driven energy economy  
6. Logging gated; modules split under 500 LOC; dead code removed  
7. Real unit tests; requirements/README hygiene  


## Execution status

Phases A–F executed on branch `cursor/execute-refactor-review-ac70`:

- **P0 distance/collision**: unified `distance.py` helpers; sensing uses linear radii via `within_radius`.
- **P0 dual update path**: consumption only in `update` (`check_for_food` / `hunt_prey`); counters increment once.
- **P0 herbivore threats**: `network_inputs.py` builds 22 channels including threat block; `neat-config.ini` `num_inputs=22`.
- **P0 breeding**: episode-local children via `register_episode_child` (no mid-gen NEAT mutation / genome id `0` hack).
- **P0 fitness/scoreboard**: `fitness_bonus` included; generation stats snapshotted before `cleanup`.
- **P1 config economy**: `starting_energy`, `food_energy_value`, `movement_cost` wired through `environment_config`.
- **P2 logging**: hot-path logs gated by `logging_util`.
- **P2/P3**: modules split (`distance`, `network_inputs`, `fitness`, `logging_util`); duplicate neat configs removed; real unit tests added (17 passing).
