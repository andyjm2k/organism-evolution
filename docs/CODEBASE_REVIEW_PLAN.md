# Codebase Review Plan: Organism Evolution (NEAT)

This plan defines a complete, staged review of the repository to find refactoring opportunities where feature implementations are **not fit for purpose**, **not optimized**, or have **implementation gaps**. It is scoped to the current Python/NEAT simulation codebase (~2.8k LOC across `src/`).

## 1. Review objectives

For each feature function, answer:

1. **Fit for purpose** — Does the implementation match the intended behavior described in the README / config / NEAT wiring?
2. **Correctness gaps** — Are there bugs, inconsistent units, dead paths, or conflicting rules that break the feature?
3. **Optimization** — Are there algorithmic, memory, or I/O costs that dominate runtime for the configured population and episode lengths?
4. **Testability / maintainability** — Can the behavior be verified and changed safely?

Deliverable of the review (when executed): a prioritized finding list with severity, owning module, root cause, and proposed refactor—not drive-by rewrites.

## 2. Inventory of feature surfaces

| Feature area | Primary modules | Key entry points |
|---|---|---|
| CLI / bootstrap | `main.py` | `run_simulation` |
| Evolutionary loop | `simulation.py` | `run`, `eval_genomes`, `evaluate_generation` |
| Runtime breeding into NEAT | `simulation.py`, `organism.py` | `add_organism`, breeding block in `take_action` |
| Sensing / NN control | `organism.py`, `config/neat-config.ini` | `take_action` (18 in / 8 out) |
| Diet / consumption | `organism.py` | `check_for_food`, `hunt_prey`, inline collision in `take_action` |
| Fitness / selection signal | `organism.py` | `calculate_fitness`, trial median in `eval_genomes` |
| Spatial / environment | `simulation.py`, `organism.py`, `food.py` | `spawn_food`, `distance_to*`, boundary logic |
| Visualization | `renderer.py`, `simulation.py` | `Renderer.render`, `visualize_neural_network` |
| Metrics / dashboard | `scoreboard.py` | `record_species`, dashboards, final summary |
| Configuration | `config/*.json`, `config/neat-config*.ini` | loaded in `main` / `Simulation.__init__` |
| Genetics module | `src/genetics` (stub), NEAT library | crossover/mutate via neat-python |
| Tests | `tests/*` | currently stubs only |

Empty / placeholder artifacts to include in review: `src/utils.py`, `src/genetics`, duplicate `neat-config copy*.ini`, missing `assets/fonts/`.

## 3. Review phases

### Phase A — Baseline & contract mapping (read-only)

**Goal:** Establish intended contracts before judging implementations.

1. Map README claims → runnable paths (`render=true/false`, logging, dashboard).
2. Map `simulation-config.json` keys → actual reads in code (flag unused / overridden constants).
3. Map NEAT `num_inputs=18` / `num_outputs=8` → exact input/output vectors built in `Organism.take_action`.
4. Document the intended lifecycle: generation → N trials → steps → fitness → speciation → scoreboard.
5. Note working-directory assumptions (`config/...` relative paths from CWD).

**Exit criteria:** A one-page “behavior contract” per feature area (inputs, outputs, side effects).

### Phase B — Correctness & fitness-for-purpose audit

**Goal:** Find logic that cannot achieve the feature’s purpose.

Review each method using the checklist in §4. Prioritize these already-suspected hotspots (confirm/deny with evidence during execution):

#### B1. Distance / collision contract (`Organism.distance_to`)

- `distance_to` returns **squared** distance; callers often compare to linear radii or wrap with an extra `math.sqrt`.
- Proximity filters in `eval_genomes` compare squared distance to linear `*_detection_radius` values.
- Breeding distance check mixes squared distance with linear radius expressions.
- Inline food eat in `take_action` uses `distance_to(food) <= 0` (effectively “exact overlap only”).
- `is_near` / `is_colliding_with` mix squared vs linear and reference undefined `_radius`.

**Fit question:** Can organisms reliably detect and consume food/prey at configured radii?

#### B2. Dual update paths (behavior duplication)

Each simulation step calls both `take_action(...)` and `update(...)`.

- Counters (`steps_taken`, breeding/food/hunt timers) increment in **both**.
- Food consumption exists in `take_action` **and** `check_for_food`.
- Predation exists in `take_action` **and** `hunt_prey` with **different rules** (radius-vs-size vs size threshold `0.8`).

**Fit question:** Is organism behavior deterministic and role-consistent, or double-applied / conflicting?

#### B3. Neural input purpose (herbivore threat sensing)

Herbivores compute threat distance/direction but **do not append threat channels** to the NN input vector; they append food channels instead. Carnivores get prey channels. Both reach 18 inputs to match NEAT config.

**Fit question:** Can herbivores evolve escape behavior if threats are never network inputs?

#### B4. Breeding vs NEAT population integrity

- Child genomes constructed with fixed ID `0`.
- `add_organism` mutates `population.population` / species maps mid-evaluation.
- `run()` contains large KeyError recovery that rebuilds speciation state.
- Energy gates disagree: readiness flag uses `>= 50`; `can_breed` requires `>= 100`.
- Children appended to `nearby_organisms` but not always to the evaluation `organisms` list used for death/fitness.

**Fit question:** Is sexual reproduction a supported evolutionary operator, or an unstable side path?

#### B5. Fitness signal & scoreboard linkage

- Trial fitness uses median of `calculate_fitness`; death forces 0 mid-trial.
- `fitness_bonus` is updated but not clearly folded into `calculate_fitness`.
- After `eval_genomes`, organisms are `cleanup()`’d and genome `.organism` attrs removed; `run()` then tries to collect `genome.organism` for `evaluate_generation` → often empty survivors path.
- Scoreboard may be updated twice (end of `eval_genomes` and `evaluate_generation`) with different survivor sets.

**Fit question:** Does selection and reporting reflect actual episode outcomes?

#### B6. Config fidelity

Config defines `starting_energy`, `food_energy_value`, `movement_cost`, `petridish_size`, `episode_length` etc., while organism logic hardcodes energy bases, food gain `75 * efficiency`, movement costs, and step counts from other keys.

**Fit question:** Are “configurable simulation parameters” actually controlling the simulation?

#### B7. Rendering / display contract

- `main.py` creates a pygame display; `Renderer` calls `set_mode` again with a different size (adds scoreboard strip).
- Fonts reference missing `assets/fonts/*`.
- `VIDEORESIZE` references unset `scoreboard_height`.
- `visualize_neural_network` output-node indexing looks incorrect; depends on unused `graphviz` import / undeclared matplotlib dependency.
- Aggressive `cleanup_resources` clears species colors every few generations (visual flicker / wasted work).

**Fit question:** Does “visual rendering + scoreboard” work as a single coherent UI?

#### B8. Dead / incomplete feature modules

- `utils.py` empty; `genetics` is a comment stub; tests are `pass`-only.
- Several organism helpers (`get_closest_*`, `get_prey_info`, `get_threat_info`, `get_target_inputs`) appear unused by `take_action`.

**Fit question:** Which APIs are real product surface vs leftover experiments?

### Phase C — Performance & resource audit

**Goal:** Identify optimization opportunities without changing intended behavior.

Workload baseline from config: `pop_size≈50`, `simulation_steps=2000`, `num_trials=3`, `num_generations` up to 2000, env 1000×1000, food 75.

Measure / estimate:

1. **Spatial queries** — O(organisms × food) and O(organisms²) per step inside trials; consider spatial hashing / grids.
2. **NN activate frequency** — every organism every step; profile vs collision/logging cost.
3. **Logging I/O** — unconditional `[DEBUG]` prints in hot paths (`calculate_attributes`, trial loops) regardless of `logging_level`.
4. **Memory** — genome/organism refs, surface caches, `gc.collect()` after every generation, pygame event double-polling.
5. **Headless path** — pygame still initialized; confirm whether display-less mode still pays render-adjacent costs.
6. **Dependency bloat** — `graphviz` imported unused; requirements incomplete vs imports.

**Exit criteria:** Hotspot table with estimated complexity and suggested refactor class (algorithmic vs structural vs I/O).

### Phase D — Architecture & maintainability audit

**Goal:** Find structural refactors that unlock safe feature work.

1. **File size / cohesion** — `organism.py` (~1150 LOC) and `simulation.py` (~750 LOC) exceed the project’s 500-line file guideline; split sensing, movement, breeding, fitness, predation.
2. **Layering** — UI (`renderer`, scoreboard stack introspection) leaking into domain; scoreboard reads caller frames for `dashboard_level`.
3. **Import style** — repeated inline `from organism import Organism` / `from scoreboard import Scoreboard` inside methods; circular-risk patterns via `organism.simulation`.
4. **Error handling** — broad try/except that hides breeding failures; speciation “repair” that may mask root bugs.
5. **Path / package layout** — scripts assume CWD; no package `__init__`; tests import `src.*` while `main` imports bare modules → inconsistent PYTHONPATH.
6. **Config duplicates** — multiple `neat-config copy*.ini` without documented purpose.

### Phase E — Test gap analysis & harness design

**Goal:** Define the minimum tests that would lock correct feature behavior before refactors.

Current state: all three test modules are empty stubs.

Proposed test matrix (unit first, then narrow integration):

| Area | Must-cover behaviors | Suggested tests |
|---|---|---|
| Distance API | squared vs linear, None positions | `test_distance_to_*` |
| Collision | food/prey radius rules single path | `test_check_for_food`, `test_hunt_prey` |
| NN inputs | fixed length 18 for both diets; threat channels present or explicitly absent by design | `test_build_network_inputs` |
| Breeding gates | energy, cooldown, boundary, partner rules | `test_can_breed_*` |
| Fitness | role penalties, bonuses, non-negative death handling | `test_calculate_fitness_*` |
| Simulation step | one step does not double-count timers/consumption | `test_eval_step_invariants` |
| Config | JSON keys applied to organism economy | `test_config_overrides_energy_costs` |
| Scoreboard | record/update/top-n idempotence | `test_record_species` |
| Add organism | IDs unique; species maps consistent | `test_add_organism_ids` (may be redesign) |

Target: each reviewed class covers ≥70% of its methods with meaningful assertions (not just construction).

### Phase F — Synthesis & refactor roadmap

**Goal:** Produce an ordered backlog.

Severity rubric:

- **P0 — Broken purpose:** Feature cannot work as designed (wrong units, unreachable consumption, selection ignores survivors).
- **P1 — Conflicting rules:** Duplicate/competing implementations change outcomes unpredictably.
- **P2 — Optimization:** Correct but scales poorly for configured workload.
- **P3 — Cleanup:** Dead code, stubs, docs/config drift, packaging.

Recommended execution order after review findings are confirmed:

1. Unify distance/collision API and single consumption path (P0/P1).
2. Align NN input schema with intended sensing (esp. herbivore threats) and NEAT config (P0).
3. Decide breeding strategy: defer offspring to next generation **or** properly integrate into NEAT without ID collisions (P0/P1).
4. Wire fitness/scoreboard to surviving evaluation results; remove post-cleanup organism collection (P0/P1).
5. Honor simulation config for energy economy; remove hardcoded duplicates (P1).
6. Spatial index + logging gates for performance (P2).
7. Split oversized modules; delete dead helpers; real tests; package layout (P2/P3).
8. Renderer display ownership / asset paths / dependency hygiene (P2/P3).

## 4. Per-function review checklist

Use this checklist for every public method in `Organism`, `Simulation`, `Renderer`, `Scoreboard`, `Food`, and `run_simulation`:

1. **Intent** — One sentence: what should this do for the user/sim?
2. **Callers** — Who calls it? Dead if none.
3. **Pre/post conditions** — Units, None handling, energy/death invariants.
4. **Config coupling** — Uses config key or hardcodes?
5. **Duplication** — Another method with same responsibility?
6. **Complexity** — Hot-path cost vs frequency.
7. **Side effects** — Mutates global NEAT/population/scoreboard/pygame?
8. **Test** — Covered? If not, what oracle proves correctness?
9. **Verdict** — Keep / fix / merge / delete / redesign.
10. **Evidence** — File + line references for the finding.

## 5. Suggested review order (module walk)

1. `food.py` + distance helpers (foundation).
2. `organism.py`: attributes → sensing inputs → movement → consume → breed → fitness → reset/cleanup.
3. `simulation.py`: spawn → `eval_genomes` inner loop → fitness aggregation → `add_organism` → `run` / speciation recovery → `evaluate_generation`.
4. `main.py` bootstrap vs `Renderer` display ownership.
5. `renderer.py` + click NN viz.
6. `scoreboard.py` recording and dashboards.
7. Config files + requirements + README drift.
8. Tests/package layout last (define harness against fixed contracts).

## 6. Working method & evidence standards

- Prefer small reproduction scripts or unit tests over long full generations when confirming bugs.
- For each finding: **symptom**, **root cause**, **impact on evolution**, **proposed refactor**, **risk**.
- Do not mix unrelated cleanups into correctness fixes.
- Keep refactors behind tests once Phase E harness exists for that area.
- Respect file size limit (≤500 LOC) when splitting during implementation phases.

## 7. Out of scope for the review itself

- Training a “good” evolved agent or tuning NEAT hyperparameters for max fitness.
- Visual redesign of the pygame UI (unless broken / blocking).
- Replacing neat-python wholesale.

## 8. Expected artifacts when this plan is executed

1. Findings log (markdown or issues) tagged P0–P3 with code references.
2. Contract doc for NN I/O, distance units, and consumption rules.
3. Refactor PR sequence matching §3 Phase F order.
4. Test suite replacing stubs for the touched classes.
5. Config/README alignment after behavior is stabilized.

## 9. Quick-start commands for reviewers

```bash
# Install deps (note: matplotlib used but may be missing from requirements)
pip install -r requirements.txt

# Smoke headless boot (expect path/CWD sensitivity)
cd /path/to/repo && python src/main.py render=false logging=normal dashboard=minimal

# Run existing tests (currently placeholders)
python -m unittest discover -s tests -v
```

Profile candidates after a correctness baseline:

- Time one `eval_genomes` call with `num_trials=1`, reduced `simulation_steps`.
- Count distance comparisons per step (organisms × organisms).
- Track RSS across generations via existing scoreboard memory helpers.

---

*Plan generated against repository state on `main` (modules under `src/`, config under `config/`). Execute phases A→F sequentially; do not start broad refactors until Phase B findings are confirmed with tests or minimal reproducers.*
