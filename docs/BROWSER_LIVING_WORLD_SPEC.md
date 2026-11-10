# Living world in the browser

Technical specification for a second runtime of the living-world harness. The Python desktop program in `src/` stays the reference. The new runtime is a static web app a visitor runs entirely in their browser, and that a cloud agent can publish with Cursor Apps hosting (the app site Grok Bot and Cursor open from the Apps dashboard).

Source of behavior: `src/living_world.py` and the modules it calls, `config/living-world-config.json`, and `config/neat-config.ini`. Desktop controls and ecology are described in `docs/LIVING_WORLD.md`.

## 1. Decision

Ship a Vite + TypeScript project in `web/`. The simulation kernel is a line-faithful port of the living-world Python modules. Drawing is WebGL2, with a Canvas 2D path that paints the same bodies when WebGL2 is unavailable. The page is one static client. It has no Python process, no long-lived server, and no database.

Publish `web/` as the app folder (`package.json` plus `index.html`). The repository root is a Python package and is the wrong folder to publish: the host may serve any file that is not a server route, and the root has no web entry point.

## 2. What the app is

One visitor, one world, running on that visitor's machine.

| Default | Value |
| --- | --- |
| World | 4000×4000 |
| Viewport | 900×900, panned inside the world |
| Carrying capacity | 120 organisms |
| Food target / cap | 200 / 320 (1.6× target) |
| Nutrient clouds | 14, each with up to 3 plant strains |
| Network | 31 inputs, 4 outputs, feed-forward |
| Selection | Every 4000 steps, only when at capacity |
| Frame pace | 60 rendered frames/s, 2 sim steps per frame (`render_stride`) |

The episodic harness (`src/simulation.py`, `config/simulation-config.json`) is out of this app. Headless training stays on the Python CLI.

## 3. Host constraints

These come from Cursor Apps hosting (`https://cursor.com/docs/cloud-agent/apps`) and the publish path a cloud agent uses. They force the shape of the port.

| Host rule | Consequence for this app |
| --- | --- |
| A publishable folder has `index.html`, or a `package.json` whose build produces one | `web/index.html` and `web/package.json` with `build` writing a static site |
| The host builds from committed source. Gitignored output is not the upload | Commit `web/src` and `web/public`. `dist/` may be gitignored |
| Functions are request handlers (`export async function GET(request)`), not a process that stays up | The tick loop lives in the browser. There is no server tick route |
| A route a visitor can call to burn unbounded compute is rejected | Do not add `/step`, `/run`, or a WebSocket that advances the world on the host |
| Private and team visibility already sign the visitor in | Do not add an auth library, login page, or users table |
| Public apps have no viewer identity | The world must run with no account and no token |
| Secrets in files, `.env`, and source maps are blocked or stripped | Simulation JSON and the NEAT ini are public data. No API keys. No `VITE_` secret |
| A Postgres database is created only if `cursor-app.json` asks for one | Do not add `cursor-app.json`. Worlds are not shared rows |
| The host does not keep a process between visits | Resume is a snapshot in the browser (IndexedDB), written only when the visitor saves |
| Grok Bot on a phone opens the same URL | The layout has to work in a narrow viewport. Keyboard shortcuts need on-screen equivalents |

The app reads no environment variables. `DATABASE_URL`, `CURSOR_APP_ID`, and identity helpers stay unused.

Grok Bot Pages (`pages.grok.bot`) are a different host. A page is a document the bot rewrites, and controls on it are not stored. This simulation is an app: the visitor runs it, and only the visitor's Save button stores it. Publish it as an App, not as a Page.

Background tabs pause the clock (`document.visibilityState`). A published page must not pin a CPU core while the visitor is elsewhere. The pygame window keeps stepping when it loses focus; the published page does not.

## 4. Architecture

```
web/
  index.html
  package.json
  vite.config.ts
  public/config/living-world-config.json   # copied at build from ../config
  public/config/neat-config.ini
  src/main.ts                # page boot, settings, save/load
  src/ui/                    # HUD, scoreboard, inspect panel, log, touch controls
  src/render/                # camera, WebGL2 world, Canvas2D fallback, genome graph
  src/worker/sim-worker.ts   # living-world loop off the UI thread
  src/sim/                   # ported kernel (one module per Python module)
  src/neat/                  # the NEAT subset this harness actually calls
  src/rng/python-random.ts   # CPython random.Random (MT19937)
  test/                      # vitest, including golden traces from Python
```

The worker owns organisms, food, eggs, genomes, and the clock. The page owns the pointer, the camera, and the canvas. Each animation frame the page asks the worker for `render_stride` steps (default 2) and one snapshot, then draws. If that work overruns the frame, the page does not schedule a second stride to catch up.

Snapshot fields are plain numbers and arrays: positions, radii, colors, organ strengths, trail points, egg positions, cloud circles, HUD counts, scoreboard rows, and the selected genome graph. The page never holds a live genome object.

`postMessage` is structured clone. No `SharedArrayBuffer`, no `SharedArrayBuffer` headers, and no cross-origin isolation. The Apps host does not promise those headers, and the population (120) does not need them.

### Modules to port

Port these files into `web/src/sim` and `web/src/neat`. Behavior follows the Python, including first-match scans and insertion order.

| Python | Browser module | Notes |
| --- | --- | --- |
| `distance.py`, `spatial.py` | `distance.ts`, `spatial.ts` | Cell walk order matches the Python nested loops |
| `food.py`, `food_ecology.py`, `plant_strain.py` | same names | Cloud drift, scarcity, strain bands, seed cap |
| `organism.py`, `network_inputs.py` | same | 31-input contract, movement, breeding gates |
| `egg.py`, `growth.py`, `combat.py` | same | Gestation, maturity, bite damage |
| `body_genes.py`, `body_traits.py` | same | Diet threshold 0.5, organ strengths |
| `fitness.py`, `rolling_fitness.py` | same | Display fitness vs selection fitness |
| `population_registry.py`, `selection_scheduler.py` | same | Capacity, cull, immigration, species ids |
| `genome_bootstrap.py`, `world_clock.py`, `species_names.py` | same | Founding wires and epoch clock |
| `camera.py` | `render/camera.ts` | Same transforms. Smoothing 0.15 |
| `genome_viz.py` | `render/genome-graph.ts` | SVG, same columns and weight colors |
| `living_world.py` loop | `worker/sim-worker.ts` | Step order in section 6 |

Leave unported: `renderer.py`, `renderer_gl.py`, `living_world_renderer.py`, `living_world_gl.py`, `renderer_common.py`, `renderer_factory.py`, `simulation.py`, `main.py`, `logging_util.py`, `scoreboard.py` (the terminal and pygame surfaces). Their visible results are reimplemented in `web/src/ui` and `web/src/render`.

`batch_inference.py` and `compiled_network.py` collapse into one compiled feed-forward pass. It is the only runtime activation path. A scalar reference pass exists for tests and must match it.

### Iteration order

JavaScript object keys that look like integers are enumerated in numeric order. NEAT node ids are integers, including negative sensor keys. Genomes, connections, species members, and the population map are `Map`s, so insertion order survives. Food, organisms, and eggs stay arrays. Grazing and hunting still hit the first overlapping item in that array and then stop.

## 5. Behavior that stays

The published app keeps the living-world rules in `docs/LIVING_WORLD.md`.

- Persistent arena. Death, starvation, and bites remove bodies. The world does not reset on an epoch.
- Nutrient clouds drift, wrap, and spawn pellets. Scarcity below 0.35 multiplies spawn by 2.5. The food cap is 1.6× `food_target_density`. Eaten pellets leave the list every tick.
- Plant strains: symbiotic at relation ≥ 0.35, predatory at ≤ −0.35, otherwise a normal meal. A strain seeds after 8 success points, and only while under the food cap. Clouds keep at most three strains.
- Bodies: color, diet, and size are genes copied from the fitter parent and nudged. Diet ≥ 0.5 hunts. Founders are green grazers. Network size still scales speed, radius, and metabolism on top of the size gene.
- Organs, drawn from enabled wires: food wires form a jaw, prey wires form a claw on a hunter or a horn on a grazer, core wires into velocity outputs form flagella. Strength below 0.25 stays invisible.
- Sensing and motors: the 31-channel vector and the 4 outputs in the README (vx, vy, breed > 0.5, rest > 0.5). Food direction is a unit vector. Other directions are scaled by their radius. Herbivores get zeros on the food channels when they are carnivores, and the reverse for prey versus threat.
- Movement noise `uniform(-0.01, 0.01) * speed` on each axis when not resting. Edges wrap and charge `boundary_penalty`.
- Breeding: energy ≥ 100, maturity ≥ 0.99, a prior meal or kill, cooldown, and a 2% edge margin. Partners share a species id and meet inside `breeding_detection_radius`. Living-world births are eggs (`egg_energy_goal` 80, `egg_hatch_delay` 150, hatch maturity 0.35). A birth at capacity displaces the lowest selection fitness.
- Combat: a carnivore bites the first grazer it is touching when `attacker.size >= prey.size * 0.8`. Health is life. Energy is fuel. A kill pays 70% of the prey's energy.
- Selection every `selection_interval_steps`: sync rolling fitness, cull `cull_fraction` only at `max_population`, immigrate with probability `immigration_rate` when under the cap, re-cluster living genomes, restore the species id a lineage already had. Compatibility starts at 3.0 and moves only inside the band `min_species_count` 4, `target_species_count` 6, `max_species_count` 9, never below the ini value. Extinction reseeds five immigrants.
- An epoch (`simulation_steps`, default 2000) is a log bucket. It does not replace the population or recolor survivors.
- Genesis: `pop_size` 60 founders, `genesis_archetype_count` 8, the four founding wires (food and prey dx/dy → vx/vy at weight 1.8), breed bias 1, rest bias −1, weight jitter 0.05. Extra hidden nodes and connections stay at 0 unless the visitor changes those keys.
- Camera: drag to pan, click a body to track, click the minimap to jump, `T` toggles track, `S` toggles sense rings, `Esc` clears the selection. HUD shows step, food count, scarcity, species count, and track state. The inspect panel shows name, diet, energy, age, meals, hunts, fitness, genome id, size, speed, organs, and the genome graph. The scoreboard lists top species with the names it minted. Sense rings use food, threat, and breeding radii. Trails keep 24 points. Draw caps stay 800 bodies, 400 minimap pellets, and 300 minimap bodies.

Config keys in `living-world-config.json` keep their names and defaults. The build copies that file and `neat-config.ini` into `web/public/config` so the Python files remain the source.

## 6. Step order

One tick matches `LivingWorldSimulation._simulation_step` and the tail of `run`:

1. Rebuild the food grid and the organism grid (cell size `max(50, detection_radius // 2)`).
2. Sense every living body, then `take_action`, then `update`, in registry order.
3. Credit rolling fitness for each survivor. Finalize display fitness on death.
4. Tick eggs. Hatch before pending births are copied into the live list.
5. Append births that registered during this tick. Drop the dead. Forget their compiled networks.
6. Tick ecology (cloud drift, regrow, prune, scarcity spawn).
7. Advance the clock.
8. On a selection boundary, run the scheduler.
9. On an epoch boundary, record the scoreboard from the best `highest_fitness` survivor and append the epoch line to the on-page log.

Rendering happens after the stride of ticks, on the snapshot. It does not run inside the tick, and it does not change fitness.

## 7. What changes

| Desktop | Published app |
| --- | --- |
| pygame window, or ModernGL with a pygame fallback | One renderer. WebGL2 when the context is created; Canvas 2D otherwise. `render_backend` is ignored |
| `clock.tick(60)` inside the render call, so a fast machine holds 60 frames and `60 × render_stride` steps per second | `requestAnimationFrame` draws 60 frames and runs `render_stride` steps per frame. No catch-up spiral. A Fast control multiplies the stride (1, 2, 8, 32) |
| Process keeps running when the window is unfocused | A hidden tab pauses. A Pause control and the Space key pause while the tab is visible |
| CLI `key=value` and JSON edits | A settings drawer edits the same keys and restarts the world. A link may set `seed` and `max_world_steps` (`?seed=1&max_world_steps=8000`) |
| stdout, `neat.StdOutReporter`, and the psutil memory block | An on-page log with the same epoch and selection lines. No memory meter and no NEAT generation reporter |
| Closing the window ends the process | Leaving the page stops the worker. Save writes an IndexedDB snapshot. Load restores one. Nothing is stored on the host |
| Fixed 900×900 viewport plus a 300px scoreboard | The world stays 4000×4000. The viewport fills the space beside the scoreboard. Under 800px wide the scoreboard becomes a drawer and the viewport fills the screen |
| Mouse only. `T`, `S`, and `Esc` | Those keys remain. The page also has buttons for pan (drag and one-finger drag), track, sense rings, clear, pause, and fast. Minimap and bodies accept taps |
| Sense rings allocated as surfaces up to 2800px across (`food_detection_radius` 1400) | Rings are stroked in world space. Same radii and colors, no giant offscreen canvas |
| Roboto files if present, otherwise the pygame default font | System UI fonts. No font CDN |
| `random` module, process-global | One `random.Random` port per world, seeded. Species-color rolls use a separate generator, as `RendererCommon` already does, so painting a new species does not reshuffle the world |
| Scoreboard class globals and names minted at record time | Names and records live in the world snapshot |
| Python `neat` package, NumPy compiled nets | The subset in section 8. Weights stay float64 (`number`). Activation sums with a scalar loop, matching `CompiledNetwork.activate` |

Query order, collision order, crossover, and mutation are part of the ecology. They are not on this list.

## 8. NEAT subset

Depend on a pinned `neat-python` version in the Python environment (record the exact version in `web/src/neat/ORACLE.md`). Port only the operations the living world calls. Do not port `Population.run`, stagnation, or generational reproduction.

| Operation | Callers |
| --- | --- |
| Parse the ini into genome, species, and reproduction parameters | Startup |
| `configure_new`, `configure_crossover`, `mutate`, `mutate_add_node`, `mutate_add_connection` | Genesis, eggs, immigrants |
| Innovation numbers, including `get_innovation_number(src, dst, "founding_motif")` | Founding wires |
| `distance` with `compatibility_disjoint_coefficient` 2.0 and `compatibility_weight_coefficient` 0.5 | Species assignment and threshold search |
| Feed-forward `activate` for `tanh`, `sigmoid`, `relu`, aggregation `sum` | Every `take_action` |
| Speciate a dict of living genomes and read `genome_to_species` | `refresh_living_species`, immigrants |

`feed_forward` is true. `initial_connection` is `unconnected`. `structural_mutation_surer` is true. Node keys follow neat-python: inputs are negative, outputs are `0 .. num_outputs-1`, hidden keys are positive. Disabled connections stay in the genome and are skipped by activation and by organ measurement. Immigration still forces child connections enabled, as `_create_immigrant_genome` does.

Golden tests, generated by the Python oracle at a fixed seed, lock:

- one founding genome's nodes, connections, and innovation ids
- one crossover-plus-mutate child
- `distance` for a handful of pairs
- `activate` on a fixed 31-vector, including an empty link list and a single-input tanh
- a 200-step world digest: positions rounded to 1e-6, energy, food count, births, species ids

A digest mismatch fails the test. Updating the golden file is a deliberate commit after the Python behavior itself changes.

## 9. Rendering

World clear color is `(18, 22, 28)`. Clouds are two translucent green discs (alpha `40 + 60 * intensity`, inner disc at 55% radius). Pellets use `PlantStrain.pigment`, radius 3, or radius 2 while `regrow_in > 0`. Eggs are `(230, 210, 160)` at radius 4. Trails fade along the 24-point path. The selection ring is `(255, 210, 60)`, two pixels outside the body. Sense rings are food `(80, 180, 100, 80)` for grazers, threat `(220, 90, 70, 80)`, and breeding `(150, 110, 220, 80)`. The minimap is 160px, top-right, with a gold viewport rectangle.

Body draw matches `draw_body`: on-screen radius `clamp(radius + 2, 6, 16)`, flagella behind, then the disc, jaw, horn, and claw. Colors come from body genes. A genome with no genes falls back to the species HSL palette in `RendererCommon.get_species_color`.

The genome graph matches `genome_viz.py`: input column, up to eight hidden nodes, output column, enabled edges only, blue for positive weights and red for negative, label `Nn Mc`.

HUD, inspect panel, scoreboard, log, and settings are DOM. They sit over the canvas. They use `textContent` for every string that originated in the world, including scientific names. The page never assigns world strings to `innerHTML`.

## 10. Save, settings, and links

A snapshot is a versioned JSON object: config, seed, RNG state, clock, genomes, organisms, eggs, ecology, fitness tables, species table, scoreboard records, and camera. Version 1 is the only version this spec defines. Load rejects a newer version with a visible error and leaves the running world alone.

Save and Load are buttons. Autosave is off, so a mis-click cannot overwrite a stored world. The host never sees the snapshot.

Restart applies settings and starts a new seed unless the visitor kept the current seed. Defaults are the copied JSON. The drawer cannot set `render_backend` or turn on a server.

## 11. Publish checklist

Before the app is published from `web/`:

- `npm run build` succeeds and the preview opens on the living world, not a blank canvas.
- The built folder contains no `.env`, no source map, and no credential file.
- No third-party script or font URL. Vite bundles the worker and the renderer.
- No server route. No `cursor-app.json`.
- Scientific names, logs, and settings errors are text nodes.
- A hidden tab drops CPU use to idle.
- The narrow layout (400px wide) still pans, selects, and opens the scoreboard.
- WebGL blocked: Canvas 2D still shows clouds, pellets, bodies, and organs.

Suggested slug when someone publishes: `living-world`. Visibility is the publisher's choice in the dialog. The app is built to run public, with no sign-in.

## 12. Tests

Vitest covers every ported class. The bar is the same one the repository already uses for Python: each class has tests for the methods that change the world, not a line-coverage ritual pasted over empty methods.

Required cases:

- distance and spatial query membership for a known set of points
- network input length 31, carnivore food channels zero, unit food direction
- graze bands, bite damage, egg fill and hatch, growth spending
- registry capacity, cull order, birth id allocation
- selection: no cull under the cap, cull at the cap, species id restored after re-cluster
- camera clamp, drag, and smooth track
- genome graph column assignment
- Python golden digest in section 8

`python3 -m unittest discover -s tests` stays green. This port does not change those tests.

## 13. Acceptance

A reviewer can publish `web/` and, in the resulting page:

1. Watch founders move, eat, and leave trails in a 4000×4000 world.
2. Pan, track a body, jump with the minimap, toggle sense rings, and read the inspect panel including the genome graph.
3. See at least one egg and one hatch in a run long enough for the founding breed bias to fire, or by lowering `breeding_cooldown` in the drawer and restarting.
4. See the epoch line and a selection line in the log.
5. Save, reload the page, and Load back to the same step and the same bodies.
6. Pause from a phone-width window using the on-screen controls, with the scoreboard in a drawer.

## 14. Out of scope

- Replacing or deleting the Python desktop app.
- Porting the episodic harness, GPU shader source from `living_world_gl.py`, or `moderngl`.
- A shared multiplayer world, accounts, or a hosted tick server.
- Bit-identical pygame pixels. Positions, colors, organ layout, and panel contents match. Text rasterization will differ.
- Automatic retrain of desktop populations inside the page.
