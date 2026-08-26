# End-to-End Bug Sweep Plan

Systematic sweep of every runtime module, config contract, and test gap.

## Sweep phases

| Phase | Scope | Method |
|---|---|---|
| 1 | Static contract audit | Map config keys → readers; NEAT I/O count; public APIs |
| 2 | Module walk | `main` → `simulation` → `organism` → `scoreboard` → `renderer` → helpers |
| 3 | Dynamic smoke | Headless `eval_genomes`, unit tests, import graph |
| 4 | Triage | Tag P0/P1/P2/P3 with repro + fix owner |
| 5 | Fix | P1 correctness first, then P2 performance/hygiene |

## Module checklist

- **distance.py** — squared vs linear API consistency
- **network_inputs.py** — length 22, herbivore threats, carnivore prey
- **fitness.py** — bonus inclusion, role penalties
- **organism.py** — single update path, breeding gates, reset completeness
- **simulation.py** — sensing units, trial fitness, scoreboard timing, headless pygame
- **scoreboard.py** — record updates, dashboard config plumbing
- **renderer.py** — display init, undefined attrs, draw arg mismatch
- **main.py** — CWD-independent paths, pygame lifecycle
- **config** — unused keys, NEAT required fields

## Sweep execution log

See `docs/BUG_SWEEP_FINDINGS.md` for confirmed issues and fix status.
