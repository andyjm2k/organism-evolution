"""Entry point for the organism evolution simulation."""

import json
import os
import sys

import neat
import pygame

from logging_util import log_always
from scoreboard import Scoreboard
from simulation import Simulation


def _project_root():
    """Return the repository root (parent of src/)."""
    # main.py lives in src/; configs live one level up.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_simulation(render=True, logging_level="normal", dashboard_level="normal"):
    """Parse CLI flags, load configs, and run the NEAT simulation."""
    # Parse simple key=value CLI overrides.
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            lowered = arg.lower()
            if lowered.startswith("render="):
                render = arg.split("=", 1)[1].lower() == "true"
            elif lowered.startswith("logging="):
                logging_level = arg.split("=", 1)[1].lower()
            elif lowered.startswith("dashboard="):
                dashboard_level = arg.split("=", 1)[1].lower()

    root = _project_root()
    # Load simulation JSON relative to the project root (CWD-independent).
    sim_config_path = os.path.join(root, "config", "simulation-config.json")
    with open(sim_config_path, encoding="utf-8") as handle:
        sim_config = json.load(handle)

    # Apply CLI overrides into the live config.
    sim_config["render"] = render
    sim_config["logging_level"] = logging_level
    sim_config["dashboard_level"] = dashboard_level
    # Renderer owns the display surface; main only initializes pygame.
    pygame.init()
    sim_config["screen"] = None
    if not render:
        log_always("=== Running Simulation in Headless Mode ===")
        log_always("Rendering disabled. Species dashboard will use the terminal.")

    log_always(f"Logging level: {logging_level.upper()}")
    log_always(f"Dashboard level: {dashboard_level.upper()}")

    # Load NEAT configuration from the project config directory.
    neat_config_path = os.path.join(root, "config", "neat-config.ini")
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config_path,
    )

    # Reset scoreboard state for a fresh run.
    Scoreboard.initialize(dashboard_level=dashboard_level)
    simulation = Simulation(neat_config, sim_config)

    try:
        simulation.run()
    except KeyboardInterrupt:
        log_always("Simulation stopped by user")
    finally:
        log_always("Performing final cleanup...")
        # Drain events only when a display exists.
        if pygame.display.get_init():
            pygame.event.get()
        if getattr(simulation, "renderer", None):
            simulation.renderer.cleanup_resources()
        pygame.quit()
        log_always("Cleanup complete.")

    log_always("Generating final simulation summary...")
    Scoreboard.display_final_summary(logging_level)


if __name__ == "__main__":
    run_simulation()
