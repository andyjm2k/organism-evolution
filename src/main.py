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


def run_simulation(render=None, logging_level=None, dashboard_level=None):
    """Parse CLI flags, load configs, and run the NEAT simulation."""
    # Track CLI overrides separately so JSON defaults remain authoritative.
    render_override = None
    logging_override = None
    dashboard_override = None
    render_backend_override = None
    batch_inference_override = None
    # Parse simple key=value CLI overrides.
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            lowered = arg.lower()
            if lowered.startswith("render="):
                render_override = arg.split("=", 1)[1].lower() == "true"
            elif lowered.startswith("logging="):
                logging_override = arg.split("=", 1)[1].lower()
            elif lowered.startswith("dashboard="):
                dashboard_override = arg.split("=", 1)[1].lower()
            elif lowered.startswith("render_backend="):
                render_backend_override = arg.split("=", 1)[1].lower()
            elif lowered.startswith("batch_inference="):
                batch_inference_override = arg.split("=", 1)[1].lower() == "true"

    root = _project_root()
    # Load simulation JSON relative to the project root (CWD-independent).
    sim_config_path = os.path.join(root, "config", "simulation-config.json")
    with open(sim_config_path, encoding="utf-8") as handle:
        sim_config = json.load(handle)

    # Prefer CLI, then explicit function args, then JSON (training defaults headless).
    if render_override is not None:
        render = render_override
    elif render is None:
        render = bool(sim_config.get("render", False))
    if logging_override is not None:
        logging_level = logging_override
    elif logging_level is None:
        logging_level = sim_config.get("logging_level", "normal")
    if dashboard_override is not None:
        dashboard_level = dashboard_override
    elif dashboard_level is None:
        dashboard_level = sim_config.get("dashboard_level", "normal")
    if render_backend_override is not None:
        sim_config["render_backend"] = render_backend_override
    if batch_inference_override is not None:
        sim_config["batch_inference"] = batch_inference_override

    # Apply resolved options into the live config.
    sim_config["render"] = render
    sim_config["logging_level"] = logging_level
    sim_config["dashboard_level"] = dashboard_level
    sim_config["screen"] = None
    # Only initialize pygame when a display will be used (A-2 headless skip).
    if render:
        pygame.init()
    else:
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
        # Drain events and quit pygame only when it was initialized.
        if pygame.get_init():
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
