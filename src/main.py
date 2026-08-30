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
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse_harness_mode():
    """Read optional harness= CLI override."""
    for arg in sys.argv[1:]:
        lowered = arg.lower()
        if lowered.startswith("harness="):
            return arg.split("=", 1)[1].lower()
    return None


def _load_sim_config(root, harness_mode=None):
    """Load episodic or living-world JSON configuration."""
    if harness_mode is None:
        harness_mode = "episodic"
    config_name = (
        "living-world-config.json"
        if harness_mode == "living_world"
        else "simulation-config.json"
    )
    sim_config_path = os.path.join(root, "config", config_name)
    with open(sim_config_path, encoding="utf-8") as handle:
        sim_config = json.load(handle)
    sim_config["harness_mode"] = harness_mode
    return sim_config


def run_simulation(render=None, logging_level=None, dashboard_level=None):
    """Parse CLI flags, load configs, and run the selected simulation harness."""
    render_override = None
    logging_override = None
    dashboard_override = None
    render_backend_override = None
    batch_inference_override = None
    harness_mode = _parse_harness_mode()
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
    sim_config = _load_sim_config(root, harness_mode)
    harness_mode = sim_config.get("harness_mode", "episodic")

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

    sim_config["render"] = render
    sim_config["logging_level"] = logging_level
    sim_config["dashboard_level"] = dashboard_level
    sim_config["screen"] = None

    if render:
        pygame.init()
    else:
        log_always("=== Running Simulation in Headless Mode ===")
        log_always("Rendering disabled. Species dashboard will use the terminal.")

    log_always(f"Harness: {harness_mode}")
    log_always(f"Logging level: {logging_level.upper()}")
    log_always(f"Dashboard level: {dashboard_level.upper()}")

    neat_config_path = os.path.join(root, "config", "neat-config.ini")
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config_path,
    )

    Scoreboard.initialize(dashboard_level=dashboard_level)
    if harness_mode == "living_world":
        from living_world import LivingWorldSimulation

        simulation = LivingWorldSimulation(neat_config, sim_config)
    else:
        simulation = Simulation(neat_config, sim_config)

    try:
        simulation.run()
    except KeyboardInterrupt:
        log_always("Simulation stopped by user")
    finally:
        log_always("Performing final cleanup...")
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
