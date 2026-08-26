"""Unit tests for simulation evaluation wiring."""

import sys
import unittest
from pathlib import Path

import neat

# Allow imports from src/ when running tests from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simulation import Simulation


class TestSimulation(unittest.TestCase):
    """Cover headless eval_genomes fitness assignment and scoreboard linkage."""

    def test_eval_genomes_assigns_fitness(self):
        # A short headless evaluation should assign numeric fitness values.
        root = Path(__file__).resolve().parents[1]
        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(root / "config" / "neat-config.ini"),
        )
        sim_config = {
            "environment_width": 200,
            "environment_height": 200,
            "num_food_items": 8,
            "simulation_steps": 10,
            "detection_radius": 80,
            "food_detection_radius": 100,
            "threat_detection_radius": 60,
            "breeding_detection_radius": 70,
            "starting_energy": 150,
            "food_energy_value": 40,
            "movement_cost": 0.05,
            "boundary_penalty": 0.5,
            "num_generations": 1,
            "render": False,
            "logging_level": "normal",
        }
        simulation = Simulation(neat_config, sim_config)
        population = neat.Population(neat_config)
        simulation.population = population
        # Evaluate a small slice of the population for speed.
        genomes = list(population.population.items())[:8]
        simulation.eval_genomes(genomes, neat_config)
        for _genome_id, genome in genomes:
            self.assertIsNotNone(genome.fitness)
            self.assertIsInstance(genome.fitness, (int, float))

    def test_legacy_config_aliases_are_honored(self):
        """Legacy petridish_size and episode_length keys map to modern fields."""
        # Load NEAT config from the project config directory.
        root = Path(__file__).resolve().parents[1]
        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(root / "config" / "neat-config.ini"),
        )
        # Use legacy keys only (no simulation_steps / environment_width).
        sim_config = {
            "petridish_size": 500,
            "episode_length": 100,
            "num_food_items": 5,
            "detection_radius": 80,
            "num_generations": 1,
            "render": False,
            "logging_level": "normal",
        }
        simulation = Simulation(neat_config, sim_config)
        # episode_length should populate simulation_steps.
        self.assertEqual(simulation.simulation_steps, 100)
        # petridish_size should populate environment width/height.
        self.assertEqual(simulation.environment_config["width"], 500)
        self.assertEqual(simulation.environment_config["height"], 500)


if __name__ == "__main__":
    unittest.main()
