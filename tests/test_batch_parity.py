"""Fitness parity tests: batch inference vs sequential NEAT activate."""

import random
import sys
import unittest
from pathlib import Path

import neat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simulation import Simulation


def _neat_config():
    """Load NEAT config from the repository."""
    root = Path(__file__).resolve().parents[1]
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )


def _base_sim_config(batch_inference):
    """Return a small headless config for parity evaluation."""
    return {
        "environment_width": 200,
        "environment_height": 200,
        "num_food_items": 10,
        "simulation_steps": 30,
        "num_trials": 1,
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
        "batch_inference": batch_inference,
        "logging_level": "normal",
    }


class TestBatchParity(unittest.TestCase):
    """Batch inference must produce identical fitness under fixed seeds."""

    def test_batch_inference_matches_sequential_fitness(self):
        """Same seed yields identical per-genome fitness batch on vs off."""
        neat_config = _neat_config()
        random.seed(424242)

        def run_once(batch_enabled):
            random.seed(424242)
            simulation = Simulation(neat_config, _base_sim_config(batch_enabled))
            population = neat.Population(neat_config)
            simulation.population = population
            genomes = list(population.population.items())[:12]
            simulation.eval_genomes(genomes, neat_config)
            return {
                genome_id: genome.fitness for genome_id, genome in genomes
            }

        sequential = run_once(False)
        batched = run_once(True)
        self.assertEqual(set(sequential.keys()), set(batched.keys()))
        for genome_id in sequential:
            self.assertAlmostEqual(
                sequential[genome_id],
                batched[genome_id],
                places=6,
                msg=f"Fitness mismatch for genome {genome_id}",
            )


if __name__ == "__main__":
    unittest.main()
