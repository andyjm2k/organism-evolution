"""Unit tests for SelectionScheduler steady-state NEAT."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import neat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from population_registry import PopulationRegistry
from rolling_fitness import RollingFitnessTracker
from selection_scheduler import SelectionScheduler, _create_immigrant_genome


def _neat_config():
    root = Path(__file__).resolve().parents[1]
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )


def _env_config():
    return {
        "width": 300,
        "height": 300,
        "detection_radius": 80,
        "food_detection_radius": 100,
        "threat_detection_radius": 60,
        "breeding_detection_radius": 70,
        "starting_energy": 150,
        "food_energy_value": 40,
        "movement_cost": 0.05,
    }


class TestSelectionScheduler(unittest.TestCase):
    """Cover interval gating, culling, and immigrant injection."""

    def setUp(self):
        self.scheduler = SelectionScheduler(
            interval_steps=100,
            max_population=10,
            cull_fraction=0.2,
            immigration_rate=1.0,
        )

    def test_should_run_on_interval_not_at_zero(self):
        self.assertFalse(self.scheduler.should_run(0))
        self.assertTrue(self.scheduler.should_run(100))
        self.assertFalse(self.scheduler.should_run(50))

    def test_run_culls_when_population_alive(self):
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        registry = PopulationRegistry(10, _env_config(), neat_config, "normal")
        genomes = list(population.population.items())[:4]
        registry.seed_from_genomes(genomes, population)
        tracker = RollingFitnessTracker({})
        for gid, _ in genomes:
            tracker.init_genome(gid)
            tracker._scores[gid] = gid
        scheduler = SelectionScheduler(100, 10, 0.2, immigration_rate=0.0)
        scheduler.run(population, registry, tracker, None, neat_config)
        self.assertLess(registry.count(), 4)

    def test_run_reseeds_when_extinct(self):
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        registry = PopulationRegistry(10, _env_config(), neat_config, "normal")
        tracker = RollingFitnessTracker({})
        self.scheduler.run(population, registry, tracker, None, neat_config)
        self.assertGreater(registry.count(), 0)

    @patch("selection_scheduler.random.random", return_value=0.0)
    def test_run_injects_immigrants_when_under_cap(self, _mock_random):
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        registry = PopulationRegistry(20, _env_config(), neat_config, "normal")
        genomes = list(population.population.items())[:3]
        registry.seed_from_genomes(genomes, population)
        tracker = RollingFitnessTracker({})
        for gid, _ in genomes:
            tracker.init_genome(gid)
            tracker._scores[gid] = 1000
        self.scheduler.run(population, registry, tracker, None, neat_config)
        self.assertGreater(registry.count(), 2)


class TestCreateImmigrantGenome(unittest.TestCase):
    """Cover immigrant genome creation helpers."""

    def test_create_random_genome_without_top_performers(self):
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        genome = _create_immigrant_genome(population, neat_config, [])
        self.assertIsNotNone(genome)
        self.assertGreater(len(genome.nodes), 0)

    def test_create_crossover_when_top_parents_exist(self):
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        ids = list(population.population.keys())[:2]
        genome = _create_immigrant_genome(population, neat_config, ids)
        self.assertIsNotNone(genome)
        self.assertGreater(len(genome.connections), 0)


if __name__ == "__main__":
    unittest.main()
