"""Unit tests for living-world ecology, fitness, and harness integration."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import neat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_ecology import FoodEcology, NutrientCloud
from living_world import LivingWorldSimulation
from rolling_fitness import RollingFitnessTracker


def _neat_config():
    root = Path(__file__).resolve().parents[1]
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )


def _minimal_living_config(**overrides):
    config = {
        "environment_width": 400,
        "environment_height": 400,
        "max_population": 20,
        "max_world_steps": 25,
        "selection_interval_steps": 1000,
        "num_food_items": 15,
        "food_target_density": 15,
        "nutrient_cloud_count": 3,
        "detection_radius": 80,
        "food_detection_radius": 100,
        "threat_detection_radius": 60,
        "breeding_detection_radius": 70,
        "starting_energy": 150,
        "food_energy_value": 40,
        "movement_cost": 0.05,
        "render": False,
        "batch_inference": False,
        "logging_level": "normal",
    }
    config.update(overrides)
    return config


class TestNutrientCloud(unittest.TestCase):
    """Cover nutrient cloud drift and bounds wrapping."""

    def test_cloud_drifts_and_wraps(self):
        cloud = NutrientCloud(10, 10, 100, 0.8, 200, 200)
        for _ in range(500):
            cloud.tick()
        x, y = cloud.position
        self.assertGreaterEqual(x, 0)
        self.assertLess(x, 200)
        self.assertGreaterEqual(y, 0)
        self.assertLess(y, 200)


class TestFoodEcology(unittest.TestCase):
    """Cover food regrowth driven by nutrient clouds."""

    def test_regrowth_increases_food_under_scarcity(self):
        config = {
            "environment_width": 500,
            "environment_height": 500,
            "food_target_density": 40,
            "num_food_items": 10,
            "nutrient_cloud_count": 4,
            "nutrient_cloud_spawn_rate": 0.5,
            "food_scarcity_threshold": 0.9,
        }
        ecology = FoodEcology(config)
        start = ecology.active_food_count()
        for _ in range(200):
            ecology.tick()
        self.assertGreater(ecology.active_food_count(), start)

    def test_scarcity_ratio_reflects_depletion(self):
        config = {
            "environment_width": 300,
            "environment_height": 300,
            "food_target_density": 100,
            "num_food_items": 20,
            "nutrient_cloud_count": 2,
        }
        ecology = FoodEcology(config)
        self.assertLess(ecology.scarcity_ratio(), 0.5)

    def test_prune_consumed_trims_large_lists(self):
        config = {
            "environment_width": 200,
            "environment_height": 200,
            "food_target_density": 10,
            "num_food_items": 5,
            "nutrient_cloud_count": 1,
        }
        ecology = FoodEcology(config)
        for _ in range(50):
            ecology.food_items.append(ecology._spawn_random())
            ecology.food_items[-1].position = None
        ecology.prune_consumed()
        self.assertLessEqual(len(ecology.food_items), 50)


class TestRollingFitness(unittest.TestCase):
    """Cover lifetime fitness accumulation and genome sync."""

    def test_offspring_bonus_increases_parent_score(self):
        tracker = RollingFitnessTracker({"fitness_offspring_weight": 100})
        tracker.init_genome(1)
        tracker.record_offspring([1, 2])
        self.assertGreaterEqual(tracker.get(1), 100)
        self.assertEqual(tracker.offspring_count(1), 1)

    def test_tick_alive_adds_survival_score(self):
        tracker = RollingFitnessTracker({"fitness_survival_weight": 2.0})
        organism = MagicMock()
        organism.genome_id = 5
        organism.is_carnivore = False
        organism.food_consumed = 0
        organism.organisms_consumed = 0
        tracker.tick_alive(organism)
        self.assertEqual(tracker.get(5), 2.0)

    def test_finalize_death_adds_display_fitness(self):
        tracker = RollingFitnessTracker({})
        organism = MagicMock()
        organism.genome_id = 3
        tracker.finalize_death(organism, display_fitness=100.0)
        self.assertAlmostEqual(tracker.get(3), 5.0)

    def test_apply_to_genomes_sets_fitness(self):
        tracker = RollingFitnessTracker({})
        tracker.init_genome(1)
        tracker._scores[1] = 42.0
        genome = MagicMock()
        tracker.apply_to_genomes({1: genome})
        self.assertEqual(genome.fitness, 42.0)

    def test_top_genome_ids_orders_by_score(self):
        tracker = RollingFitnessTracker({})
        tracker._scores = {1: 10, 2: 50, 3: 30}
        self.assertEqual(tracker.top_genome_ids(2), [2, 3])


class TestLivingWorldHarness(unittest.TestCase):
    """Cover headless living-world stepping, selection, and births."""

    def test_living_world_runs_without_reset(self):
        sim = LivingWorldSimulation(_neat_config(), _minimal_living_config())
        sim.run()
        self.assertEqual(sim.clock.step, 25)
        self.assertGreater(len(sim.food_ecology.clouds), 0)

    def test_selection_runs_at_interval(self):
        sim = LivingWorldSimulation(
            _neat_config(),
            _minimal_living_config(
                max_world_steps=100,
                selection_interval_steps=50,
            ),
        )
        sim.run()
        self.assertEqual(sim.clock.step, 100)

    def test_register_birth_adds_to_pending(self):
        sim = LivingWorldSimulation(_neat_config(), _minimal_living_config())
        sim.population = neat.Population(_neat_config())
        genomes = list(sim.population.population.items())[:2]
        sim.registry.seed_from_genomes(genomes, sim.population)
        parent_a = sim.registry.get(genomes[0][0])
        parent_b = sim.registry.get(genomes[1][0])
        child_genome = sim.neat_config.genome_type(0)
        parent_a.genome.fitness = 1.0
        parent_b.genome.fitness = 0.5
        child_genome.configure_crossover(
            parent_a.genome, parent_b.genome, sim.neat_config.genome_config
        )
        child = sim.registry._make_organism(
            child_genome, (100, 100), parent_a.species_id, 0
        )
        sim.register_birth(child, child_genome, parent_a, parent_b)
        self.assertIn(child, sim._pending_births)
        self.assertGreater(sim.fitness_tracker.get(parent_a.genome_id), 0)


if __name__ == "__main__":
    unittest.main()
