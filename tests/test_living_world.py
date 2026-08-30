"""Unit tests for living-world ecology and camera systems."""

import sys
import unittest
from pathlib import Path

import neat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera import Camera
from food_ecology import FoodEcology, NutrientCloud
from living_world import LivingWorldSimulation
from rolling_fitness import RollingFitnessTracker


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


class TestCamera(unittest.TestCase):
    """Cover viewport pan and coordinate transforms."""

    def test_pan_and_world_to_screen(self):
        camera = Camera(1000, 1000, 200, 200)
        camera.pan(50, 30)
        sx, sy = camera.world_to_screen(100, 100)
        wx, wy = camera.screen_to_world(sx, sy)
        self.assertAlmostEqual(wx, 100, places=3)
        self.assertAlmostEqual(wy, 100, places=3)

    def test_center_on_clamps_to_world(self):
        camera = Camera(1000, 1000, 200, 200)
        camera.center_on(999, 999)
        self.assertLessEqual(camera.offset_x, 800)
        self.assertLessEqual(camera.offset_y, 800)


class TestRollingFitness(unittest.TestCase):
    """Cover lifetime fitness accumulation."""

    def test_offspring_bonus_increases_parent_score(self):
        tracker = RollingFitnessTracker({"fitness_offspring_weight": 100})
        tracker.init_genome(1)
        tracker.record_offspring([1, 2])
        self.assertGreaterEqual(tracker.get(1), 100)


class TestLivingWorldHarness(unittest.TestCase):
    """Cover headless living-world stepping without reset."""

    def test_living_world_runs_without_reset(self):
        root = Path(__file__).resolve().parents[1]
        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(root / "config" / "neat-config.ini"),
        )
        sim_config = {
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
        sim = LivingWorldSimulation(neat_config, sim_config)
        sim.run()
        self.assertEqual(sim.clock.step, 25)
        self.assertGreater(len(sim.food_ecology.clouds), 0)


if __name__ == "__main__":
    unittest.main()
