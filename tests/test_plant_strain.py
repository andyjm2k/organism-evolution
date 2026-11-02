"""Unit tests for evolving plant strains and grazing."""

import random
import sys
import unittest
from pathlib import Path

import neat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food import Food
from food_ecology import FoodEcology
from organism import Organism
from plant_strain import PlantStrain, graze, make_founder_strain


def _grazer(position):
    """A herbivore standing on a pellet."""
    root = Path(__file__).resolve().parents[1]
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )
    neat.Population(config)
    genome = neat.DefaultGenome(1)
    genome.configure_new(config.genome_config)
    genome.fitness = 1.0
    env = {
        "width": 400,
        "height": 400,
        "starting_energy": 80,
        "food_energy_value": 40,
        "movement_cost": 0.01,
    }
    organism = Organism(genome, config, position, env, species_id=1, diet_key=1)
    organism.energy = 40
    return organism


class TestPlantStrain(unittest.TestCase):
    """Cover relation bands, slow mutation, and the three bite results."""

    def test_band_thresholds(self):
        """Relation is split into symbiotic, indifferent, and predatory."""
        self.assertEqual(PlantStrain(relation=0.35).band(), "symbiotic")
        self.assertEqual(PlantStrain(relation=0.0).band(), "indifferent")
        self.assertEqual(PlantStrain(relation=-0.35).band(), "predatory")

    def test_founder_strain_stays_edible(self):
        """Starting strains do not open in the predatory band."""
        for seed in range(8):
            strain = make_founder_strain(rng=random.Random(seed))
            self.assertNotEqual(strain.band(), "predatory")

    def test_mutate_stays_in_range(self):
        """A large nudge cannot leave the legal gene ranges."""
        child = PlantStrain(relation=1.0, nutrition=1.6, strike=2.5, regrowth=2.0)
        born = child.mutate(random.Random(1), variance=5.0)
        self.assertGreaterEqual(born.relation, -1.0)
        self.assertLessEqual(born.relation, 1.0)
        self.assertGreaterEqual(born.nutrition, 0.4)
        self.assertLessEqual(born.strike, 2.5)

    def test_add_success_seeds_on_the_eighth_point(self):
        """Seeds are infrequent so a strain drifts instead of flipping at once."""
        strain = PlantStrain()
        self.assertFalse(strain.add_success(3))
        self.assertFalse(strain.add_success(4))
        self.assertTrue(strain.add_success(1))
        self.assertEqual(strain.success, 0.0)

    def test_pigment_shifts_from_green_toward_red(self):
        """Predatory strains are redder than symbiotic ones."""
        helpful = PlantStrain(relation=0.8).pigment()
        hunter = PlantStrain(relation=-0.8).pigment()
        self.assertGreater(hunter[0], helpful[0])
        self.assertGreater(helpful[1], hunter[1])

    def test_indifferent_pellet_is_removed(self):
        """A pellet with no strain is still a normal meal."""
        grazer = _grazer((10, 10))
        food = Food(11, 10)
        graze(grazer, [food])
        self.assertIsNone(food.position)
        self.assertEqual(grazer.food_consumed, 1)

    def test_symbiotic_pellet_regrows_in_place(self):
        """Grazing a symbiotic plant leaves it to regrow and scores a success."""
        grazer = _grazer((10, 10))
        strain = PlantStrain(relation=0.8, nutrition=1.0, regrowth=1.0)
        food = Food(11, 10, strain=strain)
        notes = []
        food.ecology = type("Eco", (), {"note_plant_event": lambda self, item, amount: notes.append(amount)})()
        graze(grazer, [food])
        self.assertIsNotNone(food.position)
        self.assertGreater(food.regrow_in, 0)
        self.assertEqual(notes, [1.0])
        graze(grazer, [food])
        self.assertEqual(grazer.food_consumed, 1)

    def test_predatory_pellet_costs_health(self):
        """Biting a predatory plant hurts and uproots it."""
        grazer = _grazer((10, 10))
        before = grazer.health
        strain = PlantStrain(relation=-0.8, nutrition=1.0, strike=1.0)
        food = Food(11, 10, strain=strain)
        graze(grazer, [food])
        self.assertLess(grazer.health, before)
        self.assertIsNone(food.position)
        self.assertGreater(grazer.energy, 0)

    def test_note_plant_event_drops_a_seed(self):
        """Eight points of success add one mutated pellet to the cloud."""
        ecology = FoodEcology(
            {
                "environment_width": 200,
                "environment_height": 200,
                "food_target_density": 10,
                "num_food_items": 0,
                "nutrient_cloud_count": 1,
            }
        )
        cloud = ecology.clouds[0]
        strain = cloud.strains[0]
        strain.success = 7
        food = ecology._spawn_near_cloud(cloud, strain)
        before = len(ecology.food_items)
        ecology.note_plant_event(food, 1)
        self.assertGreater(len(ecology.food_items), before)
        self.assertGreaterEqual(len(cloud.strains), 2)

    def test_seed_does_not_raise_food_above_the_cap(self):
        """A successful strain still mutates when the ground is already full."""
        ecology = FoodEcology(
            {
                "environment_width": 200,
                "environment_height": 200,
                "food_target_density": 10,
                "num_food_items": 0,
                "nutrient_cloud_count": 1,
            }
        )
        cloud = ecology.clouds[0]
        strain = cloud.strains[0]
        cap = ecology.food_cap()
        while ecology.active_food_count() < cap:
            ecology.food_items.append(ecology._spawn_near_cloud(cloud, strain))
        strain.success = 7.0
        strains_before = len(cloud.strains)
        ecology.note_plant_event(ecology.food_items[0], 1)
        self.assertEqual(ecology.active_food_count(), cap)
        self.assertGreater(len(cloud.strains), strains_before)


if __name__ == "__main__":
    unittest.main()
