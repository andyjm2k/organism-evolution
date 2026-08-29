"""Unit tests for organism consumption, breeding gates, and fitness bonus."""

import sys
import unittest
from pathlib import Path

import neat

# Allow imports from src/ when running tests from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food import Food
from organism import Organism


def _load_neat_config():
    """Load the project NEAT config for constructing real genomes."""
    # Resolve config relative to the repository root.
    root = Path(__file__).resolve().parents[1]
    path = root / "config" / "neat-config.ini"
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path),
    )


def _make_organism(species_id=1, position=(400, 400), is_carnivore=None):
    """Create an organism from a fresh NEAT population genome."""
    # Population construction attaches an innovation tracker.
    config = _load_neat_config()
    population = neat.Population(config)
    _genome_id, genome = next(iter(population.population.items()))
    env = {
        "width": 800,
        "height": 600,
        "detection_radius": 200,
        "food_detection_radius": 300,
        "threat_detection_radius": 100,
        "breeding_detection_radius": 150,
        "boundary_penalty": 0.5,
        "starting_energy": 150,
        "food_energy_value": 40,
        "movement_cost": 0.05,
    }
    organism = Organism(
        genome,
        config,
        position,
        env,
        species_id=species_id,
        logging_level="normal",
    )
    # Override diet when a test needs a specific role.
    if is_carnivore is not None:
        organism.is_carnivore = is_carnivore
    return organism


class TestOrganism(unittest.TestCase):
    """Cover consumption, breeding readiness, and fitness bonus inclusion."""

    def test_starting_energy_honors_config(self):
        # Organisms should spawn near the configured starting energy.
        organism = _make_organism()
        self.assertLessEqual(organism.energy, 150.0)
        self.assertGreater(organism.energy, 0.0)

    def test_herbivore_eats_food_once(self):
        # Herbivores consume colliding food via the single update path.
        organism = _make_organism(is_carnivore=False)
        food = Food(organism.position[0] + 1, organism.position[1] + 1)
        before = organism.food_consumed
        organism.check_for_food([food])
        self.assertEqual(organism.food_consumed, before + 1)
        self.assertIsNone(food.position)

    def test_carnivore_ignores_food(self):
        # Carnivores must not forage plant food.
        organism = _make_organism(is_carnivore=True)
        food = Food(organism.position[0] + 1, organism.position[1] + 1)
        organism.check_for_food([food])
        self.assertEqual(organism.food_consumed, 0)
        self.assertIsNotNone(food.position)

    def test_can_breed_energy_gate_is_unified(self):
        # Breeding requires energy >= 100 (unified gate).
        organism = _make_organism(position=(400, 300))
        organism.energy = 99
        organism.steps_since_breeding = 1000
        self.assertFalse(organism.can_breed())
        organism.energy = 100
        self.assertTrue(organism.can_breed())

    def test_update_increments_steps_once(self):
        # Counters should increment only in update (not also in take_action).
        organism = _make_organism(is_carnivore=False)
        before = organism.steps_taken
        organism.take_action([], [], [], [])
        self.assertEqual(organism.steps_taken, before)
        organism.update([], [organism])
        self.assertEqual(organism.steps_taken, before + 1)

    def test_reset_clears_highest_fitness(self):
        # Peak fitness must not leak across episode trials.
        organism = _make_organism()
        organism.steps_taken = 10
        organism.calculate_fitness()
        self.assertGreater(organism.highest_fitness, 0.0)
        organism.reset(organism.environment_config)
        self.assertEqual(organism.highest_fitness, 0.0)

    def test_fitness_includes_bonus(self):
        organism = _make_organism()
        base = organism.calculate_fitness()
        organism.fitness_bonus += 500
        boosted = organism.calculate_fitness()
        self.assertGreater(boosted, base)

    def test_continuous_movement_updates_position(self):
        organism = _make_organism(is_carnivore=False)
        organism.energy = 200
        start = organism.position
        organism.network = _SteeringNetwork([1.0, 1.0, -1.0, -1.0])
        organism.take_action([], [], [], [])
        self.assertNotEqual(organism.position, start)
        self.assertGreater(len(organism.movement_trail), 0)

    def test_rest_output_prevents_movement(self):
        organism = _make_organism(is_carnivore=False)
        organism.energy = 200
        start = organism.position
        organism.network = _SteeringNetwork([1.0, 1.0, -1.0, 1.0])
        organism.take_action([], [], [], [])
        self.assertEqual(organism.position, start)
        self.assertEqual(len(organism.movement_trail), 0)


class _SteeringNetwork:
    """Stub network returning fixed steering outputs for movement tests."""

    def __init__(self, outputs):
        self._outputs = outputs

    def activate(self, _inputs):
        return list(self._outputs)


if __name__ == "__main__":
    unittest.main()
