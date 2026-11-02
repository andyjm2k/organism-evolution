"""Unit tests for egg gestation and the living-world fertilize path."""

import sys
import unittest
from pathlib import Path

import neat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egg import Egg, fertilize
from organism import Organism


def _pair(eggs_enabled):
    """Two ready adults of one species."""
    root = Path(__file__).resolve().parents[1]
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )
    neat.Population(config)
    env = {
        "width": 800,
        "height": 600,
        "starting_energy": 200,
        "movement_cost": 0.01,
        "breeding_cooldown": 10,
        "breeding_detection_radius": 80,
        "offspring_starting_energy": 40,
        "eggs_enabled": eggs_enabled,
        "egg_energy_goal": 20,
        "egg_hatch_delay": 3,
        "hatch_maturity": 0.35,
    }
    organisms = []
    for index, x in ((1, 400), (2, 430)):
        genome = neat.DefaultGenome(index)
        genome.configure_new(config.genome_config)
        genome.fitness = 1.0
        organism = Organism(genome, config, (x, 300), env, species_id=1, diet_key=1)
        organism.energy = 200
        organism.food_consumed = 1
        organism.steps_since_breeding = 50
        organisms.append(organism)
    return organisms


class TestEgg(unittest.TestCase):
    """Cover filling, laying, hatching, and fertilize versus instant birth."""

    def test_gestate_fills_only_while_willing(self):
        """Energy moves into the egg when desire is high, and stops when it is not."""
        parent = type("Parent", (), {"energy": 100})()
        egg = Egg(parent, parent, None, None, energy_goal=6, hatch_delay=2)
        self.assertFalse(egg.gestate(parent, desire=0.0))
        self.assertEqual(egg.stored, 0)
        self.assertFalse(egg.gestate(parent, desire=1.0))
        self.assertGreater(egg.stored, 0)
        egg.stored = 5
        self.assertTrue(egg.gestate(parent, desire=1.0))

    def test_tick_hatches_after_the_delay(self):
        """A laid egg waits out its delay before it is ready."""
        egg = Egg(None, None, None, None, energy_goal=1, hatch_delay=3)
        self.assertFalse(egg.tick())
        egg.lay((10, 10))
        self.assertFalse(egg.tick())
        self.assertFalse(egg.tick())
        self.assertTrue(egg.tick())

    def test_fertilize_starts_a_juvenile_egg_without_registering(self):
        """Eggs-enabled mating carries an egg instead of calling register_birth."""
        parent, partner = _pair(True)
        births = []

        class World:
            def register_birth(self, child, genome, parent_a, parent_b):
                births.append(child)

            def receive_egg(self, egg):
                self.egg = egg

        world = World()
        parent.simulation = world
        partner.simulation = world
        parent._try_breed([partner])
        self.assertEqual(births, [])
        self.assertIsNotNone(parent.carried_egg)
        self.assertAlmostEqual(parent.carried_egg.child.maturity, 0.35)
        self.assertLess(parent.carried_egg.child.size, parent.adult_size)
        # Desire high enough to finish the small test egg.
        parent.network = type("Net", (), {"activate": lambda self, inputs: [0, 0, 1, -1]})()
        for _ in range(30):
            if parent.carried_egg is None:
                break
            parent.take_action([], [partner], [], [partner])
        self.assertIsNone(parent.carried_egg)
        self.assertTrue(world.egg.laid)
        self.assertEqual(len(births), 0)
        self.assertFalse(world.egg.tick())
        self.assertFalse(world.egg.tick())
        self.assertTrue(world.egg.tick())


if __name__ == "__main__":
    unittest.main()
