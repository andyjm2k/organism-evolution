"""Unit tests for maturity and growth."""

import sys
import unittest
from pathlib import Path

import neat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from growth import apply_maturity, grow_step
from organism import Organism


def _organism():
    """One adult organism with plenty of energy."""
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
        "starting_energy": 200,
        "movement_cost": 0.01,
        "growth_energy_cost": 0.2,
        "growth_step": 0.05,
        "breeding_cooldown": 10,
    }
    organism = Organism(genome, config, (200, 200), env, species_id=1, diet_key=1)
    organism.energy = 180
    organism.food_consumed = 1
    organism.steps_since_breeding = 100
    return organism


class TestGrowth(unittest.TestCase):
    """Cover scaling, the breeding gate, and paid growth."""

    def test_apply_maturity_shrinks_the_body(self):
        """Half maturity is half the adult size and stomach."""
        organism = _organism()
        adult = organism.adult_size
        adult_energy = organism._adult_max_energy
        organism.maturity = 0.5
        apply_maturity(organism)
        self.assertAlmostEqual(organism.size, adult * 0.5, places=5)
        self.assertAlmostEqual(organism.max_energy, adult_energy * 0.5, places=5)

    def test_juvenile_cannot_breed(self):
        """Breeding waits until the body is grown."""
        organism = _organism()
        self.assertTrue(organism.can_breed())
        organism.maturity = 0.4
        self.assertFalse(organism.can_breed())

    def test_grow_step_spends_energy_and_raises_maturity(self):
        """Surplus energy buys a step toward adulthood."""
        organism = _organism()
        organism.maturity = 0.4
        apply_maturity(organism)
        organism.energy = organism.max_energy * 0.9
        before = organism.energy
        grow_step(organism)
        self.assertGreater(organism.maturity, 0.4)
        self.assertLess(organism.energy, before)

    def test_grow_step_waits_when_hungry(self):
        """A juvenile below the surplus ratio does not grow."""
        organism = _organism()
        organism.maturity = 0.4
        apply_maturity(organism)
        organism.energy = organism.max_energy * 0.2
        grow_step(organism)
        self.assertAlmostEqual(organism.maturity, 0.4)


if __name__ == "__main__":
    unittest.main()
