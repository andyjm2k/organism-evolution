"""Unit tests for bites and health."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from combat import apply_bite, bite_damage, hunt_prey


def _body(size, health, carnivore, position=(0, 0)):
    """Minimal creature with an empty genome so the bite uses the default pinch."""
    organism = type("Body", (), {})()
    organism.size = size
    organism.health = health
    organism.max_health = health
    organism.max_energy = 100
    organism.energy = 40
    organism.is_carnivore = carnivore
    organism.position = position
    organism.get_radius = lambda: 6
    organism.genome = type("Genome", (), {"connections": {}})()
    organism.logging_level = "quiet"
    organism.organisms_consumed = 0
    organism.steps_since_last_hunt = 0
    organism.avg_steps_between_hunts = []
    organism.fitness_bonus = 0
    return organism


class TestCombat(unittest.TestCase):
    """Cover damage, a killing bite, and the hunt loop."""

    def test_bite_damage_grows_with_size_advantage(self):
        """A larger attacker hits harder than a peer."""
        hunter = _body(2.0, 100, True)
        peer = _body(2.0, 100, False)
        smaller = _body(1.0, 100, False)
        self.assertGreater(bite_damage(hunter, smaller), bite_damage(hunter, peer))

    def test_apply_bite_kills_by_zeroing_energy(self):
        """Health at 0 is death, which the rest of the sim reads as no energy."""
        hunter = _body(2.0, 100, True)
        prey = _body(1.0, 15, False, position=(1, 0))
        self.assertTrue(apply_bite(hunter, prey))
        self.assertEqual(prey.health, 0)
        self.assertEqual(prey.energy, 0)

    def test_apply_bite_leaves_a_sturdy_prey_alive(self):
        """One pinch does not drop a large health pool."""
        hunter = _body(1.0, 100, True)
        prey = _body(1.0, 500, False)
        self.assertFalse(apply_bite(hunter, prey))
        self.assertGreater(prey.health, 0)
        self.assertEqual(prey.energy, 40)

    def test_hunt_prey_counts_a_kill_and_ignores_other_carnivores(self):
        """A touching smaller grazer is eaten. Another carnivore is not."""
        hunter = _body(2.0, 100, True, position=(10, 10))
        grazer = _body(1.0, 10, False, position=(11, 10))
        rival = _body(1.0, 10, True, position=(10, 11))
        before = hunter.energy
        hunt_prey(hunter, [hunter, rival, grazer])
        self.assertEqual(grazer.energy, 0)
        self.assertEqual(hunter.organisms_consumed, 1)
        self.assertGreater(hunter.energy, before)
        self.assertEqual(rival.energy, 40)


if __name__ == "__main__":
    unittest.main()
