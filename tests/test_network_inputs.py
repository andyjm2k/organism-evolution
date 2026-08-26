"""Unit tests for network input schema."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

# Allow imports from src/ when running tests from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from network_inputs import NUM_NETWORK_INPUTS, build_network_inputs


def _make_organism(is_carnivore=False):
    """Build a lightweight organism stub for input tests."""
    # Minimal attribute set required by build_network_inputs.
    return SimpleNamespace(
        position=(100.0, 100.0),
        last_position=(100.0, 100.0),
        energy=100.0,
        max_energy=200.0,
        speed=1.0,
        size=1.0,
        is_carnivore=is_carnivore,
        can_breed=lambda: True,
    )


class TestNetworkInputs(unittest.TestCase):
    """Ensure the NN input vector length and threat channels are correct."""

    def setUp(self):
        # Shared environment config for schema tests.
        self.env = {
            "width": 800,
            "height": 600,
            "detection_radius": 200,
            "food_detection_radius": 300,
            "threat_detection_radius": 100,
            "breeding_detection_radius": 150,
        }

    def test_input_length_is_twenty_two(self):
        # Both diets must produce exactly NUM_NETWORK_INPUTS channels.
        herbivore = _make_organism(False)
        carnivore = _make_organism(True)
        h_inputs = build_network_inputs(herbivore, [], [], [], [], self.env)
        c_inputs = build_network_inputs(carnivore, [], [], [], [], self.env)
        self.assertEqual(len(h_inputs), NUM_NETWORK_INPUTS)
        self.assertEqual(len(c_inputs), NUM_NETWORK_INPUTS)
        self.assertEqual(NUM_NETWORK_INPUTS, 22)

    def test_herbivore_receives_threat_channels(self):
        # Herbivores should encode nearby threats in channels 13-16.
        herbivore = _make_organism(False)
        threat = SimpleNamespace(position=(120.0, 100.0), is_carnivore=True)
        inputs = build_network_inputs(
            herbivore, [], [], [threat], [], self.env
        )
        # Channel 13 is threat count density; must be > 0 when a threat exists.
        self.assertGreater(inputs[13], 0.0)
        # Channel 14 is normalized distance; should be < 1 for a nearby threat.
        self.assertLess(inputs[14], 1.0)

    def test_carnivore_food_channels_are_zeroed(self):
        # Carnivores do not forage; food channels stay at the empty sentinel.
        carnivore = _make_organism(True)
        food = SimpleNamespace(position=(110.0, 100.0))
        inputs = build_network_inputs(carnivore, [food], [], [], [], self.env)
        # Empty food sentinel is [0, 1, 0, 0].
        self.assertEqual(inputs[9:13], [0.0, 1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
