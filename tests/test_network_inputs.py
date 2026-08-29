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
    """Ensure the NN input vector length and sensing channels are correct."""

    def setUp(self):
        self.env = {
            "width": 800,
            "height": 600,
            "detection_radius": 200,
            "food_detection_radius": 300,
            "threat_detection_radius": 100,
            "breeding_detection_radius": 150,
        }

    def test_input_length_is_thirty_one(self):
        herbivore = _make_organism(False)
        carnivore = _make_organism(True)
        h_inputs = build_network_inputs(herbivore, [], [], [], [], self.env)
        c_inputs = build_network_inputs(carnivore, [], [], [], [], self.env)
        self.assertEqual(len(h_inputs), NUM_NETWORK_INPUTS)
        self.assertEqual(len(c_inputs), NUM_NETWORK_INPUTS)
        self.assertEqual(NUM_NETWORK_INPUTS, 31)

    def test_herbivore_receives_threat_channels(self):
        herbivore = _make_organism(False)
        threat = SimpleNamespace(position=(120.0, 100.0), is_carnivore=True)
        inputs = build_network_inputs(
            herbivore, [], [], [threat], [], self.env
        )
        self.assertGreater(inputs[16], 0.0)
        self.assertLess(inputs[17], 1.0)

    def test_second_nearest_food_is_reported(self):
        herbivore = _make_organism(False)
        near = SimpleNamespace(position=(120.0, 100.0))
        far = SimpleNamespace(position=(180.0, 100.0))
        inputs = build_network_inputs(
            herbivore, [near, far], [], [], [], self.env
        )
        self.assertLess(inputs[13], 1.0)
        self.assertLess(inputs[10], 1.0)
        self.assertGreater(inputs[13], inputs[10])

    def test_carnivore_food_channels_are_zeroed(self):
        carnivore = _make_organism(True)
        food = SimpleNamespace(position=(110.0, 100.0))
        inputs = build_network_inputs(carnivore, [food], [], [], [], self.env)
        self.assertEqual(inputs[9:16], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def test_closest_entity_prefers_nearer_candidate(self):
        from network_inputs import _closest_entity

        herbivore = _make_organism(False)
        near = SimpleNamespace(position=(120.0, 100.0))
        far = SimpleNamespace(position=(250.0, 100.0))
        count, dist_norm, dx_norm, _dy = _closest_entity(
            herbivore, [far, near], 300
        )
        self.assertGreater(count, 0.0)
        self.assertLess(dist_norm, 250.0 / 300.0)
        self.assertGreater(dx_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
