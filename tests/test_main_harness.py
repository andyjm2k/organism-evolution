"""Unit tests for main.py harness selection and config loading."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main import _load_sim_config, _project_root, _parse_harness_mode


class TestMainHarnessRouting(unittest.TestCase):
    """Cover episodic vs living-world config loading."""

    def test_load_episodic_config(self):
        root = _project_root()
        config = _load_sim_config(root, "episodic")
        self.assertEqual(config["harness_mode"], "episodic")
        self.assertIn("simulation_steps", config)

    def test_load_living_world_config(self):
        root = _project_root()
        config = _load_sim_config(root, "living_world")
        self.assertEqual(config["harness_mode"], "living_world")
        self.assertIn("max_population", config)
        self.assertIn("nutrient_cloud_count", config)

    def test_parse_harness_mode_from_argv(self):
        with mock.patch.object(sys, "argv", ["main.py", "harness=living_world"]):
            self.assertEqual(_parse_harness_mode(), "living_world")
        with mock.patch.object(sys, "argv", ["main.py"]):
            self.assertIsNone(_parse_harness_mode())


if __name__ == "__main__":
    unittest.main()
