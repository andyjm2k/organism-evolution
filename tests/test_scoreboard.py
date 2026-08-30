"""Unit tests for scoreboard record and dashboard behavior."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scoreboard import Scoreboard


def _organism_stub(is_carnivore=False):
    """Minimal organism stand-in for record_species."""
    return SimpleNamespace(
        get_radius=lambda: 5,
        calculate_spikes=lambda: 4,
        calculate_spike_length=lambda: 3,
        is_carnivore=is_carnivore,
    )


class TestScoreboard(unittest.TestCase):
    """Cover last_seen updates and dashboard level plumbing."""

    def setUp(self):
        Scoreboard.initialize(dashboard_level="normal")

    def test_last_seen_updates_when_fitness_not_improved(self):
        # Species should remain tracked even when fitness drops.
        org = _organism_stub()
        Scoreboard.record_species("1", org, fitness=100, generation=1, config=None)
        Scoreboard.record_species("1", org, fitness=50, generation=3, config=None)
        record = Scoreboard.get_records()["1"]
        self.assertEqual(record["last_seen"], 3)
        self.assertEqual(record["highest_fitness"], 100)

    def test_dashboard_level_is_configurable(self):
        # Dashboard level should not rely on stack introspection.
        Scoreboard.set_dashboard_level("minimal")
        self.assertEqual(Scoreboard._dashboard_level, "minimal")

    def test_display_dashboard_uses_active_species_from_stats(self):
        """Terminal dashboard should show active species from generation stats."""
        org_one = _organism_stub(is_carnivore=False)
        org_two = _organism_stub(is_carnivore=True)
        Scoreboard.record_species("1", org_one, fitness=100, generation=1, config=None)
        Scoreboard.record_species("2", org_two, fitness=90, generation=1, config=None)
        stats = {
            "active_species": 2,
            "carnivores": 1,
            "herbivores": 1,
        }
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            Scoreboard.display_terminal_dashboard(
                generation=1,
                dashboard_level="normal",
                generation_stats=stats,
            )
        output = buffer.getvalue()
        self.assertIn("Active Species: 2", output)
        self.assertIn("Total Evolved: 2", output)


if __name__ == "__main__":
    unittest.main()
