"""Headless tests for shared renderer helpers (RendererCommon)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from renderer_common import RendererCommon
from scoreboard import Scoreboard


class TestRendererCommon(unittest.TestCase):
    """Cover scoreboard surface building and color helpers without a display."""

    def setUp(self):
        """Use a lightweight RendererCommon instance (no pygame window)."""
        self.common = RendererCommon(logging_level="normal")

    def test_species_color_normalized_in_unit_interval(self):
        """GPU colors should be RGB floats in 0..1."""
        red, green, blue = self.common.species_color_normalized("42", False)
        for channel in (red, green, blue):
            self.assertGreaterEqual(channel, 0.0)
            self.assertLessEqual(channel, 1.0)

    def test_build_scoreboard_surface_waiting_state(self):
        """Empty scoreboard records produce a valid waiting surface."""
        Scoreboard.initialize()
        surface = self.common.build_scoreboard_surface(300)
        self.assertEqual(surface.get_width(), self.common.scoreboard_width)
        self.assertGreater(surface.get_height(), 0)

    def test_get_species_visual_caches_surface(self):
        """Species icon surfaces should be cached by visual parameters."""
        first = self.common.get_species_visual("9", True, 6, 4, 3, num_nodes=10)
        second = self.common.get_species_visual("9", True, 6, 4, 3, num_nodes=10)
        self.assertIs(first, second)

    def test_organism_sprite_cache_key_is_stable(self):
        """Sprite cache keys should match get_species_visual parameters."""
        key = self.common.organism_sprite_cache_key("9", True, 6, 4, 3)
        self.assertEqual(key, "9_True_6_4_3")

    def test_draw_breeding_zone_writes_label(self):
        """Breeding zone helper should render border and label text."""
        import pygame

        surface = pygame.Surface((400, 400), pygame.SRCALPHA)
        self.common.draw_breeding_zone(surface, 400, 400)
        self.assertIn("breeding_zone_text", self.common.text_surfaces)
        self.assertGreater(surface.get_at((40, 40))[3], 0)

    def test_count_active_species_ignores_dead_organisms(self):
        """Active species count should include only living organisms."""
        from types import SimpleNamespace

        organisms = [
            SimpleNamespace(species_id=1, energy=100),
            SimpleNamespace(species_id=2, energy=100),
            SimpleNamespace(species_id=3, energy=0),
        ]
        self.assertEqual(self.common.count_active_species(organisms), 2)

    def test_build_hud_text_matches_active_species(self):
        """HUD text should report the same active species count as the dashboard."""
        from types import SimpleNamespace

        organisms = [
            SimpleNamespace(species_id=1, energy=50),
            SimpleNamespace(species_id=2, energy=50),
        ]
        food = [SimpleNamespace(), SimpleNamespace(), SimpleNamespace()]
        text = self.common.build_hud_text(organisms, food, generation=11)
        self.assertIn("Species: 2", text)
        self.assertIn("Gen: 11", text)
        self.assertIn("Food: 3", text)


if __name__ == "__main__":
    unittest.main()
