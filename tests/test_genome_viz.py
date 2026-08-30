"""Unit tests for genome visualization helpers."""

import sys
import unittest
from pathlib import Path

import neat
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genome_viz import _weight_color, build_genome_surface


def _neat_config():
    root = Path(__file__).resolve().parents[1]
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )


class TestGenomeViz(unittest.TestCase):
    """Cover genome surface rendering and weight coloring."""

    @classmethod
    def setUpClass(cls):
        if not pygame.get_init():
            pygame.init()
        cls.config = _neat_config()
        cls.population = neat.Population(cls.config)

    def test_build_genome_surface_returns_surface(self):
        genome = self.population.population[1]
        surface = build_genome_surface(genome, self.config)
        self.assertEqual(surface.get_width(), 260)
        self.assertEqual(surface.get_height(), 180)

    def test_build_genome_surface_handles_none_genome(self):
        surface = build_genome_surface(None, None)
        self.assertIsNotNone(surface)

    def test_build_genome_surface_with_font(self):
        genome = self.population.population[2]
        font = pygame.font.Font(None, 18)
        surface = build_genome_surface(genome, self.config, font=font)
        self.assertIsNotNone(surface)

    def test_weight_color_positive_and_negative(self):
        pos = _weight_color(20.0)
        neg = _weight_color(-20.0)
        self.assertNotEqual(pos, neg)
        self.assertEqual(len(pos), 3)
        self.assertEqual(len(neg), 3)


if __name__ == "__main__":
    unittest.main()
