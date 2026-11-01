"""Unit tests for genome-linked body traits and their panel text."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import neat
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from body_traits import BodyTraits, draw_body
from genome_bootstrap import install_founding_motif


def _neat_config():
    """Load the project NEAT config so founding wires can be installed."""
    root = Path(__file__).resolve().parents[1]
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )


def _founding_genome():
    """Return one genome with the shared living-world starting wires."""
    config = _neat_config()
    # Population construction attaches the innovation tracker the motif needs.
    neat.Population(config)
    genome = neat.DefaultGenome(1)
    genome.configure_new(config.genome_config)
    genome.fitness = 1.0
    install_founding_motif(genome, config.genome_config)
    return config, genome


def _organism(genome, is_carnivore):
    """Minimal organism stub carrying the physical fields the panel reads."""
    return SimpleNamespace(
        genome=genome,
        is_carnivore=is_carnivore,
        size=0.65,
        speed=0.85,
        get_radius=lambda: 4,
    )


class TestBodyTraits(unittest.TestCase):
    """Cover organ readout, diet-dependent labels, and panel text."""

    def test_herbivore_founder_expresses_jaw_and_horn(self):
        """Food wires are a jaw. Threat wires on a grazer are a horn."""
        _config, genome = _founding_genome()
        traits = BodyTraits.from_organism(_organism(genome, False))
        self.assertGreater(traits.jaw, 1.0)
        self.assertGreater(traits.horn, 1.0)
        self.assertEqual(traits.claw, 0.0)
        self.assertEqual(traits.flagella, 0.0)
        labels = [name for name, _strength in traits.present()]
        self.assertEqual(labels, ["Jaw", "Horn"])

    def test_carnivore_founder_expresses_jaw_and_claw(self):
        """The same prey channels are a claw once the diet is carnivore."""
        _config, genome = _founding_genome()
        traits = BodyTraits.from_organism(_organism(genome, True))
        self.assertGreater(traits.claw, 1.0)
        self.assertEqual(traits.horn, 0.0)
        labels = [name for name, _strength in traits.present()]
        self.assertIn("Claw", labels)
        self.assertNotIn("Horn", labels)

    def test_core_to_motor_wire_is_a_flagellum(self):
        """A body-sensor wire into velocity grows a flagellum."""
        config, genome = _founding_genome()
        innovation = config.genome_config.innovation_tracker.get_innovation_number(
            -1, 0, "flagellum-test"
        )
        connection = genome.create_connection(config.genome_config, -1, 0, innovation)
        connection.weight = 1.6
        connection.enabled = True
        genome.connections[connection.key] = connection
        traits = BodyTraits.from_organism(_organism(genome, False))
        self.assertGreaterEqual(traits.flagella, 1.6)
        self.assertIn("Flagella", [name for name, _strength in traits.present()])

    def test_disabled_wire_does_not_form_an_organ(self):
        """A disabled gene remains in the genome but adds no anatomy."""
        _config, genome = _founding_genome()
        for connection in genome.connections.values():
            connection.enabled = False
        traits = BodyTraits.from_organism(_organism(genome, False))
        self.assertEqual(traits.present(), [])

    def test_panel_lines_include_size_speed_and_organs(self):
        """The inspection panel lists body stats and the organs that exist."""
        _config, genome = _founding_genome()
        organism = _organism(genome, False)
        lines = BodyTraits.from_organism(organism).panel_lines(organism)
        self.assertIn("Size 0.65", lines[0])
        self.assertIn("Speed 0.85", lines[0])
        self.assertTrue(any("Jaw" in line and "Horn" in line for line in lines[1:]))

    def test_plain_genome_panel_says_plain_cell(self):
        """No sensor wires means the panel does not invent organs."""
        _config, genome = _founding_genome()
        genome.connections.clear()
        organism = _organism(genome, False)
        lines = BodyTraits.from_organism(organism).panel_lines(organism)
        self.assertIn("plain cell", lines[1])

    def test_draw_body_paints_pixels_for_expressed_organs(self):
        """Drawing a jawed body changes the surface, not only the text panel."""
        pygame.init()
        surface = pygame.Surface((80, 80), pygame.SRCALPHA)
        before = pygame.image.tostring(surface, "RGBA")
        traits = BodyTraits(jaw=3.6, claw=0.0, horn=3.6, flagella=1.5)
        draw_body(surface, 40, 40, (80, 160, 90), 6, traits)
        after = pygame.image.tostring(surface, "RGBA")
        self.assertNotEqual(before, after)
        pygame.quit()


if __name__ == "__main__":
    unittest.main()
