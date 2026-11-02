"""Unit tests for heritable body genes."""

import random
import sys
import unittest
from pathlib import Path

import neat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from body_genes import (
    BodyGenes,
    apply_body_genes,
    body_color,
    inherit_body_genes,
    make_founder_genes,
)
from organism import Organism


def _config():
    """Load the project NEAT config."""
    root = Path(__file__).resolve().parents[1]
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )


def _genome():
    """Return a configured genome with an innovation tracker attached."""
    config = _config()
    neat.Population(config)
    genome = neat.DefaultGenome(1)
    genome.configure_new(config.genome_config)
    genome.fitness = 1.0
    return config, genome


class TestBodyGenes(unittest.TestCase):
    """Cover copy, mutation bounds, inheritance, and diet expression."""

    def test_copy_matches_and_is_independent(self):
        """A copy starts equal and does not follow later edits."""
        original = BodyGenes(0.2, 0.4, 0.6, 0.1, 1.0)
        copied = original.copy()
        original.red = 0.9
        self.assertEqual(copied.red, 0.2)
        self.assertEqual(copied.color_rgb(), (51, 102, 153))

    def test_mutate_stays_inside_legal_ranges(self):
        """A large variance still cannot push pigment, diet, or size out of range."""
        genes = BodyGenes(0.0, 1.0, 0.5, 0.0, 0.75)
        genes.mutate(random.Random(1), variance=5.0)
        self.assertGreaterEqual(genes.red, 0.0)
        self.assertLessEqual(genes.red, 1.0)
        self.assertGreaterEqual(genes.green, 0.0)
        self.assertLessEqual(genes.green, 1.0)
        self.assertGreaterEqual(genes.blue, 0.0)
        self.assertLessEqual(genes.blue, 1.0)
        self.assertGreaterEqual(genes.diet, 0.0)
        self.assertLessEqual(genes.diet, 1.0)
        self.assertGreaterEqual(genes.size, 0.75)
        self.assertLessEqual(genes.size, 1.35)

    def test_diet_threshold_splits_grazers_and_hunters(self):
        """Diet below 0.5 grazes. Diet at 0.5 hunts."""
        self.assertFalse(BodyGenes(0, 0, 0, 0.49, 1).is_carnivore())
        self.assertTrue(BodyGenes(0, 0, 0, 0.5, 1).is_carnivore())

    def test_founder_genes_stay_herbivores_of_similar_color(self):
        """The starting stock grazes and stays near the shared green pigment."""
        genes = make_founder_genes(random.Random(2))
        self.assertFalse(genes.is_carnivore())
        self.assertEqual(genes.size, 1.0)
        self.assertLess(abs(genes.green - 0.70), 0.2)

    def test_child_inherits_fitter_parent_then_moves_a_little(self):
        """The child copies the fitter parent, not a blend, and is not identical."""
        _config, child = _genome()
        fitter = BodyGenes(0.8, 0.1, 0.1, 0.1, 1.0)
        other = BodyGenes(0.1, 0.1, 0.8, 0.9, 1.2)
        parent_a = type("Genome", (), {"body_genes": fitter, "fitness": 5.0})()
        parent_b = type("Genome", (), {"body_genes": other, "fitness": 1.0})()
        born = inherit_body_genes(child, parent_a, parent_b, rng=random.Random(3), variance=0.02)
        self.assertIs(child.body_genes, born)
        # Closer to the fitter red parent than to the blue parent.
        self.assertGreater(born.red, born.blue)
        self.assertFalse(born.is_carnivore())
        self.assertNotEqual((born.red, born.green, born.blue), (0.8, 0.1, 0.1))

    def test_inherit_without_parent_genes_leaves_the_child_alone(self):
        """Episodic genomes with no body genes stay on the legacy diet rule."""
        _config, child = _genome()
        parent = type("Genome", (), {})()
        self.assertIsNone(inherit_body_genes(child, parent, parent))
        self.assertFalse(hasattr(child, "body_genes"))

    def test_apply_body_genes_sets_diet_and_color(self):
        """An organism expresses the genome's diet and skin color."""
        config, genome = _genome()
        genome.body_genes = BodyGenes(0.2, 0.3, 0.4, 0.8, 1.2)
        organism = Organism(
            genome,
            config,
            (40, 40),
            {"width": 200, "height": 200, "starting_energy": 100, "movement_cost": 0.01},
            species_id=1,
            diet_key=1,
        )
        self.assertTrue(organism.is_carnivore)
        self.assertEqual(organism._diet_key, 3)
        # 0.3 * 255 is 76.5, and Python rounds halves to even.
        self.assertEqual(body_color(organism), (51, 76, 102))
        # Size gene scales the body that the network complexity produced.
        plain = genome
        plain.body_genes = BodyGenes(0.2, 0.3, 0.4, 0.8, 1.0)
        baseline = Organism(
            plain,
            config,
            (40, 40),
            {"width": 200, "height": 200, "starting_energy": 100, "movement_cost": 0.01},
            species_id=1,
        )
        genome.body_genes = BodyGenes(0.2, 0.3, 0.4, 0.8, 1.2)
        larger = Organism(
            genome,
            config,
            (40, 40),
            {"width": 200, "height": 200, "starting_energy": 100, "movement_cost": 0.01},
            species_id=1,
        )
        self.assertAlmostEqual(larger.size, baseline.size * 1.2, places=5)

    def test_apply_skips_genomes_without_genes(self):
        """Missing genes leave the organism on whatever diet it already had."""
        organism = type("Org", (), {"genome": type("G", (), {})(), "is_carnivore": False})()
        self.assertFalse(apply_body_genes(organism))
        self.assertFalse(hasattr(organism, "body_genes"))


if __name__ == "__main__":
    unittest.main()
