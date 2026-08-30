"""Unit tests for NEAT configuration validity."""

import sys
import unittest
from pathlib import Path

import neat

# Allow imports from src/ when running tests from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestNeatConfig(unittest.TestCase):
    """Ensure NEAT settings allow reproduction with the configured population."""

    def test_initial_population_can_reproduce(self):
        """Reproduction must not fail when every genome has been evaluated once."""
        root = Path(__file__).resolve().parents[1]
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(root / "config" / "neat-config.ini"),
        )
        population = neat.Population(config)
        num_species = len(population.species.species)
        min_required = num_species * max(
            config.reproduction_config.min_species_size,
            config.reproduction_config.elitism,
        )
        self.assertLessEqual(
            min_required,
            config.pop_size,
            msg=(
                f"pop_size {config.pop_size} cannot satisfy "
                f"{num_species} species with elitism "
                f"{config.reproduction_config.elitism}"
            ),
        )
        for _genome_id, genome in population.population.items():
            genome.fitness = 1.0
        population.population = population.reproduction.reproduce(
            config,
            population.species,
            config.pop_size,
            population.generation,
        )
        self.assertEqual(len(population.population), config.pop_size)


if __name__ == "__main__":
    unittest.main()
