"""Unit tests for living-world genesis genome bootstrapping."""

import sys
import unittest
from pathlib import Path

import neat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genome_bootstrap import bootstrap_immigrant_genome, bootstrap_population


def _neat_config():
    root = Path(__file__).resolve().parents[1]
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )


def _genesis_config():
    return {
        "genesis_archetype_count": 8,
        "genesis_extra_hidden_nodes": 6,
        "genesis_extra_connections": 15,
        "genesis_weight_jitter": 0.02,
        "genesis_foraging_bias": 0.3,
    }


class TestGenomeBootstrap(unittest.TestCase):
    """Cover archetype enhancement, speciation, and immigrant bootstrap."""

    def test_bootstrap_increases_nodes_and_enabled_connections(self):
        config = _neat_config()
        population = neat.Population(config)
        vanilla = list(population.population.values())[0]
        vanilla_nodes = len(vanilla.nodes)
        vanilla_enabled = sum(1 for c in vanilla.connections.values() if c.enabled)

        bootstrap_population(population, config, _genesis_config())
        sample = list(population.population.values())[0]
        enabled = sum(1 for c in sample.connections.values() if c.enabled)

        self.assertGreater(len(sample.nodes), vanilla_nodes)
        self.assertGreater(len(sample.connections), len(vanilla.connections))
        self.assertEqual(enabled, len(sample.connections))
        self.assertGreater(enabled, vanilla_enabled)

    def test_bootstrap_creates_multiple_species_with_breeding_groups(self):
        config = _neat_config()
        population = neat.Population(config)
        bootstrap_population(population, config, _genesis_config())

        species_sizes = [
            len(species.members) for species in population.species.species.values()
        ]
        self.assertGreaterEqual(len(species_sizes), 4)
        self.assertGreater(max(species_sizes), 1)

    def test_immigrant_genome_is_enhanced(self):
        config = _neat_config()
        population = neat.Population(config)
        vanilla = list(population.population.values())[0]
        immigrant = bootstrap_immigrant_genome(config, _genesis_config())

        self.assertGreater(len(immigrant.nodes), len(vanilla.nodes))
        enabled = sum(1 for c in immigrant.connections.values() if c.enabled)
        self.assertEqual(enabled, len(immigrant.connections))


if __name__ == "__main__":
    unittest.main()
