"""Unit tests for NumPy compiled NEAT networks."""

import random
import sys
import unittest
from pathlib import Path

import neat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compiled_network import CompiledNetwork


def _load_neat_config():
    """Load project NEAT config."""
    root = Path(__file__).resolve().parents[1]
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )


class TestCompiledNetwork(unittest.TestCase):
    """Ensure compiled forward pass matches neat-python activate."""

    def test_matches_neat_activate_on_random_inputs(self):
        """Compiled outputs match FeedForwardNetwork for random inputs."""
        config = _load_neat_config()
        population = neat.Population(config)
        random.seed(12345)
        for _genome_id, genome in list(population.population.items())[:8]:
            reference = neat.nn.FeedForwardNetwork.create(genome, config)
            compiled = CompiledNetwork.from_genome(genome, config)
            for _ in range(20):
                inputs = [random.random() for _ in range(31)]
                expected = reference.activate(inputs)
                actual = compiled.activate(inputs)
                for index, (exp, got) in enumerate(zip(expected, actual)):
                    self.assertAlmostEqual(
                        exp,
                        got,
                        places=9,
                        msg=f"Output {index} mismatch for genome {_genome_id}",
                    )

    def test_output_length_is_four(self):
        """Network must return four outputs per neat-config."""
        config = _load_neat_config()
        population = neat.Population(config)
        _genome_id, genome = next(iter(population.population.items()))
        compiled = CompiledNetwork.from_genome(genome, config)
        outputs = compiled.activate([0.5] * 31)
        self.assertEqual(len(outputs), 4)


if __name__ == "__main__":
    unittest.main()
