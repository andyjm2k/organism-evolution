"""Unit tests for BatchInferenceEngine registration and wiring."""

import sys
import unittest
from pathlib import Path

import neat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from batch_inference import BatchInferenceEngine
from compiled_network import CompiledNetwork
from organism import Organism


def _neat_config():
    """Load NEAT config from the project."""
    root = Path(__file__).resolve().parents[1]
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )


class TestBatchInferenceEngine(unittest.TestCase):
    """Cover compile/register/clear and organism compiled-network wiring."""

    def test_register_genome_compiles_network(self):
        """register_genome should store a CompiledNetwork for the genome id."""
        config = _neat_config()
        population = neat.Population(config)
        genome_id, genome = next(iter(population.population.items()))
        engine = BatchInferenceEngine()
        engine.register_genome(genome_id, genome, config)
        self.assertIn(genome_id, engine._networks)
        self.assertIsInstance(engine._networks[genome_id], CompiledNetwork)

    def test_clear_drops_compiled_networks(self):
        """clear() should remove all compiled networks."""
        config = _neat_config()
        population = neat.Population(config)
        genome_id, genome = next(iter(population.population.items()))
        engine = BatchInferenceEngine()
        engine.register_genome(genome_id, genome, config)
        engine.clear()
        self.assertEqual(len(engine._networks), 0)

    def test_organism_take_action_skips_neat_when_compiled(self):
        """take_action should use _compiled_network and not call neat activate."""
        config = _neat_config()
        population = neat.Population(config)
        genome_id, genome = next(iter(population.population.items()))
        env = {
            "width": 200,
            "height": 200,
            "detection_radius": 80,
            "food_detection_radius": 100,
            "threat_detection_radius": 60,
            "breeding_detection_radius": 70,
            "boundary_penalty": 0.5,
            "starting_energy": 150,
            "food_energy_value": 40,
            "movement_cost": 0.05,
        }
        engine = BatchInferenceEngine()
        engine.register_genome(genome_id, genome, config)
        organism = Organism(
            genome,
            config,
            (100, 100),
            env,
            species_id=1,
            logging_level="normal",
        )
        organism._compiled_network = engine._networks[genome_id]
        neat_called = []

        def _fail_neat(_inputs):
            neat_called.append(True)
            raise AssertionError("neat activate should not run when compiled is set")

        organism.network.activate = _fail_neat
        organism.take_action([], [], [], [])
        self.assertEqual(neat_called, [])


if __name__ == "__main__":
    unittest.main()
