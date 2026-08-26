"""Unit tests for episode-local NEAT breeding (replaces genetics stub)."""

import sys
import unittest
from pathlib import Path

import neat

# Allow imports from src/ when running tests from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism import Organism
from simulation import Simulation


def _load_neat_config():
    """Load the project NEAT config for constructing real genomes."""
    # Resolve config relative to the repository root.
    root = Path(__file__).resolve().parents[1]
    path = root / "config" / "neat-config.ini"
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path),
    )


def _fresh_genome(config):
    """Return a genome from a fresh NEAT population (innovation tracker set)."""
    # Population construction attaches the innovation tracker NEAT requires.
    population = neat.Population(config)
    _genome_id, genome = next(iter(population.population.items()))
    return genome


def _env_config():
    """Return a minimal environment config for organism tests."""
    return {
        "width": 800,
        "height": 600,
        "detection_radius": 200,
        "food_detection_radius": 300,
        "threat_detection_radius": 100,
        "breeding_detection_radius": 150,
        "boundary_penalty": 0.5,
        "starting_energy": 200,
        "food_energy_value": 40,
        "movement_cost": 0.05,
    }


class TestEpisodeBreeding(unittest.TestCase):
    """Cover episode-local offspring registration and NEAT crossover breeding."""

    def test_register_episode_child_queues_without_neat_mutation(self):
        """Children are queued on Simulation, not inserted into NEAT population."""
        # Build a headless simulation with a tiny arena for speed.
        config = _load_neat_config()
        sim_config = {
            "environment_width": 200,
            "environment_height": 200,
            "num_food_items": 4,
            "simulation_steps": 5,
            "detection_radius": 80,
            "num_generations": 1,
            "render": False,
            "logging_level": "normal",
        }
        simulation = Simulation(config, sim_config)
        population = neat.Population(config)
        simulation.population = population
        # Record population size before registering a child.
        before_count = len(population.population)
        child_genome = _fresh_genome(config)
        child = Organism(
            child_genome,
            config,
            (100, 100),
            simulation.environment_config,
            species_id=1,
            logging_level="normal",
        )
        # Register should queue the child without touching NEAT population.
        simulation.register_episode_child(child)
        self.assertEqual(len(simulation._episode_children), 1)
        self.assertIs(simulation._episode_children[0], child)
        self.assertEqual(len(population.population), before_count)

    def test_breeding_produces_episode_local_child(self):
        """Breeding with simulation attached registers an episode-local offspring."""
        # Two ready partners in the same species inside the breeding safe zone.
        config = _load_neat_config()
        env = _env_config()
        parent_a = Organism(
            _fresh_genome(config),
            config,
            (400, 300),
            env,
            species_id=2,
            logging_level="normal",
        )
        parent_b = Organism(
            _fresh_genome(config),
            config,
            (405, 300),
            env,
            species_id=2,
            logging_level="normal",
        )
        parent_a.energy = 200
        parent_b.energy = 200
        parent_a.steps_since_breeding = 1000
        parent_b.steps_since_breeding = 1000
        # Mock simulation captures register_episode_child calls.
        queued = []

        class StubSimulation:
            def register_episode_child(self, child):
                queued.append(child)

        parent_a.simulation = StubSimulation()
        parent_a._try_breed([parent_b])
        # Breeding should enqueue exactly one child when simulation is wired.
        self.assertEqual(len(queued), 1)
        self.assertIsInstance(queued[0], Organism)
        self.assertEqual(queued[0].species_id, 2)

    def test_child_genome_is_crossover_not_parent_clone(self):
        """Offspring genome is a NEAT crossover, not a copy of either parent."""
        # Crossover should produce a distinct genome object from both parents.
        config = _load_neat_config()
        env = _env_config()
        parent_a = Organism(
            _fresh_genome(config),
            config,
            (400, 300),
            env,
            species_id=3,
            logging_level="normal",
        )
        parent_b = Organism(
            _fresh_genome(config),
            config,
            (402, 300),
            env,
            species_id=3,
            logging_level="normal",
        )
        parent_a.energy = 200
        parent_b.energy = 200
        parent_a.steps_since_breeding = 1000
        parent_b.steps_since_breeding = 1000
        captured = []

        class StubSimulation:
            def register_episode_child(self, child):
                captured.append(child)

        parent_a.simulation = StubSimulation()
        parent_a._try_breed([parent_b])
        self.assertEqual(len(captured), 1)
        child = captured[0]
        # Child must not be the same object as either parent genome.
        self.assertIsNot(child.genome, parent_a.genome)
        self.assertIsNot(child.genome, parent_b.genome)
        # Crossover typically yields a non-empty connection set.
        self.assertGreater(len(child.genome.connections), 0)


if __name__ == "__main__":
    unittest.main()
