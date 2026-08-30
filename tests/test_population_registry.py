"""Unit tests for PopulationRegistry."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import neat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from population_registry import PopulationRegistry


def _neat_config():
    """Load NEAT config for registry tests."""
    root = Path(__file__).resolve().parents[1]
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )


def _env_config():
    """Minimal environment config for test organisms."""
    return {
        "width": 300,
        "height": 300,
        "detection_radius": 80,
        "food_detection_radius": 100,
        "threat_detection_radius": 60,
        "breeding_detection_radius": 70,
        "starting_energy": 150,
        "food_energy_value": 40,
        "movement_cost": 0.05,
    }


class TestPopulationRegistry(unittest.TestCase):
    """Cover registry seeding, births, removal, and culling."""

    def setUp(self):
        self.neat_config = _neat_config()
        self.registry = PopulationRegistry(
            max_population=5,
            environment_config=_env_config(),
            neat_config=self.neat_config,
            logging_level="normal",
        )
        self.population = neat.Population(self.neat_config)

    def test_seed_from_genomes_populates_registry(self):
        genomes = list(self.population.population.items())[:3]
        self.registry.seed_from_genomes(genomes, self.population)
        self.assertEqual(self.registry.count(), 3)
        for genome_id, _ in genomes:
            self.assertIsNotNone(self.registry.get(genome_id))

    def test_alive_organisms_excludes_dead(self):
        genomes = list(self.population.population.items())[:2]
        self.registry.seed_from_genomes(genomes, self.population)
        org = self.registry.get(genomes[0][0])
        org.energy = 0
        alive = self.registry.alive_organisms()
        self.assertEqual(len(alive), 1)

    def test_at_capacity_when_max_reached(self):
        genomes = list(self.population.population.items())[:5]
        self.registry.seed_from_genomes(genomes, self.population)
        self.assertTrue(self.registry.at_capacity())

    def test_add_birth_assigns_new_genome_id(self):
        genomes = list(self.population.population.items())[:1]
        self.registry.seed_from_genomes(genomes, self.population)
        parent = self.registry.get(genomes[0][0])
        child_genome = self.neat_config.genome_type(999)
        child_genome.configure_new(self.neat_config.genome_config)
        child = self.registry._make_organism(
            child_genome, (50, 50), parent.species_id, 999
        )
        new_id = self.registry.add_birth(child_genome, child)
        self.assertNotEqual(new_id, genomes[0][0])
        self.assertIs(self.registry.get(new_id), child)

    def test_remove_dead_clears_zero_energy_organisms(self):
        genomes = list(self.population.population.items())[:2]
        self.registry.seed_from_genomes(genomes, self.population)
        dead_id = genomes[0][0]
        self.registry.get(dead_id).energy = 0
        removed = self.registry.remove_dead()
        self.assertIn(dead_id, removed)
        self.assertIsNone(self.registry.get(dead_id))

    def test_cull_worst_removes_lowest_fitness(self):
        genomes = list(self.population.population.items())[:3]
        self.registry.seed_from_genomes(genomes, self.population)
        tracker = MagicMock()
        tracker.get.side_effect = lambda gid: 0 if gid == genomes[0][0] else 100
        culled = self.registry.cull_worst(tracker, count=1)
        self.assertEqual(culled, [genomes[0][0]])
        self.assertEqual(self.registry.count(), 2)

    def test_spawn_immigrant_respects_capacity(self):
        genomes = list(self.population.population.items())[:5]
        self.registry.seed_from_genomes(genomes, self.population)
        extra = self.neat_config.genome_type(9000)
        extra.configure_new(self.neat_config.genome_config)
        result = self.registry.spawn_immigrant(extra, self.population)
        self.assertIsNone(result)

    def test_all_organisms_includes_dead(self):
        genomes = list(self.population.population.items())[:2]
        self.registry.seed_from_genomes(genomes, self.population)
        self.registry.get(genomes[0][0]).energy = 0
        self.assertEqual(len(self.registry.all_organisms()), 2)

    def test_diet_stable_within_species(self):
        """Organisms in the same species share a carnivore/herbivore role."""
        genome_id, genome = next(iter(self.population.population.items()))
        org_a = self.registry._make_organism(genome, (10, 10), 4, genome_id)
        org_b = self.registry._make_organism(genome, (20, 20), 4, genome_id + 1)
        self.assertEqual(org_a.is_carnivore, org_b.is_carnivore)

    def test_diet_differs_across_species(self):
        """Different species ids can map to different diets."""
        genome_id, genome = next(iter(self.population.population.items()))
        herbivore = self.registry._make_organism(genome, (10, 10), 1, genome_id)
        carnivore = self.registry._make_organism(genome, (20, 20), 3, genome_id + 1)
        self.assertNotEqual(herbivore.is_carnivore, carnivore.is_carnivore)


if __name__ == "__main__":
    unittest.main()
