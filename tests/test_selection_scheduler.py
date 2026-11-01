"""Unit tests for SelectionScheduler steady-state NEAT."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import neat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genome_bootstrap import install_founding_motif
from population_registry import PopulationRegistry
from rolling_fitness import RollingFitnessTracker
from selection_scheduler import (
    SelectionScheduler,
    _create_immigrant_genome,
    assign_species,
    refresh_living_species,
)


def _neat_config():
    root = Path(__file__).resolve().parents[1]
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )


def _env_config():
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


class TestSelectionScheduler(unittest.TestCase):
    """Cover interval gating, culling, and immigrant injection."""

    def setUp(self):
        self.scheduler = SelectionScheduler(
            interval_steps=100,
            max_population=10,
            cull_fraction=0.2,
            immigration_rate=1.0,
        )

    def test_should_run_on_interval_not_at_zero(self):
        self.assertFalse(self.scheduler.should_run(0))
        self.assertTrue(self.scheduler.should_run(100))
        self.assertFalse(self.scheduler.should_run(50))

    def test_run_does_not_cull_when_under_max_population(self):
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        registry = PopulationRegistry(10, _env_config(), neat_config, "normal")
        genomes = list(population.population.items())[:4]
        registry.seed_from_genomes(genomes, population)
        tracker = RollingFitnessTracker({})
        for gid, _ in genomes:
            tracker.init_genome(gid)
            tracker._scores[gid] = gid
        # Immigration off so under-cap selection cannot change headcount.
        scheduler = SelectionScheduler(100, 10, 0.2, immigration_rate=0.0)
        scheduler.run(population, registry, tracker, None, neat_config)
        self.assertEqual(registry.count(), 4)

    def test_run_culls_only_when_at_max_population(self):
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        # Cap equals the seeded headcount so selection is at capacity.
        registry = PopulationRegistry(4, _env_config(), neat_config, "normal")
        genomes = list(population.population.items())[:4]
        registry.seed_from_genomes(genomes, population)
        tracker = RollingFitnessTracker({})
        for gid, _ in genomes:
            tracker.init_genome(gid)
            tracker._scores[gid] = gid
        # Immigration off so only culling can reduce the living count.
        scheduler = SelectionScheduler(100, 4, 0.25, immigration_rate=0.0)
        scheduler.run(population, registry, tracker, None, neat_config)
        self.assertLess(registry.count(), 4)

    def test_run_reseeds_when_extinct(self):
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        registry = PopulationRegistry(10, _env_config(), neat_config, "normal")
        tracker = RollingFitnessTracker({})
        self.scheduler.run(population, registry, tracker, None, neat_config)
        self.assertGreater(registry.count(), 0)

    @patch("selection_scheduler.random.random", return_value=0.0)
    def test_run_injects_immigrants_when_under_cap(self, _mock_random):
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        registry = PopulationRegistry(20, _env_config(), neat_config, "normal")
        genomes = list(population.population.items())[:3]
        registry.seed_from_genomes(genomes, population)
        tracker = RollingFitnessTracker({})
        for gid, _ in genomes:
            tracker.init_genome(gid)
            tracker._scores[gid] = 1000
        self.scheduler.run(population, registry, tracker, None, neat_config)
        self.assertGreater(registry.count(), 2)


class TestCreateImmigrantGenome(unittest.TestCase):
    """Cover immigrant genome creation helpers."""

    def test_create_random_genome_without_top_performers(self):
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        genome = _create_immigrant_genome(population, neat_config, [])
        self.assertIsNotNone(genome)
        self.assertGreater(len(genome.nodes), 0)

    def test_create_crossover_when_top_parents_exist(self):
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        ids = list(population.population.keys())[:2]
        # Founders are unconnected, so give the fitter parent one link and
        # skip mutation. Crossover must copy that disjoint gene.
        fitter = population.population[ids[0]]
        other = population.population[ids[1]]
        fitter.fitness = 10.0
        other.fitness = 1.0
        genome_config = neat_config.genome_config
        innovation = genome_config.innovation_tracker.get_innovation_number(
            -1, 0, "test"
        )
        connection = neat.DefaultGenome.create_connection(
            genome_config, -1, 0, innovation
        )
        fitter.connections[connection.key] = connection
        fitter.diet_key = 1
        other.diet_key = 1
        with patch.object(neat.DefaultGenome, "mutate", lambda self, config: None):
            genome = _create_immigrant_genome(population, neat_config, ids)
        self.assertIsNotNone(genome)
        self.assertIn((-1, 0), genome.connections)
        self.assertEqual(genome.diet_key, 1)

    def test_crossover_without_parent_diet_stays_herbivore(self):
        """A missing diet must not fall through to the species-id carnivore rule."""
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        ids = list(population.population.keys())[:2]
        for genome_id in ids:
            population.population[genome_id].fitness = 1.0
        with patch.object(neat.DefaultGenome, "mutate", lambda self, config: None):
            genome = _create_immigrant_genome(population, neat_config, ids)
        self.assertEqual(genome.diet_key, 1)

    def test_extinction_reseed_spawns_herbivores(self):
        """Crossover immigrants eat plants instead of becoming hunter-only."""
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        for genome in population.population.values():
            genome.diet_key = 1
            genome.fitness = 1.0
        registry = PopulationRegistry(10, _env_config(), neat_config, "normal")
        tracker = RollingFitnessTracker({})
        for genome_id in list(population.population.keys())[:2]:
            tracker.init_genome(genome_id)
            tracker._scores[genome_id] = 50
        scheduler = SelectionScheduler(100, 10, 0.2, immigration_rate=1.0)
        scheduler.run(population, registry, tracker, None, neat_config)
        alive = registry.alive_organisms()
        self.assertGreater(len(alive), 0)
        self.assertTrue(all(not org.is_carnivore for org in alive))


class TestLivingSpeciation(unittest.TestCase):
    """Cover distance-based species assignment for living genomes."""

    def _founding_population(self):
        """Return a population whose first genome carries the founding wires."""
        neat_config = _neat_config()
        population = neat.Population(neat_config)
        genome = next(iter(population.population.values()))
        genome.fitness = 1.0
        install_founding_motif(genome, neat_config.genome_config)
        # Start species from this motif rather than the unconnected NEAT draw.
        population.species.species = {}
        population.species.genome_to_species = {}
        assign_species(population, neat_config, genome)
        return neat_config, population, genome

    def test_weight_jitter_stays_with_founders(self):
        """Small weight noise must not isolate a genome from its parents."""
        neat_config, population, genome = self._founding_population()
        parent_sid = population.species.get_species_id(genome.key)
        child = neat_config.genome_type(888001)
        child.fitness = 1.0
        child.configure_crossover(genome, genome, neat_config.genome_config)
        for connection in child.connections.values():
            connection.weight += 0.05
        child_sid = assign_species(population, neat_config, child)
        self.assertEqual(child_sid, parent_sid)

    def test_added_node_founds_a_species(self):
        """A new hidden node is far enough to leave the founding species."""
        neat_config, population, genome = self._founding_population()
        parent_sid = population.species.get_species_id(genome.key)
        child = neat_config.genome_type(888002)
        child.fitness = 1.0
        child.configure_crossover(genome, genome, neat_config.genome_config)
        child.mutate_add_node(neat_config.genome_config)
        child_sid = assign_species(population, neat_config, child)
        self.assertNotEqual(child_sid, parent_sid)

    def test_refresh_writes_species_ids_onto_organisms(self):
        """Selection re-speciation updates the ids organisms use for mating."""
        neat_config, population, genome = self._founding_population()
        sibling = neat_config.genome_type(888003)
        sibling.fitness = 1.0
        sibling.configure_crossover(genome, genome, neat_config.genome_config)
        for connection in sibling.connections.values():
            connection.weight += 0.02
        parent = type("Org", (), {})()
        parent.genome = genome
        parent.genome_id = genome.key
        parent.energy = 10
        parent.species_id = None
        child = type("Org", (), {})()
        child.genome = sibling
        child.genome_id = sibling.key
        child.energy = 10
        child.species_id = None
        count = refresh_living_species(population, neat_config, [parent, child])
        self.assertGreaterEqual(count, 1)
        self.assertEqual(parent.species_id, child.species_id)
        self.assertIsNotNone(parent.species_id)


if __name__ == "__main__":
    unittest.main()
