"""Steady-state NEAT selection for the living-world harness."""

import random

from genome_bootstrap import bootstrap_immigrant_genome
from logging_util import log_always


class SelectionScheduler:
    """Periodic culling and immigration using steady-state NEAT (Approach B)."""

    def __init__(self, interval_steps, max_population, cull_fraction, immigration_rate):
        # Steps between selection events.
        self.interval_steps = max(1, int(interval_steps))
        # Population cap shared with the registry.
        self.max_population = max_population
        # Fraction of living organisms to cull each selection tick.
        self.cull_fraction = float(cull_fraction)
        # Probability of injecting fresh genomes when under cap.
        self.immigration_rate = float(immigration_rate)

    def should_run(self, step):
        """True on selection interval boundaries (excluding step zero)."""
        return step > 0 and step % self.interval_steps == 0

    def run(
        self, population, registry, fitness_tracker, batch_engine, neat_config, sim_config=None
    ):
        """Cull weak organisms and inject offspring of top performers."""
        alive = registry.alive_organisms()
        if not alive:
            log_always("Selection: population extinct — reseeding immigrants")
            self._inject_immigrants(
                population,
                registry,
                fitness_tracker,
                batch_engine,
                neat_config,
                sim_config,
                5,
            )
            return
        fitness_tracker.apply_to_genomes(population.population)
        cull_count = max(1, int(len(alive) * self.cull_fraction))
        culled = registry.cull_worst(fitness_tracker, cull_count)
        log_always(
            f"Selection: culled {len(culled)} organisms "
            f"({registry.count()} remain)"
        )
        deficit = self.max_population - registry.count()
        if deficit > 0 and random.random() < self.immigration_rate:
            inject = min(deficit, max(1, int(deficit * 0.3)))
            self._inject_immigrants(
                population,
                registry,
                fitness_tracker,
                batch_engine,
                neat_config,
                sim_config,
                inject,
            )

    def _inject_immigrants(
        self,
        population,
        registry,
        fitness_tracker,
        batch_engine,
        neat_config,
        sim_config,
        count,
    ):
        """Spawn crossover offspring or random genomes into open world slots."""
        created = 0
        top_ids = fitness_tracker.top_genome_ids(2)
        for _ in range(count):
            if registry.at_capacity():
                break
            genome = _create_immigrant_genome(
                population, neat_config, top_ids, sim_config
            )
            if genome is None:
                break
            genome_id = genome.key
            population.population[genome_id] = genome
            population.species.speciate(
                neat_config, population.population, population.generation
            )
            fitness_tracker.init_genome(genome_id)
            organism = registry.spawn_immigrant(genome, population, batch_engine)
            if organism is not None:
                created += 1
        if created:
            log_always(f"Selection: injected {created} immigrant organisms")


def _create_immigrant_genome(population, neat_config, top_ids, sim_config=None):
    """Build a mutated crossover child or a fresh enhanced random genome."""
    genome_id = max(population.population.keys(), default=0) + 1
    if len(top_ids) >= 2 and all(tid in population.population for tid in top_ids):
        genome = neat_config.genome_type(genome_id)
        parent_a = population.population[top_ids[0]]
        parent_b = population.population[top_ids[1]]
        parent_a.fitness = parent_a.fitness if parent_a.fitness is not None else 0.0
        parent_b.fitness = parent_b.fitness if parent_b.fitness is not None else 0.0
        genome.configure_crossover(parent_a, parent_b, neat_config.genome_config)
        genome.mutate(neat_config.genome_config)
        for connection in genome.connections.values():
            connection.enabled = True
        genome.key = genome_id
        return genome
    genome = bootstrap_immigrant_genome(neat_config, sim_config)
    genome.key = genome_id
    return genome
