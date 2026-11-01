"""Steady-state NEAT selection for the living-world harness."""

import random

from neat.species import Species

from genome_bootstrap import bootstrap_immigrant_genome
from logging_util import log_always


class SelectionScheduler:
    """Periodic capacity culling and immigration using steady-state NEAT."""

    def __init__(self, interval_steps, max_population, cull_fraction, immigration_rate):
        # Steps between selection events.
        self.interval_steps = max(1, int(interval_steps))
        # Population cap shared with the registry.
        self.max_population = max_population
        # Fraction of living organisms to cull when at max population.
        self.cull_fraction = float(cull_fraction)
        # Probability of injecting fresh genomes when under cap.
        self.immigration_rate = float(immigration_rate)

    def should_run(self, step):
        """True on selection interval boundaries (excluding step zero)."""
        return step > 0 and step % self.interval_steps == 0

    def run(
        self, population, registry, fitness_tracker, batch_engine, neat_config, sim_config=None
    ):
        """Cull only at capacity, then inject offspring of top performers."""
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
            self._refresh_species(population, registry, neat_config)
            return
        # Sync rolling fitness onto NEAT genomes before any selection work.
        fitness_tracker.apply_to_genomes(population.population)
        # Forced culling is capacity pressure only — never shrink a living
        # population that is still under the max population limit.
        if registry.count() >= self.max_population:
            cull_count = max(1, int(len(alive) * self.cull_fraction))
            culled = registry.cull_worst(fitness_tracker, cull_count)
            log_always(
                f"Selection: culled {len(culled)} organisms at capacity "
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
        # Re-cluster whoever is still alive so species ids follow the genomes.
        self._refresh_species(population, registry, neat_config)

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
            # Place the immigrant against living species instead of re-clustering
            # only the original NEAT pool, which would drop birth genomes.
            assign_species(population, neat_config, genome)
            fitness_tracker.init_genome(genome_id)
            organism = registry.spawn_immigrant(genome, population, batch_engine)
            if organism is not None:
                created += 1
        if created:
            log_always(f"Selection: injected {created} immigrant organisms")

    def _refresh_species(self, population, registry, neat_config):
        """Re-speciate living organisms and log how many species that yields."""
        species_count = refresh_living_species(
            population, neat_config, registry.alive_organisms()
        )
        log_always(f"Selection: {species_count} living species")


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
        # Species ids are not a diet. Copy the parent role, and default to the
        # herbivore founders so a reseed can eat the plants in the world.
        diet = getattr(parent_a, "diet_key", None)
        if diet is None:
            diet = getattr(parent_b, "diet_key", None)
        genome.diet_key = 1 if diet is None else diet
        return genome
    genome = bootstrap_immigrant_genome(neat_config, sim_config)
    genome.key = genome_id
    return genome


def assign_species(population, neat_config, genome):
    """Put a genome in the nearest species, or start a species when it is too far."""
    # Threshold is the neat-config compatibility distance, not a fixed cutoff.
    threshold = population.species.species_set_config.compatibility_threshold
    genome_config = neat_config.genome_config
    best_sid = None
    best_distance = None
    for sid, species in population.species.species.items():
        representative = species.representative
        if representative is None:
            continue
        # Distance uses the same formula as NEAT speciation.
        distance = genome.distance(representative, genome_config)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_sid = sid
    # No living species, or farther than the threshold: this genome founds one.
    if best_sid is None or best_distance >= threshold:
        sid = next(population.species.indexer)
        created = Species(sid, population.generation)
        created.representative = genome
        created.members = {genome.key: genome}
        population.species.species[sid] = created
        population.species.genome_to_species[genome.key] = sid
        return sid
    species = population.species.species[best_sid]
    species.members[genome.key] = genome
    population.species.genome_to_species[genome.key] = best_sid
    return best_sid


def refresh_living_species(population, neat_config, organisms):
    """Re-speciate the current living genomes and write species ids back."""
    living = {}
    for organism in organisms:
        # Dead organisms must not anchor a species representative.
        if organism.energy <= 0 or organism.genome is None:
            continue
        organism.genome.key = organism.genome_id
        living[organism.genome_id] = organism.genome
    if not living:
        return 0
    # Drop stale species first. More old species than living genomes makes
    # NEAT's speciate step fail when it looks for a new representative.
    population.species.species = {}
    population.species.genome_to_species = {}
    population.species.speciate(neat_config, living, population.generation)
    for organism in organisms:
        sid = population.species.genome_to_species.get(organism.genome_id)
        if sid is not None:
            organism.species_id = sid
    return len(population.species.species)
