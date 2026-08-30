"""Live organism registry with carrying-capacity enforcement."""

import random

from organism import Organism


class PopulationRegistry:
    """Track genome ids mapped to live organisms in the world."""

    def __init__(self, max_population, environment_config, neat_config, logging_level):
        # Hard cap on simultaneous living organisms.
        self.max_population = max_population
        # Shared arena settings passed to new organisms.
        self.environment_config = environment_config
        self.neat_config = neat_config
        self.logging_level = logging_level
        # genome_id -> Organism for all living agents.
        self._organisms = {}
        # Monotonic id allocator for in-world births.
        self._next_genome_id = 1

    def seed_from_genomes(self, genomes, population, batch_engine=None):
        """Instantiate organisms for an initial NEAT population."""
        width = self.environment_config["width"]
        height = self.environment_config["height"]
        for genome_id, genome in genomes:
            self._next_genome_id = max(self._next_genome_id, int(genome_id) + 1)
            x = random.randint(10, max(11, width - 10))
            y = random.randint(10, max(11, height - 10))
            species_id = population.species.get_species_id(genome_id)
            organism = self._make_organism(genome, (x, y), species_id, genome_id)
            if batch_engine is not None:
                batch_engine.register_genome(genome_id, genome, self.neat_config)
                organism._compiled_network = batch_engine._networks[genome_id]
            self._organisms[genome_id] = organism

    def _make_organism(self, genome, position, species_id, genome_id):
        """Construct one organism and attach metadata."""
        organism = Organism(
            genome,
            self.neat_config,
            position,
            self.environment_config,
            species_id=species_id,
            logging_level=self.logging_level,
        )
        organism.genome_id = genome_id
        organism._inference_genome_id = genome_id
        return organism

    def alive_organisms(self):
        """Return organisms with positive energy."""
        return [
            org for org in self._organisms.values() if org.energy > 0 and org.position
        ]

    def all_organisms(self):
        """Return every registered organism regardless of vitality."""
        return list(self._organisms.values())

    def get(self, genome_id):
        """Lookup one organism by genome id."""
        return self._organisms.get(genome_id)

    def count(self):
        """Count living organisms."""
        return len(self.alive_organisms())

    def at_capacity(self):
        """True when no additional births should be admitted."""
        return self.count() >= self.max_population

    def add_birth(self, child_genome, child_organism, batch_engine=None):
        """Register an in-world birth with a fresh genome id."""
        genome_id = self._next_genome_id
        self._next_genome_id += 1
        child_genome.key = genome_id
        child_organism.genome_id = genome_id
        child_organism._inference_genome_id = genome_id
        if batch_engine is not None:
            batch_engine.register_genome(genome_id, child_genome, self.neat_config)
            child_organism._compiled_network = batch_engine._networks[genome_id]
        self._organisms[genome_id] = child_organism
        return genome_id

    def remove(self, genome_id):
        """Drop a dead organism from the registry."""
        self._organisms.pop(genome_id, None)

    def remove_dead(self):
        """Remove all organisms with zero or negative energy."""
        dead_ids = [
            gid for gid, org in self._organisms.items() if org.energy <= 0
        ]
        for genome_id in dead_ids:
            self.remove(genome_id)
        return dead_ids

    def cull_worst(self, fitness_tracker, count=1):
        """Remove lowest selection-fitness living organisms."""
        alive = self.alive_organisms()
        if not alive:
            return []
        ranked = sorted(
            alive,
            key=lambda org: fitness_tracker.get(org.genome_id),
        )
        removed = []
        for organism in ranked[: max(0, count)]:
            organism.energy = 0
            removed.append(organism.genome_id)
            self.remove(organism.genome_id)
        return removed

    def spawn_immigrant(self, genome, population, batch_engine=None):
        """Place one new genome into the world at a random location."""
        if self.at_capacity():
            return None
        width = self.environment_config["width"]
        height = self.environment_config["height"]
        genome_id = genome.key
        x = random.randint(10, max(11, width - 10))
        y = random.randint(10, max(11, height - 10))
        if genome_id not in population.species.genome_to_species:
            population.species.speciate(
                self.neat_config, population.population, population.generation
            )
        species_id = population.species.get_species_id(genome_id)
        organism = self._make_organism(genome, (x, y), species_id, genome_id)
        if batch_engine is not None:
            batch_engine.register_genome(genome_id, genome, self.neat_config)
            organism._compiled_network = batch_engine._networks[genome_id]
        self._organisms[genome_id] = organism
        return organism
