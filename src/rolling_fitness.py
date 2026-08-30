"""Rolling lifetime fitness for steady-state selection in the living world."""


class RollingFitnessTracker:
    """Accumulate selection fitness across an organism's lifetime."""

    def __init__(self, weights):
        # Per-genome accumulated selection fitness.
        self._scores = {}
        # Offspring count per genome for reproductive success.
        self._offspring = {}
        # Weight configuration from simulation JSON.
        self.weights = weights

    def init_genome(self, genome_id):
        """Ensure a genome id has tracker slots."""
        self._scores.setdefault(genome_id, 0.0)
        self._offspring.setdefault(genome_id, 0)

    def get(self, genome_id):
        """Return current selection fitness for one genome."""
        return self._scores.get(genome_id, 0.0)

    def record_offspring(self, parent_ids):
        """Credit parents when an in-world birth succeeds."""
        weight = float(self.weights.get("fitness_offspring_weight", 500))
        for genome_id in parent_ids:
            self.init_genome(genome_id)
            self._offspring[genome_id] = self._offspring.get(genome_id, 0) + 1
            self._scores[genome_id] += weight

    def tick_alive(self, organism):
        """Add small survival and role-appropriate rewards each step."""
        genome_id = getattr(organism, "genome_id", None)
        if genome_id is None:
            return
        self.init_genome(genome_id)
        survival = float(self.weights.get("fitness_survival_weight", 0.5))
        self._scores[genome_id] += survival
        if organism.is_carnivore:
            self._scores[genome_id] += float(
                self.weights.get("fitness_kill_weight", 40)
            ) * 0.001 * organism.organisms_consumed
        else:
            self._scores[genome_id] += float(
                self.weights.get("fitness_food_weight", 15)
            ) * 0.001 * organism.food_consumed

    def finalize_death(self, organism, display_fitness=0.0):
        """Flush remaining display fitness into the selection score on death."""
        genome_id = getattr(organism, "genome_id", None)
        if genome_id is None:
            return
        self.init_genome(genome_id)
        self._scores[genome_id] += display_fitness * 0.05

    def offspring_count(self, genome_id):
        """Return how many offspring a genome has produced."""
        return self._offspring.get(genome_id, 0)

    def apply_to_genomes(self, genomes_dict):
        """Copy accumulated scores onto NEAT genome.fitness fields."""
        for genome_id, genome in genomes_dict.items():
            genome.fitness = self.get(genome_id)

    def top_genome_ids(self, limit=5):
        """Return highest-scoring genome ids."""
        ranked = sorted(self._scores.items(), key=lambda item: item[1], reverse=True)
        return [genome_id for genome_id, _score in ranked[:limit]]
