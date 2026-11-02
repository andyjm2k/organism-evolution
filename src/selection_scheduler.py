"""Steady-state NEAT selection for the living-world harness."""

import random

from neat.species import Species

from body_genes import inherit_body_genes
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
            self._refresh_species(population, registry, neat_config, sim_config)
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
        self._refresh_species(population, registry, neat_config, sim_config)

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
            # Place the immigrant against whoever is alive, not a stale
            # representative from a genome that has already died.
            living_ids = {
                org.genome_id
                for org in registry.alive_organisms()
                if getattr(org, "genome_id", None) is not None
            }
            assign_species(population, neat_config, genome, living_ids)
            fitness_tracker.init_genome(genome_id)
            organism = registry.spawn_immigrant(genome, population, batch_engine)
            if organism is not None:
                created += 1
        if created:
            log_always(f"Selection: injected {created} immigrant organisms")

    def _refresh_species(self, population, registry, neat_config, sim_config=None):
        """Re-speciate living organisms and log how many species that yields."""
        # Capture the threshold before retuning so the log can show a move.
        previous = float(population.species.species_set_config.compatibility_threshold)
        species_count = refresh_living_species(
            population, neat_config, registry.alive_organisms(), sim_config
        )
        updated = float(population.species.species_set_config.compatibility_threshold)
        if abs(updated - previous) >= 0.05:
            log_always(
                f"Selection: {species_count} living species "
                f"(compatibility {previous:.2f} -> {updated:.2f})"
            )
        else:
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
        # Body follows the first parent. The brain crossover already happened.
        inherit_body_genes(genome, parent_a, parent_b)
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


# Selection aims for a handful of species, and refuses to exceed one digit.
# Below the minimum it relaxes the threshold so distinct lineages can compete.
DEFAULT_TARGET_SPECIES = 6
DEFAULT_MIN_SPECIES = 4
DEFAULT_MAX_SPECIES = 9
# How far above the ini threshold a search will look. Fully diverged
# genomes at these distance weights top out inside this span.
_THRESHOLD_SEARCH_SPAN = 8.0


def assign_species(population, neat_config, genome, living_ids=None):
    """Put a genome in the nearest living species, or start one when it is too far."""
    # Drop dead members first so an ancient representative cannot isolate kin.
    if living_ids is not None:
        _drop_absent_members(population, living_ids)
    # Threshold is the neat-config compatibility distance, not a fixed cutoff.
    threshold = population.species.species_set_config.compatibility_threshold
    genome_config = neat_config.genome_config
    best_sid = None
    best_distance = None
    for sid, species in population.species.species.items():
        # Distance to the closest living member. A child of a drifted
        # lineage stays with that lineage even if the original founder is
        # still the representative and is now far away.
        for member in species.members.values():
            if member is None:
                continue
            distance = genome.distance(member, genome_config)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_sid = sid
        if species.members:
            continue
        representative = species.representative
        if representative is None:
            continue
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
    # If the representative died, follow the living member we matched.
    representative = species.representative
    rep_key = getattr(representative, "key", None)
    if representative is None or rep_key not in species.members:
        species.representative = genome
    return best_sid


def _drop_absent_members(population, living_ids):
    """Forget genomes that are no longer alive so distance uses the living cloud."""
    empty = []
    for sid, species in population.species.species.items():
        # Member ids absent from the living set are dead or culled.
        absent = [
            member_id
            for member_id in species.members
            if member_id not in living_ids
        ]
        for member_id in absent:
            species.members.pop(member_id, None)
            if population.species.genome_to_species.get(member_id) == sid:
                population.species.genome_to_species.pop(member_id, None)
        representative = species.representative
        rep_key = getattr(representative, "key", None)
        if species.members and (representative is None or rep_key not in species.members):
            # The representative has to be a genome that is still a member.
            species.representative = next(iter(species.members.values()))
        if not species.members:
            empty.append(sid)
    for sid in empty:
        # An empty species must not keep matching new births.
        population.species.species.pop(sid, None)


def _species_settings(sim_config, population_size):
    """Return the min, target, and max species counts for this population."""
    sim_config = sim_config or {}
    # Configured band. Callers can tighten it without editing code.
    target = int(sim_config.get("target_species_count", DEFAULT_TARGET_SPECIES))
    minimum = int(sim_config.get("min_species_count", DEFAULT_MIN_SPECIES))
    ceiling = int(sim_config.get("max_species_count", DEFAULT_MAX_SPECIES))
    # A species needs several members before two of them can meet and breed.
    # Never ask a small population for more species than that packing allows.
    packed = max(1, int(population_size) // 4)
    ceiling = max(1, min(max(1, ceiling), packed))
    minimum = max(1, min(minimum, ceiling))
    target = max(minimum, min(max(1, target), ceiling))
    return minimum, target, ceiling


def _baseline_threshold(population):
    """Remember the ini threshold so selection can raise it and later relax."""
    stored = getattr(population.species, "_baseline_compatibility", None)
    if stored is None:
        # First retune captures the value loaded from neat-config.ini.
        stored = float(population.species.species_set_config.compatibility_threshold)
        population.species._baseline_compatibility = stored
    return stored


def _pair_distances(genomes, genome_config):
    """Cache genetic distance for every unordered pair of living genomes."""
    keys = sorted(genomes)
    distances = {}
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            distance = genomes[left].distance(genomes[right], genome_config)
            distances[(left, right)] = distance
            distances[(right, left)] = distance
    return keys, distances


def _cluster_count(keys, distances, threshold):
    """Count species the way NEAT does when no old representatives exist."""
    representatives = []
    for genome_id in keys:
        nearest = None
        nearest_distance = None
        for representative_id in representatives:
            distance = distances[(genome_id, representative_id)]
            # NEAT joins when distance is strictly inside the threshold.
            if distance < threshold and (
                nearest_distance is None or distance < nearest_distance
            ):
                nearest = representative_id
                nearest_distance = distance
        if nearest is None:
            representatives.append(genome_id)
    return len(representatives)


def tune_compatibility_threshold(population, neat_config, genomes, sim_config=None):
    """Keep the living species count inside the single-digit band."""
    current = float(population.species.species_set_config.compatibility_threshold)
    if len(genomes) <= 1:
        return current
    minimum, target, ceiling = _species_settings(sim_config, len(genomes))
    # The ini value is the floor. Dropping below it is what produced singletons.
    baseline = _baseline_threshold(population)
    current = max(current, baseline)
    keys, distances = _pair_distances(genomes, neat_config.genome_config)
    current_count = _cluster_count(keys, distances, current)
    # Already a handful of breeding groups: leave the threshold alone.
    if minimum <= current_count <= ceiling:
        _apply_threshold(population, neat_config, current)
        return current
    best_threshold = current
    best_rank = None
    # Sample from the floor upward. Include the live threshold so a value
    # between samples can still win when it is already in use.
    candidates = [
        baseline + _THRESHOLD_SEARCH_SPAN * step / 8.0 for step in range(9)
    ]
    candidates.append(current)
    for threshold in sorted({round(value, 4) for value in candidates}):
        if threshold + 1e-9 < baseline:
            continue
        count = _cluster_count(keys, distances, threshold)
        over = max(0, count - ceiling)
        gap = abs(count - target)
        # In-ceiling counts win, then closeness to the target, then the
        # smallest move from the threshold already in use.
        rank = (over, gap, abs(threshold - current), threshold)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_threshold = threshold
    _apply_threshold(population, neat_config, best_threshold)
    return best_threshold


def _apply_threshold(population, neat_config, threshold):
    """Write one compatibility threshold onto the objects NEAT actually reads."""
    population.species.species_set_config.compatibility_threshold = threshold
    neat_config.species_set_config.compatibility_threshold = threshold


def refresh_living_species(population, neat_config, organisms, sim_config=None):
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
    # The id currently drawn on each body. Re-clustering must put it back
    # on the same lineage, or every selection pass recolors the survivors.
    previous_ids = {}
    for organism in organisms:
        if organism.energy <= 0 or organism.genome is None:
            continue
        if organism.species_id is not None:
            previous_ids[organism.genome_id] = organism.species_id
    # Widen or relax the threshold before clustering so the result can breed.
    tune_compatibility_threshold(population, neat_config, living, sim_config)
    # Drop stale species first. More old species than living genomes makes
    # NEAT's speciate step fail when it looks for a new representative.
    population.species.species = {}
    population.species.genome_to_species = {}
    population.species.speciate(neat_config, living, population.generation)
    # Speciate mints new ids. Map each cluster back onto the id its members
    # already had so an epoch does not look like a new generation.
    _restore_species_ids(population, previous_ids)
    for organism in organisms:
        sid = population.species.genome_to_species.get(organism.genome_id)
        if sid is not None:
            organism.species_id = sid
    return len(population.species.species)


def _restore_species_ids(population, previous_ids):
    """Keep each surviving lineage on the species id it already displays."""
    if not previous_ids or not population.species.species:
        return
    # Vote: the id most members already have is the one the cluster should keep.
    ranked = []
    for sid, species in population.species.species.items():
        votes = {}
        for genome_id in species.members:
            old_id = previous_ids.get(genome_id)
            if old_id is None:
                continue
            votes[old_id] = votes.get(old_id, 0) + 1
        best_id = None
        best_votes = 0
        for old_id, count in votes.items():
            # Ties break toward the lower id so the choice is stable.
            better = count > best_votes
            tied = count == best_votes and (best_id is None or old_id < best_id)
            if better or tied:
                best_id = old_id
                best_votes = count
        ranked.append((best_votes, sid, best_id))
    # Largest claim wins a shared historical id. The other cluster keeps
    # the fresh id speciate just issued, which has never been displayed.
    ranked.sort(key=lambda item: (-item[0], item[1]))
    used = set()
    assignment = {}
    for _votes, sid, best_id in ranked:
        if best_id is not None and best_id not in used:
            chosen = best_id
        elif sid not in used:
            chosen = sid
        else:
            chosen = next(population.species.indexer)
            while chosen in used:
                chosen = next(population.species.indexer)
        used.add(chosen)
        assignment[sid] = chosen
    if all(new_id == old_id for old_id, new_id in assignment.items()):
        return
    rebuilt = {}
    rebuilt_map = {}
    for sid, species in population.species.species.items():
        stable = assignment.get(sid, sid)
        # Species.key and the dict key are the id organisms and NEAT both read.
        species.key = stable
        rebuilt[stable] = species
        for genome_id in species.members:
            rebuilt_map[genome_id] = stable
    population.species.species = rebuilt
    population.species.genome_to_species = rebuilt_map
