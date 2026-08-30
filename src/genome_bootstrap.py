"""Bootstrap living-world genomes with richer topology and controlled speciation."""

import random

from logging_util import log_always

# Network input indices (see network_inputs.py) mapped to NEAT negative node keys.
_FOOD_DX_INPUT = -12
_FOOD_DY_INPUT = -13
_PREY_DX_INPUT = -18
_PREY_DY_INPUT = -19
_ENERGY_INPUT = -1

# Primary motor outputs: heading angle and speed fraction.
_ANGLE_OUTPUT = 0
_SPEED_OUTPUT = 1
_REST_OUTPUT = 3


def _clone_genome(genome_type, genome_id, template, config):
    """Return a fresh genome that copies template topology and weights."""
    clone = genome_type(genome_id)
    clone.fitness = 0.0
    template.fitness = 0.0
    clone.configure_crossover(template, template, config)
    return clone


def _enhance_archetype(genome, config, extra_hidden, extra_connections):
    """Grow an archetype with extra hidden nodes and feed-forward connections."""
    for _ in range(max(0, extra_hidden)):
        genome.mutate_add_node(config)
    for _ in range(max(0, extra_connections)):
        genome.mutate_add_connection(config)
    _ensure_connections_enabled(genome)


def _ensure_connections_enabled(genome):
    """Activate every connection gene so the phenotype uses the full graph."""
    for connection in genome.connections.values():
        connection.enabled = True


def _jitter_weights(genome, stdev):
    """Apply small Gaussian noise to enabled connection weights."""
    if stdev <= 0:
        return
    for connection in genome.connections.values():
        if connection.enabled:
            connection.weight += random.gauss(0.0, stdev)


def _bias_foraging_weights(genome, config, strength):
    """Nudge sensory→motor weights so naive networks can seek resources."""
    if strength <= 0:
        return
    pairs = [
        (_FOOD_DX_INPUT, _ANGLE_OUTPUT),
        (_FOOD_DY_INPUT, _ANGLE_OUTPUT),
        (_FOOD_DX_INPUT, _SPEED_OUTPUT),
        (_FOOD_DY_INPUT, _SPEED_OUTPUT),
        (_PREY_DX_INPUT, _ANGLE_OUTPUT),
        (_PREY_DY_INPUT, _ANGLE_OUTPUT),
        (_PREY_DX_INPUT, _SPEED_OUTPUT),
        (_PREY_DY_INPUT, _SPEED_OUTPUT),
    ]
    for in_node, out_node in pairs:
        key = (in_node, out_node)
        if key in genome.connections:
            genome.connections[key].weight += random.uniform(
                strength * 0.5, strength * 1.5
            )
        elif config.innovation_tracker is not None:
            innovation = config.innovation_tracker.get_innovation_number(
                in_node, out_node, "bootstrap_bias"
            )
            connection = genome.create_connection(config, in_node, out_node, innovation)
            connection.weight = random.uniform(strength * 0.5, strength * 1.5)
            connection.enabled = True
            genome.connections[key] = connection
    energy_key = (_ENERGY_INPUT, _REST_OUTPUT)
    if energy_key in genome.connections:
        genome.connections[energy_key].weight -= strength * 0.5


def _create_archetypes(genome_type, config, archetype_count, extra_hidden, extra_connections, foraging_bias):
    """Build diverse seed archetypes that will form distinct species clusters."""
    archetypes = []
    for index in range(archetype_count):
        archetype_id = 100_000 + index
        genome = genome_type(archetype_id)
        genome.configure_new(config)
        _enhance_archetype(genome, config, extra_hidden, extra_connections)
        _bias_foraging_weights(genome, config, foraging_bias)
        genome.fitness = 0.0
        archetypes.append(genome)
    return archetypes


def _reset_species(population):
    """Clear stale species bookkeeping before re-clustering bootstrapped genomes."""
    population.species.species = {}
    population.species.genome_to_species = {}


def bootstrap_population(population, neat_config, sim_config=None):
    """
    Replace the initial NEAT population with enhanced archetype-derived genomes.

    Each archetype receives extra hidden nodes and connections. Population members
    are cloned from archetypes (round-robin) with light weight jitter so several
    organisms share each species and can breed early in the run.
    """
    sim_config = sim_config or {}
    archetype_count = int(sim_config.get("genesis_archetype_count", 8))
    extra_hidden = int(sim_config.get("genesis_extra_hidden_nodes", 6))
    extra_connections = int(sim_config.get("genesis_extra_connections", 20))
    weight_jitter = float(sim_config.get("genesis_weight_jitter", 0.05))
    foraging_bias = float(sim_config.get("genesis_foraging_bias", 0.35))

    genome_config = neat_config.genome_config
    genome_type = neat_config.genome_type
    archetypes = _create_archetypes(
        genome_type,
        genome_config,
        archetype_count,
        extra_hidden,
        extra_connections,
        foraging_bias,
    )

    bootstrapped = {}
    for genome_id in population.population:
        template = archetypes[genome_id % archetype_count]
        genome = _clone_genome(genome_type, genome_id, template, genome_config)
        _jitter_weights(genome, weight_jitter)
        _ensure_connections_enabled(genome)
        bootstrapped[genome_id] = genome

    population.population = bootstrapped
    _reset_species(population)
    population.species.speciate(
        neat_config, population.population, population.generation
    )

    sample = next(iter(bootstrapped.values()))
    species_count = len(population.species.species)
    enabled = sum(1 for c in sample.connections.values() if c.enabled)
    log_always(
        f"Genesis bootstrap: {species_count} species, "
        f"{len(sample.nodes)} nodes, {enabled}/{len(sample.connections)} "
        f"active connections (from {archetype_count} archetypes)"
    )
    return population


def bootstrap_immigrant_genome(neat_config, sim_config=None):
    """Create one enhanced random genome for immigration or extinction reseeding."""
    sim_config = sim_config or {}
    extra_hidden = int(sim_config.get("genesis_extra_hidden_nodes", 6))
    extra_connections = int(sim_config.get("genesis_extra_connections", 20))
    foraging_bias = float(sim_config.get("genesis_foraging_bias", 0.35))
    genome_config = neat_config.genome_config
    genome = neat_config.genome_type(0)
    genome.configure_new(genome_config)
    _enhance_archetype(genome, genome_config, extra_hidden, extra_connections)
    _bias_foraging_weights(genome, genome_config, foraging_bias)
    _ensure_connections_enabled(genome)
    genome.fitness = 0.0
    return genome
