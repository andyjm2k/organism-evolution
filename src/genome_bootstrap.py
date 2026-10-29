"""Bootstrap living-world genomes with richer topology and controlled speciation."""

import random

from logging_util import log_always

# Sensor channels (network_inputs.py) as NEAT keys: key = -(index) - 1.
# Food dx/dy are channels 11 and 12. Prey dx/dy are 18 and 19.
# Partner dx/dy are channels 25 and 26.
_FOOD_DX_INPUT = -12
_FOOD_DY_INPUT = -13
_PREY_DX_INPUT = -19
_PREY_DY_INPUT = -20
_PARTNER_DX_INPUT = -26
_PARTNER_DY_INPUT = -27

# Motor outputs. 0 and 1 are signed velocity, not a heading angle.
_VX_OUTPUT = 0
_VY_OUTPUT = 1
_BREED_OUTPUT = 2
_REST_OUTPUT = 3

# Shared day-0 genome. Every founder starts here; births inherit and mutate it.
# Food and prey outweigh a mate, so a visible meal pulls harder than courtship.
_FOUNDING_WIRES = (
    (_FOOD_DX_INPUT, _VX_OUTPUT, 1.8),
    (_FOOD_DY_INPUT, _VY_OUTPUT, 1.8),
    (_PREY_DX_INPUT, _VX_OUTPUT, 1.8),
    (_PREY_DY_INPUT, _VY_OUTPUT, 1.8),
    (_PARTNER_DX_INPUT, _VX_OUTPUT, 0.65),
    (_PARTNER_DY_INPUT, _VY_OUTPUT, 0.65),
)
_FOUNDING_BIASES = {
    _VX_OUTPUT: 0.2,
    _VY_OUTPUT: 0.05,
    _BREED_OUTPUT: 1.0,
    _REST_OUTPUT: -1.0,
}


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


def install_founding_motif(genome, config):
    """Write the shared starting wiring onto one genome.

    This is initial genome content, not a runtime reflex. After this, movement
    and breeding desire are whatever the network outputs.
    """
    for node_id, bias in _FOUNDING_BIASES.items():
        node = genome.nodes.get(node_id)
        if node is not None:
            node.bias = bias
    if config.innovation_tracker is None:
        return
    for src, dst, weight in _FOUNDING_WIRES:
        key = (src, dst)
        existing = genome.connections.get(key)
        if existing is None:
            innovation = config.innovation_tracker.get_innovation_number(
                src, dst, "founding_motif"
            )
            existing = genome.create_connection(config, src, dst, innovation)
            genome.connections[key] = existing
        existing.weight = weight
        existing.enabled = True


def _bias_foraging_weights(genome, config, strength):
    """Add an optional extra nudge on the founding sensory→velocity wires."""
    if strength <= 0:
        return
    for src, dst, _base in _FOUNDING_WIRES:
        key = (src, dst)
        connection = genome.connections.get(key)
        if connection is None or config.innovation_tracker is None:
            continue
        connection.weight += random.uniform(strength * 0.5, strength * 1.5)
        connection.enabled = True


def _create_archetypes(genome_type, config, archetype_count, extra_hidden, extra_connections, foraging_bias):
    """Build diverse seed archetypes that will form distinct species clusters."""
    archetypes = []
    for index in range(archetype_count):
        archetype_id = 100_000 + index
        genome = genome_type(archetype_id)
        genome.configure_new(config)
        _enhance_archetype(genome, config, extra_hidden, extra_connections)
        genome.fitness = 0.0
        genome.diet_key = 1
        install_founding_motif(genome, config)
        _bias_foraging_weights(genome, config, foraging_bias)
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
    extra_hidden = int(sim_config.get("genesis_extra_hidden_nodes", 0))
    extra_connections = int(sim_config.get("genesis_extra_connections", 0))
    weight_jitter = float(sim_config.get("genesis_weight_jitter", 0.05))
    foraging_bias = float(sim_config.get("genesis_foraging_bias", 0.0))

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
        # Founders eat the plant food that actually exists. Species ids are an
        # accident of speciation and must not turn the whole cohort into hunters.
        genome.diet_key = getattr(template, "diet_key", 1)
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
    extra_hidden = int(sim_config.get("genesis_extra_hidden_nodes", 0))
    extra_connections = int(sim_config.get("genesis_extra_connections", 0))
    foraging_bias = float(sim_config.get("genesis_foraging_bias", 0.0))
    genome_config = neat_config.genome_config
    genome = neat_config.genome_type(0)
    genome.configure_new(genome_config)
    _enhance_archetype(genome, genome_config, extra_hidden, extra_connections)
    genome.fitness = 0.0
    genome.diet_key = 1
    install_founding_motif(genome, genome_config)
    _bias_foraging_weights(genome, genome_config, foraging_bias)
    _ensure_connections_enabled(genome)
    return genome
