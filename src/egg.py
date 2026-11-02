"""Eggs. A living-world birth is fertilized, filled, laid, then hatched."""

import random

from body_genes import inherit_body_genes
from growth import apply_maturity
from logging_util import log_detailed


class Egg:
    """One fertilized offspring, carried and then waiting in the world."""

    def __init__(self, parent, partner, child, child_genome, energy_goal, hatch_delay):
        # Parents are kept so the hatch can credit them and record lineage.
        self.parent = parent
        self.partner = partner
        self.child = child
        self.child_genome = child_genome
        self.energy_goal = float(energy_goal)
        self.hatch_delay = int(hatch_delay)
        self.stored = 0.0
        self.position = None
        self.laid = False
        self.steps_laid = 0

    def gestate(self, parent, desire):
        """Move energy from the carrier into the egg while they are willing.

        Returns True when the egg is full and ready to lay.
        """
        if self.laid or self.stored >= self.energy_goal:
            return True
        # Breed output has to stay high. A parent that stops paying pauses the egg.
        if desire <= 0.5:
            return False
        reserve = 40.0
        take = min(2.0, parent.energy - reserve, self.energy_goal - self.stored)
        if take <= 0:
            return False
        parent.energy -= take
        self.stored += take
        return self.stored >= self.energy_goal

    def lay(self, position):
        """Put the full egg down at a world position."""
        self.laid = True
        self.position = position
        self.steps_laid = 0

    def tick(self):
        """Age a laid egg. Returns True when it should hatch."""
        if not self.laid:
            return False
        self.steps_laid += 1
        return self.steps_laid >= self.hatch_delay


def uses_eggs(organism):
    """Living-world organisms with the flag gestate. Episodic births stay instant."""
    if organism.simulation is None:
        return False
    if not hasattr(organism.simulation, "register_birth"):
        return False
    return bool(organism.environment_config.get("eggs_enabled", False))


def build_offspring(parent, partner):
    """Create the child genome and body the same way an instant birth did."""
    spawn_x = (parent.position[0] + partner.position[0]) / 2 + random.uniform(-5, 5)
    spawn_y = (parent.position[1] + partner.position[1]) / 2 + random.uniform(-5, 5)
    width = parent.environment_config["width"]
    height = parent.environment_config["height"]
    spawn_x = max(0, min(spawn_x, width))
    spawn_y = max(0, min(spawn_y, height))
    parent.genome.fitness = parent.calculate_fitness()
    partner.genome.fitness = partner.calculate_fitness()
    if parent.genome.fitness >= partner.genome.fitness:
        parent1, parent2 = parent.genome, partner.genome
    else:
        parent1, parent2 = partner.genome, parent.genome
    child_genome = type(parent.genome)(0)
    child_genome.configure_crossover(parent1, parent2, parent.config.genome_config)
    child_genome.mutate(parent.config.genome_config)
    # Body genes come from the fitter parent and then mutate.
    inherit_body_genes(child_genome, parent1, parent2)
    # Local import: organism imports this module to fertilize eggs.
    from organism import Organism

    child = Organism(
        child_genome,
        parent.config,
        (spawn_x, spawn_y),
        parent.environment_config,
        species_id=parent.species_id,
        diet_key=getattr(parent, "_diet_key", parent.species_id),
        logging_level=parent.logging_level,
    )
    offspring_energy = float(
        parent.environment_config.get("offspring_starting_energy", 50)
    )
    child.energy = min(offspring_energy, child.max_energy)
    child.steps_since_breeding = 0
    log_detailed(parent.logging_level, "Offspring genome built")
    return child, child_genome


def fertilize(parent, partner):
    """Start an egg on the parent. The child exists but is not in the world yet."""
    child, child_genome = build_offspring(parent, partner)
    config = parent.environment_config
    maturity = float(config.get("hatch_maturity", 0.35))
    child.maturity = maturity
    apply_maturity(child)
    child.energy = min(child.energy, child.max_energy)
    goal = float(config.get("egg_energy_goal", 80))
    delay = int(config.get("egg_hatch_delay", 150))
    egg = Egg(parent, partner, child, child_genome, goal, delay)
    parent.carried_egg = egg
    # Both parents wait out the cooldown so one meeting is one egg.
    parent.steps_since_breeding = 0
    partner.steps_since_breeding = 0
    parent.energy = max(0, parent.energy - 10)
    partner.energy = max(0, partner.energy - 10)
    return egg


def tend_egg(parent, desire):
    """Fill a carried egg and hand it to the world when it is full."""
    egg = getattr(parent, "carried_egg", None)
    if egg is None or parent.position is None:
        return
    if not egg.gestate(parent, desire):
        return
    egg.lay(parent.position)
    parent.carried_egg = None
    simulation = parent.simulation
    if simulation is not None and hasattr(simulation, "receive_egg"):
        simulation.receive_egg(egg)
