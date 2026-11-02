"""Plant strains. Pellets are bodies of a strain that drifts between roles.

Relation above 0.35 is symbiotic: grazing spreads the plant.
Between -0.35 and 0.35 the plant is an ordinary meal.
Below -0.35 it is predatory: the bite costs the grazer health.
"""

import random

from distance import colliding

# Bands for the continuous relation gene.
SYMBIOTIC_AT = 0.35
PREDATORY_AT = -0.35
# Success points required before a strain drops a mutated seed.
SEED_EVERY = 8.0
# Much smaller than animal body-gene mutation, so plants drift slowly.
MUTATION_VARIANCE = 0.04


def _clamp(value, low, high):
    """Keep a plant gene inside its range."""
    return max(low, min(high, value))


class PlantStrain:
    """One heritable plant strategy shared by many pellets."""

    def __init__(self, relation=0.0, nutrition=1.0, strike=0.4, regrowth=1.0):
        # -1 predatory, 0 indifferent, 1 symbiotic.
        self.relation = float(relation)
        # Multiplier on the calories a grazer receives.
        self.nutrition = float(nutrition)
        # How hard a predatory bite hits.
        self.strike = float(strike)
        # How quickly a grazed symbiotic pellet returns.
        self.regrowth = float(regrowth)
        # Ecological score. Seeds spend it.
        self.success = 0.0

    def band(self):
        """Name the role this relation currently expresses."""
        if self.relation >= SYMBIOTIC_AT:
            return "symbiotic"
        if self.relation <= PREDATORY_AT:
            return "predatory"
        return "indifferent"

    def pigment(self):
        """Green when helpful, yellow-green when ordinary, red when it hunts."""
        if self.relation >= 0:
            blend = min(1.0, self.relation / max(SYMBIOTIC_AT, 0.01))
            red = int(90 * (1.0 - blend) + 40 * blend)
            green = int(190 + 40 * blend)
            blue = int(80 * (1.0 - blend) + 50 * blend)
        else:
            blend = min(1.0, -self.relation / max(-PREDATORY_AT, 0.01))
            red = int(90 + 140 * blend)
            green = int(190 * (1.0 - blend) + 50 * blend)
            blue = int(70 * (1.0 - blend) + 40 * blend)
        return (max(0, min(255, red)), max(0, min(255, green)), max(0, min(255, blue)))

    def mutate(self, rng=None, variance=MUTATION_VARIANCE):
        """Return a child strain nudged a little from this one."""
        picker = rng if rng is not None else random
        spread = float(variance)
        child = PlantStrain(
            relation=_clamp(self.relation + picker.gauss(0.0, spread), -1.0, 1.0),
            nutrition=_clamp(self.nutrition + picker.gauss(0.0, spread), 0.4, 1.6),
            strike=_clamp(self.strike + picker.gauss(0.0, spread), 0.1, 2.5),
            regrowth=_clamp(self.regrowth + picker.gauss(0.0, spread * 0.5), 0.4, 2.0),
        )
        return child

    def add_success(self, amount):
        """Record an ecological win. Returns True when a seed is due."""
        self.success += float(amount)
        if self.success < SEED_EVERY:
            return False
        self.success = 0.0
        return True


def make_founder_strain(bias=0.0, rng=None):
    """A starting strain near indifferent, so the first hours stay edible."""
    picker = rng if rng is not None else random
    relation = _clamp(bias + picker.gauss(0.0, 0.05), -0.3, 0.3)
    return PlantStrain(relation=relation, nutrition=1.0, strike=0.4, regrowth=1.0)


def pellet_color(food):
    """Draw color for a pellet, or the original green when it has no strain."""
    strain = getattr(food, "strain", None)
    if strain is None:
        return (80, 220, 90)
    return strain.pigment()


def graze(organism, food_items):
    """One herbivore bite of the first pellet it is touching.

    No strain means the old indifferent pellet: full meal, pellet removed.
    """
    if organism.energy <= 0 or organism.is_carnivore:
        return
    food_energy = float(organism.environment_config.get("food_energy_value", 75))
    food_radius = 5
    for food in food_items:
        if food.position is None or getattr(food, "regrow_in", 0) > 0:
            continue
        if not colliding(
            organism.position, organism.get_radius(), food.position, food_radius
        ):
            continue
        strain = getattr(food, "strain", None)
        band = strain.band() if strain is not None else "indifferent"
        if band == "predatory":
            _predatory_bite(organism, food, strain, food_energy)
        elif band == "symbiotic":
            _symbiotic_bite(organism, food, strain, food_energy)
        else:
            _indifferent_bite(organism, food, strain, food_energy)
        break


def _credit_meal(organism):
    """Count one plant meal toward breeding and the food timer."""
    organism.food_consumed += 1
    if organism.steps_since_last_food > 0:
        organism.avg_steps_between_food.append(organism.steps_since_last_food)
        if len(organism.avg_steps_between_food) > 10:
            organism.avg_steps_between_food.pop(0)
    organism.steps_since_last_food = 0
    organism.fitness_bonus += 10


def _pay(organism, amount):
    """Add calories without passing the stomach cap."""
    organism.energy = min(organism.max_energy, organism.energy + amount)


def _indifferent_bite(organism, food, strain, food_energy):
    """Ordinary pellet. Full meal, then the pellet is gone."""
    nutrition = 1.0 if strain is None else strain.nutrition
    _pay(organism, food_energy * organism.energy_efficiency * nutrition)
    _credit_meal(organism)
    food.position = None
    _note(food, 0.15)


def _symbiotic_bite(organism, food, strain, food_energy):
    """Grazing spreads this plant. The pellet regrows in place."""
    _pay(organism, food_energy * organism.energy_efficiency * strain.nutrition)
    _credit_meal(organism)
    # Leave the body in the world. It cannot be bitten again until it regrows.
    food.regrow_in = int(36 / max(strain.regrowth, 0.2))
    _note(food, 1.0)


def _predatory_bite(organism, food, strain, food_energy):
    """The plant bites back. A survivor can still uproot it for a small meal."""
    if getattr(organism, "health", None) is None:
        organism.max_health = 90.0
        organism.health = organism.max_health
    organism.health -= 10.0 * max(strain.strike, 0.3)
    _pay(organism, food_energy * 0.25 * strain.nutrition)
    _credit_meal(organism)
    food.position = None
    if organism.health <= 0:
        organism.health = 0
        organism.energy = 0
        _note(food, 2.0)
    else:
        _note(food, 0.5)


def _note(food, amount):
    """Tell the ecology this strain earned a step toward a seed."""
    ecology = getattr(food, "ecology", None)
    if ecology is not None and hasattr(ecology, "note_plant_event"):
        ecology.note_plant_event(food, amount)
