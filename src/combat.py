"""Health and bites. Energy stays fuel; health is the life a bite can end."""

from body_traits import BodyTraits
from distance import colliding
from logging_util import log_detailed


def bite_damage(attacker, defender):
    """Damage from claw or jaw strength and the attacker's size advantage."""
    traits = BodyTraits.from_organism(attacker)
    # A plain cell still pinches. A real claw hits harder.
    weapon = max(traits.claw, traits.jaw * 0.5, 0.8)
    scale = attacker.size / max(getattr(defender, "size", 1.0), 0.15)
    return 18.0 * weapon * scale


def apply_bite(attacker, defender):
    """Deal one bite. A killing bite leaves the defender with no energy.

    Returns True when the defender dies from this bite.
    """
    if getattr(defender, "health", None) is None:
        defender.max_health = 90.0 * max(getattr(defender, "size", 1.0), 0.25)
        defender.health = defender.max_health
    damage = bite_damage(attacker, defender)
    defender.health -= damage
    if defender.health > 0:
        return False
    defender.health = 0
    # The rest of the sim treats energy at 0 as dead.
    defender.energy = 0
    return True


def hunt_prey(organism, organisms):
    """A carnivore bites one smaller grazer it is touching."""
    if organism.energy <= 0 or not organism.is_carnivore:
        return
    for other in organisms:
        if (
            other is organism
            or other.is_carnivore
            or other.energy <= 0
            or other.position is None
            or organism.position is None
        ):
            continue
        if not colliding(
            organism.position,
            organism.get_radius(),
            other.position,
            other.get_radius(),
        ):
            continue
        # Same size gate as before: hunters take prey up to 80% of their size.
        if organism.size < other.size * 0.8:
            continue
        stored_energy = other.energy
        killed = apply_bite(organism, other)
        if not killed:
            log_detailed(
                organism.logging_level,
                f"Carnivore bit prey; health={other.health:.1f}",
            )
            break
        # The bite already zeroed the prey. Pay from the energy it was carrying.
        gain = stored_energy * 0.7
        organism.energy = min(organism.max_energy, organism.energy + gain)
        organism.organisms_consumed += 1
        if organism.steps_since_last_hunt > 0:
            organism.avg_steps_between_hunts.append(organism.steps_since_last_hunt)
            if len(organism.avg_steps_between_hunts) > 10:
                organism.avg_steps_between_hunts.pop(0)
        organism.steps_since_last_hunt = 0
        organism.fitness_bonus += 25
        log_detailed(
            organism.logging_level,
            f"Carnivore killed prey; energy={organism.energy:.1f}",
        )
        break
