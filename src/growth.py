"""Maturity. Hatchlings are a fraction of the adult body and grow by spending energy."""


def bind_adult_body(organism):
    """Remember the full-grown size, then scale the live body by maturity."""
    # Called at the end of attribute setup, after genes have scaled size.
    organism.adult_size = organism.size
    organism._adult_max_energy = organism.max_energy
    organism._adult_max_health = 90.0 * max(organism.adult_size, 0.25)
    if not hasattr(organism, "maturity"):
        # Episodic organisms and founders are born ready to breed.
        organism.maturity = 1.0
    previous_health = getattr(organism, "health", None)
    apply_maturity(organism)
    if previous_health is None:
        organism.health = organism.max_health


def apply_maturity(organism):
    """Scale size, stomach, health cap, and radius to the current maturity."""
    maturity = min(1.0, max(0.2, float(getattr(organism, "maturity", 1.0))))
    organism.maturity = maturity
    adult = getattr(organism, "adult_size", organism.size)
    organism.size = adult * maturity
    adult_energy = getattr(organism, "_adult_max_energy", organism.max_energy)
    organism.max_energy = adult_energy * maturity
    adult_health = getattr(organism, "_adult_max_health", 90.0)
    organism.max_health = adult_health * maturity
    # Init calls this before the starting energy has been assigned.
    if hasattr(organism, "energy"):
        organism.energy = min(organism.energy, organism.max_energy)
    if getattr(organism, "health", None) is not None:
        organism.health = min(organism.health, organism.max_health)
    scale = getattr(organism, "base_radius_scale", 1.0)
    node_count = getattr(organism, "node_count", 1)
    organism.base_radius = (4 + min(8, node_count / 10.0)) * scale * maturity


def grow_step(organism):
    """Spend a little energy to approach adult size while the stomach is comfortable."""
    if organism.energy <= 0 or getattr(organism, "maturity", 1.0) >= 1.0:
        return
    ratio = organism.energy / max(organism.max_energy, 1.0)
    # Hungry juveniles wait. Growth is paid from surplus.
    if ratio < 0.55:
        return
    config = organism.environment_config
    cost = float(config.get("growth_energy_cost", 0.15))
    step = float(config.get("growth_step", 0.004))
    if organism.energy <= cost + 20:
        return
    organism.energy -= cost
    organism.maturity = min(1.0, organism.maturity + step)
    apply_maturity(organism)
