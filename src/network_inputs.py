"""Build the fixed-length neural network input vector for an organism."""

import math

# Total input count must match config/neat-config.ini num_inputs.
NUM_NETWORK_INPUTS = 22


def _normalize_vector(dx, dy, scale):
    """Scale a direction vector by detection radius with divide-by-zero guard."""
    # Zero scale would explode; treat as no useful direction.
    if scale <= 0:
        return 0.0, 0.0
    # Normalize components into roughly [-1, 1] for the network.
    return dx / scale, dy / scale


def _closest_entity(organism, entities, radius):
    """Return (count_norm, dist_norm, dx_norm, dy_norm) for nearest entity."""
    # Track how many entities fall inside the sensing radius.
    count = 0
    # Start with "infinitely far" so any real candidate wins.
    best_dist = float("inf")
    # Direction placeholders until a candidate is found.
    best_dx = 0.0
    best_dy = 0.0
    # Scan all candidates using linear distance comparisons.
    for entity in entities:
        # Skip entities without a usable world position.
        if entity.position is None or organism.position is None:
            continue
        # Linear distance for sensing (not squared).
        dist = math.sqrt(
            (organism.position[0] - entity.position[0]) ** 2
            + (organism.position[1] - entity.position[1]) ** 2
        )
        # Ignore anything outside the configured sensing radius.
        if dist > radius:
            continue
        # Count every in-range entity toward local density.
        count += 1
        # Keep the nearest entity for directional cues.
        if dist < best_dist:
            best_dist = dist
            best_dx = entity.position[0] - organism.position[0]
            best_dy = entity.position[1] - organism.position[1]
    # No entity found → max distance, zero direction, zero density.
    if best_dist == float("inf"):
        return 0.0, 1.0, 0.0, 0.0
    # Normalize density against a soft cap of ten neighbors.
    count_norm = min(1.0, count / 10.0)
    # Clamp distance into [0, 1] using the sensing radius.
    dist_norm = min(1.0, best_dist / max(radius, 1.0))
    # Direction components scaled by the same radius.
    dx_norm, dy_norm = _normalize_vector(best_dx, best_dy, radius)
    # Pack the four channels expected by the network schema.
    return count_norm, dist_norm, dx_norm, dy_norm


def build_network_inputs(
    organism,
    nearby_food,
    nearby_organisms,
    nearby_threats,
    nearby_breeding_partners,
    environment_config,
):
    """
    Build a length-22 input vector:

    0-8   core body/environment
    9-12  food (herbivores; zeros for carnivores)
    13-16 prey (carnivores) or threats (herbivores)
    17-21 breeding
    """
    # Environment extents with safe fallbacks.
    width = environment_config.get("width", 800)
    height = environment_config.get("height", 600)
    # Current position used for all relative sensors.
    x, y = organism.position
    # Energy as fraction of capacity for scale-free sensing.
    energy_frac = organism.energy / max(organism.max_energy, 1.0)
    # Distance from arena center, normalized by max corner distance.
    center_x, center_y = width / 2.0, height / 2.0
    dist_center = math.hypot(x - center_x, y - center_y)
    max_dist = math.hypot(center_x, center_y) or 1.0
    center_norm = dist_center / max_dist
    # Relative position mapped into [-1, 1].
    rel_x = (x / width) * 2 - 1 if width else 0.0
    rel_y = (y / height) * 2 - 1 if height else 0.0
    # Movement since last step, normalized by speed.
    if organism.last_position != organism.position and organism.speed > 0:
        mdx = (organism.position[0] - organism.last_position[0]) / organism.speed
        mdy = (organism.position[1] - organism.last_position[1]) / organism.speed
        mag = math.hypot(mdx, mdy)
        if mag > 0:
            mdx /= mag
            mdy /= mag
    else:
        mdx, mdy = 0.0, 0.0
    # Size fed roughly in [0, 1] assuming attribute clamp near 4.
    size_norm = min(1.0, organism.size / 4.0)
    # Closest boundary distances on each axis (0 at edge, 1 at center).
    nearest_h = min(x / width, (width - x) / width) if width else 0.0
    nearest_v = min(y / height, (height - y) / height) if height else 0.0
    # Core block (9 channels).
    inputs = [
        energy_frac,
        center_norm,
        rel_x,
        rel_y,
        mdx,
        mdy,
        size_norm,
        nearest_h,
        nearest_v,
    ]
    # Food sensing radius from environment config.
    food_radius = environment_config.get(
        "food_detection_radius", environment_config.get("detection_radius", 200)
    )
    # Threat / prey / breeding radii from environment config.
    threat_radius = environment_config.get(
        "threat_detection_radius", environment_config.get("detection_radius", 200)
    )
    detection_radius = environment_config.get("detection_radius", 200)
    breeding_radius = environment_config.get(
        "breeding_detection_radius", environment_config.get("detection_radius", 200)
    )
    # Food channels: meaningful for herbivores; carnivores get zeros.
    if organism.is_carnivore:
        inputs.extend([0.0, 1.0, 0.0, 0.0])
    else:
        inputs.extend(list(_closest_entity(organism, nearby_food, food_radius)))
    # Role-specific entity channels: prey for carnivores, threats for herbivores.
    if organism.is_carnivore:
        prey = [o for o in nearby_organisms if not o.is_carnivore]
        inputs.extend(list(_closest_entity(organism, prey, detection_radius)))
    else:
        threats = nearby_threats if nearby_threats is not None else []
        inputs.extend(list(_closest_entity(organism, threats, threat_radius)))
    # Breeding partner channels shared by both diets.
    partners = nearby_breeding_partners if nearby_breeding_partners is not None else []
    partner_count, partner_dist, partner_dx, partner_dy = _closest_entity(
        organism, partners, breeding_radius
    )
    # Ready-to-breed flag must match can_breed energy/cooldown semantics.
    ready = 1.0 if organism.can_breed() else 0.0
    # Breeding block (5 channels).
    inputs.extend([partner_count, partner_dist, partner_dx, partner_dy, ready])
    # Hard assert keeps NEAT and runtime schema aligned during development.
    if len(inputs) != NUM_NETWORK_INPUTS:
        raise ValueError(
            f"Expected {NUM_NETWORK_INPUTS} inputs, built {len(inputs)}"
        )
    # Return the fixed-length vector for network.activate.
    return inputs
