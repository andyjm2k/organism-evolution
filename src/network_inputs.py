"""Build the fixed-length neural network input vector for an organism."""

import math

from distance import squared_distance

# Total input count must match config/neat-config.ini num_inputs.
NUM_NETWORK_INPUTS = 31

# Empty second-nearest sentinel: max distance, zero direction.
_EMPTY_SECOND = (1.0, 0.0, 0.0)


def _normalize_vector(dx, dy, scale):
    """Scale a direction vector by detection radius with divide-by-zero guard."""
    if scale <= 0:
        return 0.0, 0.0
    return dx / scale, dy / scale


def _entities_in_radius(organism, entities, radius):
    """Return in-radius entities sorted by squared distance (nearest first)."""
    radius_sq = radius * radius if radius > 0 else 0.0
    ranked = []
    for entity in entities:
        if entity.position is None or organism.position is None:
            continue
        dist_sq = squared_distance(organism.position, entity.position)
        if dist_sq > radius_sq:
            continue
        ranked.append((dist_sq, entity))
    ranked.sort(key=lambda item: item[0])
    return ranked


def _pack_entity_channels(organism, entity, dist_sq, radius):
    """Pack dist_norm, dx_norm, dy_norm for one sensed entity."""
    best_dist = math.sqrt(dist_sq)
    dist_norm = min(1.0, best_dist / max(radius, 1.0))
    dx = entity.position[0] - organism.position[0]
    dy = entity.position[1] - organism.position[1]
    dx_norm, dy_norm = _normalize_vector(dx, dy, radius)
    return dist_norm, dx_norm, dy_norm


def _closest_entities(organism, entities, radius):
    """
    Return nearest (count, dist, dx, dy) and second-nearest (dist, dx, dy).

    Nearest block uses local density; second block omits count.
    """
    ranked = _entities_in_radius(organism, entities, radius)
    if not ranked:
        return (0.0, 1.0, 0.0, 0.0), _EMPTY_SECOND
    count_norm = min(1.0, len(ranked) / 10.0)
    nearest_dist, nearest_entity = ranked[0]
    nearest = (
        count_norm,
        *_pack_entity_channels(organism, nearest_entity, nearest_dist, radius),
    )
    if len(ranked) < 2:
        return nearest, _EMPTY_SECOND
    second_dist, second_entity = ranked[1]
    second = _pack_entity_channels(organism, second_entity, second_dist, radius)
    return nearest, second


def _closest_entity(organism, entities, radius):
    """Return (count_norm, dist_norm, dx_norm, dy_norm) for nearest entity."""
    nearest, _second = _closest_entities(organism, entities, radius)
    return nearest


def build_network_inputs(
    organism,
    nearby_food,
    nearby_organisms,
    nearby_threats,
    nearby_breeding_partners,
    environment_config,
):
    """
    Build a length-31 input vector:

    0-8    core body/environment
    9-12   nearest food (herbivores; zeros for carnivores)
    13-15  second-nearest food
    16-19  nearest prey (carnivores) or threats (herbivores)
    20-22  second-nearest prey/threat
    23-27  nearest breeding partner + readiness
    28-30  second-nearest breeding partner
    """
    width = environment_config.get("width", 800)
    height = environment_config.get("height", 600)
    x, y = organism.position
    energy_frac = organism.energy / max(organism.max_energy, 1.0)
    center_x, center_y = width / 2.0, height / 2.0
    dist_center = math.hypot(x - center_x, y - center_y)
    max_dist = math.hypot(center_x, center_y) or 1.0
    center_norm = dist_center / max_dist
    rel_x = (x / width) * 2 - 1 if width else 0.0
    rel_y = (y / height) * 2 - 1 if height else 0.0
    if organism.last_position != organism.position and organism.speed > 0:
        mdx = (organism.position[0] - organism.last_position[0]) / organism.speed
        mdy = (organism.position[1] - organism.last_position[1]) / organism.speed
        mag = math.hypot(mdx, mdy)
        if mag > 0:
            mdx /= mag
            mdy /= mag
    else:
        mdx, mdy = 0.0, 0.0
    size_norm = min(1.0, organism.size / 4.0)
    nearest_h = min(x / width, (width - x) / width) if width else 0.0
    nearest_v = min(y / height, (height - y) / height) if height else 0.0
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
    food_radius = environment_config.get(
        "food_detection_radius", environment_config.get("detection_radius", 200)
    )
    threat_radius = environment_config.get(
        "threat_detection_radius", environment_config.get("detection_radius", 200)
    )
    detection_radius = environment_config.get("detection_radius", 200)
    breeding_radius = environment_config.get(
        "breeding_detection_radius", environment_config.get("detection_radius", 200)
    )
    if organism.is_carnivore:
        inputs.extend([0.0, 1.0, 0.0, 0.0])
        inputs.extend(list(_EMPTY_SECOND))
    else:
        food_near, food_second = _closest_entities(organism, nearby_food, food_radius)
        inputs.extend(list(food_near))
        inputs.extend(list(food_second))
    if organism.is_carnivore:
        prey = [o for o in nearby_organisms if not o.is_carnivore]
        prey_near, prey_second = _closest_entities(organism, prey, detection_radius)
        inputs.extend(list(prey_near))
        inputs.extend(list(prey_second))
    else:
        threats = nearby_threats if nearby_threats is not None else []
        threat_near, threat_second = _closest_entities(
            organism, threats, threat_radius
        )
        inputs.extend(list(threat_near))
        inputs.extend(list(threat_second))
    partners = nearby_breeding_partners if nearby_breeding_partners is not None else []
    partner_near, partner_second = _closest_entities(
        organism, partners, breeding_radius
    )
    ready = 1.0 if organism.can_breed() else 0.0
    inputs.extend(list(partner_near))
    inputs.extend([ready])
    inputs.extend(list(partner_second))
    if len(inputs) != NUM_NETWORK_INPUTS:
        raise ValueError(
            f"Expected {NUM_NETWORK_INPUTS} inputs, built {len(inputs)}"
        )
    return inputs
