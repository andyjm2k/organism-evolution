"""Fitness scoring for organisms based on survival, role, and exploration."""

import math
import random


def calculate_consumption_efficiency(steps_list, max_optimal_steps):
    """Map average steps between consumptions to an efficiency in (0, 1]."""
    # No samples means the organism never succeeded at this consumption type.
    if not steps_list:
        return 0.0
    # Average interval between successful consumptions.
    avg_steps = sum(steps_list) / len(steps_list)
    # Exponential decay: longer waits → lower efficiency.
    return math.exp(-avg_steps / max_optimal_steps)


def calculate_fitness(organism, environment_config):
    """
    Compute fitness for selection.

    Includes survival, role-specific hunting/foraging, center bias,
    exploration, boundary multiplier, and accumulated fitness_bonus.
    """
    # Base survival reward scaled by energy efficiency.
    survival_score = (
        organism.steps_taken * 0.4 + organism.energy * 0.3
    ) * organism.energy_efficiency
    # Role-specific contribution depends on diet.
    if organism.is_carnivore:
        role_score = _carnivore_role_score(organism)
    else:
        role_score = _herbivore_role_score(organism)
    # Environment size for spatial bonuses.
    width = environment_config.get("width", 800)
    height = environment_config.get("height", 600)
    # Reward staying nearer the arena center.
    center_x, center_y = width / 2.0, height / 2.0
    dist_center = math.hypot(
        organism.position[0] - center_x, organism.position[1] - center_y
    )
    max_dist = math.hypot(width / 2.0, height / 2.0) or 1.0
    center_reward = (1.0 - (dist_center / max_dist)) * 100.0
    # Exploration bonus from unique coarse grid cells visited.
    if hasattr(organism, "last_positions") and organism.last_positions:
        unique_positions = len(set(organism.last_positions))
        exploration_bonus = min(200.0, unique_positions * 2.0)
    else:
        exploration_bonus = 0.0
    # Distance to nearest boundary for breeding-safe-zone multiplier.
    boundary_distance = min(
        organism.position[0],
        organism.position[1],
        width - organism.position[0],
        height - organism.position[1],
    )
    # Safe zone matches can_breed: 10% inset from edges.
    safe_zone_boundary = min(width, height) * 0.1
    if boundary_distance >= safe_zone_boundary:
        boundary_multiplier = 1.2
    else:
        boundary_multiplier = 0.5 + (boundary_distance / safe_zone_boundary) * 0.5
    # Weighted combination of the spatial and survival terms.
    final_fitness = (
        survival_score * 0.3
        + role_score * 0.4
        + center_reward * 0.15
        + exploration_bonus * 0.15
    ) * boundary_multiplier
    # Explicit bonuses from breeding/foraging events must affect selection.
    final_fitness += getattr(organism, "fitness_bonus", 0.0)
    # Tiny noise breaks exact ties without dominating the signal.
    final_fitness += random.uniform(-0.05, 0.05)
    # Track lifetime peak for scoreboard reporting.
    if final_fitness > organism.highest_fitness:
        organism.highest_fitness = final_fitness
    # Return the scalar used as genome.fitness.
    return final_fitness


def _carnivore_role_score(organism):
    """Score hunting success with strategy and starvation penalties."""
    # Efficiency of time between successful hunts.
    hunting_efficiency = calculate_consumption_efficiency(
        organism.avg_steps_between_hunts, 100
    )
    # Size strategy bonus for small/fast or large predators.
    if organism.size < 1.0:
        size_bonus = 1.5 if organism.speed > 2.0 else 1.0
    elif organism.size > 2.0:
        size_bonus = 1.3
    else:
        size_bonus = 1.0
    # Speed strategy bonus.
    if organism.speed > 2.0:
        speed_bonus = 1.4 if organism.size < 1.5 else 1.1
    else:
        speed_bonus = 1.0
    # Territory control grows slowly with lifetime.
    territory_bonus = min(1.5, organism.steps_taken / 1000.0)
    # Reward early first kills.
    early_success_bonus = (
        1.5
        if organism.organisms_consumed > 0 and organism.steps_taken < 500
        else 1.0
    )
    # Aggregate hunting-focused score.
    role_score = (
        organism.organisms_consumed * 400
        + (organism.organisms_consumed * 250 * hunting_efficiency)
        + (organism.organisms_consumed * 150 * size_bonus)
        + (organism.organisms_consumed * 150 * speed_bonus)
        + (organism.organisms_consumed * 100 * early_success_bonus)
        + (organism.steps_taken * 0.6 * organism.speed * territory_bonus)
    )
    # Soft penalty when too long since last hunt.
    if organism.steps_since_last_hunt > 150:
        role_score *= math.exp(-organism.steps_since_last_hunt / 350)
    # Carnivores that eat plants are heavily penalized.
    if organism.food_consumed > 0:
        role_score *= 0.01
    return role_score


def _herbivore_role_score(organism):
    """Score foraging success with survival bonuses and diet penalties."""
    # Efficiency of time between successful forage events.
    foraging_efficiency = calculate_consumption_efficiency(
        organism.avg_steps_between_food, 50
    )
    # Faster herbivores get a survival bonus.
    if organism.speed > 2.0:
        speed_bonus = 1.5
    elif organism.speed > 1.5:
        speed_bonus = 1.2
    else:
        speed_bonus = 1.0
    # Extreme sizes are slightly favored (harder to hunt / catch).
    if organism.size > 1.5:
        size_bonus = 1.3
    elif organism.size < 0.8:
        size_bonus = 1.2
    else:
        size_bonus = 1.0
    # Efficiency attribute contributes directly.
    efficiency_bonus = organism.energy_efficiency * 1.2
    # Aggregate foraging-focused score.
    role_score = (
        organism.food_consumed * 350
        + (organism.food_consumed * 250 * foraging_efficiency)
        + (organism.food_consumed * 100 * speed_bonus)
        + (organism.food_consumed * 100 * size_bonus)
        + (organism.food_consumed * 50 * efficiency_bonus)
        + (organism.steps_taken * 0.5 * organism.speed)
    )
    # Soft penalty when too long since last meal.
    if organism.steps_since_last_food > 100:
        role_score *= math.exp(-organism.steps_since_last_food / 250)
    # Herbivores that eat animals are heavily penalized.
    if organism.organisms_consumed > 0:
        role_score *= 0.01
    return role_score
