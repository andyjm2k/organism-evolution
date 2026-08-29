"""Organism agent: sensing, movement, consumption, breeding, and fitness."""

import math
import random

import neat

from distance import colliding, distance, within_radius
from fitness import calculate_fitness
from logging_util import log_detailed
from network_inputs import build_network_inputs
from species_names import generate_scientific_name

class Organism:
    """A single evolved agent controlled by a NEAT feed-forward network."""

    def __init__(
        self,
        genome,
        config,
        position,
        environment_config,
        species_id=None,
        logging_level="normal",
    ):
        # Genome drives network topology and is scored after evaluation.
        self.genome = genome
        # NEAT config needed to build and mutate networks.
        self.config = config
        # Feed-forward phenotype created from the genome.
        self.network = neat.nn.FeedForwardNetwork.create(genome, config)
        # Shared simulation settings (size, radii, economy).
        self.environment_config = environment_config
        # World-space coordinates as a mutable (x, y) pair.
        self.position = position
        # Species identity used for breeding compatibility and coloring.
        self.species_id = species_id
        # Explicit bonuses awarded for breeding / notable events.
        self.fitness_bonus = 0.0
        # Peak fitness observed across the organism lifetime.
        self.highest_fitness = 0.0
        # Back-reference set by Simulation when the organism is registered.
        self.simulation = None
        # Controls whether detailed debug lines are printed.
        self.logging_level = logging_level
        # Lifetime step counter (incremented once per update).
        self.steps_taken = 0
        # Herbivore forage counter.
        self.food_consumed = 0
        # Carnivore hunt counter.
        self.organisms_consumed = 0
        # Prior position for velocity sensing.
        self.last_position = position
        # Whether the last action produced movement.
        self.was_moving = False
        # Cooldown bookkeeping for breeding attempts.
        self.steps_since_breeding = 1000
        self.breeding_cooldown = 200
        # Time since last successful forage / hunt for fitness curves.
        self.steps_since_last_food = 0
        self.steps_since_last_hunt = 0
        self.avg_steps_between_food = []
        self.avg_steps_between_hunts = []
        # Coarse exploration trail for fitness.
        self.last_positions = []
        # Diet type is stable within a species id.
        self.is_carnivore = self._diet_from_species(species_id)
        # Derive speed/size/energy from genome structure + config economy.
        self._calculate_attributes()
        # Start from configured starting energy (not always max capacity).
        self.energy = self._starting_energy
        log_detailed(
            self.logging_level,
            f"Organism created at {self.position} "
            f"({'Carnivore' if self.is_carnivore else 'Herbivore'})",
        )

    @staticmethod
    def _diet_from_species(species_id):
        """Map species id to a stable carnivore/herbivore diet."""
        # Missing species → random diet for throwaway agents.
        if species_id is None:
            return random.random() > 0.5
        # Coerce string ids to integers when possible.
        if isinstance(species_id, str):
            try:
                num_id = int(species_id)
            except ValueError:
                num_id = hash(species_id)
        else:
            num_id = species_id
        # Half of species ids are carnivores (mod 4 in {2, 3}).
        return abs(num_id) % 4 >= 2

    def _calculate_attributes(self):
        """Derive physical attributes from genome complexity and config."""
        # Count nodes and connections as complexity proxies.
        self.node_count = len(self.genome.nodes)
        self.connection_count = len(self.genome.connections)
        # Positive node keys are treated as hidden nodes for visuals.
        self.hidden_count = len([n for n in self.genome.nodes.keys() if n > 0])
        # Role-specific base stats.
        base_speed = 1.2 if self.is_carnivore else 1.3
        base_size = 1.1 if self.is_carnivore else 1.0
        base_energy = 300 if self.is_carnivore else 250
        base_efficiency = 1.5 if self.is_carnivore else 1.8
        # Complexity scales speed/size within a clamped band.
        speed_factor = 0.25 + (self.node_count / 10.0)
        size_factor = 0.25 + (self.connection_count / 10.0)
        self.speed = base_speed * min(4.0, max(0.25, speed_factor))
        self.size = base_size * min(4.0, max(0.25, size_factor))
        # Capacity scales with size; starting energy comes from config when set.
        energy_mult = float(self.environment_config.get("energy_multiplier", 3.0))
        self.max_energy = base_energy * self.size * energy_mult
        # Prefer configured starting energy as the episode spawn amount.
        starting = self.environment_config.get("starting_energy")
        self._starting_energy = (
            float(starting) if starting is not None else self.max_energy
        )
        # Never allow start above capacity.
        self._starting_energy = min(self._starting_energy, self.max_energy)
        self.energy_efficiency = base_efficiency * (2.0 - (self.size * 0.1))
        # Movement cost prefers config, else role default.
        cfg_move = self.environment_config.get("movement_cost")
        base_move = float(cfg_move) if cfg_move is not None else (
            0.007 if self.is_carnivore else 0.005
        )
        self.movement_energy_cost = base_move * (self.speed ** 1.1) * (self.size ** 0.5)
        # Visual radius / spikes derived from network shape.
        self._calculate_visual_properties()
        log_detailed(
            self.logging_level,
            f"Attributes size={self.size:.2f} speed={self.speed:.2f} "
            f"max_energy={self.max_energy:.1f}",
        )

    def _calculate_visual_properties(self):
        """Map network complexity onto drawable spike geometry."""
        # Body radius grows slowly with node count.
        self.base_radius = 4 + min(8, self.node_count / 10.0)
        # Spike count tracks hidden complexity.
        self.num_spikes = min(16, max(3, 3 + self.hidden_count))
        # Spike length tracks connection density.
        if self.node_count > 1:
            density = self.connection_count / self.node_count
            self.spike_length = min(8, max(3, 3 + density * 2))
        else:
            self.spike_length = 3

    def take_action(
        self,
        nearby_food,
        nearby_organisms,
        nearby_threats=None,
        nearby_breeding_partners=None,
    ):
        """Activate the network, move, and optionally attempt breeding."""
        if self.energy <= 0:
            return
        self.last_position = self.position
        inputs = build_network_inputs(
            self,
            nearby_food,
            nearby_organisms,
            nearby_threats,
            nearby_breeding_partners,
            self.environment_config,
        )
        compiled = getattr(self, "_compiled_network", None)
        if compiled is not None:
            network_output = compiled.activate(inputs)
        else:
            network_output = self.network.activate(inputs)
        movement_outputs = network_output[:7]
        breeding_desire = network_output[7]
        # Small noise prevents permanent hard ties on the first generation.
        noisy = [v + random.uniform(-0.01, 0.01) for v in movement_outputs]
        direction_index = noisy.index(max(noisy))
        # Seven discrete headings around the circle.
        angles = [0, 60, 120, 180, 240, 300, 360]
        width = self.environment_config.get("width", 800)
        height = self.environment_config.get("height", 600)
        move_x = move_y = 0.0
        if direction_index < len(angles):
            angle_rad = math.radians(angles[direction_index])
            move_x = self.speed * math.cos(angle_rad)
            move_y = self.speed * math.sin(angle_rad)
            # Light jitter avoids perfectly straight tracks.
            move_x += random.uniform(-0.1, 0.1) * self.speed
            move_y += random.uniform(-0.1, 0.1) * self.speed
        # Clamp inside the arena (update may wrap with penalty).
        new_x = max(0, min(self.position[0] + move_x, width))
        new_y = max(0, min(self.position[1] + move_y, height))
        if abs(move_x) > 0 or abs(move_y) > 0:
            self.position = (new_x, new_y)
            self.was_moving = True
            # Pay movement energy from config-derived cost.
            self.energy = max(0, self.energy - self.movement_energy_cost)
        else:
            self.was_moving = False
        # Breeding attempt when desire and readiness align.
        if breeding_desire > 0.5 and self.can_breed():
            self._try_breed(nearby_breeding_partners or nearby_organisms)

    def update(self, food_items, organisms):
        """Advance per-step timers, resting cost, boundaries, and consumption."""
        # Single place that increments lifetime / cooldown counters.
        self.steps_taken += 1
        self.steps_since_breeding += 1
        self.steps_since_last_food += 1
        self.steps_since_last_hunt += 1
        # Cap energy at maximum capacity.
        self.energy = min(self.energy, self.max_energy)
        # Invalid position abort.
        if self.position is None:
            return
        # Tiny survival bonus accumulates for staying alive.
        self.fitness_bonus += 0.05
        # Resting metabolism scales with size.
        self.energy -= 0.01 * self.size
        # Track coarse cells for exploration fitness.
        rounded = (
            round(self.position[0] / 10) * 10,
            round(self.position[1] / 10) * 10,
        )
        self.last_positions.append(rounded)
        if len(self.last_positions) > 100:
            self.last_positions.pop(0)
        # Boundary wrap with configurable energy penalty.
        self._apply_boundary_rules()
        # Single consumption path (not duplicated in take_action).
        self.check_for_food(food_items)
        if self.is_carnivore:
            self.hunt_prey(organisms)

    def _apply_boundary_rules(self):
        """Wrap at edges and apply boundary energy penalty."""
        x, y = self.position
        width = self.environment_config["width"]
        height = self.environment_config["height"]
        penalty = self.environment_config.get("boundary_penalty", 0.5)
        wrapped = False
        if x < 0:
            x = width
            wrapped = True
        elif x > width:
            x = 0
            wrapped = True
        if y < 0:
            y = height
            wrapped = True
        elif y > height:
            y = 0
            wrapped = True
        if wrapped:
            self.energy *= 1 - penalty
        self.position = (x, y)

    def check_for_food(self, food_items):
        """Herbivores consume the first colliding food pellet."""
        # Carnivores and dead agents do not forage.
        if self.energy <= 0 or self.is_carnivore:
            return
        food_energy = float(self.environment_config.get("food_energy_value", 75))
        food_radius = 5
        for food in food_items:
            if food.position is None:
                continue
            if colliding(self.position, self.get_radius(), food.position, food_radius):
                gain = food_energy * self.energy_efficiency
                self.energy = min(self.max_energy, self.energy + gain)
                self.food_consumed += 1
                if self.steps_since_last_food > 0:
                    self.avg_steps_between_food.append(self.steps_since_last_food)
                    if len(self.avg_steps_between_food) > 10:
                        self.avg_steps_between_food.pop(0)
                self.steps_since_last_food = 0
                food.position = None
                self.fitness_bonus += 10
                log_detailed(
                    self.logging_level,
                    f"Herbivore ate food; energy={self.energy:.1f}",
                )
                break

    def hunt_prey(self, organisms):
        """Carnivores consume one eligible smaller prey on collision."""
        if self.energy <= 0 or not self.is_carnivore:
            return
        for other in organisms:
            if (
                other is self
                or other.is_carnivore
                or other.energy <= 0
                or other.position is None
                or self.position is None
            ):
                continue
            if not colliding(
                self.position, self.get_radius(), other.position, other.get_radius()
            ):
                continue
            # May eat prey up to 80% of own size (inclusive margin via >=).
            if self.size >= other.size * 0.8:
                gain = other.energy * 0.7
                self.energy = min(self.max_energy, self.energy + gain)
                self.organisms_consumed += 1
                other.energy = 0
                if self.steps_since_last_hunt > 0:
                    self.avg_steps_between_hunts.append(self.steps_since_last_hunt)
                    if len(self.avg_steps_between_hunts) > 10:
                        self.avg_steps_between_hunts.pop(0)
                self.steps_since_last_hunt = 0
                self.fitness_bonus += 25
                log_detailed(
                    self.logging_level,
                    f"Carnivore ate prey; energy={self.energy:.1f}",
                )
                break

    def can_breed(self):
        """Return True when energy, cooldown, and boundary rules allow breeding."""
        # Unified energy gate (was inconsistently 50 vs 100).
        if self.energy < 100:
            return False
        if self.steps_since_breeding < self.breeding_cooldown:
            return False
        if self.position is None:
            return False
        width = self.environment_config.get("width", 800)
        height = self.environment_config.get("height", 600)
        min_distance = min(width, height) * 0.1
        dist_edge = min(
            self.position[0],
            self.position[1],
            width - self.position[0],
            height - self.position[1],
        )
        return dist_edge >= min_distance

    def _try_breed(self, candidates):
        """Breed with a valid partner; offspring stay episode-local."""
        # Filter partners that share species and are themselves ready.
        partners = []
        for org in candidates:
            if org is self:
                continue
            if org.species_id != self.species_id:
                continue
            if not org.can_breed():
                continue
            if not within_radius(
                self.position,
                org.position,
                (self.get_radius() + org.get_radius()) * 1.5,
            ):
                continue
            partners.append(org)
        if not partners:
            return
        partner = partners[0]
        # Pay energy and reset cooldowns for both parents.
        self.energy -= 50
        partner.energy -= 50
        self.steps_since_breeding = 0
        partner.steps_since_breeding = 0
        # Spawn near the midpoint of the parents.
        spawn_x = (self.position[0] + partner.position[0]) / 2 + random.uniform(-5, 5)
        spawn_y = (self.position[1] + partner.position[1]) / 2 + random.uniform(-5, 5)
        width = self.environment_config["width"]
        height = self.environment_config["height"]
        spawn_x = max(0, min(spawn_x, width))
        spawn_y = max(0, min(spawn_y, height))
        # Rank parents by current fitness for NEAT crossover order.
        self.genome.fitness = self.calculate_fitness()
        partner.genome.fitness = partner.calculate_fitness()
        if self.genome.fitness >= partner.genome.fitness:
            parent1, parent2 = self.genome, partner.genome
        else:
            parent1, parent2 = partner.genome, self.genome
        # Use a disposable id; child is NOT inserted into the NEAT population.
        child_genome = type(self.genome)(0)
        child_genome.configure_crossover(parent1, parent2, self.config.genome_config)
        child_genome.mutate(self.config.genome_config)
        child = Organism(
            child_genome,
            self.config,
            (spawn_x, spawn_y),
            self.environment_config,
            species_id=self.species_id,
            logging_level=self.logging_level,
        )
        # Register only with the live episode list via simulation hook.
        if self.simulation is not None:
            self.simulation.register_episode_child(child)
            self.fitness_bonus += 100
            partner.fitness_bonus += 100
            log_detailed(
                self.logging_level,
                f"Breeding produced episode-local child in species {self.species_id}",
            )
        else:
            # Refund if we cannot place the child anywhere.
            self.energy += 50
            partner.energy += 50

    def calculate_fitness(self):
        """Delegate fitness calculation to the fitness module."""
        value = calculate_fitness(self, self.environment_config)
        # Track lifetime peak for scoreboard reporting.
        if value > self.highest_fitness:
            self.highest_fitness = value
        return value

    def distance_to(self, other):
        """Return linear Euclidean distance to another positioned object."""
        # Public API is linear units so callers compare against linear radii.
        if other is None or getattr(other, "position", None) is None:
            return float("inf")
        if self.position is None:
            return float("inf")
        return distance(self.position, other.position)

    def get_radius(self):
        """Return drawable/collision radius clamped to a readable range."""
        return max(2, min(20, self.base_radius))

    def get_active_node_count(self):
        """Return total node count for renderer spike scaling."""
        return len(self.genome.nodes)

    def calculate_spikes(self):
        """Return spike count used by the scoreboard visuals."""
        type_bonus = 2 if self.is_carnivore else 0
        return min(16, max(3, 3 + self.hidden_count + type_bonus))

    def calculate_spike_length(self):
        """Return spike length used by the scoreboard visuals."""
        if self.node_count > 1:
            density = self.connection_count / self.node_count
            length = (3 + density * 2) * (1.2 if self.is_carnivore else 1.0)
        else:
            length = 3
        return min(8, max(3, length))

    @staticmethod
    def generate_scientific_name():
        """Mint a random binomial-style name for a successful species."""
        return generate_scientific_name()

    def reset(self, environment_config):
        """Reset episode state while keeping the genome/network."""
        self.environment_config = environment_config
        width = environment_config.get("width", 800)
        height = environment_config.get("height", 600)
        self.position = (
            random.randint(10, max(11, width - 10)),
            random.randint(10, max(11, height - 10)),
        )
        self.last_position = self.position
        self.steps_taken = 0
        self.food_consumed = 0
        self.organisms_consumed = 0
        self.was_moving = False
        self.fitness_bonus = 0.0
        self.highest_fitness = 0.0
        self.steps_since_breeding = 1000
        self.steps_since_last_food = 0
        self.steps_since_last_hunt = 0
        self.last_positions = []
        self.is_carnivore = self._diet_from_species(self.species_id)
        self._calculate_attributes()
        # Reset energy to configured starting amount for the new episode.
        self.energy = self._starting_energy

    def cleanup(self):
        """Drop heavy references after evaluation completes."""
        self.network = None
        self.genome = None
        self.config = None
        self.simulation = None
        self._compiled_network = None
        self.avg_steps_between_food.clear()
        self.avg_steps_between_hunts.clear()
        self.last_positions.clear()
