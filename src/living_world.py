"""Continuous living-world simulation harness with steady-state NEAT selection."""

import json
import random

import neat
import pygame

from distance import within_radius
from food_ecology import FoodEcology
from living_world_renderer import LivingWorldRenderer
from logging_util import log_always, log_detailed
from organism import Organism
from population_registry import PopulationRegistry
from rolling_fitness import RollingFitnessTracker
from scoreboard import Scoreboard
from selection_scheduler import SelectionScheduler
from spatial import SpatialGrid
from world_clock import WorldClock


class LivingWorldSimulation:
    """Run a persistent ecology that evolves under predation and food scarcity."""

    def __init__(self, neat_config, sim_config):
        # NEAT configuration for genome operations.
        self.neat_config = neat_config
        if isinstance(sim_config, str):
            with open(sim_config, "r", encoding="utf-8") as handle:
                self.sim_config = json.load(handle)
        else:
            self.sim_config = sim_config
        self.logging_level = self.sim_config.get("logging_level", "normal")
        self.dashboard_level = self.sim_config.get("dashboard_level", "normal")
        Scoreboard.set_dashboard_level(self.dashboard_level)
        # Shared environment contract for organisms.
        detection = self.sim_config["detection_radius"]
        self.environment_config = {
            "width": self.sim_config["environment_width"],
            "height": self.sim_config["environment_height"],
            "boundary_penalty": self.sim_config.get("boundary_penalty", 0.5),
            "detection_radius": detection,
            "food_detection_radius": self.sim_config.get(
                "food_detection_radius", detection
            ),
            "threat_detection_radius": self.sim_config.get(
                "threat_detection_radius", detection
            ),
            "breeding_detection_radius": self.sim_config.get(
                "breeding_detection_radius", detection
            ),
            "starting_energy": self.sim_config.get("starting_energy"),
            "food_energy_value": self.sim_config.get("food_energy_value", 40),
            "movement_cost": self.sim_config.get("movement_cost"),
        }
        # Ecology and population subsystems.
        self.food_ecology = FoodEcology(self.sim_config)
        self.max_population = int(self.sim_config.get("max_population", 120))
        self.registry = PopulationRegistry(
            self.max_population,
            self.environment_config,
            neat_config,
            self.logging_level,
        )
        self.fitness_tracker = RollingFitnessTracker(self.sim_config)
        self.selection = SelectionScheduler(
            self.sim_config.get("selection_interval_steps", 4000),
            self.max_population,
            self.sim_config.get("cull_fraction", 0.1),
            self.sim_config.get("immigration_rate", 0.03),
        )
        self.clock = WorldClock(
            epoch_length=int(self.sim_config.get("simulation_steps", 2000))
        )
        self.max_world_steps = self.sim_config.get("max_world_steps")
        # Spatial indexing reused each step.
        self._grid_cell_size = max(50, detection // 2)
        self._food_grid = SpatialGrid(self._grid_cell_size)
        self._org_grid = SpatialGrid(self._grid_cell_size)
        self.detection_radius = detection
        self.food_detection_radius = self.environment_config["food_detection_radius"]
        self.threat_detection_radius = self.environment_config[
            "threat_detection_radius"
        ]
        self.breeding_detection_radius = self.environment_config[
            "breeding_detection_radius"
        ]
        self.render_stride = max(1, int(self.sim_config.get("render_stride", 2)))
        self.batch_inference = bool(self.sim_config.get("batch_inference", True))
        self._batch_engine = None
        if self.batch_inference:
            from batch_inference import BatchInferenceEngine

            self._batch_engine = BatchInferenceEngine()
        self.population = None
        self.renderer = None
        if self.sim_config.get("render", True):
            self.renderer = LivingWorldRenderer(self.sim_config, self.logging_level)
            self.renderer.set_environment_config(self.environment_config)
        self._pending_births = []

    def register_birth(self, child, child_genome, parent_a, parent_b):
        """Admit an in-world birth into the live population."""
        if self.registry.at_capacity():
            self.registry.cull_worst(self.fitness_tracker, 1)
        genome_id = self.registry.add_birth(
            child_genome, child, self._batch_engine
        )
        child.simulation = self
        self.fitness_tracker.init_genome(genome_id)
        self.fitness_tracker.record_offspring(
            [getattr(parent_a, "genome_id", None), getattr(parent_b, "genome_id", None)]
        )
        self._pending_births.append(child)
        parent_a.fitness_bonus += 100
        parent_b.fitness_bonus += 100
        log_detailed(
            self.logging_level,
            f"Living birth genome={genome_id} species={child.species_id}",
        )

    def _nearby_entities(self, organism, organisms, food_grid, org_grid):
        """Return sensed entities using spatial grids."""
        nearby_food = list(
            food_grid.query(organism.position, self.food_detection_radius)
        )
        nearby_organisms = list(
            org_grid.query(organism.position, self.detection_radius)
        )
        nearby_threats = []
        nearby_partners = []
        for other in nearby_organisms:
            if other is organism:
                continue
            if other.is_carnivore and not organism.is_carnivore:
                if within_radius(
                    organism.position,
                    other.position,
                    self.threat_detection_radius,
                ):
                    nearby_threats.append(other)
            elif other.species_id == organism.species_id:
                if within_radius(
                    organism.position,
                    other.position,
                    self.breeding_detection_radius,
                ):
                    nearby_partners.append(other)
        return nearby_food, nearby_organisms, nearby_threats, nearby_partners

    def _simulation_step(self):
        """Advance the world by one tick."""
        organisms = self.registry.alive_organisms()
        food_grid = self._food_grid
        org_grid = self._org_grid
        food_grid.clear()
        org_grid.clear()
        for food in self.food_ecology.food_items:
            if food.position is not None:
                food_grid.insert(food, food.position)
        for other in organisms:
            if other.position is not None and other.energy > 0:
                org_grid.insert(other, other.position)
        next_organisms = []
        sensing_cache = {}
        for organism in organisms:
            if organism.energy <= 0:
                continue
            sensing_cache[organism] = self._nearby_entities(
                organism, organisms, food_grid, org_grid
            )
        for organism in organisms:
            if organism.energy <= 0:
                continue
            nearby_food, nearby_orgs, nearby_threats, nearby_partners = sensing_cache[
                organism
            ]
            organism.take_action(
                nearby_food, nearby_orgs, nearby_threats, nearby_partners
            )
            organism.update(nearby_food, nearby_orgs)
            self.fitness_tracker.tick_alive(organism)
            if organism.energy > 0:
                next_organisms.append(organism)
            else:
                self.fitness_tracker.finalize_death(
                    organism, organism.calculate_fitness()
                )
        if self._pending_births:
            next_organisms.extend(self._pending_births)
            self._pending_births.clear()
        for genome_id in self.registry.remove_dead():
            if self._batch_engine is not None:
                self._batch_engine._networks.pop(genome_id, None)
        self.food_ecology.tick()
        if self.clock.step % 500 == 0:
            self.food_ecology.prune_consumed()
        self.clock.tick()
        return next_organisms

    def _handle_events(self, organisms):
        """Pump pygame events for quit, pan, and selection."""
        if not pygame.get_init():
            return True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if self.renderer is not None:
                self.renderer.handle_event(event, organisms)
        return True

    def _maybe_selection(self):
        """Run steady-state selection on interval boundaries."""
        if not self.selection.should_run(self.clock.step):
            return
        for organism in self.registry.all_organisms():
            organism.simulation = self
        self.selection.run(
            self.population,
            self.registry,
            self.fitness_tracker,
            self._batch_engine,
            self.neat_config,
        )
        for organism in self.registry.all_organisms():
            organism.simulation = self

    def _record_scoreboard(self):
        """Update scoreboard from the current best survivor."""
        survivors = self.registry.alive_organisms()
        if not survivors:
            return
        best = max(survivors, key=lambda org: org.highest_fitness)
        Scoreboard.record_species(
            species_id=str(best.species_id),
            organism=best,
            fitness=best.highest_fitness,
            generation=self.clock.epoch,
            config=self.neat_config,
        )

    def run(self):
        """Start the continuous living-world loop."""
        self.population = neat.Population(self.neat_config)
        self.population.add_reporter(neat.StdOutReporter(True))
        genomes = list(self.population.population.items())
        self.registry.seed_from_genomes(
            genomes, self.population, self._batch_engine
        )
        for organism in self.registry.all_organisms():
            organism.simulation = self
            self.fitness_tracker.init_genome(organism.genome_id)
        log_always(
            f"Living world started: {self.registry.count()} organisms, "
            f"world {self.environment_config['width']}x"
            f"{self.environment_config['height']}"
        )
        while self.clock.alive(self.max_world_steps):
            organisms = self._simulation_step()
            if not self._handle_events(organisms):
                break
            self._maybe_selection()
            if self.clock.step % self.clock.epoch_length == 0:
                self._record_scoreboard()
                log_always(
                    f"Epoch {self.clock.epoch} step {self.clock.step}: "
                    f"{self.registry.count()} alive, "
                    f"food {self.food_ecology.active_food_count()}"
                )
            if self.renderer and self.clock.step % self.render_stride == 0:
                self.renderer.set_world_step(self.clock.step)
                self.renderer.set_generation(self.clock.epoch)
                if not self.renderer.render(
                    organisms,
                    self.food_ecology.food_items,
                    self.food_ecology.clouds,
                    self.food_ecology,
                ):
                    break
        log_always(f"Living world ended at step {self.clock.step}")
        Scoreboard.display_final_summary(self.logging_level)
