"""NEAT simulation loop: trials, sensing, scoreboard, and optional rendering."""

import json
import random

import neat
import pygame

from distance import within_radius
from food import Food
from logging_util import log_always, log_detailed
from organism import Organism
from scoreboard import Scoreboard


class Simulation:
    """Evolve organisms in a 2D foraging / predation environment."""

    def __init__(self, neat_config, sim_config):
        # Keep the NEAT config for population construction.
        self.neat_config = neat_config
        # Load JSON from disk when a path is provided.
        if isinstance(sim_config, str):
            with open(sim_config, "r", encoding="utf-8") as handle:
                self.sim_config = json.load(handle)
        else:
            self.sim_config = sim_config
        # Verbosity gate for hot-path logging.
        self.logging_level = self.sim_config.get("logging_level", "normal")
        # Sensible default when food count is omitted.
        if "num_food_items" not in self.sim_config:
            self.sim_config["num_food_items"] = 30
        # Food state for the active trial.
        self.food_items = []
        self.spawn_food()
        # Optional pygame renderer.
        self.renderer = None
        if self.sim_config.get("render", False):
            from renderer import Renderer

            screen_size = max(
                self.sim_config["environment_width"],
                self.sim_config["environment_height"],
            )
            self.renderer = Renderer(screen_size)
        # Shared environment contract consumed by organisms.
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
            "food_energy_value": self.sim_config.get("food_energy_value", 75),
            "movement_cost": self.sim_config.get("movement_cost"),
        }
        # Cached loop parameters.
        self.simulation_steps = self.sim_config["simulation_steps"]
        self.detection_radius = detection
        self.food_detection_radius = self.environment_config["food_detection_radius"]
        self.threat_detection_radius = self.environment_config[
            "threat_detection_radius"
        ]
        self.breeding_detection_radius = self.environment_config[
            "breeding_detection_radius"
        ]
        self.num_generations = self.sim_config["num_generations"]
        # Fresh scoreboard for this run.
        Scoreboard._species_records = {}
        # Filled by run().
        self.population = None
        # Episode-local children awaiting insertion into the live cast.
        self._episode_children = []
        # Stats captured before organism cleanup.
        self._last_generation_stats = None

    def spawn_food(self):
        """Clear and respawn food across the arena."""
        self.food_items.clear()
        margin = 10
        count = self.sim_config["num_food_items"]
        log_detailed(self.logging_level, f"Spawning {count} food items")
        width = self.sim_config["environment_width"]
        height = self.sim_config["environment_height"]
        for _ in range(count):
            x = random.randint(margin, width - margin)
            y = random.randint(margin, height - margin)
            self.food_items.append(
                Food(x, y, log_creation=self.logging_level == "detailed")
            )

    def register_episode_child(self, child):
        """Queue an episode-local child without mutating the NEAT population."""
        child.simulation = self
        self._episode_children.append(child)

    def eval_genomes(self, genomes, config):
        """Score genomes across a fixed number of environment trials."""
        genome_to_organism = {}
        log_detailed(self.logging_level, "=== Creating Organisms ===")
        # Instantiate one organism per genome.
        for genome_id, genome in genomes:
            x = random.randint(10, self.sim_config["environment_width"] - 10)
            y = random.randint(10, self.sim_config["environment_height"] - 10)
            species_id = self.population.species.get_species_id(genome_id)
            organism = Organism(
                genome,
                config,
                (x, y),
                self.environment_config,
                species_id=species_id,
                logging_level=self.logging_level,
            )
            organism.simulation = self
            genome_to_organism[genome_id] = organism
        num_trials = 3
        log_always(f"Starting evaluation with {len(genome_to_organism)} organisms")
        for trial in range(num_trials):
            self.spawn_food()
            self._episode_children.clear()
            # Reset NEAT-owned organisms for the new trial.
            organisms = []
            for genome_id, _genome in genomes:
                organism = genome_to_organism[genome_id]
                organism.reset(self.environment_config)
                organisms.append(organism)
            for step in range(self.simulation_steps):
                # Only pump events when a display exists (headless-safe).
                if pygame.display.get_init():
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self._finalize_genomes(
                                genomes, genome_to_organism, organisms
                            )
                            return
                        if event.type == pygame.MOUSEBUTTONDOWN and self.renderer:
                            self.handle_click(event.pos, organisms)
                for organism in organisms[:]:
                    if organism.energy <= 0:
                        if organism in organisms:
                            organisms.remove(organism)
                        continue
                    nearby_food = [
                        food
                        for food in self.food_items
                        if food.position is not None
                        and within_radius(
                            organism.position,
                            food.position,
                            self.food_detection_radius,
                        )
                    ]
                    nearby_organisms = []
                    nearby_threats = []
                    nearby_partners = []
                    for other in organisms:
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
                        if within_radius(
                            organism.position,
                            other.position,
                            self.detection_radius,
                        ):
                            nearby_organisms.append(other)
                    organism.take_action(
                        nearby_food,
                        nearby_organisms,
                        nearby_threats,
                        nearby_partners,
                    )
                    organism.update(self.food_items, organisms)
                # Admit episode-local children into the live cast.
                if self._episode_children:
                    organisms.extend(self._episode_children)
                    self._episode_children.clear()
                if not organisms:
                    log_always(
                        f"All organisms died at step {step + 1}/"
                        f"{self.simulation_steps} in trial {trial + 1}"
                    )
                    break
                if step % 50 == 0:
                    alive = len(organisms)
                    avg_energy = sum(o.energy for o in organisms) / max(1, alive)
                    log_detailed(
                        self.logging_level,
                        f"Trial {trial + 1}, step {step}: {alive} alive, "
                        f"avg energy {avg_energy:.2f}",
                    )
                if self.renderer and not self.renderer.render(
                    organisms, self.food_items
                ):
                    self._finalize_genomes(genomes, genome_to_organism, organisms)
                    return
            # Store this trial's fitness for each NEAT genome.
            for genome_id, genome in genomes:
                organism = genome_to_organism[genome_id]
                if not hasattr(genome, "trial_fitnesses"):
                    genome.trial_fitnesses = []
                if organism in organisms and organism.energy > 0:
                    genome.trial_fitnesses.append(organism.calculate_fitness())
                else:
                    genome.trial_fitnesses.append(0.0)
        self._finalize_genomes(genomes, genome_to_organism, organisms)

    def _finalize_genomes(self, genomes, genome_to_organism, organisms):
        """Assign median fitness, update scoreboard, then cleanup organisms."""
        for genome_id, genome in genomes:
            trials = getattr(genome, "trial_fitnesses", [])
            genome.fitness = sorted(trials)[len(trials) // 2] if trials else 0.0
            if hasattr(genome, "trial_fitnesses"):
                delattr(genome, "trial_fitnesses")
        survivors = [
            genome_to_organism[genome_id]
            for genome_id, _ in genomes
            if genome_to_organism[genome_id] in organisms
            and genome_to_organism[genome_id].energy > 0
        ]
        if survivors:
            best = max(survivors, key=lambda org: org.highest_fitness)
            Scoreboard.record_species(
                species_id=str(best.species_id),
                organism=best,
                fitness=best.highest_fitness,
                generation=self.population.generation,
                config=self.neat_config,
            )
            # Snapshot primitive stats before organisms are cleaned.
            self._last_generation_stats = {
                "count": len(survivors),
                "carnivores": sum(1 for o in survivors if o.is_carnivore),
                "herbivores": sum(1 for o in survivors if not o.is_carnivore),
                "avg_fitness": sum(o.highest_fitness for o in survivors)
                / len(survivors),
                "max_fitness": max(o.highest_fitness for o in survivors),
                "avg_energy": sum(o.energy for o in survivors) / len(survivors),
                "avg_size": sum(o.size for o in survivors) / len(survivors),
                "avg_speed": sum(o.speed for o in survivors) / len(survivors),
                "best_species_id": best.species_id,
                "best_fitness": best.highest_fitness,
                "best_is_carnivore": best.is_carnivore,
                "best_food": best.food_consumed,
                "best_kills": best.organisms_consumed,
            }
            self._print_generation_dashboard(
                self.population.generation, self._last_generation_stats
            )
        else:
            self._last_generation_stats = None
            log_always("No organisms survived to record in scoreboard")
        for organism in genome_to_organism.values():
            organism.cleanup()
        genome_to_organism.clear()

    def _print_generation_dashboard(self, generation, stats):
        """Emit a compact generation summary from snapshot stats."""
        log_always("=== Generation Evaluation ===")
        log_always(f"Generation: {generation}")
        log_always(f"Number of organisms: {stats['count']}")
        log_always(
            f"Organism Types: {stats['carnivores']} Carnivores, "
            f"{stats['herbivores']} Herbivores"
        )
        log_always(
            f"Fitness Stats: Avg: {stats['avg_fitness']:.2f}, "
            f"Max: {stats['max_fitness']:.2f}"
        )
        log_always(f"Energy Avg: {stats['avg_energy']:.2f}")
        log_always(f"Size Avg: {stats['avg_size']:.2f}")
        log_always(f"Speed Avg: {stats['avg_speed']:.2f}")
        log_always("Best organism found:")
        log_always(f"- Species ID: {stats['best_species_id']}")
        log_always(f"- Highest Fitness: {stats['best_fitness']:.2f}")
        diet = "Carnivore" if stats["best_is_carnivore"] else "Herbivore"
        log_always(f"- Type: {diet}")
        log_always(f"- Food Consumed: {stats['best_food']}")
        log_always(f"- Organisms Consumed: {stats['best_kills']}")

    def handle_click(self, pos, organisms):
        """Log clicks on organisms when a renderer is active."""
        if not self.renderer:
            return
        log_detailed(self.logging_level, f"Click at {pos}")
        for organism in organisms:
            if organism.position and within_radius(pos, organism.position, 10):
                log_detailed(
                    self.logging_level,
                    f"Clicked organism species={organism.species_id}",
                )
                break

    def run(self, max_generations=None):
        """Evolve the population until generations or fitness threshold."""
        self.population = neat.Population(self.neat_config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(neat.StatisticsReporter())
        fitness_threshold = self.neat_config.fitness_threshold
        limit = self.num_generations if max_generations is None else max_generations
        while True:
            gen = self.population.generation
            if self.renderer:
                self.renderer.set_generation(gen)
            if gen >= limit:
                log_always(f"Reached maximum generations ({limit})")
                break
            winner = self.population.run(self.eval_genomes, 1)
            if self.renderer:
                self.renderer.cleanup_resources()
                # Drain events only when a display is active.
                if pygame.display.get_init():
                    pygame.event.get()
            if (
                winner is not None
                and winner.fitness is not None
                and winner.fitness >= fitness_threshold
            ):
                log_always(f"Fitness threshold ({fitness_threshold}) reached!")
                break
            positive = sum(
                1
                for genome in self.population.population.values()
                if genome.fitness is not None and genome.fitness > 0
            )
            log_always(f"Generation {gen} complete")
            log_always(f"Number of genomes with positive fitness: {positive}")
            if self._last_generation_stats:
                Scoreboard.display_terminal_dashboard(gen)
            else:
                log_always("Warning: No organisms survived this generation")
            if pygame.display.get_init():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
