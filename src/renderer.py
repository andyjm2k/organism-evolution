import math
import os
import random

import colorsys
import pygame

from logging_util import log_detailed


class Renderer:
    """Pygame renderer for the simulation arena and species scoreboard."""

    def __init__(self, size, logging_level="normal"):
        # Avoid re-initializing pygame when main already did so.
        if not pygame.get_init():
            pygame.init()
        self.logging_level = logging_level
        self.scoreboard_width = 300
        self.scoreboard_height = size
        total_width = size + self.scoreboard_width
        if not pygame.display.get_init():
            self.screen = pygame.display.set_mode((total_width, size))
        else:
            self.screen = pygame.display.get_surface()
            if self.screen is None:
                self.screen = pygame.display.set_mode((total_width, size))
        pygame.display.set_caption("Evolution Simulation")
        self.scoreboard_rect = pygame.Rect(size, 0, self.scoreboard_width, size)
        self.clock = pygame.time.Clock()
        self.generation = 0
        self.species_colors = {}
        self.species_surfaces = {}
        self.text_surfaces = {}
        self.last_cache_clear = 0
        log_detailed(logging_level, f"Renderer initialized with screen size: {size}x{size}")

        pygame.font.init()
        font_dir = os.path.join(
            os.path.dirname(__file__), "..", "assets", "fonts"
        )
        try:
            self.title_font = pygame.font.Font(
                os.path.join(font_dir, "Roboto-Bold.ttf"), 36
            )
            self.header_font = pygame.font.Font(
                os.path.join(font_dir, "Roboto-Medium.ttf"), 28
            )
            self.font = pygame.font.Font(
                os.path.join(font_dir, "Roboto-Regular.ttf"), 20
            )
        except (FileNotFoundError, OSError):
            log_detailed(logging_level, "Falling back to default font")
            self.title_font = pygame.font.Font(None, 36)
            self.header_font = pygame.font.Font(None, 28)
            self.font = pygame.font.Font(None, 20)

        self.colors = {
            "background": (255, 255, 255),
            "food": (0, 255, 0),
            "text": (0, 0, 0),
            "card": (240, 240, 240),
            "border": (200, 200, 200),
        }

    def get_species_color(self, species_id, is_carnivore):
        """Return a stable color for a species id."""
        if species_id not in self.species_colors:
            hash_val = hash(species_id)
            random.seed(hash_val)
            if is_carnivore:
                hue = random.uniform(0, 60) / 360.0
                saturation = random.uniform(0.7, 1.0)
                lightness = random.uniform(0.3, 0.7)
            else:
                hue = random.uniform(30, 120) / 360.0
                saturation = random.uniform(0.4, 0.8)
                lightness = random.uniform(0.2, 0.6)
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            self.species_colors[species_id] = tuple(int(x * 255) for x in rgb)
            random.seed()
        return self.species_colors[species_id]

    def render(self, organisms, food_items):
        """Draw one simulation frame; return False when the window closes."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill((255, 255, 255))
        main_width = self.screen.get_width() - self.scoreboard_width
        main_height = self.screen.get_height()
        breeding_boundary_x = main_width * 0.1
        breeding_boundary_y = main_height * 0.1
        breeding_width = main_width - (breeding_boundary_x * 2)
        breeding_height = main_height - (breeding_boundary_y * 2)
        breeding_rect = pygame.Rect(
            breeding_boundary_x,
            breeding_boundary_y,
            breeding_width,
            breeding_height,
        )
        pygame.draw.rect(self.screen, (240, 250, 240), breeding_rect, 1)
        if not hasattr(self, "breeding_zone_text"):
            self.breeding_zone_text = self.font.render(
                "Breeding Safe Zone", True, (100, 180, 100)
            )
        self.screen.blit(
            self.breeding_zone_text,
            (breeding_boundary_x + 5, breeding_boundary_y - 25),
        )

        render_limit = 500
        if food_items:
            render_food = food_items[:render_limit]
            for food in render_food:
                if food.position is not None:
                    pos = (int(food.position[0]), int(food.position[1]))
                    pygame.draw.circle(self.screen, (0, 255, 0), pos, 4)

        if organisms:
            render_organisms = organisms[:render_limit]
            for organism in render_organisms:
                if organism.position is not None and organism.energy > 0:
                    pos = (int(organism.position[0]), int(organism.position[1]))
                    species_id_str = str(organism.species_id)
                    color = self.species_colors.get(species_id_str) or self.get_species_color(
                        species_id_str, organism.is_carnivore
                    )
                    self.draw_organism_with_spikes(
                        self.screen,
                        pos,
                        color,
                        organism.get_radius(),
                        organism.num_spikes,
                        organism.get_active_node_count(),
                        organism.is_carnivore,
                    )

        if organisms:
            species_count = len(set(org.species_id for org in organisms))
            debug_text = (
                f"Food: {len(food_items)} | Gen: {self.generation} | "
                f"Species: {species_count}"
            )
            self.screen.blit(self.font.render(debug_text, True, (0, 0, 0)), (10, 10))

        self._render_scoreboard()
        pygame.display.flip()
        self.clock.tick(60)
        return True

    def set_generation(self, generation):
        """Update generation label and periodically clear caches."""
        if generation != self.generation:
            self.cleanup_resources(light=True)
        self.generation = generation
        if generation % 10 == 0 and generation > self.last_cache_clear:
            self.species_colors.clear()
            self.species_surfaces.clear()
            self.text_surfaces.clear()
            self.last_cache_clear = generation

    def cleanup_resources(self, light=False):
        """Clear renderer caches; full mode also clears pygame scratch state."""
        self.species_surfaces.clear()
        self.text_surfaces.clear()
        if light:
            return
        self.species_colors.clear()
        if self.screen is not None:
            self.screen.fill((255, 255, 255))

    def draw_organism_with_spikes(
        self, screen, position, color, base_radius, num_spikes, num_nodes, is_carnivore
    ):
        """Draw an organism body and optional spikes."""
        radius = min(20, max(5, base_radius + (num_nodes / 10)))
        pygame.draw.circle(screen, color, position, radius)
        if num_spikes <= 0:
            return
        center_x, center_y = position
        spike_length = radius * (0.6 if is_carnivore else 0.4)
        for i in range(num_spikes):
            angle = (2 * math.pi * i) / num_spikes
            start_x = center_x + radius * math.cos(angle)
            start_y = center_y + radius * math.sin(angle)
            end_x = center_x + (radius + spike_length) * math.cos(angle)
            end_y = center_y + (radius + spike_length) * math.sin(angle)
            pygame.draw.line(
                screen, color, (start_x, start_y), (end_x, end_y), 2
            )
            if is_carnivore:
                pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 3)

    def get_species_visual(
        self, species_id, is_carnivore, radius, num_spikes, spike_length, num_nodes=10
    ):
        """Return a cached surface for scoreboard species cards."""
        cache_key = (
            f"{species_id}_{is_carnivore}_{radius}_{num_spikes}_{spike_length}"
        )
        if cache_key not in self.species_surfaces:
            surface_size = max(20, int(radius * 4))
            surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
            color = self.get_species_color(species_id, is_carnivore)
            center = (surface_size // 2, surface_size // 2)
            self.draw_organism_with_spikes(
                surface,
                center,
                color,
                radius,
                num_spikes,
                num_nodes,
                is_carnivore,
            )
            self.species_surfaces[cache_key] = surface
        return self.species_surfaces[cache_key]

    def _render_scoreboard(self):
        """Draw the in-window species leaderboard."""
        from scoreboard import Scoreboard

        pygame.draw.rect(
            self.screen, self.colors["background"], self.scoreboard_rect
        )
        top_species = Scoreboard.get_top_species(10)
        if not top_species:
            if "waiting" not in self.text_surfaces:
                self.text_surfaces["waiting"] = self.font.render(
                    "Waiting for species data...", True, self.colors["text"]
                )
            text = self.text_surfaces["waiting"]
            text_pos = text.get_rect(
                centerx=self.scoreboard_rect.centerx,
                centery=self.scoreboard_rect.centery,
            )
            self.screen.blit(text, text_pos)
            return

        if "header" not in self.text_surfaces:
            self.text_surfaces["header"] = self.header_font.render(
                "Top Species", True, self.colors["text"]
            )
        self.screen.blit(
            self.text_surfaces["header"], (self.scoreboard_rect.x + 20, 5)
        )

        y_offset = 50
        for species_id, record in top_species[:8]:
            card_height = 90
            card_rect = pygame.Rect(
                self.scoreboard_rect.x + 10,
                y_offset,
                self.scoreboard_width - 20,
                card_height,
            )
            pygame.draw.rect(
                self.screen, self.colors["card"], card_rect, border_radius=5
            )
            pygame.draw.rect(
                self.screen,
                self.colors["border"],
                card_rect,
                width=1,
                border_radius=5,
            )
            radius = record.get("size", 12) / 2
            num_spikes = record.get("num_spikes", 3)
            is_carnivore = record.get("is_carnivore", False)
            species_visual = self.get_species_visual(
                species_id,
                is_carnivore,
                radius,
                num_spikes,
                record.get("spike_length", 3),
                num_nodes=int(record.get("size", 12)),
            )
            visual_pos = (
                card_rect.x + 25 - species_visual.get_width() // 2,
                card_rect.y + card_height // 2 - species_visual.get_height() // 2,
            )
            self.screen.blit(species_visual, visual_pos)

            name = record["scientific_name"] or f"Species {species_id}"
            name_color = self.get_species_color(species_id, is_carnivore)
            text_key = f"name_{name}"
            if text_key not in self.text_surfaces:
                self.text_surfaces[text_key] = self.font.render(
                    name, True, name_color
                )
            self.screen.blit(
                self.text_surfaces[text_key], (card_rect.x + 50, card_rect.y + 15)
            )

            fitness_text = f"Fitness: {int(record['highest_fitness']):,}"
            text_key = f"fitness_{fitness_text}"
            if text_key not in self.text_surfaces:
                self.text_surfaces[text_key] = self.font.render(
                    fitness_text, True, self.colors["text"]
                )
            self.screen.blit(
                self.text_surfaces[text_key], (card_rect.x + 50, card_rect.y + 40)
            )

            gen_text = f"Gen {record['first_seen']} → {record['last_seen']}"
            text_key = f"gen_{gen_text}"
            if text_key not in self.text_surfaces:
                self.text_surfaces[text_key] = self.font.render(
                    gen_text, True, self.colors["text"]
                )
            self.screen.blit(
                self.text_surfaces[text_key], (card_rect.x + 50, card_rect.y + 65)
            )
            y_offset += card_height + 10

        if len(self.text_surfaces) > 100:
            essential_keys = ["header", "waiting"]
            self.text_surfaces = {
                key: value
                for key, value in self.text_surfaces.items()
                if key in essential_keys
            }
