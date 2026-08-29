"""Pygame software renderer for the simulation arena and scoreboard."""

import math

import pygame

from logging_util import log_detailed
from renderer_common import RendererCommon


class PygameRenderer(RendererCommon):
    """CPU pygame renderer (default backend)."""

    def __init__(self, size, logging_level="normal"):
        super().__init__(logging_level=logging_level)
        if not pygame.get_init():
            pygame.init()
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
        self._hud_key = None
        self._hud_surface = None
        log_detailed(logging_level, f"Pygame renderer initialized with screen size: {size}x{size}")

    def render(self, organisms, food_items):
        """Draw one simulation frame; events are pumped by Simulation."""
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
            for food in food_items[:render_limit]:
                if food.position is not None:
                    pos = (int(food.position[0]), int(food.position[1]))
                    pygame.draw.circle(self.screen, (0, 255, 0), pos, 4)

        if organisms:
            self.draw_movement_trails(self.screen, organisms[:render_limit])
            if self.selected_organism and self.show_sense_rings:
                self.draw_sense_rings(self.screen, self.selected_organism)
            for organism in organisms[:render_limit]:
                if organism.position is not None and organism.energy > 0:
                    pos = (int(organism.position[0]), int(organism.position[1]))
                    species_id_str = str(organism.species_id)
                    visual = self.get_species_visual(
                        species_id_str,
                        organism.is_carnivore,
                        organism.get_radius(),
                        organism.num_spikes,
                        organism.spike_length,
                        num_nodes=organism.get_active_node_count(),
                    )
                    self.screen.blit(
                        visual,
                        (
                            pos[0] - visual.get_width() // 2,
                            pos[1] - visual.get_height() // 2,
                        ),
                    )
            if self.selected_organism is not None:
                self.draw_selection_highlight(self.screen, self.selected_organism)

        if organisms:
            species_count = len(set(org.species_id for org in organisms))
            food_count = len(food_items) if food_items else 0
            hud_key = (food_count, self.generation, species_count)
            if self._hud_key != hud_key:
                debug_text = (
                    f"Food: {food_count} | Gen: {self.generation} | "
                    f"Species: {species_count}"
                )
                self._hud_surface = self.font.render(debug_text, True, self.colors["text"])
                self._hud_key = hud_key
            self.screen.blit(self._hud_surface, (10, 10))

        scoreboard_surface = self.build_scoreboard_surface(self.scoreboard_height)
        self.screen.blit(scoreboard_surface, self.scoreboard_rect.topleft)
        pygame.display.flip()
        self.clock.tick(60)
        return True

    def cleanup_resources(self, light=False):
        """Clear caches and optionally wipe the screen."""
        super().cleanup_resources(light=light)
        if light or self.screen is None:
            return
        self.screen.fill((255, 255, 255))

    def draw_organism_with_spikes(
        self, screen, position, color, base_radius, num_spikes, num_nodes, is_carnivore
    ):
        """Draw an organism body and optional spikes (legacy helper)."""
        radius = min(20, max(5, base_radius + (num_nodes / 10)))
        pygame.draw.circle(screen, color, position, radius)
        if num_spikes <= 0:
            return
        center_x, center_y = position
        spike_length = radius * (0.6 if is_carnivore else 0.4)
        for index in range(num_spikes):
            angle = (2 * math.pi * index) / num_spikes
            start_x = center_x + radius * math.cos(angle)
            start_y = center_y + radius * math.sin(angle)
            end_x = center_x + (radius + spike_length) * math.cos(angle)
            end_y = center_y + (radius + spike_length) * math.sin(angle)
            pygame.draw.line(
                screen, color, (start_x, start_y), (end_x, end_y), 2
            )
            if is_carnivore:
                pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 3)


# Backward-compatible alias used by older imports.
Renderer = PygameRenderer
