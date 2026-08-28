"""Shared renderer helpers: species colors and scoreboard surface building."""

import colorsys
import os
import random

import pygame


class RendererCommon:
    """Mixin-style helpers shared by pygame and ModernGL render backends."""

    def __init__(self, logging_level="normal"):
        # Verbosity for optional renderer diagnostics.
        self.logging_level = logging_level
        self.scoreboard_width = 300
        self.generation = 0
        # Stable per-species RGB colors.
        self.species_colors = {}
        # Cached pygame text/sprite surfaces.
        self.species_surfaces = {}
        self.text_surfaces = {}
        self.last_cache_clear = 0
        self._init_fonts()

    def _init_fonts(self):
        """Load project fonts or fall back to pygame defaults."""
        pygame.font.init()
        font_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts")
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

    def species_color_normalized(self, species_id, is_carnivore):
        """Return species RGB in 0..1 for GPU shaders."""
        red, green, blue = self.get_species_color(species_id, is_carnivore)
        return (red / 255.0, green / 255.0, blue / 255.0)

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
        """Clear renderer caches."""
        self.species_surfaces.clear()
        self.text_surfaces.clear()
        if light:
            return
        self.species_colors.clear()

    def build_scoreboard_surface(self, scoreboard_height):
        """Rasterize the scoreboard panel to an RGBA pygame surface."""
        from scoreboard import Scoreboard

        surface = pygame.Surface(
            (self.scoreboard_width, scoreboard_height), pygame.SRCALPHA
        )
        surface.fill(self.colors["background"])
        top_species = Scoreboard.get_top_species(10)
        if not top_species:
            if "waiting" not in self.text_surfaces:
                self.text_surfaces["waiting"] = self.font.render(
                    "Waiting for species data...", True, self.colors["text"]
                )
            text = self.text_surfaces["waiting"]
            text_pos = text.get_rect(
                centerx=self.scoreboard_width // 2,
                centery=scoreboard_height // 2,
            )
            surface.blit(text, text_pos)
            return surface

        if "header" not in self.text_surfaces:
            self.text_surfaces["header"] = self.header_font.render(
                "Top Species", True, self.colors["text"]
            )
        surface.blit(self.text_surfaces["header"], (20, 5))

        y_offset = 50
        for species_id, record in top_species[:8]:
            card_height = 90
            card_rect = pygame.Rect(10, y_offset, self.scoreboard_width - 20, card_height)
            pygame.draw.rect(surface, self.colors["card"], card_rect, border_radius=5)
            pygame.draw.rect(
                surface, self.colors["border"], card_rect, width=1, border_radius=5
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
            surface.blit(species_visual, visual_pos)

            name = record["scientific_name"] or f"Species {species_id}"
            name_color = self.get_species_color(species_id, is_carnivore)
            text_key = f"name_{name}"
            if text_key not in self.text_surfaces:
                self.text_surfaces[text_key] = self.font.render(
                    name, True, name_color
                )
            surface.blit(self.text_surfaces[text_key], (card_rect.x + 50, card_rect.y + 15))

            fitness_text = f"Fitness: {int(record['highest_fitness']):,}"
            text_key = f"fitness_{fitness_text}"
            if text_key not in self.text_surfaces:
                self.text_surfaces[text_key] = self.font.render(
                    fitness_text, True, self.colors["text"]
                )
            surface.blit(
                self.text_surfaces[text_key], (card_rect.x + 50, card_rect.y + 40)
            )

            gen_text = f"Gen {record['first_seen']} → {record['last_seen']}"
            text_key = f"gen_{gen_text}"
            if text_key not in self.text_surfaces:
                self.text_surfaces[text_key] = self.font.render(
                    gen_text, True, self.colors["text"]
                )
            surface.blit(
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
        return surface

    def get_species_visual(
        self, species_id, is_carnivore, radius, num_spikes, spike_length, num_nodes=10
    ):
        """Return a cached pygame surface for a species icon."""
        import math

        cache_key = (
            f"{species_id}_{is_carnivore}_{radius}_{num_spikes}_{spike_length}"
        )
        if cache_key not in self.species_surfaces:
            surface_size = max(20, int(radius * 4))
            surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
            color = self.get_species_color(species_id, is_carnivore)
            center = (surface_size // 2, surface_size // 2)
            draw_radius = min(20, max(5, radius + (num_nodes / 10)))
            pygame.draw.circle(surface, color, center, int(draw_radius))
            if num_spikes > 0:
                spike_len = draw_radius * (0.6 if is_carnivore else 0.4)
                for index in range(num_spikes):
                    angle = (2 * math.pi * index) / num_spikes
                    sx = center[0] + draw_radius * math.cos(angle)
                    sy = center[1] + draw_radius * math.sin(angle)
                    ex = center[0] + (draw_radius + spike_len) * math.cos(angle)
                    ey = center[1] + (draw_radius + spike_len) * math.sin(angle)
                    pygame.draw.line(surface, color, (sx, sy), (ex, ey), 2)
            self.species_surfaces[cache_key] = surface
        return self.species_surfaces[cache_key]
