"""Pygame renderer for the living-world harness with camera, minimap, and tracking."""

import pygame

from camera import Camera
from genome_viz import build_genome_surface
from logging_util import log_detailed
from renderer_common import RendererCommon


class LivingWorldRenderer(RendererCommon):
    """Large-world viewport with pan, minimap, and organism inspection panel."""

    def __init__(self, sim_config, logging_level="normal"):
        super().__init__(logging_level=logging_level)
        if not pygame.get_init():
            pygame.init()
        self.viewport_width = int(sim_config.get("viewport_width", 900))
        self.viewport_height = int(sim_config.get("viewport_height", 900))
        self.world_width = int(sim_config["environment_width"])
        self.world_height = int(sim_config["environment_height"])
        self.minimap_size = int(sim_config.get("minimap_size", 160))
        total_width = self.viewport_width + self.scoreboard_width
        if not pygame.display.get_init():
            self.screen = pygame.display.set_mode((total_width, self.viewport_height))
        else:
            self.screen = pygame.display.get_surface()
            if self.screen is None:
                self.screen = pygame.display.set_mode((total_width, self.viewport_height))
        pygame.display.set_caption("Living World Evolution")
        self.scoreboard_rect = pygame.Rect(
            self.viewport_width, 0, self.scoreboard_width, self.viewport_height
        )
        self.clock = pygame.time.Clock()
        self.camera = Camera(
            self.world_width,
            self.world_height,
            self.viewport_width,
            self.viewport_height,
            smoothing=float(sim_config.get("camera_track_smoothing", 0.15)),
        )
        self.track_selected = True
        self.world_step = 0
        self._hud_key = None
        self._hud_surface = None
        self._panel_surfaces = {}
        log_detailed(
            logging_level,
            f"Living world renderer: world {self.world_width}x{self.world_height}, "
            f"viewport {self.viewport_width}x{self.viewport_height}",
        )

    def set_world_step(self, step):
        """Update HUD world step counter."""
        self.world_step = step

    def handle_event(self, event, organisms):
        """Process mouse events for pan, select, and track."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            if self._in_viewport(pos):
                if self._point_in_minimap(pos):
                    self._minimap_jump(pos)
                else:
                    world_pos = self._screen_to_world_viewport(pos)
                    picked = self._pick_organism(world_pos, organisms)
                    if picked is not None:
                        self.set_selected_organism(picked)
                        self.track_selected = True
                        self.camera.set_track_target(picked.position)
                    else:
                        self.camera.begin_drag(pos)
            elif event.button == 1:
                pass
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.camera.end_drag()
        elif event.type == pygame.MOUSEMOTION and self.camera.is_dragging():
            self.camera.drag_to(event.pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                self.toggle_sense_rings()
            elif event.key == pygame.K_ESCAPE:
                self.set_selected_organism(None)
                self.track_selected = False
                self.camera.clear_track()
            elif event.key == pygame.K_t:
                self.track_selected = not self.track_selected
                if not self.track_selected:
                    self.camera.clear_track()

    def render(self, organisms, food_items, nutrient_clouds, food_ecology):
        """Draw one living-world frame."""
        self.camera.update_tracking()
        if (
            self.track_selected
            and self.selected_organism is not None
            and self.selected_organism.energy > 0
            and self.selected_organism.position is not None
        ):
            self.camera.set_track_target(self.selected_organism.position)

        viewport = pygame.Surface((self.viewport_width, self.viewport_height))
        viewport.fill((18, 22, 28))
        self._draw_nutrient_clouds(viewport, nutrient_clouds)
        self._draw_food(viewport, food_items)
        if organisms:
            self._draw_trails(viewport, organisms)
            if self.selected_organism and self.show_sense_rings:
                self._draw_sense_rings_world(viewport, self.selected_organism)
            for organism in organisms[:800]:
                self._draw_organism(viewport, organism)
            if self.selected_organism is not None:
                self._draw_selection(viewport, self.selected_organism)
        self._draw_hud(viewport, organisms, food_ecology)
        self._draw_minimap(viewport, organisms, food_items, nutrient_clouds)
        if self.selected_organism is not None:
            self._draw_organism_panel(viewport, self.selected_organism)

        self.screen.fill((255, 255, 255))
        self.screen.blit(viewport, (0, 0))
        scoreboard_surface = self.build_scoreboard_surface(self.viewport_height)
        self.screen.blit(scoreboard_surface, self.scoreboard_rect.topleft)
        pygame.display.flip()
        self.clock.tick(60)
        return True

    def _in_viewport(self, pos):
        """True when a screen point lies inside the world viewport."""
        x, y = pos
        return 0 <= x < self.viewport_width and 0 <= y < self.viewport_height

    def _screen_to_world_viewport(self, screen_pos):
        """Map viewport screen coordinates to world coordinates."""
        return self.camera.screen_to_world(screen_pos[0], screen_pos[1])

    def _world_to_viewport(self, world_x, world_y):
        """Map world coordinates to viewport pixel coordinates."""
        return self.camera.world_to_screen(world_x, world_y)

    def _minimap_rect(self):
        """Return minimap rectangle within the viewport."""
        pad = 10
        size = self.minimap_size
        return pygame.Rect(
            self.viewport_width - size - pad,
            pad,
            size,
            size,
        )

    def _point_in_minimap(self, pos):
        """True when a click targets the minimap."""
        return self._minimap_rect().collidepoint(pos)

    def _minimap_jump(self, screen_pos):
        """Center the camera on a minimap click location."""
        rect = self._minimap_rect()
        rel_x = (screen_pos[0] - rect.x) / rect.width
        rel_y = (screen_pos[1] - rect.y) / rect.height
        world_x = rel_x * self.world_width
        world_y = rel_y * self.world_height
        self.camera.center_on(world_x, world_y)
        self.track_selected = False
        self.camera.clear_track()

    def _pick_organism(self, world_pos, organisms):
        """Return the organism closest to a world click."""
        best = None
        best_dist = float("inf")
        for organism in organisms:
            if organism.position is None or organism.energy <= 0:
                continue
            dx = world_pos[0] - organism.position[0]
            dy = world_pos[1] - organism.position[1]
            hit = max(12, organism.get_radius() + 8)
            dist_sq = dx * dx + dy * dy
            if dist_sq <= hit * hit and dist_sq < best_dist:
                best_dist = dist_sq
                best = organism
        return best

    def _draw_nutrient_clouds(self, surface, clouds):
        """Draw semi-transparent nutrient cloud fields."""
        for cloud in clouds:
            sx, sy = self._world_to_viewport(cloud.position[0], cloud.position[1])
            if not self._on_screen(sx, sy, cloud.radius):
                continue
            radius = int(cloud.radius)
            alpha = int(40 + 60 * cloud.intensity)
            cloud_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            color = (60, 200, 90, alpha)
            pygame.draw.circle(cloud_surface, color, (radius, radius), radius)
            inner = (120, 240, 140, alpha // 2)
            pygame.draw.circle(
                cloud_surface, inner, (radius, radius), int(radius * 0.55)
            )
            surface.blit(cloud_surface, (sx - radius, sy - radius))

    def _draw_food(self, surface, food_items):
        """Draw food pellets in viewport space."""
        for food in food_items:
            if food.position is None:
                continue
            sx, sy = self._world_to_viewport(food.position[0], food.position[1])
            if self._on_screen(sx, sy, 6):
                pygame.draw.circle(surface, (80, 220, 90), (sx, sy), 3)

    def _draw_trails(self, surface, organisms):
        """Draw movement trails with camera offset applied."""
        for organism in organisms:
            trail = getattr(organism, "movement_trail", None)
            if not trail or len(trail) < 2:
                continue
            color = self.get_species_color(
                str(organism.species_id), organism.is_carnivore
            )
            for index in range(1, len(trail)):
                a = int(50 + 150 * index / len(trail))
                c = (
                    min(255, color[0] * a // 255),
                    min(255, color[1] * a // 255),
                    min(255, color[2] * a // 255),
                )
                p0 = self._world_to_viewport(trail[index - 1][0], trail[index - 1][1])
                p1 = self._world_to_viewport(trail[index][0], trail[index][1])
                pygame.draw.line(surface, c, p0, p1, 2)

    def _draw_sense_rings_world(self, surface, organism):
        """Draw sense rings with world-to-screen transform."""
        if organism.position is None:
            return
        config = self.environment_config or {}
        sx, sy = self._world_to_viewport(organism.position[0], organism.position[1])
        rings = []
        if not organism.is_carnivore:
            rings.append(
                (
                    config.get("food_detection_radius", 200),
                    (80, 180, 100, 80),
                )
            )
        rings.append((config.get("threat_detection_radius", 100), (220, 90, 70, 80)))
        rings.append(
            (config.get("breeding_detection_radius", 150), (150, 110, 220, 80))
        )
        for radius, color in rings:
            if radius <= 0:
                continue
            ring = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(ring, color, (radius, radius), int(radius), 1)
            surface.blit(ring, (sx - radius, sy - radius))

    def _draw_organism(self, surface, organism):
        """Blit one organism sprite at its world position."""
        if organism.position is None or organism.energy <= 0:
            return
        sx, sy = self._world_to_viewport(organism.position[0], organism.position[1])
        if not self._on_screen(sx, sy, 30):
            return
        visual = self.get_species_visual(
            str(organism.species_id),
            organism.is_carnivore,
            organism.get_radius(),
            organism.calculate_spikes(),
            organism.calculate_spike_length(),
            num_nodes=organism.get_active_node_count(),
        )
        surface.blit(
            visual,
            (sx - visual.get_width() // 2, sy - visual.get_height() // 2),
        )

    def _draw_selection(self, surface, organism):
        """Highlight the tracked organism."""
        if organism.position is None:
            return
        sx, sy = self._world_to_viewport(organism.position[0], organism.position[1])
        pygame.draw.circle(
            surface,
            (255, 210, 60),
            (sx, sy),
            int(organism.get_radius()) + 8,
            2,
        )

    def _draw_hud(self, surface, organisms, food_ecology):
        """Draw top-left status text."""
        species_count = len({org.species_id for org in organisms}) if organisms else 0
        food_count = food_ecology.active_food_count()
        scarcity = food_ecology.scarcity_ratio()
        hud_key = (food_count, self.world_step, species_count, int(scarcity * 100))
        if self._hud_key != hud_key:
            track = "ON" if self.track_selected and self.selected_organism else "OFF"
            text = (
                f"Step {self.world_step} | Food {food_count} "
                f"({scarcity:.0%}) | Species {species_count} | Track {track}"
            )
            self._hud_surface = self.font.render(text, True, (220, 230, 240))
            self._hud_key = hud_key
        bg = pygame.Surface(
            (self._hud_surface.get_width() + 16, self._hud_surface.get_height() + 8),
            pygame.SRCALPHA,
        )
        bg.fill((0, 0, 0, 140))
        surface.blit(bg, (8, 8))
        surface.blit(self._hud_surface, (16, 12))

    def _draw_minimap(self, surface, organisms, food_items, clouds):
        """Draw world overview with viewport indicator in the top-right."""
        rect = self._minimap_rect()
        pygame.draw.rect(surface, (10, 14, 18), rect, border_radius=4)
        pygame.draw.rect(surface, (70, 90, 110), rect, 1, border_radius=4)
        scale_x = rect.width / self.world_width
        scale_y = rect.height / self.world_height

        def to_mini(wx, wy):
            return (
                int(rect.x + wx * scale_x),
                int(rect.y + wy * scale_y),
            )

        for cloud in clouds:
            mx, my = to_mini(cloud.position[0], cloud.position[1])
            r = max(2, int(cloud.radius * scale_x * 0.5))
            pygame.draw.circle(surface, (40, 100, 50), (mx, my), r)
        for food in food_items[:400]:
            if food.position is None:
                continue
            fx, fy = to_mini(food.position[0], food.position[1])
            surface.set_at((fx, fy), (80, 200, 90))
        for organism in organisms[:300]:
            if organism.position is None or organism.energy <= 0:
                continue
            ox, oy = to_mini(organism.position[0], organism.position[1])
            color = self.get_species_color(
                str(organism.species_id), organism.is_carnivore
            )
            pygame.draw.circle(surface, color, (ox, oy), 2)
        vx0, vy0, vx1, vy1 = self.camera.viewport_rect_world()
        mini_rect = pygame.Rect(
            int(rect.x + vx0 * scale_x),
            int(rect.y + vy0 * scale_y),
            max(4, int((vx1 - vx0) * scale_x)),
            max(4, int((vy1 - vy0) * scale_y)),
        )
        pygame.draw.rect(surface, (255, 220, 80), mini_rect, 2)
        label = self.font.render("Map", True, (180, 190, 200))
        surface.blit(label, (rect.x + 6, rect.y + 4))

    def _draw_organism_panel(self, surface, organism):
        """Draw stats and genome graph for the selected organism."""
        panel_w, panel_h = 280, 320
        panel = pygame.Rect(12, self.viewport_height - panel_h - 12, panel_w, panel_h)
        panel_surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surface.fill((20, 24, 32, 230))
        pygame.draw.rect(panel_surface, (90, 110, 130), panel_surface.get_rect(), 1, 8)
        diet = "Carnivore" if organism.is_carnivore else "Herbivore"
        name = getattr(organism, "scientific_name", None) or f"Species {organism.species_id}"
        lines = [
            name[:28],
            f"{diet} | Energy {organism.energy:.0f}/{organism.max_energy:.0f}",
            f"Age {organism.steps_taken} steps",
            f"Food {organism.food_consumed} | Hunts {organism.organisms_consumed}",
            f"Fitness {organism.highest_fitness:.0f}",
            f"Genome {getattr(organism, 'genome_id', '?')}",
        ]
        y = 12
        for index, line in enumerate(lines):
            color = (240, 245, 250) if index == 0 else (180, 190, 200)
            font = self.header_font if index == 0 else self.font
            text = font.render(line, True, color)
            panel_surface.blit(text, (14, y))
            y += 28 if index == 0 else 22
        genome_surface = build_genome_surface(
            organism.genome,
            organism.config,
            width=panel_w - 28,
            height=130,
            font=self.font,
        )
        panel_surface.blit(genome_surface, (14, y + 4))
        hint = self.font.render("Drag to pan | T track | Esc clear", True, (120, 130, 140))
        panel_surface.blit(hint, (14, panel_h - 24))
        surface.blit(panel_surface, panel.topleft)

    def _on_screen(self, sx, sy, margin):
        """Rough visibility test for viewport culling."""
        return (
            -margin <= sx <= self.viewport_width + margin
            and -margin <= sy <= self.viewport_height + margin
        )
