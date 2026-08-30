"""ModernGL GPU renderer for arena entities (Phase 4)."""

import struct

import moderngl
import pygame
from pygame import DOUBLEBUF, OPENGL

from logging_util import log_detailed
from renderer_common import RendererCommon

# GLSL 330: instanced unit quad scaled to entity radius with circular fragment mask.
_CIRCLE_VERTEX_SHADER = """
#version 330 core
in vec2 in_corner;
in vec2 in_center;
in float in_radius;
in vec3 in_color;
uniform vec2 u_arena_size;
out vec3 v_color;
out vec2 v_local;
void main() {
    vec2 world = in_center + in_corner * in_radius;
    vec2 ndc = (world / u_arena_size) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = in_color;
    v_local = in_corner;
}
"""

_CIRCLE_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
in vec2 v_local;
out vec4 fragColor;
void main() {
    float dist = length(v_local);
    float alpha = 1.0 - smoothstep(0.92, 1.05, dist);
    if (alpha < 0.01) {
        discard;
    }
    fragColor = vec4(v_color, alpha);
}
"""

# GLSL 330: world-space textured quad for cached organism sprites.
_SPRITE_VERTEX_SHADER = """
#version 330 core
in vec2 in_corner;
in vec2 in_center;
in float in_half_size;
uniform vec2 u_arena_size;
out vec2 v_uv;
void main() {
    vec2 world = in_center + in_corner * in_half_size;
    vec2 ndc = (world / u_arena_size) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_corner * 0.5 + 0.5;
}
"""

_SPRITE_FRAGMENT_SHADER = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_texture;
out vec4 fragColor;
void main() {
    fragColor = texture(u_texture, v_uv);
}
"""

_PANEL_VERTEX_SHADER = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_PANEL_FRAGMENT_SHADER = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_texture;
out vec4 fragColor;
void main() {
    fragColor = texture(u_texture, v_uv);
}
"""


class ModernGLRenderer(RendererCommon):
    """GPU renderer with circle food, sprite organisms, and shared overlays."""

    def __init__(self, size, logging_level="normal"):
        super().__init__(logging_level=logging_level)
        if not pygame.get_init():
            pygame.init()
        self.arena_size = size
        self.scoreboard_height = size
        total_width = size + self.scoreboard_width
        self.screen = pygame.display.set_mode((total_width, size), OPENGL | DOUBLEBUF)
        pygame.display.set_caption("Evolution Simulation (GPU)")
        self.ctx = moderngl.create_context()
        self.clock = pygame.time.Clock()
        self._circle_prog = self.ctx.program(
            vertex_shader=_CIRCLE_VERTEX_SHADER,
            fragment_shader=_CIRCLE_FRAGMENT_SHADER,
        )
        self._sprite_prog = self.ctx.program(
            vertex_shader=_SPRITE_VERTEX_SHADER,
            fragment_shader=_SPRITE_FRAGMENT_SHADER,
        )
        self.panel_prog = self.ctx.program(
            vertex_shader=_PANEL_VERTEX_SHADER,
            fragment_shader=_PANEL_FRAGMENT_SHADER,
        )
        quad = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
        self._quad_vbo = self.ctx.buffer(struct.pack("12f", *quad))
        self._instance_vbo = self.ctx.buffer(reserve=500 * 6 * 4)
        self._circle_vao = self.ctx.vertex_array(
            self._circle_prog,
            [
                (self._quad_vbo, "2f", "in_corner"),
                (
                    self._instance_vbo,
                    "2f 1f 3f/i",
                    "in_center",
                    "in_radius",
                    "in_color",
                ),
            ],
        )
        self._sprite_instance_vbo = self.ctx.buffer(reserve=6 * 4)
        self._sprite_vao = self.ctx.vertex_array(
            self._sprite_prog,
            [
                (self._quad_vbo, "2f", "in_corner"),
                (
                    self._sprite_instance_vbo,
                    "2f 1f/i",
                    "in_center",
                    "in_half_size",
                ),
            ],
        )
        panel_verts = [
            -1.0, -1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.0,
            -1.0, -1.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 0.0,
        ]
        self._panel_vbo = self.ctx.buffer(struct.pack("24f", *panel_verts))
        self._panel_vao = self.ctx.vertex_array(
            self.panel_prog, [(self._panel_vbo, "2f 2f", "in_pos", "in_uv")]
        )
        self._scoreboard_texture = None
        self._hud_texture = None
        self._overlay_texture = None
        self._hud_key = None
        self._hud_surface = None
        self._gl_sprite_textures = {}
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        log_detailed(logging_level, f"ModernGL renderer initialized ({size}x{size})")

    def render(self, organisms, food_items):
        """Draw one GPU frame; events are handled in Simulation."""
        self.ctx.clear(1.0, 1.0, 1.0)
        self._draw_background_overlay(organisms)
        self._draw_food(food_items)
        self._draw_organisms(organisms)
        self._draw_selection_overlay(organisms)
        self._draw_scoreboard_panel()
        self._draw_hud(organisms, food_items)
        pygame.display.flip()
        self.clock.tick(60)
        return True

    def _draw_food(self, food_items):
        """Draw food pellets as instanced green circles."""
        if not food_items:
            return
        render_limit = 500
        food_color = (
            self.colors["food"][0] / 255.0,
            self.colors["food"][1] / 255.0,
            self.colors["food"][2] / 255.0,
        )
        instances = []
        for food in food_items[:render_limit]:
            if food.position is None:
                continue
            instances.extend(
                [
                    food.position[0],
                    food.position[1],
                    4.0,
                    food_color[0],
                    food_color[1],
                    food_color[2],
                ]
            )
        if not instances:
            return
        self._circle_prog["u_arena_size"].value = (
            float(self.arena_size),
            float(self.arena_size),
        )
        data = struct.pack(f"{len(instances)}f", *instances)
        if self._instance_vbo.size < len(data):
            self._instance_vbo.orphan(size=len(data))
        self._instance_vbo.write(data)
        self._circle_vao.render(moderngl.TRIANGLES, instances=len(instances) // 6)

    def _draw_organisms(self, organisms):
        """Draw organisms using cached species sprite textures."""
        if not organisms:
            return
        render_limit = 500
        self._sprite_prog["u_arena_size"].value = (
            float(self.arena_size),
            float(self.arena_size),
        )
        for organism in organisms[:render_limit]:
            if organism.position is None or organism.energy <= 0:
                continue
            species_id_str = str(organism.species_id)
            base_radius = float(organism.get_radius())
            num_spikes = int(getattr(organism, "num_spikes", 0))
            spike_length = float(getattr(organism, "spike_length", 3.0))
            num_nodes = int(getattr(organism, "get_active_node_count", lambda: 10)())
            visual = self.get_species_visual(
                species_id_str,
                organism.is_carnivore,
                base_radius,
                num_spikes,
                spike_length,
                num_nodes=num_nodes,
            )
            cache_key = self.organism_sprite_cache_key(
                species_id_str,
                organism.is_carnivore,
                base_radius,
                num_spikes,
                spike_length,
            )
            texture = self._get_sprite_texture(cache_key, visual)
            half_size = visual.get_width() / 2.0
            instance = struct.pack(
                "3f",
                organism.position[0],
                organism.position[1],
                half_size,
            )
            self._sprite_instance_vbo.write(instance)
            texture.use(location=0)
            self._sprite_vao.render(moderngl.TRIANGLES, instances=1)

    def _get_sprite_texture(self, cache_key, surface):
        """Upload and cache a pygame sprite surface as a GL texture."""
        surface = surface.convert_alpha()
        width, height = surface.get_size()
        cached = self._gl_sprite_textures.get(cache_key)
        if cached is None or cached.size != (width, height):
            if cached is not None:
                cached.release()
            texture = self.ctx.texture((width, height), 4)
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._gl_sprite_textures[cache_key] = texture
        texture = self._gl_sprite_textures[cache_key]
        texture.write(pygame.image.tostring(surface, "RGBA", False))
        return texture

    def _draw_background_overlay(self, organisms):
        """Rasterize breeding zone and movement trails beneath entities."""
        overlay = pygame.Surface((self.arena_size, self.arena_size), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 0))
        self.draw_breeding_zone(overlay, self.arena_size, self.arena_size)
        if organisms:
            render_limit = 500
            self.draw_movement_trails(overlay, organisms[:render_limit])
        self._upload_overlay_texture(overlay)
        width, height = overlay.get_size()
        self._blit_panel_texture(self._overlay_texture, 0, 0, width, height)

    def _draw_food(self, food_items):
        """Rasterize sense rings and selection highlight above organisms."""
        if not organisms or self.selected_organism is None:
            return
        overlay = pygame.Surface((self.arena_size, self.arena_size), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 0))
        if self.show_sense_rings:
            self.draw_sense_rings(overlay, self.selected_organism)
        self.draw_selection_highlight(overlay, self.selected_organism)
        self._upload_overlay_texture(overlay)
        width, height = overlay.get_size()
        self._blit_panel_texture(self._overlay_texture, 0, 0, width, height)

    def _upload_overlay_texture(self, overlay):
        """Write a pygame overlay surface to the reusable GL overlay texture."""
        width, height = overlay.get_size()
        if self._overlay_texture is None or self._overlay_texture.size != (width, height):
            if self._overlay_texture is not None:
                self._overlay_texture.release()
            self._overlay_texture = self.ctx.texture((width, height), 4)
            self._overlay_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._overlay_texture.write(pygame.image.tostring(overlay, "RGBA", False))

    def _draw_scoreboard_panel(self):
        """Upload cached scoreboard pygame surface as a GL texture."""
        surface = self.build_scoreboard_surface(self.scoreboard_height)
        surface = surface.convert_alpha()
        width, height = surface.get_size()
        if (
            self._scoreboard_texture is None
            or self._scoreboard_texture.size != (width, height)
        ):
            if self._scoreboard_texture is not None:
                self._scoreboard_texture.release()
            self._scoreboard_texture = self.ctx.texture(surface.get_size(), 4)
            self._scoreboard_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._scoreboard_texture.write(pygame.image.tostring(surface, "RGBA", False))
        self._blit_panel_texture(self._scoreboard_texture, self.arena_size, 0, width, height)

    def _draw_hud(self, organisms, food_items):
        """Draw cached HUD text in the top-left arena corner."""
        if not organisms:
            return
        species_count = self.count_active_species(organisms)
        food_count = len(food_items) if food_items else 0
        hud_key = (food_count, self.generation, species_count)
        if self._hud_key != hud_key:
            hud_surface = self.font.render(
                self.build_hud_text(organisms, food_items, self.generation),
                True,
                self.colors["text"],
            )
            hud_surface = hud_surface.convert_alpha()
            self._hud_surface = hud_surface
            self._hud_key = hud_key
        else:
            hud_surface = self._hud_surface
        width, height = hud_surface.get_size()
        if self._hud_texture is None or self._hud_texture.size != (width, height):
            if self._hud_texture is not None:
                self._hud_texture.release()
            self._hud_texture = self.ctx.texture(hud_surface.get_size(), 4)
        self._hud_texture.write(pygame.image.tostring(hud_surface, "RGBA", False))
        self._blit_panel_texture(self._hud_texture, 10, 10, width, height)

    def _blit_panel_texture(self, texture, x, y, width, height):
        """Draw a screen-space texture quad at pixel coordinates."""
        screen_w, screen_h = self.screen.get_size()
        left = (x / screen_w) * 2.0 - 1.0
        right = ((x + width) / screen_w) * 2.0 - 1.0
        top = 1.0 - (y / screen_h) * 2.0
        bottom = 1.0 - ((y + height) / screen_h) * 2.0
        panel_verts = [
            left, bottom, 0.0, 1.0,
            right, bottom, 1.0, 1.0,
            right, top, 1.0, 0.0,
            left, bottom, 0.0, 1.0,
            right, top, 1.0, 0.0,
            left, top, 0.0, 0.0,
        ]
        self._panel_vbo.write(struct.pack("24f", *panel_verts))
        texture.use(location=0)
        self._panel_vao.render(moderngl.TRIANGLES)

    def cleanup_resources(self, light=False):
        """Release GL textures and clear shared caches."""
        super().cleanup_resources(light=light)
        if light:
            self._release_sprite_textures()
            return
        self._release_sprite_textures()
        if self._scoreboard_texture is not None:
            self._scoreboard_texture.release()
            self._scoreboard_texture = None
        if self._hud_texture is not None:
            self._hud_texture.release()
            self._hud_texture = None
        if self._overlay_texture is not None:
            self._overlay_texture.release()
            self._overlay_texture = None

    def _release_sprite_textures(self):
        """Drop cached organism sprite GL textures."""
        for texture in self._gl_sprite_textures.values():
            texture.release()
        self._gl_sprite_textures.clear()
