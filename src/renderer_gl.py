"""ModernGL GPU renderer for arena entities (Phase 4)."""

import struct

import moderngl
import pygame
from pygame import DOUBLEBUF, OPENGL

from logging_util import log_detailed
from renderer_common import RendererCommon

# GLSL 330: instanced unit quad scaled to organism radius / food point size.
_VERTEX_SHADER = """
#version 330 core
in vec2 in_corner;
in vec2 in_center;
in float in_radius;
in vec3 in_color;
uniform vec2 u_arena_size;
out vec3 v_color;
void main() {
    vec2 world = in_center + in_corner * in_radius;
    vec2 ndc = (world / u_arena_size) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = in_color;
}
"""

_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_color, 1.0);
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
    """GPU instanced draw for food/organisms; scoreboard via cached textures."""

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
        self.prog = self.ctx.program(
            vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER
        )
        self.panel_prog = self.ctx.program(
            vertex_shader=_PANEL_VERTEX_SHADER,
            fragment_shader=_PANEL_FRAGMENT_SHADER,
        )
        # Unit square corners for two triangles (instanced).
        quad = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
        self._quad_vbo = self.ctx.buffer(struct.pack("12f", *quad))
        self._instance_vbo = self.ctx.buffer(reserve=500 * 6 * 4)
        self._vao = self.ctx.vertex_array(
            self.prog,
            [
                (self._quad_vbo, "2f", "in_corner"),
                (self._instance_vbo, "2f 1f 3f/i", "in_center", "in_radius", "in_color"),
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
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        log_detailed(logging_level, f"ModernGL renderer initialized ({size}x{size})")

    def render(self, organisms, food_items):
        """Draw one GPU frame; events are handled in Simulation."""
        self.ctx.clear(1.0, 1.0, 1.0)
        self.prog["u_arena_size"].value = (float(self.arena_size), float(self.arena_size))
        instances = []
        render_limit = 500
        if food_items:
            for food in food_items[:render_limit]:
                if food.position is None:
                    continue
                instances.extend(
                    [
                        food.position[0],
                        food.position[1],
                        4.0,
                        0.0,
                        1.0,
                        0.0,
                    ]
                )
        if organisms:
            for organism in organisms[:render_limit]:
                if organism.position is None or organism.energy <= 0:
                    continue
                color = self.species_color_normalized(
                    str(organism.species_id), organism.is_carnivore
                )
                radius = float(organism.get_radius())
                instances.extend(
                    [
                        organism.position[0],
                        organism.position[1],
                        radius,
                        color[0],
                        color[1],
                        color[2],
                    ]
                )
        if instances:
            data = struct.pack(f"{len(instances)}f", *instances)
            if self._instance_vbo.size < len(data):
                self._instance_vbo.orphan(size=len(data))
            self._instance_vbo.write(data)
            self._vao.render(moderngl.TRIANGLES, instances=len(instances) // 6)

        self._draw_arena_overlays(organisms)
        self._draw_scoreboard_panel()
        self._draw_hud(organisms, food_items)
        pygame.display.flip()
        self.clock.tick(60)
        return True

    def _draw_arena_overlays(self, organisms):
        """Rasterize trails and selection overlays onto the GL arena."""
        if not organisms:
            return
        overlay = pygame.Surface((self.arena_size, self.arena_size), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 0))
        render_limit = 500
        self.draw_movement_trails(overlay, organisms[:render_limit])
        if self.selected_organism and self.show_sense_rings:
            self.draw_sense_rings(overlay, self.selected_organism)
        if self.selected_organism is not None:
            self.draw_selection_highlight(overlay, self.selected_organism)
        width, height = overlay.get_size()
        if self._overlay_texture is None or self._overlay_texture.size != (width, height):
            if self._overlay_texture is not None:
                self._overlay_texture.release()
            self._overlay_texture = self.ctx.texture((width, height), 4)
            self._overlay_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._overlay_texture.write(pygame.image.tostring(overlay, "RGBA", False))
        self._blit_panel_texture(self._overlay_texture, 0, 0, width, height)

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
        species_count = len(set(org.species_id for org in organisms))
        food_count = len(food_items) if food_items else 0
        hud_key = (food_count, self.generation, species_count)
        if self._hud_key != hud_key:
            debug_text = (
                f"Food: {food_count} | Gen: {self.generation} | "
                f"Species: {species_count}"
            )
            hud_surface = self.font.render(debug_text, True, self.colors["text"])
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
            return
        if self._scoreboard_texture is not None:
            self._scoreboard_texture.release()
            self._scoreboard_texture = None
        if self._hud_texture is not None:
            self._hud_texture.release()
            self._hud_texture = None
        if self._overlay_texture is not None:
            self._overlay_texture.release()
            self._overlay_texture = None
