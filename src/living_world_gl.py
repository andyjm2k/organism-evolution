"""GPU living-world renderer. Food and bodies are instanced; UI stays pygame."""

import struct

import moderngl
import pygame
from pygame import DOUBLEBUF, OPENGL

from living_world_renderer import LivingWorldRenderer
from logging_util import log_detailed
from plant_strain import pellet_color

# World position minus the camera offset, then into the full window.
# The scoreboard sits to the right of the viewport, so screen size includes it.
_CIRCLE_VERTEX_SHADER = """
#version 330 core
in vec2 in_corner;
in vec2 in_center;
in float in_radius;
in vec3 in_color;
uniform vec2 u_screen_size;
uniform vec2 u_camera;
out vec3 v_color;
out vec2 v_local;
void main() {
    // Radius has to be read here. Otherwise the linker drops the attribute
    // and the instanced buffer cannot be attached.
    vec2 screen = (in_center - u_camera) + in_corner * in_radius;
    vec2 ndc = (screen / u_screen_size) * 2.0 - 1.0;
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


class LivingWorldGLRenderer(LivingWorldRenderer):
    """Living-world view that draws pellets and bodies as GPU instances."""

    def _create_screen(self, total_width):
        """Open an OpenGL window. The GPU context is created after this."""
        self.screen = pygame.display.set_mode(
            (total_width, self.viewport_height),
            OPENGL | DOUBLEBUF,
        )

    def __init__(self, sim_config, logging_level="normal"):
        super().__init__(sim_config, logging_level=logging_level)
        pygame.display.set_caption("Living World Evolution (GPU)")
        self.ctx = moderngl.create_context()
        self._circle_prog = self.ctx.program(
            vertex_shader=_CIRCLE_VERTEX_SHADER,
            fragment_shader=_CIRCLE_FRAGMENT_SHADER,
        )
        self.panel_prog = self.ctx.program(
            vertex_shader=_PANEL_VERTEX_SHADER,
            fragment_shader=_PANEL_FRAGMENT_SHADER,
        )
        quad = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
        self._quad_vbo = self.ctx.buffer(struct.pack("12f", *quad))
        # Room for a few thousand pellets before the buffer grows.
        self._instance_vbo = self.ctx.buffer(reserve=4096 * 6 * 4)
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
        self._textures = {}
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        log_detailed(
            logging_level,
            f"Living world GPU renderer: viewport {self.viewport_width}x"
            f"{self.viewport_height}",
        )

    def render(self, organisms, food_items, nutrient_clouds, food_ecology):
        """Draw one GPU frame of the living world."""
        self.camera.update_tracking()
        if (
            self.track_selected
            and self.selected_organism is not None
            and self.selected_organism.energy > 0
            and self.selected_organism.position is not None
        ):
            self.camera.set_track_target(self.selected_organism.position)
        self.ctx.clear(18 / 255.0, 22 / 255.0, 28 / 255.0, 1.0)
        self._draw_world_overlay(organisms, nutrient_clouds)
        self._draw_instances(self._collect_instances(organisms, food_items))
        self._draw_ui_overlay(organisms, food_items, nutrient_clouds, food_ecology)
        self._draw_scoreboard()
        pygame.display.flip()
        self.clock.tick(60)
        return True

    def _draw_world_overlay(self, organisms, nutrient_clouds):
        """Clouds and trails stay on a pygame surface behind the instances."""
        overlay = pygame.Surface(
            (self.viewport_width, self.viewport_height), pygame.SRCALPHA
        )
        overlay.fill((18, 22, 28, 255))
        self._draw_nutrient_clouds(overlay, nutrient_clouds)
        if organisms:
            self._draw_trails(overlay, organisms)
        self._blit_surface("world", overlay, 0, 0)

    def _collect_instances(self, organisms, food_items):
        """Pack on-screen pellets, eggs, and bodies into one instance list."""
        instances = []
        for food in food_items:
            if food.position is None:
                continue
            color = pellet_color(food)
            radius = 2.0 if getattr(food, "regrow_in", 0) > 0 else 3.5
            self._append_circle(instances, food.position, radius, color)
        for egg in getattr(self, "_eggs_to_draw", ()) or ():
            if getattr(egg, "position", None) is None:
                continue
            self._append_circle(instances, egg.position, 4.5, (230, 210, 160))
        selected = self.selected_organism
        for organism in organisms:
            if organism is selected or organism.position is None or organism.energy <= 0:
                continue
            color = self._organism_color(organism)
            self._append_circle(
                instances, organism.position, float(organism.get_radius()), color
            )
        return instances

    def _append_circle(self, instances, position, radius, color):
        """Add one circle if it overlaps the current viewport."""
        margin = radius + 8.0
        sx = position[0] - self.camera.offset_x
        sy = position[1] - self.camera.offset_y
        if sx < -margin or sy < -margin:
            return
        if sx > self.viewport_width + margin or sy > self.viewport_height + margin:
            return
        instances.extend(
            [
                float(position[0]),
                float(position[1]),
                float(radius),
                color[0] / 255.0,
                color[1] / 255.0,
                color[2] / 255.0,
            ]
        )

    def _draw_instances(self, instances):
        """Upload every circle and draw them in one instanced call."""
        if not instances:
            return
        screen_w, screen_h = self.screen.get_size()
        self._circle_prog["u_screen_size"].value = (float(screen_w), float(screen_h))
        self._circle_prog["u_camera"].value = (
            float(self.camera.offset_x),
            float(self.camera.offset_y),
        )
        data = struct.pack(f"{len(instances)}f", *instances)
        if self._instance_vbo.size < len(data):
            self._instance_vbo.orphan(size=len(data))
        self._instance_vbo.write(data)
        self._circle_vao.render(moderngl.TRIANGLES, instances=len(instances) // 6)

    def _draw_ui_overlay(self, organisms, food_items, nutrient_clouds, food_ecology):
        """HUD, minimap, selection, and the tracked body with its organs."""
        ui = pygame.Surface(
            (self.viewport_width, self.viewport_height), pygame.SRCALPHA
        )
        if self.selected_organism is not None and self.selected_organism.energy > 0:
            self._draw_organism(ui, self.selected_organism)
            self._draw_selection(ui, self.selected_organism)
            if self.show_sense_rings:
                self._draw_sense_rings_world(ui, self.selected_organism)
            self._draw_organism_panel(ui, self.selected_organism)
        self._draw_hud(ui, organisms, food_ecology)
        self._draw_minimap(ui, organisms, food_items, nutrient_clouds)
        self._blit_surface("ui", ui, 0, 0)

    def _draw_scoreboard(self):
        """Scoreboard stays a pygame surface on the right of the viewport."""
        surface = self.build_scoreboard_surface(self.viewport_height)
        self._blit_surface("scoreboard", surface, self.viewport_width, 0)

    def _blit_surface(self, name, surface, x, y):
        """Upload one pygame surface and draw it in screen pixels."""
        surface = surface.convert_alpha()
        width, height = surface.get_size()
        texture = self._textures.get(name)
        if texture is None or texture.size != (width, height):
            if texture is not None:
                texture.release()
            texture = self.ctx.texture((width, height), 4)
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._textures[name] = texture
        texture.write(pygame.image.tostring(surface, "RGBA", False))
        self._blit_texture(texture, x, y, width, height)

    def _blit_texture(self, texture, x, y, width, height):
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
        """Release uploaded textures. The window itself stays open."""
        super().cleanup_resources(light=light)
        for texture in self._textures.values():
            texture.release()
        self._textures.clear()
