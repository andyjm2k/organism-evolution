"""Renderer selection for the living-world harness."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gl_support import opengl_available
from living_world import living_world_renderer_for


class TestLivingWorldBackend(unittest.TestCase):
    """The living world can pick pygame or the GPU renderer by name."""

    def test_pygame_is_the_software_renderer(self):
        """The default backend stays the pygame viewport."""
        renderer_cls = living_world_renderer_for("pygame")
        self.assertEqual(renderer_cls.__name__, "LivingWorldRenderer")

    def test_moderngl_is_the_gpu_renderer(self):
        """render_backend=moderngl selects the instanced living-world renderer."""
        try:
            renderer_cls = living_world_renderer_for("moderngl")
        except ImportError:
            self.skipTest("moderngl is not installed")
        self.assertEqual(renderer_cls.__name__, "LivingWorldGLRenderer")
        self.assertTrue(issubclass(renderer_cls, living_world_renderer_for("pygame")))

    @unittest.skipUnless(opengl_available(), "OpenGL display/context not available")
    def test_gpu_frame_draws_a_body(self):
        """One instanced body must change the framebuffer, not leave it blank."""
        import pygame

        from living_world_gl import LivingWorldGLRenderer

        renderer = LivingWorldGLRenderer(
            {
                "viewport_width": 320,
                "viewport_height": 240,
                "environment_width": 800,
                "environment_height": 600,
                "minimap_size": 64,
                "camera_track_smoothing": 1.0,
            },
            "normal",
        )
        try:
            organism = type("Organism", (), {})()
            organism.position = (40.0, 40.0)
            organism.energy = 10.0
            organism.species_id = 1
            organism.is_carnivore = False
            organism.body_genes = None
            organism.genome = None
            organism.movement_trail = []
            organism.get_radius = lambda: 12.0
            food = type("Food", (), {"position": None, "regrow_in": 0, "strain": None})()

            class Ecology:
                clouds = []

                def active_food_count(self):
                    return 0

                def scarcity_ratio(self):
                    return 1.0

            renderer.track_selected = False
            renderer.selected_organism = None
            renderer.render([organism], [food], [], Ecology())
            width, height = renderer.screen.get_size()
            # GL origin is the bottom. The body sits near the top-left of the viewport.
            sample = renderer.ctx.screen.read(
                viewport=(30, height - 50, 24, 24),
                components=3,
            )
            background = (18, 22, 28)
            self.assertTrue(
                any(abs(sample[index] - background[index % 3]) > 12 for index in range(len(sample)))
            )
        finally:
            renderer.cleanup_resources()
            pygame.display.quit()


if __name__ == "__main__":
    unittest.main()
