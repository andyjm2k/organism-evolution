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
    def test_gpu_frame_draws_organs_past_the_body(self):
        """A founding herbivore's jaw is visible beyond the cell, as in pygame."""
        import neat
        import pygame

        from genome_bootstrap import install_founding_motif
        from living_world_gl import LivingWorldGLRenderer

        root = Path(__file__).resolve().parents[1]
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(root / "config" / "neat-config.ini"),
        )
        neat.Population(config)
        genome = neat.DefaultGenome(1)
        genome.configure_new(config.genome_config)
        genome.fitness = 1.0
        install_founding_motif(genome, config.genome_config)
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
            organism.genome = genome
            organism.movement_trail = []
            # Cell radius 14. The jaw continues past that, to the right.
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
            _width, height = renderer.screen.get_size()
            # Outside the cell disk and on the jaw. GL y grows upward.
            sample = renderer.ctx.screen.read(
                viewport=(54, height - 48, 12, 16),
                components=3,
            )
            background = (18, 22, 28)
            self.assertTrue(
                any(
                    abs(sample[index] - background[index % 3]) > 20
                    for index in range(len(sample))
                )
            )
        finally:
            renderer.cleanup_resources()
            pygame.display.quit()


if __name__ == "__main__":
    unittest.main()
