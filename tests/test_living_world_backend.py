"""Renderer selection for the living-world harness."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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


if __name__ == "__main__":
    unittest.main()
