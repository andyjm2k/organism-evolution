"""GPU smoke tests for ModernGLRenderer (skip when OpenGL is unavailable)."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gl_support import opengl_available, skip_reason

try:
    from renderer_gl import ModernGLRenderer
except ImportError:
    ModernGLRenderer = None


@unittest.skipUnless(
    opengl_available() and ModernGLRenderer is not None,
    skip_reason() or "moderngl not installed",
)
class TestModernGLRendererSmoke(unittest.TestCase):
    """Live OpenGL smoke tests; skipped in headless CI without libGL."""

    def setUp(self):
        """Create a small GPU renderer for each test."""
        self.renderer = ModernGLRenderer(200, logging_level="normal")

    def tearDown(self):
        """Release GL resources after each test."""
        if getattr(self, "renderer", None) is not None:
            self.renderer.cleanup_resources()
            try:
                import pygame

                if pygame.display.get_init():
                    pygame.display.quit()
                if pygame.get_init():
                    pygame.quit()
            except Exception:
                pass

    def test_init_creates_moderngl_context(self):
        """ModernGLRenderer should expose a valid moderngl context."""
        self.assertIsNotNone(self.renderer.ctx)
        version = self.renderer.ctx.info.get("GL_VERSION")
        self.assertTrue(version)

    def test_render_single_frame_with_stubs(self):
        """render() should draw one frame without error for stub entities."""
        food = SimpleNamespace(position=(50.0, 50.0))
        organism = SimpleNamespace(
            position=(100.0, 100.0),
            energy=100.0,
            species_id=1,
            is_carnivore=False,
            get_radius=lambda: 8.0,
        )
        result = self.renderer.render([organism], [food])
        self.assertTrue(result)

    def test_render_with_empty_lists(self):
        """render() should succeed with no organisms or food."""
        self.assertTrue(self.renderer.render([], []))

    def test_cleanup_releases_textures(self):
        """Full cleanup should drop scoreboard/HUD GL textures."""
        self.renderer.render([], [])
        self.renderer.cleanup_resources(light=False)
        self.assertIsNone(self.renderer._scoreboard_texture)
        self.assertIsNone(self.renderer._hud_texture)

    def test_texture_writes_do_not_flip(self):
        """Verify textures are written without vertical flipping."""
        # Check renderer_gl.py source code for correct flip parameter
        source_file = Path(__file__).resolve().parents[1] / "src" / "renderer_gl.py"
        with open(source_file, 'r') as f:
            content = f.read()
        
        # All texture writes should use False for the flip parameter
        # to avoid double-flipping with OpenGL coordinate system
        self.assertIn('pygame.image.tostring(overlay, "RGBA", False)', content,
                     "Overlay texture should not be flipped")
        self.assertIn('pygame.image.tostring(surface, "RGBA", False)', content,
                     "Scoreboard texture should not be flipped")
        self.assertIn('pygame.image.tostring(hud_surface, "RGBA", False)', content,
                     "HUD texture should not be flipped")
        
        # Ensure we're not using True anywhere for texture writes
        self.assertNotIn('pygame.image.tostring(overlay, "RGBA", True)', content,
                        "Should not flip overlay texture")
        self.assertNotIn('pygame.image.tostring(surface, "RGBA", True)', content,
                        "Should not flip scoreboard texture")
        self.assertNotIn('pygame.image.tostring(hud_surface, "RGBA", True)', content,
                        "Should not flip HUD texture")


class TestModernGLRendererHeadlessSkip(unittest.TestCase):
    """Verify skip helper behaves predictably on this host."""

    def test_skip_reason_is_string_when_unavailable(self):
        """When GL is missing, skip_reason explains why tests are skipped."""
        if opengl_available():
            self.assertIsNone(skip_reason())
        else:
            self.assertIn("OpenGL", skip_reason())


if __name__ == "__main__":
    unittest.main()
