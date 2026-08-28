"""Tests for renderer backend factory."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from renderer_factory import create_renderer


class TestRendererFactory(unittest.TestCase):
    """Factory selects backend and falls back when GPU init fails."""

    def test_moderngl_fallback_on_init_error(self):
        """moderngl backend falls back to pygame when GL init fails."""
        mock_gl = MagicMock(side_effect=RuntimeError("no gl"))
        mock_pygame = MagicMock(return_value="pygame-renderer")
        with patch("renderer_gl.ModernGLRenderer", mock_gl, create=True):
            with patch("renderer.PygameRenderer", mock_pygame):
                result = create_renderer(100, backend="moderngl")
        self.assertEqual(result, "pygame-renderer")
        mock_pygame.assert_called_once()

    def test_pygame_backend_requested(self):
        """Explicit pygame backend constructs PygameRenderer."""
        mock_pygame = MagicMock(return_value="pygame-renderer")
        with patch("renderer.PygameRenderer", mock_pygame):
            result = create_renderer(100, backend="pygame")
        self.assertEqual(result, "pygame-renderer")


if __name__ == "__main__":
    unittest.main()
