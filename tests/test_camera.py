"""Unit tests for Camera viewport transforms and panning."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera import Camera


class TestCamera(unittest.TestCase):
    """Cover viewport pan, drag, tracking, and coordinate transforms."""

    def test_pan_and_world_to_screen_roundtrip(self):
        camera = Camera(1000, 1000, 200, 200)
        camera.pan(50, 30)
        sx, sy = camera.world_to_screen(100, 100)
        wx, wy = camera.screen_to_world(sx, sy)
        self.assertAlmostEqual(wx, 100, places=3)
        self.assertAlmostEqual(wy, 100, places=3)

    def test_center_on_clamps_to_world(self):
        camera = Camera(1000, 1000, 200, 200)
        camera.center_on(999, 999)
        self.assertLessEqual(camera.offset_x, 800)
        self.assertLessEqual(camera.offset_y, 800)

    def test_viewport_rect_world_matches_offset(self):
        camera = Camera(800, 600, 200, 150)
        camera.pan(100, 50)
        x0, y0, x1, y1 = camera.viewport_rect_world()
        self.assertAlmostEqual(x1 - x0, 200)
        self.assertAlmostEqual(y1 - y0, 150)

    def test_drag_updates_offset(self):
        camera = Camera(1000, 1000, 200, 200)
        camera.begin_drag((100, 100))
        camera.drag_to((150, 130))
        self.assertTrue(camera.is_dragging())
        camera.end_drag()
        self.assertFalse(camera.is_dragging())

    def test_drag_clears_track_target(self):
        camera = Camera(1000, 1000, 200, 200)
        camera.set_track_target((500, 500))
        camera.begin_drag((10, 10))
        camera.update_tracking()
        camera.end_drag()

    def test_smooth_center_on_moves_toward_target(self):
        camera = Camera(1000, 1000, 200, 200, smoothing=0.5)
        start_x = camera.offset_x
        camera.center_on(800, 800, smooth=True)
        self.assertNotEqual(camera.offset_x, start_x)
        self.assertLess(camera.offset_x, 700)

    def test_clamp_offset_keeps_viewport_inside_world(self):
        camera = Camera(500, 500, 200, 200)
        camera.offset_x = -50
        camera.offset_y = 600
        camera.clamp_offset()
        self.assertGreaterEqual(camera.offset_x, 0)
        self.assertLessEqual(camera.offset_y, 300)

    def test_zoom_at_keeps_cursor_world_point_fixed(self):
        camera = Camera(4000, 4000, 800, 600)
        camera.center_on(2000, 2000)
        before = camera.screen_to_world(400, 300)
        camera.zoom_at(400, 300, 1)
        self.assertGreater(camera.zoom, 1.0)
        after = camera.screen_to_world(400, 300)
        self.assertAlmostEqual(after[0], before[0], places=3)
        self.assertAlmostEqual(after[1], before[1], places=3)

    def test_zoom_clamps_to_min_max(self):
        camera = Camera(4000, 4000, 800, 600)
        for _ in range(30):
            camera.zoom_at(400, 300, 1)
        self.assertAlmostEqual(camera.zoom, camera.max_zoom, places=6)
        for _ in range(60):
            camera.zoom_at(400, 300, -1)
        self.assertAlmostEqual(camera.zoom, camera.min_zoom, places=6)

    def test_world_to_screen_scales_with_zoom(self):
        camera = Camera(4000, 4000, 800, 600)
        camera.zoom_at(0, 0, 1)
        sx, sy = camera.world_to_screen(100, 100)
        self.assertAlmostEqual(sx, 100 * camera.zoom, places=3)
        wx, wy = camera.screen_to_world(sx, sy)
        self.assertAlmostEqual(wx, 100, places=3)
        self.assertAlmostEqual(wy, 100, places=3)

    def test_viewport_rect_shrinks_when_zoomed(self):
        camera = Camera(4000, 4000, 800, 600)
        camera.zoom_at(0, 0, 1)
        x0, y0, x1, y1 = camera.viewport_rect_world()
        self.assertAlmostEqual(x1 - x0, 800 / camera.zoom, places=3)
        self.assertAlmostEqual(y1 - y0, 600 / camera.zoom, places=3)

    def test_drag_scales_with_zoom(self):
        camera = Camera(4000, 4000, 800, 600)
        camera.center_on(2000, 2000)
        camera.zoom_at(400, 300, 1)
        start_x = camera.offset_x
        camera.begin_drag((400, 300))
        camera.drag_to((500, 300))
        self.assertAlmostEqual(camera.offset_x, start_x - 100 / camera.zoom, places=3)


if __name__ == "__main__":
    unittest.main()
