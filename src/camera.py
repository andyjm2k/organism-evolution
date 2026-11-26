"""Viewport camera for panning across a large world."""


class Camera:
    """Translate between world coordinates and screen viewport coordinates."""

    def __init__(
        self,
        world_width,
        world_height,
        viewport_width,
        viewport_height,
        smoothing=0.15,
        min_zoom=0.5,
        max_zoom=3.0,
    ):
        # Full simulation arena size.
        self.world_width = float(world_width)
        self.world_height = float(world_height)
        # Visible window size in pixels.
        self.viewport_width = float(viewport_width)
        self.viewport_height = float(viewport_height)
        # Top-left corner of the viewport in world space.
        self.offset_x = 0.0
        self.offset_y = 0.0
        # Zoom factor (1.0 = one world unit per screen pixel).
        self.zoom = 1.0
        self.min_zoom = float(min_zoom)
        self.max_zoom = float(max_zoom)
        # Follow smoothing factor for tracked organisms (0 = instant).
        self.smoothing = float(smoothing)
        # Optional tracked world position (cx, cy).
        self._track_target = None
        # Mouse-drag pan state.
        self._dragging = False
        self._drag_start = None
        self._drag_offset_start = None

    def clamp_offset(self):
        """Keep the viewport inside world boundaries."""
        max_x = max(0.0, self.world_width - self.viewport_width / self.zoom)
        max_y = max(0.0, self.world_height - self.viewport_height / self.zoom)
        self.offset_x = max(0.0, min(self.offset_x, max_x))
        self.offset_y = max(0.0, min(self.offset_y, max_y))

    def zoom_at(self, screen_x, screen_y, delta):
        """Zoom toward the cursor, keeping the world point under it fixed.

        delta > 0 zooms in one step, delta < 0 zooms out. Clamped to
        [min_zoom, max_zoom]; a no-op at the limits.
        """
        factor = 1.1 if delta > 0 else 1.0 / 1.1
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))
        if new_zoom == self.zoom:
            return
        world_x, world_y = self.screen_to_world(screen_x, screen_y)
        self.zoom = new_zoom
        self.offset_x = world_x - screen_x / new_zoom
        self.offset_y = world_y - screen_y / new_zoom
        self.clamp_offset()

    def pan(self, dx, dy):
        """Move the viewport by pixel deltas."""
        self.offset_x -= dx
        self.offset_y -= dy
        self._track_target = None
        self.clamp_offset()

    def center_on(self, world_x, world_y, smooth=False):
        """Center the viewport on a world point."""
        target_x = world_x - self.viewport_width / self.zoom / 2.0
        target_y = world_y - self.viewport_height / self.zoom / 2.0
        if smooth and self.smoothing > 0:
            self.offset_x += (target_x - self.offset_x) * self.smoothing
            self.offset_y += (target_y - self.offset_y) * self.smoothing
        else:
            self.offset_x = target_x
            self.offset_y = target_y
        self.clamp_offset()

    def set_track_target(self, position):
        """Follow an organism position each frame."""
        self._track_target = position

    def clear_track(self):
        """Stop auto-following."""
        self._track_target = None

    def update_tracking(self):
        """Apply smooth tracking when a target is set."""
        if self._track_target is None:
            return
        self.center_on(self._track_target[0], self._track_target[1], smooth=True)

    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to viewport pixel coordinates."""
        sx = (world_x - self.offset_x) * self.zoom
        sy = (world_y - self.offset_y) * self.zoom
        return int(sx), int(sy)

    def screen_to_world(self, screen_x, screen_y):
        """Convert viewport pixel coordinates to world coordinates."""
        wx = screen_x / self.zoom + self.offset_x
        wy = screen_y / self.zoom + self.offset_y
        return wx, wy

    def viewport_rect_world(self):
        """Return world-space bounds of the current viewport."""
        return (
            self.offset_x,
            self.offset_y,
            self.offset_x + self.viewport_width / self.zoom,
            self.offset_y + self.viewport_height / self.zoom,
        )

    def begin_drag(self, screen_pos):
        """Start mouse-drag panning."""
        self._dragging = True
        self._drag_start = screen_pos
        self._drag_offset_start = (self.offset_x, self.offset_y)
        self.clear_track()

    def drag_to(self, screen_pos):
        """Update pan offset while dragging."""
        if not self._dragging or self._drag_start is None:
            return
        dx = screen_pos[0] - self._drag_start[0]
        dy = screen_pos[1] - self._drag_start[1]
        # A screen drag moves the world by the zoomed distance.
        self.offset_x = self._drag_offset_start[0] - dx / self.zoom
        self.offset_y = self._drag_offset_start[1] - dy / self.zoom
        self.clamp_offset()

    def end_drag(self):
        """Stop mouse-drag panning."""
        self._dragging = False
        self._drag_start = None
        self._drag_offset_start = None

    def is_dragging(self):
        """Return True while the user is panning with the mouse."""
        return self._dragging
