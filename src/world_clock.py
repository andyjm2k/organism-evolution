"""Monotonic world clock for the living-world simulation harness."""


class WorldClock:
    """Track elapsed simulation steps and derived world epochs."""

    def __init__(self, epoch_length=2000):
        # Current simulation step index (never reset in living-world mode).
        self.step = 0
        # Steps per named epoch for dashboard grouping.
        self.epoch_length = max(1, int(epoch_length))
        # Count of completed epochs since world start.
        self.epoch = 0

    def tick(self):
        """Advance one simulation step and roll epoch when threshold crossed."""
        self.step += 1
        if self.step % self.epoch_length == 0:
            self.epoch += 1

    def alive(self, max_steps):
        """Return False when an optional max step limit is reached."""
        if max_steps is None:
            return True
        return self.step < int(max_steps)
