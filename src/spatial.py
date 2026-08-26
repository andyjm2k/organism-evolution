"""Uniform grid spatial index for radius queries in the simulation arena."""

from collections import defaultdict

import math

from distance import within_radius


class SpatialGrid:
    """Hash entities into cells for fast local radius queries."""

    def __init__(self, cell_size=100):
        # Cell edge length in world units (tuned to detection radii).
        self.cell_size = max(1, int(cell_size))
        # Map (cx, cy) cell keys to entity lists.
        self._cells = defaultdict(list)

    def clear(self):
        """Remove all indexed entities."""
        self._cells.clear()

    def _cell(self, position):
        """Return the grid cell key for a world position."""
        x, y = position
        return (
            int(x // self.cell_size),
            int(y // self.cell_size),
        )

    def insert(self, entity, position):
        """Place an entity into the cell covering its position."""
        if position is None:
            return
        self._cells[self._cell(position)].append(entity)

    def query(self, position, radius):
        """Yield entities within linear radius of position."""
        if position is None or radius <= 0:
            return
        # Number of cells to inspect in each direction.
        span = int(math.ceil(radius / self.cell_size)) + 1
        cx, cy = self._cell(position)
        seen = set()
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                for entity in self._cells.get((cx + dx, cy + dy), ()):
                    entity_id = id(entity)
                    if entity_id in seen:
                        continue
                    seen.add(entity_id)
                    entity_pos = getattr(entity, "position", None)
                    if entity_pos is None:
                        continue
                    if within_radius(position, entity_pos, radius):
                        yield entity
