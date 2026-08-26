"""Distance and collision helpers with explicit squared vs linear units."""

import math


def squared_distance(pos_a, pos_b):
    """Return squared Euclidean distance between two (x, y) positions."""
    # Guard missing positions so callers can treat them as infinitely far.
    if pos_a is None or pos_b is None:
        return float("inf")
    # Delta on x axis between the two points.
    dx = pos_a[0] - pos_b[0]
    # Delta on y axis between the two points.
    dy = pos_a[1] - pos_b[1]
    # Squared length avoids an expensive sqrt for radius comparisons.
    return dx * dx + dy * dy


def distance(pos_a, pos_b):
    """Return Euclidean (linear) distance between two positions."""
    # Reuse squared helper then take the real square root once.
    return math.sqrt(squared_distance(pos_a, pos_b))


def within_radius(pos_a, pos_b, radius):
    """Return True if pos_b is within linear radius of pos_a."""
    # Negative or zero radius never contains another distinct point usefully.
    if radius <= 0:
        return False
    # Compare squared distance to squared radius (correct unit pairing).
    return squared_distance(pos_a, pos_b) <= (radius * radius)


def colliding(pos_a, radius_a, pos_b, radius_b):
    """Return True if two circles centered at the positions overlap."""
    # Combined radius is the maximum center distance that still overlaps.
    combined = radius_a + radius_b
    # Delegate to within_radius so collision uses the same unit rules.
    return within_radius(pos_a, pos_b, combined)
