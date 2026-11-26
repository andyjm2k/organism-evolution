"""Physical traits expressed by a living-world genome.

Hidden nodes and sensor wires are the organs. Food wires form a jaw,
prey wires form a claw on a carnivore or a horn on a herbivore, and
core-body wires into the motors form flagella.
"""

import math

import pygame

# NEAT input key = -(channel index) - 1. These match network_inputs.py.
# Food channels are 9-15, prey or threat channels are 16-22, core body is 0-8.
_FOOD_CHANNELS = range(9, 16)
_PREY_CHANNELS = range(16, 23)
_CORE_CHANNELS = range(0, 9)
# Velocity outputs. A flagellum has to drive movement, not breeding or rest.
_VELOCITY_OUTPUTS = (0, 1)
# Ignore tiny mutated weights so a plain wire does not flicker on and off.
_VISIBLE_STRENGTH = 0.25


def _channel_for_key(node_key):
    """Return the sensor channel for a NEAT input key, or None for other nodes."""
    # Only negative keys are sensors. Outputs and hidden nodes are non-negative.
    if not isinstance(node_key, int) or node_key >= 0:
        return None
    return -node_key - 1


def _sensor_class(node_key):
    """Classify a sensor key as food, prey, core, or neither."""
    channel = _channel_for_key(node_key)
    if channel is None:
        return None
    if channel in _FOOD_CHANNELS:
        return "food"
    if channel in _PREY_CHANNELS:
        return "prey"
    if channel in _CORE_CHANNELS:
        return "core"
    return None


def _drives_body(target_key):
    """True when a wire lands on a motor output or a hidden organ node."""
    # Outputs 0 and 1 are velocity. Keys above the four outputs are hidden nodes.
    if not isinstance(target_key, int) or target_key < 0:
        return False
    return target_key in _VELOCITY_OUTPUTS or target_key > 3


class BodyTraits:
    """Strength of each organ read off one organism's current genome."""

    def __init__(self, jaw, claw, horn, flagella):
        # Absolute connection weight feeding each organ.
        self.jaw = float(jaw)
        self.claw = float(claw)
        self.horn = float(horn)
        self.flagella = float(flagella)

    @classmethod
    def from_organism(cls, organism):
        """Sum enabled sensor wires into jaw, claw, horn, and flagella."""
        genome = getattr(organism, "genome", None)
        if genome is None:
            return cls(0.0, 0.0, 0.0, 0.0)
        jaw = 0.0
        claw = 0.0
        horn = 0.0
        flagella = 0.0
        carnivore = bool(getattr(organism, "is_carnivore", False))
        for key, connection in genome.connections.items():
            # Disabled genes are still in the genome but do not build anatomy.
            if not getattr(connection, "enabled", True):
                continue
            source, target = key
            kind = _sensor_class(source)
            if kind is None or not _drives_body(target):
                continue
            strength = abs(float(connection.weight))
            if kind == "food":
                jaw += strength
            elif kind == "prey":
                # The same channels are prey for hunters and threats for grazers.
                if carnivore:
                    claw += strength
                else:
                    horn += strength
            elif kind == "core":
                flagella += strength
        return cls(jaw, claw, horn, flagella)

    def present(self):
        """Return (label, strength) pairs strong enough to show on the body."""
        named = (
            ("Jaw", self.jaw),
            ("Claw", self.claw),
            ("Horn", self.horn),
            ("Flagella", self.flagella),
        )
        return [(label, strength) for label, strength in named if strength >= _VISIBLE_STRENGTH]

    def panel_lines(self, organism):
        """Text rows for the living-world inspection panel."""
        # Size and speed are the body stats already derived from node and wire counts.
        lines = [f"Size {organism.size:.2f} | Speed {organism.speed:.2f}"]
        labels = [f"{label} {strength:.1f}" for label, strength in self.present()]
        if not labels:
            lines.append("Body: plain cell")
            return lines
        # Two organs per row so the panel stays inside the viewport width.
        lines.append("Body: " + " | ".join(labels[:2]))
        if len(labels) > 2:
            lines.append("      " + " | ".join(labels[2:]))
        return lines


def _rotate(offset, heading):
    """Rotate a local (x, y) offset by heading radians (screen space, y down)."""
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    x, y = offset
    return (x * cos_h - y * sin_h, x * sin_h + y * cos_h)


def _at(center, offset, heading):
    """Place a local offset at a body center, rotated to the heading."""
    rx, ry = _rotate(offset, heading)
    return (center[0] + rx, center[1] + ry)


def draw_body(surface, x, y, color, radius, traits, heading=0.0):
    """Draw the cell and any organs that the genome currently expresses.

    Organs are drawn in body space — jaw forward, flagella behind — and
    rotated to `heading`, so the creature faces its direction of travel.
    """
    # On-screen radius stays readable when the world radius is only a few pixels.
    body_radius = max(6, min(16, int(radius) + 2))
    center = (int(x), int(y))
    # Flagella sit behind the body, opposite the mouth.
    if traits.flagella >= _VISIBLE_STRENGTH:
        _draw_flagella(surface, center, body_radius, traits.flagella, color, heading)
    pygame.draw.circle(surface, color, center, body_radius)
    if traits.jaw >= _VISIBLE_STRENGTH:
        _draw_jaw(surface, center, body_radius, traits.jaw, color, heading)
    if traits.horn >= _VISIBLE_STRENGTH:
        _draw_horn(surface, center, body_radius, traits.horn, color, heading)
    if traits.claw >= _VISIBLE_STRENGTH:
        _draw_claw(surface, center, body_radius, traits.claw, color, heading)


def _organ_length(radius, strength):
    """Longer organs for stronger wires, with a cap so they stay on screen."""
    return radius + min(18, int(strength * 2))


def _tint(color, toward, amount):
    """Blend a species color toward an organ highlight."""
    return tuple(
        max(0, min(255, int(color[index] * (1.0 - amount) + toward[index] * amount)))
        for index in range(3)
    )


def _draw_jaw(surface, center, radius, strength, color, heading=0.0):
    """Mouth wedge on the leading side. Length tracks food-wire strength."""
    length = _organ_length(radius, strength)
    tint = _tint(color, (240, 190, 120), 0.55)
    tip = _at(center, (length, 0), heading)
    upper = _at(center, (radius - 1, -max(3, radius // 2)), heading)
    lower = _at(center, (radius - 1, max(3, radius // 2)), heading)
    pygame.draw.polygon(surface, tint, (upper, tip, lower))


def _draw_horn(surface, center, radius, strength, color, heading=0.0):
    """Single spike above the body. Length tracks threat-wire strength."""
    length = _organ_length(radius, strength)
    tint = _tint(color, (245, 245, 230), 0.65)
    left = _at(center, (-3, -radius + 1), heading)
    right = _at(center, (3, -radius + 1), heading)
    tip = _at(center, (0, -length), heading)
    pygame.draw.polygon(surface, tint, (left, right, tip))


def _draw_claw(surface, center, radius, strength, color, heading=0.0):
    """Two pincers in front of the body. Reach tracks prey-wire strength."""
    length = _organ_length(radius, strength)
    tint = _tint(color, (230, 170, 130), 0.5)
    spread = max(3, radius // 2)
    for sign in (-1, 1):
        start = _at(center, (radius - 2, sign * spread), heading)
        end = _at(center, (length, sign * (spread + 3)), heading)
        pygame.draw.line(surface, tint, start, end, 2)


def _draw_flagella(surface, center, radius, strength, color, heading=0.0):
    """Tail opposite the mouth. Length tracks core-to-motor wire strength."""
    length = _organ_length(radius, strength)
    tint = _tint(color, (140, 220, 180), 0.55)
    # Three segments make a short wave instead of a rigid spike.
    points = [_at(center, (-radius + 1, 0), heading)]
    for index in range(1, 4):
        along = radius + (length - radius) * index / 3
        wave = 3 * math.sin(index * 1.4) * (1 if index % 2 else -1)
        points.append(_at(center, (-along, wave), heading))
    pygame.draw.lines(surface, tint, False, points, 2)
