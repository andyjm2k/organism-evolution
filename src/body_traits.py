"""Physical traits expressed by a living-world genome.

Hidden nodes and sensor wires are the organs. Food wires form a jaw,
prey wires form a claw on a carnivore or a horn on a herbivore, and
core-body wires into the motors form flagella.
"""

import math

import pygame

from distance import colliding

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

    def food_reach_bonus(self):
        """Extra world units a jaw adds to herbivore bite range."""
        # Founder jaws are about 3.6, which reaches a few units past the body.
        return min(12.0, self.jaw * 1.25)

    def food_energy_multiplier(self):
        """How much more energy a jaw pulls from one pellet."""
        return 1.0 + min(0.6, self.jaw * 0.06)

    def hunt_reach_bonus(self):
        """Extra world units claws add to a carnivore's grab range."""
        return min(14.0, self.claw * 2.0)

    def prey_energy_fraction(self):
        """Share of prey energy a bite takes. Jaws raise the share."""
        return min(0.95, 0.7 + self.jaw * 0.03)

    def effective_speed(self, base_speed):
        """Speed after flagella. No flagella leaves the base speed unchanged."""
        return float(base_speed) * (1.0 + min(1.0, self.flagella * 0.18))

    def movement_cost_multiplier(self):
        """Flagella burn more energy per step of movement."""
        return 1.0 + min(0.8, self.flagella * 0.12)

    def horn_reach_bonus(self):
        """Extra world units a horn reaches beyond the body."""
        return min(8.0, self.horn * 1.0)

    def horn_damage(self, size):
        """Energy removed by one horn strike. Scales with horn and body size."""
        if self.horn < _VISIBLE_STRENGTH:
            return 0.0
        return self.horn * max(0.2, float(size)) * 2.0

    def panel_lines(self, organism):
        """Text rows for the living-world inspection panel."""
        # Shown speed includes flagella, so the panel matches how fast the body moves.
        speed = self.effective_speed(getattr(organism, "speed", 0.0))
        lines = [f"Size {organism.size:.2f} | Speed {speed:.2f}"]
        labels = [f"{label} {strength:.1f}" for label, strength in self.present()]
        if not labels:
            lines.append("Body: plain cell")
            return lines
        # Two organs per row so the panel stays inside the viewport width.
        lines.append("Body: " + " | ".join(labels[:2]))
        if len(labels) > 2:
            lines.append("      " + " | ".join(labels[2:]))
        return lines


def apply_horn_strikes(attacker, organisms):
    """Stab the other diet when a horn reaches them. Herds do not gore themselves."""
    traits = BodyTraits.from_organism(attacker)
    damage = traits.horn_damage(getattr(attacker, "size", 1.0))
    if damage <= 0.0 or getattr(attacker, "position", None) is None:
        return
    # Horn reach is the body radius plus a bonus from horn-wire strength.
    reach = attacker.get_radius() + traits.horn_reach_bonus()
    attacker_carnivore = bool(getattr(attacker, "is_carnivore", False))
    for other in organisms:
        if other is attacker or getattr(other, "energy", 0) <= 0:
            continue
        if getattr(other, "position", None) is None:
            continue
        # Same diet is not a target, so horned herbivores can crowd safely.
        if bool(getattr(other, "is_carnivore", False)) == attacker_carnivore:
            continue
        if not colliding(attacker.position, reach, other.position, other.get_radius()):
            continue
        other.energy = max(0.0, other.energy - damage)


def draw_body(surface, x, y, color, radius, traits):
    """Draw the cell and any organs that the genome currently expresses."""
    # On-screen radius stays readable when the world radius is only a few pixels.
    body_radius = max(6, min(16, int(radius) + 2))
    center = (int(x), int(y))
    # Flagella sit behind the body, opposite the mouth.
    if traits.flagella >= _VISIBLE_STRENGTH:
        _draw_flagella(surface, center, body_radius, traits.flagella, color)
    pygame.draw.circle(surface, color, center, body_radius)
    if traits.jaw >= _VISIBLE_STRENGTH:
        _draw_jaw(surface, center, body_radius, traits.jaw, color)
    if traits.horn >= _VISIBLE_STRENGTH:
        _draw_horn(surface, center, body_radius, traits.horn, color)
    if traits.claw >= _VISIBLE_STRENGTH:
        _draw_claw(surface, center, body_radius, traits.claw, color)


def _organ_length(radius, strength):
    """Longer organs for stronger wires, with a cap so they stay on screen."""
    return radius + min(18, int(strength * 2))


def _tint(color, toward, amount):
    """Blend a species color toward an organ highlight."""
    return tuple(
        max(0, min(255, int(color[index] * (1.0 - amount) + toward[index] * amount)))
        for index in range(3)
    )


def _draw_jaw(surface, center, radius, strength, color):
    """Mouth wedge on the leading side. Length tracks food-wire strength."""
    length = _organ_length(radius, strength)
    tint = _tint(color, (240, 190, 120), 0.55)
    tip = (center[0] + length, center[1])
    upper = (center[0] + radius - 1, center[1] - max(3, radius // 2))
    lower = (center[0] + radius - 1, center[1] + max(3, radius // 2))
    pygame.draw.polygon(surface, tint, (upper, tip, lower))


def _draw_horn(surface, center, radius, strength, color):
    """Single spike above the body. Length tracks threat-wire strength."""
    length = _organ_length(radius, strength)
    tint = _tint(color, (245, 245, 230), 0.65)
    left = (center[0] - 3, center[1] - radius + 1)
    right = (center[0] + 3, center[1] - radius + 1)
    tip = (center[0], center[1] - length)
    pygame.draw.polygon(surface, tint, (left, right, tip))


def _draw_claw(surface, center, radius, strength, color):
    """Two pincers in front of the body. Reach tracks prey-wire strength."""
    length = _organ_length(radius, strength)
    tint = _tint(color, (230, 170, 130), 0.5)
    spread = max(3, radius // 2)
    for sign in (-1, 1):
        start = (center[0] + radius - 2, center[1] + sign * spread)
        end = (center[0] + length, center[1] + sign * (spread + 3))
        pygame.draw.line(surface, tint, start, end, 2)


def _draw_flagella(surface, center, radius, strength, color):
    """Tail opposite the mouth. Length tracks core-to-motor wire strength."""
    length = _organ_length(radius, strength)
    tint = _tint(color, (140, 220, 180), 0.55)
    # Three segments make a short wave instead of a rigid spike.
    points = [ (center[0] - radius + 1, center[1]) ]
    for index in range(1, 4):
        along = radius + (length - radius) * index / 3
        wave = 3 * math.sin(index * 1.4) * (1 if index % 2 else -1)
        points.append((center[0] - along, center[1] + wave))
    pygame.draw.lines(surface, tint, False, points, 2)
