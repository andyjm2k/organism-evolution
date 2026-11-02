"""Heritable body genes, kept apart from the NEAT brain.

The Bibites store color, diet, and body plan in genes that mutate when
an egg hatches. The brain evolves on its own. This is that split: a
small vector copied from the fitter parent and nudged, so a child looks
and eats like its parent instead of taking a color from its species id.
"""

import random

# Pigment and diet live on 0..1. Size stays near 1 so metabolism does not jump.
_COLOR_MIN = 0.0
_COLOR_MAX = 1.0
_SIZE_MIN = 0.75
_SIZE_MAX = 1.35
# Diet at or above this value hunts. Below it grazes.
_CARNIVORE_DIET = 0.5
# How far a typical birth nudges each gene. Color drifts; diet drifts slower.
_DEFAULT_VARIANCE = 0.06
# Founders start as similar green grazers. Births spread them from there.
_FOUNDER_RED = 0.42
_FOUNDER_GREEN = 0.70
_FOUNDER_BLUE = 0.28
_FOUNDER_DIET = 0.08


def _clamp(value, low, high):
    """Keep a gene inside its legal range."""
    return max(low, min(high, value))


class BodyGenes:
    """Color, diet, and size carried on one genome and passed to its children."""

    def __init__(self, red, green, blue, diet, size):
        # Skin pigments, each in 0..1.
        self.red = float(red)
        self.green = float(green)
        self.blue = float(blue)
        # 0 grazes, 1 hunts. The switch sits at _CARNIVORE_DIET.
        self.diet = float(diet)
        # Multiplier on the body derived from the network. 1 leaves it unchanged.
        self.size = float(size)

    def copy(self):
        """Return a new gene set with the same values."""
        return BodyGenes(self.red, self.green, self.blue, self.diet, self.size)

    def mutate(self, rng=None, variance=_DEFAULT_VARIANCE):
        """Nudge every gene by a small gaussian and clamp it."""
        # One generator so tests can replay a birth without touching the world RNG.
        picker = rng if rng is not None else random
        spread = float(variance)
        self.red = _clamp(self.red + picker.gauss(0.0, spread), _COLOR_MIN, _COLOR_MAX)
        self.green = _clamp(self.green + picker.gauss(0.0, spread), _COLOR_MIN, _COLOR_MAX)
        self.blue = _clamp(self.blue + picker.gauss(0.0, spread), _COLOR_MIN, _COLOR_MAX)
        # Diet moves slower than pigment so a lineage does not flip role in one birth.
        self.diet = _clamp(
            self.diet + picker.gauss(0.0, spread * 0.45), _COLOR_MIN, _COLOR_MAX
        )
        self.size = _clamp(self.size + picker.gauss(0.0, spread * 0.5), _SIZE_MIN, _SIZE_MAX)
        return self

    def color_rgb(self):
        """Return the skin color as an 0..255 RGB tuple for pygame."""
        return (
            int(round(self.red * 255)),
            int(round(self.green * 255)),
            int(round(self.blue * 255)),
        )

    def is_carnivore(self):
        """True when the diet gene has crossed into hunting."""
        return self.diet >= _CARNIVORE_DIET


def make_founder_genes(rng=None):
    """Return the shared starting body: a green grazer with a little pigment noise."""
    genes = BodyGenes(_FOUNDER_RED, _FOUNDER_GREEN, _FOUNDER_BLUE, _FOUNDER_DIET, 1.0)
    # A small nudge so founders are not one flat color, still obviously one stock.
    genes.mutate(rng, variance=0.03)
    # The nudge must not turn the founding cohort into hunters. Plants are the food.
    genes.diet = min(0.2, max(0.0, genes.diet))
    genes.size = 1.0
    return genes


def inherit_body_genes(child_genome, parent_a, parent_b, rng=None, variance=_DEFAULT_VARIANCE):
    """Copy the fitter parent's body genes onto a child and mutate them.

    parent_a is the fitter parent when the caller ranked them. A child with
    no parent genes is left untouched so episodic tests keep the old diet rule.
    """
    source = getattr(parent_a, "body_genes", None)
    if source is None:
        source = getattr(parent_b, "body_genes", None)
    if source is None:
        return None
    child_genes = source.copy()
    child_genes.mutate(rng, variance=variance)
    child_genome.body_genes = child_genes
    return child_genes


def apply_body_genes(organism):
    """Express body genes onto diet and remember them on the organism.

    Returns True when the genome carried genes. Callers without genes keep
    the species-id diet they already computed.
    """
    genome = getattr(organism, "genome", None)
    genes = getattr(genome, "body_genes", None)
    if genes is None:
        return False
    organism.body_genes = genes
    organism.is_carnivore = genes.is_carnivore()
    # diet_key 1 grazes and 3 hunts under the legacy modulo rule, so births
    # that copy diet_key stay on the gene's side of the diet.
    organism._diet_key = 3 if organism.is_carnivore else 1
    return True


def body_color(organism):
    """Return the gene skin color, or None when this organism has no body genes."""
    genes = getattr(organism, "body_genes", None)
    if genes is None:
        genome = getattr(organism, "genome", None)
        genes = getattr(genome, "body_genes", None)
    if genes is None:
        return None
    return genes.color_rgb()
