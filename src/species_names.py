"""Scientific name fragments for species labeling."""

import random

GENUS_PREFIXES = [
    "Neo", "Cyber", "Digi", "Techno", "Synth", "Bio", "Quantum", "Meta",
    "Nano", "Robo", "Auto", "Proto", "Mega", "Ultra", "Hyper", "Super",
]
GENUS_SUFFIXES = [
    "bot", "tron", "droid", "mind", "form", "morph", "ware", "byte",
    "mech", "flex", "gen", "zoid", "pod", "roid", "node", "net",
]
SPECIES_PREFIXES = [
    "micro", "macro", "multi", "omni", "uni", "poly", "pseudo", "quasi",
    "semi", "sub", "super", "trans", "ultra", "anti", "meta", "para",
]
SPECIES_SUFFIXES = [
    "formis", "ensis", "oides", "atus", "inus", "alis", "arius", "osus",
    "ivus", "ilis", "icus", "anus", "eus", "ius", "aris", "ifer",
]


def generate_scientific_name():
    """Mint a random binomial-style name for a successful species."""
    genus = random.choice(GENUS_PREFIXES) + random.choice(GENUS_SUFFIXES)
    species = random.choice(SPECIES_PREFIXES) + random.choice(SPECIES_SUFFIXES)
    return f"{genus} {species}"
