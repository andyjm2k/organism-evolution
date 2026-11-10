/** Binomial labels for the scoreboard. Uses the world random stream. */

import type { PythonRandom } from "../rng/python-random";

const GENUS_PREFIXES = ["Neo", "Cyber", "Digi", "Techno", "Synth", "Bio", "Quantum", "Meta", "Nano", "Robo", "Auto", "Proto", "Mega", "Ultra", "Hyper", "Super"];
const GENUS_SUFFIXES = ["bot", "tron", "droid", "mind", "form", "morph", "ware", "byte", "mech", "flex", "gen", "zoid", "pod", "roid", "node", "net"];
const SPECIES_PREFIXES = ["micro", "macro", "multi", "omni", "uni", "poly", "pseudo", "quasi", "semi", "sub", "super", "trans", "ultra", "anti", "meta", "para"];
const SPECIES_SUFFIXES = ["formis", "ensis", "oides", "atus", "inus", "alis", "arius", "osus", "ivus", "ilis", "icus", "anus", "eus", "ius", "aris", "ifer"];

export function scientificName(rng: PythonRandom): string {
  return `${rng.choice(GENUS_PREFIXES)}${rng.choice(GENUS_SUFFIXES)} ${rng.choice(SPECIES_PREFIXES)}${rng.choice(SPECIES_SUFFIXES)}`;
}
