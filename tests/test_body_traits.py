"""Unit tests for genome-linked body traits and their panel text."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import neat
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from body_traits import BodyTraits, apply_horn_strikes, draw_body
from food import Food
from genome_bootstrap import install_founding_motif
from organism import Organism


def _neat_config():
    """Load the project NEAT config so founding wires can be installed."""
    root = Path(__file__).resolve().parents[1]
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(root / "config" / "neat-config.ini"),
    )


def _founding_genome():
    """Return one genome with the shared living-world starting wires."""
    config = _neat_config()
    # Population construction attaches the innovation tracker the motif needs.
    neat.Population(config)
    genome = neat.DefaultGenome(1)
    genome.configure_new(config.genome_config)
    genome.fitness = 1.0
    install_founding_motif(genome, config.genome_config)
    return config, genome


def _living(genome, config, position, is_carnivore):
    """Real organism so bite, horn, and speed effects can be exercised."""
    env = {
        "width": 800,
        "height": 600,
        "detection_radius": 200,
        "food_detection_radius": 300,
        "threat_detection_radius": 100,
        "breeding_detection_radius": 150,
        "boundary_penalty": 0.5,
        "starting_energy": 200,
        "food_energy_value": 40,
        "movement_cost": 0.05,
    }
    organism = Organism(
        genome,
        config,
        position,
        env,
        species_id=1,
        diet_key=1 if not is_carnivore else 2,
        logging_level="normal",
    )
    organism.is_carnivore = is_carnivore
    return organism


def _organism(genome, is_carnivore):
    """Minimal organism stub carrying the physical fields the panel reads."""
    return SimpleNamespace(
        genome=genome,
        is_carnivore=is_carnivore,
        size=0.65,
        speed=0.85,
        get_radius=lambda: 4,
    )


class TestBodyTraits(unittest.TestCase):
    """Cover organ readout, diet-dependent labels, and panel text."""

    def test_herbivore_founder_expresses_jaw_and_horn(self):
        """Food wires are a jaw. Threat wires on a grazer are a horn."""
        _config, genome = _founding_genome()
        traits = BodyTraits.from_organism(_organism(genome, False))
        self.assertGreater(traits.jaw, 1.0)
        self.assertGreater(traits.horn, 1.0)
        self.assertEqual(traits.claw, 0.0)
        self.assertEqual(traits.flagella, 0.0)
        labels = [name for name, _strength in traits.present()]
        self.assertEqual(labels, ["Jaw", "Horn"])

    def test_carnivore_founder_expresses_jaw_and_claw(self):
        """The same prey channels are a claw once the diet is carnivore."""
        _config, genome = _founding_genome()
        traits = BodyTraits.from_organism(_organism(genome, True))
        self.assertGreater(traits.claw, 1.0)
        self.assertEqual(traits.horn, 0.0)
        labels = [name for name, _strength in traits.present()]
        self.assertIn("Claw", labels)
        self.assertNotIn("Horn", labels)

    def test_core_to_motor_wire_is_a_flagellum(self):
        """A body-sensor wire into velocity grows a flagellum."""
        config, genome = _founding_genome()
        innovation = config.genome_config.innovation_tracker.get_innovation_number(
            -1, 0, "flagellum-test"
        )
        connection = genome.create_connection(config.genome_config, -1, 0, innovation)
        connection.weight = 1.6
        connection.enabled = True
        genome.connections[connection.key] = connection
        traits = BodyTraits.from_organism(_organism(genome, False))
        self.assertGreaterEqual(traits.flagella, 1.6)
        self.assertIn("Flagella", [name for name, _strength in traits.present()])

    def test_disabled_wire_does_not_form_an_organ(self):
        """A disabled gene remains in the genome but adds no anatomy."""
        _config, genome = _founding_genome()
        for connection in genome.connections.values():
            connection.enabled = False
        traits = BodyTraits.from_organism(_organism(genome, False))
        self.assertEqual(traits.present(), [])

    def test_panel_lines_include_size_speed_and_organs(self):
        """The inspection panel lists body stats and the organs that exist."""
        _config, genome = _founding_genome()
        organism = _organism(genome, False)
        lines = BodyTraits.from_organism(organism).panel_lines(organism)
        self.assertIn("Size 0.65", lines[0])
        self.assertIn("Speed 0.85", lines[0])
        self.assertTrue(any("Jaw" in line and "Horn" in line for line in lines[1:]))

    def test_plain_genome_panel_says_plain_cell(self):
        """No sensor wires means the panel does not invent organs."""
        _config, genome = _founding_genome()
        genome.connections.clear()
        organism = _organism(genome, False)
        lines = BodyTraits.from_organism(organism).panel_lines(organism)
        self.assertIn("plain cell", lines[1])

    def test_draw_body_paints_pixels_for_expressed_organs(self):
        """Drawing a jawed body changes the surface, not only the text panel."""
        pygame.init()
        surface = pygame.Surface((80, 80), pygame.SRCALPHA)
        before = pygame.image.tostring(surface, "RGBA")
        traits = BodyTraits(jaw=3.6, claw=0.0, horn=3.6, flagella=1.5)
        draw_body(surface, 40, 40, (80, 160, 90), 6, traits)
        after = pygame.image.tostring(surface, "RGBA")
        self.assertNotEqual(before, after)
        pygame.quit()

    def test_jaw_reaches_farther_and_yields_more_energy(self):
        """A jaw bites beyond the body and takes more energy from a pellet."""
        config, genome = _founding_genome()
        jawed = _living(genome, config, (100, 100), False)
        _config_b, plain_genome = _founding_genome()
        plain = _living(plain_genome, _config_b, (100, 100), False)
        for connection in plain.genome.connections.values():
            connection.enabled = False
        # Just outside the body, inside the jaw's extra reach.
        offset = jawed.get_radius() + 5 + 2
        near = Food(100 + offset, 100)
        far = Food(100 + offset, 100)
        before_jaw = jawed.energy
        before_plain = plain.energy
        jawed.check_for_food([near])
        plain.check_for_food([far])
        self.assertIsNone(near.position)
        self.assertIsNotNone(far.position)
        self.assertGreater(jawed.energy - before_jaw, 0)
        # A pellet in contact range yields more to the jaw. Keep both below
        # their energy cap so the extra yield is not clipped.
        touched_jaw = Food(102, 100)
        touched_plain = Food(102, 100)
        jawed.energy = 50
        plain.energy = 50
        jaw_start = jawed.energy
        plain_start = plain.energy
        jawed.check_for_food([touched_jaw])
        plain.check_for_food([touched_plain])
        self.assertGreater(jawed.energy - jaw_start, plain.energy - plain_start)

    def test_claws_grab_prey_beyond_body_contact(self):
        """Claws catch a herbivore that is not touching the hunter's body."""
        config, genome = _founding_genome()
        hunter = _living(genome, config, (100, 100), True)
        _config_b, prey_genome = _founding_genome()
        prey = _living(prey_genome, _config_b, (100, 100), False)
        gap = hunter.get_radius() + prey.get_radius() + 3
        prey.position = (100 + gap, 100)
        before = prey.energy
        hunter.hunt_prey([hunter, prey])
        self.assertLess(prey.energy, before)
        self.assertGreater(prey.energy, 0)
        self.assertEqual(hunter.organisms_consumed, 0)

    def test_horn_damages_the_other_diet_only(self):
        """A herbivore horn stabs a carnivore and leaves another grazer alone."""
        config, genome = _founding_genome()
        grazer = _living(genome, config, (50, 50), False)
        _config_b, hunter_genome = _founding_genome()
        hunter = _living(hunter_genome, _config_b, (50, 50), True)
        for connection in hunter.genome.connections.values():
            connection.enabled = False
        _config_c, friend_genome = _founding_genome()
        friend = _living(friend_genome, _config_c, (50, 50), False)
        for connection in friend.genome.connections.values():
            connection.enabled = False
        hunter.energy = 180
        friend.energy = 180
        apply_horn_strikes(grazer, [grazer, hunter, friend])
        self.assertLess(hunter.energy, 180)
        self.assertEqual(friend.energy, 180)

    def test_flagella_increase_speed_and_movement_cost(self):
        """Flagella move farther per step and spend more energy doing it."""
        config, fast_genome = _founding_genome()
        innovation = config.genome_config.innovation_tracker.get_innovation_number(
            -1, 0, "flagellum-speed"
        )
        connection = fast_genome.create_connection(
            config.genome_config, -1, 0, innovation
        )
        connection.weight = 2.0
        connection.enabled = True
        fast_genome.connections[connection.key] = connection
        _config_b, slow_genome = _founding_genome()
        fast = _living(fast_genome, config, (200, 200), False)
        slow = _living(slow_genome, _config_b, (200, 200), False)

        class _FixedNet:
            """Network stub so both bodies try to move the same way."""

            def activate(self, _inputs):
                return (1.0, 0.0, -1.0, -1.0)

        for organism in (fast, slow):
            organism.network = _FixedNet()
            organism._compiled_network = None
            organism.energy = 150
        fast.take_action([], [], [], [])
        slow.take_action([], [], [], [])
        self.assertGreater(fast.position[0], slow.position[0])
        self.assertLess(fast.energy, slow.energy)


if __name__ == "__main__":
    unittest.main()
