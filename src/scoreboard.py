"""Terminal and pygame scoreboard for species tracking."""

from logging_util import log_always


class Scoreboard:
    """Tracks top species across generations."""

    _species_records = {}
    _initialized = False
    _memory_stats = []
    _dashboard_level = "normal"

    @classmethod
    def initialize(cls, dashboard_level="normal"):
        """Initialize or reset the scoreboard."""
        cls._species_records = {}
        cls._initialized = True
        cls._dashboard_level = dashboard_level
        log_always("Scoreboard initialized")

    @classmethod
    def set_dashboard_level(cls, level):
        """Set dashboard verbosity without reinitializing records."""
        cls._dashboard_level = level or "normal"

    @classmethod
    def record_species(cls, species_id, organism, fitness, generation, config):
        """Record or update a species performance entry."""
        species_id = str(species_id)
        if species_id not in cls._species_records:
            from organism import Organism

            scientific_name = Organism.generate_scientific_name()
        else:
            scientific_name = cls._species_records[species_id]["scientific_name"]

        size = organism.get_radius() * 2
        num_spikes = organism.calculate_spikes()
        spike_length = organism.calculate_spike_length()
        is_carnivore = organism.is_carnivore

        record = {
            "scientific_name": scientific_name,
            "highest_fitness": fitness,
            "first_seen": generation,
            "last_seen": generation,
            "size": size,
            "num_spikes": num_spikes,
            "spike_length": spike_length,
            "is_carnivore": is_carnivore,
        }

        if species_id in cls._species_records:
            existing = cls._species_records[species_id]
            existing["last_seen"] = generation
            if fitness > existing["highest_fitness"]:
                existing.update(record)
        else:
            cls._species_records[species_id] = record

    @classmethod
    def get_top_species(cls, n):
        """Return top N species sorted by highest fitness."""
        if not cls._initialized:
            cls.initialize()
        if not cls._species_records:
            return []
        sorted_species = sorted(
            cls._species_records.items(),
            key=lambda item: item[1]["highest_fitness"],
            reverse=True,
        )
        return sorted_species[:n]

    @classmethod
    def get_records(cls):
        """Return all species records."""
        if not cls._initialized:
            cls.initialize()
        return cls._species_records

    @classmethod
    def display_terminal_dashboard(cls, generation=None, dashboard_level=None):
        """Print the terminal species dashboard."""
        level = dashboard_level or cls._dashboard_level or "normal"
        species_limit = 5 if level == "minimal" else 10
        top_species = cls.get_top_species(species_limit)

        if level == "minimal":
            print("\n" + "=" * 80)
            title = (
                f"DASHBOARD SUMMARY - GENERATION {generation}"
                if generation is not None
                else "DASHBOARD SUMMARY"
            )
            print(" " * 25 + title)
            print("=" * 80)
            print(
                f"Total Species: {len(cls._species_records)} | "
                f"Top Species: {len(top_species)}"
            )
            if top_species:
                _sid, best = top_species[0]
                print(
                    f"Best Species: {best['scientific_name']} "
                    f"(Fitness: {best['highest_fitness']:.2f})"
                )
            cls._print_memory_usage(generation, compact=True)
            print("=" * 80)
            return

        print("\n" + "=" * 100)
        title = (
            f"TOP SPECIES DASHBOARD - GENERATION {generation}"
            if generation is not None
            else "TOP SPECIES DASHBOARD"
        )
        print(" " * 30 + title)
        print("=" * 100)

        total = len(cls._species_records)
        carnivores = sum(
            1 for record in cls._species_records.values() if record["is_carnivore"]
        )
        print(
            f"Total Species: {total} | Carnivores: {carnivores} | "
            f"Herbivores: {total - carnivores}"
        )
        print("-" * 100)
        print(
            f"{'Rank':<4} {'Species Name':<30} {'Type':<10} {'Fitness':<12} "
            f"{'First Seen':<10} {'Last Seen':<10} {'Size':<8} {'Spikes':<8}"
        )
        print("-" * 100)

        for rank, (_species_id, record) in enumerate(top_species, 1):
            species_type = "Carnivore" if record["is_carnivore"] else "Herbivore"
            print(
                f"{rank:<4} {record['scientific_name']:<30} {species_type:<10} "
                f"{record['highest_fitness']:<12.2f} {record['first_seen']:<10} "
                f"{record['last_seen']:<10} {record['size']:<8.2f} "
                f"{record['num_spikes']:<8}"
            )

        if level == "detailed" and generation is not None and generation > 0:
            print("\n" + "-" * 100)
            print("Generation Trends:")
            new_species = sum(
                1
                for record in cls._species_records.values()
                if record["first_seen"] == generation
            )
            extinct = sum(
                1
                for record in cls._species_records.values()
                if record["last_seen"] < generation - 1
            )
            long_lived = sum(
                1
                for record in cls._species_records.values()
                if (record["last_seen"] - record["first_seen"]) > 5
            )
            print(
                f"New Species: {new_species} | Extinct Species: {extinct} | "
                f"Long-lived Species (>5 gens): {long_lived}"
            )

        cls._print_memory_usage(generation, compact=False)
        print("=" * 100)

    @classmethod
    def _print_memory_usage(cls, generation=None, compact=False):
        """Print current process memory usage when psutil is available."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            if generation is not None:
                cls._memory_stats.append((generation, memory_mb))
                if len(cls._memory_stats) > 20:
                    cls._memory_stats.pop(0)
            if compact:
                print(f"Memory Usage: {memory_mb:.2f} MB")
                return
            print("\n" + "-" * 100)
            print("Memory Usage Statistics:")
            print(f"Current Memory Usage: {memory_mb:.2f} MB")
        except ImportError:
            if not compact:
                print("\n" + "-" * 100)
                print("Memory Usage Statistics: (psutil not installed)")

    @classmethod
    def display_final_summary(cls, logging_level="normal"):
        """Print a comprehensive summary after the simulation ends."""
        print("\n\n" + "=" * 100)
        print(" " * 35 + "SIMULATION FINAL SUMMARY")
        print("=" * 100)

        total_species = len(cls._species_records)
        if total_species == 0:
            print("No species were recorded during this simulation.")
            print("=" * 100)
            return

        carnivores = sum(
            1 for record in cls._species_records.values() if record["is_carnivore"]
        )
        herbivores = total_species - carnivores
        all_fitness = [
            record["highest_fitness"] for record in cls._species_records.values()
        ]
        avg_fitness = sum(all_fitness) / len(all_fitness)
        lifespans = [
            record["last_seen"] - record["first_seen"] + 1
            for record in cls._species_records.values()
        ]
        top_limit = 15 if logging_level == "detailed" else 10
        top_species = cls.get_top_species(top_limit)

        print("ECOSYSTEM OVERVIEW:")
        print(f"Total Species Evolved: {total_species}")
        print(
            f"Carnivores: {carnivores} ({carnivores/total_species*100:.1f}%) | "
            f"Herbivores: {herbivores} ({herbivores/total_species*100:.1f}%)"
        )
        print(
            f"Average Species Lifespan: {sum(lifespans)/len(lifespans):.1f} generations | "
            f"Maximum Lifespan: {max(lifespans)} generations"
        )
        print("\nFITNESS STATISTICS:")
        print(f"Average Fitness: {avg_fitness:.2f}")
        print(f"Maximum Fitness: {max(all_fitness):.2f}")
        print(f"Minimum Fitness: {min(all_fitness):.2f}")

        if top_species:
            _sid, best = top_species[0]
            print("\nMOST SUCCESSFUL SPECIES:")
            print(f"Name: {best['scientific_name']}")
            print(f"Type: {'Carnivore' if best['is_carnivore'] else 'Herbivore'}")
            print(f"Highest Fitness: {best['highest_fitness']:.2f}")
            print(
                f"Lifespan: {best['last_seen'] - best['first_seen'] + 1} generations "
                f"(Gen {best['first_seen']} - Gen {best['last_seen']})"
            )

        print("\n" + "-" * 100)
        print("FINAL SPECIES LEADERBOARD:")
        print("-" * 100)
        print(
            f"{'Rank':<4} {'Species Name':<30} {'Type':<10} {'Fitness':<12} "
            f"{'Lifespan':<10} {'Gen Range':<12} {'Size':<8}"
        )
        print("-" * 100)
        for rank, (_species_id, record) in enumerate(top_species, 1):
            lifespan = record["last_seen"] - record["first_seen"] + 1
            gen_range = f"{record['first_seen']} - {record['last_seen']}"
            species_type = "Carnivore" if record["is_carnivore"] else "Herbivore"
            print(
                f"{rank:<4} {record['scientific_name']:<30} {species_type:<10} "
                f"{record['highest_fitness']:<12.2f} {lifespan:<10} "
                f"{gen_range:<12} {record['size']:<8.2f}"
            )
        cls._print_memory_usage(compact=False)
        print("=" * 100)
