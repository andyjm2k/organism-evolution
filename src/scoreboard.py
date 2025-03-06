class Scoreboard:
    _species_records = {}  # Class variable to store species records
    _initialized = False   # Track if scoreboard has been initialized
    _memory_stats = []     # Track memory usage over time

    @classmethod
    def initialize(cls):
        """Initialize or reset the scoreboard"""
        cls._species_records = {}
        cls._initialized = True
        print("Scoreboard initialized")

    @classmethod
    def record_species(cls, species_id, organism, fitness, generation, config):
        """Record a species' performance"""
        species_id = str(species_id)
        
        # Generate scientific name for new species
        if species_id not in cls._species_records:
            from organism import Organism
            scientific_name = Organism.generate_scientific_name()
        else:
            scientific_name = cls._species_records[species_id]['scientific_name']
        
        # Get the visual properties directly from the organism's methods
        size = organism.get_radius() * 2  # Convert radius to diameter for size
        num_spikes = organism.calculate_spikes()  # Use existing method from Organism class
        spike_length = organism.calculate_spike_length()  # Use existing method from Organism class
        is_carnivore = organism.is_carnivore  # Use existing method from Organism class
        
        species_record = {
            'scientific_name': scientific_name,
            'highest_fitness': fitness,
            'first_seen': generation if species_id not in cls._species_records else cls._species_records[species_id]['first_seen'],
            'last_seen': generation,
            'size': size,
            'num_spikes': num_spikes,
            'spike_length': spike_length,
            'is_carnivore': is_carnivore
        }
        
        # Update record if fitness is higher
        if species_id in cls._species_records:
            if fitness > cls._species_records[species_id]['highest_fitness']:
                cls._species_records[species_id].update(species_record)
        else:
            cls._species_records[species_id] = species_record

    @classmethod
    def get_top_species(cls, n):
        """Get the top N species by highest fitness"""
        if not cls._initialized:
            cls.initialize()
            
        # print(f"\n=== Getting Top Species ===")
        # print(f"Total species records: {len(cls._species_records)}")
        
        if not cls._species_records:
            # print("No species records found!")
            return []
            
        sorted_species = sorted(
            cls._species_records.items(),
            key=lambda x: x[1]['highest_fitness'],
            reverse=True
        )
        
        result = sorted_species[:n]
        # print(f"Returning {len(result)} top species")
        return result
        
    @classmethod
    def get_records(cls):
        """Get all species records"""
        if not cls._initialized:
            cls.initialize()
        return cls._species_records
        
    @classmethod
    def display_terminal_dashboard(cls, generation=None):
        """Display the top species dashboard in the terminal"""
        # Get the dashboard level from the simulation config if available
        try:
            import inspect
            import sys
            
            # Try to get the dashboard_level from simulation config
            # Look for a variable 'sim_config' in the caller's frame or its caller
            frame = inspect.currentframe().f_back
            dashboard_level = "normal"  # Default
            
            # Search through the stack for sim_config
            while frame:
                if 'sim_config' in frame.f_locals:
                    if 'dashboard_level' in frame.f_locals['sim_config']:
                        dashboard_level = frame.f_locals['sim_config']['dashboard_level']
                        break
                frame = frame.f_back
        except:
            dashboard_level = "normal"  # Default to normal if unable to determine
        
        # Get top species, limit depending on dashboard level
        species_limit = 5 if dashboard_level == "minimal" else 10
        top_species = cls.get_top_species(species_limit)
        
        # For minimal dashboard, only show a condensed version
        if dashboard_level == "minimal":
            print("\n" + "="*80)
            if generation is not None:
                print(f" "*25 + f"DASHBOARD SUMMARY - GENERATION {generation}")
            else:
                print(" "*30 + "DASHBOARD SUMMARY")
            print("="*80)
            
            total_species = len(cls._species_records)
            print(f"Total Species: {total_species} | Top Species: {len(top_species)}")
            
            if top_species:
                best_species_id, best_record = top_species[0]
                print(f"Best Species: {best_record['scientific_name']} (Fitness: {best_record['highest_fitness']:.2f})")
            
            # Display brief memory stats
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_usage_mb = memory_info.rss / (1024 * 1024)
                print(f"Memory Usage: {memory_usage_mb:.2f} MB")
            except:
                pass
            
            print("="*80)
            return
        
        # Full dashboard display for normal and detailed modes
        print("\n" + "="*100)
        if generation is not None:
            print(f" "*30 + f"TOP SPECIES DASHBOARD - GENERATION {generation}")
        else:
            print(" "*35 + "TOP SPECIES DASHBOARD")
        print("="*100)
        
        # Total species count and generation stats
        total_species = len(cls._species_records)
        carnivores = sum(1 for record in cls._species_records.values() if record['is_carnivore'])
        herbivores = total_species - carnivores
        
        print(f"Total Species: {total_species} | Carnivores: {carnivores} | Herbivores: {herbivores}")
        print("-"*100)
        
        # Header row with more attributes
        print(f"{'Rank':<4} {'Species Name':<30} {'Type':<10} {'Fitness':<12} {'First Seen':<10} {'Last Seen':<10} {'Size':<8} {'Spikes':<8}")
        print("-"*100)
        
        # Print each species with detailed information
        for rank, (species_id, record) in enumerate(top_species, 1):
            species_name = record['scientific_name']
            species_type = "Carnivore" if record['is_carnivore'] else "Herbivore"
            fitness = f"{record['highest_fitness']:.2f}"
            first_gen = record['first_seen']
            last_gen = record['last_seen']
            size = f"{record['size']:.2f}"
            spikes = record['num_spikes']
            
            print(f"{rank:<4} {species_name:<30} {species_type:<10} {fitness:<12} {first_gen:<10} {last_gen:<10} {size:<8} {spikes:<8}")
        
        # Show trend information for detailed dashboard
        if dashboard_level == "detailed" and generation is not None and generation > 0:
            print("\n" + "-"*100)
            print("Generation Trends:")
            
            # Calculate stats for this generation
            new_species = sum(1 for record in cls._species_records.values() if record['first_seen'] == generation)
            extinct_species = sum(1 for record in cls._species_records.values() if record['last_seen'] < generation - 1)
            
            # More detailed stats for species that have existed for multiple generations
            long_lived_species = sum(1 for record in cls._species_records.values() 
                                    if (record['last_seen'] - record['first_seen']) > 5)
            
            print(f"New Species: {new_species} | Extinct Species: {extinct_species} | Long-lived Species (>5 gens): {long_lived_species}")
            
            # Show species turnover rate if we have enough data
            if generation > 5:
                species_5_gens_ago = sum(1 for record in cls._species_records.values() 
                                       if record['first_seen'] <= generation - 5)
                surviving_old_species = sum(1 for record in cls._species_records.values() 
                                         if record['first_seen'] <= generation - 5 and record['last_seen'] >= generation - 1)
                
                if species_5_gens_ago > 0:
                    survival_rate = (surviving_old_species / species_5_gens_ago) * 100
                    print(f"5-Generation Survival Rate: {survival_rate:.1f}% ({surviving_old_species}/{species_5_gens_ago} species)")
        
        # Display memory usage
        cls.display_memory_usage(generation)
        
        print("="*100)
    
    @classmethod
    def display_memory_usage(cls, generation=None):
        """Display memory usage statistics"""
        try:
            import psutil
            import os
            
            # Get current process
            process = psutil.Process(os.getpid())
            
            # Get memory info in MB
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)
            
            # Store memory stats for trending
            if generation is not None:
                cls._memory_stats.append((generation, memory_usage_mb))
                # Keep only the most recent 20 generations
                if len(cls._memory_stats) > 20:
                    cls._memory_stats.pop(0)
            
            print("\n" + "-"*100)
            print("Memory Usage Statistics:")
            print(f"Current Memory Usage: {memory_usage_mb:.2f} MB")
            
            # Display memory trend if we have enough data
            if len(cls._memory_stats) > 1:
                # Calculate trend
                prev_gen, prev_mem = cls._memory_stats[-2]
                current_gen, current_mem = cls._memory_stats[-1]
                change = current_mem - prev_mem
                
                change_sign = "+" if change > 0 else ""
                print(f"Memory Change: {change_sign}{change:.2f} MB from Generation {prev_gen} to {current_gen}")
                
                # Display warning if memory is consistently growing
                if len(cls._memory_stats) > 5:
                    all_increasing = all(cls._memory_stats[i][1] < cls._memory_stats[i+1][1] for i in range(len(cls._memory_stats)-5, len(cls._memory_stats)-1))
                    if all_increasing and change > 0:
                        print("WARNING: Memory usage has been consistently increasing over the last 5 generations.")
        
        except ImportError:
            print("\n" + "-"*100)
            print("Memory Usage Statistics: (psutil module not installed)")
            print("To see memory statistics, install psutil with: pip install psutil")
            
    @classmethod
    def display_final_summary(cls, logging_level="normal"):
        """Display a comprehensive final summary of the simulation results"""
        print("\n\n" + "="*100)
        print(" "*35 + "SIMULATION FINAL SUMMARY")
        print("="*100)
        
        total_species = len(cls._species_records)
        if total_species == 0:
            print("No species were recorded during this simulation.")
            print("="*100)
            return
            
        # Get environment distribution stats
        carnivores = sum(1 for record in cls._species_records.values() if record['is_carnivore'])
        herbivores = total_species - carnivores
        
        # Get fitness statistics
        all_fitness = [record['highest_fitness'] for record in cls._species_records.values()]
        avg_fitness = sum(all_fitness) / len(all_fitness)
        max_fitness = max(all_fitness)
        min_fitness = min(all_fitness)
        
        # Get longevity stats
        lifespans = [record['last_seen'] - record['first_seen'] + 1 for record in cls._species_records.values()]
        avg_lifespan = sum(lifespans) / len(lifespans)
        max_lifespan = max(lifespans)
        
        # Get top species
        top_limit = 15 if logging_level == "detailed" else 10
        top_species = cls.get_top_species(top_limit)
        
        # Display summary statistics
        print(f"ECOSYSTEM OVERVIEW:")
        print(f"Total Species Evolved: {total_species}")
        print(f"Carnivores: {carnivores} ({carnivores/total_species*100:.1f}%) | Herbivores: {herbivores} ({herbivores/total_species*100:.1f}%)")
        print(f"Average Species Lifespan: {avg_lifespan:.1f} generations | Maximum Lifespan: {max_lifespan} generations")
        print(f"\nFITNESS STATISTICS:")
        print(f"Average Fitness: {avg_fitness:.2f}")
        print(f"Maximum Fitness: {max_fitness:.2f}")
        print(f"Minimum Fitness: {min_fitness:.2f}")
        
        # Find the most successful species
        if top_species:
            _, best_species = top_species[0]
            print(f"\nMOST SUCCESSFUL SPECIES:")
            print(f"Name: {best_species['scientific_name']}")
            print(f"Type: {'Carnivore' if best_species['is_carnivore'] else 'Herbivore'}")
            print(f"Highest Fitness: {best_species['highest_fitness']:.2f}")
            print(f"Lifespan: {best_species['last_seen'] - best_species['first_seen'] + 1} generations (Gen {best_species['first_seen']} - Gen {best_species['last_seen']})")
            print(f"Physical Attributes: Size = {best_species['size']:.2f}, Spikes = {best_species['num_spikes']}")
        
        # Display leaderboard
        print("\n" + "-"*100)
        print("FINAL SPECIES LEADERBOARD:")
        print("-"*100)
        
        # Header row
        print(f"{'Rank':<4} {'Species Name':<30} {'Type':<10} {'Fitness':<12} {'Lifespan':<10} {'Gen Range':<12} {'Size':<8}")
        print("-"*100)
        
        # Display each top species
        for rank, (species_id, record) in enumerate(top_species, 1):
            species_name = record['scientific_name']
            species_type = "Carnivore" if record['is_carnivore'] else "Herbivore"
            fitness = f"{record['highest_fitness']:.2f}"
            lifespan = record['last_seen'] - record['first_seen'] + 1
            gen_range = f"{record['first_seen']} - {record['last_seen']}"
            size = f"{record['size']:.2f}"
            
            print(f"{rank:<4} {species_name:<30} {species_type:<10} {fitness:<12} {lifespan:<10} {gen_range:<12} {size:<8}")
        
        # Display detailed stats if in detailed mode
        if logging_level == "detailed":
            print("\n" + "-"*100)
            print("DETAILED STATISTICS:")
            
            # Species type distribution by fitness
            carnivore_fitness = [r['highest_fitness'] for r in cls._species_records.values() if r['is_carnivore']]
            herbivore_fitness = [r['highest_fitness'] for r in cls._species_records.values() if not r['is_carnivore']]
            
            if carnivore_fitness:
                avg_carnivore_fitness = sum(carnivore_fitness) / len(carnivore_fitness)
                print(f"Average Carnivore Fitness: {avg_carnivore_fitness:.2f}")
            
            if herbivore_fitness:
                avg_herbivore_fitness = sum(herbivore_fitness) / len(herbivore_fitness)
                print(f"Average Herbivore Fitness: {avg_herbivore_fitness:.2f}")
            
            # Size and attribute analysis
            all_sizes = [r['size'] for r in cls._species_records.values()]
            avg_size = sum(all_sizes) / len(all_sizes)
            all_spikes = [r['num_spikes'] for r in cls._species_records.values()]
            avg_spikes = sum(all_spikes) / len(all_spikes)
            
            print(f"Average Organism Size: {avg_size:.2f}")
            print(f"Average Spikes Count: {avg_spikes:.2f}")
            
            # Correlation analysis
            size_fitness_pairs = [(r['size'], r['highest_fitness']) for r in cls._species_records.values()]
            sorted_by_size = sorted(size_fitness_pairs, key=lambda x: x[0])
            if len(sorted_by_size) > 1:
                smallest_third = sorted_by_size[:len(sorted_by_size)//3]
                largest_third = sorted_by_size[-len(sorted_by_size)//3:]
                
                avg_small_fitness = sum(f for _, f in smallest_third) / len(smallest_third)
                avg_large_fitness = sum(f for _, f in largest_third) / len(largest_third)
                
                print(f"Average Fitness of Smallest Third: {avg_small_fitness:.2f}")
                print(f"Average Fitness of Largest Third: {avg_large_fitness:.2f}")
                
                if avg_small_fitness > avg_large_fitness:
                    print("Trend: Smaller organisms tended to be more successful")
                else:
                    print("Trend: Larger organisms tended to be more successful")
        
        # Display final memory usage
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)
            
            print("\n" + "-"*100)
            print(f"Final Memory Usage: {memory_usage_mb:.2f} MB")
        except:
            pass
            
        print("="*100)