import neat
import pygame
from organism import Organism as OrganismClass
from food import Food
import random
import json
import graphviz
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, neat_config, sim_config):
        # Store NEAT config
        self.neat_config = neat_config
        
        # Load simulation configuration
        if isinstance(sim_config, str):
            # If sim_config is a file path, load it
            with open(sim_config, 'r') as f:
                self.sim_config = json.load(f)
        else:
            # If sim_config is already a dictionary, use it directly
            self.sim_config = sim_config
        
        # Set logging level
        self.logging_level = self.sim_config.get('logging_level', 'normal')
        
        # Verify food configuration
        if 'num_food_items' not in self.sim_config:
            self.sim_config['num_food_items'] = 30  # More food
        
        # Only print detailed logs if logging level is set to detailed
        if self.logging_level == 'detailed':
            print(f"Configured for {self.sim_config['num_food_items']} food items")
        
        # Initialize food items
        self.food_items = []
        self.spawn_food()
        
        # Initialize renderer if rendering is enabled
        self.renderer = None
        if self.sim_config.get('render', False):
            from renderer import Renderer
            screen_size = max(
                self.sim_config['environment_width'],
                self.sim_config['environment_height']
            )
            self.renderer = Renderer(screen_size)
        
        # Add this line to store environment configuration
        self.environment_config = {
            'width': self.sim_config['environment_width'],
            'height': self.sim_config['environment_height'],
            'boundary_penalty': self.sim_config.get('boundary_penalty', 0.5),
            'detection_radius': self.sim_config['detection_radius'],
            'food_detection_radius': self.sim_config.get('food_detection_radius', self.sim_config['detection_radius']),
            'threat_detection_radius': self.sim_config.get('threat_detection_radius', self.sim_config['detection_radius']),
            'breeding_detection_radius': self.sim_config.get('breeding_detection_radius', self.sim_config['detection_radius'])
        }
        
        # Get simulation attributes from config
        self.simulation_steps = self.sim_config['simulation_steps']
        self.detection_radius = self.sim_config['detection_radius']
        self.food_detection_radius = self.sim_config.get('food_detection_radius', self.detection_radius)
        self.threat_detection_radius = self.sim_config.get('threat_detection_radius', self.detection_radius)
        self.breeding_detection_radius = self.sim_config.get('breeding_detection_radius', self.detection_radius)
        self.num_generations = self.sim_config['num_generations']
        
        # Initialize the scoreboard at simulation start
        from scoreboard import Scoreboard
        Scoreboard._species_records = {}  # Reset species records
        
        # Store the population reference
        self.population = None

    def spawn_food(self):
        """Spawn food items in the environment"""
        self.food_items.clear()
        margin = 10
        num_food = self.sim_config['num_food_items']
        
        # Only print detailed logs if logging level is set to detailed
        if self.logging_level == 'detailed':
            print(f"Spawning {num_food} food items")
        
        for _ in range(num_food):
            x = random.randint(margin, self.sim_config['environment_width'] - margin)
            y = random.randint(margin, self.sim_config['environment_height'] - margin)
            self.food_items.append(Food(x, y, log_creation=self.logging_level == 'detailed'))

    def eval_genomes(self, genomes, config):
        organisms = []
        genome_to_organism = {}
        
        # Only print detailed logs if logging level is set to detailed
        if self.logging_level == 'detailed':
            print("\n=== Creating Organisms ===")
        
        # Convert genomes list to dictionary for easier lookup
        genomes_dict = dict(genomes)
        
        # Create organisms for each genome
        for genome_id, genome in genomes:
            x = random.randint(10, self.sim_config['environment_width'] - 10)
            y = random.randint(10, self.sim_config['environment_height'] - 10)
            
            # Get species ID
            species_id = self.population.species.get_species_id(genome_id)
            
            # Only print detailed logs if logging level is set to detailed
            if self.logging_level == 'detailed':
                print(f"Creating organism for genome {genome_id} (Species {species_id})")
            
            try:
                # Use fully qualified import to avoid naming conflicts
                from organism import Organism as OrganismClass
                organism = OrganismClass(
                    genome=genome,
                    config=config,
                    position=(x, y),
                    environment_config=self.environment_config,
                    species_id=species_id,
                    logging_level=self.logging_level
                )
                # Set the simulation reference
                organism.simulation = self
                
                # Only print detailed logs if logging level is set to detailed
                if self.logging_level == 'detailed':
                    print(f"Successfully created organism")
            except Exception as e:
                print(f"Error creating organism: {e}")
                raise
            
            organisms.append(organism)
            genome_to_organism[genome_id] = organism
            genome.organism = organism
        
        # Record species information
        species_dict = {}
        for sid, species in self.population.species.species.items():
            for gid in species.members:
                species_dict[gid] = sid
        
        # Run multiple evaluation rounds for each genome
        num_trials = 3
        
        # Print number of organisms at the start
        print(f"\n[DEBUG] Starting evaluation with {len(organisms)} organisms")
        
        for trial in range(num_trials):
            # Reset environment and organisms
            self.spawn_food()
            for organism in organisms:
                organism.reset(self.environment_config)
            
            # Print number of organisms at the start of each trial
            print(f"[DEBUG] Trial {trial+1}: {len(organisms)} organisms at start")
            
            # Run simulation for fixed number of steps
            for step in range(self.simulation_steps):
                # Handle pygame events - only process mouse events if rendering is enabled
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.MOUSEBUTTONDOWN and self.renderer:
                        # Pass the current organisms list to handle_click
                        self.handle_click(event.pos, organisms)
                
                # Create a copy of the list to safely remove organisms
                for organism in organisms[:]:
                    # Remove dead organisms
                    if organism.energy <= 0:
                        if organism in organisms:
                            organisms.remove(organism)
                            # Set fitness to 0 for the corresponding genome
                            for genome_id, org in genome_to_organism.items():
                                if org == organism:
                                    # Use the dictionary lookup instead
                                    genome = genomes_dict[genome_id]
                                    genome.fitness = 0
                                    
                                    # Only print detailed logs if logging level is set to detailed
                                    if self.logging_level == 'detailed':
                                        print(f"Organism {genome_id} died - setting fitness to 0")
                        continue
                    
                    # Use different detection radii for different entity types
                    food_detection_radius = self.food_detection_radius
                    nearby_food = [f for f in self.food_items if organism.distance_to(f) < food_detection_radius]
                    
                    # For organisms, we need to separate threats (carnivores) from potential breeding partners
                    nearby_organisms = []
                    nearby_threats = []
                    nearby_breeding_partners = []
                    
                    for other in organisms:
                        if other == organism:
                            continue
                            
                        # Check if the other organism is a potential threat (carnivore)
                        if other.is_carnivore and not organism.is_carnivore:
                            if organism.distance_to(other) < self.threat_detection_radius:
                                nearby_threats.append(other)
                                
                        # Check if the other organism is a potential breeding partner (same species)
                        elif other.species_id == organism.species_id:
                            if organism.distance_to(other) < self.breeding_detection_radius:
                                nearby_breeding_partners.append(other)
                                
                        # For general detection (includes all organisms within standard detection radius)
                        if organism.distance_to(other) < self.detection_radius:
                            nearby_organisms.append(other)
                    
                    # Pass all the different nearby entities to the organism
                    organism.take_action(nearby_food, nearby_organisms, nearby_threats, nearby_breeding_partners)
                    organism.update(self.food_items, organisms)
                
                # End episode early if no organisms remain
                if not organisms:
                    print(f"[DEBUG] All organisms died at step {step+1}/{self.simulation_steps} in trial {trial+1}")
                    break
                
                # Print statistics every 50 steps
                if step % 50 == 0:
                    alive_organisms = len(organisms)
                    avg_energy = sum(org.energy for org in organisms) / max(1, alive_organisms)
                    print(f"[DEBUG] Trial {trial+1}, Step {step}: {alive_organisms} organisms alive, avg energy: {avg_energy:.2f}")
                
                # Render current state
                if self.renderer:
                    if not self.renderer.render(organisms, self.food_items):
                        return  # Exit if window is closed
            
            # Check surviving organisms at the end of this trial
            print(f"[DEBUG] Trial {trial+1} complete: {len(organisms)} organisms survived")
            
            # Store fitness for this trial
            fitness_values = []
            for genome_id, genome in genomes:
                organism = genome_to_organism.get(genome_id)
                if organism and organism in organisms:
                    trial_fitness = organism.calculate_fitness()
                    fitness_values.append(trial_fitness)
                    
                    # Debug fitness calculation
                    if trial_fitness > 0:
                        print(f"[DEBUG] Organism {genome_id} fitness: {trial_fitness}, energy: {organism.energy}")
                    
                    if not hasattr(genome, 'trial_fitnesses'):
                        genome.trial_fitnesses = []
                    genome.trial_fitnesses.append(trial_fitness)
                    
                    # Store the organism's highest fitness in the genome
                    if not hasattr(genome, 'highest_fitness'):
                        genome.highest_fitness = 0
                    genome.highest_fitness = max(genome.highest_fitness, organism.highest_fitness)
                else:
                    # If organism died during trial, set fitness to 0
                    if not hasattr(genome, 'trial_fitnesses'):
                        genome.trial_fitnesses = []
                    genome.trial_fitnesses.append(0)
            
            # Print fitness statistics
            if fitness_values:
                avg_fitness = sum(fitness_values) / len(fitness_values)
                max_fitness = max(fitness_values, default=0)
                positive_fitness = sum(1 for f in fitness_values if f > 0)
                print(f"[DEBUG] Trial {trial+1} fitness stats - Avg: {avg_fitness:.3f}, Max: {max_fitness:.3f}, Positive: {positive_fitness}/{len(fitness_values)}")
            else:
                print("[DEBUG] No organisms survived to calculate fitness")
        
        # Use median fitness across trials
        final_fitnesses = []
        for genome_id, genome in genomes:
            if hasattr(genome, 'trial_fitnesses'):
                genome.fitness = sorted(genome.trial_fitnesses)[len(genome.trial_fitnesses)//2]
                final_fitnesses.append(genome.fitness)
            else:
                genome.fitness = 0
                final_fitnesses.append(0)
                
                # Only print detailed logs if logging level is set to detailed
                if self.logging_level == 'detailed':
                    print(f"Genome {genome_id} died - fitness: 0")
        
        # Print final fitness statistics
        if final_fitnesses:
            avg_fitness = sum(final_fitnesses) / len(final_fitnesses)
            max_fitness = max(final_fitnesses, default=0)
            positive_fitness = sum(1 for f in final_fitnesses if f > 0)
            print(f"[DEBUG] Final fitness stats - Avg: {avg_fitness:.3f}, Max: {max_fitness:.3f}, Positive: {positive_fitness}/{len(final_fitnesses)}")
        
        # After evaluating all trials, update the scoreboard
        if organisms:  # Only update scoreboard if there are surviving organisms
            best_organism = max(organisms, key=lambda org: org.highest_fitness)
            
            # Only print detailed logs if logging level is set to detailed
            if self.logging_level == 'detailed':
                print(f"\n=== Best Organism This Generation ===")
                print(f"Species ID: {best_organism.species_id}")
                print(f"Highest Fitness: {best_organism.highest_fitness}")
            
            # Record the species
            from scoreboard import Scoreboard
            species_id = str(best_organism.species_id)
            scientific_name = None
            if species_id not in Scoreboard._species_records:
                from organism import Organism as OrganismClass
                scientific_name = OrganismClass.generate_scientific_name()
            
            Scoreboard.record_species(
                species_id=species_id,
                organism=best_organism,
                fitness=best_organism.highest_fitness,
                generation=self.population.generation,
                config=config
            )
        else:
            print("\nNo organisms survived to record in scoreboard")
        
        # Remove organism references from genomes before finishing
        for genome_id, genome in genomes:
            if hasattr(genome, 'organism'):
                delattr(genome, 'organism')
            
            # Clean up trial_fitnesses to avoid growing memory
            if hasattr(genome, 'trial_fitnesses'):
                delattr(genome, 'trial_fitnesses')
        
        # Clear references to help garbage collection
        for organism in organisms[:]:
            # Call the organism's cleanup method
            organism.cleanup()
            
        # Clear local references
        organisms.clear()
        genome_to_organism.clear()
        genomes_dict.clear()

    def visualize_neural_network(self, organism):
        """Visualize the neural network of the given organism using matplotlib."""
        # Only visualize if rendering is enabled
        if not self.renderer:
            return
            
        plt.clf()  # Clear the current figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate node positions
        input_nodes = [n for n in organism.genome.nodes.keys() if n < 0]
        output_nodes = list(range(len(input_nodes), len(input_nodes) + organism.genome.nodes))
        hidden_nodes = [n for n in organism.genome.nodes.keys() if n not in input_nodes and n not in output_nodes]
        
        # Position nodes in layers
        node_positions = {}
        
        # Input layer
        for i, node in enumerate(input_nodes):
            node_positions[node] = (-1, i/len(input_nodes) - 0.5)
        
        # Hidden layer
        if hidden_nodes:
            for i, node in enumerate(hidden_nodes):
                node_positions[node] = (0, i/len(hidden_nodes) - 0.5)
        
        # Output layer
        for i, node in enumerate(output_nodes):
            node_positions[node] = (1, i/len(output_nodes) - 0.5)
        
        # Draw connections
        for conn in organism.genome.connections.values():
            if conn.enabled:
                start = node_positions[conn.key[0]]
                end = node_positions[conn.key[1]]
                color = 'blue' if conn.weight > 0 else 'red'
                width = abs(conn.weight)
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color=color, linewidth=width, alpha=0.6)
        
        # Draw nodes
        for node, pos in node_positions.items():
            ax.plot(pos[0], pos[1], 'o', markersize=10, 
                   color='lightblue', markeredgecolor='black')
            ax.annotate(f'Node {node}', (pos[0], pos[1]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_title('Neural Network Visualization')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1, 1)
        ax.axis('equal')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to allow the window to appear

    def handle_click(self, pos, organisms):
        """Handle mouse click events to visualize the neural network of the clicked organism."""
        # Only process clicks if rendering is enabled
        if not self.renderer:
            return
            
        # Only print detailed logs if logging level is set to detailed
        if self.logging_level == 'detailed':
            print(f"Click detected at position: {pos}")
            
        for organism in organisms:
            if organism.position and self.is_click_on_organism(pos, organism.position):
                # Only print detailed logs if logging level is set to detailed
                if self.logging_level == 'detailed':
                    print(f"Organism clicked: {organism}")
                    
                self.visualize_neural_network(organism)
                break

    def is_click_on_organism(self, click_pos, org_pos):
        """Check if a click is on an organism based on its position."""
        click_x, click_y = click_pos
        org_x, org_y = org_pos
        
        # Only print detailed logs if logging level is set to detailed
        if self.logging_level == 'detailed':
            print(f"Checking click on organism at: {org_pos}")
            
        return (click_x - org_x) ** 2 + (click_y - org_y) ** 2 <= 100  # Assuming a radius of 10 for simplicity

    def run(self, max_generations=100):
        """Run the simulation for a specified number of generations"""
        # Setup initial state
        self.generation = 0
        
        # Create the population
        self.population = neat.Population(self.neat_config)
        
        # Add reporters
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        
        # Get termination condition from NEAT config
        fitness_threshold = self.neat_config.fitness_threshold
        max_generations = self.num_generations
        
        # Initial species setup for rendering and dashboard
        if self.renderer and len(self.population.species.species) > 0:
            for sid, species in self.population.species.species.items():
                # Initialize species colors - renderer will handle this automatically
                # when rendering organisms of this species
                species_id_str = str(sid)
                if species_id_str not in self.renderer.species_colors:
                    # Get a random organism from this species to determine if carnivore
                    is_carnivore = False
                    if len(species.members) > 0:
                        for genome_id in species.members:
                            if genome_id in self.population.population:
                                genome = self.population.population[genome_id]
                                # Try to infer if carnivore based on genome
                                # This is just a placeholder - actual implementation may vary
                                is_carnivore = len(genome.connections) > 10  # Simple heuristic
                                break
                    # This will cache the color for this species ID
                    self.renderer.get_species_color(species_id_str, is_carnivore)
                
        # Set the clock for the simulation
        clock = pygame.time.Clock()
        
        # Main evolutionary loop
        while True:
            gen = self.population.generation
            if self.renderer:
                self.renderer.set_generation(gen)
            
            # Check max generations
            if gen >= max_generations:
                print(f"\nReached maximum generations ({max_generations})")
                break
            
            # Run one generation with proper error handling
            try:
                winner = self.population.run(self.eval_genomes, 1)
            except KeyError as e:
                # Handle the specific KeyError during speciation
                error_str = str(e)
                traceback_str = str(e.__traceback__)
                
                print(f"\nHandling NEAT speciation error: {error_str}")
                
                # Save the current population
                current_genomes = self.population.population
                
                # Rebuild the species tracking
                self.population.species.genome_to_species = {}
                
                # Create a fresh 'unspeciated' list with all current genomes
                # This addresses the KeyError in unspeciated.remove()
                self.population.species.unspeciated = list(current_genomes.keys())
                
                # Special handling for KeyError: 0 which often occurs with breeding
                if error_str == "0" and hasattr(self.population.species, 'unspeciated'):
                    print("Handling special case for genome ID 0...")
                    # Ensure genome ID 0 is not in unspeciated if it doesn't exist in population
                    if 0 in self.population.species.unspeciated and 0 not in current_genomes:
                        self.population.species.unspeciated.remove(0)
                    # Add genome ID 0 to population if needed
                    if 0 not in current_genomes and 0 in self.population.species.genome_to_species:
                        species_id = self.population.species.genome_to_species[0]
                        # Remove from genome_to_species mapping
                        del self.population.species.genome_to_species[0]
                        # Also remove from the species members
                        if species_id in self.population.species.species:
                            if 0 in self.population.species.species[species_id].members:
                                del self.population.species.species[species_id].members[0]
                
                # Re-run speciation with the cleaned up data structures
                try:
                    self.population.species.speciate(self.population.config, current_genomes, gen)
                    print("Population tracking rebuilt successfully.")
                except Exception as speciation_error:
                    print(f"Failed to rebuild population tracking: {speciation_error}")
                    print("Resetting population to ensure simulation can continue...")
                    
                    # Last resort: reset species completely and force respeciation
                    self.population.species = neat.species.Species(self.population.config.species_set_config)
                    self.population.species.speciate(self.population.config, current_genomes, gen)
                    print("Population species reset completed.")
            except Exception as e:
                print(f"\nUnexpected error during evolution: {e}")
                raise
            
            # Force garbage collection after each generation
            import gc
            gc.collect()
            
            # Additional cleanup for memory management when rendering
            if self.renderer:
                # Call renderer cleanup
                self.renderer.cleanup_resources()
                
                # Check for retained pygame event references
                pygame.event.get()  # Clear the event queue
                
                # Extra GC run to collect renderer-related objects
                gc.collect()
            
            # Check if fitness threshold was reached
            if winner and winner.fitness >= fitness_threshold:
                print(f"\nFitness threshold ({fitness_threshold}) reached!")
                break
            
            # Get count of genomes with positive fitness (Option 2)
            surviving_count = sum(1 for genome in self.population.population.values() if genome.fitness is not None and genome.fitness > 0)
            
            # Always print generation summary regardless of logging level
            print(f"\nGeneration {gen} complete")
            print(f"Number of surviving organisms: {surviving_count}")
            
            # Get all organisms from the current generation for evaluation
            current_organisms = []
            for genome_id, genome in self.population.population.items():
                if hasattr(genome, 'organism'):
                    organism = genome.organism
                    # Only collect organisms that are still alive
                    if organism.energy > 0:
                        # Set the species_id from the genome's species
                        species_id = self.population.species.get_species_id(genome_id)
                        organism.species_id = species_id
                        current_organisms.append(organism)
            
            # Only evaluate if there are surviving organisms
            if current_organisms:
                # Evaluate the generation and update scoreboard
                self.evaluate_generation(current_organisms, gen)
                
                # Always display dashboard in terminal regardless of rendering
                from scoreboard import Scoreboard
                print("\n=== Generation Dashboard ===")
                Scoreboard.display_terminal_dashboard(gen)
            else:
                print("Warning: No organisms survived this generation")

            # Add event handling for mouse clicks - only process events if rendering is enabled
            if self.renderer:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        self.handle_click(event.pos, current_organisms)
            else:
                # In headless mode, just check for quit events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

    def evaluate_generation(self, organisms, generation):
        """Evaluate the current generation and update the scoreboard"""
        # Always print generation evaluation header
        print("\n=== Generation Evaluation ===")
        print(f"Generation: {generation}")
        print(f"Number of organisms: {len(organisms)}")
        
        # Check if there are any organisms to evaluate
        if not organisms:
            print("Warning: No organisms survived to be evaluated")
            return
        
        # Calculate statistics for this generation
        carnivores = sum(1 for org in organisms if org.is_carnivore)
        herbivores = len(organisms) - carnivores
        
        avg_fitness = sum(org.highest_fitness for org in organisms) / len(organisms)
        max_fitness = max(org.highest_fitness for org in organisms)
        min_fitness = min(org.highest_fitness for org in organisms)
        
        avg_energy = sum(org.energy for org in organisms) / len(organisms)
        avg_size = sum(org.size for org in organisms) / len(organisms)
        avg_speed = sum(org.speed for org in organisms) / len(organisms)
        
        # Print enhanced statistics
        print(f"\nOrganism Types: {carnivores} Carnivores, {herbivores} Herbivores")
        print(f"Fitness Stats: Avg: {avg_fitness:.2f}, Max: {max_fitness:.2f}, Min: {min_fitness:.2f}")
        print(f"Energy Avg: {avg_energy:.2f}")
        print(f"Size Avg: {avg_size:.2f}")
        print(f"Speed Avg: {avg_speed:.2f}")
        
        # Find unique species in this generation
        species_in_gen = {}
        for org in organisms:
            species_id = str(org.species_id)
            if species_id not in species_in_gen:
                species_in_gen[species_id] = []
            species_in_gen[species_id].append(org)
        
        print(f"\nUnique Species: {len(species_in_gen)}")
        
        # Find the organism with the highest fitness
        best_organism = max(organisms, key=lambda org: org.highest_fitness)
        
        # Print basic information about the best organism
        print(f"\nBest organism found:")
        print(f"- Species ID: {best_organism.species_id}")
        print(f"- Highest Fitness: {best_organism.highest_fitness:.2f}")
        print(f"- Type: {'Carnivore' if best_organism.is_carnivore else 'Herbivore'}")
        print(f"- Size: {best_organism.size:.2f}")
        print(f"- Speed: {best_organism.speed:.2f}")
        print(f"- Energy Efficiency: {best_organism.energy_efficiency:.2f}")
        print(f"- Food Consumed: {best_organism.food_consumed}")
        print(f"- Organisms Consumed: {best_organism.organisms_consumed}")
        
        # Record the best performing species
        from scoreboard import Scoreboard
        
        # Only generate scientific name if this is a new species
        species_id = str(best_organism.species_id)  # Convert to string for consistency
        scientific_name = None
        
        if species_id not in Scoreboard._species_records:
            from organism import Organism as OrganismClass
            scientific_name = OrganismClass.generate_scientific_name()
            
            # Only print detailed logs if logging level is set to detailed
            if self.logging_level == 'detailed':
                print(f"Generated new scientific name: {scientific_name}")
        
        # Only print detailed logs if logging level is set to detailed
        if self.logging_level == 'detailed':
            print(f"Recording species {species_id} with fitness {best_organism.highest_fitness}")
        
        Scoreboard.record_species(
            species_id=species_id,
            organism=best_organism,
            fitness=best_organism.highest_fitness,
            generation=generation,
            config=self.neat_config
        )
        
        # Only print detailed logs if logging level is set to detailed
        if self.logging_level == 'detailed':
            # Verify recording
            records = Scoreboard.get_records()
            print(f"\nCurrent Scoreboard Records: {len(records)}")
            for sid, record in records.items():
                print(f"Species {sid}: {record['scientific_name']} - Fitness: {record['highest_fitness']}")

    def add_organism(self, organism):
        """Add a new organism to the simulation during runtime (e.g. from breeding)"""
        # Set the simulation reference for the new organism
        organism.simulation = self
        
        try:
            # Get the next available genome ID from the population
            # Start from a higher number to avoid ID conflicts
            next_genome_id = max(self.population.population.keys()) + 10
            
            # Create a new genome for the organism and add it to the population
            organism.genome.fitness = 0  # Set initial fitness
            self.population.population[next_genome_id] = organism.genome
            
            # Find the parent's species
            parent_species = None
            parent_species_id = str(organism.species_id)
            
            for sid, species in self.population.species.species.items():
                if str(sid) == parent_species_id:
                    parent_species = species
                    break
            
            # IMPORTANT: Make sure this new genome is properly tracked in all NEAT data structures
            if parent_species is None:
                # If parent species not found, create a new one
                self.population.species.new_species(next_genome_id)
                species_id = self.population.species.get_species_id(next_genome_id)
                
                # Additional safety: ensure the species exists
                if species_id not in self.population.species.species:
                    # If the species wasn't created, create it manually
                    from neat.species import Species as NEATSpecies
                    self.population.species.species[species_id] = NEATSpecies(species_id)
                    self.population.species.species[species_id].members = {next_genome_id: organism.genome}
            else:
                # Add to parent's species
                species_id = organism.species_id
                # Add the genome to the species members
                parent_species.members[next_genome_id] = organism.genome
                # Update the genome-to-species mapping
                self.population.species.genome_to_species[next_genome_id] = species_id
            
            # Make sure this genome is no longer in the unspeciated list if it exists there
            if hasattr(self.population.species, 'unspeciated'):
                # Create the unspeciated list if it doesn't exist
                if not hasattr(self.population.species, 'unspeciated'):
                    self.population.species.unspeciated = []
                    
                # Make sure the new genome ID is not in unspeciated list
                if next_genome_id in self.population.species.unspeciated:
                    self.population.species.unspeciated.remove(next_genome_id)
            
            # Set the species ID on the organism
            organism.species_id = species_id
            
            # Only print detailed logs if logging level is set to detailed
            if self.logging_level == 'detailed':
                print(f"Added new organism to species {species_id} with genome ID {next_genome_id}")
            
            # Return the genome ID
            return next_genome_id
            
        except Exception as e:
            print(f"Error adding organism: {e}")
            # Return a placeholder ID in case of error
            return -1