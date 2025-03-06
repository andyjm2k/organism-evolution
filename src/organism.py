import neat
import random
import math

# Add these constants before the generate_scientific_name method
GENUS_PREFIXES = [
    "Neo", "Cyber", "Digi", "Techno", "Synth", "Bio", "Quantum", "Meta",
    "Nano", "Robo", "Auto", "Proto", "Mega", "Ultra", "Hyper", "Super"
]

GENUS_SUFFIXES = [
    "bot", "tron", "droid", "mind", "form", "morph", "ware", "byte",
    "mech", "flex", "gen", "zoid", "pod", "roid", "node", "net"
]

# Add these new constants alongside the existing ones
SPECIES_PREFIXES = [
    "micro", "macro", "multi", "omni", "uni", "poly", "pseudo", "quasi",
    "semi", "sub", "super", "trans", "ultra", "anti", "meta", "para"
]

SPECIES_SUFFIXES = [
    "formis", "ensis", "oides", "atus", "inus", "alis", "arius", "osus",
    "ivus", "ilis", "icus", "anus", "eus", "ius", "aris", "ifer"
]

class Organism:
    def __init__(self, genome, config, position, environment_config, species_id=None, logging_level="normal"):
        self.genome = genome
        self.config = config
        self.network = neat.nn.FeedForwardNetwork.create(genome, config)
        self.environment_config = environment_config
        self.position = position
        self.species_id = species_id
        self.fitness_bonus = 0
        self.highest_fitness = 0
        self.simulation = None  # Will be set by the simulation when organism is added
        self.logging_level = logging_level
        
        # Initialize counters
        self.steps_taken = 0
        self.food_consumed = 0
        self.organisms_consumed = 0
        self.last_position = position
        self.was_moving = False
        
        # Add breeding cooldown tracking
        self.steps_since_breeding = 1000  # Start ready to breed
        self.breeding_cooldown = 200  # Number of steps required between breeding attempts
        
        # Add time-based consumption tracking
        self.steps_since_last_food = 0
        self.steps_since_last_hunt = 0
        self.avg_steps_between_food = []
        self.avg_steps_between_hunts = []
        
        # Determine organism type based on species_id to ensure consistency within species
        if species_id is not None:
            if isinstance(species_id, str):
                try:
                    num_id = int(species_id)
                except ValueError:
                    num_id = hash(species_id)
            else:
                num_id = species_id
            
            # Use modulo 4 to get a better distribution (25% chance each for stable type)
            mod_val = abs(num_id) % 4
            self.is_carnivore = mod_val >= 2  # 0,1 = herbivore, 2,3 = carnivore
        else:
            self.is_carnivore = random.random() > 0.5
        
        # Calculate evolutionary attributes based on genome
        self.calculate_attributes()
        
        # Initialize energy based on size
        self.energy = self.max_energy
        
        # Only print detailed logs if logging level is set to detailed
        if self.logging_level == 'detailed':
            print(f"Organism created at position {self.position}")
            print(f"Type: {'Carnivore' if self.is_carnivore else 'Herbivore'}")
            print(f"Speed: {self.speed:.2f}, Size: {self.size:.2f}, Energy Efficiency: {self.energy_efficiency:.2f}")
    
    def calculate_attributes(self):
        """Calculate evolutionary attributes based on genome structure"""
        # Use genome properties to determine attributes
        self.node_count = len(self.genome.nodes)
        self.connection_count = len(self.genome.connections)
        self.hidden_count = len([node for node in self.genome.nodes.keys() if node > 0])  # Positive IDs are hidden nodes
        
        # Base attributes - rebalanced for better herbivore survival
        base_speed = 1.2 if self.is_carnivore else 1.3  # Herbivores are faster for escape
        base_size = 1.1 if self.is_carnivore else 1.0   # Carnivores remain slightly larger
        base_energy = 300 if self.is_carnivore else 250  # More balanced energy pools - INCREASED for better survival
        base_efficiency = 1.5 if self.is_carnivore else 1.8  # Herbivores are more efficient
        
        # Calculate normalized values from genome (0.25 to 4.0 range)
        speed_factor = 0.25 + (self.node_count / 10)  # More nodes = potentially faster
        size_factor = 0.25 + (self.connection_count / 10)  # More connections = potentially larger
        
        # Calculate primary attributes with expanded ranges
        self.speed = base_speed * min(4.0, max(0.25, speed_factor))
        self.size = base_size * min(4.0, max(0.25, size_factor))
        
        # Trade-offs:
        # 1. Larger size = more max energy but faster energy drain
        # 2. Higher speed = faster movement but more energy cost
        # 3. Energy efficiency is inverse to size (smaller = more efficient)
        self.max_energy = base_energy * self.size * 3.0  # Increased energy multiplier - INCREASED
        self.energy_efficiency = base_efficiency * (2.0 - (self.size * 0.1))  # Reduced size penalty
        
        # Movement energy cost - adjusted for better sustainability
        base_movement_cost = 0.007 if self.is_carnivore else 0.005  # Reduced movement cost - DECREASED
        self.movement_energy_cost = base_movement_cost * (self.speed ** 1.1) * (self.size ** 0.5)  # Reduced scaling
        
        # Set initial energy to max
        self.energy = self.max_energy
        
        # Debug info to diagnose fitness issues
        print(f"[DEBUG] Organism attributes: Size={self.size:.2f}, Speed={self.speed:.2f}, Efficiency={self.energy_efficiency:.2f}")
        print(f"[DEBUG] Energy values: Max={self.max_energy:.2f}, Movement Cost={self.movement_energy_cost:.4f}")
        print(f"[DEBUG] Type: {'Carnivore' if self.is_carnivore else 'Herbivore'}, Nodes={self.node_count}, Connections={self.connection_count}")
        
        # Visual properties now based on neural network structure
        self.calculate_visual_properties()

    def calculate_visual_properties(self):
        """Calculate visual properties based on neural network structure"""
        # Base radius represents total number of nodes (input + hidden + output)
        base_size = 4  # Minimum size
        node_factor = self.node_count / 10  # Scale factor for total nodes
        self.base_radius = base_size + min(8, node_factor)  # Clamp maximum size increase
        
        # Number of spikes represents number of hidden nodes
        base_spikes = 3  # Minimum number of spikes
        hidden_factor = self.hidden_count  # One spike per hidden node
        self.num_spikes = min(16, max(base_spikes, base_spikes + hidden_factor))
        
        # Spike length represents connection density
        if self.node_count > 1:  # Avoid division by zero
            # Calculate connection density (connections per node)
            connection_density = self.connection_count / self.node_count
            # Scale spike length based on connection density
            self.spike_length = min(8, max(3, 3 + (connection_density * 2)))
        else:
            self.spike_length = 3  # Minimum spike length

    def take_action(self, nearby_food, nearby_organisms, nearby_threats=None, nearby_breeding_partners=None):
        if self.energy <= 0:
            return
            
        # Update consumption timers
        self.steps_since_last_food += 1
        self.steps_since_last_hunt += 1
        self.steps_since_breeding += 1  # Update breeding cooldown
        
        # Store previous position and energy
        self.last_position = self.position
        prev_energy = self.energy
        
        # Get environment dimensions with fallback
        width = self.environment_config.get('width', self.environment_config.get('environment_width', 800))
        height = self.environment_config.get('height', self.environment_config.get('environment_height', 600))
        
        # Calculate relative position in environment (-1 to 1 range for both dimensions)
        rel_x = (self.position[0] / width) * 2 - 1
        rel_y = (self.position[1] / height) * 2 - 1
        
        # Calculate center-relative position with normalized distance
        center_x = width / 2
        center_y = height / 2
        dx_to_center = self.position[0] - center_x
        dy_to_center = self.position[1] - center_y
        dist_to_center = math.sqrt(dx_to_center**2 + dy_to_center**2)
        max_dist = math.sqrt(center_x**2 + center_y**2)
        normalized_center_dist = dist_to_center / max_dist
        
        # Calculate boundary distances (normalized 0-1, equal weighting for all directions)
        left_dist = self.position[0] / width
        right_dist = (width - self.position[0]) / width
        top_dist = self.position[1] / height
        bottom_dist = (height - self.position[1]) / height
        
        # Calculate movement vector (normalized and balanced)
        if self.last_position != self.position:
            dx = (self.position[0] - self.last_position[0]) / self.speed
            dy = (self.position[1] - self.last_position[1]) / self.speed
            # Normalize movement vector
            movement_mag = math.sqrt(dx*dx + dy*dy)
            if movement_mag > 0:
                dx /= movement_mag
                dy /= movement_mag
        else:
            dx = dy = 0.0
        
        # Core inputs that both types need
        inputs = [
            self.energy / self.max_energy,     # Energy level (1)
            normalized_center_dist,            # Distance from center (2)
            rel_x,                            # Relative X position (-1 to 1) (3)
            rel_y,                            # Relative Y position (-1 to 1) (4)
            dx,                               # Normalized X movement (5)
            dy,                               # Normalized Y movement (6)
            self.size,                        # Size of organism (7)
            min(left_dist, right_dist),       # Closest horizontal boundary (8)
            min(top_dist, bottom_dist),       # Closest vertical boundary (9)
        ]

        if self.is_carnivore:
            # Process nearby organisms for hunting
            prey_count = 0
            closest_prey_dist = float('inf')
            closest_prey_dx = 0
            closest_prey_dy = 0
            
            for org in nearby_organisms:
                if not org.is_carnivore:  # Found potential prey
                    prey_count += 1
                    dist = math.sqrt(self.distance_to(org))
                    if dist < closest_prey_dist:
                        closest_prey_dist = dist
                        if org.position[0] != self.position[0]:
                            closest_prey_dx = (org.position[0] - self.position[0]) / self.environment_config['detection_radius']
                        if org.position[1] != self.position[1]:
                            closest_prey_dy = (org.position[1] - self.position[1]) / self.environment_config['detection_radius']
            
            # Normalize prey distance
            closest_prey_dist = min(1.0, closest_prey_dist / self.environment_config['detection_radius'])
            
            inputs.extend([
                prey_count / 10.0,          # Local prey density (10)
                closest_prey_dist,          # Distance to nearest prey (11)
                closest_prey_dx,            # X direction to prey (12)
                closest_prey_dy,            # Y direction to prey (13)
            ])
        else:
            # Process nearby food and threats for herbivores
            food_count = len(nearby_food)
            closest_food_dist = float('inf')
            closest_food_dx = 0
            closest_food_dy = 0
            
            # Use food_detection_radius for normalization
            food_detection_radius = self.environment_config.get('food_detection_radius', self.environment_config['detection_radius'])
            
            for food in nearby_food:
                if food.position is not None:
                    dist = math.sqrt(self.distance_to(food))
                    if dist < closest_food_dist:
                        closest_food_dist = dist
                        if food.position[0] != self.position[0]:
                            closest_food_dx = (food.position[0] - self.position[0]) / food_detection_radius
                        if food.position[1] != self.position[1]:
                            closest_food_dy = (food.position[1] - self.position[1]) / food_detection_radius
            
            # Find closest threat - use the dedicated nearby_threats list if provided
            threat_count = 0
            closest_threat_dist = float('inf')
            closest_threat_dx = 0
            closest_threat_dy = 0
            
            # Use threat_detection_radius for normalization
            threat_detection_radius = self.environment_config.get('threat_detection_radius', self.environment_config['detection_radius'])
            
            # Use the dedicated threats list if provided, otherwise fall back to filtering nearby_organisms
            threats_to_check = nearby_threats if nearby_threats is not None else [org for org in nearby_organisms if org.is_carnivore]
            
            for org in threats_to_check:
                threat_count += 1
                dist = math.sqrt(self.distance_to(org))
                if dist < closest_threat_dist:
                    closest_threat_dist = dist
                    if org.position[0] != self.position[0]:
                        closest_threat_dx = (org.position[0] - self.position[0]) / threat_detection_radius
                    if org.position[1] != self.position[1]:
                        closest_threat_dy = (org.position[1] - self.position[1]) / threat_detection_radius
            
            # Normalize distances using their respective detection radii
            closest_food_dist = min(1.0, closest_food_dist / food_detection_radius)
            closest_threat_dist = min(1.0, closest_threat_dist / threat_detection_radius)
            
            inputs.extend([
                food_count / 10.0,          # Local food density (10)
                closest_food_dist,          # Distance to nearest food (11)
                closest_food_dx,            # X direction to food (12)
                closest_food_dy,            # Y direction to food (13)
            ])

        # Add breeding-related inputs
        partner_count = 0
        closest_partner_dist = float('inf')
        closest_partner_dx = 0
        closest_partner_dy = 0

        # Use breeding_detection_radius for normalization
        breeding_detection_radius = self.environment_config.get('breeding_detection_radius', self.environment_config['detection_radius'])
        
        # Use the dedicated breeding partners list if provided, otherwise fall back to filtering nearby_organisms
        partners_to_check = nearby_breeding_partners if nearby_breeding_partners is not None else [
            org for org in nearby_organisms if org.species_id == self.species_id and org.energy >= 50
        ]

        for org in partners_to_check:
            partner_count += 1
            dist = math.sqrt(self.distance_to(org))
            if dist < closest_partner_dist:
                closest_partner_dist = dist
                if org.position[0] != self.position[0]:
                    closest_partner_dx = (org.position[0] - self.position[0]) / breeding_detection_radius
                if org.position[1] != self.position[1]:
                    closest_partner_dy = (org.position[1] - self.position[1]) / breeding_detection_radius

        # Normalize partner distance using breeding detection radius
        closest_partner_dist = min(1.0, closest_partner_dist / breeding_detection_radius)

        # Add breeding-related inputs (consolidated)
        inputs.extend([
            partner_count / 5.0,         # Local partner density (14)
            closest_partner_dist,        # Distance to nearest partner (15)
            closest_partner_dx,          # X direction to partner (16)
            closest_partner_dy,          # Y direction to partner (17)
            self.energy >= 50 and self.steps_since_breeding >= self.breeding_cooldown,  # Ready to breed flag (18)
        ])

        # Get network output and apply movement
        output = self.network.activate(inputs)
        
        # First 7 outputs are movement directions, last output is breeding desire
        movement_outputs = output[:7]
        breeding_desire = output[7]
        
        # Add small random noise to break ties and prevent initial bias
        movement_with_noise = [x + random.uniform(-0.01, 0.01) for x in movement_outputs]
        max_movement_index = movement_with_noise.index(max(movement_with_noise))
        
        # Apply movement based on speed attribute
        move_x = 0
        move_y = 0
        
        # Define movement angles (in degrees, clockwise from right)
        angles = [
            0,    # Right
            60,   # Down-Right
            120,  # Down
            180,  # Left
            240,  # Up-Left
            300,  # Up
            360   # Right (again, to complete the circle)
        ]
        
        # Convert the chosen angle to radians and calculate movement
        if max_movement_index < len(angles):
            angle_rad = math.radians(angles[max_movement_index])
            move_x = self.speed * math.cos(angle_rad)
            move_y = self.speed * math.sin(angle_rad)
            
            # Add small random variation to prevent straight-line movement
            variation = 0.1  # 10% maximum variation
            move_x += random.uniform(-variation, variation) * self.speed
            move_y += random.uniform(-variation, variation) * self.speed
        
        # Update position with boundary checks
        new_x = max(0, min(self.position[0] + move_x, width))
        new_y = max(0, min(self.position[1] + move_y, height))
        
        # Only update position if we actually moved
        if abs(move_x) > 0 or abs(move_y) > 0:
            self.position = (new_x, new_y)
            self.was_moving = True
        else:
            self.was_moving = False
        
        # Update energy
        self.energy = max(1, self.energy - self.movement_energy_cost)
        self.steps_taken += 1
        
        # Check for breeding if breeding desire is high and conditions are met
        if (breeding_desire > 0.5 and 
            self.energy >= 50 and  # Increased energy requirement
            self.steps_since_breeding >= self.breeding_cooldown):  # Check cooldown
            nearby_same_species = [org for org in nearby_organisms 
                                 if org.species_id == self.species_id 
                                 and org.energy >= 50  # Increased partner energy requirement
                                 and org.steps_since_breeding >= org.breeding_cooldown  # Check partner cooldown
                                 and self.distance_to(org) <= (self.get_radius() + org.get_radius()) * 1.5]  # Use radius-based distance
            
            if nearby_same_species:
                partner = nearby_same_species[0]
                
                # Only print detailed logs if logging level is set to detailed
                if self.logging_level == 'detailed':
                    print(f"\n=== Breeding Event ===")
                    print(f"Parent 1: Species {self.species_id}")
                    print(f"- Type: {'Carnivore' if self.is_carnivore else 'Herbivore'}")
                    print(f"- Position: {self.position}")
                    print(f"- Energy: {self.energy} -> {self.energy - 50}")
                    print(f"- Steps since last breeding: {self.steps_since_breeding}")
                    print(f"- Fitness: {self.calculate_fitness()}")
                    
                    print(f"\nParent 2: Species {partner.species_id}")
                    print(f"- Type: {'Carnivore' if partner.is_carnivore else 'Herbivore'}")
                    print(f"- Position: {partner.position}")
                    print(f"- Energy: {partner.energy} -> {partner.energy - 50}")
                    print(f"- Steps since last breeding: {partner.steps_since_breeding}")
                    print(f"- Fitness: {partner.calculate_fitness()}")
                
                # Check if both organisms can breed
                can_self_breed = self.can_breed()
                can_partner_breed = partner.can_breed()
                
                # Check if breeding failed due to boundary restriction
                if self.energy >= 100 and self.steps_since_breeding >= self.breeding_cooldown and not can_self_breed:
                    # This means we failed the boundary distance check
                    if self.logging_level == 'detailed':
                        print(f"Organism {self.species_id} cannot breed due to being too close to environment boundary")
                
                if partner.energy >= 100 and partner.steps_since_breeding >= partner.breeding_cooldown and not can_partner_breed:
                    # Partner failed the boundary distance check
                    if self.logging_level == 'detailed':
                        print(f"Partner organism {partner.species_id} cannot breed due to being too close to environment boundary")
                
                if can_self_breed and can_partner_breed:
                    # Breed with the partner
                    if self.logging_level == 'detailed':
                        print(f"Organisms breeding: {self.species_id} with {partner.species_id}")
                    
                    try:
                        # Both parents pay energy cost and reset breeding cooldown
                        self.energy -= 50
                        partner.energy -= 50
                        self.steps_since_breeding = 0
                        partner.steps_since_breeding = 0
                        
                        # Create a new position near the parents
                        spawn_x = (self.position[0] + partner.position[0]) / 2
                        spawn_y = (self.position[1] + partner.position[1]) / 2
                        
                        # Add small random offset
                        spawn_x += random.uniform(-5, 5)
                        spawn_y += random.uniform(-5, 5)
                        
                        # Keep within boundaries
                        width = self.environment_config['width']
                        height = self.environment_config['height']
                        spawn_x = max(0, min(spawn_x, width))
                        spawn_y = max(0, min(spawn_y, height))
                        
                        # Create new genome using NEAT's crossover
                        # Set fitness values on parent genomes before crossover
                        current_fitness = self.calculate_fitness()
                        partner_fitness = partner.calculate_fitness()
                        self.genome.fitness = current_fitness
                        partner.genome.fitness = partner_fitness
                        
                        # Choose parent order based on fitness
                        if current_fitness > partner_fitness:
                            parent1, parent2 = self.genome, partner.genome
                        else:
                            parent1, parent2 = partner.genome, self.genome
                        
                        # Create a new genome of the same type as the parents
                        child_genome = neat.DefaultGenome(0)  # ID doesn't matter, will be replaced
                        # Perform crossover using the instance method
                        child_genome.configure_crossover(parent1, parent2, self.config.genome_config)
                        # Mutate the child's genome
                        child_genome.mutate(self.config.genome_config)
                        
                        # Create new organism
                        child = Organism(
                            genome=child_genome,
                            config=self.config,
                            position=(spawn_x, spawn_y),
                            environment_config=self.environment_config,
                            species_id=self.species_id,  # Inherit species from parents
                            logging_level=self.logging_level  # Pass logging level to child
                        )
                        
                        # Set simulation reference for the child
                        if self.simulation is not None:
                            child.simulation = self.simulation
                            # Add the child organism to the simulation
                            genome_id = self.simulation.add_organism(child)
                            
                            # Only add the organism to nearby_organisms if successfully added to simulation
                            if genome_id > 0:
                                # Only print detailed logs if logging level is set to detailed
                                if self.logging_level == 'detailed':
                                    print(f"\nChild Organism Created:")
                                    print(f"- Genome ID: {genome_id}")
                                    print(f"- Species: {child.species_id}")
                                    print(f"- Type: {'Carnivore' if child.is_carnivore else 'Herbivore'}")
                                    print(f"- Position: {child.position}")
                                    print(f"- Energy: {child.energy}")
                                    print(f"- Speed: {child.speed:.2f}")
                                    print(f"- Size: {child.size:.2f}")
                                    print(f"- Energy Efficiency: {child.energy_efficiency:.2f}")
                                
                                # Give fitness bonus to both parents
                                self.fitness_bonus += 100
                                partner.fitness_bonus += 100
                                
                                # Add child to the list of organisms being evaluated
                                if child not in nearby_organisms:
                                    nearby_organisms.append(child)
                            else:
                                # If adding to simulation failed, refund some energy
                                self.energy += 25
                                partner.energy += 25
                                if self.logging_level == 'detailed':
                                    print("Failed to add child organism to simulation")
                        else:
                            # Always print this warning regardless of logging level
                            print("Warning: Could not add child organism - no simulation reference")
                    except Exception as e:
                        # Log the error but don't crash the simulation
                        if self.logging_level == 'detailed':
                            print(f"Error during breeding: {str(e)}")
                        
                        # Refund energy to parents if breeding fails
                        self.energy += 50
                        partner.energy += 50
                        # Don't reset cooldown so they can try again soon
                        self.steps_since_breeding = self.breeding_cooldown - 50
                        partner.steps_since_breeding = partner.breeding_cooldown - 50
        
        # Check for food collision
        if nearby_food:
            for food in nearby_food:
                if self.distance_to(food) <= 0:  # Changed from fixed distance to radius-based
                    energy_gain = 75 * self.energy_efficiency  # Increased food energy value
                    self.energy = min(self.max_energy, self.energy + energy_gain)
                    self.food_consumed += 1
                    food.position = None
                    break
        
        # Check for organism collision
        if nearby_organisms:
            for other in nearby_organisms:
                if other.energy > 0 and self.distance_to(other) <= 0:  # Changed from fixed distance to radius-based
                    # Larger organisms can eat smaller ones
                    if self.get_radius() > other.get_radius():
                        self.energy += other.energy / 2
                        other.energy = 0
                        self.organisms_consumed += 1
                        # Only print detailed logs if logging level is set to detailed
                        if self.logging_level == 'detailed' and other.energy == 0:
                            print(f"Organism consumed another at {self.position}")

    def distance_to(self, other):
        """Calculate distance between this organism and another object"""
        if self.position is None or other.position is None:
            return float('inf')
        
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return dx*dx + dy*dy  # Return squared distance to avoid sqrt

    def distance_to_sqrt(self, other):
        """Calculate actual distance (with sqrt) when needed"""
        if self.position is None or other.position is None:
            return float('inf')
        
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return math.sqrt(dx*dx + dy*dy)

    def update(self, food_items, organisms):
        """Update organism position and state"""
        self.steps_taken += 1
        self.steps_since_breeding += 1
        self.steps_since_last_food += 1
        self.steps_since_last_hunt += 1
        
        # Cap energy at maximum
        self.energy = min(self.energy, self.max_energy)
        
        # Check for valid position
        if self.position is None:
            print("Warning: Organism has no position!")
            return
        
        # Add small survival bonus just for staying alive
        self.fitness_bonus += 0.05
        
        # Deduct resting energy cost - proportional to size
        rest_cost = 0.01 * self.size
        self.energy -= rest_cost
        
        # Add position to history for exploration tracking
        if hasattr(self, 'last_positions'):
            if len(self.last_positions) >= 100:  # Limit history
                self.last_positions.pop(0)
            # Round positions to track general areas rather than exact coordinates
            rounded_pos = (round(self.position[0] / 10) * 10, round(self.position[1] / 10) * 10)
            self.last_positions.append(rounded_pos)
        else:
            self.last_positions = []
        
        # Boundary checking - wrap around or bounce
        x, y = self.position
        width = self.environment_config['width']
        height = self.environment_config['height']
        boundary_penalty = self.environment_config.get('boundary_penalty', 0.5)
        
        # Wrap around the world
        if x < 0:
            x = width
            # Apply boundary penalty
            self.energy *= (1 - boundary_penalty)
        elif x > width:
            x = 0
            # Apply boundary penalty
            self.energy *= (1 - boundary_penalty)
            
        if y < 0:
            y = height
            # Apply boundary penalty
            self.energy *= (1 - boundary_penalty)
        elif y > height:
            y = 0
            # Apply boundary penalty
            self.energy *= (1 - boundary_penalty)
            
        # Update position
        self.position = (x, y)
        
        # Check for nearby food and eat if close enough and appropriate for diet
        self.check_for_food(food_items)
        
        # Check for prey if we're a carnivore
        if self.is_carnivore:
            self.hunt_prey(organisms)

    def is_near(self, food):
        # Check if either position is None
        if self.position is None or food.position is None:
            return False
        # Check if the organism is near a food item using its radius
        return self.distance_to(food) < self.get_radius() + 4  # 4 is food radius

    def can_breed(self):
        """Check if the organism is ready to breed."""
        # Check energy and cooldown first
        if not (self.energy >= 100 and self.steps_since_breeding >= self.breeding_cooldown):
            return False
            
        # Check if away from boundaries
        if self.position is None:
            return False
            
        # Get environment dimensions
        width = self.environment_config.get('width', self.environment_config.get('environment_width', 800))
        height = self.environment_config.get('height', self.environment_config.get('environment_height', 600))
        
        # Calculate minimum boundary distance (require at least 10% of environment dimension from any edge)
        min_distance = min(width, height) * 0.1
        
        # Calculate distance to each boundary
        dist_to_left = self.position[0]
        dist_to_right = width - self.position[0]
        dist_to_top = self.position[1]
        dist_to_bottom = height - self.position[1]
        
        # Minimum distance to any boundary
        min_boundary_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        
        # Only allow breeding if sufficiently far from any boundary
        return min_boundary_dist >= min_distance

    def calculate_fitness(self):
        """Calculate fitness based on various factors and organism type"""
        # Base survival metrics considering efficiency
        survival_score = (
            self.steps_taken * 0.4 +      # Base step reward
            self.energy * 0.3             # Energy value weight
        ) * self.energy_efficiency        # Scale by efficiency
        
        # Calculate consumption efficiency
        def calculate_consumption_efficiency(steps_list, max_optimal_steps):
            if not steps_list:
                return 0.0
            avg_steps = sum(steps_list) / len(steps_list)
            # Efficiency decreases exponentially as average steps increase
            efficiency = math.exp(-avg_steps / max_optimal_steps)
            return efficiency
        
        # Role-specific scoring
        if self.is_carnivore:
            # Calculate hunting efficiency (optimal = 100 steps between hunts)
            hunting_efficiency = calculate_consumption_efficiency(self.avg_steps_between_hunts, 100)
            
            # Size-based hunting bonus to encourage different strategies
            size_bonus = 1.0
            if self.size < 1.0:  # Small and agile hunters
                size_bonus = 1.5 if self.speed > 2.0 else 1.0
            elif self.size > 2.0:  # Large predators
                size_bonus = 1.3
                
            # Speed-based hunting bonus
            speed_bonus = 1.0
            if self.speed > 2.0:  # Fast hunters
                speed_bonus = 1.4 if self.size < 1.5 else 1.1
            
            # Territory control bonus (based on movement patterns)
            territory_bonus = min(1.5, self.steps_taken / 1000.0)
            
            # Early hunting success bonus - rewards getting first kills
            early_success_bonus = 1.0
            if self.organisms_consumed > 0 and self.steps_taken < 500:
                early_success_bonus = 1.5
            
            # Carnivore scoring focuses on hunting and domination with diverse strategy bonuses
            role_score = (
                self.organisms_consumed * 400 +  # Increased base hunting reward
                (self.organisms_consumed * 250 * hunting_efficiency) +  # Increased efficiency bonus
                (self.organisms_consumed * 150 * size_bonus) +  # Size strategy bonus
                (self.organisms_consumed * 150 * speed_bonus) +  # Speed strategy bonus
                (self.organisms_consumed * 100 * early_success_bonus) +  # Early success bonus
                (self.steps_taken * 0.6 * self.speed * territory_bonus)  # Movement and territory reward
            )
            
            # More gradual penalty for not hunting
            if self.steps_since_last_hunt > 150:  # Increased from 100
                role_score *= math.exp(-self.steps_since_last_hunt / 350)  # Increased from 300
            
            # Heavy penalty for breaking carnivore rules
            if self.food_consumed > 0:
                role_score *= 0.01
                
        else:
            # Calculate foraging efficiency (optimal = 50 steps between food)
            foraging_efficiency = calculate_consumption_efficiency(self.avg_steps_between_food, 50)
            
            # Speed-based survival bonus for herbivores
            speed_bonus = 1.0
            if self.speed > 2.0:  # Fast herbivores get bonus
                speed_bonus = 1.5
            elif self.speed > 1.5:  # Moderately fast herbivores get smaller bonus
                speed_bonus = 1.2
            
            # Size-based survival bonus
            size_bonus = 1.0
            if self.size > 1.5:  # Larger herbivores are harder to hunt
                size_bonus = 1.3
            elif self.size < 0.8:  # Very small herbivores are harder to catch
                size_bonus = 1.2
            
            # Efficiency bonus for good foraging
            efficiency_bonus = self.energy_efficiency * 1.2
            
            # Herbivore scoring focuses on efficient foraging and survival
            role_score = (
                self.food_consumed * 350 +  # Increased base foraging reward (from 250 to 350)
                (self.food_consumed * 250 * foraging_efficiency) +  # Increased efficiency bonus (from 200 to 250)
                (self.food_consumed * 100 * speed_bonus) +  # Speed survival bonus
                (self.food_consumed * 100 * size_bonus) +  # Size survival bonus
                (self.food_consumed * 50 * efficiency_bonus) +  # Efficiency bonus
                (self.steps_taken * 0.5 * self.speed)  # Movement reward scaled by speed
            )
            
            # More gradual penalty for not eating - increased threshold and decay factor
            if self.steps_since_last_food > 100:  # Increased from 50 to 100
                role_score *= math.exp(-self.steps_since_last_food / 250)  # Increased from 150 to 250
            
            # Heavy penalty for breaking herbivore rules
            if self.organisms_consumed > 0:
                role_score *= 0.01
        
        # Get environment dimensions with fallback
        width = self.environment_config.get('width', self.environment_config.get('environment_width', 800))
        height = self.environment_config.get('height', self.environment_config.get('environment_height', 600))
        
        # Calculate distance from center of environment
        center_x = width / 2
        center_y = height / 2
        distance_from_center = math.sqrt((self.position[0] - center_x)**2 + (self.position[1] - center_y)**2)
        max_distance = math.sqrt((width/2)**2 + (height/2)**2)
        center_reward = (1 - (distance_from_center / max_distance)) * 100
        
        # Add exploration bonus
        if hasattr(self, 'last_positions'):
            unique_positions = len(set(self.last_positions))
            exploration_bonus = min(200, unique_positions * 2)
        else:
            exploration_bonus = 0
        
        # Boundary penalty/bonus - now with enhanced central bias
        boundary_distance = min(
            self.position[0], self.position[1],
            width - self.position[0],
            height - self.position[1]
        )
        
        # Calculate the safe breeding zone boundary distance (10% of smaller dimension)
        safe_zone_boundary = min(width, height) * 0.1
        
        # Enhanced multiplier that gives stronger bonus for staying in breeding zone
        if boundary_distance >= safe_zone_boundary:
            # Extra bonus for staying in breeding safe zone
            boundary_multiplier = 1.2  # 20% bonus for staying away from edges
        else:
            # Linear penalty for getting closer to edge
            boundary_multiplier = 0.5 + (boundary_distance / safe_zone_boundary) * 0.5
        
        # Combine all factors with adjusted weights
        final_fitness = (
            (survival_score * 0.3) +  # Base survival
            (role_score * 0.4) +      # Role-specific behavior
            (center_reward * 0.15) +   # Center-seeking reward
            (exploration_bonus * 0.15)  # Exploration bonus
        ) * boundary_multiplier      # Apply boundary penalty/bonus
        
        # Add small random noise to break symmetry (reduced magnitude)
        final_fitness += random.uniform(-0.05, 0.05)
        
        # Update highest fitness if current is higher
        if final_fitness > self.highest_fitness:
            self.highest_fitness = final_fitness
        
        return final_fitness

    def get_closest_food_info(self, nearby_food):
        if not nearby_food:
            return [1.0, 0.0]
        
        # Find closest food using squared distances
        closest_food = min(nearby_food, key=lambda f: self.distance_to(f))
        squared_distance = self.distance_to(closest_food)
        normalized_distance = min(1.0, math.sqrt(squared_distance) / self.environment_config['detection_radius'])
        
        dx = closest_food.position[0] - self.position[0]
        dy = closest_food.position[1] - self.position[1]
        angle = math.atan2(dy, dx) / math.pi
        
        return [normalized_distance, angle]

    def get_closest_organism_info(self, nearby_organisms):
        if not nearby_organisms:
            return [1.0, 0.0, 0.0]
        
        # Find closest organism using squared distances
        closest_organism = min(nearby_organisms, key=lambda o: self.distance_to(o))
        squared_distance = self.distance_to(closest_organism)
        normalized_distance = min(1.0, math.sqrt(squared_distance) / self.environment_config['detection_radius'])
        
        energy_difference = (self.energy - closest_organism.energy) / 200.0
        
        dx = closest_organism.position[0] - self.position[0]
        dy = closest_organism.position[1] - self.position[1]
        angle = math.atan2(dy, dx) / math.pi
        
        return [normalized_distance, energy_difference, angle]

    def get_closest_same_species_info(self, nearby_organisms):
        same_species = [org for org in nearby_organisms if org.species_id == self.species_id]
        
        if not same_species:
            return [1.0, 0.0, 0.0]  # Max distance, no direction
        
        closest_organism = min(same_species, key=lambda o: self.distance_to(o))
        distance = self.distance_to(closest_organism)
        normalized_distance = min(1.0, distance / self.environment_config['detection_radius'])
        
        # Calculate direction vector
        dx = closest_organism.position[0] - self.position[0]
        dy = closest_organism.position[1] - self.position[1]
        
        return [
            normalized_distance,  # How far is the same-species organism
            dx / self.environment_config['detection_radius'],  # X direction
            dy / self.environment_config['detection_radius']   # Y direction
        ]

    def get_hidden_neuron_count(self):
        # NEAT uses negative numbers for input nodes, positive for hidden/output
        hidden_nodes = [node for node in self.genome.nodes.keys() 
                       if node > 0]  # Positive numbers are hidden/output nodes
        return len(hidden_nodes)

    def get_active_node_count(self):
        # Count all nodes (input, hidden, and output) that are active in the genome
        return len([node for node in self.genome.nodes.keys()])

    def get_radius(self):
        """Calculate the organism's radius based on its size"""
        return max(2, min(20, self.base_radius))

    def is_colliding_with(self, other):
        """Check if this organism is colliding with another object"""
        distance = self.distance_to(other)
        if hasattr(other, '_radius'):  # Is an organism
            return distance <= (self._radius + other._radius)
        else:  # Is food
            return distance <= (self._radius + 4)  # Food radius is 4

    @staticmethod
    def generate_scientific_name():
        """Generate a unique scientific name for a successful species"""
        genus = random.choice(GENUS_PREFIXES) + random.choice(GENUS_SUFFIXES)
        species = random.choice(SPECIES_PREFIXES) + random.choice(SPECIES_SUFFIXES)
        return f"{genus} {species}"

    def get_prey_info(self, nearby_organisms):
        potential_prey = [o for o in nearby_organisms if o.energy < self.energy]
        if not potential_prey:
            return [1.0, 0.0, 0.0, 0.0]  # No prey nearby
        
        closest_prey = min(potential_prey, key=lambda o: self.distance_to(o))
        distance = self.distance_to(closest_prey)
        normalized_distance = min(1.0, distance / self.environment_config['detection_radius'])
        energy_difference = (self.energy - closest_prey.energy) / 200.0
        
        dx = closest_prey.position[0] - self.position[0]
        dy = closest_prey.position[1] - self.position[1]
        
        return [
            normalized_distance,
            energy_difference,
            dx / self.environment_config['detection_radius'],
            dy / self.environment_config['detection_radius']
        ]

    def get_threat_info(self, nearby_organisms):
        threats = [o for o in nearby_organisms if o.energy > self.energy]
        if not threats:
            return [1.0, 0.0, 0.0, 0.0]  # No threats nearby
        
        closest_threat = min(threats, key=lambda o: self.distance_to(o))
        distance = self.distance_to(closest_threat)
        normalized_distance = min(1.0, distance / self.environment_config['detection_radius'])
        energy_difference = (self.energy - closest_threat.energy) / 200.0
        
        dx = closest_threat.position[0] - self.position[0]
        dy = closest_threat.position[1] - self.position[1]
        
        return [
            normalized_distance,
            energy_difference,
            dx / self.environment_config['detection_radius'],
            dy / self.environment_config['detection_radius']
        ]

    def reset(self, environment_config):
        # Reset position to a random location
        self.environment_config = environment_config
        
        # Use correct environment_config keys
        width = environment_config.get('width', environment_config.get('environment_width', 800))
        height = environment_config.get('height', environment_config.get('environment_height', 600))
        
        x = random.randint(10, width - 10)
        y = random.randint(10, height - 10)
        self.position = (x, y)
        self.last_position = self.position
        
        # Reset energy and tracking variables
        self.energy = self.max_energy
        self.steps_taken = 0
        self.food_consumed = 0
        self.organisms_consumed = 0
        self.was_moving = False
        self.fitness_bonus = 0
        self.steps_since_breeding = 1000  # Start ready to breed
        self.steps_since_last_food = 0
        self.steps_since_last_hunt = 0
        
        # Don't reset these - they're cumulative
        # self.avg_steps_between_food = []
        # self.avg_steps_between_hunts = []
        
        # Determine organism type based on species_id to ensure consistency within species
        if self.species_id is not None:
            if isinstance(self.species_id, str):
                try:
                    num_id = int(self.species_id)
                except ValueError:
                    num_id = hash(self.species_id)
            else:
                num_id = self.species_id
            
            # Use modulo 4 to get a better distribution (25% chance each for stable type)
            mod_val = abs(num_id) % 4
            self.is_carnivore = mod_val >= 2  # 0,1 = herbivore, 2,3 = carnivore
        else:
            self.is_carnivore = random.random() > 0.5
        
        # Calculate evolutionary attributes based on genome
        self.calculate_attributes()
        
        # Only print detailed logs if logging level is set to detailed
        if self.logging_level == 'detailed':
            print(f"Organism created at position {self.position}")
            print(f"Type: {'Carnivore' if self.is_carnivore else 'Herbivore'}")
            print(f"Speed: {self.speed:.2f}, Size: {self.size:.2f}, Energy Efficiency: {self.energy_efficiency:.2f}")
    
    def cleanup(self):
        """Clean up references to help garbage collection"""
        if hasattr(self, 'network'):
            self.network = None
        if hasattr(self, 'genome'):
            self.genome = None
        if hasattr(self, 'config'):
            self.config = None
        if hasattr(self, 'simulation'):
            self.simulation = None
        
        # Clear any large lists that might be holding memory
        if hasattr(self, 'avg_steps_between_food'):
            self.avg_steps_between_food.clear()
        if hasattr(self, 'avg_steps_between_hunts'):
            self.avg_steps_between_hunts.clear()

    def calculate_spikes(self):
        """Calculate number of spikes based on neural network structure"""
        # Base number of spikes represents minimum complexity
        base_spikes = 3
        
        # Add spikes based on hidden nodes (more hidden nodes = more spikes)
        hidden_spikes = self.hidden_count
        
        # Carnivores get a slight bonus to spikes
        type_bonus = 2 if self.is_carnivore else 0
        
        return min(16, max(3, base_spikes + hidden_spikes + type_bonus))
    
    def calculate_spike_length(self):
        """Calculate spike length based on neural network density"""
        # Base length represents minimum complexity
        base_length = 3
        
        # Calculate connection density (connections per node)
        if self.node_count > 1:
            connection_density = self.connection_count / self.node_count
            density_factor = connection_density * 2
        else:
            density_factor = 0
        
        # Carnivores get slightly longer spikes
        type_multiplier = 1.2 if self.is_carnivore else 1.0
        
        spike_length = (base_length + density_factor) * type_multiplier
        return min(8, max(3, spike_length))  # Clamp between 3 and 8

    def get_target_inputs(self, target, detection_radius):
        """Helper method to get normalized distance and angle to a target"""
        if target is None:
            return [1.0, 0.0]  # Max distance, no direction
            
        dist = math.sqrt(self.distance_to(target))
        normalized_dist = min(1.0, dist / detection_radius)
        
        dx = target.position[0] - self.position[0]
        dy = target.position[1] - self.position[1]
        angle = math.atan2(dy, dx) / math.pi
        
        return [normalized_dist, angle]

    def check_for_food(self, food_items):
        """Check for nearby food and consume if close enough"""
        if self.energy <= 0 or self.is_carnivore:
            return
            
        # Food detection radius from environment config
        food_detection_radius = self.environment_config.get('food_detection_radius', 
                                                          self.environment_config['detection_radius'])

        # Find food items that are close enough to consume
        for food in food_items:
            if food.position is None:
                continue
                
            # Use radius-based collision detection
            if self.distance_to(food) <= (self.get_radius() + 5) ** 2:  # Food radius is approximately 5
                # Calculate energy gain based on efficiency
                energy_gain = 75 * self.energy_efficiency
                
                # Add energy up to max_energy
                self.energy = min(self.max_energy, self.energy + energy_gain)
                
                # Update stats
                self.food_consumed += 1
                
                # Update time-tracking for consumption patterns
                if self.steps_since_last_food > 0:
                    self.avg_steps_between_food.append(self.steps_since_last_food)
                    # Keep history limited to prevent memory buildup
                    if len(self.avg_steps_between_food) > 10:
                        self.avg_steps_between_food.pop(0)
                self.steps_since_last_food = 0
                
                # Mark food as consumed
                food.position = None
                
                # Add fitness bonus for successful foraging
                self.fitness_bonus += 10
                
                # Only print detailed logs if logging level is set to detailed
                if self.logging_level == 'detailed':
                    print(f"Herbivore consumed food at {self.position}, energy now {self.energy:.2f}")
                
                # Only consume one food item per update
                break

    def hunt_prey(self, organisms):
        """Hunt and consume other organisms if close enough"""
        if self.energy <= 0 or not self.is_carnivore:
            return
        
        # Only carnivores can hunt
        for other in organisms:
            # Skip self, other carnivores, or already dead organisms
            if (other == self or other.is_carnivore or other.energy <= 0 or 
                other.position is None or self.position is None):
                continue
                
            # Use radius-based collision detection
            combined_radius = (self.get_radius() + other.get_radius()) ** 2
            if self.distance_to(other) <= combined_radius:
                # Carnivores can only eat smaller or same-sized organisms
                if self.size >= other.size * 0.8:  # Can eat organisms up to 80% of own size
                    # Calculate energy gain - proportional to prey's remaining energy
                    energy_gain = other.energy * 0.7  # Conversion efficiency
                    
                    # Add energy up to max_energy
                    self.energy = min(self.max_energy, self.energy + energy_gain)
                    
                    # Update stats
                    self.organisms_consumed += 1
                    
                    # Kill the prey
                    other.energy = 0
                    
                    # Update time-tracking for hunting patterns
                    if self.steps_since_last_hunt > 0:
                        self.avg_steps_between_hunts.append(self.steps_since_last_hunt)
                        # Keep history limited to prevent memory buildup
                        if len(self.avg_steps_between_hunts) > 10:
                            self.avg_steps_between_hunts.pop(0)
                    self.steps_since_last_hunt = 0
                    
                    # Add fitness bonus for successful hunting
                    self.fitness_bonus += 25
                    
                    # Only print detailed logs if logging level is set to detailed
                    if self.logging_level == 'detailed':
                        print(f"Carnivore consumed organism at {self.position}, energy now {self.energy:.2f}")
                    
                    # Only consume one organism per update
                    break
