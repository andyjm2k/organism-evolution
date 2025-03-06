import pygame
import random
import math
import os
import colorsys

class Renderer:
    def __init__(self, size):
        pygame.init()
        
        # Define scoreboard width first
        self.scoreboard_width = 300
        
        # Create single window with extra width for scoreboard
        total_width = size + self.scoreboard_width
        self.screen = pygame.display.set_mode((total_width, size))
        pygame.display.set_caption("Evolution Simulation")
        
        # Define scoreboard area as a rect
        self.scoreboard_rect = pygame.Rect(size, 0, self.scoreboard_width, size)
        
        self.clock = pygame.time.Clock()
        self.generation = 0
        self.species_colors = {}      # Cache for species colors
        self.species_surfaces = {}    # Cache for rendered species visuals
        self.text_surfaces = {}       # Cache for rendered text
        self.last_cache_clear = 0     # Track when we last cleared the cache
        print(f"Renderer initialized with screen size: {size}x{size}")

        # Enhanced fonts and colors
        pygame.font.init()
        try:
            self.title_font = pygame.font.Font('assets/fonts/Roboto-Bold.ttf', 36)
            self.header_font = pygame.font.Font('assets/fonts/Roboto-Medium.ttf', 28)
            self.font = pygame.font.Font('assets/fonts/Roboto-Regular.ttf', 20)
        except:
            print("Falling back to default font")
            self.title_font = pygame.font.Font(None, 36)
            self.header_font = pygame.font.Font(None, 28)
            self.font = pygame.font.Font(None, 20)

        # Color scheme
        self.colors = {
            'background': (255, 255, 255),  # White
            'food': (0, 255, 0),           # Green
            'text': (0, 0, 0),             # Black
            'card': (240, 240, 240),       # Light gray
            'border': (200, 200, 200),     # Medium gray
        }

    def get_species_color(self, species_id, is_carnivore):
        """Generate a consistent color for each species with distinction between types"""
        if species_id not in self.species_colors:
            # Use hash of species_id for consistent but random-looking values
            hash_val = hash(species_id)
            random.seed(hash_val)
            
            if is_carnivore:
                # Carnivores: light yellow (60°) through orange to dark red (0°)
                hue = random.uniform(0, 60) / 360.0  # Convert to 0-1 range
                saturation = random.uniform(0.7, 1.0)  # High saturation for vivid colors
                lightness = random.uniform(0.3, 0.7)   # Vary lightness for depth
            else:
                # Herbivores: light green (120°) through to dark brown (30°)
                hue = random.uniform(30, 120) / 360.0  # Convert to 0-1 range
                saturation = random.uniform(0.4, 0.8)  # More earthy saturation
                lightness = random.uniform(0.2, 0.6)   # Darker for natural look
            
            # Convert HSL to RGB
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            # Convert from 0-1 range to 0-255 range
            self.species_colors[species_id] = tuple(int(x * 255) for x in rgb)
            
            # Reset random seed to avoid affecting other random operations
            random.seed()
        
        return self.species_colors[species_id]

    def render(self, organisms, food_items):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.VIDEORESIZE:
                # Handle scoreboard window resizing
                if event.w != self.scoreboard_width or event.h != self.scoreboard_height:
                    self.scoreboard_width = max(300, event.w)
                    self.scoreboard_height = max(400, event.h)

        # Clear screen with white - ensure this runs every frame
        self.screen.fill((255, 255, 255))  

        # Draw breeding safe zone boundary (subtle indication)
        main_width = self.screen.get_width() - self.scoreboard_width
        main_height = self.screen.get_height()
        
        # Calculate breeding boundary (10% inset from edges)
        breeding_boundary_x = main_width * 0.1
        breeding_boundary_y = main_height * 0.1
        breeding_width = main_width - (breeding_boundary_x * 2)
        breeding_height = main_height - (breeding_boundary_y * 2)
        
        # Draw a subtle rectangle showing breeding safe zone
        breeding_rect = pygame.Rect(breeding_boundary_x, breeding_boundary_y, breeding_width, breeding_height)
        pygame.draw.rect(self.screen, (240, 250, 240), breeding_rect, 1)  # Very light green outline
        
        # Add a small indicator about breeding zone
        if not hasattr(self, 'breeding_zone_text'):
            self.breeding_zone_text = self.font.render("Breeding Safe Zone", True, (100, 180, 100))
        self.screen.blit(self.breeding_zone_text, (breeding_boundary_x + 5, breeding_boundary_y - 25))

        # Limit the number of food items rendered to conserve memory if there are too many
        render_limit = 500  # Maximum number of objects to render
        
        # Render food items only if they exist and are alive
        if food_items:
            # Only render a subset if there are too many food items
            render_food = food_items
            if len(food_items) > render_limit:
                render_food = food_items[:render_limit]
                
            for food in render_food:
                if food.position is not None:  # Only render if position exists
                    pos = (int(food.position[0]), int(food.position[1]))
                    pygame.draw.circle(self.screen, (0, 255, 0), pos, 4)
                
        # Render organisms with species-specific colors
        if organisms:
            # Only render a subset if there are too many organisms
            render_organisms = organisms
            if len(organisms) > render_limit:
                render_organisms = organisms[:render_limit]
                
            for organism in render_organisms:
                if organism.position is not None and organism.energy > 0:
                    pos = (int(organism.position[0]), int(organism.position[1]))
                    # Use existing color if already in cache to avoid unnecessary calculations
                    species_id_str = str(organism.species_id)
                    if species_id_str in self.species_colors:
                        color = self.species_colors[species_id_str]
                    else:
                        color = self.get_species_color(species_id_str, organism.is_carnivore)
                    
                    self.draw_organism_with_spikes(
                        self.screen, pos, color,
                        organism.get_radius(),
                        organism.num_spikes,
                        organism.get_active_node_count(),
                        organism.is_carnivore
                    )

        # Update debug info to show species count
        if organisms:
            species_count = len(set(org.species_id for org in organisms))
            debug_text = f"Food: {len(food_items)} | Gen: {self.generation} | Species: {species_count}"
            text = self.font.render(debug_text, True, (0, 0, 0))
            self.screen.blit(text, (10, 10))
            # Add memory usage info to debug display
            try:
                import psutil
                import os
                memory_info = psutil.Process(os.getpid()).memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                memory_text = f"Memory: {memory_mb:.1f} MB"
                memory_display = self.font.render(memory_text, True, (0, 0, 0))
                self.screen.blit(memory_display, (10, 40))
            except ImportError:
                pass

        # Render scoreboard
        self._render_scoreboard()
        
        # Update the display and limit framerate
        pygame.display.flip()
        self.clock.tick(60)
        
        # Handle events again to catch any that occurred during rendering
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.cleanup_resources()
                return False
        return True

    def set_generation(self, generation):
        """Update the current generation number"""
        # Clear resources if generation changes to prevent memory leaks
        if generation != self.generation:
            self.cleanup_resources()
            
        self.generation = generation
        
        # Clear the species_colors cache every 5 generations to avoid memory growth (more aggressive)
        if generation % 5 == 0 and generation > self.last_cache_clear:
            self.species_colors.clear()
            self.species_surfaces.clear()
            self.text_surfaces.clear()
            self.last_cache_clear = generation
            # Force a garbage collection cycle
            import gc
            gc.collect()
            
    def cleanup_resources(self):
        """Clean up renderer resources to prevent memory leaks"""
        # Explicitly clear surface caches
        for surface in self.species_surfaces.values():
            # Attempt to help pygame release the surface
            surface.fill((0, 0, 0, 0))
            del surface
        
        for surface in self.text_surfaces.values():
            # Attempt to help pygame release the surface
            del surface
        
        # Clear all cache dictionaries
        self.species_colors.clear()
        self.species_surfaces.clear()
        self.text_surfaces.clear()
        
        # Force the Python garbage collector to run
        import gc
        gc.collect()
        
        # Explicitly clean up pygame surfaces if they exist
        try:
            # This helps release video memory
            pygame.display.update()
            self.screen.fill((0, 0, 0))  # Fill with black
            pygame.display.update()      # Update display
            pygame.time.delay(10)        # Short delay to allow cleanup
            
            # Create a small temporary surface to help flush memory
            temp_surface = pygame.Surface((10, 10))
            temp_surface.fill((0, 0, 0))
            self.screen.blit(temp_surface, (0, 0))
            pygame.display.update()
            del temp_surface
            
            # Force another garbage collection cycle after pygame operations
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Failed to clean up pygame resources: {e}")

    def draw_organism_with_spikes(self, screen, position, color, base_radius, num_spikes, num_nodes, is_carnivore):
        """Draw an organism with spikes that reflect its attributes"""
        # Calculate dynamic radius based on node count
        radius = min(20, max(5, base_radius + (num_nodes / 10)))
        
        # Draw the base circle
        pygame.draw.circle(screen, color, position, radius)
        
        if num_spikes > 0:
            # Calculate spike positions around the circle
            center_x, center_y = position
            spike_length = radius * (0.6 if is_carnivore else 0.4)  # Carnivores get longer spikes
            
            for i in range(num_spikes):
                angle = (2 * math.pi * i) / num_spikes
                
                # Calculate spike start point (on circle)
                start_x = center_x + radius * math.cos(angle)
                start_y = center_y + radius * math.sin(angle)
                
                # Calculate spike end point (outside circle)
                end_x = center_x + (radius + spike_length) * math.cos(angle)
                end_y = center_y + (radius + spike_length) * math.sin(angle)
                
                # Draw the spike
                pygame.draw.line(screen, color, (start_x, start_y), (end_x, end_y), 2)
                
                # Add spiky tips for carnivores
                if is_carnivore:
                    pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 3)

    def get_species_visual(self, species_id, is_carnivore, radius, num_spikes, spike_length):
        """Get or create a surface with the species visual"""
        cache_key = f"{species_id}_{is_carnivore}_{radius}_{num_spikes}_{spike_length}"
        
        if cache_key not in self.species_surfaces:
            # Create a new surface for this species visual
            surface_size = int(radius * 4)  # Make sure there's room for spikes
            surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
            
            # Get species color
            color = self.get_species_color(species_id, is_carnivore)
            
            # Draw the organism on the surface
            center = (surface_size // 2, surface_size // 2)
            self.draw_organism_with_spikes(
                surface, center, color, radius, num_spikes, spike_length, is_carnivore
            )
            
            # Cache the surface
            self.species_surfaces[cache_key] = surface
            
        return self.species_surfaces[cache_key]

    def _render_scoreboard(self):
        """Render the scoreboard with a modern, clean design"""
        # Clear scoreboard area with background color
        pygame.draw.rect(self.screen, self.colors['background'], self.scoreboard_rect)    
        # Get top species
        from scoreboard import Scoreboard
        top_species = Scoreboard.get_top_species(10)  # Get top 10 species
        
        if not top_species:
            # Show placeholder text if no species recorded
            if "waiting" not in self.text_surfaces:
                self.text_surfaces["waiting"] = self.font.render("Waiting for species data...", True, self.colors['text'])
            text = self.text_surfaces["waiting"]
            text_pos = text.get_rect(centerx=self.scoreboard_rect.centerx, centery=self.scoreboard_rect.centery)
            self.screen.blit(text, text_pos)
            return

        # Render leaderboard header
        if "header" not in self.text_surfaces:
            self.text_surfaces["header"] = self.header_font.render("Top Species", True, self.colors['text'])
        self.screen.blit(self.text_surfaces["header"], (self.scoreboard_rect.x + 20, 5))
        
        # Render each species card with a limit to prevent excessive memory usage
        max_cards = 8  # Limit the number of cards to render
        y_offset = 50
        for i, (species_id, record) in enumerate(top_species[:max_cards]):
            card_height = 90
            card_rect = pygame.Rect(
                self.scoreboard_rect.x + 10,
                y_offset,
                self.scoreboard_width - 20,
                card_height
            )
            
            # Draw card background
            pygame.draw.rect(self.screen, self.colors['card'], card_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.colors['border'], card_rect, width=1, border_radius=5)
            
            # Use the stored visual properties for rendering
            radius = record.get('size', 12) / 2  # Convert size back to radius
            num_spikes = record.get('num_spikes', 3)
            spike_length = record.get('spike_length', 3)
            is_carnivore = record.get('is_carnivore', False)
            
            # Get cached species visual surface
            species_visual = self.get_species_visual(
                species_id, is_carnivore, radius, num_spikes, spike_length
            )
            
            # Calculate position to center the visual in the circle area
            visual_pos = (
                card_rect.x + 25 - species_visual.get_width() // 2,
                card_rect.y + card_height // 2 - species_visual.get_height() // 2
            )
            
            # Blit the cached visual
            self.screen.blit(species_visual, visual_pos)
            
            # Species name with custom color based on rank
            name = record['scientific_name'] or f'Species {species_id}'
            name_color = self.get_species_color(species_id, record.get('is_carnivore', False))
            
            # Use cached text surface if available
            text_key = f"name_{name}"
            if text_key not in self.text_surfaces:
                self.text_surfaces[text_key] = self.font.render(name, True, name_color)
            self.screen.blit(self.text_surfaces[text_key], (card_rect.x + 50, card_rect.y + 15))
            
            # Fitness score
            fitness_text = f"Fitness: {int(record['highest_fitness']):,}"
            text_key = f"fitness_{fitness_text}"
            if text_key not in self.text_surfaces:
                self.text_surfaces[text_key] = self.font.render(fitness_text, True, self.colors['text'])
            self.screen.blit(self.text_surfaces[text_key], (card_rect.x + 50, card_rect.y + 40))
            
            # Generation range
            gen_text = f"Gen {record['first_seen']} → {record['last_seen']}"
            text_key = f"gen_{gen_text}"
            if text_key not in self.text_surfaces:
                self.text_surfaces[text_key] = self.font.render(gen_text, True, self.colors['text'])
            self.screen.blit(self.text_surfaces[text_key], (card_rect.x + 50, card_rect.y + 65))
            
            y_offset += card_height + 10
            
        # Limit cache size to prevent unlimited growth
        if len(self.text_surfaces) > 100:  # Arbitrary limit
            # Keep only essential surfaces
            essential_keys = ["header", "waiting"]
            self.text_surfaces = {k: v for k, v in self.text_surfaces.items() if k in essential_keys}
