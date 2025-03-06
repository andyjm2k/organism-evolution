import neat
import json
import pygame
import sys
from simulation import Simulation
from organism import Organism
from scoreboard import Scoreboard

def run_simulation(render=True, logging_level="normal", dashboard_level="normal"):
    # Parse command line arguments
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.lower().startswith('render='):
                render_value = arg.split('=')[1].lower()
                render = render_value == 'true'
            elif arg.lower().startswith('logging='):
                logging_value = arg.split('=')[1].lower()
                logging_level = logging_value
            elif arg.lower().startswith('dashboard='):
                dashboard_value = arg.split('=')[1].lower()
                dashboard_level = dashboard_value
    
    # Load configuration
    with open('config/simulation-config.json') as f:
        sim_config = json.load(f)
    
    # Update rendering and logging settings in config
    sim_config['render'] = render
    sim_config['logging_level'] = logging_level
    sim_config['dashboard_level'] = dashboard_level
    
    # Initialize pygame and set up display only if rendering is enabled
    if render:
        pygame.init()
        screen_width = sim_config['screen_width']
        screen_height = sim_config['screen_height'] 
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Organism Evolution")
        sim_config['screen'] = screen
    else:
        # Still need to initialize pygame for event handling, but no display
        pygame.init()
        sim_config['screen'] = None
        print("\n=== Running Simulation in Headless Mode ===")
        print("Rendering disabled. Species dashboard will be displayed in terminal.")

    # Print logging level
    if logging_level == "detailed":
        print(f"Logging level: DETAILED - All debug messages will be shown")
    else:
        print(f"Logging level: NORMAL - Only essential messages will be shown")
        
    # Print dashboard level
    if dashboard_level == "detailed":
        print(f"Dashboard level: DETAILED - Full dashboard statistics will be shown")
    elif dashboard_level == "minimal":
        print(f"Dashboard level: MINIMAL - Only summary information will be shown")
    else:
        print(f"Dashboard level: NORMAL - Standard dashboard statistics will be shown")

    # Load NEAT configuration
    config_path = 'config/neat-config.ini'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Initialize scoreboard
    Scoreboard.initialize()

    # Create the simulation
    simulation = Simulation(config, sim_config)
    
    try:
        simulation.run()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        # Final cleanup to ensure all resources are released
        print("\nPerforming final cleanup...")
        
        # Clear any remaining pygame events
        pygame.event.get()
        
        # If there's a renderer, clear its resources
        if hasattr(simulation, 'renderer') and simulation.renderer:
            simulation.renderer.cleanup_resources()
        
        # Force final garbage collection
        import gc
        gc.collect()
        
        # Quit pygame
        pygame.quit()
        
        print("Cleanup complete.")
        
    # Display final scoreboard results - show full dashboard for headless mode
    # and the new comprehensive summary for both logging modes
    print("\nGenerating final simulation summary...")
    Scoreboard.display_final_summary(logging_level)

if __name__ == '__main__':
    run_simulation()
