# NEAT Simulation

This project simulates the genetic evolution of simple cell organisms using the NEAT algorithm. Organisms compete for survival in a petri dish by consuming food and each other, developing traits to gain advantages.

## Features

- Configurable simulation parameters
- Visual rendering of the simulation
- Genetic evolution using NEAT
- Terminal dashboard for headless mode

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation with visual rendering (default):
   ```bash
   python src/main.py
   ```

3. Run the simulation in headless mode (no visual rendering):
   ```bash
   python src/main.py render=false
   ```
   
   In headless mode, a terminal dashboard will display the top species after each generation.

## Running Options

- **With Rendering**: The simulation will display a visual representation of the organisms and food in the environment, along with a scoreboard showing the top species.
  ```bash
  python src/main.py render=true
  ```

- **Without Rendering (Headless Mode)**: The simulation will run without visual output, displaying only a terminal dashboard of the top species after each generation. This mode is useful for faster simulations or when running on servers without a display.
  ```bash
  python src/main.py render=false
  ```

## Configuration

- `config/neat-config.ini`: NEAT algorithm settings
- `config/simulation-config.json`: Simulation parameters

## License

MIT License
