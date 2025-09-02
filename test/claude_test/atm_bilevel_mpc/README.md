# Air Traffic Management with Bi-Level Model Predictive Control

A Python implementation of a bi-level Model Predictive Control (MPC) system for air traffic management in terminal areas, specifically designed for Stockholm Arlanda airport's point merge system.

## Overview

This system implements a hierarchical control architecture with two levels:

- **Upper-Level Controller (BL-RC)**: Strategic planning over longer horizons (20 minutes), generating conflict-free reference trajectories and set-points
- **Lower-Level Controller (BL-DC)**: Tactical control with high update frequency (1 second), tracking set-points while ensuring immediate safety constraints

The system replaces traditional air traffic controller instructions with automated turn point calculations, transmitted to aircraft via datalink/CPDLC.

## Key Features

- **Bi-level MPC Architecture**: Hierarchical control with strategic and tactical layers
- **Point Merge System**: Implementation of Stockholm Arlanda terminal area procedures
- **Separation Constraint Enforcement**: Automatic 3 nautical mile minimum separation
- **CPDLC Communication**: Simulated datalink for instruction transmission
- **Real-time Visualization**: Comprehensive plotting and animation capabilities
- **Comprehensive Testing**: Unit tests and example scenarios

## System Components

### Aircraft Model (`src/models/aircraft.py`)
- Point-mass aircraft dynamics
- Configurable performance parameters
- State tracking and trajectory prediction

### Point Merge System (`src/systems/point_merge.py`)
- Stockholm Arlanda terminal area layout
- Sequencing leg management
- Turn point calculation
- Merge sequence optimization

### Controllers
- **Upper Level** (`src/controllers/upper_level.py`): Strategic planning and set-point generation
- **Lower Level** (`src/controllers/lower_level.py`): Tactical MPC with CasADi optimization

### Safety Systems
- **Separation Monitor** (`src/utils/separation_monitor.py`): Real-time conflict detection and resolution
- **Emergency Procedures**: Immediate avoidance maneuvers

### Communication
- **CPDLC Interface** (`src/communication/cpdlc_interface.py`): Datalink message simulation

### Visualization
- **Real-time Plotting** (`src/visualization/plotter.py`): Trajectory visualization and system analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd atm_bilevel_mpc
```

2. Install dependencies:
```bash
pip install -e .
```

### Required Dependencies
- `numpy`: Numerical computations
- `scipy`: Scientific computing
- `matplotlib`: Visualization
- `casadi`: Optimization (MPC)
- `cvxpy`: Convex optimization
- `pandas`: Data analysis
- `pydantic`: Data validation

## Quick Start

### Basic Simulation

```python
from src.simulation import ATMSimulation, SimulationConfig

# Create simulation configuration
config = SimulationConfig(
    total_time=1800.0,      # 30 minutes
    aircraft_arrival_rate=0.033,  # ~120 aircraft/hour
    max_aircraft=15
)

# Run simulation
simulation = ATMSimulation(config)
results = simulation.run_simulation()

# Print summary
print(f"Processed {results['metrics']['total_aircraft_processed']} aircraft")
print(f"Separation violations: {results['metrics']['separation_violations']}")
```

### Running the Example

```bash
cd examples
python basic_simulation.py
```

This will run a 15-minute simulation and generate visualization plots in the `example_output/` directory.

## Usage Examples

### Creating Aircraft

```python
from src.models.aircraft import Aircraft, AircraftState, AircraftConfig

# Create aircraft configuration
config = AircraftConfig(
    max_speed=250.0,    # m/s
    min_speed=70.0,     # m/s
    max_turn_rate=3.0   # degrees/second
)

# Create initial state
state = AircraftState(
    position=np.array([10000, 5000, 2000]),  # x, y, z in meters
    velocity=np.array([200, 100, 0]),        # vx, vy, vz in m/s
    heading=np.pi/4,    # radians
    altitude=2000.0,    # meters
    speed=220.0,        # m/s
    timestamp=0.0       # seconds
)

# Create aircraft
aircraft = Aircraft("AC001", state, config)
```

### Point Merge System

```python
from src.systems.point_merge import PointMergeSystem

# Initialize point merge system
pms = PointMergeSystem()

# Assign aircraft to sequencing leg
leg_assignment = pms.assign_aircraft_to_leg(aircraft)

# Generate reference trajectory
waypoints, times = pms.generate_reference_trajectory(aircraft)

# Calculate turn point for sequencing
turn_point = pms.get_turn_point(aircraft, target_sequence_position=2)
```

### Bi-Level Control

```python
from src.controllers.upper_level import UpperLevelController
from src.controllers.lower_level import LowerLevelController

# Initialize controllers
upper_controller = UpperLevelController(pms)
lower_controller = LowerLevelController()

# Strategic planning (upper level)
setpoints = upper_controller.plan_strategic_trajectories(aircraft_list, current_time)

# Tactical control (lower level)
current_setpoint = upper_controller.get_current_setpoint(aircraft.id, current_time)
control_input = lower_controller.compute_control(
    aircraft, current_setpoint, nearby_aircraft, current_time
)
```

### Visualization

```python
from src.visualization.plotter import ATMVisualizer

# Create visualizer
visualizer = ATMVisualizer(pms)

# Plot terminal area
fig = visualizer.plot_terminal_area(aircraft_list, show_legs=True)

# Plot trajectory history
fig = visualizer.plot_trajectory_history(trajectory_data)

# Create animation
animation = visualizer.create_animation(trajectory_data)
```

## System Architecture

```
┌─────────────────────────────────────────┐
│           Upper-Level Controller         │
│              (BL-RC)                    │
│  • Strategic planning (20 min horizon)  │
│  • Conflict-free trajectories          │
│  • Set-point generation                │
└─────────────┬───────────────────────────┘
              │ Set-points
              ▼
┌─────────────────────────────────────────┐
│           Lower-Level Controller         │
│              (BL-DC)                    │
│  • Tactical control (3 min horizon)     │
│  • MPC optimization                     │
│  • Real-time safety                    │
└─────────────┬───────────────────────────┘
              │ Control commands
              ▼
┌─────────────────────────────────────────┐
│            Aircraft Dynamics            │
│  • Point-mass model                     │
│  • Performance constraints              │
│  • State propagation                    │
└─────────────────────────────────────────┘
```

## Configuration

### Simulation Parameters

```python
config = SimulationConfig(
    time_step=1.0,                  # Simulation time step
    total_time=3600.0,              # Total simulation time
    upper_level_update_interval=60.0, # Upper controller update
    lower_level_update_interval=1.0,  # Lower controller update
    aircraft_arrival_rate=0.05,     # Aircraft per second
    max_aircraft=20                 # Maximum aircraft in system
)
```

### Aircraft Configuration

```python
config = AircraftConfig(
    max_speed=250.0,        # Maximum speed (m/s)
    min_speed=70.0,         # Minimum speed (m/s)
    max_turn_rate=3.0,      # Maximum turn rate (deg/s)
    max_climb_rate=15.0,    # Maximum climb rate (m/s)
    max_descent_rate=-10.0  # Maximum descent rate (m/s)
)
```

### Separation Constraints

```python
constraints = SeparationConstraints(
    minimum_separation=5556.0,    # 3 NM in meters
    warning_separation=7408.0,    # 4 NM warning
    critical_separation=3704.0,   # 2 NM critical
    emergency_separation=1852.0,  # 1 NM emergency
    vertical_separation=300.0     # 1000 feet
)
```

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Run specific tests:

```bash
python -m pytest tests/test_aircraft_model.py -v
```

## Performance

The system is designed for real-time operation:

- **Real-time factor**: Typically 10-50x faster than real-time
- **Aircraft capacity**: Handles 15-20 aircraft simultaneously
- **Update frequency**: 1 Hz for tactical control, 1/60 Hz for strategic planning
- **Separation monitoring**: Continuous with 5-second prediction resolution

## Limitations and Future Work

### Current Limitations
- Simplified aircraft dynamics (point-mass model)
- No wind or weather effects
- Simplified CPDLC communication model
- Single runway operations only

### Future Enhancements
- Advanced aircraft performance models
- Weather integration
- Multi-runway operations
- Machine learning integration for traffic prediction
- Real ATC system integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. **Point Merge System**: EUROCONTROL Point Merge Implementation Guidelines
2. **Model Predictive Control**: Camacho, E.F. and Bordons, C., "Model Predictive Control"
3. **Air Traffic Management**: ICAO Doc 9426 - Air Traffic Services Planning Manual
4. **Bi-level Optimization**: Colson, B., et al., "An overview of bilevel optimization"

## Contact

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

**Note**: This is a research prototype for educational and research purposes. It is not certified for operational air traffic control use.