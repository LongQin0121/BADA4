"""
Basic simulation example for the ATM bi-level MPC system.

This example demonstrates how to set up and run a simple simulation
of the air traffic management system for Stockholm Arlanda airport.
"""
import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation import ATMSimulation, SimulationConfig
from models.aircraft import Aircraft, AircraftState, AircraftConfig, FlightPhase
from visualization.plotter import ATMVisualizer, create_summary_report


def create_test_aircraft() -> Aircraft:
    """Create a test aircraft with predefined parameters."""
    # Initial position: 50 km northeast of merge point
    initial_position = np.array([35000.0, 35000.0, 2500.0])
    
    # Initial velocity: heading southwest towards merge point
    heading = np.radians(225)  # Southwest
    speed = 200.0  # m/s
    initial_velocity = np.array([
        speed * np.cos(heading),
        speed * np.sin(heading),
        -2.0  # Slight descent
    ])
    
    initial_state = AircraftState(
        position=initial_position,
        velocity=initial_velocity,
        heading=heading,
        altitude=2500.0,
        speed=speed,
        timestamp=0.0
    )
    
    config = AircraftConfig(
        max_speed=260.0,
        min_speed=75.0,
        max_turn_rate=3.0,
        max_climb_rate=12.0,
        max_descent_rate=-8.0
    )
    
    aircraft = Aircraft(
        aircraft_id="SAS001",
        initial_state=initial_state,
        config=config,
        flight_phase=FlightPhase.ARRIVAL
    )
    
    return aircraft


def run_basic_simulation():
    """Run a basic simulation with predefined parameters."""
    print("Setting up basic ATM simulation...")
    
    # Create simulation configuration
    config = SimulationConfig(
        time_step=2.0,      # Larger time step for faster execution
        total_time=300.0,   # 5 minutes for demonstration
        aircraft_arrival_rate=0.01,  # Lower rate for demonstration
        max_aircraft=5,     # Fewer aircraft for demonstration
        log_interval=30.0
    )
    
    # Create simulation
    simulation = ATMSimulation(config)
    
    # Add a test aircraft manually
    test_aircraft = create_test_aircraft()
    simulation.add_aircraft(test_aircraft)
    
    print(f"Starting simulation for {config.total_time} seconds...")
    print("Initial aircraft:", test_aircraft.id)
    
    # Run simulation
    results = simulation.run_simulation()
    
    return results, simulation


def analyze_results(results, simulation):
    """Analyze and display simulation results."""
    print("\n" + "="*60)
    print("SIMULATION ANALYSIS")
    print("="*60)
    
    # Create summary report
    summary = create_summary_report(results)
    print(summary)
    
    # Create visualizations
    print("Generating visualizations...")
    
    visualizer = ATMVisualizer(simulation.point_merge_system)
    
    try:
        # Plot final state
        final_state_fig = visualizer.plot_terminal_area(
            simulation.aircraft_list, 
            show_legs=True, 
            show_separation_circles=True
        )
        final_state_fig.suptitle("Final Simulation State")
        final_state_fig.show()
        
        # Plot trajectory history if available
        if 'trajectory_data' in results and results['trajectory_data']:
            traj_fig = visualizer.plot_trajectory_history(results['trajectory_data'])
            traj_fig.show()
        
        # Plot separation events if any occurred
        if 'events_log' in results and results['events_log']:
            sep_events = [e for e in results['events_log'] 
                         if e['type'] == 'separation_event']
            if sep_events:
                sep_fig = visualizer.plot_separation_events(results['events_log'])
                sep_fig.show()
        
        # Plot system performance
        if 'statistics_data' in results and results['statistics_data']:
            perf_fig = visualizer.plot_system_performance(results['statistics_data'])
            perf_fig.show()
        
        # Save plots
        visualizer.save_plots(results, "example_output/")
        
        print("Visualizations generated and saved to 'example_output/' directory")
        
    except ImportError as e:
        print(f"Visualization skipped (matplotlib not available): {e}")
    except Exception as e:
        print(f"Error generating visualizations: {e}")


def demonstrate_point_merge_system(simulation):
    """Demonstrate the point merge system functionality."""
    print("\n" + "="*50)
    print("POINT MERGE SYSTEM DEMONSTRATION")
    print("="*50)
    
    pms = simulation.point_merge_system
    
    # Display merge point information
    print(f"Merge Point: {pms.merge_point.name}")
    print(f"Position: {pms.merge_point.position}")
    print(f"Approach Altitude: {pms.merge_point.approach_altitude}m")
    
    # Display sequencing legs
    print("\nSequencing Legs:")
    for leg_type, leg in pms.sequencing_legs.items():
        print(f"  {leg_type.value.upper()}:")
        print(f"    Entry Point: {leg.entry_point}")
        print(f"    Length: {leg.length/1000:.1f} km")
        print(f"    Speed Constraint: {leg.speed_constraint} m/s")
    
    # Display current aircraft assignments
    if pms.aircraft_assignments:
        print("\nAircraft Assignments:")
        for aircraft_id, leg_type in pms.aircraft_assignments.items():
            print(f"  {aircraft_id}: {leg_type.value.upper()} leg")
    else:
        print("\nNo aircraft currently assigned to legs")
    
    # Display separation constraints
    print(f"\nSeparation Requirements:")
    print(f"  Minimum: {pms.min_separation_distance/1852:.1f} NM")
    print(f"  Time separation: {pms.min_time_separation} seconds")


def demonstrate_controllers(simulation):
    """Demonstrate controller functionality."""
    print("\n" + "="*50)
    print("CONTROLLER DEMONSTRATION")
    print("="*50)
    
    upper_controller = simulation.upper_controller
    lower_controller = simulation.lower_controller
    
    print("Upper-Level Controller (BL-RC):")
    print(f"  Strategic horizon: {upper_controller.control_horizon.strategic_horizon}s")
    print(f"  Tactical horizon: {upper_controller.control_horizon.tactical_horizon}s")
    print(f"  Update interval: {upper_controller.control_horizon.update_interval}s")
    
    print("\nLower-Level Controller (BL-DC):")
    print(f"  Prediction horizon: {lower_controller.tactical_horizon.prediction_horizon}s")
    print(f"  Control horizon: {lower_controller.tactical_horizon.control_horizon}s")
    print(f"  Time step: {lower_controller.tactical_horizon.time_step}s")
    
    # Display current setpoints if available
    if upper_controller.current_setpoints:
        print(f"\nCurrent setpoints available for {len(upper_controller.current_setpoints)} aircraft")
    
    # Display control history
    if lower_controller.control_history:
        print(f"Control history maintained for {len(lower_controller.control_history)} aircraft")


def main():
    """Main example execution."""
    print("ATM Bi-Level MPC System - Basic Example")
    print("="*50)
    
    try:
        # Run simulation
        results, simulation = run_basic_simulation()
        
        # Demonstrate system components
        demonstrate_point_merge_system(simulation)
        demonstrate_controllers(simulation)
        
        # Analyze results
        analyze_results(results, simulation)
        
        print("\n" + "="*50)
        print("Example completed successfully!")
        print("Check the 'example_output/' directory for visualization plots.")
        
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()