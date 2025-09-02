#!/usr/bin/env python3
"""
Quick test to verify the system works.
"""
import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation import ATMSimulation, SimulationConfig

def quick_test():
    """Run a very short simulation to test basic functionality."""
    print("Running quick test...")
    
    # Create minimal configuration
    config = SimulationConfig(
        time_step=5.0,      # Larger time step for faster execution
        total_time=60.0,    # Only 1 minute
        aircraft_arrival_rate=0.01,  # Very low rate
        max_aircraft=3,     # Limit aircraft
        log_interval=20.0
    )
    
    # Create and run simulation
    simulation = ATMSimulation(config)
    
    try:
        results = simulation.run_simulation()
        
        print("\n" + "="*40)
        print("QUICK TEST RESULTS")
        print("="*40)
        print(f"Aircraft processed: {results['metrics']['total_aircraft_processed']}")
        print(f"Separation violations: {results['metrics']['separation_violations']}")
        print(f"Final aircraft count: {results['final_aircraft_count']}")
        print(f"Computation time: {results.get('computation_time', 0):.2f}s")
        print("Test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)