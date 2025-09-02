"""
Main simulation runner for bi-level MPC air traffic management system.

This module integrates all components and provides the main simulation loop
for the Stockholm airport terminal area air traffic management system.
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
import time
from dataclasses import dataclass

try:
    from .models.aircraft import Aircraft, AircraftState, AircraftConfig, FlightPhase
    from .systems.point_merge import PointMergeSystem, SequenceType
    from .controllers.upper_level import UpperLevelController, SetPoint
    from .controllers.lower_level import LowerLevelController, ControlInput
    from .utils.separation_monitor import SeparationMonitor, SeparationConstraints
except ImportError:
    from models.aircraft import Aircraft, AircraftState, AircraftConfig, FlightPhase
    from systems.point_merge import PointMergeSystem, SequenceType
    from controllers.upper_level import UpperLevelController, SetPoint
    from controllers.lower_level import LowerLevelController, ControlInput
    from utils.separation_monitor import SeparationMonitor, SeparationConstraints


@dataclass
class SimulationConfig:
    """Simulation configuration parameters."""
    time_step: float = 1.0                  # Simulation time step in seconds
    total_time: float = 3600.0              # Total simulation time (1 hour)
    upper_level_update_interval: float = 60.0  # Upper level update every minute
    lower_level_update_interval: float = 1.0   # Lower level update every second
    
    # Aircraft generation
    aircraft_arrival_rate: float = 0.05     # Aircraft per second (180/hour)
    max_aircraft: int = 20                  # Maximum aircraft in system
    
    # Logging and output
    log_interval: float = 10.0              # Log every 10 seconds
    save_trajectory: bool = True
    save_statistics: bool = True


class ATMSimulation:
    """
    Main simulation class for air traffic management system.
    
    This class coordinates all components of the bi-level MPC system,
    manages aircraft lifecycle, and provides simulation control.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize simulation.
        
        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        
        # Initialize system components
        self.point_merge_system = PointMergeSystem()
        self.upper_controller = UpperLevelController(self.point_merge_system)
        self.lower_controller = LowerLevelController()
        self.separation_monitor = SeparationMonitor()
        
        # Simulation state
        self.current_time = 0.0
        self.aircraft_list: List[Aircraft] = []
        self.aircraft_counter = 0
        
        # Control state
        self.last_upper_update = 0.0
        self.current_setpoints: Dict[str, List[SetPoint]] = {}
        
        # Data logging
        self.trajectory_data: Dict[str, List[Dict]] = {}
        self.statistics_data: List[Dict] = []
        self.events_log: List[Dict] = []
        
        # Performance metrics
        self.metrics = {
            'total_aircraft_processed': 0,
            'average_flight_time': 0.0,
            'separation_violations': 0,
            'emergency_events': 0,
            'fuel_consumption': 0.0,
            'delays': 0.0
        }
        
    def run_simulation(self) -> Dict:
        """
        Run the complete simulation.
        
        Returns:
            Simulation results and statistics
        """
        print(f"Starting ATM simulation for {self.config.total_time} seconds...")
        
        start_time = time.time()
        
        while self.current_time < self.config.total_time:
            self._simulation_step()
            self.current_time += self.config.time_step
            
            # Log progress
            if self.current_time % self.config.log_interval == 0:
                self._log_progress()
        
        end_time = time.time()
        
        # Finalize simulation
        results = self._finalize_simulation()
        results['computation_time'] = end_time - start_time
        
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        return results
    
    def _simulation_step(self) -> None:
        """Execute one simulation time step."""
        # 1. Generate new aircraft if needed
        self._generate_aircraft()
        
        # 2. Update upper-level controller (strategic planning)
        if (self.current_time - self.last_upper_update >= 
            self.config.upper_level_update_interval):
            self._update_upper_level_control()
            self.last_upper_update = self.current_time
        
        # 3. Update lower-level controller (tactical control)
        self._update_lower_level_control()
        
        # 4. Monitor separation constraints
        self._monitor_separation()
        
        # 5. Update aircraft dynamics
        self._update_aircraft_dynamics()
        
        # 6. Remove aircraft that have completed their flight
        self._remove_completed_aircraft()
        
        # 7. Log data
        if self.config.save_trajectory:
            self._log_trajectory_data()
    
    def _generate_aircraft(self) -> None:
        """Generate new aircraft arrivals."""
        # Simple Poisson process for aircraft generation
        if (len(self.aircraft_list) < self.config.max_aircraft and
            np.random.random() < self.config.aircraft_arrival_rate * self.config.time_step):
            
            new_aircraft = self._create_new_aircraft()
            self.aircraft_list.append(new_aircraft)
            self.aircraft_counter += 1
            
            # Initialize trajectory logging
            if self.config.save_trajectory:
                self.trajectory_data[new_aircraft.id] = []
    
    def _create_new_aircraft(self) -> Aircraft:
        """Create a new aircraft with random initial conditions."""
        aircraft_id = f"AC{self.aircraft_counter:03d}"
        
        # Random entry point around the terminal area
        entry_angle = np.random.uniform(0, 2 * np.pi)
        entry_distance = np.random.uniform(40000, 60000)  # 40-60 km from center
        
        initial_position = np.array([
            entry_distance * np.cos(entry_angle),
            entry_distance * np.sin(entry_angle),
            np.random.uniform(2000, 4000)  # 2-4 km altitude
        ])
        
        # Initial heading towards airport (rough)
        heading_to_center = np.arctan2(-initial_position[1], -initial_position[0])
        heading_variation = np.random.uniform(-np.pi/6, np.pi/6)  # ±30 degrees
        initial_heading = heading_to_center + heading_variation
        
        # Initial speed
        initial_speed = np.random.uniform(180, 250)  # 180-250 m/s
        
        # Initial velocity
        initial_velocity = np.array([
            initial_speed * np.cos(initial_heading),
            initial_speed * np.sin(initial_heading),
            np.random.uniform(-5, 5)  # Small vertical velocity
        ])
        
        initial_state = AircraftState(
            position=initial_position,
            velocity=initial_velocity,
            heading=initial_heading,
            altitude=initial_position[2],
            speed=initial_speed,
            timestamp=self.current_time
        )
        
        # Random aircraft configuration
        config = AircraftConfig(
            max_speed=np.random.uniform(240, 260),
            min_speed=np.random.uniform(65, 75),
            max_turn_rate=np.random.uniform(2.5, 3.5),
            max_climb_rate=np.random.uniform(12, 18),
            max_descent_rate=np.random.uniform(-12, -8)
        )
        
        aircraft = Aircraft(
            aircraft_id=aircraft_id,
            initial_state=initial_state,
            config=config,
            flight_phase=FlightPhase.ARRIVAL
        )
        
        return aircraft
    
    def _update_upper_level_control(self) -> None:
        """Update upper-level controller."""
        if not self.aircraft_list:
            return
        
        # Update strategic planning
        self.current_setpoints = self.upper_controller.update_setpoints(
            self.aircraft_list, self.current_time
        )
        
        # Log strategic decisions
        self._log_strategic_decisions()
    
    def _update_lower_level_control(self) -> None:
        """Update lower-level controller for all aircraft."""
        for aircraft in self.aircraft_list:
            # Get current setpoint
            current_setpoint = self.upper_controller.get_current_setpoint(
                aircraft.id, self.current_time
            )
            
            if current_setpoint:
                # Get nearby aircraft for conflict avoidance
                nearby_aircraft = self._get_nearby_aircraft(aircraft)
                
                # Compute control input
                control_input = self.lower_controller.compute_control(
                    aircraft, current_setpoint, nearby_aircraft, self.current_time
                )
                
                # Apply control input to aircraft
                self._apply_control_input(aircraft, control_input)
    
    def _get_nearby_aircraft(self, reference_aircraft: Aircraft, 
                           radius: float = 20000.0) -> List[Aircraft]:
        """Get aircraft within specified radius of reference aircraft."""
        nearby = []
        
        for aircraft in self.aircraft_list:
            if aircraft.id != reference_aircraft.id:
                distance = reference_aircraft.distance_to(aircraft)
                if distance <= radius:
                    nearby.append(aircraft)
        
        return nearby
    
    def _apply_control_input(self, aircraft: Aircraft, control_input: ControlInput) -> None:
        """Apply control input to aircraft."""
        # Convert control input to aircraft dynamics format
        control_array = np.array([
            control_input.speed_command,
            control_input.heading_rate,
            control_input.climb_rate
        ])
        
        # Update aircraft with control input
        aircraft.update_dynamics(self.config.time_step, control_array)
    
    def _monitor_separation(self) -> None:
        """Monitor separation constraints."""
        # Monitor current separation
        current_events = self.separation_monitor.monitor_separation(
            self.aircraft_list, self.current_time
        )
        
        # Handle any violations
        if current_events:
            self._handle_separation_events(current_events)
        
        # Check for emergency situations
        emergency_conflicts = self.lower_controller.check_immediate_conflicts(
            self.aircraft_list
        )
        
        if emergency_conflicts:
            self._handle_emergency_conflicts(emergency_conflicts)
    
    def _handle_separation_events(self, events) -> None:
        """Handle separation violation events."""
        for event in events:
            self.events_log.append({
                'time': self.current_time,
                'type': 'separation_event',
                'level': event.level.value,
                'aircraft1': event.aircraft1_id,
                'aircraft2': event.aircraft2_id,
                'distance': event.distance
            })
            
            # Update metrics
            if event.level.value in ['violation', 'critical', 'emergency']:
                self.metrics['separation_violations'] += 1
            if event.level.value == 'emergency':
                self.metrics['emergency_events'] += 1
    
    def _handle_emergency_conflicts(self, conflicts) -> None:
        """Handle emergency conflicts."""
        for aircraft1_id, aircraft2_id in conflicts:
            aircraft1 = next((a for a in self.aircraft_list if a.id == aircraft1_id), None)
            aircraft2 = next((a for a in self.aircraft_list if a.id == aircraft2_id), None)
            
            if aircraft1 and aircraft2:
                # Apply emergency avoidance for both aircraft
                emergency_control1 = self.lower_controller.emergency_avoidance(
                    aircraft1, aircraft2, self.current_time
                )
                emergency_control2 = self.lower_controller.emergency_avoidance(
                    aircraft2, aircraft1, self.current_time
                )
                
                self._apply_control_input(aircraft1, emergency_control1)
                self._apply_control_input(aircraft2, emergency_control2)
                
                # Log emergency event
                self.events_log.append({
                    'time': self.current_time,
                    'type': 'emergency_avoidance',
                    'aircraft1': aircraft1_id,
                    'aircraft2': aircraft2_id
                })
    
    def _update_aircraft_dynamics(self) -> None:
        """Update aircraft dynamics for all aircraft."""
        for aircraft in self.aircraft_list:
            # Aircraft dynamics are updated in _apply_control_input
            # Here we could add additional dynamics like wind effects
            pass
    
    def _remove_completed_aircraft(self) -> None:
        """Remove aircraft that have completed their approach."""
        completed_aircraft = []
        
        for aircraft in self.aircraft_list:
            # Check if aircraft has reached the merge point
            distance_to_merge = np.linalg.norm(
                aircraft.state.position - self.point_merge_system.merge_point.position
            )
            
            if distance_to_merge < 1000.0:  # Within 1 km of merge point
                completed_aircraft.append(aircraft)
                self.metrics['total_aircraft_processed'] += 1
                
                # Calculate flight time
                flight_time = self.current_time - (aircraft.state.timestamp - 
                                                 self.current_time + aircraft.state.timestamp)
                
                # Log completion
                self.events_log.append({
                    'time': self.current_time,
                    'type': 'aircraft_completed',
                    'aircraft_id': aircraft.id,
                    'flight_time': flight_time
                })
        
        # Remove completed aircraft
        for aircraft in completed_aircraft:
            self.aircraft_list.remove(aircraft)
            
            # Clean up assignments
            if aircraft.id in self.point_merge_system.aircraft_assignments:
                leg_type = self.point_merge_system.aircraft_assignments[aircraft.id]
                if aircraft.id in self.point_merge_system.sequence_order[leg_type]:
                    self.point_merge_system.sequence_order[leg_type].remove(aircraft.id)
                del self.point_merge_system.aircraft_assignments[aircraft.id]
    
    def _log_trajectory_data(self) -> None:
        """Log trajectory data for all aircraft."""
        for aircraft in self.aircraft_list:
            trajectory_point = {
                'time': self.current_time,
                'position': aircraft.state.position.copy(),
                'velocity': aircraft.state.velocity.copy(),
                'heading': aircraft.state.heading,
                'speed': aircraft.state.speed,
                'altitude': aircraft.state.altitude
            }
            
            if aircraft.id in self.trajectory_data:
                self.trajectory_data[aircraft.id].append(trajectory_point)
    
    def _log_strategic_decisions(self) -> None:
        """Log strategic decisions from upper-level controller."""
        strategic_data = {
            'time': self.current_time,
            'aircraft_count': len(self.aircraft_list),
            'sequence_order': self.upper_controller.strategic_sequence.copy(),
            'leg_assignments': self.point_merge_system.aircraft_assignments.copy()
        }
        
        if self.config.save_statistics:
            self.statistics_data.append(strategic_data)
    
    def _log_progress(self) -> None:
        """Log simulation progress."""
        print(f"Time: {self.current_time:6.0f}s | "
              f"Aircraft: {len(self.aircraft_list):2d} | "
              f"Completed: {self.metrics['total_aircraft_processed']:3d} | "
              f"Violations: {self.metrics['separation_violations']:3d}")
    
    def _finalize_simulation(self) -> Dict:
        """Finalize simulation and compute final statistics."""
        # Compute final metrics
        if self.metrics['total_aircraft_processed'] > 0:
            total_flight_time = sum(
                event['flight_time'] for event in self.events_log 
                if event['type'] == 'aircraft_completed'
            )
            self.metrics['average_flight_time'] = (
                total_flight_time / self.metrics['total_aircraft_processed']
            )
        
        # Get separation monitoring report
        separation_report = self.separation_monitor.get_separation_report()
        
        # Compile results
        results = {
            'metrics': self.metrics,
            'separation_report': separation_report,
            'events_log': self.events_log,
            'final_aircraft_count': len(self.aircraft_list),
            'simulation_time': self.current_time
        }
        
        if self.config.save_trajectory:
            results['trajectory_data'] = self.trajectory_data
        
        if self.config.save_statistics:
            results['statistics_data'] = self.statistics_data
        
        return results
    
    def add_aircraft(self, aircraft: Aircraft) -> None:
        """Add an aircraft to the simulation."""
        self.aircraft_list.append(aircraft)
        if self.config.save_trajectory:
            self.trajectory_data[aircraft.id] = []
    
    def get_current_status(self) -> Dict:
        """Get current simulation status."""
        return {
            'time': self.current_time,
            'aircraft_count': len(self.aircraft_list),
            'aircraft_list': [
                {
                    'id': aircraft.id,
                    'position': aircraft.state.position.tolist(),
                    'speed': aircraft.state.speed,
                    'heading': np.degrees(aircraft.state.heading)
                }
                for aircraft in self.aircraft_list
            ],
            'separation_events': len(self.separation_monitor.current_events),
            'metrics': self.metrics.copy()
        }


def main():
    """Main simulation entry point."""
    # Create simulation configuration
    config = SimulationConfig(
        total_time=1800.0,  # 30 minutes
        aircraft_arrival_rate=0.033,  # About 120 aircraft per hour
        max_aircraft=15
    )
    
    # Create and run simulation
    simulation = ATMSimulation(config)
    results = simulation.run_simulation()
    
    # Print summary
    print("\n" + "="*50)
    print("SIMULATION SUMMARY")
    print("="*50)
    print(f"Total aircraft processed: {results['metrics']['total_aircraft_processed']}")
    print(f"Average flight time: {results['metrics']['average_flight_time']:.1f} seconds")
    print(f"Separation violations: {results['metrics']['separation_violations']}")
    print(f"Emergency events: {results['metrics']['emergency_events']}")
    print(f"Final aircraft in system: {results['final_aircraft_count']}")
    
    return results


if __name__ == "__main__":
    main()