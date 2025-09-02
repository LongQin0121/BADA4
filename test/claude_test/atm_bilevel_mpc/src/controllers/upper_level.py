"""
Upper-level controller (BL-RC) for strategic air traffic management.

This controller implements the rough control layer that generates set-points
for the lower-level controller, focusing on overall strategy and conflict-free
trajectory planning.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
import casadi as ca

try:
    from ..models.aircraft import Aircraft, AircraftState
    from ..systems.point_merge import PointMergeSystem, SequenceType
except ImportError:
    from models.aircraft import Aircraft, AircraftState
    from systems.point_merge import PointMergeSystem, SequenceType


@dataclass
class SetPoint:
    """Set-point data structure for lower-level controller."""
    position: np.ndarray      # Target position [x, y, z]
    time: float              # Target time
    speed_range: Tuple[float, float]  # (min_speed, max_speed)
    altitude_range: Tuple[float, float]  # (min_alt, max_alt)
    heading_constraint: Optional[float] = None  # Preferred heading


@dataclass
class ControlHorizon:
    """Control horizon parameters."""
    strategic_horizon: float = 1200.0    # 20 minutes strategic planning
    tactical_horizon: float = 300.0      # 5 minutes tactical adjustment
    update_interval: float = 60.0        # Update every minute
    
    
class UpperLevelController:
    """
    Upper-level controller for bi-level MPC air traffic management.
    
    This controller performs strategic planning over longer horizons,
    generating conflict-free reference trajectories and set-points.
    """
    
    def __init__(self, point_merge_system: PointMergeSystem):
        """
        Initialize upper-level controller.
        
        Args:
            point_merge_system: Point merge system instance
        """
        self.pms = point_merge_system
        self.control_horizon = ControlHorizon()
        
        # Strategic planning parameters
        self.conflict_buffer = 7500.0  # 7.5km conflict detection buffer
        self.planning_resolution = 30.0  # 30-second time steps
        
        # Optimization weights
        self.weights = {
            'fuel_efficiency': 1.0,
            'time_deviation': 2.0,
            'separation_violation': 100.0,
            'smooth_trajectory': 0.5
        }
        
        # Current set-points for each aircraft
        self.current_setpoints: Dict[str, List[SetPoint]] = {}
        
        # Strategic sequence decisions
        self.strategic_sequence: List[str] = []
        
    def plan_strategic_trajectories(self, aircraft_list: List[Aircraft], 
                                  current_time: float) -> Dict[str, List[SetPoint]]:
        """
        Plan strategic trajectories for all aircraft.
        
        Args:
            aircraft_list: List of aircraft to plan for
            current_time: Current simulation time
            
        Returns:
            Dictionary mapping aircraft IDs to set-point lists
        """
        # Step 1: Assign aircraft to sequencing legs if not already assigned
        self._assign_aircraft_to_legs(aircraft_list)
        
        # Step 2: Calculate optimal merge sequence
        self.strategic_sequence = self._optimize_merge_sequence(aircraft_list)
        
        # Step 3: Generate conflict-free trajectories
        setpoints_dict = {}
        
        for aircraft in aircraft_list:
            setpoints = self._generate_aircraft_setpoints(
                aircraft, self.strategic_sequence, current_time
            )
            setpoints_dict[aircraft.id] = setpoints
            
        self.current_setpoints = setpoints_dict
        return setpoints_dict
    
    def _assign_aircraft_to_legs(self, aircraft_list: List[Aircraft]) -> None:
        """Assign unassigned aircraft to sequencing legs."""
        for aircraft in aircraft_list:
            if aircraft.id not in self.pms.aircraft_assignments:
                # Determine optimal leg assignment
                best_leg = self._find_optimal_leg_assignment(aircraft, aircraft_list)
                self.pms.assign_aircraft_to_leg(aircraft, best_leg)
    
    def _find_optimal_leg_assignment(self, aircraft: Aircraft, 
                                   all_aircraft: List[Aircraft]) -> SequenceType:
        """
        Find optimal leg assignment considering traffic distribution.
        
        Args:
            aircraft: Aircraft to assign
            all_aircraft: All aircraft in the system
            
        Returns:
            Optimal sequencing leg type
        """
        leg_costs = {}
        
        for leg_type in SequenceType:
            cost = self._calculate_leg_assignment_cost(aircraft, leg_type, all_aircraft)
            leg_costs[leg_type] = cost
            
        return min(leg_costs, key=leg_costs.get)
    
    def _calculate_leg_assignment_cost(self, aircraft: Aircraft, 
                                     leg_type: SequenceType, 
                                     all_aircraft: List[Aircraft]) -> float:
        """Calculate cost of assigning aircraft to a specific leg."""
        leg = self.pms.sequencing_legs[leg_type]
        
        # Distance cost
        distance_cost = np.linalg.norm(
            aircraft.state.position[:2] - leg.entry_point[:2]
        )
        
        # Traffic load cost
        current_load = len(self.pms.sequence_order[leg_type])
        load_cost = current_load * 50000.0  # 50km penalty per aircraft
        
        # Conflict potential cost
        conflict_cost = self._estimate_conflict_potential(aircraft, leg_type, all_aircraft)
        
        return distance_cost + load_cost + conflict_cost
    
    def _estimate_conflict_potential(self, aircraft: Aircraft, 
                                   leg_type: SequenceType,
                                   all_aircraft: List[Aircraft]) -> float:
        """Estimate potential conflicts on a leg."""
        conflict_cost = 0.0
        leg = self.pms.sequencing_legs[leg_type]
        
        # Check conflicts with aircraft already on this leg
        for other_aircraft in all_aircraft:
            if (other_aircraft.id != aircraft.id and 
                other_aircraft.id in self.pms.aircraft_assignments and
                self.pms.aircraft_assignments[other_aircraft.id] == leg_type):
                
                # Estimate future separation
                time_horizon = 300.0  # 5 minutes
                future_separation = self._estimate_future_separation(
                    aircraft, other_aircraft, time_horizon
                )
                
                if future_separation < self.pms.min_separation_distance:
                    conflict_cost += 100000.0  # High penalty for conflicts
                    
        return conflict_cost
    
    def _estimate_future_separation(self, aircraft1: Aircraft, aircraft2: Aircraft, 
                                  time_horizon: float) -> float:
        """Estimate minimum separation between two aircraft over time horizon."""
        dt = 30.0  # 30-second intervals
        num_steps = int(time_horizon / dt)
        min_separation = float('inf')
        
        for i in range(num_steps):
            t = i * dt
            # Simple constant velocity prediction
            pos1 = aircraft1.state.position + aircraft1.state.velocity * t
            pos2 = aircraft2.state.position + aircraft2.state.velocity * t
            
            separation = np.linalg.norm(pos1[:2] - pos2[:2])
            min_separation = min(min_separation, separation)
            
        return min_separation
    
    def _optimize_merge_sequence(self, aircraft_list: List[Aircraft]) -> List[str]:
        """
        Optimize merge sequence using simplified optimization.
        
        Args:
            aircraft_list: List of aircraft
            
        Returns:
            Optimal sequence order
        """
        # For now, use the point merge system's sequence calculation
        # In a more sophisticated implementation, this would involve
        # multi-objective optimization considering fuel, delay, and fairness
        
        return self.pms.calculate_merge_sequence(aircraft_list)
    
    def _generate_aircraft_setpoints(self, aircraft: Aircraft, 
                                   sequence: List[str],
                                   current_time: float) -> List[SetPoint]:
        """
        Generate set-points for a specific aircraft.
        
        Args:
            aircraft: Aircraft to generate set-points for
            sequence: Overall merge sequence
            current_time: Current time
            
        Returns:
            List of set-points
        """
        setpoints = []
        
        # Get reference trajectory from point merge system
        waypoints, times = self.pms.generate_reference_trajectory(
            aircraft, self.control_horizon.strategic_horizon
        )
        
        # Find aircraft's position in sequence
        sequence_position = sequence.index(aircraft.id) if aircraft.id in sequence else 0
        
        # Calculate required timing adjustments
        target_merge_time = current_time + 600.0 + sequence_position * self.pms.min_time_separation
        
        # Generate set-points along the trajectory
        time_step = self.control_horizon.update_interval
        num_setpoints = int(self.control_horizon.strategic_horizon / time_step)
        
        for i in range(num_setpoints):
            setpoint_time = current_time + i * time_step
            
            # Interpolate position along reference trajectory
            position = self._interpolate_trajectory_position(
                waypoints, times, setpoint_time
            )
            
            # Calculate speed constraints based on timing requirements
            speed_range = self._calculate_speed_constraints(
                aircraft, position, setpoint_time, target_merge_time
            )
            
            # Altitude constraints
            altitude_range = (position[2] - 100.0, position[2] + 100.0)
            
            setpoint = SetPoint(
                position=position,
                time=setpoint_time,
                speed_range=speed_range,
                altitude_range=altitude_range
            )
            setpoints.append(setpoint)
            
        return setpoints
    
    def _interpolate_trajectory_position(self, waypoints: List[np.ndarray], 
                                       times: List[float], 
                                       target_time: float) -> np.ndarray:
        """Interpolate position along trajectory at target time."""
        if not waypoints or not times:
            return np.array([0.0, 0.0, 0.0])
        
        # Find the two waypoints that bracket the target time
        for i in range(len(times) - 1):
            if times[i] <= target_time <= times[i + 1]:
                # Linear interpolation
                t_ratio = (target_time - times[i]) / (times[i + 1] - times[i])
                return waypoints[i] + t_ratio * (waypoints[i + 1] - waypoints[i])
        
        # Extrapolate if outside time range
        if target_time <= times[0]:
            return waypoints[0]
        else:
            return waypoints[-1]
    
    def _calculate_speed_constraints(self, aircraft: Aircraft, 
                                   target_position: np.ndarray,
                                   setpoint_time: float,
                                   target_merge_time: float) -> Tuple[float, float]:
        """Calculate speed constraints for a set-point."""
        # Base speed constraints from aircraft configuration
        min_speed = aircraft.config.min_speed
        max_speed = aircraft.config.max_speed
        
        # Adjust based on timing requirements
        current_pos = aircraft.state.position
        distance_to_target = np.linalg.norm(target_position - current_pos)
        time_to_target = setpoint_time - aircraft.state.timestamp
        
        if time_to_target > 0:
            required_speed = distance_to_target / time_to_target
            
            # Adjust constraints to meet timing
            if required_speed < min_speed:
                # Need to slow down - use minimum speed
                return (min_speed, min_speed * 1.1)
            elif required_speed > max_speed:
                # Need to speed up - use maximum speed
                return (max_speed * 0.9, max_speed)
            else:
                # Normal operation - allow some flexibility
                return (required_speed * 0.9, required_speed * 1.1)
        
        return (min_speed, max_speed)
    
    def update_setpoints(self, aircraft_list: List[Aircraft], 
                        current_time: float) -> Dict[str, List[SetPoint]]:
        """
        Update set-points based on current situation.
        
        Args:
            aircraft_list: Current aircraft list
            current_time: Current time
            
        Returns:
            Updated set-points dictionary
        """
        # Check if major replanning is needed
        if self._needs_replanning(aircraft_list, current_time):
            return self.plan_strategic_trajectories(aircraft_list, current_time)
        
        # Otherwise, update existing set-points
        for aircraft in aircraft_list:
            if aircraft.id in self.current_setpoints:
                self._update_aircraft_setpoints(aircraft, current_time)
        
        return self.current_setpoints
    
    def _needs_replanning(self, aircraft_list: List[Aircraft], current_time: float) -> bool:
        """Determine if major replanning is required."""
        # Check for new aircraft
        current_aircraft_ids = {aircraft.id for aircraft in aircraft_list}
        tracked_aircraft_ids = set(self.current_setpoints.keys())
        
        if current_aircraft_ids != tracked_aircraft_ids:
            return True
        
        # Check for significant conflicts
        for aircraft in aircraft_list:
            for other_aircraft in aircraft_list:
                if aircraft.id != other_aircraft.id:
                    if not self.pms.check_separation_constraint(aircraft, other_aircraft):
                        # Check if this is a predicted conflict vs actual
                        future_separation = self._estimate_future_separation(
                            aircraft, other_aircraft, 180.0  # 3 minutes
                        )
                        if future_separation < self.pms.min_separation_distance:
                            return True
        
        return False
    
    def _update_aircraft_setpoints(self, aircraft: Aircraft, current_time: float) -> None:
        """Update set-points for a specific aircraft."""
        if aircraft.id not in self.current_setpoints:
            return
        
        setpoints = self.current_setpoints[aircraft.id]
        
        # Remove outdated set-points
        setpoints = [sp for sp in setpoints if sp.time > current_time]
        
        # Add new set-points if needed
        if setpoints:
            last_time = setpoints[-1].time
            if last_time < current_time + self.control_horizon.strategic_horizon:
                # Generate additional set-points
                additional_setpoints = self._generate_additional_setpoints(
                    aircraft, last_time, current_time + self.control_horizon.strategic_horizon
                )
                setpoints.extend(additional_setpoints)
        
        self.current_setpoints[aircraft.id] = setpoints
    
    def _generate_additional_setpoints(self, aircraft: Aircraft, 
                                     start_time: float, 
                                     end_time: float) -> List[SetPoint]:
        """Generate additional set-points to extend horizon."""
        # This is a simplified implementation
        # In practice, this would involve more sophisticated trajectory extension
        
        additional_setpoints = []
        time_step = self.control_horizon.update_interval
        
        for t in np.arange(start_time + time_step, end_time, time_step):
            # Simple extrapolation
            future_pos = aircraft.state.position + aircraft.state.velocity * (t - aircraft.state.timestamp)
            
            setpoint = SetPoint(
                position=future_pos,
                time=t,
                speed_range=(aircraft.config.min_speed, aircraft.config.max_speed),
                altitude_range=(future_pos[2] - 100, future_pos[2] + 100)
            )
            additional_setpoints.append(setpoint)
            
        return additional_setpoints
    
    def get_current_setpoint(self, aircraft_id: str, current_time: float) -> Optional[SetPoint]:
        """
        Get the current set-point for an aircraft.
        
        Args:
            aircraft_id: Aircraft identifier
            current_time: Current time
            
        Returns:
            Current set-point or None
        """
        if aircraft_id not in self.current_setpoints:
            return None
        
        setpoints = self.current_setpoints[aircraft_id]
        
        # Find the set-point closest to current time
        for setpoint in setpoints:
            if setpoint.time >= current_time:
                return setpoint
                
        return None