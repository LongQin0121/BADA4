"""
Point Merge System implementation for Stockholm airport terminal area.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

try:
    from ..models.aircraft import Aircraft, AircraftState
except ImportError:
    from models.aircraft import Aircraft, AircraftState


class SequenceType(Enum):
    """Sequencing area type."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


@dataclass
class MergePoint:
    """Merge point configuration."""
    position: np.ndarray  # [x, y, z] in meters
    name: str
    runway_heading: float  # radians
    approach_altitude: float = 1000.0  # meters
    
    
@dataclass
class SequencingLeg:
    """Sequencing leg configuration."""
    entry_point: np.ndarray   # Entry point [x, y, z]
    direction: float          # Direction of the leg in radians
    length: float            # Length of the leg in meters
    sequence_type: SequenceType
    speed_constraint: float = 220.0  # m/s (约 430 knots)


class PointMergeSystem:
    """
    Point Merge System for Stockholm Arlanda airport.
    
    This system manages aircraft sequencing in the terminal area using
    a point merge procedure with multiple sequencing legs.
    """
    
    def __init__(self):
        """Initialize the Point Merge System with Stockholm Arlanda configuration."""
        
        # Stockholm Arlanda coordinates (approximate, in local coordinate system)
        # Origin at the merge point for simplification
        self.merge_point = MergePoint(
            position=np.array([0.0, 0.0, 300.0]),  # 300m altitude for merge
            name="ARLANDA_MERGE",
            runway_heading=np.radians(80),  # Runway 08L/26R heading
            approach_altitude=300.0
        )
        
        # Define sequencing legs around the merge point
        # Using a simplified geometry with 4 legs
        leg_length = 30000.0  # 30 km legs
        
        self.sequencing_legs = {
            SequenceType.NORTH: SequencingLeg(
                entry_point=np.array([0.0, leg_length, 1000.0]),
                direction=np.radians(180),  # South towards merge
                length=leg_length,
                sequence_type=SequenceType.NORTH,
                speed_constraint=200.0
            ),
            SequenceType.SOUTH: SequencingLeg(
                entry_point=np.array([0.0, -leg_length, 1000.0]),
                direction=np.radians(0),    # North towards merge
                length=leg_length,
                sequence_type=SequenceType.SOUTH,
                speed_constraint=200.0
            ),
            SequenceType.EAST: SequencingLeg(
                entry_point=np.array([leg_length, 0.0, 1000.0]),
                direction=np.radians(270),  # West towards merge
                length=leg_length,
                sequence_type=SequenceType.EAST,
                speed_constraint=200.0
            ),
            SequenceType.WEST: SequencingLeg(
                entry_point=np.array([-leg_length, 0.0, 1000.0]),
                direction=np.radians(90),   # East towards merge
                length=leg_length,
                sequence_type=SequenceType.WEST,
                speed_constraint=200.0
            )
        }
        
        # Minimum separation requirements
        self.min_separation_distance = 5556.0  # 3 nautical miles in meters
        self.min_time_separation = 120.0       # 2 minutes minimum time separation
        
        # Aircraft assignments to legs
        self.aircraft_assignments: Dict[str, SequenceType] = {}
        self.sequence_order: Dict[SequenceType, List[str]] = {
            leg_type: [] for leg_type in SequenceType
        }
        
    def assign_aircraft_to_leg(self, aircraft: Aircraft, preferred_leg: Optional[SequenceType] = None) -> SequenceType:
        """
        Assign aircraft to a sequencing leg based on its position and traffic load.
        
        Args:
            aircraft: Aircraft to assign
            preferred_leg: Preferred sequencing leg (if any)
            
        Returns:
            Assigned sequencing leg type
        """
        if preferred_leg and self._is_leg_available(preferred_leg):
            assigned_leg = preferred_leg
        else:
            # Find the closest available leg
            assigned_leg = self._find_best_leg(aircraft)
        
        self.aircraft_assignments[aircraft.id] = assigned_leg
        self.sequence_order[assigned_leg].append(aircraft.id)
        
        return assigned_leg
    
    def _find_best_leg(self, aircraft: Aircraft) -> SequenceType:
        """Find the best sequencing leg for an aircraft."""
        min_distance = float('inf')
        best_leg = SequenceType.NORTH
        
        for leg_type, leg in self.sequencing_legs.items():
            distance = np.linalg.norm(aircraft.state.position[:2] - leg.entry_point[:2])
            traffic_load = len(self.sequence_order[leg_type])
            
            # Weighted score: distance + traffic load penalty
            score = distance + traffic_load * 10000.0  # 10km penalty per aircraft
            
            if score < min_distance:
                min_distance = score
                best_leg = leg_type
                
        return best_leg
    
    def _is_leg_available(self, leg_type: SequenceType) -> bool:
        """Check if a leg has capacity for more aircraft."""
        return len(self.sequence_order[leg_type]) < 5  # Max 5 aircraft per leg
    
    def generate_reference_trajectory(self, aircraft: Aircraft, time_horizon: float = 600.0) -> Tuple[List[np.ndarray], List[float]]:
        """
        Generate reference trajectory for an aircraft in the point merge system.
        
        Args:
            aircraft: Aircraft to generate trajectory for
            time_horizon: Time horizon for trajectory in seconds
            
        Returns:
            Tuple of (waypoints, times)
        """
        if aircraft.id not in self.aircraft_assignments:
            raise ValueError(f"Aircraft {aircraft.id} not assigned to any leg")
        
        leg_type = self.aircraft_assignments[aircraft.id]
        leg = self.sequencing_legs[leg_type]
        
        # Calculate trajectory waypoints
        waypoints = []
        times = []
        
        # Current position
        current_pos = aircraft.state.position.copy()
        current_time = 0.0
        
        # 1. Navigate to leg entry point
        entry_waypoint = leg.entry_point.copy()
        entry_time = self._calculate_travel_time(current_pos, entry_waypoint, aircraft.state.speed)
        waypoints.append(entry_waypoint)
        times.append(entry_time)
        
        # 2. Follow the sequencing leg
        leg_waypoints, leg_times = self._generate_leg_trajectory(
            leg, aircraft, entry_time, time_horizon
        )
        waypoints.extend(leg_waypoints)
        times.extend(leg_times)
        
        # 3. Final approach to merge point
        merge_waypoint = self.merge_point.position.copy()
        if times:
            merge_time = times[-1] + self._calculate_travel_time(
                waypoints[-1], merge_waypoint, leg.speed_constraint
            )
            waypoints.append(merge_waypoint)
            times.append(merge_time)
        
        return waypoints, times
    
    def _generate_leg_trajectory(self, leg: SequencingLeg, aircraft: Aircraft, 
                               start_time: float, time_horizon: float) -> Tuple[List[np.ndarray], List[float]]:
        """Generate trajectory along a sequencing leg."""
        waypoints = []
        times = []
        
        # Number of waypoints along the leg
        num_waypoints = 5
        
        # Calculate waypoints along the leg
        for i in range(1, num_waypoints + 1):
            # Position along the leg
            progress = i / num_waypoints
            
            # Calculate position
            leg_vector = np.array([
                np.cos(leg.direction),
                np.sin(leg.direction),
                0.0
            ]) * leg.length * progress
            
            waypoint = leg.entry_point + leg_vector
            waypoint[2] = self._interpolate_altitude(
                leg.entry_point[2], 
                self.merge_point.position[2], 
                progress
            )
            
            # Calculate time
            travel_time = self._calculate_travel_time(
                leg.entry_point if i == 1 else waypoints[-1],
                waypoint,
                leg.speed_constraint
            )
            time = (times[-1] if times else start_time) + travel_time
            
            waypoints.append(waypoint)
            times.append(time)
            
        return waypoints, times
    
    def _interpolate_altitude(self, start_alt: float, end_alt: float, progress: float) -> float:
        """Interpolate altitude along trajectory."""
        return start_alt + (end_alt - start_alt) * progress
    
    def _calculate_travel_time(self, pos1: np.ndarray, pos2: np.ndarray, speed: float) -> float:
        """Calculate travel time between two positions."""
        distance = np.linalg.norm(pos2 - pos1)
        return distance / speed
    
    def calculate_merge_sequence(self, aircraft_list: List[Aircraft]) -> List[str]:
        """
        Calculate optimal merge sequence considering separation constraints.
        
        Args:
            aircraft_list: List of aircraft to sequence
            
        Returns:
            Optimal sequence order (aircraft IDs)
        """
        # Calculate estimated arrival times at merge point
        arrival_times = {}
        
        for aircraft in aircraft_list:
            if aircraft.id in self.aircraft_assignments:
                # Estimate time to merge point
                merge_distance = np.linalg.norm(
                    aircraft.state.position - self.merge_point.position
                )
                # Simplified estimation
                estimated_time = merge_distance / aircraft.state.speed
                arrival_times[aircraft.id] = aircraft.state.timestamp + estimated_time
        
        # Sort by arrival time
        sorted_aircraft = sorted(arrival_times.items(), key=lambda x: x[1])
        
        # Apply separation constraints
        sequence = []
        last_merge_time = 0.0
        
        for aircraft_id, arrival_time in sorted_aircraft:
            # Ensure minimum time separation
            merge_time = max(arrival_time, last_merge_time + self.min_time_separation)
            sequence.append(aircraft_id)
            last_merge_time = merge_time
            
        return sequence
    
    def check_separation_constraint(self, aircraft1: Aircraft, aircraft2: Aircraft) -> bool:
        """
        Check if two aircraft meet separation requirements.
        
        Args:
            aircraft1: First aircraft
            aircraft2: Second aircraft
            
        Returns:
            True if separation is adequate
        """
        distance = aircraft1.horizontal_distance_to(aircraft2)
        return distance >= self.min_separation_distance
    
    def get_turn_point(self, aircraft: Aircraft, target_sequence_position: int) -> Optional[np.ndarray]:
        """
        Calculate turn point for aircraft to achieve desired sequence position.
        
        Args:
            aircraft: Aircraft to calculate turn point for
            target_sequence_position: Desired position in sequence (0-based)
            
        Returns:
            Turn point coordinates or None if not applicable
        """
        if aircraft.id not in self.aircraft_assignments:
            return None
        
        leg_type = self.aircraft_assignments[aircraft.id]
        leg = self.sequencing_legs[leg_type]
        
        # Calculate required delay based on sequence position
        required_delay = target_sequence_position * self.min_time_separation
        
        # Calculate turn point on the leg to achieve this delay
        current_distance_to_merge = np.linalg.norm(
            aircraft.state.position - self.merge_point.position
        )
        
        # Simple calculation: extend path to create delay
        extra_distance = required_delay * leg.speed_constraint
        
        # Turn point is further back on the leg
        leg_progress = max(0.2, 1.0 - (extra_distance / leg.length))
        
        leg_vector = np.array([
            np.cos(leg.direction),
            np.sin(leg.direction),
            0.0
        ]) * leg.length * leg_progress
        
        turn_point = leg.entry_point + leg_vector
        
        return turn_point