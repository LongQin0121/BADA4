"""
Aircraft model for air traffic management simulation.
"""
from typing import Tuple, Optional, List
import numpy as np
from dataclasses import dataclass
from enum import Enum


class FlightPhase(Enum):
    """Flight phase enumeration."""
    ARRIVAL = "arrival"
    DEPARTURE = "departure"
    CRUISE = "cruise"
    APPROACH = "approach"


@dataclass
class AircraftState:
    """Aircraft state representation."""
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    heading: float        # heading angle in radians
    altitude: float       # altitude in meters
    speed: float          # ground speed in m/s
    timestamp: float      # time in seconds


@dataclass
class AircraftConfig:
    """Aircraft configuration parameters."""
    max_speed: float = 250.0          # m/s (about 485 knots)
    min_speed: float = 70.0           # m/s (about 136 knots)
    max_turn_rate: float = 3.0        # degrees per second
    max_climb_rate: float = 15.0      # m/s
    max_descent_rate: float = -10.0   # m/s
    wingspan: float = 35.0            # meters
    length: float = 37.0              # meters


class Aircraft:
    """
    Aircraft model for simulation and control.
    
    This class represents an aircraft in the terminal area, with dynamics
    suitable for MPC-based control in the point merge system.
    """
    
    def __init__(self, 
                 aircraft_id: str,
                 initial_state: AircraftState,
                 config: Optional[AircraftConfig] = None,
                 flight_phase: FlightPhase = FlightPhase.ARRIVAL):
        """
        Initialize aircraft model.
        
        Args:
            aircraft_id: Unique identifier for the aircraft
            initial_state: Initial state of the aircraft
            config: Aircraft configuration parameters
            flight_phase: Current flight phase
        """
        self.id = aircraft_id
        self.state = initial_state
        self.config = config or AircraftConfig()
        self.flight_phase = flight_phase
        
        # Control inputs (to be set by controllers)
        self.commanded_speed: Optional[float] = None
        self.commanded_heading: Optional[float] = None
        self.commanded_altitude: Optional[float] = None
        
        # Reference trajectory (set-points from upper-level controller)
        self.reference_waypoints: List[np.ndarray] = []
        self.reference_times: List[float] = []
        
    def update_dynamics(self, dt: float, control_input: Optional[np.ndarray] = None) -> None:
        """
        Update aircraft dynamics using simple point-mass model.
        
        Args:
            dt: Time step in seconds
            control_input: Control input [speed_cmd, heading_rate_cmd, climb_rate_cmd]
        """
        if control_input is not None:
            speed_cmd, heading_rate_cmd, climb_rate_cmd = control_input
            
            # Update commanded values
            self.commanded_speed = np.clip(speed_cmd, 
                                         self.config.min_speed, 
                                         self.config.max_speed)
            
            # Update heading with turn rate constraints
            max_heading_rate = np.radians(self.config.max_turn_rate)
            heading_rate = np.clip(heading_rate_cmd, 
                                 -max_heading_rate, 
                                 max_heading_rate)
            self.state.heading += heading_rate * dt
            self.state.heading = self.state.heading % (2 * np.pi)
            
            # Update altitude with climb rate constraints
            climb_rate = np.clip(climb_rate_cmd,
                               self.config.max_descent_rate,
                               self.config.max_climb_rate)
            self.state.altitude += climb_rate * dt
        
        # Update position based on current speed and heading
        self.state.speed = self.commanded_speed or self.state.speed
        
        # Convert heading to velocity components
        vx = self.state.speed * np.cos(self.state.heading)
        vy = self.state.speed * np.sin(self.state.heading)
        vz = (self.commanded_altitude - self.state.altitude) / dt if self.commanded_altitude else 0
        
        self.state.velocity = np.array([vx, vy, vz])
        
        # Update position
        self.state.position += self.state.velocity * dt
        self.state.timestamp += dt
    
    def get_future_position(self, time_horizon: float, num_points: int = 10) -> np.ndarray:
        """
        Predict future positions assuming current control inputs.
        
        Args:
            time_horizon: Prediction horizon in seconds
            num_points: Number of prediction points
            
        Returns:
            Array of future positions [num_points, 3]
        """
        dt = time_horizon / num_points
        future_positions = np.zeros((num_points, 3))
        
        # Simple constant velocity prediction
        for i in range(num_points):
            t = (i + 1) * dt
            future_pos = self.state.position + self.state.velocity * t
            future_positions[i] = future_pos
            
        return future_positions
    
    def distance_to(self, other: 'Aircraft') -> float:
        """
        Calculate 3D distance to another aircraft.
        
        Args:
            other: Another aircraft
            
        Returns:
            Distance in meters
        """
        return np.linalg.norm(self.state.position - other.state.position)
    
    def horizontal_distance_to(self, other: 'Aircraft') -> float:
        """
        Calculate horizontal distance to another aircraft.
        
        Args:
            other: Another aircraft
            
        Returns:
            Horizontal distance in meters
        """
        pos_diff = self.state.position[:2] - other.state.position[:2]
        return np.linalg.norm(pos_diff)
    
    def set_reference_trajectory(self, waypoints: List[np.ndarray], times: List[float]) -> None:
        """
        Set reference trajectory from upper-level controller.
        
        Args:
            waypoints: List of 3D waypoints [x, y, z]
            times: Corresponding time stamps
        """
        self.reference_waypoints = waypoints.copy()
        self.reference_times = times.copy()
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get current state as a vector for optimization.
        
        Returns:
            State vector [x, y, z, vx, vy, vz, heading, speed]
        """
        return np.concatenate([
            self.state.position,
            self.state.velocity,
            [self.state.heading, self.state.speed]
        ])
    
    def __repr__(self) -> str:
        return (f"Aircraft(id={self.id}, "
                f"pos={self.state.position}, "
                f"speed={self.state.speed:.1f}, "
                f"heading={np.degrees(self.state.heading):.1f}°)")