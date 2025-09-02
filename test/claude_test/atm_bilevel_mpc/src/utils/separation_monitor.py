"""
Separation constraint enforcement and monitoring utilities.

This module provides comprehensive separation monitoring and constraint
enforcement for air traffic management, ensuring 3 nautical miles minimum
separation as required.
"""
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from ..models.aircraft import Aircraft
except ImportError:
    from models.aircraft import Aircraft


class SeparationLevel(Enum):
    """Separation violation severity levels."""
    SAFE = "safe"
    WARNING = "warning"      # Within 4 NM
    VIOLATION = "violation"  # Within 3 NM
    CRITICAL = "critical"    # Within 2 NM
    EMERGENCY = "emergency"  # Within 1 NM


@dataclass
class SeparationEvent:
    """Separation monitoring event."""
    aircraft1_id: str
    aircraft2_id: str
    distance: float
    level: SeparationLevel
    timestamp: float
    predicted_cpa_time: Optional[float] = None  # Closest point of approach time
    predicted_cpa_distance: Optional[float] = None  # Closest point of approach distance
    relative_velocity: Optional[np.ndarray] = None
    
    
@dataclass
class SeparationConstraints:
    """Separation constraint configuration."""
    minimum_separation: float = 5556.0      # 3 NM in meters
    warning_separation: float = 7408.0      # 4 NM in meters
    critical_separation: float = 3704.0     # 2 NM in meters
    emergency_separation: float = 1852.0    # 1 NM in meters
    
    # Vertical separation minimums
    vertical_separation: float = 300.0      # 1000 feet in meters
    
    # Time-based separation
    wake_turbulence_time: float = 120.0     # 2 minutes for wake turbulence
    runway_separation_time: float = 90.0    # 1.5 minutes for runway operations
    
    # Prediction parameters
    prediction_horizon: float = 300.0       # 5 minutes prediction
    prediction_resolution: float = 5.0      # 5-second intervals


class SeparationMonitor:
    """
    Comprehensive separation monitoring and constraint enforcement system.
    
    This class provides real-time monitoring of aircraft separation,
    prediction of future conflicts, and enforcement of separation constraints.
    """
    
    def __init__(self, constraints: Optional[SeparationConstraints] = None):
        """
        Initialize separation monitor.
        
        Args:
            constraints: Separation constraint configuration
        """
        self.constraints = constraints or SeparationConstraints()
        
        # Event tracking
        self.current_events: List[SeparationEvent] = []
        self.event_history: List[SeparationEvent] = []
        
        # Conflict prediction
        self.predicted_conflicts: Dict[Tuple[str, str], SeparationEvent] = {}
        
        # Statistics
        self.statistics = {
            'total_violations': 0,
            'critical_events': 0,
            'emergency_events': 0,
            'false_alarms': 0,
            'min_observed_separation': float('inf'),
            'average_separation': 0.0
        }
        
    def monitor_separation(self, aircraft_list: List[Aircraft], 
                         current_time: float) -> List[SeparationEvent]:
        """
        Monitor current separation between all aircraft pairs.
        
        Args:
            aircraft_list: List of aircraft to monitor
            current_time: Current simulation time
            
        Returns:
            List of current separation events
        """
        current_events = []
        
        for i, aircraft1 in enumerate(aircraft_list):
            for aircraft2 in aircraft_list[i+1:]:
                event = self._check_aircraft_pair(aircraft1, aircraft2, current_time)
                if event:
                    current_events.append(event)
        
        self.current_events = current_events
        self._update_statistics()
        
        return current_events
    
    def _check_aircraft_pair(self, aircraft1: Aircraft, aircraft2: Aircraft, 
                           current_time: float) -> Optional[SeparationEvent]:
        """Check separation between two aircraft."""
        # Calculate current separation
        horizontal_distance = aircraft1.horizontal_distance_to(aircraft2)
        vertical_distance = abs(aircraft1.state.position[2] - aircraft2.state.position[2])
        
        # Determine separation level
        level = self._classify_separation_level(horizontal_distance, vertical_distance)
        
        if level == SeparationLevel.SAFE:
            return None
        
        # Calculate relative velocity and CPA prediction
        rel_velocity = aircraft1.state.velocity - aircraft2.state.velocity
        cpa_time, cpa_distance = self._predict_closest_approach(
            aircraft1, aircraft2, rel_velocity
        )
        
        event = SeparationEvent(
            aircraft1_id=aircraft1.id,
            aircraft2_id=aircraft2.id,
            distance=horizontal_distance,
            level=level,
            timestamp=current_time,
            predicted_cpa_time=cpa_time,
            predicted_cpa_distance=cpa_distance,
            relative_velocity=rel_velocity
        )
        
        return event
    
    def _classify_separation_level(self, horizontal_distance: float, 
                                 vertical_distance: float) -> SeparationLevel:
        """Classify separation level based on distances."""
        # If vertical separation is adequate, less concern about horizontal
        if vertical_distance >= self.constraints.vertical_separation:
            horizontal_threshold = self.constraints.minimum_separation * 0.8
        else:
            horizontal_threshold = self.constraints.minimum_separation
        
        if horizontal_distance >= self.constraints.warning_separation:
            return SeparationLevel.SAFE
        elif horizontal_distance >= horizontal_threshold:
            return SeparationLevel.WARNING
        elif horizontal_distance >= self.constraints.critical_separation:
            return SeparationLevel.VIOLATION
        elif horizontal_distance >= self.constraints.emergency_separation:
            return SeparationLevel.CRITICAL
        else:
            return SeparationLevel.EMERGENCY
    
    def _predict_closest_approach(self, aircraft1: Aircraft, aircraft2: Aircraft,
                                rel_velocity: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Predict closest point of approach between two aircraft."""
        # Relative position
        rel_pos = aircraft1.state.position - aircraft2.state.position
        
        # Only consider horizontal components for now
        rel_pos_2d = rel_pos[:2]
        rel_vel_2d = rel_velocity[:2]
        
        # If relative velocity is very small, aircraft are not converging
        rel_speed = np.linalg.norm(rel_vel_2d)
        if rel_speed < 1e-6:  # Practically stationary relative to each other
            return None, np.linalg.norm(rel_pos_2d)
        
        # Time to closest approach: t = -(rel_pos · rel_vel) / |rel_vel|²
        t_cpa = -np.dot(rel_pos_2d, rel_vel_2d) / (rel_speed ** 2)
        
        # Only consider future closest approaches
        if t_cpa <= 0:
            return None, np.linalg.norm(rel_pos_2d)
        
        # Distance at closest approach
        pos_at_cpa = rel_pos_2d + rel_vel_2d * t_cpa
        distance_at_cpa = np.linalg.norm(pos_at_cpa)
        
        return t_cpa, distance_at_cpa
    
    def predict_conflicts(self, aircraft_list: List[Aircraft], 
                         current_time: float) -> Dict[Tuple[str, str], SeparationEvent]:
        """
        Predict future conflicts within the prediction horizon.
        
        Args:
            aircraft_list: List of aircraft
            current_time: Current time
            
        Returns:
            Dictionary of predicted conflicts
        """
        predicted_conflicts = {}
        
        for i, aircraft1 in enumerate(aircraft_list):
            for aircraft2 in aircraft_list[i+1:]:
                conflict = self._predict_pair_conflict(aircraft1, aircraft2, current_time)
                if conflict:
                    pair_key = tuple(sorted([aircraft1.id, aircraft2.id]))
                    predicted_conflicts[pair_key] = conflict
        
        self.predicted_conflicts = predicted_conflicts
        return predicted_conflicts
    
    def _predict_pair_conflict(self, aircraft1: Aircraft, aircraft2: Aircraft,
                             current_time: float) -> Optional[SeparationEvent]:
        """Predict conflict between two aircraft."""
        dt = self.constraints.prediction_resolution
        num_steps = int(self.constraints.prediction_horizon / dt)
        
        min_separation = float('inf')
        conflict_time = None
        
        for step in range(num_steps):
            t = step * dt
            
            # Predict positions (simple constant velocity model)
            pos1 = aircraft1.state.position + aircraft1.state.velocity * t
            pos2 = aircraft2.state.position + aircraft2.state.velocity * t
            
            # Calculate separation
            horizontal_sep = np.linalg.norm(pos1[:2] - pos2[:2])
            vertical_sep = abs(pos1[2] - pos2[2])
            
            # Check if separation is violated
            effective_min_sep = (self.constraints.minimum_separation 
                               if vertical_sep < self.constraints.vertical_separation
                               else self.constraints.minimum_separation * 0.8)
            
            if horizontal_sep < min_separation:
                min_separation = horizontal_sep
                if horizontal_sep < effective_min_sep:
                    conflict_time = current_time + t
        
        # Create conflict event if violation predicted
        if conflict_time is not None:
            level = self._classify_separation_level(min_separation, 0)  # Assume no vertical sep
            
            return SeparationEvent(
                aircraft1_id=aircraft1.id,
                aircraft2_id=aircraft2.id,
                distance=min_separation,
                level=level,
                timestamp=conflict_time,
                predicted_cpa_time=conflict_time,
                predicted_cpa_distance=min_separation
            )
        
        return None
    
    def enforce_separation_constraints(self, aircraft_list: List[Aircraft],
                                     current_time: float) -> Dict[str, Dict]:
        """
        Enforce separation constraints and generate corrective actions.
        
        Args:
            aircraft_list: List of aircraft
            current_time: Current time
            
        Returns:
            Dictionary of corrective actions for each aircraft
        """
        corrective_actions = {}
        
        # Get current violations
        current_events = self.monitor_separation(aircraft_list, current_time)
        
        # Get predicted conflicts
        predicted_conflicts = self.predict_conflicts(aircraft_list, current_time)
        
        # Generate corrective actions for current violations
        for event in current_events:
            if event.level in [SeparationLevel.VIOLATION, SeparationLevel.CRITICAL, 
                             SeparationLevel.EMERGENCY]:
                actions = self._generate_corrective_actions(event, aircraft_list)
                corrective_actions.update(actions)
        
        # Generate preventive actions for predicted conflicts
        for conflict in predicted_conflicts.values():
            if conflict.predicted_cpa_distance < self.constraints.minimum_separation:
                actions = self._generate_preventive_actions(conflict, aircraft_list)
                corrective_actions.update(actions)
        
        return corrective_actions
    
    def _generate_corrective_actions(self, event: SeparationEvent, 
                                   aircraft_list: List[Aircraft]) -> Dict[str, Dict]:
        """Generate corrective actions for a separation violation."""
        actions = {}
        
        # Find the aircraft objects
        aircraft1 = next((a for a in aircraft_list if a.id == event.aircraft1_id), None)
        aircraft2 = next((a for a in aircraft_list if a.id == event.aircraft2_id), None)
        
        if not aircraft1 or not aircraft2:
            return actions
        
        # Determine which aircraft should take action
        # Prefer the aircraft with more maneuvering capability
        if event.level == SeparationLevel.EMERGENCY:
            # Both aircraft take immediate action
            actions[aircraft1.id] = self._emergency_action(aircraft1, aircraft2)
            actions[aircraft2.id] = self._emergency_action(aircraft2, aircraft1)
        elif event.level == SeparationLevel.CRITICAL:
            # Primary aircraft takes strong action, secondary takes supporting action
            primary, secondary = self._select_primary_aircraft(aircraft1, aircraft2)
            actions[primary.id] = self._critical_action(primary, secondary)
            actions[secondary.id] = self._supporting_action(secondary, primary)
        else:  # VIOLATION
            # One aircraft takes moderate action
            primary, _ = self._select_primary_aircraft(aircraft1, aircraft2)
            actions[primary.id] = self._moderate_action(primary, aircraft2 if primary == aircraft1 else aircraft1)
        
        return actions
    
    def _generate_preventive_actions(self, conflict: SeparationEvent,
                                   aircraft_list: List[Aircraft]) -> Dict[str, Dict]:
        """Generate preventive actions for predicted conflicts."""
        actions = {}
        
        # Find the aircraft objects
        aircraft1 = next((a for a in aircraft_list if a.id == conflict.aircraft1_id), None)
        aircraft2 = next((a for a in aircraft_list if a.id == conflict.aircraft2_id), None)
        
        if not aircraft1 or not aircraft2:
            return actions
        
        # Less aggressive actions for prevention
        if conflict.predicted_cpa_distance < self.constraints.critical_separation:
            primary, _ = self._select_primary_aircraft(aircraft1, aircraft2)
            actions[primary.id] = self._preventive_action(primary, aircraft2 if primary == aircraft1 else aircraft1)
        
        return actions
    
    def _select_primary_aircraft(self, aircraft1: Aircraft, 
                               aircraft2: Aircraft) -> Tuple[Aircraft, Aircraft]:
        """Select which aircraft should be primary for conflict resolution."""
        # Simple heuristic: aircraft with higher speed has priority
        # (assuming it's easier to slow down than speed up)
        if aircraft1.state.speed > aircraft2.state.speed:
            return aircraft2, aircraft1  # Slower aircraft adjusts
        else:
            return aircraft1, aircraft2
    
    def _emergency_action(self, aircraft: Aircraft, threat: Aircraft) -> Dict:
        """Generate emergency avoidance action."""
        rel_pos = aircraft.state.position - threat.state.position
        avoidance_direction = rel_pos[:2] / np.linalg.norm(rel_pos[:2])
        
        return {
            'type': 'emergency_avoidance',
            'priority': 'immediate',
            'actions': {
                'turn_direction': avoidance_direction,
                'turn_rate': 'maximum',
                'speed_change': -0.3,  # Reduce speed by 30%
                'altitude_change': 150.0 if rel_pos[2] > 0 else -150.0  # 500 feet
            }
        }
    
    def _critical_action(self, aircraft: Aircraft, threat: Aircraft) -> Dict:
        """Generate critical avoidance action."""
        rel_pos = aircraft.state.position - threat.state.position
        avoidance_direction = rel_pos[:2] / np.linalg.norm(rel_pos[:2])
        
        return {
            'type': 'critical_avoidance',
            'priority': 'high',
            'actions': {
                'turn_direction': avoidance_direction,
                'turn_rate': 'high',
                'speed_change': -0.2,  # Reduce speed by 20%
                'altitude_change': 100.0 if rel_pos[2] > 0 else -100.0
            }
        }
    
    def _supporting_action(self, aircraft: Aircraft, primary: Aircraft) -> Dict:
        """Generate supporting action for secondary aircraft."""
        return {
            'type': 'supporting_action',
            'priority': 'medium',
            'actions': {
                'speed_change': -0.1,  # Slight speed reduction
                'maintain_heading': True
            }
        }
    
    def _moderate_action(self, aircraft: Aircraft, other: Aircraft) -> Dict:
        """Generate moderate corrective action."""
        rel_pos = aircraft.state.position - other.state.position
        avoidance_direction = rel_pos[:2] / np.linalg.norm(rel_pos[:2])
        
        return {
            'type': 'moderate_correction',
            'priority': 'medium',
            'actions': {
                'turn_direction': avoidance_direction,
                'turn_rate': 'moderate',
                'speed_change': -0.1
            }
        }
    
    def _preventive_action(self, aircraft: Aircraft, other: Aircraft) -> Dict:
        """Generate preventive action."""
        return {
            'type': 'preventive_action',
            'priority': 'low',
            'actions': {
                'speed_change': -0.05,  # Small speed adjustment
                'path_adjustment': 'minor'
            }
        }
    
    def _update_statistics(self) -> None:
        """Update monitoring statistics."""
        for event in self.current_events:
            if event.level == SeparationLevel.VIOLATION:
                self.statistics['total_violations'] += 1
            elif event.level == SeparationLevel.CRITICAL:
                self.statistics['critical_events'] += 1
            elif event.level == SeparationLevel.EMERGENCY:
                self.statistics['emergency_events'] += 1
            
            # Update minimum observed separation
            if event.distance < self.statistics['min_observed_separation']:
                self.statistics['min_observed_separation'] = event.distance
    
    def get_separation_report(self) -> Dict:
        """Get comprehensive separation monitoring report."""
        return {
            'current_events': len(self.current_events),
            'predicted_conflicts': len(self.predicted_conflicts),
            'statistics': self.statistics.copy(),
            'events_by_level': {
                level.value: len([e for e in self.current_events if e.level == level])
                for level in SeparationLevel
            }
        }
    
    def is_separation_adequate(self, aircraft1: Aircraft, aircraft2: Aircraft) -> bool:
        """Check if separation between two aircraft is adequate."""
        horizontal_distance = aircraft1.horizontal_distance_to(aircraft2)
        vertical_distance = abs(aircraft1.state.position[2] - aircraft2.state.position[2])
        
        # Apply different standards based on vertical separation
        if vertical_distance >= self.constraints.vertical_separation:
            return horizontal_distance >= self.constraints.minimum_separation * 0.8
        else:
            return horizontal_distance >= self.constraints.minimum_separation