"""
Lower-level controller (BL-DC) for fine-grained air traffic control.

This controller implements the detailed control layer that tracks set-points
from the upper-level controller while ensuring immediate safety constraints
and smooth trajectory execution.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import casadi as ca
from scipy.optimize import minimize

try:
    from ..models.aircraft import Aircraft, AircraftState
    from ..controllers.upper_level import SetPoint
except ImportError:
    from models.aircraft import Aircraft, AircraftState
    from controllers.upper_level import SetPoint


@dataclass
class ControlInput:
    """Control input for aircraft."""
    speed_command: float      # Commanded speed in m/s
    heading_rate: float       # Heading rate command in rad/s
    climb_rate: float         # Climb rate command in m/s
    timestamp: float          # Time when command was issued


@dataclass
class TacticalHorizon:
    """Tactical control horizon parameters."""
    prediction_horizon: float = 180.0    # 3 minutes prediction
    control_horizon: float = 60.0        # 1 minute control
    time_step: float = 5.0               # 5-second time steps
    update_frequency: float = 1.0        # Update every second


class LowerLevelController:
    """
    Lower-level controller for bi-level MPC air traffic management.
    
    This controller performs tactical control with high update frequency,
    tracking set-points from the upper-level controller while ensuring
    immediate safety and smoothness constraints.
    """
    
    def __init__(self):
        """Initialize lower-level controller."""
        self.tactical_horizon = TacticalHorizon()
        
        # MPC parameters
        self.prediction_steps = int(self.tactical_horizon.prediction_horizon / 
                                  self.tactical_horizon.time_step)
        self.control_steps = int(self.tactical_horizon.control_horizon / 
                               self.tactical_horizon.time_step)
        
        # Safety constraints
        self.emergency_separation = 3704.0  # 2 nautical miles emergency separation
        self.min_separation = 5556.0        # 3 nautical miles minimum separation
        self.conflict_horizon = 120.0       # 2 minutes conflict prediction
        
        # Control weights
        self.weights = {
            'setpoint_tracking': 10.0,
            'control_effort': 1.0,
            'control_smoothness': 2.0,
            'separation_violation': 1000.0,
            'emergency_separation': 10000.0,
            'speed_deviation': 5.0,
            'altitude_deviation': 8.0
        }
        
        # Control history for smoothness
        self.control_history: Dict[str, List[ControlInput]] = {}
        
        # Current control commands
        self.current_commands: Dict[str, ControlInput] = {}
        
    def compute_control(self, aircraft: Aircraft, 
                       setpoint: SetPoint,
                       nearby_aircraft: List[Aircraft],
                       current_time: float) -> ControlInput:
        """
        Compute optimal control input for an aircraft.
        
        Args:
            aircraft: Aircraft to control
            setpoint: Current set-point from upper-level controller
            nearby_aircraft: List of nearby aircraft for conflict avoidance
            current_time: Current simulation time
            
        Returns:
            Optimal control input
        """
        # Initialize control history if needed
        if aircraft.id not in self.control_history:
            self.control_history[aircraft.id] = []
        
        # Set up MPC optimization problem
        control_input = self._solve_mpc(aircraft, setpoint, nearby_aircraft, current_time)
        
        # Store control command
        self.current_commands[aircraft.id] = control_input
        self.control_history[aircraft.id].append(control_input)
        
        # Limit history length
        max_history = 60  # Keep last 60 commands (1 minute at 1Hz)
        if len(self.control_history[aircraft.id]) > max_history:
            self.control_history[aircraft.id] = self.control_history[aircraft.id][-max_history:]
        
        return control_input
    
    def _solve_mpc(self, aircraft: Aircraft, 
                   setpoint: SetPoint,
                   nearby_aircraft: List[Aircraft],
                   current_time: float) -> ControlInput:
        """
        Solve MPC optimization problem using CasADi.
        
        Args:
            aircraft: Aircraft to control
            setpoint: Target set-point
            nearby_aircraft: Nearby aircraft for conflict avoidance
            current_time: Current time
            
        Returns:
            Optimal control input
        """
        # Create optimization variables
        opti = ca.Opti()
        
        # State variables: [x, y, z, vx, vy, vz, heading, speed]
        n_states = 8
        n_controls = 3  # [speed_cmd, heading_rate, climb_rate]
        
        # Decision variables
        X = opti.variable(n_states, self.prediction_steps + 1)  # States
        U = opti.variable(n_controls, self.control_steps)       # Controls
        
        # Parameters
        X0 = opti.parameter(n_states)  # Initial state
        X_ref = opti.parameter(n_states, self.prediction_steps + 1)  # Reference trajectory
        
        # Set initial state
        current_state = aircraft.get_state_vector()
        opti.set_value(X0, current_state)
        
        # Generate reference trajectory
        ref_trajectory = self._generate_reference_trajectory(
            aircraft, setpoint, current_time
        )
        opti.set_value(X_ref, ref_trajectory)
        
        # Initial constraint
        opti.subject_to(X[:, 0] == X0)
        
        # Dynamics constraints
        dt = self.tactical_horizon.time_step
        for k in range(self.prediction_steps):
            # Get control input (use last control if beyond control horizon)
            if k < self.control_steps:
                u_k = U[:, k]
            else:
                u_k = U[:, -1]  # Hold last control
            
            # Aircraft dynamics
            x_next = self._aircraft_dynamics(X[:, k], u_k, dt)
            opti.subject_to(X[:, k + 1] == x_next)
        
        # Control constraints
        for k in range(self.control_steps):
            # Speed constraints
            opti.subject_to(U[0, k] >= aircraft.config.min_speed)
            opti.subject_to(U[0, k] <= aircraft.config.max_speed)
            
            # Heading rate constraints
            max_heading_rate = np.radians(aircraft.config.max_turn_rate)
            opti.subject_to(U[1, k] >= -max_heading_rate)
            opti.subject_to(U[1, k] <= max_heading_rate)
            
            # Climb rate constraints
            opti.subject_to(U[2, k] >= aircraft.config.max_descent_rate)
            opti.subject_to(U[2, k] <= aircraft.config.max_climb_rate)
        
        # Set-point constraints (soft)
        speed_min, speed_max = setpoint.speed_range
        alt_min, alt_max = setpoint.altitude_range
        
        for k in range(self.prediction_steps + 1):
            # Speed range constraints (soft)
            speed = X[7, k]
            opti.subject_to(speed >= speed_min * 0.9)  # Allow 10% deviation
            opti.subject_to(speed <= speed_max * 1.1)
            
            # Altitude range constraints (soft)
            altitude = X[2, k]
            opti.subject_to(altitude >= alt_min - 150.0)  # Allow 150m deviation
            opti.subject_to(altitude <= alt_max + 150.0)
        
        # Separation constraints with nearby aircraft
        self._add_separation_constraints(opti, X, nearby_aircraft, current_time)
        
        # Objective function
        objective = self._build_objective(opti, X, U, X_ref, aircraft)
        opti.minimize(objective)
        
        # Solver settings
        opti.solver('ipopt', {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 100,
            'ipopt.acceptable_tol': 1e-6,
            'ipopt.warm_start_init_point': 'yes'
        })
        
        try:
            # Solve optimization
            sol = opti.solve()
            
            # Extract first control input
            u_opt = sol.value(U[:, 0])
            
            return ControlInput(
                speed_command=float(u_opt[0]),
                heading_rate=float(u_opt[1]),
                climb_rate=float(u_opt[2]),
                timestamp=current_time
            )
            
        except Exception as e:
            # Fallback to simple controller if optimization fails
            print(f"MPC optimization failed for aircraft {aircraft.id}: {e}")
            return self._fallback_controller(aircraft, setpoint, current_time)
    
    def _aircraft_dynamics(self, state: ca.MX, control: ca.MX, dt: float) -> ca.MX:
        """
        Aircraft dynamics model for MPC.
        
        Args:
            state: Current state [x, y, z, vx, vy, vz, heading, speed]
            control: Control input [speed_cmd, heading_rate, climb_rate]
            dt: Time step
            
        Returns:
            Next state
        """
        # Extract state variables
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        heading = state[6]
        speed = state[7]
        
        # Extract control inputs
        speed_cmd, heading_rate, climb_rate = control[0], control[1], control[2]
        
        # Update heading
        heading_next = heading + heading_rate * dt
        
        # Update speed (simple first-order dynamics)
        speed_next = speed + (speed_cmd - speed) * dt / 10.0  # 10-second time constant
        
        # Update velocity components
        vx_next = speed_next * ca.cos(heading_next)
        vy_next = speed_next * ca.sin(heading_next)
        vz_next = climb_rate
        
        # Update position
        x_next = x + vx_next * dt
        y_next = y + vy_next * dt
        z_next = z + vz_next * dt
        
        return ca.vertcat(x_next, y_next, z_next, vx_next, vy_next, vz_next, 
                         heading_next, speed_next)
    
    def _generate_reference_trajectory(self, aircraft: Aircraft, 
                                     setpoint: SetPoint,
                                     current_time: float) -> np.ndarray:
        """Generate reference trajectory for MPC horizon."""
        ref_trajectory = np.zeros((8, self.prediction_steps + 1))
        
        # Simple reference: linear interpolation to set-point
        current_state = aircraft.get_state_vector()
        target_position = setpoint.position
        target_speed = np.mean(setpoint.speed_range)
        
        for k in range(self.prediction_steps + 1):
            progress = k / self.prediction_steps if self.prediction_steps > 0 else 0
            
            # Interpolate position
            ref_position = (1 - progress) * current_state[:3] + progress * target_position
            
            # Calculate required velocity
            if k > 0:
                dt = self.tactical_horizon.time_step
                ref_velocity = (ref_position - ref_trajectory[:3, k-1]) / dt
            else:
                ref_velocity = current_state[3:6]
            
            # Target heading towards set-point
            if k < self.prediction_steps:
                direction = target_position[:2] - ref_position[:2]
                if np.linalg.norm(direction) > 1e-6:
                    ref_heading = np.arctan2(direction[1], direction[0])
                else:
                    ref_heading = current_state[6]
            else:
                ref_heading = ref_trajectory[6, k-1]
            
            # Reference state
            ref_trajectory[:, k] = np.array([
                ref_position[0], ref_position[1], ref_position[2],  # position
                ref_velocity[0], ref_velocity[1], ref_velocity[2],  # velocity
                ref_heading,                                        # heading
                target_speed                                        # speed
            ])
        
        return ref_trajectory
    
    def _add_separation_constraints(self, opti: ca.Opti, X: ca.MX, 
                                  nearby_aircraft: List[Aircraft],
                                  current_time: float) -> None:
        """Add separation constraints to optimization problem."""
        dt = self.tactical_horizon.time_step
        
        for other_aircraft in nearby_aircraft:
            # Predict other aircraft trajectory (simple constant velocity)
            other_pos = other_aircraft.state.position
            other_vel = other_aircraft.state.velocity
            
            for k in range(self.prediction_steps + 1):
                t = k * dt
                # Predicted position of other aircraft
                other_pred_pos = other_pos + other_vel * t
                
                # Horizontal separation constraint
                pos_diff = X[:2, k] - other_pred_pos[:2]
                horizontal_dist_sq = ca.dot(pos_diff, pos_diff)
                
                # Minimum separation constraint (soft)
                min_sep_sq = self.min_separation ** 2
                opti.subject_to(horizontal_dist_sq >= min_sep_sq * 0.8)  # Soft constraint
    
    def _build_objective(self, opti: ca.Opti, X: ca.MX, U: ca.MX, 
                        X_ref: ca.MX, aircraft: Aircraft) -> ca.MX:
        """Build MPC objective function."""
        objective = 0
        
        # Set-point tracking cost
        for k in range(self.prediction_steps + 1):
            state_error = X[:, k] - X_ref[:, k]
            # Weighted tracking error
            position_error = ca.dot(state_error[:3], state_error[:3])
            speed_error = state_error[7] ** 2
            heading_error = state_error[6] ** 2
            
            objective += self.weights['setpoint_tracking'] * (
                position_error + speed_error + heading_error
            )
        
        # Control effort cost
        for k in range(self.control_steps):
            control_effort = ca.dot(U[:, k], U[:, k])
            objective += self.weights['control_effort'] * control_effort
        
        # Control smoothness cost
        for k in range(self.control_steps - 1):
            control_diff = U[:, k+1] - U[:, k]
            objective += self.weights['control_smoothness'] * ca.dot(control_diff, control_diff)
        
        # Add smoothness with previous control if available
        if aircraft.id in self.control_history and self.control_history[aircraft.id]:
            last_control = self.control_history[aircraft.id][-1]
            last_u = np.array([last_control.speed_command, 
                             last_control.heading_rate, 
                             last_control.climb_rate])
            first_control_diff = U[:, 0] - last_u
            objective += self.weights['control_smoothness'] * ca.dot(first_control_diff, first_control_diff)
        
        return objective
    
    def _fallback_controller(self, aircraft: Aircraft, 
                           setpoint: SetPoint,
                           current_time: float) -> ControlInput:
        """
        Simple fallback controller when MPC fails.
        
        Args:
            aircraft: Aircraft to control
            setpoint: Target set-point
            current_time: Current time
            
        Returns:
            Simple control input
        """
        # Simple proportional controller
        current_pos = aircraft.state.position
        target_pos = setpoint.position
        
        # Speed command
        target_speed = np.mean(setpoint.speed_range)
        speed_error = target_speed - aircraft.state.speed
        speed_cmd = aircraft.state.speed + np.clip(speed_error * 0.1, -5.0, 5.0)
        speed_cmd = np.clip(speed_cmd, aircraft.config.min_speed, aircraft.config.max_speed)
        
        # Heading command
        direction = target_pos[:2] - current_pos[:2]
        if np.linalg.norm(direction) > 1e-6:
            target_heading = np.arctan2(direction[1], direction[0])
            heading_error = target_heading - aircraft.state.heading
            
            # Normalize heading error to [-pi, pi]
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            
            heading_rate = np.clip(heading_error * 0.5, 
                                 -np.radians(aircraft.config.max_turn_rate),
                                 np.radians(aircraft.config.max_turn_rate))
        else:
            heading_rate = 0.0
        
        # Climb rate command
        altitude_error = target_pos[2] - current_pos[2]
        climb_rate = np.clip(altitude_error * 0.1,
                           aircraft.config.max_descent_rate,
                           aircraft.config.max_climb_rate)
        
        return ControlInput(
            speed_command=speed_cmd,
            heading_rate=heading_rate,
            climb_rate=climb_rate,
            timestamp=current_time
        )
    
    def emergency_avoidance(self, aircraft: Aircraft, 
                          threat_aircraft: Aircraft,
                          current_time: float) -> ControlInput:
        """
        Emergency avoidance maneuver when separation is critically low.
        
        Args:
            aircraft: Aircraft to control
            threat_aircraft: Threatening aircraft
            current_time: Current time
            
        Returns:
            Emergency control input
        """
        # Calculate relative position
        rel_pos = aircraft.state.position - threat_aircraft.state.position
        rel_distance = np.linalg.norm(rel_pos[:2])
        
        if rel_distance < self.emergency_separation:
            # Emergency maneuver: turn away and slow down
            avoidance_direction = rel_pos[:2] / np.linalg.norm(rel_pos[:2])
            target_heading = np.arctan2(avoidance_direction[1], avoidance_direction[0])
            
            heading_error = target_heading - aircraft.state.heading
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            
            # Maximum turn rate for emergency
            heading_rate = np.sign(heading_error) * np.radians(aircraft.config.max_turn_rate)
            
            # Reduce speed
            speed_cmd = max(aircraft.config.min_speed, aircraft.state.speed * 0.8)
            
            return ControlInput(
                speed_command=speed_cmd,
                heading_rate=heading_rate,
                climb_rate=0.0,  # Maintain altitude during emergency
                timestamp=current_time
            )
        
        return self.current_commands.get(aircraft.id, ControlInput(
            speed_command=aircraft.state.speed,
            heading_rate=0.0,
            climb_rate=0.0,
            timestamp=current_time
        ))
    
    def check_immediate_conflicts(self, aircraft_list: List[Aircraft]) -> List[Tuple[str, str]]:
        """
        Check for immediate conflicts requiring emergency action.
        
        Args:
            aircraft_list: List of all aircraft
            
        Returns:
            List of aircraft ID pairs in immediate conflict
        """
        conflicts = []
        
        for i, aircraft1 in enumerate(aircraft_list):
            for j, aircraft2 in enumerate(aircraft_list[i+1:], i+1):
                distance = aircraft1.horizontal_distance_to(aircraft2)
                
                if distance < self.emergency_separation:
                    conflicts.append((aircraft1.id, aircraft2.id))
                    
        return conflicts