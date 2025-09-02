"""
Test cases for aircraft model functionality.
"""
import unittest
import numpy as np

from src.models.aircraft import Aircraft, AircraftState, AircraftConfig, FlightPhase


class TestAircraftModel(unittest.TestCase):
    """Test cases for Aircraft class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.initial_position = np.array([10000.0, 5000.0, 1000.0])
        self.initial_velocity = np.array([100.0, 50.0, 0.0])
        self.initial_heading = np.pi / 4  # 45 degrees
        self.initial_speed = 150.0
        
        self.initial_state = AircraftState(
            position=self.initial_position,
            velocity=self.initial_velocity,
            heading=self.initial_heading,
            altitude=1000.0,
            speed=self.initial_speed,
            timestamp=0.0
        )
        
        self.config = AircraftConfig(
            max_speed=250.0,
            min_speed=70.0,
            max_turn_rate=3.0,
            max_climb_rate=15.0,
            max_descent_rate=-10.0
        )
        
        self.aircraft = Aircraft(
            aircraft_id="TEST001",
            initial_state=self.initial_state,
            config=self.config,
            flight_phase=FlightPhase.ARRIVAL
        )
    
    def test_aircraft_initialization(self):
        """Test aircraft initialization."""
        self.assertEqual(self.aircraft.id, "TEST001")
        self.assertEqual(self.aircraft.flight_phase, FlightPhase.ARRIVAL)
        np.testing.assert_array_equal(self.aircraft.state.position, self.initial_position)
        np.testing.assert_array_equal(self.aircraft.state.velocity, self.initial_velocity)
        self.assertEqual(self.aircraft.state.heading, self.initial_heading)
        self.assertEqual(self.aircraft.state.speed, self.initial_speed)
    
    def test_dynamics_update(self):
        """Test aircraft dynamics update."""
        dt = 1.0
        control_input = np.array([200.0, np.radians(2.0), 5.0])  # speed, heading_rate, climb_rate
        
        initial_position = self.aircraft.state.position.copy()
        initial_heading = self.aircraft.state.heading
        
        self.aircraft.update_dynamics(dt, control_input)
        
        # Check that position has changed
        self.assertFalse(np.array_equal(self.aircraft.state.position, initial_position))
        
        # Check that heading has changed
        expected_heading = initial_heading + np.radians(2.0) * dt
        self.assertAlmostEqual(self.aircraft.state.heading, expected_heading, places=5)
        
        # Check that speed command is within limits
        self.assertLessEqual(self.aircraft.commanded_speed, self.config.max_speed)
        self.assertGreaterEqual(self.aircraft.commanded_speed, self.config.min_speed)
    
    def test_control_limits(self):
        """Test that control inputs are properly limited."""
        dt = 1.0
        
        # Test speed limits
        excessive_speed = self.config.max_speed + 50.0
        control_input = np.array([excessive_speed, 0.0, 0.0])
        self.aircraft.update_dynamics(dt, control_input)
        self.assertEqual(self.aircraft.commanded_speed, self.config.max_speed)
        
        # Test turn rate limits
        excessive_turn_rate = np.radians(10.0)  # 10 deg/s > max 3 deg/s
        control_input = np.array([150.0, excessive_turn_rate, 0.0])
        initial_heading = self.aircraft.state.heading
        self.aircraft.update_dynamics(dt, control_input)
        
        max_heading_change = np.radians(self.config.max_turn_rate) * dt
        actual_heading_change = abs(self.aircraft.state.heading - initial_heading)
        self.assertLessEqual(actual_heading_change, max_heading_change + 1e-6)
    
    def test_future_position_prediction(self):
        """Test future position prediction."""
        time_horizon = 10.0
        num_points = 5
        
        future_positions = self.aircraft.get_future_position(time_horizon, num_points)
        
        self.assertEqual(future_positions.shape, (num_points, 3))
        
        # Check that future positions are reasonable
        dt = time_horizon / num_points
        for i in range(num_points):
            expected_pos = self.initial_position + self.initial_velocity * (i + 1) * dt
            np.testing.assert_array_almost_equal(future_positions[i], expected_pos, decimal=3)
    
    def test_distance_calculation(self):
        """Test distance calculation between aircraft."""
        # Create second aircraft
        other_position = np.array([15000.0, 8000.0, 1200.0])
        other_state = AircraftState(
            position=other_position,
            velocity=np.array([120.0, 60.0, 0.0]),
            heading=np.pi/3,
            altitude=1200.0,
            speed=140.0,
            timestamp=0.0
        )
        
        other_aircraft = Aircraft("TEST002", other_state, self.config)
        
        # Test 3D distance
        expected_distance = np.linalg.norm(self.initial_position - other_position)
        calculated_distance = self.aircraft.distance_to(other_aircraft)
        self.assertAlmostEqual(calculated_distance, expected_distance, places=3)
        
        # Test horizontal distance
        expected_horizontal = np.linalg.norm(self.initial_position[:2] - other_position[:2])
        calculated_horizontal = self.aircraft.horizontal_distance_to(other_aircraft)
        self.assertAlmostEqual(calculated_horizontal, expected_horizontal, places=3)
    
    def test_reference_trajectory(self):
        """Test reference trajectory setting and retrieval."""
        waypoints = [
            np.array([12000.0, 6000.0, 800.0]),
            np.array([8000.0, 4000.0, 600.0]),
            np.array([0.0, 0.0, 300.0])
        ]
        times = [10.0, 20.0, 30.0]
        
        self.aircraft.set_reference_trajectory(waypoints, times)
        
        self.assertEqual(len(self.aircraft.reference_waypoints), 3)
        self.assertEqual(len(self.aircraft.reference_times), 3)
        np.testing.assert_array_equal(self.aircraft.reference_waypoints[0], waypoints[0])
        self.assertEqual(self.aircraft.reference_times[1], 20.0)
    
    def test_state_vector(self):
        """Test state vector generation."""
        state_vector = self.aircraft.get_state_vector()
        
        expected_vector = np.concatenate([
            self.initial_position,
            self.initial_velocity,
            [self.initial_heading, self.initial_speed]
        ])
        
        np.testing.assert_array_equal(state_vector, expected_vector)
        self.assertEqual(len(state_vector), 8)
    
    def test_aircraft_config(self):
        """Test aircraft configuration validation."""
        # Test default configuration
        default_config = AircraftConfig()
        self.assertGreater(default_config.max_speed, default_config.min_speed)
        self.assertGreater(default_config.max_climb_rate, 0)
        self.assertLess(default_config.max_descent_rate, 0)
        
        # Test custom configuration
        custom_config = AircraftConfig(
            max_speed=300.0,
            min_speed=80.0,
            max_turn_rate=2.5
        )
        self.assertEqual(custom_config.max_speed, 300.0)
        self.assertEqual(custom_config.min_speed, 80.0)
        self.assertEqual(custom_config.max_turn_rate, 2.5)


class TestAircraftState(unittest.TestCase):
    """Test cases for AircraftState class."""
    
    def test_state_creation(self):
        """Test aircraft state creation."""
        position = np.array([1000.0, 2000.0, 3000.0])
        velocity = np.array([100.0, 50.0, 10.0])
        heading = np.pi / 2
        altitude = 3000.0
        speed = 150.0
        timestamp = 123.456
        
        state = AircraftState(
            position=position,
            velocity=velocity,
            heading=heading,
            altitude=altitude,
            speed=speed,
            timestamp=timestamp
        )
        
        np.testing.assert_array_equal(state.position, position)
        np.testing.assert_array_equal(state.velocity, velocity)
        self.assertEqual(state.heading, heading)
        self.assertEqual(state.altitude, altitude)
        self.assertEqual(state.speed, speed)
        self.assertEqual(state.timestamp, timestamp)


if __name__ == '__main__':
    unittest.main()