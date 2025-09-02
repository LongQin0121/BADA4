#!/usr/bin/env python3
"""
Simplified demonstration without complex MPC optimization.
This version shows the system architecture working with simpler controllers.
"""
import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.aircraft import Aircraft, AircraftState, AircraftConfig, FlightPhase
from systems.point_merge import PointMergeSystem, SequenceType
from utils.separation_monitor import SeparationMonitor


def create_demo_aircraft(aircraft_id, entry_angle, distance=40000):
    """Create a demo aircraft."""
    # Position around the merge point
    initial_position = np.array([
        distance * np.cos(entry_angle),
        distance * np.sin(entry_angle),
        2000.0  # 2 km altitude
    ])
    
    # Heading towards center
    heading = np.arctan2(-initial_position[1], -initial_position[0])
    speed = 200.0
    
    initial_velocity = np.array([
        speed * np.cos(heading),
        speed * np.sin(heading),
        -2.0  # Slight descent
    ])
    
    initial_state = AircraftState(
        position=initial_position,
        velocity=initial_velocity,
        heading=heading,
        altitude=2000.0,
        speed=speed,
        timestamp=0.0
    )
    
    config = AircraftConfig()
    
    return Aircraft(aircraft_id, initial_state, config, FlightPhase.ARRIVAL)


def demonstrate_point_merge_system():
    """Demonstrate the point merge system functionality."""
    print("="*60)
    print("POINT MERGE SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Create point merge system
    pms = PointMergeSystem()
    
    print(f"Merge Point: {pms.merge_point.name}")
    print(f"Position: {pms.merge_point.position}")
    print(f"Minimum Separation: {pms.min_separation_distance/1852:.1f} NM")
    
    # Create some demo aircraft
    aircraft_list = [
        create_demo_aircraft("SAS001", np.radians(45), 50000),   # Northeast
        create_demo_aircraft("DLH002", np.radians(135), 45000),  # Southeast  
        create_demo_aircraft("BAW003", np.radians(225), 40000),  # Southwest
        create_demo_aircraft("AFR004", np.radians(315), 35000),  # Northwest
    ]
    
    print(f"\nCreated {len(aircraft_list)} demonstration aircraft:")
    for aircraft in aircraft_list:
        pos = aircraft.state.position[:2]/1000
        print(f"  {aircraft.id}: Position ({pos[0]:.1f}, {pos[1]:.1f}) km")
    
    # Assign aircraft to legs
    print(f"\nAssigning aircraft to sequencing legs:")
    for aircraft in aircraft_list:
        leg_type = pms.assign_aircraft_to_leg(aircraft)
        print(f"  {aircraft.id} -> {leg_type.value.upper()} leg")
    
    # Generate reference trajectories
    print(f"\nGenerating reference trajectories:")
    for aircraft in aircraft_list:
        try:
            waypoints, times = pms.generate_reference_trajectory(aircraft)
            print(f"  {aircraft.id}: {len(waypoints)} waypoints over {times[-1]:.0f}s")
        except Exception as e:
            print(f"  {aircraft.id}: Error generating trajectory - {e}")
    
    # Calculate merge sequence
    sequence = pms.calculate_merge_sequence(aircraft_list)
    print(f"\nOptimal merge sequence: {' -> '.join(sequence)}")
    
    # Calculate turn points
    print(f"\nTurn points for sequencing:")
    for i, aircraft_id in enumerate(sequence):
        aircraft = next(a for a in aircraft_list if a.id == aircraft_id)
        turn_point = pms.get_turn_point(aircraft, i)
        if turn_point is not None:
            tp = turn_point[:2]/1000
            print(f"  {aircraft_id}: Turn at ({tp[0]:.1f}, {tp[1]:.1f}) km")
        else:
            print(f"  {aircraft_id}: No turn point required")
    
    return aircraft_list, pms


def demonstrate_separation_monitoring(aircraft_list):
    """Demonstrate separation monitoring."""
    print("\n" + "="*60)
    print("SEPARATION MONITORING DEMONSTRATION")
    print("="*60)
    
    # Create separation monitor
    monitor = SeparationMonitor()
    
    # Check current separations
    events = monitor.monitor_separation(aircraft_list, 0.0)
    
    print(f"Current separation events: {len(events)}")
    for event in events:
        print(f"  {event.aircraft1_id} <-> {event.aircraft2_id}: "
              f"{event.distance/1852:.1f} NM ({event.level.value})")
    
    # Predict future conflicts
    conflicts = monitor.predict_conflicts(aircraft_list, 0.0)
    
    print(f"\nPredicted conflicts: {len(conflicts)}")
    for pair, conflict in conflicts.items():
        print(f"  {conflict.aircraft1_id} <-> {conflict.aircraft2_id}: "
              f"Min separation {conflict.predicted_cpa_distance/1852:.1f} NM "
              f"at time {conflict.predicted_cpa_time:.0f}s")
    
    # Check separation adequacy
    print(f"\nPairwise separation adequacy:")
    for i, aircraft1 in enumerate(aircraft_list):
        for aircraft2 in aircraft_list[i+1:]:
            is_adequate = monitor.is_separation_adequate(aircraft1, aircraft2)
            distance = aircraft1.horizontal_distance_to(aircraft2)
            status = "✓" if is_adequate else "✗"
            print(f"  {aircraft1.id} <-> {aircraft2.id}: "
                  f"{distance/1852:.1f} NM {status}")
    
    # Get monitoring report
    report = monitor.get_separation_report()
    print(f"\nMonitoring Statistics:")
    print(f"  Events by level: {report['events_by_level']}")
    
    return monitor


def simulate_simple_movement(aircraft_list, pms, monitor, duration=300):
    """Simulate simple aircraft movement without complex MPC."""
    print("\n" + "="*60)
    print("SIMPLE MOVEMENT SIMULATION")
    print("="*60)
    
    dt = 10.0  # 10 second time steps
    time_steps = int(duration / dt)
    
    print(f"Simulating {duration}s with {dt}s time steps...")
    
    for step in range(time_steps):
        current_time = step * dt
        
        # Simple movement: each aircraft moves towards merge point
        for aircraft in aircraft_list:
            # Calculate direction to merge point
            to_merge = pms.merge_point.position - aircraft.state.position
            distance_to_merge = np.linalg.norm(to_merge)
            
            if distance_to_merge > 2000:  # Still far from merge
                # Normalize direction
                direction = to_merge / distance_to_merge
                
                # Simple speed control: slower when closer
                target_speed = min(220.0, max(120.0, distance_to_merge / 200))
                
                # Update aircraft state
                aircraft.state.velocity = direction * target_speed
                aircraft.state.position += aircraft.state.velocity * dt
                aircraft.state.speed = target_speed
                aircraft.state.heading = np.arctan2(direction[1], direction[0])
                aircraft.state.timestamp = current_time
        
        # Monitor separation every 30 seconds
        if step % 3 == 0:
            events = monitor.monitor_separation(aircraft_list, current_time)
            violations = [e for e in events if e.level.value in ['violation', 'critical', 'emergency']]
            
            distances = [f'{a.horizontal_distance_to(aircraft_list[(i+1)%len(aircraft_list)])/1852:.1f}' for i, a in enumerate(aircraft_list)][:3]
            print(f"  Time {current_time:3.0f}s: {len(violations)} violations, "
                  f"distances: {distances} NM")
    
    print(f"\nFinal positions:")
    for aircraft in aircraft_list:
        distance_to_merge = np.linalg.norm(aircraft.state.position - pms.merge_point.position)
        print(f"  {aircraft.id}: {distance_to_merge/1000:.1f} km from merge point")


def main():
    """Main demonstration."""
    print("ATM Bi-Level MPC System - Simplified Demonstration")
    print("="*60)
    
    try:
        # Demonstrate point merge system
        aircraft_list, pms = demonstrate_point_merge_system()
        
        # Demonstrate separation monitoring
        monitor = demonstrate_separation_monitoring(aircraft_list)
        
        # Simulate simple movement
        simulate_simple_movement(aircraft_list, pms, monitor)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Point Merge System - Aircraft assignment and sequencing")
        print("✓ Reference Trajectory Generation")
        print("✓ Separation Monitoring - Real-time conflict detection")
        print("✓ Turn Point Calculation")
        print("✓ Simple Aircraft Movement Simulation")
        print("\nNote: This demo uses simplified controllers.")
        print("The full MPC optimization is available but requires parameter tuning.")
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()