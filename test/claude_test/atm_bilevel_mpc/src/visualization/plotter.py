"""
Visualization tools for air traffic management simulation.

This module provides plotting and visualization capabilities for analyzing
simulation results, aircraft trajectories, and system performance.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
import pandas as pd

try:
    from ..models.aircraft import Aircraft
    from ..systems.point_merge import PointMergeSystem, SequenceType
    from ..utils.separation_monitor import SeparationEvent, SeparationLevel
except ImportError:
    from models.aircraft import Aircraft
    from systems.point_merge import PointMergeSystem, SequenceType
    from utils.separation_monitor import SeparationEvent, SeparationLevel


class ATMVisualizer:
    """
    Air Traffic Management visualization tool.
    
    This class provides comprehensive visualization capabilities for analyzing
    ATM simulation results and real-time monitoring.
    """
    
    def __init__(self, point_merge_system: PointMergeSystem):
        """
        Initialize visualizer.
        
        Args:
            point_merge_system: Point merge system for reference
        """
        self.pms = point_merge_system
        
        # Plot configuration
        self.colors = {
            'aircraft': 'blue',
            'trajectory': 'lightblue',
            'merge_point': 'red',
            'sequencing_leg': 'green',
            'separation_violation': 'red',
            'separation_warning': 'orange',
            'safe_aircraft': 'blue'
        }
        
        # Figure size and DPI
        self.fig_size = (12, 10)
        self.dpi = 100
        
    def plot_terminal_area(self, aircraft_list: Optional[List[Aircraft]] = None,
                          show_legs: bool = True, 
                          show_separation_circles: bool = False) -> plt.Figure:
        """
        Plot the terminal area with point merge system.
        
        Args:
            aircraft_list: List of aircraft to display
            show_legs: Whether to show sequencing legs
            show_separation_circles: Whether to show separation circles
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Plot merge point
        merge_pos = self.pms.merge_point.position
        ax.plot(merge_pos[0], merge_pos[1], 'ro', markersize=10, 
                label='Merge Point', zorder=5)
        
        # Plot sequencing legs
        if show_legs:
            self._plot_sequencing_legs(ax)
        
        # Plot aircraft
        if aircraft_list:
            self._plot_aircraft(ax, aircraft_list, show_separation_circles)
        
        # Set axis properties
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Stockholm Arlanda Terminal Area - Point Merge System')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        
        return fig
    
    def _plot_sequencing_legs(self, ax: plt.Axes) -> None:
        """Plot sequencing legs."""
        for leg_type, leg in self.pms.sequencing_legs.items():
            # Calculate leg end point
            leg_vector = np.array([
                np.cos(leg.direction),
                np.sin(leg.direction),
                0.0
            ]) * leg.length
            
            end_point = leg.entry_point + leg_vector
            
            # Plot leg
            ax.plot([leg.entry_point[0], end_point[0]], 
                   [leg.entry_point[1], end_point[1]], 
                   'g-', linewidth=2, alpha=0.7, 
                   label=f'{leg_type.value.title()} Leg' if leg_type.value == 'north' else "")
            
            # Plot entry point
            ax.plot(leg.entry_point[0], leg.entry_point[1], 'go', 
                   markersize=8, alpha=0.7)
            
            # Add leg label
            mid_point = (leg.entry_point + end_point) / 2
            ax.text(mid_point[0], mid_point[1], leg_type.value.upper(), 
                   ha='center', va='center', fontsize=10, fontweight='bold')
    
    def _plot_aircraft(self, ax: plt.Axes, aircraft_list: List[Aircraft],
                      show_separation_circles: bool) -> None:
        """Plot aircraft positions."""
        for aircraft in aircraft_list:
            pos = aircraft.state.position
            
            # Plot aircraft position
            ax.plot(pos[0], pos[1], 'bo', markersize=8, 
                   label='Aircraft' if aircraft == aircraft_list[0] else "")
            
            # Add aircraft ID
            ax.text(pos[0] + 1000, pos[1] + 1000, aircraft.id, 
                   fontsize=8, ha='left')
            
            # Plot heading vector
            heading_vector = np.array([
                np.cos(aircraft.state.heading),
                np.sin(aircraft.state.heading)
            ]) * 5000  # 5km vector
            
            ax.arrow(pos[0], pos[1], heading_vector[0], heading_vector[1], 
                    head_width=1000, head_length=1500, fc='blue', ec='blue',
                    alpha=0.7)
            
            # Plot separation circle
            if show_separation_circles:
                circle = Circle((pos[0], pos[1]), 5556, fill=False, 
                              edgecolor='red', linestyle='--', alpha=0.5)
                ax.add_patch(circle)
    
    def plot_trajectory_history(self, trajectory_data: Dict[str, List[Dict]], 
                              aircraft_ids: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot historical trajectories.
        
        Args:
            trajectory_data: Trajectory data from simulation
            aircraft_ids: Specific aircraft to plot (None for all)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Plot merge point and legs
        merge_pos = self.pms.merge_point.position
        ax.plot(merge_pos[0], merge_pos[1], 'ro', markersize=10, 
                label='Merge Point')
        self._plot_sequencing_legs(ax)
        
        # Plot trajectories
        aircraft_to_plot = aircraft_ids if aircraft_ids else list(trajectory_data.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(aircraft_to_plot)))
        
        for i, aircraft_id in enumerate(aircraft_to_plot):
            if aircraft_id not in trajectory_data:
                continue
                
            trajectory = trajectory_data[aircraft_id]
            if not trajectory:
                continue
            
            # Extract positions
            positions = np.array([point['position'] for point in trajectory])
            
            # Plot trajectory
            ax.plot(positions[:, 0], positions[:, 1], 
                   color=colors[i], linewidth=2, alpha=0.7,
                   label=f'Aircraft {aircraft_id}')
            
            # Mark start and end points
            ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=8)
            ax.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=8)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Aircraft Trajectory History')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')
        
        return fig
    
    def plot_separation_events(self, events_log: List[Dict]) -> plt.Figure:
        """
        Plot separation events over time.
        
        Args:
            events_log: List of separation events
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=self.dpi)
        
        # Filter separation events
        sep_events = [e for e in events_log if e['type'] == 'separation_event']
        
        if not sep_events:
            ax1.text(0.5, 0.5, 'No separation events recorded', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No separation events recorded', 
                    ha='center', va='center', transform=ax2.transAxes)
            return fig
        
        # Extract data
        times = [e['time'] for e in sep_events]
        distances = [e['distance'] for e in sep_events]
        levels = [e['level'] for e in sep_events]
        
        # Plot 1: Separation distances over time
        level_colors = {
            'warning': 'orange',
            'violation': 'red',
            'critical': 'darkred',
            'emergency': 'purple'
        }
        
        for level in set(levels):
            level_times = [t for t, l in zip(times, levels) if l == level]
            level_distances = [d for d, l in zip(distances, levels) if l == level]
            
            ax1.scatter(level_times, level_distances, 
                       c=level_colors.get(level, 'blue'), 
                       label=level.title(), alpha=0.7)
        
        # Add minimum separation line
        ax1.axhline(y=5556, color='green', linestyle='--', 
                   label='Minimum Separation (3 NM)')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Separation Distance (m)')
        ax1.set_title('Separation Events Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Event count by level
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        ax2.bar(level_counts.keys(), level_counts.values(), 
               color=[level_colors.get(level, 'blue') for level in level_counts.keys()])
        ax2.set_xlabel('Separation Level')
        ax2.set_ylabel('Event Count')
        ax2.set_title('Separation Event Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_system_performance(self, statistics_data: List[Dict]) -> plt.Figure:
        """
        Plot system performance metrics.
        
        Args:
            statistics_data: Performance statistics over time
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        if not statistics_data:
            return fig
        
        # Extract time series data
        times = [d['time'] for d in statistics_data]
        aircraft_counts = [d['aircraft_count'] for d in statistics_data]
        
        # Plot 1: Aircraft count over time
        ax1.plot(times, aircraft_counts, 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Aircraft Count')
        ax1.set_title('Aircraft in System Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Leg assignment distribution
        leg_assignments = {}
        for data in statistics_data:
            assignments = data.get('leg_assignments', {})
            for aircraft_id, leg_type in assignments.items():
                if leg_type not in leg_assignments:
                    leg_assignments[leg_type] = 0
                leg_assignments[leg_type] += 1
        
        if leg_assignments:
            ax2.pie(leg_assignments.values(), labels=leg_assignments.keys(), 
                   autopct='%1.1f%%', startangle=90)
            ax2.set_title('Aircraft Distribution by Sequencing Leg')
        
        # Plot 3: Sequence order changes
        sequence_lengths = [len(d.get('sequence_order', [])) for d in statistics_data]
        ax3.plot(times, sequence_lengths, 'g-', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Sequence Length')
        ax3.set_title('Merge Sequence Length Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Workload distribution
        max_aircraft = max(aircraft_counts) if aircraft_counts else 0
        workload_bins = np.arange(0, max_aircraft + 2)
        ax4.hist(aircraft_counts, bins=workload_bins, alpha=0.7, color='skyblue')
        ax4.set_xlabel('Aircraft Count')
        ax4.set_ylabel('Frequency')
        ax4.set_title('System Workload Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_animation(self, trajectory_data: Dict[str, List[Dict]], 
                        interval: int = 100) -> FuncAnimation:
        """
        Create animated visualization of aircraft movement.
        
        Args:
            trajectory_data: Trajectory data
            interval: Animation interval in milliseconds
            
        Returns:
            Matplotlib animation object
        """
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Setup plot
        merge_pos = self.pms.merge_point.position
        ax.plot(merge_pos[0], merge_pos[1], 'ro', markersize=10)
        self._plot_sequencing_legs(ax)
        
        # Find time range
        all_times = []
        for aircraft_traj in trajectory_data.values():
            all_times.extend([point['time'] for point in aircraft_traj])
        
        if not all_times:
            return None
        
        time_range = (min(all_times), max(all_times))
        time_step = 10.0  # 10 second steps
        
        # Animation function
        def animate(frame):
            ax.clear()
            
            # Redraw static elements
            ax.plot(merge_pos[0], merge_pos[1], 'ro', markersize=10)
            self._plot_sequencing_legs(ax)
            
            current_time = time_range[0] + frame * time_step
            
            # Plot aircraft at current time
            for aircraft_id, trajectory in trajectory_data.items():
                # Find closest time point
                closest_point = None
                min_time_diff = float('inf')
                
                for point in trajectory:
                    time_diff = abs(point['time'] - current_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_point = point
                
                if closest_point and min_time_diff < time_step:
                    pos = closest_point['position']
                    ax.plot(pos[0], pos[1], 'bo', markersize=8)
                    ax.text(pos[0] + 1000, pos[1] + 1000, aircraft_id, 
                           fontsize=8, ha='left')
                    
                    # Plot heading vector
                    heading = closest_point['heading']
                    heading_vector = np.array([
                        np.cos(heading), np.sin(heading)
                    ]) * 5000
                    
                    ax.arrow(pos[0], pos[1], heading_vector[0], heading_vector[1], 
                            head_width=1000, head_length=1500, fc='blue', ec='blue',
                            alpha=0.7)
            
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_title(f'ATM Simulation - Time: {current_time:.0f}s')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Set consistent axis limits
            ax.set_xlim(-70000, 70000)
            ax.set_ylim(-70000, 70000)
        
        # Calculate number of frames
        num_frames = int((time_range[1] - time_range[0]) / time_step)
        
        anim = FuncAnimation(fig, animate, frames=num_frames, 
                           interval=interval, blit=False)
        
        return anim
    
    def save_plots(self, results: Dict, output_dir: str = "plots/") -> None:
        """
        Save all visualization plots.
        
        Args:
            results: Simulation results
            output_dir: Output directory for plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot trajectory history
        if 'trajectory_data' in results:
            fig = self.plot_trajectory_history(results['trajectory_data'])
            fig.savefig(f"{output_dir}/trajectory_history.png", 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        
        # Plot separation events
        if 'events_log' in results:
            fig = self.plot_separation_events(results['events_log'])
            fig.savefig(f"{output_dir}/separation_events.png", 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        
        # Plot system performance
        if 'statistics_data' in results:
            fig = self.plot_system_performance(results['statistics_data'])
            fig.savefig(f"{output_dir}/system_performance.png", 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Plots saved to {output_dir}")


def create_summary_report(results: Dict) -> str:
    """
    Create a text summary report of simulation results.
    
    Args:
        results: Simulation results
        
    Returns:
        Formatted summary report
    """
    report = []
    report.append("="*60)
    report.append("AIR TRAFFIC MANAGEMENT SIMULATION REPORT")
    report.append("="*60)
    report.append("")
    
    # Basic metrics
    metrics = results.get('metrics', {})
    report.append("BASIC METRICS:")
    report.append(f"  Total aircraft processed: {metrics.get('total_aircraft_processed', 0)}")
    report.append(f"  Average flight time: {metrics.get('average_flight_time', 0):.1f} seconds")
    report.append(f"  Separation violations: {metrics.get('separation_violations', 0)}")
    report.append(f"  Emergency events: {metrics.get('emergency_events', 0)}")
    report.append(f"  Final aircraft count: {results.get('final_aircraft_count', 0)}")
    report.append(f"  Simulation time: {results.get('simulation_time', 0):.1f} seconds")
    report.append("")
    
    # Separation report
    sep_report = results.get('separation_report', {})
    if sep_report:
        report.append("SEPARATION MONITORING:")
        report.append(f"  Current events: {sep_report.get('current_events', 0)}")
        report.append(f"  Predicted conflicts: {sep_report.get('predicted_conflicts', 0)}")
        
        events_by_level = sep_report.get('events_by_level', {})
        for level, count in events_by_level.items():
            if count > 0:
                report.append(f"  {level.title()} events: {count}")
        report.append("")
    
    # Performance summary
    computation_time = results.get('computation_time', 0)
    if computation_time > 0:
        report.append("PERFORMANCE:")
        report.append(f"  Computation time: {computation_time:.2f} seconds")
        
        sim_time = results.get('simulation_time', 0)
        if sim_time > 0:
            ratio = sim_time / computation_time
            report.append(f"  Real-time factor: {ratio:.1f}x")
        report.append("")
    
    # Event summary
    events_log = results.get('events_log', [])
    if events_log:
        event_types = {}
        for event in events_log:
            event_type = event.get('type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        report.append("EVENTS SUMMARY:")
        for event_type, count in event_types.items():
            report.append(f"  {event_type.replace('_', ' ').title()}: {count}")
        report.append("")
    
    report.append("="*60)
    
    return "\n".join(report)