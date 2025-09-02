"""
CPDLC (Controller-Pilot Data Link Communications) interface simulation.

This module simulates the datalink communication between air traffic control
and aircraft for transmitting turn instructions and other control commands.
"""
from typing import List, Dict, Optional, Any
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import time
import json

try:
    from ..models.aircraft import Aircraft
    from ..controllers.lower_level import ControlInput
except ImportError:
    from models.aircraft import Aircraft
    from controllers.lower_level import ControlInput


class MessageType(Enum):
    """CPDLC message types."""
    TURN_INSTRUCTION = "TURN_INSTRUCTION"
    SPEED_INSTRUCTION = "SPEED_INSTRUCTION"
    ALTITUDE_INSTRUCTION = "ALTITUDE_INSTRUCTION"
    ROUTE_AMENDMENT = "ROUTE_AMENDMENT"
    CLEARANCE = "CLEARANCE"
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    EMERGENCY = "EMERGENCY"


class MessageStatus(Enum):
    """Message transmission status."""
    PENDING = "PENDING"
    TRANSMITTED = "TRANSMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


@dataclass
class CPDLCMessage:
    """CPDLC message structure."""
    message_id: str
    aircraft_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp_created: float
    timestamp_transmitted: Optional[float] = None
    timestamp_acknowledged: Optional[float] = None
    timestamp_executed: Optional[float] = None
    status: MessageStatus = MessageStatus.PENDING
    priority: int = 5  # 1 (highest) to 10 (lowest)
    retry_count: int = 0
    max_retries: int = 3
    
    def to_uplink_format(self) -> str:
        """Convert message to uplink format."""
        return json.dumps({
            'msg_id': self.message_id,
            'type': self.message_type.value,
            'content': self.content,
            'timestamp': self.timestamp_created,
            'priority': self.priority
        })
    
    def from_downlink_format(self, data: str) -> None:
        """Parse downlink message format."""
        try:
            parsed = json.loads(data)
            self.status = MessageStatus(parsed.get('status', 'ACKNOWLEDGED'))
            self.timestamp_acknowledged = parsed.get('timestamp', None)
        except (json.JSONDecodeError, ValueError):
            self.status = MessageStatus.FAILED


@dataclass
class DataLinkConfig:
    """Datalink configuration parameters."""
    transmission_delay: float = 2.0      # Average transmission delay (seconds)
    transmission_variance: float = 1.0   # Transmission delay variance
    message_loss_rate: float = 0.02      # 2% message loss rate
    acknowledgment_timeout: float = 30.0  # Timeout for acknowledgment
    execution_timeout: float = 60.0      # Timeout for execution
    max_queue_size: int = 100            # Maximum message queue size


class CPDLCInterface:
    """
    CPDLC interface for air traffic control datalink communications.
    
    This class simulates the datalink system used to transmit control
    instructions from ground to aircraft, including turn points, speed
    changes, and other tactical instructions.
    """
    
    def __init__(self, config: Optional[DataLinkConfig] = None):
        """
        Initialize CPDLC interface.
        
        Args:
            config: Datalink configuration parameters
        """
        self.config = config or DataLinkConfig()
        
        # Message queues
        self.outbound_queue: List[CPDLCMessage] = []
        self.pending_messages: Dict[str, CPDLCMessage] = {}
        self.message_history: List[CPDLCMessage] = []
        
        # Message ID counter
        self.message_counter = 0
        
        # Communication statistics
        self.statistics = {
            'messages_sent': 0,
            'messages_acknowledged': 0,
            'messages_executed': 0,
            'messages_failed': 0,
            'messages_rejected': 0,
            'average_response_time': 0.0,
            'total_response_time': 0.0
        }
        
        # Aircraft communication status
        self.aircraft_status: Dict[str, Dict] = {}
    
    def send_turn_instruction(self, aircraft_id: str, turn_point: np.ndarray,
                             heading: float, current_time: float,
                             priority: int = 5) -> str:
        """
        Send turn instruction to aircraft.
        
        Args:
            aircraft_id: Target aircraft ID
            turn_point: Turn point coordinates [x, y, z]
            heading: Target heading in radians
            current_time: Current simulation time
            priority: Message priority (1-10)
            
        Returns:
            Message ID
        """
        message_content = {
            'instruction_type': 'TURN_AT_POINT',
            'turn_point': {
                'x': float(turn_point[0]),
                'y': float(turn_point[1]),
                'z': float(turn_point[2])
            },
            'target_heading': float(np.degrees(heading)),
            'execution_time': current_time + 10.0,  # Execute in 10 seconds
            'clearance_text': f"TURN RIGHT/LEFT HEADING {int(np.degrees(heading)):03d} AT POINT"
        }
        
        return self._create_and_queue_message(
            aircraft_id, MessageType.TURN_INSTRUCTION, 
            message_content, current_time, priority
        )
    
    def send_speed_instruction(self, aircraft_id: str, target_speed: float,
                              current_time: float, priority: int = 5) -> str:
        """
        Send speed instruction to aircraft.
        
        Args:
            aircraft_id: Target aircraft ID
            target_speed: Target speed in m/s
            current_time: Current simulation time
            priority: Message priority
            
        Returns:
            Message ID
        """
        speed_knots = int(target_speed * 1.944)  # Convert m/s to knots
        
        message_content = {
            'instruction_type': 'SPEED_CHANGE',
            'target_speed_ms': float(target_speed),
            'target_speed_knots': speed_knots,
            'execution_time': current_time + 5.0,
            'clearance_text': f"REDUCE/INCREASE SPEED TO {speed_knots} KNOTS"
        }
        
        return self._create_and_queue_message(
            aircraft_id, MessageType.SPEED_INSTRUCTION,
            message_content, current_time, priority
        )
    
    def send_altitude_instruction(self, aircraft_id: str, target_altitude: float,
                                 current_time: float, priority: int = 5) -> str:
        """
        Send altitude instruction to aircraft.
        
        Args:
            aircraft_id: Target aircraft ID
            target_altitude: Target altitude in meters
            current_time: Current simulation time
            priority: Message priority
            
        Returns:
            Message ID
        """
        altitude_feet = int(target_altitude * 3.281)  # Convert meters to feet
        
        message_content = {
            'instruction_type': 'ALTITUDE_CHANGE',
            'target_altitude_m': float(target_altitude),
            'target_altitude_ft': altitude_feet,
            'execution_time': current_time + 15.0,
            'clearance_text': f"CLIMB/DESCEND TO FLIGHT LEVEL {altitude_feet // 100:03d}"
        }
        
        return self._create_and_queue_message(
            aircraft_id, MessageType.ALTITUDE_INSTRUCTION,
            message_content, current_time, priority
        )
    
    def send_route_amendment(self, aircraft_id: str, waypoints: List[np.ndarray],
                           current_time: float, priority: int = 5) -> str:
        """
        Send route amendment to aircraft.
        
        Args:
            aircraft_id: Target aircraft ID
            waypoints: List of waypoint coordinates
            current_time: Current simulation time
            priority: Message priority
            
        Returns:
            Message ID
        """
        waypoint_list = []
        for i, wp in enumerate(waypoints):
            waypoint_list.append({
                'name': f"WP{i+1:02d}",
                'x': float(wp[0]),
                'y': float(wp[1]),
                'z': float(wp[2])
            })
        
        message_content = {
            'instruction_type': 'ROUTE_AMENDMENT',
            'waypoints': waypoint_list,
            'execution_time': current_time + 20.0,
            'clearance_text': f"PROCEED DIRECT TO WAYPOINTS AS AMENDED"
        }
        
        return self._create_and_queue_message(
            aircraft_id, MessageType.ROUTE_AMENDMENT,
            message_content, current_time, priority
        )
    
    def send_emergency_instruction(self, aircraft_id: str, instruction: str,
                                 current_time: float) -> str:
        """
        Send emergency instruction with highest priority.
        
        Args:
            aircraft_id: Target aircraft ID
            instruction: Emergency instruction text
            current_time: Current simulation time
            
        Returns:
            Message ID
        """
        message_content = {
            'instruction_type': 'EMERGENCY',
            'emergency_instruction': instruction,
            'execution_time': current_time + 1.0,  # Immediate execution
            'clearance_text': f"EMERGENCY: {instruction}"
        }
        
        return self._create_and_queue_message(
            aircraft_id, MessageType.EMERGENCY,
            message_content, current_time, priority=1
        )
    
    def _create_and_queue_message(self, aircraft_id: str, message_type: MessageType,
                                content: Dict, current_time: float,
                                priority: int) -> str:
        """Create and queue a new message."""
        self.message_counter += 1
        message_id = f"MSG{self.message_counter:06d}"
        
        message = CPDLCMessage(
            message_id=message_id,
            aircraft_id=aircraft_id,
            message_type=message_type,
            content=content,
            timestamp_created=current_time,
            priority=priority
        )
        
        # Add to outbound queue (sorted by priority)
        self.outbound_queue.append(message)
        self.outbound_queue.sort(key=lambda m: (m.priority, m.timestamp_created))
        
        # Limit queue size
        if len(self.outbound_queue) > self.config.max_queue_size:
            dropped_message = self.outbound_queue.pop()
            self.statistics['messages_failed'] += 1
        
        return message_id
    
    def process_communications(self, current_time: float) -> Dict[str, List[CPDLCMessage]]:
        """
        Process communication queue and simulate message transmission.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dictionary of transmitted messages by aircraft ID
        """
        transmitted_messages = {}
        
        # Process outbound queue
        messages_to_transmit = []
        for message in self.outbound_queue:
            if self._should_transmit_message(message, current_time):
                messages_to_transmit.append(message)
        
        # Remove transmitted messages from queue
        for message in messages_to_transmit:
            self.outbound_queue.remove(message)
            
            # Simulate transmission
            if self._simulate_transmission(message, current_time):
                message.status = MessageStatus.TRANSMITTED
                message.timestamp_transmitted = current_time
                self.pending_messages[message.message_id] = message
                
                # Add to transmitted messages
                if message.aircraft_id not in transmitted_messages:
                    transmitted_messages[message.aircraft_id] = []
                transmitted_messages[message.aircraft_id].append(message)
                
                self.statistics['messages_sent'] += 1
            else:
                # Transmission failed
                message.status = MessageStatus.FAILED
                message.retry_count += 1
                
                if message.retry_count < message.max_retries:
                    # Retry later
                    self.outbound_queue.append(message)
                else:
                    # Max retries reached
                    self.statistics['messages_failed'] += 1
                    self.message_history.append(message)
        
        # Process acknowledgments and timeouts
        self._process_pending_messages(current_time)
        
        return transmitted_messages
    
    def _should_transmit_message(self, message: CPDLCMessage, current_time: float) -> bool:
        """Determine if a message should be transmitted now."""
        # Check if enough time has passed since creation (simulate processing delay)
        if current_time - message.timestamp_created < 1.0:
            return False
        
        # Emergency messages are transmitted immediately
        if message.message_type == MessageType.EMERGENCY:
            return True
        
        # Check aircraft communication status
        aircraft_status = self.aircraft_status.get(message.aircraft_id, {})
        last_transmission = aircraft_status.get('last_transmission_time', 0.0)
        
        # Minimum time between transmissions to same aircraft
        if current_time - last_transmission < 2.0:
            return False
        
        return True
    
    def _simulate_transmission(self, message: CPDLCMessage, current_time: float) -> bool:
        """Simulate message transmission with delays and failures."""
        # Simulate message loss
        if np.random.random() < self.config.message_loss_rate:
            return False
        
        # Update aircraft communication status
        if message.aircraft_id not in self.aircraft_status:
            self.aircraft_status[message.aircraft_id] = {}
        
        self.aircraft_status[message.aircraft_id]['last_transmission_time'] = current_time
        
        return True
    
    def _process_pending_messages(self, current_time: float) -> None:
        """Process pending messages for acknowledgments and timeouts."""
        completed_messages = []
        
        for message_id, message in self.pending_messages.items():
            # Simulate acknowledgment (simplified - assume all messages are acknowledged)
            if (message.status == MessageStatus.TRANSMITTED and 
                message.timestamp_acknowledged is None):
                
                # Random acknowledgment delay
                ack_delay = np.random.normal(5.0, 2.0)  # Average 5 seconds
                if current_time - message.timestamp_transmitted >= ack_delay:
                    message.status = MessageStatus.ACKNOWLEDGED
                    message.timestamp_acknowledged = current_time
                    self.statistics['messages_acknowledged'] += 1
            
            # Simulate execution
            elif (message.status == MessageStatus.ACKNOWLEDGED and
                  message.timestamp_executed is None):
                
                execution_time = message.content.get('execution_time', current_time)
                if current_time >= execution_time:
                    message.status = MessageStatus.EXECUTED
                    message.timestamp_executed = current_time
                    self.statistics['messages_executed'] += 1
                    
                    # Calculate response time
                    response_time = current_time - message.timestamp_created
                    self.statistics['total_response_time'] += response_time
                    
                    completed_messages.append(message_id)
            
            # Check for timeouts
            elif message.status == MessageStatus.TRANSMITTED:
                if (current_time - message.timestamp_transmitted > 
                    self.config.acknowledgment_timeout):
                    message.status = MessageStatus.FAILED
                    self.statistics['messages_failed'] += 1
                    completed_messages.append(message_id)
        
        # Move completed messages to history
        for message_id in completed_messages:
            message = self.pending_messages.pop(message_id)
            self.message_history.append(message)
        
        # Update average response time
        if self.statistics['messages_executed'] > 0:
            self.statistics['average_response_time'] = (
                self.statistics['total_response_time'] / 
                self.statistics['messages_executed']
            )
    
    def receive_downlink_message(self, aircraft_id: str, message_data: str,
                               current_time: float) -> None:
        """
        Process downlink message from aircraft.
        
        Args:
            aircraft_id: Source aircraft ID
            message_data: Message data
            current_time: Current time
        """
        try:
            parsed_data = json.loads(message_data)
            message_id = parsed_data.get('msg_id')
            
            if message_id in self.pending_messages:
                message = self.pending_messages[message_id]
                status = parsed_data.get('status', 'ACKNOWLEDGED')
                
                if status == 'ACKNOWLEDGED':
                    message.status = MessageStatus.ACKNOWLEDGED
                    message.timestamp_acknowledged = current_time
                elif status == 'REJECTED':
                    message.status = MessageStatus.REJECTED
                    self.statistics['messages_rejected'] += 1
                
        except (json.JSONDecodeError, KeyError):
            # Invalid downlink message
            pass
    
    def get_aircraft_messages(self, aircraft_id: str, 
                            status_filter: Optional[MessageStatus] = None) -> List[CPDLCMessage]:
        """
        Get messages for a specific aircraft.
        
        Args:
            aircraft_id: Aircraft ID
            status_filter: Optional status filter
            
        Returns:
            List of messages
        """
        messages = []
        
        # Check outbound queue
        messages.extend([m for m in self.outbound_queue if m.aircraft_id == aircraft_id])
        
        # Check pending messages
        messages.extend([m for m in self.pending_messages.values() 
                        if m.aircraft_id == aircraft_id])
        
        # Check history
        messages.extend([m for m in self.message_history 
                        if m.aircraft_id == aircraft_id])
        
        # Apply status filter
        if status_filter:
            messages = [m for m in messages if m.status == status_filter]
        
        return sorted(messages, key=lambda m: m.timestamp_created)
    
    def get_communication_statistics(self) -> Dict:
        """Get communication statistics."""
        return {
            'statistics': self.statistics.copy(),
            'queue_status': {
                'outbound_queue_size': len(self.outbound_queue),
                'pending_messages': len(self.pending_messages),
                'total_messages_history': len(self.message_history)
            },
            'aircraft_status': {
                aircraft_id: status.copy() 
                for aircraft_id, status in self.aircraft_status.items()
            }
        }
    
    def clear_old_messages(self, current_time: float, max_age: float = 3600.0) -> None:
        """
        Clear old messages from history.
        
        Args:
            current_time: Current time
            max_age: Maximum age of messages to keep (seconds)
        """
        cutoff_time = current_time - max_age
        
        self.message_history = [
            m for m in self.message_history 
            if m.timestamp_created > cutoff_time
        ]