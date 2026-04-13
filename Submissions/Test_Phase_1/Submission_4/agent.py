"""
Braitenberg Vehicle implementation for OBELIX.
Direct sensor-motor coupling with wall avoidance and box pushing.
"""

import os
import numpy as np
from typing import Sequence, Tuple

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

# Sensor layout indices in obs[0:16] (8 sensors x 2 ranges: near=odd, far=even)
# Order: L90-far, L90-near, L45-far, L45-near, L22-far, L22-near, R22-far, R22-near, 
#        R45-far, R45-near, R90-far, R90-near, R90b-far, R90b-near, L90b-far, L90b-near

# Group sensors by direction and range
LEFT_SENSORS = [0, 1, 2, 3, 4, 5, 14, 15]  # L90, L45, L22, L90b (back)
RIGHT_SENSORS = [6, 7, 8, 9, 10, 11, 12, 13]  # R22, R45, R90, R90b (back)
FRONT_NEAR = [5, 7]  # L22-near, R22-near (most forward)
FRONT_FAR = [4, 6]  # L22-far, R22-far
SIDE_NEAR = [1, 3, 9, 11]  # L90-near, L45-near, R45-near, R90-near

IR_SENSOR = 16  # Infrared (contact)
STUCK_FLAG = 17


class BraitenbergAgent:
    """
    Braitenberg-style reactive agent with wall/box discrimination.
    """
    
    def __init__(self):
        # Track if we're attached to box (for push behavior)
        self.attached = False
        self.consecutive_forward = 0
    
    def count_active_sensors(self, obs: np.ndarray, sensor_list: list) -> int:
        """Count active sensors in a group."""
        return int(sum(obs[i] for i in sensor_list))
    
    def is_likely_wall(self, obs: np.ndarray) -> bool:
        """
        Wall detection: many sensors active = large object = wall.
        Box detection: few sensors active = small object = box.
        """
        total_active = int(sum(obs[0:16]))
        # Wall: 4+ sensors, Box: 1-3 sensors
        return total_active >= 4
    
    def is_box_in_front(self, obs: np.ndarray) -> bool:
        """Check if box (not wall) is directly ahead."""
        front_near = self.count_active_sensors(obs, [5, 7])  # L22-near, R22-near
        total_active = int(sum(obs[0:16]))
        
        # Box pattern: front sensors active, but total sensors low
        if front_near > 0 and total_active <= 3:
            return True
        
        # IR sensor indicates direct contact
        if obs[IR_SENSOR] > 0.5:
            return True
        
        return False
    
    def get_motor_activation(self, obs: np.ndarray) -> Tuple[float, float]:
        """
        Braitenberg-style sensor-motor coupling.
        Returns (left_motor, right_motor) activation levels.
        """
        # Calculate weighted sensor activations
        left_activation = (
            2.0 * obs[5] +   # L22-near (front-left) - highest weight
            1.5 * obs[3] +   # L45-near
            1.0 * obs[1] +   # L90-near
            0.5 * obs[4] +   # L22-far
            0.3 * obs[2] +   # L45-far
            0.2 * obs[0]     # L90-far
        )
        
        right_activation = (
            2.0 * obs[7] +   # R22-near (front-right) - highest weight
            1.5 * obs[9] +   # R45-near
            1.0 * obs[11] +  # R90-near
            0.5 * obs[6] +   # R22-far
            0.3 * obs[8] +   # R45-far
            0.2 * obs[10]    # R90-far
        )
        
        # Back sensors (for escaping)
        back_left = obs[14] + obs[15]  # L90b
        back_right = obs[12] + obs[13]  # R90b
        
        return left_activation, right_activation, back_left, back_right
    
    def select_action(self, obs: np.ndarray, rng: np.random.Generator) -> str:
        """
        Braitenberg sensor-motor coupling with wall avoidance.
        """
        stuck = obs[STUCK_FLAG] > 0.5
        
        # Check attachment via IR sensor
        if obs[IR_SENSOR] > 0.5:
            self.attached = True
        
        # === PUSH MODE: Box attached ===
        if self.attached:
            # Push forward while avoiding walls
            if self.is_likely_wall(obs) or stuck:
                # Wall ahead - steer toward side with fewer sensors
                left_count = self.count_active_sensors(obs, LEFT_SENSORS)
                right_count = self.count_active_sensors(obs, RIGHT_SENSORS)
                
                if left_count < right_count:
                    return "L22"  # Steer left (away from wall on right)
                else:
                    return "R22"  # Steer right (away from wall on left)
            
            # Clear path - push forward
            self.consecutive_forward += 1
            return "FW"
        
        # === FIND MODE: Not attached ===
        
        # If stuck, back away with rotation
        if stuck:
            self.consecutive_forward = 0
            # Rotate away from obstacle
            left_count = self.count_active_sensors(obs, LEFT_SENSORS)
            right_count = self.count_active_sensors(obs, RIGHT_SENSORS)
            
            if left_count > right_count:
                return "R45"  # Turn right (away from left obstacle)
            else:
                return "L45"  # Turn left (away from right obstacle)
        
        # Wall detection - large object, many sensors
        if self.is_likely_wall(obs):
            self.consecutive_forward = 0
            # Wall avoidance: turn away from wall
            left_count = self.count_active_sensors(obs, LEFT_SENSORS)
            right_count = self.count_active_sensors(obs, RIGHT_SENSORS)
            
            # Turn toward side with fewer detections (away from wall)
            if left_count > right_count:
                return "R22"  # Wall on left, turn right
            elif right_count > left_count:
                return "L22"  # Wall on right, turn left
            else:
                # Wall ahead, big turn
                return rng.choice(["L45", "R45"])
        
        # Box detection - small object, few sensors
        if self.is_box_in_front(obs):
            self.consecutive_forward += 1
            # Box ahead - move forward to attach
            return "FW"
        
        # Braitenberg sensor-motor coupling for search
        left_act, right_act, back_left, back_right = self.get_motor_activation(obs)
        
        # If no detection, use back sensors to escape dead ends
        if left_act == 0 and right_act == 0:
            if back_left > 0 or back_right > 0:
                # Something behind, turn around
                return rng.choice(["L45", "R45"])
            
            # No detection - spiral search pattern
            self.consecutive_forward += 1
            if self.consecutive_forward % 10 < 7:
                return "FW"  # Move forward most of the time
            else:
                return "L22"  # Slight turn occasionally
        
        # Braitenberg: turn toward higher activation (source seeking)
        if left_act > right_act:
            # Source on left - turn left (positive tropism)
            if left_act > 2.0:
                return "L45"  # Strong turn
            else:
                return "L22"  # Gentle turn
        elif right_act > left_act:
            # Source on right - turn right
            if right_act > 2.0:
                return "R45"  # Strong turn
            else:
                return "R22"  # Gentle turn
        else:
            # Balanced - move forward
            self.consecutive_forward += 1
            return "FW"


# Global agent
_AGENT = None


def _load_agent():
    """Load Braitenberg agent."""
    global _AGENT
    if _AGENT is None:
        _AGENT = BraitenbergAgent()
    return _AGENT


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Braitenberg policy function.
    """
    agent = _load_agent()
    return agent.select_action(obs, rng)