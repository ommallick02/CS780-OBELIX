"""
Heuristic agent for OBELIX with wall avoidance and box pushing.
Uses sensor patterns to distinguish box (small, few sensors) from walls (large, many sensors).
"""

import numpy as np
from typing import Sequence, Tuple, Optional

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

# Sensor layout indices in obs[0:16] (8 sensors × 2 ranges)
# Near: indices 0,2,4,6,8,10,12,14 (even)
# Far:  indices 1,3,5,7,9,11,13,15 (odd)
# Order: Left-90, Left-90, Left-45, Left-22, Right-22, Right-45, Right-90, Right-90

SENSOR_ORDER = [
    "L90_far", "L90_near",      # indices 0,1  - far left
    "L45_far", "L45_near",      # indices 2,3  - left
    "L22_far", "L22_near",      # indices 4,5  - front-left
    "R22_far", "R22_near",      # indices 6,7  - front-right
    "R45_far", "R45_near",      # indices 8,9  - right
    "R90_far", "R90_near",      # indices 10,11 - far right
    "R90b_far", "R90b_near",    # indices 12,13 - far right (back)
    "L90b_far", "L90b_near",    # indices 14,15 - far left (back)
]

# IR sensor is obs[16], stuck flag is obs[17]


def count_active_sensors(obs: np.ndarray) -> int:
    """Count how many sonar sensors are active (box or wall detected)."""
    return int(np.sum(obs[0:16]))


def get_front_sensors(obs: np.ndarray) -> Tuple[int, int]:
    """
    Get front sensor readings.
    Returns: (near_count, far_count) for front sector (L22, R22, R45, L45 center area)
    """
    # Front sensors: L22 (4,5), R22 (6,7), plus central R45/L45 partially
    front_indices = [4, 5, 6, 7]  # L22 and R22 (the most forward-pointing)
    near = sum(obs[i] for i in front_indices if i % 2 == 1)  # odd = near
    far = sum(obs[i] for i in front_indices if i % 2 == 0)  # even = far
    return near, far


def get_left_sensors(obs: np.ndarray) -> int:
    """Sum of left side sensors (L90, L45, L22)."""
    left_indices = [0, 1, 2, 3, 4, 5]
    return sum(obs[i] for i in left_indices)


def get_right_sensors(obs: np.ndarray) -> int:
    """Sum of right side sensors (R22, R45, R90, R90b)."""
    right_indices = [6, 7, 8, 9, 10, 11, 12, 13]
    return sum(obs[i] for i in right_indices)


def estimate_box_direction(obs: np.ndarray) -> Optional[str]:
    """
    Estimate where the box is based on sensor pattern.
    Returns: 'left', 'right', 'front', 'back', or None if not detected.
    """
    left = get_left_sensors(obs)
    right = get_right_sensors(obs)
    front_near, front_far = get_front_sensors(obs)
    front = front_near + front_far
    
    # Back sensors (14, 15 are L90b - back left)
    back = obs[14] + obs[15] + obs[12] + obs[13]
    
    if left == 0 and right == 0 and front == 0 and back == 0:
        return None  # No detection
    
    # Find strongest direction
    scores = {
        'left': left,
        'right': right,
        'front': front * 2,  # Weight front more (it's where we can go)
        'back': back * 0.5   # Weight back less (harder to reach)
    }
    
    return max(scores, key=scores.get)


def is_likely_wall(obs: np.ndarray) -> bool:
    """
    Heuristic: walls are big and light up many sensors.
    Box is small and lights up 1-2 sensors.
    """
    total_sensors = count_active_sensors(obs)
    # Wall heuristic: 4+ sensors active suggests large object (wall)
    # Box heuristic: 1-2 sensors active suggests small object (box)
    return total_sensors >= 4


def is_box_in_front(obs: np.ndarray) -> bool:
    """Check if box (not wall) is directly ahead."""
    # IR sensor indicates direct contact/very close
    if obs[16] > 0.5:
        return True
    
    front_near, front_far = get_front_sensors(obs)
    total_active = count_active_sensors(obs)
    
    # Box pattern: front sensors active, but not too many total sensors
    if front_near > 0 and total_active <= 3:
        return True
    
    # Strong front detection with limited side detection = box ahead
    if front_near + front_far >= 2 and total_active <= 4:
        return True
    
    return False


def is_stuck(obs: np.ndarray) -> bool:
    """Check if stuck against wall/boundary."""
    return obs[17] > 0.5


def select_rotation_towards(direction: Optional[str], rng: np.random.Generator) -> str:
    """Select rotation action to face the target direction."""
    if direction == 'left':
        return "L22"  # Small left turn
    elif direction == 'right':
        return "R22"  # Small right turn
    elif direction == 'back':
        # Turn around - big rotation
        return rng.choice(["L45", "R45"])
    else:
        # No detection or front - small search rotation
        return rng.choice(["L22", "R22"])


# Global state
_attached = False
_last_action = None
_search_direction = None  # Persist search direction


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Heuristic policy:
    1. If not attached: rotate to find box, avoid walls, move forward when box ahead
    2. If attached: push forward while avoiding walls
    """
    global _attached, _last_action, _search_direction
    
    # Check attachment via IR sensor or stuck flag with box contact
    # (We track this heuristically - no direct attachment signal in obs)
    # Actually, we can infer: if we were moving forward and IR was on, now we're "attached"
    
    ir_active = obs[16] > 0.5
    stuck = is_stuck(obs)
    
    # Update attachment state
    if ir_active and _last_action == "FW":
        _attached = True
    # If we've been pushing but lost contact and not stuck, might be done
    # (Episode ends on success, so this is just for safety)
    
    # === PUSH MODE: Box is attached ===
    if _attached:
        # Priority: avoid walls while pushing
        if is_likely_wall(obs) or stuck:
            # Wall ahead! Try to rotate away while still pushing
            # Prefer the direction with fewer sensors active
            left_count = get_left_sensors(obs)
            right_count = get_right_sensors(obs)
            
            if left_count < right_count:
                action = "L22"  # Steer left
            else:
                action = "R22"  # Steer right
            
            _last_action = action
            return action
        
        # Clear path - push forward
        action = "FW"
        _last_action = action
        return action
    
    # === FIND MODE: Not attached yet ===
    
    # If stuck, back up by rotating
    if stuck:
        _search_direction = None
        action = rng.choice(["L45", "R45"])
        _last_action = action
        return action
    
    # If wall detected (many sensors), rotate away
    if is_likely_wall(obs):
        # Rotate away from wall
        left_count = get_left_sensors(obs)
        right_count = get_right_sensors(obs)
        
        # Turn towards side with fewer detections (away from wall)
        if left_count > right_count:
            action = "R22"  # Wall on left, turn right
        elif right_count > left_count:
            action = "L22"  # Wall on right, turn left
        else:
            action = rng.choice(["L45", "R45"])  # Big turn to escape
        
        _search_direction = None  # Reset search
        _last_action = action
        return action
    
    # Box detection: small number of sensors, front preferred
    if is_box_in_front(obs):
        # Box is ahead and it's not a wall pattern - move forward!
        action = "FW"
        _last_action = action
        return action
    
    # Box detected but not in front - rotate towards it
    direction = estimate_box_direction(obs)
    
    if direction is not None and direction != 'front':
        action = select_rotation_towards(direction, rng)
        _search_direction = direction
        _last_action = action
        return action
    
    # No detection - search pattern
    # Continue in consistent search direction or pick new one
    if _search_direction is None:
        _search_direction = rng.choice(['left', 'right'])
    
    if _search_direction == 'left':
        action = "L22"
    else:
        action = "R22"
    
    _last_action = action
    return action