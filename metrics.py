# metrics.py
# Utility functions for calculating exercise performance metrics.

import numpy as np
import torch

# Joint indices in the 19-joint skeleton (0-indexed)
# Based on the skeleton structure from model.py bone_list
JOINT_INDICES = {
    # These are indices in the 19-joint array (after selection from 25-joint skeleton)
    'hip_center': 8,      # Joint 8 connects to left/right hip
    'left_hip': 9,        # Joint 9
    'right_hip': 12,      # Joint 12
    'left_knee': 10,      # Joint 10
    'right_knee': 13,     # Joint 13
    'left_ankle': 11,     # Joint 11
    'right_ankle': 14,    # Joint 14
    'left_shoulder': 5,   # Joint 5 (from bone list [1, 5])
    'right_shoulder': 2,  # Joint 2 (from bone list [1, 2])
}

def pose_to_joints(pose_array):
    """
    Convert flattened pose array back to [num_frames, num_joints, 3].

    IMPORTANT: The flattening in data_loader.py is (coords, joints) major:
        all_poses[frames, 3, 19].reshape(frames, 57)
    i.e., coordinates vary slowest, joints vary fastest.

    That means to invert we must first reshape to (frames, 3, 19) and then
    transpose to (frames, 19, 3).

    Args:
        pose_array: numpy array or tensor of shape [57, num_frames]
                    or [num_frames, 57], where 57 = 19 joints * 3 coords.

    Returns:
        numpy array of shape [num_frames, 19, 3].
    """
    if isinstance(pose_array, torch.Tensor):
        pose_array = pose_array.cpu().numpy()

    pose_array = np.asarray(pose_array)

    if pose_array.ndim != 2 or 57 not in pose_array.shape:
        raise ValueError(f"Unexpected pose shape: {pose_array.shape}")

    if pose_array.shape[0] == 57:
        # [57, T] -> [T, 3, 19] -> [T, 19, 3]
        num_frames = pose_array.shape[1]
        tmp = pose_array.T.reshape(num_frames, 3, 19)
        pose_reshaped = np.transpose(tmp, (0, 2, 1))
    else:
        # [T, 57] -> [T, 3, 19] -> [T, 19, 3]
        num_frames = pose_array.shape[0]
        tmp = pose_array.reshape(num_frames, 3, 19)
        pose_reshaped = np.transpose(tmp, (0, 2, 1))

    return pose_reshaped

def get_joint_position(pose_joints, joint_name, frame_idx=None):
    """
    Get position of a joint.
    
    Args:
        pose_joints: [num_frames, 19, 3] array
        joint_name: string key from JOINT_INDICES
        frame_idx: frame index (None for all frames)
    
    Returns:
        [3] array if frame_idx specified, else [num_frames, 3] array
    """
    joint_idx = JOINT_INDICES[joint_name]
    if frame_idx is not None:
        return pose_joints[frame_idx, joint_idx]
    return pose_joints[:, joint_idx]

def calculate_angle(p1, p2, p3):
    """
    Calculate angle at p2 formed by points p1-p2-p3.
    
    Args:
        p1, p2, p3: [3] arrays representing 3D points
    
    Returns:
        angle in radians
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    # Calculate angle
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return angle

def calculate_squat_speed(pose_joints, joint_name='hip_center'):
    """
    Calculate squat speed metrics from hip joint movement.
    
    Args:
        pose_joints: [num_frames, 19, 3] array
        joint_name: joint to track (default: 'hip_center')
    
    Returns:
        dict with speed metrics
    """
    hip_positions = get_joint_position(pose_joints, joint_name)  # [num_frames, 3]
    num_frames = hip_positions.shape[0]
    
    # Calculate velocity (units/frame) for each frame
    velocities = np.zeros(num_frames)
    for i in range(1, num_frames):
        # Euclidean distance between consecutive frames
        velocities[i] = np.linalg.norm(hip_positions[i] - hip_positions[i-1])
    
    # Find bottom of squat (maximum Y position, assuming Y is vertical)
    # In typical 3D pose, Y might be up or down depending on coordinate system
    # We'll use the frame with maximum Y (lowest point in typical systems)
    y_positions = hip_positions[:, 1]  # Y coordinate
    bottom_frame = np.argmax(y_positions)  # Frame with highest Y (lowest point)
    
    # Split into descent (0 to bottom) and ascent (bottom to end)
    descent_velocities = velocities[1:bottom_frame+1] if bottom_frame > 0 else velocities[1:2]
    ascent_velocities = velocities[bottom_frame+1:] if bottom_frame < num_frames - 1 else velocities[-1:]
    
    # Calculate metrics
    avg_speed_descent = np.mean(descent_velocities) if len(descent_velocities) > 0 else 0.0
    avg_speed_ascent = np.mean(ascent_velocities) if len(ascent_velocities) > 0 else 0.0
    avg_speed_combined = np.mean(velocities[1:]) if num_frames > 1 else 0.0
    
    # Peak speed
    peak_speed = np.max(velocities)
    peak_speed_frame = np.argmax(velocities)
    peak_speed_position = hip_positions[peak_speed_frame].tolist()
    
    return {
        "average_speed_descent": float(avg_speed_descent),
        "average_speed_ascent": float(avg_speed_ascent),
        "average_speed_combined": float(avg_speed_combined),
        "peak_speed": {
            "value": float(peak_speed),
            "frame": int(peak_speed_frame),
            "position": peak_speed_position
        }
    }

def calculate_squat_depth(pose_joints):
    """
    Calculate squat depth (minimum hip height).
    
    Args:
        pose_joints: [num_frames, 19, 3] array
    
    Returns:
        dict with depth metrics
    """
    hip_positions = get_joint_position(pose_joints, 'hip_center')  # [num_frames, 3]
    y_positions = hip_positions[:, 1]  # Y coordinate
    
    # Initial hip height
    initial_height = float(y_positions[0])
    
    # Minimum hip height (deepest point)
    min_height = float(np.min(y_positions))
    min_height_frame = int(np.argmin(y_positions))
    
    # Depth (distance from initial to minimum)
    depth = float(initial_height - min_height)
    
    return {
        "initial_height": initial_height,
        "minimum_height": min_height,
        "depth": depth,
        "depth_frame": min_height_frame
    }

def calculate_knee_angles(pose_joints):
    """
    Calculate knee angles throughout the squat.
    
    Args:
        pose_joints: [num_frames, 19, 3] array
    
    Returns:
        dict with knee angle metrics
    """
    num_frames = pose_joints.shape[0]
    left_knee_angles = []
    right_knee_angles = []
    
    for frame_idx in range(num_frames):
        # Left knee angle: left_hip - left_knee - left_ankle
        left_hip = get_joint_position(pose_joints, 'left_hip', frame_idx)
        left_knee = get_joint_position(pose_joints, 'left_knee', frame_idx)
        left_ankle = get_joint_position(pose_joints, 'left_ankle', frame_idx)
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)
        left_knee_angles.append(left_angle)
        
        # Right knee angle: right_hip - right_knee - right_ankle
        right_hip = get_joint_position(pose_joints, 'right_hip', frame_idx)
        right_knee = get_joint_position(pose_joints, 'right_knee', frame_idx)
        right_ankle = get_joint_position(pose_joints, 'right_ankle', frame_idx)
        right_angle = calculate_angle(right_hip, right_knee, right_ankle)
        right_knee_angles.append(right_angle)
    
    left_knee_angles = np.array(left_knee_angles)
    right_knee_angles = np.array(right_knee_angles)
    
    return {
        "left_knee": {
            "angles_rad": left_knee_angles.tolist(),
            "angles_deg": np.degrees(left_knee_angles).tolist(),
            "min_angle_rad": float(np.min(left_knee_angles)),
            "max_angle_rad": float(np.max(left_knee_angles)),
            "min_angle_deg": float(np.degrees(np.min(left_knee_angles))),
            "max_angle_deg": float(np.degrees(np.max(left_knee_angles))),
            "range_rad": float(np.max(left_knee_angles) - np.min(left_knee_angles)),
            "range_deg": float(np.degrees(np.max(left_knee_angles) - np.min(left_knee_angles)))
        },
        "right_knee": {
            "angles_rad": right_knee_angles.tolist(),
            "angles_deg": np.degrees(right_knee_angles).tolist(),
            "min_angle_rad": float(np.min(right_knee_angles)),
            "max_angle_rad": float(np.max(right_knee_angles)),
            "min_angle_deg": float(np.degrees(np.min(right_knee_angles))),
            "max_angle_deg": float(np.degrees(np.max(right_knee_angles))),
            "range_rad": float(np.max(right_knee_angles) - np.min(right_knee_angles)),
            "range_deg": float(np.degrees(np.max(right_knee_angles) - np.min(right_knee_angles)))
        }
    }

def calculate_hip_angles(pose_joints):
    """
    Calculate hip angles throughout the squat.
    Hip angle is calculated using: shoulder - hip - knee
    
    Args:
        pose_joints: [num_frames, 19, 3] array
    
    Returns:
        dict with hip angle metrics
    """
    num_frames = pose_joints.shape[0]
    left_hip_angles = []
    right_hip_angles = []
    
    for frame_idx in range(num_frames):
        # Left hip angle: left_shoulder - left_hip - left_knee
        left_shoulder = get_joint_position(pose_joints, 'left_shoulder', frame_idx)
        left_hip = get_joint_position(pose_joints, 'left_hip', frame_idx)
        left_knee = get_joint_position(pose_joints, 'left_knee', frame_idx)
        left_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        left_hip_angles.append(left_angle)
        
        # Right hip angle: right_shoulder - right_hip - right_knee
        right_shoulder = get_joint_position(pose_joints, 'right_shoulder', frame_idx)
        right_hip = get_joint_position(pose_joints, 'right_hip', frame_idx)
        right_knee = get_joint_position(pose_joints, 'right_knee', frame_idx)
        right_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        right_hip_angles.append(right_angle)
    
    left_hip_angles = np.array(left_hip_angles)
    right_hip_angles = np.array(right_hip_angles)
    
    return {
        "left_hip": {
            "angles_rad": left_hip_angles.tolist(),
            "angles_deg": np.degrees(left_hip_angles).tolist(),
            "min_angle_rad": float(np.min(left_hip_angles)),
            "max_angle_rad": float(np.max(left_hip_angles)),
            "min_angle_deg": float(np.degrees(np.min(left_hip_angles))),
            "max_angle_deg": float(np.degrees(np.max(left_hip_angles))),
            "range_rad": float(np.max(left_hip_angles) - np.min(left_hip_angles)),
            "range_deg": float(np.degrees(np.max(left_hip_angles) - np.min(left_hip_angles)))
        },
        "right_hip": {
            "angles_rad": right_hip_angles.tolist(),
            "angles_deg": np.degrees(right_hip_angles).tolist(),
            "min_angle_rad": float(np.min(right_hip_angles)),
            "max_angle_rad": float(np.max(right_hip_angles)),
            "min_angle_deg": float(np.degrees(np.min(right_hip_angles))),
            "max_angle_deg": float(np.degrees(np.max(right_hip_angles))),
            "range_rad": float(np.max(right_hip_angles) - np.min(right_hip_angles)),
            "range_deg": float(np.degrees(np.max(right_hip_angles) - np.min(right_hip_angles)))
        }
    }

def calculate_balance_metrics(pose_joints):
    """
    Calculate balance and stability metrics.
    
    Args:
        pose_joints: [num_frames, 19, 3] array
    
    Returns:
        dict with balance metrics
    """
    left_ankle_positions = get_joint_position(pose_joints, 'left_ankle')  # [num_frames, 3]
    right_ankle_positions = get_joint_position(pose_joints, 'right_ankle')  # [num_frames, 3]
    hip_positions = get_joint_position(pose_joints, 'hip_center')  # [num_frames, 3]
    
    # Calculate center of mass (approximate as hip position)
    com_x = hip_positions[:, 0]  # X coordinate
    com_z = hip_positions[:, 2]  # Z coordinate
    
    # Base of support (line between ankles)
    base_center_x = (left_ankle_positions[:, 0] + right_ankle_positions[:, 0]) / 2
    base_center_z = (left_ankle_positions[:, 2] + right_ankle_positions[:, 2]) / 2
    
    # Distance from COM to base center (lateral stability)
    lateral_displacement = np.sqrt((com_x - base_center_x)**2 + (com_z - base_center_z)**2)
    
    # Base of support width
    base_width = np.linalg.norm(left_ankle_positions - right_ankle_positions, axis=1)
    
    return {
        "lateral_displacement": {
            "values": lateral_displacement.tolist(),
            "mean": float(np.mean(lateral_displacement)),
            "std": float(np.std(lateral_displacement)),
            "max": float(np.max(lateral_displacement))
        },
        "base_of_support_width": {
            "values": base_width.tolist(),
            "mean": float(np.mean(base_width)),
            "std": float(np.std(base_width)),
            "min": float(np.min(base_width)),
            "max": float(np.max(base_width))
        }
    }

def calculate_rep_symmetry(pose_joints):
    """
    Calculate symmetry metrics between left and right sides.
    
    Args:
        pose_joints: [num_frames, 19, 3] array
    
    Returns:
        dict with symmetry metrics
    """
    # Get joint positions
    left_knee_positions = get_joint_position(pose_joints, 'left_knee')
    right_knee_positions = get_joint_position(pose_joints, 'right_knee')
    left_hip_positions = get_joint_position(pose_joints, 'left_hip')
    right_hip_positions = get_joint_position(pose_joints, 'right_hip')
    left_ankle_positions = get_joint_position(pose_joints, 'left_ankle')
    right_ankle_positions = get_joint_position(pose_joints, 'right_ankle')
    
    # Calculate knee angles for symmetry comparison
    knee_angles = calculate_knee_angles(pose_joints)
    left_knee_angles = np.array(knee_angles['left_knee']['angles_rad'])
    right_knee_angles = np.array(knee_angles['right_knee']['angles_rad'])
    
    # Calculate hip angles for symmetry comparison
    hip_angles = calculate_hip_angles(pose_joints)
    left_hip_angles = np.array(hip_angles['left_hip']['angles_rad'])
    right_hip_angles = np.array(hip_angles['right_hip']['angles_rad'])
    
    # Calculate differences
    knee_angle_diff = np.abs(left_knee_angles - right_knee_angles)
    hip_angle_diff = np.abs(left_hip_angles - right_hip_angles)
    
    # Vertical position differences (Y coordinate)
    knee_height_diff = np.abs(left_knee_positions[:, 1] - right_knee_positions[:, 1])
    hip_height_diff = np.abs(left_hip_positions[:, 1] - right_hip_positions[:, 1])
    ankle_height_diff = np.abs(left_ankle_positions[:, 1] - right_ankle_positions[:, 1])
    
    return {
        "knee_angle_symmetry": {
            "differences_rad": knee_angle_diff.tolist(),
            "mean_difference_rad": float(np.mean(knee_angle_diff)),
            "mean_difference_deg": float(np.degrees(np.mean(knee_angle_diff))),
            "max_difference_rad": float(np.max(knee_angle_diff)),
            "max_difference_deg": float(np.degrees(np.max(knee_angle_diff)))
        },
        "hip_angle_symmetry": {
            "differences_rad": hip_angle_diff.tolist(),
            "mean_difference_rad": float(np.mean(hip_angle_diff)),
            "mean_difference_deg": float(np.degrees(np.mean(hip_angle_diff))),
            "max_difference_rad": float(np.max(hip_angle_diff)),
            "max_difference_deg": float(np.degrees(np.max(hip_angle_diff)))
        },
        "vertical_symmetry": {
            "knee_height_difference": {
                "values": knee_height_diff.tolist(),
                "mean": float(np.mean(knee_height_diff)),
                "max": float(np.max(knee_height_diff))
            },
            "hip_height_difference": {
                "values": hip_height_diff.tolist(),
                "mean": float(np.mean(hip_height_diff)),
                "max": float(np.max(hip_height_diff))
            },
            "ankle_height_difference": {
                "values": ankle_height_diff.tolist(),
                "mean": float(np.mean(ankle_height_diff)),
                "max": float(np.max(ankle_height_diff))
            }
        }
    }

def calculate_all_metrics(pose_array):
    """
    Calculate all performance metrics for a squat rep.
    
    Args:
        pose_array: numpy array of shape [57, num_frames] or [num_frames, 57]
                   or torch tensor of same shape
    
    Returns:
        dict with all calculated metrics
    """
    # Convert to joint format
    pose_joints = pose_to_joints(pose_array)
    
    metrics = {
        "squat_speed": calculate_squat_speed(pose_joints),
        "squat_depth": calculate_squat_depth(pose_joints),
        "knee_angles": calculate_knee_angles(pose_joints),
        "hip_angles": calculate_hip_angles(pose_joints),
        "balance": calculate_balance_metrics(pose_joints),
        "symmetry": calculate_rep_symmetry(pose_joints)
    }
    
    return metrics


