# visualize_data.py
# Loads the 'data/data_3D.pickle' file, displays a menu of all
# available sequences, and generates an INTERACTIVE 3D plot
# with a slider.
#
# Requires: matplotlib, numpy, pandas

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider # Import Slider
import sys

import config

# --- Configuration from your project files ---

# We need to manually define these here to replicate the data loading
# From data_loader.py:
JOINT_INDICES_TO_USE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 21, 22, 24]
# From config.py:
NUM_JOINTS = config.NUM_JOINTS # 19
NUM_COORDS = config.NUM_COORDS # 3
NUM_NODES = config.NUM_NODES   # 57
# From model.py:
BONE_LIST = [
    [0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7],
    [8, 9], [8, 12], [9, 10], [10, 11], [11, 17], [11, 18],
    [12, 13], [13, 14], [14, 15]
]
# --- New Configuration ---
LABEL_MAP = {
    1: "Correct"
    # Other labels will be shown as "Mistake (ID: #)"
}
# No longer saving a file, so no template needed.


def load_pickle_data(data_path):
    """
    Loads the raw pickle file.
    """
    print(f"Loading data from '{data_path}'...")
    try:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at '{data_path}'.", file=sys.stderr)
        print("Please run 'python create_mock_data.py' first.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading pickle file: {e}", file=sys.stderr)
        return None

def get_all_sequence_descriptors(data):
    """
    Scans the data and returns a list of dictionaries, one for each
    unique sequence (rep) found.
    """
    print("Scanning for all available sequences...")
    labels_df = pd.DataFrame(data['labels'], columns=['act', 'sub', 'lab', 'rep', 'frame'])
    labels_df[['lab', 'rep']] = labels_df[['lab', 'rep']].astype(int)

    # Group by all identifying factors
    grouped = labels_df.groupby(['act', 'sub', 'lab', 'rep'])
    
    descriptors = []
    for (act, sub, lab, rep), group in grouped:
        frame_indices = group.index.values
        descriptors.append({
            'act': act,
            'sub': sub,
            'lab': lab,
            'rep': rep,
            'frame_indices': frame_indices,
            'frame_count': len(frame_indices)
        })
        
    if not descriptors:
        print("Error: No sequences found in the data.", file=sys.stderr)
        return None

    return descriptors

def get_pose_data(data, frame_indices):
    """
    Extracts the specified pose data and normalizes it
    by anchoring the feet to the floor at (0,0,0).
    Returns:
        A numpy array of shape [seq_len, 19, 3]
    """
    try:
        # Index on axis 2 (joints)
        all_poses = data['poses'][:, :, JOINT_INDICES_TO_USE]
        
        # Get the raw pose sequence: [seq_len, 3_coords, 19_joints]
        pose_seq_raw_untransposed = all_poses[frame_indices]
        
        # Transpose to (seq_len, 19_joints, 3_coords) for visualization
        pose_seq_raw = pose_seq_raw_untransposed.transpose(0, 2, 1)
        
        if pose_seq_raw.shape[0] == 0: # Handle empty sequence
            return pose_seq_raw

        # --- Normalize to anchor feet to the floor ---
        # Get first frame (19, 3) to find the anchor point
        frame_0 = pose_seq_raw[0] 
        
        # Get ankle positions (indices 11 and 14)
        left_ankle_pos = frame_0[14]
        right_ankle_pos = frame_0[11]
        
        # Calculate midpoint between ankles
        mid_ankle_pos = (left_ankle_pos + right_ankle_pos) / 2.0
        
        # Get Y-coords of feet (indices 15, 17, 18)
        foot_joints_y = frame_0[[15, 17, 18], 1]
        lowest_y = np.min(foot_joints_y)
        
        # Create anchor point: X/Z from ankle midpoint, Y from lowest foot
        anchor_point = np.array([mid_ankle_pos[0], lowest_y, mid_ankle_pos[2]])
        
        # Subtract this anchor from all points in all frames
        normalized_pose_seq = pose_seq_raw - anchor_point
        
        print("Normalization complete: Anchored to feet.")
        return normalized_pose_seq
        
    except IndexError as e:
        print(f"Error filtering joints: {e}", file=sys.stderr)
        print("The data pickle file might be malformed or have fewer joints than expected (25).", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error extracting pose data: {e}", file=sys.stderr)
        return None

def show_interactive_plot(pose_seq, title):
    """
    Creates an interactive 3D plot with a slider.
    pose_seq shape: [seq_len, 19, 3]
    """
    seq_len, num_joints, num_coords = pose_seq.shape
    print(f"Showing interactive plot for sequence of length {seq_len} frames...")

    fig = plt.figure(figsize=(10, 9))
    # Make room for slider
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.2) 

    # Find data limits for consistent plotting
    min_vals = np.min(pose_seq, axis=(0, 1))
    max_vals = np.max(pose_seq, axis=(0, 1))
    
    # Create a buffer
    mid = (max_vals + min_vals) / 2
    span = (max_vals - min_vals).max() * 0.7 # A bit tighter
    
    x_lim = [mid[0] - span, mid[0] + span]
    y_lim = [mid[1] - span, mid[1] + span]
    z_lim = [mid[2] - span, mid[2] + span]

    def draw_frame(frame_num):
        """Helper function to draw a single frame."""
        ax.cla() # Clear the axis
        
        pose = pose_seq[frame_num] # [19, 3]
        x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]
        
        # Plot joints
        ax.scatter(x, y, z, c='r', marker='o')
        
        # Plot bones
        for (j1, j2) in BONE_LIST:
            ax.plot([x[j1], x[j2]], [y[j1], y[j2]], [z[j1], z[j2]], 'k-')
            
        # Set consistent view and labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        # Invert Z and Y to match common 3D pose conventions (Y up)
        ax.view_init(elev=15, azim=-75)
        ax.set_title(f"Frame {frame_num + 1} / {seq_len}\n{title}")

    # --- Create Slider ---
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03]) # x, y, width, height
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=seq_len - 1,
        valinit=0,
        valstep=1 # Go frame by frame
    )

    # --- Update function for slider ---
    def update(val):
        frame = int(slider.val)
        draw_frame(frame)
        fig.canvas.draw_idle() # Redraw the figure

    slider.on_changed(update)

    # Draw the initial frame
    draw_frame(0)
    
    # Show the plot
    print("Opening interactive plot window...")
    print("NOTE: You can click and drag the 3D plot to rotate the view.")
    plt.show()


def main():
    data = load_pickle_data(config.DATA_PATH)
    if data is None:
        return

    descriptors = get_all_sequence_descriptors(data)
    if not descriptors:
        return
        
    print("\n--- Available Sequences to Visualize ---")
    for i, desc in enumerate(descriptors):
        lab_name = LABEL_MAP.get(desc['lab'], f"Mistake (ID: {desc['lab']})")
        print(f"  [{i+1}] {desc['act']} - Sub: {desc['sub']} - Rep: {desc['rep']} - Label: {lab_name} ({desc['frame_count']} frames)")
    print("------------------------------------------")

    # Get user choice
    choice_idx = -1
    while True:
        try:
            choice_str = input(f"Enter the number of the sequence to visualize (1-{len(descriptors)}): ")
            choice_idx = int(choice_str) - 1 # Convert to 0-based index
            if 0 <= choice_idx < len(descriptors):
                break # Valid choice
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(descriptors)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    # Load and animate the selected sequence
    selected_desc = descriptors[choice_idx]
    print(f"\nLoading selected sequence: [{choice_idx+1}]...")
    
    pose_sequence = get_pose_data(data, selected_desc['frame_indices'])
    
    if pose_sequence is not None:
        # --- FIX ---
        # Original code referenced OUTPUT_FILENAME_TEMPLATE which was removed.
        # Create the title string directly using an f-string.
        lab_name = LABEL_MAP.get(selected_desc['lab'], f"Mistake (ID: {selected_desc['lab']})")
        title_str = f"{selected_desc['act']} - Sub: {selected_desc['sub']} - Rep: {selected_desc['rep']} - Label: {lab_name}"
        # --- END FIX ---
        
        # Call the new interactive plot function
        show_interactive_plot(pose_sequence, title_str)

if __name__ == "__main__":
    main()