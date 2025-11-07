import pickle
import numpy as np
import os
import sys

# Import the configuration from your main project
try:
    import config
except ImportError:
    print("ERROR: Could not import config.py.")
    print("Please make sure this script is in the same root folder as config.py.")
    sys.exit(1)

# --- CONFIGURATION ---
# The path to the data file we want to check.
# PICKLE_PATH = config.DATA_PATH 
PICKLE_PATH = r"data/barbell_data.pickle"
# --- END CONFIGURATION ---


def print_check(message, status):
    """Helper function to print formatted check messages."""
    if status == "pass":
        print(f"  [PASS] {message}")
    elif status == "warn":
        print(f"  [WARN] {message}")
    else:
        print(f"  [FAIL] {message}")

def validate_pickle_file(file_path):
    """
    Loads and validates the barbell_data.pickle file against
    the project's config.py settings.
    """
    print(f"--- Validating Data File: {file_path} ---")
    
    # 1. File Existence
    if not os.path.exists(file_path):
        print_check(f"File not found at '{file_path}'", "fail")
        return False
    print_check("File exists", "pass")
    
    # 2. Pickle Loading
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print_check("File loaded successfully (is a valid pickle)", "pass")
    except Exception as e:
        print_check(f"Failed to load pickle file. Error: {e}", "fail")
        return False

    # 3. Top-Level Structure
    if not isinstance(data, dict):
        print_check(f"Data is not a dictionary (dict). Got: {type(data)}", "fail")
        return False
        
    errors = 0
    
    # 4. Check for 'labels' key
    if 'labels' not in data:
        print_check("Missing required key: 'labels'", "fail")
        errors += 1
    else:
        print_check("Found key: 'labels'", "pass")
        
    # 5. Check for 'poses' key
    if 'poses' not in data:
        print_check("Missing required key: 'poses'", "fail")
        errors += 1
    else:
        print_check("Found key: 'poses'", "pass")
        
    if errors > 0:
        print("\nStopping validation due to missing critical keys.")
        return False
        
    # --- 'labels' Key Validation ---
    print("\nValidating 'labels'...")
    labels = data['labels']
    if not isinstance(labels, (np.ndarray, list)):
        print_check(f"'labels' must be a list or numpy array. Got: {type(labels)}", "fail")
        return False
    
    try:
        labels = np.asarray(labels)
    except Exception as e:
        print_check(f"Could not convert 'labels' to numpy array. Error: {e}", "fail")
        return False

    if labels.ndim != 2 or labels.shape[1] != 5:
        print_check(f"Shape must be (N, 5). Got: {labels.shape}", "fail")
        return False
    
    num_frames_from_labels = labels.shape[0]
    print_check(f"Shape is (N, 5). Found N={num_frames_from_labels} frames.", "pass")

    # --- 'poses' Key Validation ---
    print("\nValidating 'poses'...")
    poses = data['poses']
    if not isinstance(poses, np.ndarray):
        print_check(f"'poses' must be a numpy array. Got: {type(poses)}", "fail")
        return False

    if poses.ndim != 3:
        print_check(f"Shape must be (N, 3, 21). Got {poses.ndim} dimensions.", "fail")
        errors += 1
    elif poses.shape[1] != config.NUM_COORDS:
        print_check(f"Dimension 1 must be {config.NUM_COORDS} (for x,y,z). Got: {poses.shape[1]}", "fail")
        errors += 1
    elif poses.shape[2] != config.NUM_JOINTS:
        print_check(f"Dimension 2 must be {config.NUM_JOINTS} (from config.py). Got: {poses.shape[2]}", "fail")
        errors += 1
    
    num_frames_from_poses = poses.shape[0]
    print_check(f"Shape is (N, 3, 21). Found N={num_frames_from_poses} frames.", "pass")

    # --- Cross-Validation ---
    print("\nValidating consistency...")
    if num_frames_from_labels != num_frames_from_poses:
        print_check(f"Frame count mismatch! 'labels' has {num_frames_from_labels} frames, "
                    f"but 'poses' has {num_frames_from_poses} frames.", "fail")
        errors += 1
    else:
        print_check("Frame counts match between 'labels' and 'poses'", "pass")

    # --- 'labels' Content Sanity Check ---
    print("\nValidating 'labels' content...")
    max_label = int(np.max(labels[:, 2]))
    min_label = int(np.min(labels[:, 2]))
    
    if min_label < 0:
        print_check(f"Found negative label ID: {min_label}. Labels must be >= 0.", "fail")
        errors += 1
    
    if max_label >= config.NUM_CLASSES:
        print_check(f"Found label ID {max_label}, but config.NUM_CLASSES is {config.NUM_CLASSES}. "
                    f"Labels must be 0-indexed (i.e., 0 to {config.NUM_CLASSES - 1}).", "fail")
        errors += 1
    else:
        print_check(f"All label IDs (0-{max_label}) are within range (0-{config.NUM_CLASSES - 1})", "pass")
    
    if len(np.unique(labels[:, 1])) < 2:
        print_check(f"Only found 1 unique subject ID. You need at least 2 subjects "
                    f"for a train/test split.", "warn")
    else:
        print_check(f"Found {len(np.unique(labels[:, 1]))} unique subject IDs. Good!", "pass")

    # --- Final Result ---
    if errors == 0:
        print("\n--- Validation SUCCESS ---")
        print("Your `barbell_data.pickle` file is correctly formatted.")
        return True
    else:
        print(f"\n--- Validation FAILED ---")
        print(f"Found {errors} critical error(s). Please fix the data and try again.")
        return False

if __name__ == "__main__":
    validate_pickle_file(PICKLE_PATH)