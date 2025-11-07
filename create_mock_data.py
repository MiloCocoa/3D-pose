import pickle
import numpy as np
import os
import sys

try:
    import config
except ImportError:
    print("ERROR: Could not import config.py.")
    print("Please make sure this script is in the same root folder as config.py.")
    sys.exit(1)

# --- Configuration ---
NUM_FRAMES_TO_GENERATE = 1500 # Total number of frames in our mock dataset
NUM_SUBJECTS = 3            # Must be > 1 to pass the validator's warning
NUM_REPS_PER_SUBJECT = 10   # Mock reps
# PICKLE_PATH = config.DATA_PATH
PICKLE_PATH = r"data/barbell_data.pickle"
# ---------------------

def generate_mock_data():
    """
    Generates a mock barbell_data.pickle file with the correct structure
    and saves it to the path specified in config.py.
    """
    print(f"--- Generating Mock Data ---")
    print(f"Target file: {PICKLE_PATH}")

    if not os.path.exists(os.path.dirname(PICKLE_PATH)):
        print(f"Creating directory: {os.path.dirname(PICKLE_PATH)}")
        os.makedirs(os.path.dirname(PICKLE_PATH))

    # 1. Create 'labels' array: Shape (N, 5)
    # [exercise_id, subject_id, label_id, repetition_id, frame_index]
    
    # col 0: exercise_id (always 0 for this project)
    col_exercise = np.zeros(NUM_FRAMES_TO_GENERATE, dtype=int)
    
    # col 1: subject_id (from 0 to NUM_SUBJECTS-1)
    col_subject = np.random.randint(0, NUM_SUBJECTS, size=NUM_FRAMES_TO_GENERATE)
    
    # col 2: label_id (from 0 to NUM_CLASSES-1)
    col_label = np.random.randint(0, config.NUM_CLASSES, size=NUM_FRAMES_TO_GENERATE)
    
    # col 3: repetition_id (e.g., 0 to 9)
    col_rep = np.random.randint(0, NUM_REPS_PER_SUBJECT, size=NUM_FRAMES_TO_GENERATE)
    
    # col 4: frame_index (must be 0 to N-1)
    col_frame = np.arange(NUM_FRAMES_TO_GENERATE)
    
    # Stack them into the final (N, 5) array
    labels_array = np.stack([
        col_exercise,
        col_subject,
        col_label,
        col_rep,
        col_frame
    ], axis=1)

    print(f"Generated 'labels' array with shape: {labels_array.shape}")

    # 2. Create 'poses' array: Shape (N, 3, 21)
    # (N, NUM_COORDS, NUM_JOINTS)
    poses_array = np.random.rand(
        NUM_FRAMES_TO_GENERATE,
        config.NUM_COORDS,
        config.NUM_JOINTS
    )
    
    print(f"Generated 'poses' array with shape: {poses_array.shape}")

    # 3. Create final dictionary
    mock_data = {
        'labels': labels_array,
        'poses': poses_array
    }

    # 4. Save to pickle file
    try:
        with open(PICKLE_PATH, 'wb') as f:
            pickle.dump(mock_data, f)
        print(f"\n[SUCCESS] Mock data saved to {PICKLE_PATH}")
    except Exception as e:
        print(f"\n[FAIL] Could not save pickle file. Error: {e}")

if __name__ == "__main__":
    generate_mock_data()