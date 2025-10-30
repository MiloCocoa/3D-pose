# data_loader.py
# Handles loading, pairing, and preprocessing the dataset.

import torch
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d

import config

class ExerciseDataset(Dataset):
    """
    Custom PyTorch Dataset for the EC3D exercise data.
    - Loads the raw data.
    - Pairs incorrect sequences with correct ones.
    - Resamples all sequences to a fixed length (config.NUM_FRAMES).
    """
    def __init__(self, data_path, subject_ids, is_cuda=False):
        """
        Loads and pairs the data.
        subject_ids: A list of subject IDs for this split (e.g., [0, 1, 2]).
        """
        self.num_frames = config.NUM_FRAMES
        self.num_nodes = config.NUM_NODES
        self.is_cuda = is_cuda
        
        # 1. Load and filter data
        correct_poses, incorrect_poses = self._load_data(data_path, subject_ids)
        
        # 2. Pair sequences
        self.pairs = self._pair_sequences(correct_poses, incorrect_poses) 
        # self.pairs is now a list of:
        # (incorrect_pose_array, correct_pose_array, mistake_label)

    def _load_data(self, data_path, subject_indices):
        """
        Loads the data_3D.pickle file, filters by subject *index*,
        and groups frames into sequences.
        This is a corrected version that mimics the original repo's logic.
        """
        with open(data_path, "rb") as f:
            data_gt = pickle.load(f)
        
        # In this project, data_gt is the same as data, as add_data is None
        data = data_gt 

        # --- Create 'labels' dataframe (for incorrect poses) ---
        labels_df = pd.DataFrame(data['labels'], columns=['act', 'sub', 'lab', 'rep', 'frame'])
        labels_df[['lab', 'rep']] = labels_df[['lab', 'rep']].astype(int)

        # --- Create 'labels_gt' dataframe (for correct poses) ---
        labels_gt_df = pd.DataFrame(data_gt['labels'], columns=['act', 'sub', 'lab', 'rep', 'frame'])
        labels_gt_df[['lab', 'rep']] = labels_gt_df[['lab', 'rep']].astype(int)

        # --- THIS IS THE FIX ---
        # Get all unique subjects from the data
        all_subjects = labels_df[['act', 'sub', 'lab', 'rep']].drop_duplicates().groupby('sub').count().rep
        
        # Select the subjects for this split using the *indices* (e.g., [0, 1, 2])
        subjects_to_use = all_subjects.index[subject_indices]
        
        # Filter BOTH dataframes by the *actual* subject IDs
        labels_df = labels_df[labels_df['sub'].isin(subjects_to_use)]
        labels_gt_df = labels_gt_df[labels_gt_df['sub'].isin(subjects_to_use)]
        # --- END FIX ---

        # Select the 19 joints (as in the original repo)
        joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 21, 22, 24]
        all_poses = data['poses'][:, :, joint_indices]
        all_poses_gt = data_gt['poses'][:, :, joint_indices]

        correct_poses = {}   # Key: (act, sub, rep), Val: pose_array
        incorrect_poses = {} # Key: (act, sub, lab, rep), Val: pose_array

        # Group frames for CORRECT poses (lab == 1)
        grouped_gt = labels_gt_df[labels_gt_df['lab'] == 1].groupby(['act', 'sub', 'rep'])
        for (act, sub, rep), group in grouped_gt:
            frame_indices = group.index.values 
            pose_seq = all_poses_gt[frame_indices].reshape(-1, self.num_nodes).T
            correct_poses[(act, sub, rep)] = pose_seq
            
        # Group frames for INCORRECT poses (lab != 1)
        grouped_inc = labels_df[labels_df['lab'] != 1].groupby(['act', 'sub', 'lab', 'rep'])
        for (act, sub, lab, rep), group in grouped_inc:
            frame_indices = group.index.values 
            pose_seq = all_poses[frame_indices].reshape(-1, self.num_nodes).T
            incorrect_poses[(act, sub, lab, rep)] = pose_seq
                
        return correct_poses, incorrect_poses

    def _pair_sequences(self, correct_poses, incorrect_poses):
        """
        Pairs every 'incorrect' sequence with its 'correct' counterpart
        from the same subject, exercise, and repetition number.
        """
        pairs = []
        for (act, sub, lab, rep), inc_pose in incorrect_poses.items():
            # Find the corresponding "correct" repetition key
            # --- THIS IS THE KEY FIX ---
            correct_key = (act, sub, rep) # The key for correct_poses
            
            # Check if this corresponding correct sequence exists
            if correct_key in correct_poses:
                cor_pose = correct_poses[correct_key]
                
                # We subtract 1 from the label to make it 0-indexed (0-11)
                # This is crucial for nn.CrossEntropyLoss
                pairs.append((inc_pose, cor_pose, lab - 1))
                
        return pairs

    def _resample_sequence(self, seq, target_len):
        """
        Resamples a sequence (shape: [num_nodes, seq_len])
        to [num_nodes, target_len] using linear interpolation.
        """
        seq = np.asarray(seq)
        seq_len = seq.shape[1]
        
        # Handle edge case: sequence is too short for interpolation
        if seq_len <= 1:
            # Just repeat the single frame
            return torch.tensor(np.tile(seq, (1, target_len)), dtype=torch.float32)
            
        x = np.linspace(0, 1, seq_len)
        x_new = np.linspace(0, 1, target_len)
        
        # interp1d expects (features, time) and our data is (nodes, time),
        # which works correctly with axis=1.
        f = interp1d(x, seq, kind='linear', axis=1, bounds_error=False, fill_value="extrapolate")
        resampled_seq = f(x_new)
        return torch.tensor(resampled_seq, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        """
        Returns (input_pose, target_pose, label)
        All are torch tensors.
        input_pose: [num_nodes, num_frames]
        target_pose: [num_nodes, num_frames]
        label: long
        """
        inc_pose_raw, cor_pose_raw, label = self.pairs[index]
        
        input_pose = self._resample_sequence(inc_pose_raw, self.num_frames)
        target_pose = self._resample_sequence(cor_pose_raw, self.num_frames)
        
        # Final shape is [57, 100], which our model expects
        
        return input_pose, target_pose, torch.tensor(label, dtype=torch.long)

def create_dataloaders(data_path, batch_size=None):
    """
    Creates and returns the train and test DataLoaders.
    If batch_size not given, uses config.BATCH_SIZE
    """
    batch_size = batch_size or config.BATCH_SIZE
    
    # Original repo splits were [0, 1], [2], [3]
    # We'll use [0, 1, 2] for training and [3] for testing
    train_subjects = [0, 1, 2]
    test_subjects = [3]

    train_dataset = ExerciseDataset(data_path, train_subjects)
    test_dataset = ExerciseDataset(data_path, test_subjects)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader