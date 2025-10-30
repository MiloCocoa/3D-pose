# config.py
# This file stores all hyperparameters and configuration settings
# for easy modification and reproducibility.

import torch

# --- Training ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
EPOCHS = 200
LR_DECAY_STEP = 100
LR_DECAY_GAMMA = 0.9

# --- Model ---
HIDDEN_DIM = 256
NUM_GCN_BLOCKS = 2  # Number of GCN blocks in the shared backbone
DROPOUT = 0.6
BETA = 1.0  # Loss balancing factor (loss = loss_class + BETA * loss_corr)

# --- Data ---
# We will resample all sequences to a fixed length for stable batching
# This replaces the need for DCT.
NUM_FRAMES = 100
NUM_JOINTS = 19  # We use 19 joints as in the original repo
NUM_COORDS = 3   # (x, y, z)
NUM_NODES = NUM_JOINTS * NUM_COORDS # 19 * 3 = 57 nodes in the graph
NUM_CLASSES = 12  # Number of mistake classes (for classification)

# --- File Paths ---
DATA_PATH = "data/data_3D.pickle"
SAVE_DIR = "runs/"