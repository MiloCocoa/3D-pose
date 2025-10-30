# utils.py
# Helper functions for setting seeds, logging, etc.

import torch
import numpy as np
import random
import logging

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_logger(log_file_path):
    """Creates a logger that writes to both a file and the console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
    
    return logger