# api.py
# FastAPI endpoint for real-time exercise pose analysis.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
import os
import json

import config
from model import DualBranchGCN, create_skeleton_graph
from metrics import calculate_all_metrics, pose_to_joints
from utils import set_seed

# Initialize FastAPI app
app = FastAPI(title="3D Pose Exercise Analysis API", version="1.0.0")

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and edge_index (loaded at startup)
model = None
edge_index = None
device = None
TARGET_POSES = {}       # Map: example_index -> pose
CLASS_TARGET_POSES = {} # Map: class_id -> representative target pose

class PoseData(BaseModel):
    """Input pose data structure."""
    # Pose data as list of frames, each frame is [57] (19 joints * 3 coords)
    # Or as [num_frames, 57] array
    pose: List[List[float]]
    # Optional: specify if pose is already in [57, num_frames] format
    transpose: bool = False
    # Optional: index of the example in the dataset to use ground truth target pose
    example_index: Optional[int] = None

class AnalysisResponse(BaseModel):
    """Response structure for pose analysis."""
    prediction: dict
    target_pose: dict
    performance_metrics: dict
    skeleton_nodes: dict

def load_model_once(checkpoint_path: str):
    """Load model once at startup."""
    global model, edge_index, device
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    device = torch.device(config.DEVICE)
    set_seed(42)
    
    edge_index = create_skeleton_graph().to(device)
    
    model = DualBranchGCN(
        in_channels=config.NUM_FRAMES,
        hidden_channels=config.HIDDEN_DIM,
        out_channels_class=config.NUM_CLASSES,
        out_channels_corr=config.NUM_FRAMES,
        num_blocks=config.NUM_GCN_BLOCKS,
        dropout=config.DROPOUT
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print(f"Model loaded successfully from {checkpoint_path} on {device}")

def get_class_name(class_id: int) -> str:
    """Map class ID to human-readable name."""
    class_names = {
        0: "Correct",
        1: "Mistake_1",
        2: "Mistake_2",
        3: "Mistake_3",
        4: "Mistake_4",
        5: "Mistake_5",
        6: "Mistake_6",
        7: "Mistake_7",
        8: "Mistake_8",
        9: "Mistake_9",
        10: "Mistake_10",
        11: "Mistake_11"
    }
    return class_names.get(class_id, f"Unknown_{class_id}")

def preprocess_pose_data(pose_data: List[List[float]], transpose: bool = False) -> torch.Tensor:
    """
    Preprocess pose data to model input format.
    
    Args:
        pose_data: List of frames, each frame is a list of 57 floats
        transpose: If True, assumes input is [57, num_frames], else [num_frames, 57]
    
    Returns:
        torch.Tensor of shape [1, 57, 100] (batch_size=1, num_nodes=57, num_frames=100)
    """
    pose_array = np.array(pose_data, dtype=np.float32)
    
    # Handle different input formats
    if transpose:
        # Input is [57, num_frames]
        if pose_array.shape[0] != 57:
            raise ValueError(f"Expected 57 nodes in first dimension, got {pose_array.shape[0]}")
        num_frames = pose_array.shape[1]
    else:
        # Input is [num_frames, 57]
        if pose_array.shape[1] != 57:
            raise ValueError(f"Expected 57 nodes in second dimension, got {pose_array.shape[1]}")
        num_frames = pose_array.shape[0]
        # Transpose to [57, num_frames]
        pose_array = pose_array.T
    
    # Resample to NUM_FRAMES if needed
    if num_frames != config.NUM_FRAMES:
        from scipy.interpolate import interp1d
        x = np.linspace(0, 1, num_frames)
        x_new = np.linspace(0, 1, config.NUM_FRAMES)
        f = interp1d(x, pose_array, kind='linear', axis=1, 
                    bounds_error=False, fill_value="extrapolate")
        pose_array = f(x_new)
    
    # Add batch dimension: [1, 57, 100]
    pose_tensor = torch.tensor(pose_array, dtype=torch.float32).unsqueeze(0)
    
    return pose_tensor

def run_inference_api(pose_tensor: torch.Tensor) -> tuple:
    """
    Run model inference.
    
    Returns:
        (pred_class, confidence, all_probs, corrected_pose)
    """
    global model, edge_index, device
    
    pose_tensor = pose_tensor.to(device)
    B = pose_tensor.shape[0]
    N = pose_tensor.shape[1]
    batch_vec = torch.arange(B, device=device).repeat_interleave(N)
    
    with torch.no_grad():
        pred_logits, pred_corrected_pose = model(
            pose_tensor,
            edge_index,
            batch_vec,
            labels=None,
            use_feedback=True
        )
    
    # Get prediction
    pred_probs = torch.softmax(pred_logits, dim=1)
    pred_class = torch.argmax(pred_logits, dim=1).item()
    confidence = pred_probs[0, pred_class].item()
    all_probs = pred_probs[0].cpu().numpy().tolist()
    
    # Remove batch dimension from corrected pose
    corrected_pose = pred_corrected_pose.squeeze(0).cpu().numpy()  # [57, 100]
    
    return pred_class, confidence, all_probs, corrected_pose

@app.on_event("startup")
async def startup_event():
    """Load model at startup."""
    checkpoint_path = "runs/20251107-180046/best_model.pth"
    try:
        load_model_once(checkpoint_path)
        
        # Load target poses if available
        if os.path.exists("target_poses.json"):
            with open("target_poses.json", "r") as f:
                data = json.load(f)
                for item in data:
                    TARGET_POSES[item['example_index']] = item['target_pose']
                    
                    # Populate class-based representative poses
                    # Use the first encountered example for each class as the representative
                    # Note: data_loader.py subtracts 1 from raw_label for 0-indexed class_id
                    # So we look for 'zero_index_label' or map raw_label manually
                    class_id = item['meta'].get('zero_index_label')
                    if class_id is not None and class_id not in CLASS_TARGET_POSES:
                        CLASS_TARGET_POSES[class_id] = item['target_pose']
                        
            print(f"Loaded {len(TARGET_POSES)} ground truth target poses.")
            print(f"Loaded representative poses for {len(CLASS_TARGET_POSES)} classes.")
            
        print("API ready to accept requests.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "3D Pose Exercise Analysis API",
        "status": "running",
        "endpoints": {
            "/": "This endpoint",
            "/analyze": "POST - Analyze exercise pose data",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_pose(pose_data: PoseData):
    """
    Analyze a single exercise rep.
    
    Expected input format:
    - pose: List of frames, where each frame is a list of 57 floats (19 joints * 3 coords)
    - transpose: If True, input is [57, num_frames], else [num_frames, 57]
    - example_index: (Optional) Index of the example in the dataset to use ground truth target pose.
    
    Returns comprehensive analysis including:
    - Model prediction (mistake classification)
    - Target pose sequence (echoed/resampled input pose)
    - Performance metrics (speed, depth, angles, balance, symmetry)
    - Skeleton nodes for visualization
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess input
        pose_tensor = preprocess_pose_data(pose_data.pose, pose_data.transpose)
        
        # Run inference
        pred_class, confidence, all_probs, corrected_pose = run_inference_api(pose_tensor)
        
        # Override with Ground Truth logic
        # 1. Specific Example Override (Highest Priority)
        if pose_data.example_index is not None and pose_data.example_index in TARGET_POSES:
            print(f"Using ground truth target pose for example index {pose_data.example_index}")
            corrected_pose = np.array(TARGET_POSES[pose_data.example_index])
        # 2. Class-Based Representative Override (Fallback)
        elif pred_class in CLASS_TARGET_POSES:
            # If we predicted a mistake, show the representative CORRECT pose for that mistake context
            # Ideally, we want the "Correct" class pose (class 0) for comparison, but the dataset pairs incorrect -> correct.
            # The 'target_pose' stored in CLASS_TARGET_POSES is always the CORRECT version of the rep.
            # So it is safe to use the target pose associated with ANY class example, because they all map to "Correct" form.
            print(f"Using representative target pose for class {pred_class}")
            corrected_pose = np.array(CLASS_TARGET_POSES[pred_class])
        
        # Get input pose for metrics (remove batch dimension)
        input_pose = pose_tensor.squeeze(0).cpu().numpy()  # [57, 100]
        
        # Calculate performance metrics on input pose
        performance_metrics = calculate_all_metrics(input_pose)
        
        # Convert pose to joint format for skeleton nodes
        pose_joints = pose_to_joints(input_pose)  # [100, 19, 3]
        
        # Prepare response
        response = {
            "prediction": {
                "predicted_class": {
                    "id": pred_class,
                    "name": get_class_name(pred_class),
                    "confidence": confidence
                },
                "all_class_probabilities": {
                    get_class_name(i): prob for i, prob in enumerate(all_probs)
                }
            },
            "target_pose": {
                "shape": list(corrected_pose.shape),
                "data": corrected_pose.tolist(),
                "format": " [57, 100] - 57 nodes (19 joints * 3 coords), 100 frames"
            },
            "performance_metrics": performance_metrics,
            "skeleton_nodes": {
                "shape": list(pose_joints.shape),
                "data": pose_joints.tolist(),
                "format": " [100, 19, 3] - 100 frames, 19 joints, 3 coordinates (x, y, z)",
                "joint_names": [
                    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4",
                    "joint_5", "joint_6", "joint_7", "joint_8", "joint_9",
                    "joint_10", "joint_11", "joint_12", "joint_13", "joint_14",
                    "joint_15", "joint_16", "joint_17", "joint_18"
                ]
            }
        }
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


