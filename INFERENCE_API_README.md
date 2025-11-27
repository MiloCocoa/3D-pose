# Inference and API Documentation

This document describes the inference script and FastAPI endpoint for analyzing exercise poses.

## Files Created

1. **`inference.py`** - Script to run inference on a single validation rep
2. **`metrics.py`** - Utility module for calculating performance metrics
3. **`api.py`** - FastAPI endpoint for real-time pose analysis

## 1. Inference Script (`inference.py`)

### Usage

```bash
python inference.py <rep_index> [options]
```

### Arguments

- `rep_index` (required): Index of the rep in the validation set (0-based)
- `--checkpoint` (optional): Path to model checkpoint (default: `runs/20251107-180046/best_model.pth`)
- `--output` (optional): Output JSON file path (default: `inference_output_<rep_index>.json`)
- `--data_path` (optional): Path to data pickle file (default: from `config.py`)

### Example

```bash
python inference.py 0
python inference.py 5 --output results/rep_5.json
```

### Output Format

The script generates a JSON file with:
- True label and predicted class
- Confidence scores for all classes
- Input pose, target pose, and corrected pose data
- All data in list format for easy parsing

## 2. Performance Metrics (`metrics.py`)

This module provides comprehensive performance metrics for squat analysis:

### Available Metrics

1. **Squat Speed** (`calculate_squat_speed`)
   - Average speed (descent, ascent, combined) in units/frame
   - Peak speed value and position

2. **Squat Depth** (`calculate_squat_depth`)
   - Initial height, minimum height, depth
   - Frame at which minimum depth occurs

3. **Knee Angles** (`calculate_knee_angles`)
   - Left and right knee angles throughout the rep
   - Min, max, and range (in radians and degrees)

4. **Hip Angles** (`calculate_hip_angles`)
   - Left and right hip angles throughout the rep
   - Min, max, and range (in radians and degrees)

5. **Balance Metrics** (`calculate_balance_metrics`)
   - Lateral displacement (center of mass stability)
   - Base of support width

6. **Rep Symmetry** (`calculate_rep_symmetry`)
   - Knee angle symmetry (left vs right)
   - Hip angle symmetry
   - Vertical symmetry (height differences)

### Usage

```python
from metrics import calculate_all_metrics, pose_to_joints

# pose_array shape: [57, num_frames] or [num_frames, 57]
metrics = calculate_all_metrics(pose_array)
```

## 3. FastAPI Endpoint (`api.py`)

### Installation

Make sure to install the API dependencies:

```bash
pip install fastapi uvicorn[standard] pydantic
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### Starting the Server

```bash
python api.py
```

Or using uvicorn directly:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Root Endpoint

**GET** `/`

Returns API information and available endpoints.

#### 2. Health Check

**GET** `/health`

Returns API health status and model loading status.

#### 3. Analyze Pose

**POST** `/analyze`

Analyzes a single exercise rep and returns comprehensive results.

**Request Body:**

```json
{
  "pose": [
    [x1, y1, z1, x2, y2, z2, ...],  // Frame 1: 57 floats (19 joints * 3 coords)
    [x1, y1, z1, x2, y2, z2, ...],  // Frame 2
    ...                              // More frames
  ],
  "transpose": false  // Optional: if true, input is [57, num_frames] instead of [num_frames, 57]
}
```

**Response:**

```json
{
  "prediction": {
    "predicted_class": {
      "id": 0,
      "name": "Correct",
      "confidence": 0.95
    },
    "all_class_probabilities": {
      "Correct": 0.95,
      "Mistake_1": 0.02,
      ...
    }
  },
  "corrected_pose": {
    "shape": [57, 100],
    "data": [...],
    "format": " [57, 100] - 57 nodes (19 joints * 3 coords), 100 frames"
  },
  "performance_metrics": {
    "squat_speed": {...},
    "squat_depth": {...},
    "knee_angles": {...},
    "hip_angles": {...},
    "balance": {...},
    "symmetry": {...}
  },
  "skeleton_nodes": {
    "shape": [100, 19, 3],
    "data": [...],
    "format": " [100, 19, 3] - 100 frames, 19 joints, 3 coordinates (x, y, z)",
    "joint_names": ["joint_0", "joint_1", ...]
  }
}
```

### Example API Usage

**Using curl:**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "pose": [[...57 floats...], [...57 floats...], ...],
    "transpose": false
  }'
```

**Using Python:**

```python
import requests
import json

url = "http://localhost:8000/analyze"
data = {
    "pose": pose_data,  # Your pose data as list of lists
    "transpose": False
}

response = requests.post(url, json=data)
result = response.json()

print(f"Predicted: {result['prediction']['predicted_class']['name']}")
print(f"Confidence: {result['prediction']['predicted_class']['confidence']}")
```

### API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Notes

1. **Pose Data Format**: The API expects pose data as a list of frames, where each frame contains 57 floats (19 joints × 3 coordinates). The data will be automatically resampled to 100 frames if needed.

2. **Model Loading**: The model is loaded once at server startup. Make sure the checkpoint path in `api.py` is correct.

3. **CORS**: CORS is enabled for all origins. In production, restrict this to specific domains.

4. **Performance**: The API processes requests synchronously. For high-throughput scenarios, consider using async processing or a queue system.

## Troubleshooting

1. **Model not loading**: Check that the checkpoint path exists and is correct in `api.py` (startup_event function).

2. **Input format errors**: Ensure your pose data has exactly 57 values per frame (19 joints × 3 coordinates).

3. **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`


