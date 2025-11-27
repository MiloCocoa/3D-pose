# 3D Barbell Squat Form Correction

This project is a modern re-implementation of the concepts from the paper "3D Pose Based Feedback for Physical Exercises," adapted for a new task: analyzing and correcting 3D barbell squat form.

The goal is to build a model that can perform two tasks:

1. **Mistake Classification:** Identify common mistakes in a barbell squat (e.g., "Butt Wink," "Bar Tilt," "Knees Inward").
2. **Pose Correction:** Generate a "corrected" 3D pose sequence to show the user how to perform the squat correctly.

This implementation uses **PyTorch** and **PyTorch Geometric (PyG)**, and it replaces the original paper's DCT-based (Discrete Cosine Transform) input with resampled raw 3D coordinate sequences for improved stability and reproducibility.

## Project Structure

Here is an overview of the key files in this repository:

- `main.py`: The main script to start and run the training and evaluation pipeline.
- `model.py`: Defines the dual-branch Graph Convolutional Network (GCN) architecture.
- `config.py`: **Crucial file.** Contains all hyperparameters, data paths, and model settings (e.g., number of joints, classes, hidden dimensions).
- `data_loader.py`: Loads the raw `.pickle` data, performs pairing (incorrect rep -> correct rep), and resamples sequences to a fixed length.
- `train.py`: Contains the core `train_one_epoch` and `evaluate` functions.
- `utils.py`: Helper functions for setting random seeds and configuring logging.
- `data/`: This directory should contain your dataset.
- `validate_data.py`: A utility script to check if your custom dataset pickle file is formatted correctly.
- `create_mock_data.py`: A utility script to generate a correctly formatted, random "dummy" dataset to test the pipeline.
- `data_collection_plan.md`: A detailed guide on how to collect and label a high-quality barbell squat dataset.

## 1. Installation

This project requires Python 3.10+ and PyTorch.

### Step 1: Clone and Set Up Environment

```
git clone [https://your-repo-url.git](https://your-repo-url.git)
cd exercise-correction-project
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

### Step 2: Install PyTorch

First, install PyTorch according to your specific CUDA version. Visit the [PyTorch website](https://pytorch.org/get-started/locally/ "null") for the correct command.

**Example (for CUDA 12.1):**

```
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

### Step 3: Install PyTorch Geometric (PyG)

PyG requires special CUDA-compiled libraries. You **must** install these packages separately, matching your PyTorch and CUDA versions.

**Example (for PyTorch 2.3.0 and CUDA 12.1):**

```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv --index-url [https://data.pyg.org/whl/torch-2.3.0+cu121.html](https://data.pyg.org/whl/torch-2.3.0+cu121.html)
```

## Frontend web client

The repository now ships with a minimal React UI in [`frontend/`](frontend) that calls the FastAPI `/analyze` endpoint. Use it to paste pose JSON, upload a file exported from `pose_examples.json`, and inspect predictions/metrics without touching Python notebooks.

1. Make sure the API server (`uvicorn api:app --reload`) is running and reachable from the browser.
2. In a new terminal:
   ```
   cd frontend
   npm install          # first run only
   npm run dev
   ```
3. Visit the printed Vite URL (defaults to `http://localhost:5173`). The UI assumes the backend lives at `http://localhost:8000`; override it by creating a `.env` file in `frontend/` with `VITE_API_BASE_URL=https://your-host`.

The form exposes a JSON textarea, optional file upload, and a transpose toggle. Responses are summarized (class name, confidence, metric preview) and the raw JSON is available for copy/paste into other tools.