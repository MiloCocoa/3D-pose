"""
visualize_inference.py

Interactive visualization for model inference on validation reps.

This script:
- Loads the trained model and validation (test) dataset
- Lets you choose a validation rep index via command line or prompt
- Runs inference to get:
    - Predicted class + confidence
    - Corrected pose sequence
- Visualizes:
    - Original input pose
    - Corrected pose
  side-by-side in an interactive 3D plot with a frame slider.

Usage:
    python visualize_inference.py
    python visualize_inference.py --rep_index 0
"""

import argparse
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider

import config
from data_loader import create_dataloaders
from metrics import pose_to_joints
from model import DualBranchGCN, create_skeleton_graph
from utils import set_seed


# Bone connections for the 19-joint skeleton (same as in model.py / visualize_data.py)
BONE_LIST = [
    [0, 1],
    [1, 2],
    [1, 5],
    [1, 8],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [8, 9],
    [8, 12],
    [9, 10],
    [10, 11],
    [11, 17],
    [11, 18],
    [12, 13],
    [13, 14],
    [14, 15],
]


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
        11: "Mistake_11",
    }
    return class_names.get(class_id, f"Unknown_{class_id}")


def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained model and skeleton graph."""
    edge_index = create_skeleton_graph().to(device)

    model = DualBranchGCN(
        in_channels=config.NUM_FRAMES,
        hidden_channels=config.HIDDEN_DIM,
        out_channels_class=config.NUM_CLASSES,
        out_channels_corr=config.NUM_FRAMES,
        num_blocks=config.NUM_GCN_BLOCKS,
        dropout=config.DROPOUT,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model, edge_index


def run_inference_single(
    model: DualBranchGCN,
    edge_index: torch.Tensor,
    input_pose: torch.Tensor,
    device: torch.device,
):
    """
    Run inference on a single input pose.

    Args:
        input_pose: [57, 100] tensor
    """
    input_pose = input_pose.unsqueeze(0).to(device)  # [1, 57, 100]
    B = input_pose.shape[0]
    N = input_pose.shape[1]
    batch_vec = torch.arange(B, device=device).repeat_interleave(N)

    with torch.no_grad():
        pred_logits, pred_corrected_pose = model(
            input_pose,
            edge_index,
            batch_vec,
            labels=None,
            use_feedback=True,
        )

    pred_probs = torch.softmax(pred_logits, dim=1)
    pred_class = int(torch.argmax(pred_logits, dim=1).item())
    confidence = float(pred_probs[0, pred_class].item())
    all_probs = pred_probs[0].cpu().numpy()

    corrected_pose = pred_corrected_pose.squeeze(0).cpu()  # [57, 100]
    return pred_class, confidence, all_probs, corrected_pose


def normalize_pose_sequence(pose_joints: np.ndarray) -> np.ndarray:
    """
    Normalize pose sequence by anchoring feet to the floor (same logic as visualize_data.py).

    Args:
        pose_joints: [seq_len, 19, 3]
    """
    if pose_joints.shape[0] == 0:
        return pose_joints

    pose_norm = pose_joints.copy()

    frame0 = pose_norm[0]
    left_ankle_pos = frame0[14]
    right_ankle_pos = frame0[11]
    mid_ankle_pos = (left_ankle_pos + right_ankle_pos) / 2.0

    foot_joints_y = frame0[[15, 17, 18], 1]
    lowest_y = np.min(foot_joints_y)

    anchor_point = np.array([mid_ankle_pos[0], lowest_y, mid_ankle_pos[2]])
    pose_norm = pose_norm - anchor_point
    return pose_norm


def visualize_sequences(
    input_pose_57xT: torch.Tensor,
    target_pose_57xT: torch.Tensor,
    true_label: Optional[int],
    pred_class: int,
    confidence: float,
):
    """
    Visualize original and target-correct pose side-by-side with a frame slider.
    Left: original input rep.
    Right: ground-truth corrected rep (target_pose from dataset).
    """
    # Convert to numpy [57, T]
    input_np = input_pose_57xT.cpu().numpy()
    target_np = target_pose_57xT.cpu().numpy()

    # Convert to [T, 19, 3] for easier plotting
    input_joints = normalize_pose_sequence(pose_to_joints(input_np))  # [T, 19, 3]
    target_joints = normalize_pose_sequence(
        pose_to_joints(target_np)
    )  # [T, 19, 3]

    seq_len = input_joints.shape[0]

    # Compute global limits across both sequences for consistent scaling
    all_points = np.concatenate(
        [input_joints.reshape(seq_len, -1, 3), target_joints.reshape(seq_len, -1, 3)],
        axis=1,
    )
    min_vals = np.min(all_points, axis=(0, 1))
    max_vals = np.max(all_points, axis=(0, 1))

    mid = (max_vals + min_vals) / 2.0
    span = (max_vals - min_vals).max() * 0.7
    x_lim = [mid[0] - span, mid[0] + span]
    y_lim = [mid[1] - span, mid[1] + span]
    z_lim = [mid[2] - span, mid[2] + span]

    fig = plt.figure(figsize=(14, 7))
    plt.subplots_adjust(bottom=0.2, wspace=0.3)

    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    true_name = get_class_name(true_label) if true_label is not None else "Unknown"
    pred_name = get_class_name(pred_class)

    title_base = (
        f"True: {true_name} | Pred: {pred_name} (conf: {confidence:.3f})"
    )

    def draw_frame(frame_idx: int):
        """Draw a single frame in both subplots."""
        ax1.cla()
        ax2.cla()

        # Original pose
        pose_in = input_joints[frame_idx]  # [19, 3]
        x1, y1, z1 = pose_in[:, 0], pose_in[:, 1], pose_in[:, 2]
        ax1.scatter(x1, y1, z1, c="r", marker="o")
        for j1, j2 in BONE_LIST:
            ax1.plot([x1[j1], x1[j2]], [y1[j1], y1[j2]], [z1[j1], z1[j2]], "k-")
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)
        ax1.set_zlim(z_lim)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.view_init(elev=15, azim=-75)
        ax1.set_title(f"Original Pose\nFrame {frame_idx + 1}/{seq_len}")

        # Target-correct pose
        pose_corr = target_joints[frame_idx]
        x2, y2, z2 = pose_corr[:, 0], pose_corr[:, 1], pose_corr[:, 2]
        ax2.scatter(x2, y2, z2, c="b", marker="o")
        for j1, j2 in BONE_LIST:
            ax2.plot([x2[j1], x2[j2]], [y2[j1], y2[j2]], [z2[j1], z2[j2]], "k-")
        ax2.set_xlim(x_lim)
        ax2.set_ylim(y_lim)
        ax2.set_zlim(z_lim)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.view_init(elev=15, azim=-75)
        ax2.set_title(f"Target Correct Pose\nFrame {frame_idx + 1}/{seq_len}")

        fig.suptitle(title_base, fontsize=14)

        # print("Corrected frame stats:",
        #     "x range", float(x2.min()), "to", float(x2.max()),
        #     "y range", float(y2.min()), "to", float(y2.max()),
        #     "z range", float(z2.min()), "to", float(z2.max()))

    # Slider
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="Frame",
        valmin=0,
        valmax=seq_len - 1,
        valinit=0,
        valstep=1,
    )

    def on_slider_change(val):
        frame = int(slider.val)
        draw_frame(frame)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)

    # Initial frame
    draw_frame(0)

    print("Opening interactive window. Drag to rotate; use slider to scrub frames.")
    plt.show()


def choose_rep_index_interactively(dataset_len: int) -> int:
    """Prompt user to choose a validation rep index."""
    print(f"Validation set contains {dataset_len} reps (indices 0 to {dataset_len - 1}).")
    while True:
        try:
            choice = input(
                f"Enter rep index to visualize (0-{dataset_len - 1}): "
            ).strip()
            idx = int(choice)
            if 0 <= idx < dataset_len:
                return idx
            print("Invalid index. Try again.")
        except ValueError:
            print("Please enter a valid integer index.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize model inference (original vs corrected pose) on a validation rep."
    )
    parser.add_argument(
        "--rep_index",
        type=int,
        default=None,
        help="Validation rep index to visualize (0-based). If not provided, you will be prompted.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/20251107-180046/best_model.pth",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=config.DATA_PATH,
        help="Path to data pickle file (default: from config.DATA_PATH).",
    )

    args = parser.parse_args()

    # Basic setup
    set_seed(42)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    print(f"Loading model from {args.checkpoint}...")
    model, edge_index = load_model(args.checkpoint, device)
    print("Model loaded.")

    # Load validation data
    print(f"Loading data from {args.data_path}...")
    _, test_loader = create_dataloaders(args.data_path)
    test_dataset = test_loader.dataset

    if len(test_dataset) == 0:
        print("Validation dataset is empty.")
        sys.exit(1)

    # Determine which rep to visualize
    if args.rep_index is not None:
        rep_index = args.rep_index
        if rep_index < 0 or rep_index >= len(test_dataset):
            print(f"rep_index must be between 0 and {len(test_dataset) - 1}")
            sys.exit(1)
    else:
        rep_index = choose_rep_index_interactively(len(test_dataset))

    # Get the chosen sample (input incorrect rep, target correct rep, label)
    input_pose, target_pose, true_label = test_dataset[rep_index]
    metadata = getattr(test_dataset, "get_metadata", lambda idx: None)(rep_index)

    print(f"Selected rep {rep_index} | True label: {get_class_name(int(true_label.item()))}")
    if metadata:
        print(
            "Metadata:",
            f"act={metadata.get('act')},",
            f"subject={metadata.get('subject')},",
            f"raw_label={metadata.get('raw_label')},",
            f"rep={metadata.get('rep')}"
        )

    # Run inference
    print("Running inference...")
    pred_class, confidence, all_probs, corrected_pose = run_inference_single(
        model, edge_index, input_pose, device
    )

    print(
        f"Prediction: {get_class_name(pred_class)} "
        f"(conf: {confidence:.4f}) | True: {get_class_name(int(true_label.item()))}"
    )

    # Visualize original vs target-correct (ground truth)
    visualize_sequences(
        input_pose_57xT=input_pose,
        target_pose_57xT=target_pose,
        true_label=int(true_label.item()),
        pred_class=pred_class,
        confidence=confidence,
    )


if __name__ == "__main__":
    main()


