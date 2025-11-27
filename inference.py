# inference.py
# Script to run inference on a single exercise rep from the validation set.
# Usage: python inference.py <rep_index>
# Example: python inference.py 0

import torch
import json
import argparse
import os
import sys

import config
from data_loader import ExerciseDataset, create_dataloaders
from model import DualBranchGCN, create_skeleton_graph
from utils import set_seed

def load_model(checkpoint_path, device):
    """Load the trained model from checkpoint."""
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
    
    return model, edge_index

def run_inference(model, edge_index, input_pose, device):
    """Run inference on a single pose sequence."""
    # input_pose shape: [num_nodes, num_frames] = [57, 100]
    # Add batch dimension: [1, 57, 100]
    input_pose = input_pose.unsqueeze(0).to(device)
    
    B = input_pose.shape[0]
    N = input_pose.shape[1]
    batch_vec = torch.arange(B, device=device).repeat_interleave(N)
    
    with torch.no_grad():
        pred_logits, pred_corrected_pose = model(
            input_pose, 
            edge_index, 
            batch_vec, 
            labels=None, 
            use_feedback=True
        )
    
    # Get predicted class and confidence
    pred_probs = torch.softmax(pred_logits, dim=1)
    pred_class = torch.argmax(pred_logits, dim=1).item()
    confidence = pred_probs[0, pred_class].item()
    
    # Remove batch dimension from corrected pose
    pred_corrected_pose = pred_corrected_pose.squeeze(0)  # [57, 100]
    
    return pred_class, confidence, pred_probs[0].cpu().numpy().tolist(), pred_corrected_pose.cpu().numpy().tolist()

def get_class_name(class_id):
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

def main():
    parser = argparse.ArgumentParser(description='Run inference on a single exercise rep')
    parser.add_argument('rep_index', type=int, help='Index of the rep in the validation set (0-based)')
    parser.add_argument('--checkpoint', type=str, default='runs/20251107-180046/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (default: inference_output_<rep_index>.json)')
    parser.add_argument('--data_path', type=str, default=config.DATA_PATH,
                       help='Path to data pickle file')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Setup device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    
    print(f"Loading model from {args.checkpoint}...")
    model, edge_index = load_model(args.checkpoint, device)
    print("Model loaded successfully.")
    
    # Load validation data
    print(f"Loading data from {args.data_path}...")
    _, test_loader = create_dataloaders(args.data_path)
    test_dataset = test_loader.dataset
    
    if args.rep_index < 0 or args.rep_index >= len(test_dataset):
        print(f"Error: rep_index must be between 0 and {len(test_dataset) - 1}")
        sys.exit(1)
    
    # Get the rep
    input_pose, target_pose, true_label = test_dataset[args.rep_index]
    print(f"Processing rep {args.rep_index} (true label: {get_class_name(true_label.item())})")
    
    # Run inference
    print("Running inference...")
    pred_class, confidence, all_probs, corrected_pose = run_inference(
        model, edge_index, input_pose, device
    )
    
    # Prepare output
    output_data = {
        "rep_index": args.rep_index,
        "true_label": {
            "id": int(true_label.item()),
            "name": get_class_name(true_label.item())
        },
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
        "input_pose": {
            "shape": list(input_pose.shape),
            "data": input_pose.cpu().numpy().tolist()
        },
        "target_pose": {
            "shape": list(target_pose.shape),
            "data": target_pose.cpu().numpy().tolist()
        },
        "corrected_pose": {
            "shape": [57, 100],
            "data": corrected_pose
        }
    }
    
    # Save to JSON
    output_path = args.output or f"inference_output_{args.rep_index}.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nInference complete!")
    print(f"True Label: {get_class_name(true_label.item())}")
    print(f"Predicted: {get_class_name(pred_class)} (confidence: {confidence:.4f})")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()


