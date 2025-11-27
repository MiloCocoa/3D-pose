"""
export_pose_examples.py

Utility script to extract pose sequences specifically from the
validation split (subject set used for testing) into a JSON file that is
easy to copy-paste into the API (`api.py`) or other tools.

Each exported example contains:
- Basic metadata (act, subject, label, repetition, frame_count)
- Pose data as a list of frames, each frame is a list of 57 floats
  (19 joints * 3 coordinates), shape: [num_frames, 57]
- A `transpose` flag compatible with the FastAPI input schema

Usage:
    python export_pose_examples.py

Optional arguments:
    --output pose_examples.json      # change output path
    --max_examples 10                # limit how many sequences to export
"""

import argparse
import json
from typing import Any, Dict, List

import config
from data_loader import create_dataloaders


def safe_int(value: Any) -> Any:
    """Attempt to cast to int; if not possible, return original value."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return value


def label_name_from_id(lab: int) -> str:
    """Simple mapping of label id to a human-readable name."""
    if lab == 1:
        return "Correct"
    return f"Mistake_raw_label_{lab}"


def export_examples(
    data_path: str,
    output_path: str,
    max_examples: int,
) -> None:
    """
    Main export routine.

    - Loads raw data
    - Groups into sequences
    - Extracts up to `max_examples` sequences
    - Saves to JSON
    """
    print("Loading validation split via data_loader.create_dataloaders()...")
    _, test_loader = create_dataloaders(data_path, batch_size=1)
    test_dataset = test_loader.dataset

    if len(test_dataset) == 0:
        raise RuntimeError("Validation dataset is empty. Nothing to export.")

    limit = min(max_examples, len(test_dataset))
    print(f"Exporting {limit} validation sequence(s) to JSON...")

    examples: List[Dict[str, Any]] = []
    for idx in range(limit):
        input_pose, _, label = test_dataset[idx]
        metadata = (
            test_dataset.get_metadata(idx)
            if hasattr(test_dataset, "get_metadata")
            else {}
        )

        pose_np = input_pose.detach().cpu().numpy().T  # [num_frames, 57]

        example_meta: Dict[str, Any] = {
            "act": metadata.get("act"),
            "subject": safe_int(metadata.get("subject")),
            "raw_label": safe_int(metadata.get("raw_label")),
            "zero_index_label": safe_int(metadata.get("zero_index_label")),
            "rep": safe_int(metadata.get("rep")),
            "frame_count": pose_np.shape[0],
        }

        if example_meta["raw_label"] is not None:
            example_meta["raw_label_name"] = label_name_from_id(
                int(example_meta["raw_label"])
            )

        example = {
            "example_index": idx,
            "meta": example_meta,
            "pose": pose_np.tolist(),
            "transpose": False,
            "label_index": int(label.item()),
        }
        examples.append(example)

    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Export complete. Saved {len(examples)} examples to '{output_path}'.")
    print("")
    print("You can now open this JSON file and copy-paste any `pose` list")
    print("directly into the `/analyze` endpoint body for testing:")
    print("")
    print('  { "pose": <PASTE_POSE_LIST_HERE>, "transpose": false }')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export example pose sequences from data_3D.pickle to JSON."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=config.DATA_PATH,
        help="Path to data pickle file (default: from config.DATA_PATH).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pose_examples.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=10,
        help="Maximum number of sequences to export.",
    )

    args = parser.parse_args()
    export_examples(args.data_path, args.output, args.max_examples)


if __name__ == "__main__":
    main()


