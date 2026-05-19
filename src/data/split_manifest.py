import argparse
import hashlib
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"clip_path", "label", "video_id", "clip_id"}


def hash_video_id(video_id: str) -> int:
    """Generate a unique hash for a video ID.

    Args:
        video_id (str): The original video ID.

    Returns:
        int: A unique hash integer derived from the video ID.
    """
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()
    return int(digest, 16)


def assign_split_from_hashmod(hashmod10: int) -> str:
    """Assign a video ID to a dataset split (train, val, test) based on its hash.

    Args:
        hashmod10 (int): The hash of the video ID modulo 10.

    Returns:
        str: The assigned split ("train", "val", or "test").
    """
    if hashmod10 < 7:
        return "train"
    if hashmod10 < 9:
        return "val"
    
    return "test"

    
def validate_manifest(samples: pd.DataFrame) -> None:
    """Validate that the manifest contains the required columns.

    Args:
        samples (pd.DataFrame): The DataFrame containing the manifest data.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_columns = REQUIRED_COLUMNS - set(samples.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing_columns)}"
        )


def split_manifest(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Split a manifest CSV file into train, val, and test sets based on video IDs.

    Args:
        manifest_path (str | Path): Path to the input manifest CSV file.
        output_dir (str | Path): Directory where the split manifest files will be saved.

    Returns:
        dict[str, Path]: A dictionary mapping split names ("train", "val", "test") to their corresponding manifest file paths.
    
    Raises:
        FileNotFoundError: If the input manifest file does not exist.
        ValueError: If the manifest is missing required columns.
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")
    
    samples = pd.read_csv(
        manifest_path,
        dtype={
            "clip_path": str,
            "label": str,
            "video_id": str,
            "clip_id": str,
        },
    )

    validate_manifest(samples)

    samples = samples.copy()
    samples["video_id_hashmod10"] = samples["video_id"].map(
        lambda vid: hash_video_id(vid) % 10
    )
    samples["split"] = samples["video_id_hashmod10"].map(assign_split_from_hashmod)

    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": output_dir / "train_manifest.csv",
        "val": output_dir / "val_manifest.csv",
        "test": output_dir / "test_manifest.csv",
    }

    for split_name, split_path in split_paths.items():
        split_samples = samples[samples["split"] == split_name]
        split_samples.to_csv(split_path, index=False, encoding="utf-8")

    return split_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a manifest CSV file into train, val, and test sets.")
    parser.add_argument("--manifest-path", type=Path, required=True, help="Path to the input manifest CSV file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where the split manifest files will be saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_manifest(
        manifest_path=args.manifest_path, 
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
