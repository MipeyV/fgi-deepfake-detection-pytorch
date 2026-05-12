import argparse
import json
import shutil
from pathlib import Path


def prepare_dfdc_sample(input_dir: Path, output_dir: Path) -> None:
    """Prepare a sample of the DFDC dataset by copying videos into separate real and fake directories based on metadata.

    Args:
        input_dir (Path): The directory containing the original DFDC dataset with videos and metadata.json.
        output_dir (Path): The directory where the prepared sample will be stored, with 'real' and 'fake' subdirectories.
    """
    metadata_path = input_dir / "metadata.json"

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    for filename, info in metadata.items():
        source_path = input_dir / filename

        if not source_path.is_file():
            continue

        label = info["label"].lower()

        if label == "real":
            destination_path = real_dir / filename
        elif label == "fake":
            destination_path = fake_dir / filename
        else:
            raise ValueError(f"Unknown label for {filename}: {info['label']}")

        shutil.copy2(source_path, destination_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for preparing the DFDC sample.

    Returns:
        argparse.Namespace: The parsed arguments containing input and output directory paths.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dfdc_sample(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()