import argparse
from pathlib import Path

from src.data.preprocessing_pipeline import preprocess_dataset, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FGI deepfake preprocessing pipeline")

    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess real and fake videos into synchronized audio-visual clips.",
    )

    preprocess_parser.add_argument("--real-dir", type=Path, required=True)
    preprocess_parser.add_argument("--fake-dir", type=Path, required=True)
    preprocess_parser.add_argument("--output-dir", type=Path, required=True)
    preprocess_parser.add_argument("--fps", type=int, default=30)
    preprocess_parser.add_argument("--clip-size", type=int, default=30)
    preprocess_parser.add_argument("--sample-rate", type=int, default=48000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "preprocess":
        clip_paths = preprocess_dataset(
            real_dir=args.real_dir,
            fake_dir=args.fake_dir,
            output_dir=args.output_dir,
            fps=args.fps,
            clip_size=args.clip_size,
            sample_rate=args.sample_rate,
        )

        write_manifest(
            clip_paths=clip_paths,
            manifest_path=args.output_dir / "manifest.csv",
        )

        print(f"Preprocessing complete. Created {len(clip_paths)} clips.")
        print(f"Manifest written to {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()