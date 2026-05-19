import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float tensor with shape [C, H, W]."""
    image_array = np.array(image, copy=True)
    return torch.from_numpy(image_array).permute(2, 0, 1).float().div(255)


class DeepFakeClipDataset(Dataset):
    def __init__(self, manifest_path: str | Path, frame_transform=None):
        self.manifest_path = Path(manifest_path)
        self.samples = pd.read_csv(
            self.manifest_path,
            dtype={
                "clip_path": str,
                "label": str,
                "video_id": str,
                "clip_id": str,
            },
        )
        self.frame_transform = frame_transform or pil_to_tensor
        self.label_to_idx = {"real": 0, "fake": 1}
        self._validate_manifest()


    def _validate_manifest(self) -> None:
        """Validate that the manifest contains the required columns and valid labels.

        Raises:
            ValueError: If required columns are missing or if labels are invalid.
        """
        required_columns = {"clip_path", "label", "video_id", "clip_id"}
        missing_columns = required_columns - set(self.samples.columns)

        if missing_columns:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing_columns)}"
            )

        valid_labels = set(self.label_to_idx)
        invalid_labels = set(self.samples["label"]) - valid_labels

        if invalid_labels:
            raise ValueError(f"Manifest contains invalid labels: {sorted(invalid_labels)}")


    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)
    

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing frames, label index, clip path, video ID, and clip ID.
        """
        sample = self.samples.iloc[idx]

        clip_path = Path(sample["clip_path"])
        label = self.label_to_idx[sample["label"]]
        frames = self._load_frames(clip_path)

        return {
            "frames": frames,
            "clip_path": clip_path,
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": sample["video_id"],
            "clip_id": sample["clip_id"],
        }
    

    def _load_frames(self, clip_path: str | Path) -> torch.Tensor:
        """Load video frames from the given clip path.

        Args:
            clip_path (str): The path to the video clip.

        Returns:
            torch.Tensor: A tensor containing the video frames.
        
        Raises:
            FileNotFoundError: If the clip path does not exist or if no frames are found.
        """
        clip_path = Path(clip_path)

        if not clip_path.is_dir():
            raise FileNotFoundError(f"Clip directory does not exist: {clip_path}")
        
        frame_paths = sorted(clip_path.glob("*.jpg"))

        if not frame_paths:
            raise FileNotFoundError(f"No frames found in clip directory: {clip_path}")
        
        frames = []

        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                frame = image.convert("RGB")

            frame = self.frame_transform(frame)
            frames.append(frame)

        return torch.stack(frames)
