from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DeepFakeClipDataset


def collate_deepfake_batch(batch: list[dict]) -> dict:
    """Collate deepfake samples into a batch.

    Tensor fields are stacked, while metadata fields are kept as lists.

    Args:
        batch (list[dict]): A list of samples, where each sample is a dictionary containing:
            - "frames": Tensor of shape [num_frames, C, H, W]
            - "audio": Tensor of shape [1, num_samples]
            - "label": Tensor scalar (0 for real, 1 for fake)
            - "clip_path": Path to the clip directory
            - "video_id": ID of the source video
            - "clip_id": ID of the clip within the video
    
    Returns:
        dict: A dictionary containing the collated batch with the same keys as the input samples,
    """
    return {
        "frames": torch.stack([sample["frames"] for sample in batch]),
        "audio": torch.stack([sample["audio"] for sample in batch]),
        "label": torch.stack([sample["label"] for sample in batch]),
        "clip_path": [sample["clip_path"] for sample in batch],
        "video_id": [sample["video_id"] for sample in batch],
        "clip_id": [sample["clip_id"] for sample in batch],
    }


def create_dataloader(
    manifest_path: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    frame_transform=None,
    collate_fn=collate_deepfake_batch,
) -> DataLoader:
    """Create a Dataloader for preprocessed deepfake clips.
    
    Args:
        manifest_path (str | Path): Path to the manifest CSV file.
        batch_size (int, optional): Number of samples per batch. Defaults to 8.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.
        frame_transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. Defaults to None.
        collate_fn (callable, optional): Function used to merge samples into batches.

    Returns:
        DataLoader: A PyTorch DataLoader for the DeepFakeClipDataset.
    
    Raises:
        FileNotFoundError: If the manifest file does not exist.
    
    """
    if not Path(manifest_path).is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

    dataset = DeepFakeClipDataset(
        manifest_path, 
        frame_transform=frame_transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
