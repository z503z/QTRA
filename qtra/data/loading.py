"""Resolve dataset layouts without embedding local machine paths."""
from pathlib import Path


def qag_dataset(root, split):
    from .qag import DatasetNew
    root = Path(root)
    return DatasetNew(
        str(root / "json" / f"{split}.json"),
        str(root / "image") + "/",
        str(root / "mask") + "/",
    )


def cdvqa_dataset(root, second_root, split):
    from .cdvqa import CDVQADataset
    return CDVQADataset(root, second_root, split)
