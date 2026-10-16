"""Thresholded test predictions in Kaggle format."""
from pathlib import Path

from .config import CLASSES
from .data import read_test


def save_submission(probs, path, threshold=0.5):
    """`image_id` plus one 0/1 column per class, rows in test.csv order."""
    submission = read_test()
    if len(probs) != len(submission):
        raise ValueError(f"{len(probs)} predictions for {len(submission)} test images")

    submission[CLASSES] = (probs > threshold).astype(int)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    print(f"saved {path}: {len(submission):,} rows")
    return submission
