"""Paths and constants shared by all notebooks."""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]  # Homework 2 folder
DATA = ROOT / "data" / "kaggle"  # Kaggle zip as is, read-only
PROCESSED = ROOT / "data" / "processed"  # generated: folds, caches
SUBMISSIONS = ROOT / "data" / "submissions"  # CSVs for Kaggle upload
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"
CHECKPOINTS = ROOT / "checkpoints"

CLASSES = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
VIP = ["Moris", "Motya", "Biatrix"]  # individual animals, rare classes
N_FOLDS = 5
SEED = 95431

IMG_SIZE = 224
MEAN = STD = (0.5, 0.5, 0.5)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
