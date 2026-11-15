"""Single place for seeds, paths and project-wide settings."""
from pathlib import Path

SEED = 42
EXPERIMENT_SEEDS = [0, 1, 2, 3, 4]

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
WEIGHTS_DIR = ROOT / "nn" / "weights"

# Chosen by Person A after exploring OULAD (data/make_data.py make_dataset defaults).
COURSE = "BBB"
TRAIN_PRESENTATIONS: list[str] = ["2013B", "2013J", "2014B"]
TEST_PRESENTATION = "2014J"

# Planner limits
MAX_SESSIONS_PER_ASSIGNMENT = 10
N_ASSIGNMENTS = 3
