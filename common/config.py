"""Single place for seeds, paths and project-wide settings."""
from pathlib import Path

SEED = 42
EXPERIMENT_SEEDS = [0, 1, 2, 3, 4]

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
WEIGHTS_DIR = ROOT / "nn" / "weights"

# Chosen in Week 1 after exploring OULAD (Person A). Placeholder until then.
COURSE = "TODO"
TRAIN_PRESENTATIONS: list[str] = []   # earlier presentations, e.g. ["2013J", "2014B"]
TEST_PRESENTATION = "TODO"            # later presentation, e.g. "2014J"

# Planner limits
MAX_SESSIONS_PER_ASSIGNMENT = 10
N_ASSIGNMENTS = 3
