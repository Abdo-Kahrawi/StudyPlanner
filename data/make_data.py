"""Download and preprocess OULAD reproducibly. Owner: Person A (Week 2).

Usage:
    python -m data.make_data            # download + build data/processed/features.parquet

Steps to implement:
    1. Download the figshare CSV zip into data/raw/ (no registration), verify checksum.
    2. Filter studentVle.csv in chunks (pd.read_csv(chunksize=...)) to common.config.COURSE.
    3. Join vle.csv activity_type -> 3 ACTIVITY_GROUPS; aggregate to weekly totals.
    4. Build one row per (student, assessment) with schema.FEATURES and TARGET.
    5. Split by presentation: TRAIN_PRESENTATIONS vs TEST_PRESENTATION.
    6. Fix all randomness with common.config.SEED.
"""
from common import config


def download() -> None:
    raise NotImplementedError("Person A, Week 2")


def build_features() -> None:
    raise NotImplementedError("Person A, Week 2")


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    download()
    build_features()


if __name__ == "__main__":
    main()
