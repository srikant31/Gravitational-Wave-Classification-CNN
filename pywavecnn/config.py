"""Central configuration for PyWaveCNN.

All paths are resolved relative to the repository root (the parent of this
package) so scripts work regardless of the current working directory they
are launched from — this fixes the old ``../data`` / ``../figures`` relative
paths that broke unless you ran ``python main.py`` from inside ``src/``.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Repository root = parent of the `pywavecnn` package directory
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"

TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
RAW_EXTRACT_DIR = DATA_DIR / "TrainingSet"
ARCHIVE_PATH = DATA_DIR / "trainingsetv1d1.tar.gz"
METADATA_CSV_PATH = DATA_DIR / "trainingset_v1d1_metadata.csv"

IMAGES_URL = "https://zenodo.org/records/1476551/files/trainingsetv1d1.tar.gz?download=1"
METADATA_URL = "https://zenodo.org/records/1476551/files/trainingset_v1d1_metadata.csv?download=1"

# Gravity Spy contour plots have a fixed border/axis frame; these bounds crop
# it down to the plot area only.
CROP_ROWS = (66, 532)
CROP_COLS = (105, 671)


@dataclass
class TrainingConfig:
    img_size: tuple = (400, 400)
    batch_size: int = 32
    epochs: int = 20
    seed: int = 220301
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    early_stopping_patience: int = 5
    early_stopping_baseline: float = 0.95
    num_workers: int = field(default_factory=lambda: max(1, __import__("os").cpu_count() - 1))

    def __post_init__(self):
        total = round(self.train_split + self.val_split + self.test_split, 6)
        if total != 1.0:
            raise ValueError(
                f"train/val/test splits must sum to 1.0, got {total}"
            )
