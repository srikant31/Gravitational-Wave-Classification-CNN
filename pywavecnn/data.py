"""Download, extract, preprocess and split the Gravity Spy dataset.

Fixes vs. the original ``data_handler.py``:
  * ``dataset_ready()`` (was ``check_files``) actually checks the train/
    validation/test split directories contain images, instead of checking
    whether ``data/`` is non-empty — the old version always returned True
    because ``data/.gitkeep`` made the directory "non-empty" even before
    anything was downloaded, so the "skip if already prepared" logic never
    worked.
  * ``process_images_in_class`` had a bug where the ``except`` branch
    referenced an undefined ``class_folder`` variable instead of
    ``class_path``, which raised ``NameError`` and crashed the whole worker
    process the first time a corrupt image was hit.
  * Downloads use a request timeout, are verified with ``tarfile.is_tarfile``
    before extracting, and partial/corrupt downloads are removed instead of
    being silently left on disk to break the next run.
  * All paths use ``pathlib`` and the shared config module instead of
    hand-built ``'../data/' + filename`` strings.
"""

import logging
import os
import shutil
import tarfile
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import requests
from skimage import io

from . import config

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30  # seconds, applied per chunk read via requests' stream


def download_file(url: str, destination: Path) -> None:
    """Stream-download ``url`` to ``destination``, cleaning up on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            chunk_size = 8192
            downloaded = 0
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        print(
                            f"\r{destination.name}: "
                            f"{downloaded / (1024 ** 3):.2f} GB downloaded...",
                            end="",
                        )
            print()
    except requests.exceptions.RequestException as err:
        # Remove whatever partial file we wrote so a later run doesn't
        # mistake it for a complete, valid download.
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Failed to download {url}: {err}") from err


def download_dataset() -> None:
    download_file(config.IMAGES_URL, config.ARCHIVE_PATH)
    download_file(config.METADATA_URL, config.METADATA_CSV_PATH)


def dataset_ready() -> bool:
    """True if the train/validation/test split already has data in it."""
    return all(
        d.exists() and any(d.iterdir())
        for d in (config.TRAIN_DIR, config.VALID_DIR, config.TEST_DIR)
    )


def extract_archive() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not tarfile.is_tarfile(config.ARCHIVE_PATH):
        raise RuntimeError(
            f"{config.ARCHIVE_PATH} is not a valid tar.gz archive "
            "(the download may have failed or been interrupted)."
        )

    logger.info("Extracting %s", config.ARCHIVE_PATH.name)
    with tarfile.open(config.ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(path=config.DATA_DIR)
    logger.info("Extracted to %s", config.DATA_DIR)


def split_dataset(
    train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1, seed: int = 220301
) -> None:
    source_folder = config.RAW_EXTRACT_DIR

    classes = [p for p in source_folder.iterdir() if p.is_dir()]
    rng = np.random.default_rng(seed)

    for cls_folder in classes:
        files = os.listdir(cls_folder)
        rng.shuffle(files)

        train_end = int(len(files) * train_size)
        val_end = train_end + int(len(files) * val_size)

        splits = {
            "train": files[:train_end],
            "validation": files[train_end:val_end],
            "test": files[val_end:],
        }

        for split_name, split_files in splits.items():
            dest_folder = config.DATA_DIR / split_name / cls_folder.name
            dest_folder.mkdir(parents=True, exist_ok=True)
            for filename in split_files:
                shutil.move(str(cls_folder / filename), str(dest_folder / filename))


def clear_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def process_images_in_class(class_path: Path, row_bounds: tuple, col_bounds: tuple) -> None:
    """Crop every image in ``class_path`` to the plot area, dropping unreadable files."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for image_file in os.listdir(class_path):
            image_path = class_path / image_file

            # Ignore macOS AppleDouble metadata files
            if image_file.startswith("._"):
                image_path.unlink(missing_ok=True)
                continue

            if not image_path.is_file() or image_path.suffix.lower() not in (
                ".png",
                ".jpg",
                ".jpeg",
            ):
                continue

            try:
                image_data = io.imread(image_path)
                image_data = image_data[
                    row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1], :3
                ]
                io.imsave(image_path, image_data)
            except Exception as e:  # noqa: BLE001 - corrupt/unreadable image, skip it
                logger.warning("Dropping unreadable image %s (%s)", image_path, e)
                image_path.unlink(missing_ok=True)


def process_images_in_classes(num_workers: int = 4) -> None:
    """Crop every class's images in parallel."""
    class_paths = [p for p in config.RAW_EXTRACT_DIR.iterdir() if p.is_dir()]

    with Pool(num_workers) as pool:
        pool.starmap(
            process_images_in_class,
            [(class_path, config.CROP_ROWS, config.CROP_COLS) for class_path in class_paths],
        )


def prepare_dataset(cfg: "config.TrainingConfig", force: bool = False) -> None:
    """Download, extract, crop and split the dataset if it isn't ready yet.

    Set ``force=True`` to wipe and re-download even if a prepared split
    already exists.
    """
    if dataset_ready() and not force:
        logger.info("Dataset already prepared in %s, skipping download.", config.DATA_DIR)
        return

    clear_path(config.METADATA_CSV_PATH)
    for split_dir in (config.TRAIN_DIR, config.VALID_DIR, config.TEST_DIR, config.RAW_EXTRACT_DIR):
        clear_path(split_dir)

    download_dataset()
    extract_archive()
    process_images_in_classes(cfg.num_workers)
    split_dataset(cfg.train_split, cfg.val_split, cfg.test_split, cfg.seed)

    clear_path(config.RAW_EXTRACT_DIR)
    clear_path(config.ARCHIVE_PATH)
