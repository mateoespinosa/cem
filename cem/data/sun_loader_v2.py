"""
Minimal SUN (Xian) dataloader v2.

This module intentionally mirrors the simple notebook pipeline:
- Load Xian SUN ResNet-101 features from res101.mat
- Load labels and class attributes from att_splits.mat
- Build per-sample concept targets via class lookup
- Stratified 80/10/10 split over all samples
- Return DataLoaders that emit (x, y, c)
"""

import os
import urllib.request
import zipfile

import numpy as np
import torch
from pytorch_lightning import seed_everything
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


DATASET_DIR = os.environ.get("DATASET_DIR", "data/SUN/")
DEFAULT_NUM_ATTRIBUTES = 102
DEFAULT_XIAN_URL = "http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"
DEFAULT_XIAN_DIR = "xlsa17/data/SUN"
DEFAULT_XIAN_ARCHIVE = "xlsa17.zip"


def _resolve_path(root_dir, path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(root_dir, path)


def _download_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def _ensure_xian_files(root_dir, config):
    xian_dir = _resolve_path(root_dir, config.get("xian_sun_dir", DEFAULT_XIAN_DIR))
    res101_path = os.path.join(xian_dir, "res101.mat")
    splits_path = os.path.join(xian_dir, "att_splits.mat")

    if os.path.exists(res101_path) and os.path.exists(splits_path):
        return res101_path, splits_path

    if not config.get("download_xian_sun_if_missing", True):
        raise ValueError(
            "Missing Xian SUN files. Set download_xian_sun_if_missing=true "
            "or provide xian_sun_dir with res101.mat and att_splits.mat."
        )

    archive_path = _resolve_path(
        root_dir,
        config.get("xian_sun_archive", DEFAULT_XIAN_ARCHIVE),
    )
    xian_url = config.get("xian_sun_url", DEFAULT_XIAN_URL)

    if not os.path.exists(archive_path):
        print(f"Downloading Xian SUN archive from {xian_url}...")
        _download_file(xian_url, archive_path)

    print(f"Extracting Xian SUN archive at {archive_path}...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(root_dir)

    if not (os.path.exists(res101_path) and os.path.exists(splits_path)):
        raise ValueError(
            "Could not find res101.mat/att_splits.mat after extracting xlsa17.zip."
        )

    return res101_path, splits_path


def _mat_cell_to_str(cell):
    arr = np.array(cell)
    while isinstance(arr, np.ndarray) and arr.dtype == object and arr.size > 0:
        arr = arr.reshape(-1)[0]
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        if arr.size == 0:
            return ""
        if arr.dtype.kind in ["U", "S"]:
            return "".join(arr.reshape(-1).astype(str)).strip()
        if arr.dtype.kind in ["i", "u"]:
            return "".join(chr(int(x)) for x in arr.reshape(-1)).strip()
    return str(arr).strip()


def _parse_attribute_names(splits):
    if "attributes" not in splits:
        return [f"attr_{i}" for i in range(DEFAULT_NUM_ATTRIBUTES)]
    raw = np.array(splits["attributes"]).reshape(-1)
    names = [_mat_cell_to_str(cell) for cell in raw]
    names = [name if name else f"attr_{i}" for i, name in enumerate(names)]
    return names


def _build_tensors(root_dir, config, seed):
    res101_path, splits_path = _ensure_xian_files(root_dir, config)
    res = loadmat(res101_path)
    splits = loadmat(splits_path)

    features = np.array(res["features"], dtype=np.float32)
    if features.shape[0] == 2048 and features.shape[1] != 2048:
        features = features.T
    if features.shape[1] != 2048:
        raise ValueError(f"Expected 2048-d feature vectors, got shape {features.shape}.")

    labels = np.array(res["labels"]).reshape(-1).astype(np.int64)
    if labels.min() == 1:
        labels = labels - 1

    attributes = np.array(splits["att"], dtype=np.float32).T

    # Normalize each concept dimension to [0, 1], matching notebook logic.
    attr_min = attributes.min(axis=0, keepdims=True)
    attr_max = attributes.max(axis=0, keepdims=True)
    attributes_norm = (attributes - attr_min) / (attr_max - attr_min + 1e-8)

    concept_targets = attributes_norm[labels]

    val_fraction = float(config.get("xian_val_fraction", 0.1))
    test_fraction = float(config.get("xian_test_fraction", 0.1))
    if val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("xian_val_fraction and xian_test_fraction must both be > 0.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("xian_val_fraction + xian_test_fraction must be < 1.")

    indices = np.arange(features.shape[0])
    holdout_fraction = val_fraction + test_fraction

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=holdout_fraction,
        stratify=labels,
        random_state=seed,
    )

    test_size_within_temp = test_fraction / holdout_fraction
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_size_within_temp,
        stratify=labels[temp_idx],
        random_state=seed,
    )

    print(
        "SUN v2 Xian stats: "
        f"samples={features.shape[0]} classes={int(labels.max()) + 1} concepts={attributes.shape[1]}"
    )
    print(
        "SUN v2 splits: "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )

    split_map = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    tensors = {}
    for split, idx in split_map.items():
        x = torch.from_numpy(features[idx]).float()
        y = torch.from_numpy(labels[idx]).long()
        c = torch.from_numpy(concept_targets[idx]).float()
        tensors[split] = (x, y, c)

    class_names = []
    if "allclasses_names" in splits:
        class_names = [_mat_cell_to_str(cell) for cell in np.array(splits["allclasses_names"]).reshape(-1)]
    if not class_names:
        class_names = [f"class_{i}" for i in range(int(labels.max()) + 1)]

    attribute_names = _parse_attribute_names(splits)
    return tensors, class_names, attribute_names


def load_data(
    split,
    batch_size,
    root_dir=DATASET_DIR,
    num_workers=1,
    dataset_transform=lambda x: x,
    augment_data=False,
    seed=42,
    config=None,
    dataset_size=None,
    image_size=224,
    concept_transform=None,
    additional_sample_transform=None,
):
    del augment_data, image_size, concept_transform, additional_sample_transform

    if config is None:
        config = {}

    tensors, class_names, _ = _build_tensors(root_dir, config, seed)
    x, y, c = tensors[split]

    if dataset_size is not None:
        if 0 < dataset_size < 1:
            dataset_size = int(np.ceil(len(x) * dataset_size))
        dataset_size = int(min(len(x), dataset_size))
        selected = torch.randperm(len(x))[:dataset_size]
        x = x[selected]
        y = y[selected]
        c = c[selected]

    ds = TensorDataset(x, y, c)
    ds.class_names = class_names
    ds = dataset_transform(ds)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        num_workers=num_workers,
    )


def get_num_labels(*args, **kwargs):
    return int(kwargs.get("n_tasks", 717))


def get_num_attributes(*args, **kwargs):
    return int(kwargs.get("n_concepts", DEFAULT_NUM_ATTRIBUTES))


def _compute_imbalance(train_dl, n_concepts):
    positives = np.zeros((n_concepts,), dtype=np.float64)
    total = 0
    for _, _, concepts in train_dl:
        c = concepts.detach().cpu().numpy()
        positives += np.sum(c, axis=0)
        total += c.shape[0]
    if total == 0:
        return np.zeros((n_concepts,), dtype=np.float32)
    positives = np.clip(positives, 1e-8, total - 1e-8)
    return (total / positives) - 1


def generate_data(
    config,
    root_dir=DATASET_DIR,
    seed=42,
    output_dataset_vars=False,
    rerun=False,
    dataset_transform=lambda x: x,
    training_transform=None,
    train_sample_transform=None,
    test_sample_transform=None,
    val_sample_transform=None,
):
    del rerun, train_sample_transform, test_sample_transform, val_sample_transform

    if root_dir is None:
        root_dir = DATASET_DIR
    seed_everything(seed)

    training_transform = training_transform if training_transform is not None else dataset_transform

    batch_size = config.get("batch_size", 256)
    num_workers = config.get("num_workers", 8)
    dataset_size = config.get("dataset_size", None)

    dataset_opts = {
        "xian_sun_url": config.get("xian_sun_url", DEFAULT_XIAN_URL),
        "xian_sun_dir": config.get("xian_sun_dir", DEFAULT_XIAN_DIR),
        "xian_sun_archive": config.get("xian_sun_archive", DEFAULT_XIAN_ARCHIVE),
        "download_xian_sun_if_missing": config.get("download_xian_sun_if_missing", True),
        "xian_val_fraction": config.get("xian_val_fraction", 0.1),
        "xian_test_fraction": config.get("xian_test_fraction", 0.1),
    }

    # Build once so all splits share exact same tensors/indices for this seed.
    tensors, class_names, attribute_names = _build_tensors(root_dir, dataset_opts, seed)

    def _make_dl(split, ds_transform):
        x, y, c = tensors[split]
        if dataset_size is not None:
            if 0 < dataset_size < 1:
                ds_count = int(np.ceil(len(x) * dataset_size))
            else:
                ds_count = int(dataset_size)
            ds_count = int(min(len(x), ds_count))
            selected = torch.randperm(len(x))[:ds_count]
            x_local = x[selected]
            y_local = y[selected]
            c_local = c[selected]
        else:
            x_local, y_local, c_local = x, y, c

        ds = TensorDataset(x_local, y_local, c_local)
        ds.class_names = class_names
        ds = ds_transform(ds)

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            num_workers=num_workers,
        )

    train_dl = _make_dl("train", training_transform)
    val_dl = _make_dl("val", dataset_transform)
    test_dl = _make_dl("test", dataset_transform)

    n_concepts = tensors["train"][2].shape[1]
    n_tasks = len(class_names)
    concept_group_map = {i: [i] for i in range(n_concepts)}

    if config.get("weight_loss", False):
        imbalance = _compute_imbalance(train_dl, n_concepts)
    else:
        imbalance = None

    if not output_dataset_vars:
        return train_dl, val_dl, test_dl

    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        (n_concepts, n_tasks, concept_group_map),
    )


def get_concept_descriptions(
    config,
    root_dir=DATASET_DIR,
    seed=42,
    rerun=False,
):
    del rerun
    _, _, attribute_names = _build_tensors(root_dir, config or {}, seed)
    return attribute_names
