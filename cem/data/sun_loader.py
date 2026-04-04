"""
Dataloader for SUN dataset variants using per-class summary attributes.

This loader expects image-level class labels and class-level attribute signatures
(102 attributes by default, following common SUN attribute setups). When the
image tree is missing, it can bootstrap SUN397 from Hugging Face into the
configured root directory after confirmation.
"""

import os
import sys
import shutil
import io
import struct
import time
import tarfile
import tempfile
import urllib.request
import zipfile
import zlib
from contextlib import contextmanager
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
try:
    from datasets import load_dataset, Image as HFDatasetsImage
    from datasets.download.download_config import DownloadConfig
except ImportError:  # pragma: no cover - optional dependency path
    load_dataset = None
    HFDatasetsImage = None
    DownloadConfig = None
from pytorch_lightning import seed_everything
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, Subset


########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

# CAN BE OVERWRITTEN WITH AN ENV VARIABLE DATASET_DIR
DATASET_DIR = os.environ.get("DATASET_DIR", "data/SUN/")
DEFAULT_NUM_ATTRIBUTES = 102
DEFAULT_HF_DATASET_ID = "tanganke/sun397"
DEFAULT_HF_CACHE_DIR = ".hf_cache"
DOWNLOAD_COMPLETE_MARKER = ".sun397_download_complete"
DOWNLOAD_LOCK_FILENAME = ".sun397_download.lock"
SUN_ATTRIBUTE_DB_URL = "https://cs.brown.edu/people/gmpatter/Attributes/SUNAttributeDB.tar.gz"
SUN_ATTRIBUTE_DB_ARCHIVE = "SUNAttributeDB.tar.gz"
SUN_ATTRIBUTE_DB_DIR = "SUNAttributeDB"
SUN_ATTRIBUTE_DB_IMAGES_URL = "https://cs.brown.edu/people/gmpatter/Attributes/SUNAttributeDB_Images.tar.gz"
SUN_ATTRIBUTE_DB_IMAGES_ARCHIVE = "SUNAttributeDB_Images.tar.gz"
SUN_ATTRIBUTE_DB_SPLIT_PREFIX = "sun_attribute_db"
XIAN_SUN_ZIP_URL = "http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"
XIAN_SUN_DATA_DIR = "xlsa17/data/SUN"
XIAN_SUN_RES101_MEMBER = "xlsa17/data/SUN/res101.mat"
XIAN_SUN_SPLITS_MEMBER = "xlsa17/data/SUN/att_splits.mat"
XIAN_SUN_SPLIT_PREFIX = "xian_sun"


MANUAL_SUN_CLASS_FALLBACKS = {
    "ice cream parlor": "parlor",
    "volleyball court indoor": "volleyball court/outdoor",
}


########################################################
## HELPERS
########################################################


def _normalize_class_name(name):
    return name.replace("\\", "/").strip().strip("/").lower()


def _canonical_xian_split_strategy(split_strategy):
    strategy = str(split_strategy).strip().lower()
    if strategy in ["official", "benchmark"]:
        return "official"
    if strategy in ["original", "original_sizes", "stratified", "standard"]:
        return "original_sizes"
    if strategy in ["trainval_test", "trainval"]:
        return "trainval_test"
    raise ValueError(
        f"Unsupported xian_split_strategy '{split_strategy}'."
    )


def _canonical_fraction_tag(value):
    # Keep file-name-safe, deterministic tags for cache keys.
    return str(float(value)).replace(".", "p")


def _get_xian_split_fractions(config):
    val_fraction = float(
        config.get(
            "xian_val_fraction",
            0.1,
        )
    )
    test_fraction = float(
        config.get(
            "xian_test_fraction",
            0.1,
        )
    )
    return val_fraction, test_fraction


def _first_existing_file(root_dir, candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_path = candidate
        if not os.path.isabs(candidate_path):
            candidate_path = os.path.join(root_dir, candidate_path)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def _resolve_path(root_dir, path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(root_dir, path)


def _download_file(url, destination):
    with urllib.request.urlopen(url, timeout=120) as response:
        with open(destination, "wb") as out_file:
            shutil.copyfileobj(response, out_file)


def _normalize_label_name(name):
    return name.replace("\\", "/").replace("_", " ").replace("/", " ").strip().lower()


def _mat_cell_to_str(cell):
    arr = np.array(cell).reshape(-1)
    if len(arr) == 0:
        return ""
    return str(arr[0]).strip()


def _extract_class_from_sun_attribute_image_path(image_rel_path):
    image_rel_path = image_rel_path.replace("\\", "/").strip("/")
    parts = image_rel_path.split("/")
    if len(parts) < 3:
        return ""
    # Paths use format like a/abbey/image.jpg or p/poolroom/home/image.jpg.
    return "/".join(parts[1:-1]).strip()


def _get_target_sun_class_names(root_dir, config):
    class_names_file = _first_existing_file(
        root_dir,
        [
            config.get("class_names_file", None),
            "sun_classes.txt",
            "classes.txt",
        ],
    )
    if class_names_file is not None:
        return _parse_classes_file(class_names_file)

    images_root = _find_existing_image_root(root_dir, config)
    if images_root is not None:
        class_to_paths = _collect_image_paths_by_class(images_root)
        if class_to_paths:
            return sorted(class_to_paths.keys())
    return None


def _best_label_match(target_name, candidate_names):
    target_tokens = set(_normalize_label_name(target_name).split())
    if not target_tokens:
        return None

    best_name = None
    best_score = -1.0
    for candidate in candidate_names:
        cand_tokens = set(_normalize_label_name(candidate).split())
        if not cand_tokens:
            continue
        overlap = len(target_tokens.intersection(cand_tokens))
        score = overlap / max(1, len(target_tokens.union(cand_tokens)))
        if score > best_score:
            best_score = score
            best_name = candidate

    if best_score < 0.5:
        return None
    return best_name


def _build_sun_attribute_class_matrix(class_names, attr_db_classes, attr_db_labels):
    class_to_rows = {}
    for idx, class_name in enumerate(attr_db_classes):
        class_to_rows.setdefault(class_name, []).append(idx)

    class_to_mean = {
        class_name: np.mean(attr_db_labels[row_indices, :], axis=0)
        for class_name, row_indices in class_to_rows.items()
    }

    normalized_lookup = {
        _normalize_label_name(class_name): class_name
        for class_name in class_to_mean
    }

    class_matrix = np.zeros((len(class_names), DEFAULT_NUM_ATTRIBUTES), dtype=np.float32)
    missing = []
    for idx, target_class in enumerate(class_names):
        normalized_target = _normalize_label_name(target_class)
        source_class = normalized_lookup.get(normalized_target)

        if source_class is None:
            manual_source = MANUAL_SUN_CLASS_FALLBACKS.get(normalized_target)
            if manual_source is not None:
                source_class = normalized_lookup.get(_normalize_label_name(manual_source))

        if source_class is None:
            source_class = _best_label_match(target_class, list(class_to_mean.keys()))

        if source_class is None:
            missing.append(target_class)
            continue

        class_matrix[idx, :] = class_to_mean[source_class].astype(np.float32)

    if missing:
        raise ValueError(
            "Could not align SUN Attribute Database classes for: "
            f"{missing[:10]}"
        )

    return class_matrix


def _ensure_sun_attribute_db_dir(root_dir, config):
    attr_db_dir = _first_existing_file(
        root_dir,
        [
            config.get("sun_attribute_db_dir", None),
            SUN_ATTRIBUTE_DB_DIR,
        ],
    )
    if attr_db_dir is not None:
        return attr_db_dir

    if not config.get("download_attribute_db_if_missing", True):
        return None

    archive_path = _resolve_path(
        root_dir,
        config.get("sun_attribute_db_archive", SUN_ATTRIBUTE_DB_ARCHIVE),
    )
    attr_db_url = config.get("sun_attribute_db_url", SUN_ATTRIBUTE_DB_URL)

    os.makedirs(root_dir, exist_ok=True)
    if not os.path.exists(archive_path):
        _download_file(attr_db_url, archive_path)

    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(root_dir)

    extracted_dir = os.path.join(root_dir, SUN_ATTRIBUTE_DB_DIR)
    if not os.path.isdir(extracted_dir):
        return None
    return extracted_dir


def _load_sun_attribute_db_annotation_data(root_dir, config):
    attr_db_dir = _ensure_sun_attribute_db_dir(root_dir, config)
    if attr_db_dir is None:
        raise ValueError(
            "SUN Attribute DB directory was not found. Set `sun_attribute_db_dir` "
            "or enable `download_attribute_db_if_missing`."
        )

    attrs_mat_path = os.path.join(attr_db_dir, "attributes.mat")
    images_mat_path = os.path.join(attr_db_dir, "images.mat")
    labels_mat_path = os.path.join(attr_db_dir, "attributeLabels_continuous.mat")

    if not (os.path.exists(attrs_mat_path) and os.path.exists(images_mat_path) and os.path.exists(labels_mat_path)):
        raise ValueError(
            "SUN Attribute Database files are incomplete. Expected attributes.mat, "
            "images.mat, and attributeLabels_continuous.mat."
        )

    attrs_data = loadmat(attrs_mat_path)
    images_data = loadmat(images_mat_path)
    labels_data = loadmat(labels_mat_path)

    attr_names_arr = np.array(attrs_data.get("attributes"))
    image_paths_arr = np.array(images_data.get("images"))
    attr_labels = np.array(labels_data.get("labels_cv"), dtype=np.float32)

    if attr_names_arr.size == 0 or image_paths_arr.size == 0 or attr_labels.size == 0:
        raise ValueError("SUN Attribute Database MAT files were empty.")

    attr_names = [_mat_cell_to_str(cell) for cell in attr_names_arr.reshape(-1)]
    image_paths = [_mat_cell_to_str(cell) for cell in image_paths_arr.reshape(-1)]
    attr_db_classes = [_extract_class_from_sun_attribute_image_path(p) for p in image_paths]
    return attr_names, image_paths, attr_db_classes, attr_labels


def _get_sun_attribute_db_image_roots(root_dir, config):
    roots = []
    configured_images_dir = config.get("images_dir", None)
    if configured_images_dir is not None:
        roots.append(_resolve_path(root_dir, configured_images_dir))
    roots.append(os.path.join(root_dir, "SUN397"))
    roots.append(os.path.join(root_dir, "SUNAttributeDB_Images"))
    # Brown archive extracts to 'images/' directory
    roots.append(os.path.join(root_dir, "images"))
    roots.append(root_dir)

    deduped = []
    seen = set()
    for candidate in roots:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate):
            deduped.append(candidate)
    return deduped


def _resolve_sun_attribute_db_image_path(image_rel_path, image_roots):
    rel_norm = image_rel_path.replace("\\", "/").strip("/")
    rel_candidates = [rel_norm]
    if rel_norm.startswith("SUN397/"):
        rel_candidates.append(rel_norm[len("SUN397/"):])

    for base_dir in image_roots:
        for rel_candidate in rel_candidates:
            abs_path = os.path.join(base_dir, rel_candidate)
            if os.path.exists(abs_path):
                return os.path.abspath(abs_path)
    return None


def _has_enough_sun_attribute_db_image_matches(image_paths, image_roots, min_matches=25):
    matches = 0
    checks = min(len(image_paths), 300)
    for rel_path in image_paths[:checks]:
        if _resolve_sun_attribute_db_image_path(rel_path, image_roots) is not None:
            matches += 1
            if matches >= min_matches:
                return True
    return False


def _maybe_bootstrap_sun_attribute_db_images(root_dir, config, image_paths):
    if not config.get("download_attribute_db_images_if_missing", True):
        return

    image_roots = _get_sun_attribute_db_image_roots(root_dir, config)
    if _has_enough_sun_attribute_db_image_matches(image_paths, image_roots):
        return

    archive_path = _resolve_path(
        root_dir,
        config.get("sun_attribute_db_images_archive", SUN_ATTRIBUTE_DB_IMAGES_ARCHIVE),
    )
    image_url = config.get("sun_attribute_db_images_url", SUN_ATTRIBUTE_DB_IMAGES_URL)

    os.makedirs(root_dir, exist_ok=True)
    if not os.path.exists(archive_path):
        _download_file(image_url, archive_path)

    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(root_dir)


def _build_sun_attribute_db_split_payloads(root_dir, config, seed):
    _, image_paths, attr_db_classes, _ = _load_sun_attribute_db_annotation_data(
        root_dir,
        config,
    )

    _maybe_bootstrap_sun_attribute_db_images(root_dir, config, image_paths)

    image_roots = _get_sun_attribute_db_image_roots(root_dir, config)
    class_to_paths = {}
    missing = []
    for rel_path, class_name in zip(image_paths, attr_db_classes):
        if not class_name:
            continue
        abs_path = _resolve_sun_attribute_db_image_path(rel_path, image_roots)
        if abs_path is None:
            missing.append(rel_path)
            continue
        class_to_paths.setdefault(class_name, []).append(abs_path)

    if not class_to_paths:
        raise ValueError(
            "No SUN Attribute DB images were found locally. Place SUN397 images "
            "under `images_dir` (or root_dir/SUN397), matching paths listed in "
            "SUNAttributeDB/images.mat."
        )

    if missing:
        raise ValueError(
            "SUN Attribute DB mode requires the exact annotated image paths. "
            f"Missing {len(missing)} images; examples: {missing[:5]}"
        )

    (
        class_names,
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
    ) = _stratified_train_val_test_split(
        class_to_paths,
        seed=seed,
        val_fraction=config.get("val_fraction", 0.1),
        test_fraction=config.get("test_fraction", 0.1),
    )

    split_payloads = {
        "train": {"paths": train_paths, "labels": train_labels},
        "val": {"paths": val_paths, "labels": val_labels},
        "test": {"paths": test_paths, "labels": test_labels},
    }
    return class_names, split_payloads


def _ensure_sun_attribute_db_concepts(root_dir, config, class_names=None):
    if not config.get("use_sun_attribute_database", True):
        return

    target_attr_names = _resolve_path(
        root_dir,
        config.get("attribute_names_file", "sun_attributes.txt"),
    )
    target_attr_matrix = _resolve_path(
        root_dir,
        config.get("class_attributes_file", "sun_attrs_per_class_binary_0.npy"),
    )
    if os.path.exists(target_attr_names) and os.path.exists(target_attr_matrix):
        return

    target_attr_names_dir = os.path.dirname(target_attr_names)
    target_attr_matrix_dir = os.path.dirname(target_attr_matrix)
    if target_attr_names_dir:
        os.makedirs(target_attr_names_dir, exist_ok=True)
    if target_attr_matrix_dir:
        os.makedirs(target_attr_matrix_dir, exist_ok=True)

    attr_names, _, attr_db_classes, attr_labels = _load_sun_attribute_db_annotation_data(
        root_dir,
        config,
    )

    if class_names is None:
        if config.get("use_sun_attribute_db_dataset", False):
            class_names = sorted(set(attr_db_classes))
        else:
            class_names = _get_target_sun_class_names(root_dir, config)
    if class_names is None:
        return

    class_matrix = _build_sun_attribute_class_matrix(
        class_names,
        attr_db_classes,
        attr_labels,
    )

    with open(target_attr_names, "w") as f:
        for name in attr_names[:DEFAULT_NUM_ATTRIBUTES]:
            f.write(name + "\n")

    np.save(target_attr_matrix, class_matrix[:, :DEFAULT_NUM_ATTRIBUTES])


def _load_mat_from_bytes(raw_bytes):
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp_file:
        tmp_file.write(raw_bytes)
        tmp_path = tmp_file.name
    try:
        return loadmat(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _download_remote_zip_member(url, member_name, destination, tail_bytes=16 * 1024 * 1024):
    head_request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head_request, timeout=120) as response:
        content_length = int(response.headers.get("Content-Length"))

    tail_sizes = [tail_bytes, tail_bytes * 2, tail_bytes * 4]
    for current_tail_bytes in tail_sizes:
        current_tail_bytes = min(current_tail_bytes, content_length)
        tail_start = max(0, content_length - current_tail_bytes)
        tail_request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={tail_start}-"},
        )
        with urllib.request.urlopen(tail_request, timeout=300) as response:
            tail_data = response.read()

        try:
            archive = zipfile.ZipFile(io.BytesIO(tail_data))
            info = archive.getinfo(member_name)
        except Exception:
            continue

        header_offset = tail_start + info.header_offset
        header_request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={header_offset}-{header_offset + 200}"},
        )
        with urllib.request.urlopen(header_request, timeout=120) as response:
            local_header = response.read()

        if local_header[:4] != b"PK\x03\x04":
            raise ValueError(f"Could not read local zip header for {member_name}.")

        _, _, _, compression_method, _, _, _, _, _, file_name_len, extra_len = struct.unpack(
            "<IHHHHHIIIHH",
            local_header[:30],
        )
        file_start = header_offset + 30 + file_name_len + extra_len
        data_request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={file_start}-{file_start + info.compress_size - 1}"},
        )
        with urllib.request.urlopen(data_request, timeout=600) as response:
            compressed_bytes = response.read()

        if compression_method == zipfile.ZIP_STORED:
            raw_bytes = compressed_bytes
        elif compression_method == zipfile.ZIP_DEFLATED:
            raw_bytes = zlib.decompress(compressed_bytes, -15)
        else:
            raise ValueError(
                f"Unsupported compression method {compression_method} for {member_name}."
            )

        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as out_file:
            out_file.write(raw_bytes)
        return destination

    raise ValueError(f"Could not locate {member_name} in remote zip archive {url}.")


def _ensure_xian_sun_representation_files(root_dir, config):
    if not config.get("use_xian_sun_representation_dataset", False):
        return None

    xian_dir = _resolve_path(root_dir, config.get("xian_sun_dir", XIAN_SUN_DATA_DIR))
    res101_path = os.path.join(xian_dir, "res101.mat")
    splits_path = os.path.join(xian_dir, "att_splits.mat")

    if os.path.exists(res101_path) and os.path.exists(splits_path):
        return xian_dir

    if not config.get("download_xian_sun_if_missing", True):
        raise ValueError(
            "Xian SUN representation files are missing. Enable `download_xian_sun_if_missing` "
            "or place xlsa17/data/SUN/res101.mat and att_splits.mat under the root directory."
        )

    xian_url = config.get("xian_sun_url", XIAN_SUN_ZIP_URL)
    _download_remote_zip_member(
        xian_url,
        XIAN_SUN_RES101_MEMBER,
        res101_path,
        tail_bytes=int(config.get("xian_sun_tail_bytes", 16 * 1024 * 1024)),
    )
    _download_remote_zip_member(
        xian_url,
        XIAN_SUN_SPLITS_MEMBER,
        splits_path,
        tail_bytes=int(config.get("xian_sun_tail_bytes", 16 * 1024 * 1024)),
    )
    return xian_dir


def _build_xian_sun_class_attribute_matrix(class_names, attr_names, class_attr_values, threshold=0.5):
    if class_attr_values.shape[0] == len(class_names):
        class_attributes = class_attr_values
    elif class_attr_values.shape[1] == len(class_names):
        class_attributes = class_attr_values.T
    else:
        raise ValueError(
            f"Could not align SUN class-attribute matrix of shape {class_attr_values.shape} "
            f"with {len(class_names)} classes."
        )

    class_attributes = np.array(class_attributes, dtype=np.float32)
    if threshold is not None:
        class_attributes = (class_attributes >= threshold).astype(np.float32)

    return class_attributes[:len(class_names), :DEFAULT_NUM_ATTRIBUTES]


def _ensure_xian_sun_representation_concepts(root_dir, config, class_names=None):
    if not config.get("use_xian_sun_representation_dataset", False):
        return

    target_class_names = _resolve_path(
        root_dir,
        config.get("class_names_file", f"{XIAN_SUN_SPLIT_PREFIX}_classes.txt"),
    )
    target_class_attributes = _resolve_path(
        root_dir,
        config.get("class_attributes_file", f"{XIAN_SUN_SPLIT_PREFIX}_attrs_per_class_binary_0.npy"),
    )
    if os.path.exists(target_class_names) and os.path.exists(target_class_attributes):
        return

    xian_dir = _ensure_xian_sun_representation_files(root_dir, config)
    if xian_dir is None:
        return

    splits_data = loadmat(os.path.join(xian_dir, "att_splits.mat"))
    raw_class_names = np.array(splits_data["allclasses_names"]).reshape(-1)
    attr_matrix = np.array(splits_data.get("original_att", splits_data.get("att")), dtype=np.float32)

    if class_names is None:
        class_names = [_mat_cell_to_str(cell) for cell in raw_class_names]

    class_attributes = _build_xian_sun_class_attribute_matrix(
        class_names=class_names,
        attr_names=[],
        class_attr_values=attr_matrix,
        threshold=config.get("xian_attribute_threshold", 0.5),
    )

    with open(target_class_names, "w") as class_names_file:
        for class_name in class_names:
            class_names_file.write(class_name + "\n")

    np.save(target_class_attributes, class_attributes)


def _load_xian_sun_representation_data(root_dir, config):
    xian_dir = _ensure_xian_sun_representation_files(root_dir, config)
    if xian_dir is None:
        raise ValueError("Xian SUN representation files could not be initialized.")

    res101_path = os.path.join(xian_dir, "res101.mat")
    splits_path = os.path.join(xian_dir, "att_splits.mat")
    res101_data = loadmat(res101_path)
    splits_data = loadmat(splits_path)

    feature_matrix = np.array(res101_data["features"], dtype=np.float32)
    if feature_matrix.shape[0] == 2048 and feature_matrix.shape[1] != 2048:
        feature_matrix = feature_matrix.T
    if feature_matrix.shape[1] != 2048:
        raise ValueError(
            f"Expected 2048-dimensional SUN features but found shape {feature_matrix.shape}."
        )

    labels = np.array(res101_data["labels"], dtype=np.int64).reshape(-1)
    if labels.min() == 1:
        labels = labels - 1

    class_names = [_mat_cell_to_str(cell) for cell in np.array(splits_data["allclasses_names"]).reshape(-1)]
    attr_matrix = np.array(splits_data.get("original_att", splits_data.get("att")), dtype=np.float32)
    class_attribute_matrix = _build_xian_sun_class_attribute_matrix(
        class_names=class_names,
        attr_names=[],
        class_attr_values=attr_matrix,
        threshold=config.get("xian_attribute_threshold", 0.5),
    )

    return feature_matrix, labels, class_names, class_attribute_matrix, splits_data


def _build_xian_sun_split_payloads(root_dir, config, seed):
    features, labels, class_names, class_attribute_matrix, splits_data = _load_xian_sun_representation_data(
        root_dir,
        config,
    )

    split_strategy = _canonical_xian_split_strategy(
        config.get("xian_split_strategy", "official")
    )
    val_fraction, test_fraction = _get_xian_split_fractions(config)
    if split_strategy == "official":
        train_indices = np.array(splits_data["train_loc"]).reshape(-1) - 1
        val_indices = np.array(splits_data["val_loc"]).reshape(-1) - 1
        test_seen = np.array(splits_data.get("test_seen_loc", [])).reshape(-1)
        test_unseen = np.array(splits_data.get("test_unseen_loc", [])).reshape(-1)
        if len(test_seen) or len(test_unseen):
            test_indices = np.concatenate([test_seen, test_unseen]).reshape(-1) - 1
        else:
            test_indices = np.array(splits_data.get("test_loc", [])).reshape(-1) - 1
    elif split_strategy == "original_sizes":
        class_to_indices = {}
        for sample_index, class_idx in enumerate(labels.tolist()):
            class_to_indices.setdefault(int(class_idx), []).append(sample_index)
        (
            train_indices,
            _,
            val_indices,
            _,
            test_indices,
            _,
        ) = _stratified_train_val_test_split_indices(
            class_to_indices,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
    elif split_strategy == "trainval_test":
        train_indices = np.array(splits_data["trainval_loc"]).reshape(-1) - 1
        val_indices = np.array(splits_data["val_loc"]).reshape(-1) - 1
        test_seen = np.array(splits_data.get("test_seen_loc", [])).reshape(-1)
        test_unseen = np.array(splits_data.get("test_unseen_loc", [])).reshape(-1)
        if len(test_seen) or len(test_unseen):
            test_indices = np.concatenate([test_seen, test_unseen]).reshape(-1) - 1
        else:
            test_indices = np.array(splits_data.get("test_loc", [])).reshape(-1) - 1

    split_payloads = {
        "train": {
            "features": features[train_indices],
            "labels": labels[train_indices],
        },
        "val": {
            "features": features[val_indices],
            "labels": labels[val_indices],
        },
        "test": {
            "features": features[test_indices],
            "labels": labels[test_indices],
        },
    }
    return class_names, class_attribute_matrix, split_payloads


@contextmanager
def _temporary_hf_cache_env(cache_dir):
    old_env = {name: os.environ.get(name) for name in [
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
    ]}
    try:
        if cache_dir is not None:
            hub_cache_dir = os.path.join(cache_dir, "hub")
            datasets_cache_dir = os.path.join(cache_dir, "datasets")
            os.environ["HF_HOME"] = cache_dir
            os.environ["HF_HUB_CACHE"] = hub_cache_dir
            os.environ["HF_DATASETS_CACHE"] = datasets_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
        yield
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _find_existing_image_root(root_dir, config):
    candidate_roots = []
    images_dir = config.get("images_dir", None)
    if images_dir is not None:
        candidate_roots.append(_resolve_path(root_dir, images_dir))
    candidate_roots.append(os.path.join(root_dir, "SUN397"))
    candidate_roots.append(root_dir)

    for candidate_root in candidate_roots:
        if not os.path.isdir(candidate_root):
            continue
        if _collect_image_paths_by_class(candidate_root):
            return candidate_root
    return None


def _maybe_bootstrap_sun397_images(root_dir, config):
    existing_root = _find_existing_image_root(root_dir, config)
    if existing_root is not None:
        return existing_root

    if not config.get("download_if_missing", False):
        return None

    if load_dataset is None:
        raise ImportError(
            "datasets is required to bootstrap SUN397 from Hugging Face. "
            "Install it or disable download_if_missing."
        )

    dataset_id = config.get("download_dataset_id", DEFAULT_HF_DATASET_ID)
    images_root = _resolve_path(root_dir, config.get("images_dir", "SUN397"))
    cache_dir = _resolve_path(
        root_dir,
        config.get("download_cache_dir", DEFAULT_HF_CACHE_DIR),
    )
    download_num_proc = int(config.get("download_num_proc", 1))
    download_max_retries = int(config.get("download_max_retries", 1))
    marker_path = os.path.join(images_root, DOWNLOAD_COMPLETE_MARKER)
    lock_path = os.path.join(images_root, DOWNLOAD_LOCK_FILENAME)
    lock_wait_seconds = int(config.get("download_lock_wait_seconds", 7200))
    stale_lock_seconds = int(config.get("download_stale_lock_seconds", 6 * 3600))

    if os.path.exists(marker_path):
        existing_root = _find_existing_image_root(root_dir, config)
        if existing_root is not None:
            return existing_root

    os.makedirs(images_root, exist_ok=True)

    lock_fd = None
    wait_start = time.time()
    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, str(os.getpid()).encode("utf-8"))
        except FileExistsError:
            existing_root = _find_existing_image_root(root_dir, config)
            if existing_root is not None and os.path.exists(marker_path):
                return existing_root

            try:
                lock_age = time.time() - os.path.getmtime(lock_path)
                if lock_age > stale_lock_seconds:
                    os.remove(lock_path)
                    continue
            except FileNotFoundError:
                continue

            if time.time() - wait_start > lock_wait_seconds:
                raise RuntimeError(
                    "Timed out waiting for SUN397 bootstrap lock. "
                    "If no download is running, remove the lock file at "
                    f"{lock_path}."
                )
            time.sleep(2.0)

    # Another process may have completed the download while we were waiting.
    existing_root = _find_existing_image_root(root_dir, config)
    if existing_root is not None and os.path.exists(marker_path):
        os.close(lock_fd)
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        return existing_root

    if sys.stdin.isatty():
        response = input(
            f"SUN397 images are missing under {root_dir}. Download {dataset_id} "
            f"into {images_root}? [y/N]: "
        ).strip().lower()
    else:
        response = "y"
    if response not in {"y", "yes"}:
        os.close(lock_fd)
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        raise RuntimeError("SUN397 download cancelled by user.")

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    dataset = None
    class_names = None
    try:
        with _temporary_hf_cache_env(cache_dir):
            download_config = None
            if DownloadConfig is not None:
                download_config = DownloadConfig(
                    cache_dir=cache_dir,
                    num_proc=max(1, download_num_proc),
                    max_retries=max(0, download_max_retries),
                    resume_download=True,
                )
            dataset = load_dataset(
                dataset_id,
                cache_dir=cache_dir,
                download_config=download_config,
            )
            label_feature = None

            for split_name in dataset.keys():
                split_dataset = dataset[split_name]
                if HFDatasetsImage is not None:
                    split_dataset = split_dataset.cast_column(
                        "image",
                        HFDatasetsImage(decode=False),
                    )
                if label_feature is None:
                    label_feature = split_dataset.features["label"]
                    class_names = list(label_feature.names)

                for row_index, example in enumerate(split_dataset):
                    class_name = label_feature.int2str(int(example["label"]))
                    class_dir = os.path.join(images_root, class_name)
                    os.makedirs(class_dir, exist_ok=True)

                    image = example["image"]
                    if isinstance(image, dict):
                        image_bytes = image.get("bytes", None)
                        image_path = image.get("path", None)
                        if image_bytes is not None:
                            image = Image.open(io.BytesIO(image_bytes))
                        elif image_path is not None:
                            image = Image.open(image_path)
                        else:
                            raise ValueError(
                                "Unsupported Hugging Face image record with no "
                                "bytes or path."
                            )
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    image_path = os.path.join(class_dir, f"{split_name}_{row_index:06d}.jpg")
                    image.save(image_path, format="JPEG", quality=95)
    finally:
        if cache_dir is not None and os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        if lock_fd is not None:
            os.close(lock_fd)
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass

    class_name_paths = {
        os.path.join(root_dir, "classes.txt"),
        os.path.join(root_dir, "sun_classes.txt"),
    }
    for class_name_path in class_name_paths:
        if os.path.exists(class_name_path):
            continue
        with open(class_name_path, "w") as f:
            for class_name in class_names:
                f.write(class_name + "\n")

    with open(marker_path, "w") as f:
        f.write(dataset_id + "\n")

    return images_root


def _read_lines(path):
    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def _parse_classes_file(path):
    class_names = []
    for line in _read_lines(path):
        if "\t" in line:
            parts = line.split("\t")
            class_names.append(parts[-1].strip())
        elif " " in line and line.split(" ", 1)[0].isdigit():
            class_names.append(line.split(" ", 1)[1].strip())
        else:
            class_names.append(line.strip())
    return class_names


def _extract_class_from_relpath(relpath):
    relpath = relpath.replace("\\", "/")
    if relpath.startswith("SUN397/"):
        relpath = relpath[len("SUN397/"):]
    parts = relpath.split("/")
    if len(parts) <= 1:
        return ""
    return "/".join(parts[:-1]).strip()


def _extract_concept_names_from_mat(mat_path):
    data = loadmat(mat_path)
    for key, val in data.items():
        if key.startswith("__"):
            continue
        arr = np.array(val)
        if arr.dtype == object:
            flat = arr.reshape(-1)
            out = []
            for elem in flat:
                elem_arr = np.array(elem).reshape(-1)
                if len(elem_arr) == 0:
                    continue
                txt = str(elem_arr[0]).strip()
                if txt:
                    out.append(txt)
            if len(out) >= DEFAULT_NUM_ATTRIBUTES:
                return out
        elif arr.ndim == 1 and arr.dtype.kind in ["U", "S"]:
            out = [str(x).strip() for x in arr.tolist() if str(x).strip()]
            if len(out) >= DEFAULT_NUM_ATTRIBUTES:
                return out
    return None


def _extract_attribute_matrix_from_mat(mat_path, expected_num_classes=None):
    data = loadmat(mat_path)
    candidates = []
    for key, val in data.items():
        if key.startswith("__"):
            continue
        arr = np.array(val)
        if arr.ndim != 2:
            continue
        if min(arr.shape) < DEFAULT_NUM_ATTRIBUTES:
            continue
        candidates.append(arr)

    if not candidates:
        raise ValueError(
            f"Could not find a valid 2D attribute matrix in MAT file {mat_path}."
        )

    def _score(shape):
        score = 0
        if DEFAULT_NUM_ATTRIBUTES in shape:
            score += 100
        if expected_num_classes is not None and expected_num_classes in shape:
            score += 50
        score -= abs(shape[1] - DEFAULT_NUM_ATTRIBUTES)
        return score

    arr = sorted(candidates, key=lambda x: _score(x.shape), reverse=True)[0]
    arr = np.array(arr, dtype=np.float32)
    if arr.shape[0] == DEFAULT_NUM_ATTRIBUTES and arr.shape[1] != DEFAULT_NUM_ATTRIBUTES:
        arr = arr.T
    if arr.shape[1] != DEFAULT_NUM_ATTRIBUTES and arr.shape[0] == DEFAULT_NUM_ATTRIBUTES:
        arr = arr.T
    return arr


def _load_concept_semantics(root_dir, config):
    concept_names_path = _first_existing_file(
        root_dir,
        [
            config.get("attribute_names_file", None),
            "sun_attributes.txt",
            "attributes.txt",
            "attribute_names.txt",
            "SUNattributeDB/attributes.txt",
            "SUNAttributeDB/attributes.txt",
        ],
    )
    if concept_names_path is not None:
        concept_names = _read_lines(concept_names_path)
        if len(concept_names) >= DEFAULT_NUM_ATTRIBUTES:
            return concept_names[:DEFAULT_NUM_ATTRIBUTES]

    concept_names_mat = _first_existing_file(
        root_dir,
        [
            config.get("attribute_names_mat_file", None),
            "attributes.mat",
            "SUNattributeDB/attributes.mat",
            "SUNAttributeDB/attributes.mat",
        ],
    )
    if concept_names_mat is not None:
        concept_names = _extract_concept_names_from_mat(concept_names_mat)
        if concept_names is not None and len(concept_names) >= DEFAULT_NUM_ATTRIBUTES:
            return concept_names[:DEFAULT_NUM_ATTRIBUTES]

    # Safe fallback that keeps dimensions consistent when no names are provided.
    return [f"sun_attribute_{i:03d}" for i in range(DEFAULT_NUM_ATTRIBUTES)]


def _load_class_attribute_matrix(root_dir, config, class_names):
    class_attr_path = _first_existing_file(
        root_dir,
        [
            config.get("class_attributes_file", None),
            "sun_attrs_per_class_binary_0.npy",
            "sun_attrs_per_class_binary.npy",
            "sun_attrs_per_class.npy",
            "class_attributes.npy",
            "attributes_per_class.npy",
            "sun_attrs_per_class_binary_0.npz",
            "class_attributes.npz",
            "class_attributes.txt",
            "attributes_per_class.txt",
            "sun_attrs_per_class_binary_0.mat",
            "class_attributes.mat",
        ],
    )
    if class_attr_path is None:
        _ensure_sun_attribute_db_concepts(root_dir, config, class_names=class_names)
        class_attr_path = _first_existing_file(
            root_dir,
            [
                config.get("class_attributes_file", None),
                "sun_attrs_per_class_binary_0.npy",
                "sun_attrs_per_class_binary.npy",
                "sun_attrs_per_class.npy",
                "class_attributes.npy",
                "attributes_per_class.npy",
                "sun_attrs_per_class_binary_0.npz",
                "class_attributes.npz",
                "class_attributes.txt",
                "attributes_per_class.txt",
                "sun_attrs_per_class_binary_0.mat",
                "class_attributes.mat",
            ],
        )

    if class_attr_path is None:
        raise ValueError(
            "Could not locate a per-class SUN attribute matrix. Please provide "
            "`class_attributes_file` in dataset_config."
        )

    if class_attr_path.endswith(".npy"):
        class_attributes = np.load(class_attr_path)
    elif class_attr_path.endswith(".npz"):
        loaded = np.load(class_attr_path)
        valid_keys = [
            key for key in loaded.files
            if np.array(loaded[key]).ndim == 2
        ]
        if not valid_keys:
            raise ValueError(
                f"No valid 2D attribute matrix found in NPZ file {class_attr_path}."
            )
        class_attributes = np.array(loaded[valid_keys[0]])
    elif class_attr_path.endswith(".mat"):
        class_attributes = _extract_attribute_matrix_from_mat(
            class_attr_path,
            expected_num_classes=len(class_names),
        )
    else:
        class_attributes = np.genfromtxt(class_attr_path, dtype=np.float32)

    class_attributes = np.array(class_attributes, dtype=np.float32)
    if class_attributes.ndim != 2:
        raise ValueError(
            f"Expected a 2D class-attribute matrix in {class_attr_path}; "
            f"found shape {class_attributes.shape}."
        )

    # Align orientation to [num_classes, num_attributes].
    if class_attributes.shape[0] == DEFAULT_NUM_ATTRIBUTES and class_attributes.shape[1] != DEFAULT_NUM_ATTRIBUTES:
        class_attributes = class_attributes.T
    if class_attributes.shape[1] != DEFAULT_NUM_ATTRIBUTES:
        raise ValueError(
            f"Expected {DEFAULT_NUM_ATTRIBUTES} SUN attributes but found "
            f"shape {class_attributes.shape}."
        )

    if class_attributes.shape[0] < len(class_names):
        raise ValueError(
            f"Class-attribute matrix has {class_attributes.shape[0]} classes, "
            f"but dataset has {len(class_names)} classes."
        )

    if class_attributes.shape[0] > len(class_names):
        attr_class_names_path = _first_existing_file(
            root_dir,
            [
                config.get("attribute_class_names_file", None),
                "sun_classes.txt",
                "classes.txt",
            ],
        )
        if attr_class_names_path is None:
            class_attributes = class_attributes[:len(class_names)]
        else:
            attr_class_names = _parse_classes_file(attr_class_names_path)
            attr_name_map = {
                _normalize_class_name(name): idx
                for idx, name in enumerate(attr_class_names)
            }
            selected_rows = []
            missing_classes = []
            for class_name in class_names:
                normalized = _normalize_class_name(class_name)
                if normalized in attr_name_map:
                    selected_rows.append(attr_name_map[normalized])
                else:
                    missing_classes.append(class_name)
            if missing_classes:
                raise ValueError(
                    "Could not align class-attribute rows to image classes for: "
                    f"{missing_classes[:10]}"
                )
            class_attributes = class_attributes[selected_rows]

    threshold = config.get("class_attribute_threshold", 0.5)
    if not np.array_equal(class_attributes, class_attributes.astype(bool)):
        class_attributes = (class_attributes >= threshold).astype(np.float32)

    return class_attributes


def _collect_image_paths_by_class(base_dir, allowed_exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif")):
    class_to_paths = {}
    for root, _, files in os.walk(base_dir):
        for file_name in files:
            if not file_name.lower().endswith(allowed_exts):
                continue
            abs_path = os.path.abspath(os.path.join(root, file_name))
            rel_path = os.path.relpath(abs_path, base_dir)
            class_name = _extract_class_from_relpath(rel_path)
            if not class_name:
                continue
            class_to_paths.setdefault(class_name, []).append(abs_path)
    return class_to_paths


def _stratified_train_val_test_split(class_to_paths, seed, val_fraction=0.2, test_fraction=0.2):
    rng = np.random.default_rng(seed)
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    class_names = sorted(class_to_paths.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        paths = np.array(sorted(class_to_paths[class_name]))
        if len(paths) < 3:
            # Keep tiny classes train-heavy while preserving all samples.
            splits = [len(paths), 0, 0]
        else:
            n_test = max(1, int(np.floor(len(paths) * test_fraction)))
            n_val = max(1, int(np.floor(len(paths) * val_fraction)))
            n_train = max(1, len(paths) - n_test - n_val)
            while n_train + n_val + n_test > len(paths):
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    n_train -= 1
            splits = [n_train, n_val, n_test]

        perm = rng.permutation(len(paths))
        paths = paths[perm]
        n_train, n_val, n_test = splits

        for p in paths[:n_train]:
            train_paths.append(p)
            train_labels.append(class_to_idx[class_name])
        for p in paths[n_train:n_train + n_val]:
            val_paths.append(p)
            val_labels.append(class_to_idx[class_name])
        for p in paths[n_train + n_val:n_train + n_val + n_test]:
            test_paths.append(p)
            test_labels.append(class_to_idx[class_name])

    return (
        class_names,
        np.array(train_paths),
        np.array(train_labels),
        np.array(val_paths),
        np.array(val_labels),
        np.array(test_paths),
        np.array(test_labels),
    )


def _stratified_train_val_test_split_indices(class_to_indices, seed, val_fraction=0.2, test_fraction=0.2):
    rng = np.random.default_rng(seed)
    train_indices, train_labels = [], []
    val_indices, val_labels = [], []
    test_indices, test_labels = [], []

    class_ids = sorted(class_to_indices.keys())
    for class_id in class_ids:
        indices = np.array(sorted(class_to_indices[class_id]))
        if len(indices) < 3:
            splits = [len(indices), 0, 0]
        else:
            n_test = max(1, int(np.floor(len(indices) * test_fraction)))
            n_val = max(1, int(np.floor(len(indices) * val_fraction)))
            n_train = max(1, len(indices) - n_test - n_val)
            while n_train + n_val + n_test > len(indices):
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    n_train -= 1
            splits = [n_train, n_val, n_test]

        perm = rng.permutation(len(indices))
        indices = indices[perm]
        n_train, n_val, n_test = splits

        train_chunk = indices[:n_train]
        val_chunk = indices[n_train:n_train + n_val]
        test_chunk = indices[n_train + n_val:n_train + n_val + n_test]

        train_indices.extend(train_chunk.tolist())
        train_labels.extend([class_id] * len(train_chunk))
        val_indices.extend(val_chunk.tolist())
        val_labels.extend([class_id] * len(val_chunk))
        test_indices.extend(test_chunk.tolist())
        test_labels.extend([class_id] * len(test_chunk))

    return (
        np.array(train_indices, dtype=np.int64),
        np.array(train_labels, dtype=np.int64),
        np.array(val_indices, dtype=np.int64),
        np.array(val_labels, dtype=np.int64),
        np.array(test_indices, dtype=np.int64),
        np.array(test_labels, dtype=np.int64),
    )


########################################################
## Dataset Loader
########################################################


class SUNDataset(Dataset):
    """Torch dataset for SUN images with per-class attribute concepts."""

    def __init__(
        self,
        root_dir,
        split="train",
        augment_data=False,
        image_size=224,
        concept_transform=None,
        sample_transform=None,
        seed=42,
        config=None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.config = config or {}
        self.concept_transform = concept_transform

        self.use_xian_representation_dataset = self.config.get(
            "use_xian_sun_representation_dataset",
            False,
        )
        if self.use_xian_representation_dataset:
            _ensure_xian_sun_representation_files(self.root_dir, self.config)
        else:
            _maybe_bootstrap_sun397_images(self.root_dir, self.config)

        if not os.path.exists(self.root_dir):
            raise ValueError(
                f"{self.root_dir} does not exist yet. Please download the "
                f"dataset first."
            )

        if self.use_xian_representation_dataset:
            self.transform = sample_transform
        elif split == "train":
            self.transform = get_transform_sun(
                train=True,
                augment_data=augment_data,
                image_size=image_size,
                sample_transform=sample_transform,
            )
        else:
            self.transform = get_transform_sun(
                train=False,
                augment_data=augment_data,
                image_size=image_size,
                sample_transform=sample_transform,
            )

        (
            self.class_names,
            self.class_to_index,
            self.class_attribute_matrix,
            split_payload,
        ) = self._load_or_generate_splits(seed=seed)

        if self.use_xian_representation_dataset:
            self.sample_features = split_payload["features"].astype(np.float32)
            self.sample_labels = split_payload["labels"].astype(int)
            self.sample_paths = None
        else:
            self.img_paths = split_payload["paths"]
            self.img_labels = split_payload["labels"].astype(int)
        print(f"{split.upper()} SUN dataset has: {len(self)} samples")

    def _load_or_generate_splits(self, seed):
        use_attribute_db_dataset = self.config.get("use_sun_attribute_db_dataset", False)
        use_xian_representation_dataset = self.config.get(
            "use_xian_sun_representation_dataset",
            False,
        )

        if use_xian_representation_dataset:
            xian_strategy = _canonical_xian_split_strategy(
                self.config.get("xian_split_strategy", "original_sizes")
            )
            xian_val_fraction, xian_test_fraction = _get_xian_split_fractions(self.config)
            xian_fraction_tag = (
                f"vf{_canonical_fraction_tag(xian_val_fraction)}"
                f"_tf{_canonical_fraction_tag(xian_test_fraction)}"
            )
            class_names_file = _first_existing_file(
                self.root_dir,
                [
                    self.config.get("class_names_file", None),
                    f"{XIAN_SUN_SPLIT_PREFIX}_classes.txt",
                ],
            )
            split_prefix = f"{XIAN_SUN_SPLIT_PREFIX}_{xian_strategy}_{xian_fraction_tag}_"
        elif use_attribute_db_dataset:
            class_names_file = _first_existing_file(
                self.root_dir,
                [
                    self.config.get("class_names_file", None),
                    f"{SUN_ATTRIBUTE_DB_SPLIT_PREFIX}_classes.txt",
                ],
            )
            split_prefix = f"{SUN_ATTRIBUTE_DB_SPLIT_PREFIX}_"
        else:
            class_names_file = _first_existing_file(
                self.root_dir,
                [
                    self.config.get("class_names_file", None),
                    "sun_classes.txt",
                    "classes.txt",
                ],
            )
            split_prefix = ""

        split_file = os.path.join(self.root_dir, f"{split_prefix}{self.split}_split.npz")
        all_split_files_exist = all(
            os.path.exists(os.path.join(self.root_dir, f"{split_prefix}{x}_split.npz"))
            for x in ["train", "val", "test"]
        )
        should_regenerate = self.config.get("regenerate_splits", False)

        if (not all_split_files_exist) or should_regenerate:
            if use_xian_representation_dataset:
                print(
                    f"Generating Xian SUN splits with strategy '{xian_strategy}' "
                    f"(val_fraction={xian_val_fraction}, test_fraction={xian_test_fraction}) "
                    f"(regenerate_splits={should_regenerate}, all_split_files_exist={all_split_files_exist})."
                )
            class_to_paths = None

            if use_xian_representation_dataset:
                class_names, class_attribute_matrix, split_payloads = _build_xian_sun_split_payloads(
                    self.root_dir,
                    self.config,
                    seed=seed,
                )
            elif use_attribute_db_dataset:
                class_names, split_payloads = _build_sun_attribute_db_split_payloads(
                    self.root_dir,
                    self.config,
                    seed=seed,
                )
            else:
                # Option A: explicit train/val/test directories.
                split_dirs_exist = all(
                    os.path.isdir(os.path.join(self.root_dir, x))
                    for x in ["train", "val", "test"]
                )
                if split_dirs_exist:
                    class_names = []
                    split_payloads = {}
                    for split in ["train", "val", "test"]:
                        base_dir = os.path.join(self.root_dir, split)
                        current_class_to_paths = _collect_image_paths_by_class(base_dir)
                        if split == "train":
                            class_names = sorted(current_class_to_paths.keys())
                        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
                        paths = []
                        labels = []
                        for class_name in class_names:
                            for p in sorted(current_class_to_paths.get(class_name, [])):
                                paths.append(p)
                                labels.append(class_to_idx[class_name])
                        split_payloads[split] = {
                            "paths": np.array(paths),
                            "labels": np.array(labels),
                        }
                else:
                    images_root = _find_existing_image_root(self.root_dir, self.config)
                    if images_root is None:
                        configured_images_root = self.config.get("images_dir", None)
                        if configured_images_root is None:
                            sun397_candidate = os.path.join(self.root_dir, "SUN397")
                            images_root = sun397_candidate if os.path.isdir(sun397_candidate) else self.root_dir
                        else:
                            images_root = _resolve_path(self.root_dir, configured_images_root)

                    class_to_paths = _collect_image_paths_by_class(images_root)

                    if len(class_to_paths) == 0:
                        raise ValueError(
                            "Could not find SUN images. Expected either train/val/test "
                            "folders or a directory with class subfolders under "
                            f"{images_root}."
                        )

                    (
                        class_names,
                        train_paths,
                        train_labels,
                        val_paths,
                        val_labels,
                        test_paths,
                        test_labels,
                    ) = _stratified_train_val_test_split(
                        class_to_paths,
                        seed=seed,
                        val_fraction=self.config.get("val_fraction", 0.2),
                        test_fraction=self.config.get("test_fraction", 0.2),
                    )

                    split_payloads = {
                        "train": {"paths": train_paths, "labels": train_labels},
                        "val": {"paths": val_paths, "labels": val_labels},
                        "test": {"paths": test_paths, "labels": test_labels},
                    }

                class_attribute_matrix = None

            for split_name, payload in split_payloads.items():
                split_save_kwargs = {
                    "labels": payload["labels"],
                    "class_names": np.array(class_names, dtype=object),
                }
                if use_xian_representation_dataset:
                    split_save_kwargs["features"] = payload["features"]
                else:
                    split_save_kwargs["paths"] = payload["paths"]
                np.savez(
                    os.path.join(self.root_dir, f"{split_prefix}{split_name}_split.npz"),
                    **split_save_kwargs,
                )
        elif use_xian_representation_dataset:
            print(
                f"Loading cached Xian SUN splits with strategy '{xian_strategy}' "
                f"(val_fraction={xian_val_fraction}, test_fraction={xian_test_fraction}) "
                f"from prefix '{split_prefix}'."
            )

            if class_names_file is None:
                if use_xian_representation_dataset:
                    generated_name = f"{XIAN_SUN_SPLIT_PREFIX}_classes.txt"
                elif use_attribute_db_dataset:
                    generated_name = f"{SUN_ATTRIBUTE_DB_SPLIT_PREFIX}_classes.txt"
                else:
                    generated_name = "classes.txt"
                generated_path = os.path.join(self.root_dir, generated_name)
                with open(generated_path, "w") as f:
                    for name in class_names:
                        f.write(name + "\n")
                class_names_file = generated_path

        if class_names_file is not None:
            class_names = _parse_classes_file(class_names_file)
        else:
            saved_split = np.load(split_file, allow_pickle=True)
            if "class_names" in saved_split:
                class_names = list(saved_split["class_names"])
            else:
                all_paths = saved_split["paths"]
                all_labels = saved_split["labels"]
                inv_map = {}
                for path, lbl in zip(all_paths, all_labels):
                    inv_map[int(lbl)] = _extract_class_from_relpath(
                        os.path.relpath(path, self.root_dir)
                    )
                class_names = [inv_map[i] for i in sorted(inv_map.keys())]

        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        if use_xian_representation_dataset:
            _ensure_xian_sun_representation_concepts(
                self.root_dir,
                self.config,
                class_names=class_names,
            )
        class_attribute_matrix = _load_class_attribute_matrix(
            self.root_dir,
            self.config,
            class_names=class_names,
        )

        split_payload = np.load(split_file, allow_pickle=True)
        return class_names, class_to_idx, class_attribute_matrix, split_payload

    def __len__(self):
        if self.use_xian_representation_dataset:
            return len(self.sample_labels)
        return len(self.img_paths)

    def __getitem__(self, index):
        if self.use_xian_representation_dataset:
            sample = torch.FloatTensor(self.sample_features[index])
            if self.transform:
                sample = self.transform(sample)
            label_idx = int(self.sample_labels[index])
        else:
            sample = Image.open(self.img_paths[index])
            if sample.mode != "RGB":
                sample = sample.convert("RGB")
            if self.transform:
                sample = self.transform(sample)

            label_idx = int(self.img_labels[index])
        concepts = self.class_attribute_matrix[label_idx, :]
        if self.concept_transform is not None:
            concepts = self.concept_transform(concepts)
        return sample, label_idx, torch.FloatTensor(concepts)


def get_transform_sun(
    train,
    augment_data,
    image_size=224,
    sample_transform=None,
):
    """Helper function to build SUN image transforms."""
    scale = 256.0 / 224.0
    sample_transform = sample_transform if sample_transform is not None else (lambda x: x)

    if (not train) or (not augment_data):
        return transforms.Compose([
            transforms.Resize((int(image_size * scale), int(image_size * scale))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            sample_transform,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=2,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        sample_transform,
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_data(
    split,
    batch_size,
    image_size=224,
    root_dir="data/SUN/",
    num_workers=1,
    dataset_transform=lambda x: x,
    dataset_size=None,
    augment_data=False,
    concept_transform=None,
    additional_sample_transform=None,
    seed=42,
    config=None,
):
    """Generates a dataloader for SUN."""
    dataset = SUNDataset(
        split=split,
        root_dir=root_dir,
        augment_data=augment_data,
        image_size=image_size,
        concept_transform=concept_transform,
        sample_transform=additional_sample_transform,
        seed=seed,
        config=config,
    )

    if dataset_size is not None:
        selected = np.random.permutation(len(dataset))
        if 0 < dataset_size < 1:
            dataset_size = int(np.ceil(len(dataset) * dataset_size))
        selected = selected[: int(np.ceil(dataset_size))]
        dataset = Subset(dataset, selected)

    drop_last = split == "train"
    shuffle = split == "train"
    return DataLoader(
        dataset_transform(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


########################
## Data Module
########################


def get_num_labels(*args, **kwargs):
    """
    This call is needed to satisfy the loader API used in this codebase.

    Returns:
        int: number of SUN class labels.
    """
    # When unknown at config-time, this is overwritten by generate_data outputs.
    return int(kwargs.get("n_tasks", 397))


def get_num_attributes(*args, **kwargs):
    """
    This call is needed to satisfy the loader API used in this codebase.

    Returns:
        int: number of SUN attributes.
    """
    return DEFAULT_NUM_ATTRIBUTES


def _compute_selected_concepts(config, root_dir, n_concepts, rerun=False):
    sampling_percent = config.get("sampling_percent", 1)
    sampling_groups = config.get("sampling_groups", False)

    if sampling_percent == 1:
        return list(range(n_concepts)), {i: [i] for i in range(n_concepts)}

    if sampling_groups:
        # For SUN we use singleton groups by default.
        all_groups = {i: [i] for i in range(n_concepts)}
        new_n_groups = int(np.ceil(len(all_groups) * sampling_percent))
        selected_groups_file = os.path.join(
            root_dir,
            f"selected_groups_sampling_{sampling_percent}.npy",
        )
        if (not rerun) and os.path.exists(selected_groups_file):
            selected_groups = np.load(selected_groups_file)
        else:
            selected_groups = sorted(
                np.random.permutation(len(all_groups))[:new_n_groups]
            )
            np.save(selected_groups_file, selected_groups)

        selected_concepts = sorted(set(int(g) for g in selected_groups))
    else:
        new_n_concepts = int(np.ceil(n_concepts * sampling_percent))
        selected_concepts_file = os.path.join(
            root_dir,
            f"selected_concepts_sampling_{sampling_percent}.npy",
        )
        if (not rerun) and os.path.exists(selected_concepts_file):
            selected_concepts = np.load(selected_concepts_file).tolist()
        else:
            selected_concepts = sorted(
                np.random.permutation(n_concepts)[:new_n_concepts].tolist()
            )
            np.save(selected_concepts_file, np.array(selected_concepts))

    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_concepts)}
    concept_group_map = {
        remap[old_idx]: [remap[old_idx]]
        for old_idx in selected_concepts
    }
    return selected_concepts, concept_group_map


def _get_loss_weight_cache_path(config, root_dir):
    configured_path = config.get("loss_weight_cache_file", None)
    if configured_path is not None:
        return _resolve_path(root_dir, configured_path)

    prefix = (
        SUN_ATTRIBUTE_DB_SPLIT_PREFIX
        if config.get("use_sun_attribute_db_dataset", False)
        else "sun397"
    )
    return os.path.join(root_dir, f"{prefix}_concept_loss_weights_cache.npz")


def _compute_or_load_imbalance(
    train_dl,
    n_concepts,
    config,
    root_dir,
    selected_concepts,
):
    cache_enabled = config.get("cache_loss_weights", True)
    force_recompute = config.get("recompute_loss_weights", False)
    cache_path = _get_loss_weight_cache_path(config, root_dir)

    selected_signature = ",".join(str(int(x)) for x in selected_concepts)
    metadata = {
        "n_concepts": int(n_concepts),
        "selected_signature": selected_signature,
        "class_attributes_file": str(config.get("class_attributes_file", "")),
        "class_attribute_threshold": float(config.get("class_attribute_threshold", 0.5)),
        "sampling_percent": float(config.get("sampling_percent", 1)),
        "sampling_groups": bool(config.get("sampling_groups", False)),
        "dataset_size": str(config.get("dataset_size", None)),
        "use_sun_attribute_db_dataset": bool(config.get("use_sun_attribute_db_dataset", False)),
    }

    if cache_enabled and (not force_recompute) and os.path.exists(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            cached_imbalance = np.array(loaded["imbalance"], dtype=np.float32)
            cached_metadata = loaded["metadata"].item()
            if (
                isinstance(cached_metadata, dict)
                and cached_imbalance.shape[0] == n_concepts
                and all(cached_metadata.get(k) == metadata.get(k) for k in metadata)
            ):
                print(f"Loaded cached concept loss weights from {cache_path}")
                return cached_imbalance
        except Exception:
            pass

    attribute_count = np.zeros((n_concepts,), dtype=np.float64)
    samples_seen = 0
    for _, _, c in train_dl:
        c = c.cpu().detach().numpy()
        attribute_count += np.sum(c, axis=0)
        samples_seen += c.shape[0]
    imbalance = (samples_seen / (attribute_count - 1 + 1e-8)).astype(np.float32)

    if cache_enabled:
        cache_parent = os.path.dirname(cache_path)
        if cache_parent:
            os.makedirs(cache_parent, exist_ok=True)
        np.savez(
            cache_path,
            imbalance=imbalance,
            metadata=np.array(metadata, dtype=object),
        )
        print(f"Saved concept loss weights cache to {cache_path}")

    return imbalance


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
    if root_dir is None:
        root_dir = DATASET_DIR
    seed_everything(seed)
    print("config:", config)

    training_transform = training_transform if training_transform is not None else dataset_transform

    val_subsample = config.get("val_subsample", None)
    image_size = config.get("image_size", 224)
    num_workers = config.get("num_workers", 8)
    dataset_size = config.get("dataset_size", None)
    augment_data = config.get("augment_data", False)
    batch_size = config.get("batch_size", 32)

    use_attribute_db_dataset = config.get("use_sun_attribute_db_dataset", False)
    use_xian_representation_dataset = config.get(
        "use_xian_sun_representation_dataset",
        False,
    )
    class_names_file = config.get("class_names_file", None)
    class_attributes_file = config.get("class_attributes_file", None)

    if use_xian_representation_dataset and class_names_file in [None, "sun_classes.txt", "classes.txt", f"{SUN_ATTRIBUTE_DB_SPLIT_PREFIX}_classes.txt"]:
        class_names_file = f"{XIAN_SUN_SPLIT_PREFIX}_classes.txt"
    if use_xian_representation_dataset and class_attributes_file in [
        None,
        "sun_attrs_per_class_binary_0.npy",
        "sun_attrs_per_class_binary.npy",
        "sun_attrs_per_class.npy",
        f"{SUN_ATTRIBUTE_DB_SPLIT_PREFIX}_attrs_per_class_binary_0.npy",
    ]:
        class_attributes_file = f"{XIAN_SUN_SPLIT_PREFIX}_attrs_per_class_binary_0.npy"

    if use_attribute_db_dataset and class_names_file in [None, "sun_classes.txt", "classes.txt"]:
        class_names_file = f"{SUN_ATTRIBUTE_DB_SPLIT_PREFIX}_classes.txt"
    if use_attribute_db_dataset and class_attributes_file in [
        None,
        "sun_attrs_per_class_binary_0.npy",
        "sun_attrs_per_class_binary.npy",
        "sun_attrs_per_class.npy",
    ]:
        class_attributes_file = f"{SUN_ATTRIBUTE_DB_SPLIT_PREFIX}_attrs_per_class_binary_0.npy"

    dataset_opts = {
        "class_names_file": class_names_file,
        "attribute_names_file": config.get("attribute_names_file", None),
        "attribute_names_mat_file": config.get("attribute_names_mat_file", None),
        "class_attributes_file": class_attributes_file,
        "attribute_class_names_file": config.get("attribute_class_names_file", None),
        "images_dir": config.get("images_dir", None),
        "use_sun_attribute_db_dataset": use_attribute_db_dataset,
        "use_xian_sun_representation_dataset": use_xian_representation_dataset,
        "download_if_missing": config.get("download_if_missing", False),
        "download_dataset_id": config.get("download_dataset_id", DEFAULT_HF_DATASET_ID),
        "download_cache_dir": config.get("download_cache_dir", DEFAULT_HF_CACHE_DIR),
        "download_num_proc": config.get("download_num_proc", 1),
        "download_max_retries": config.get("download_max_retries", 1),
        "download_lock_wait_seconds": config.get("download_lock_wait_seconds", 7200),
        "download_stale_lock_seconds": config.get("download_stale_lock_seconds", 6 * 3600),
        "use_sun_attribute_database": config.get("use_sun_attribute_database", True),
        "download_attribute_db_if_missing": config.get("download_attribute_db_if_missing", True),
        "sun_attribute_db_url": config.get("sun_attribute_db_url", SUN_ATTRIBUTE_DB_URL),
        "sun_attribute_db_archive": config.get("sun_attribute_db_archive", SUN_ATTRIBUTE_DB_ARCHIVE),
        "sun_attribute_db_dir": config.get("sun_attribute_db_dir", SUN_ATTRIBUTE_DB_DIR),
        "download_attribute_db_images_if_missing": config.get("download_attribute_db_images_if_missing", True),
        "sun_attribute_db_images_url": config.get("sun_attribute_db_images_url", SUN_ATTRIBUTE_DB_IMAGES_URL),
        "sun_attribute_db_images_archive": config.get("sun_attribute_db_images_archive", SUN_ATTRIBUTE_DB_IMAGES_ARCHIVE),
        "download_xian_sun_if_missing": config.get("download_xian_sun_if_missing", True),
        "xian_sun_url": config.get("xian_sun_url", XIAN_SUN_ZIP_URL),
        "xian_sun_dir": config.get("xian_sun_dir", XIAN_SUN_DATA_DIR),
        "xian_sun_tail_bytes": config.get("xian_sun_tail_bytes", 16 * 1024 * 1024),
        "xian_split_strategy": config.get("xian_split_strategy", "original_sizes"),
        "xian_val_fraction": config.get("xian_val_fraction", 0.1),
        "xian_test_fraction": config.get("xian_test_fraction", 0.1),
        "xian_attribute_threshold": config.get("xian_attribute_threshold", 0.5),
        "regenerate_splits": config.get("regenerate_splits", False),
        "class_attribute_threshold": config.get("class_attribute_threshold", 0.5),
        "val_fraction": config.get("val_fraction", 0.1 if use_attribute_db_dataset else 0.2),
        "test_fraction": config.get("test_fraction", 0.1 if use_attribute_db_dataset else 0.2),
    }

    # In SUN397 mode, optionally bootstrap full SUN397 images from Hugging Face.
    if not use_attribute_db_dataset and (not use_xian_representation_dataset):
        _maybe_bootstrap_sun397_images(root_dir, dataset_opts)
    if use_xian_representation_dataset:
        _ensure_xian_sun_representation_concepts(root_dir, dataset_opts)
    if use_attribute_db_dataset:
        _ensure_sun_attribute_db_concepts(root_dir, dataset_opts)

    concept_names = _load_concept_semantics(root_dir, config)
    n_concepts = len(concept_names)
    selected_concepts, concept_group_map = _compute_selected_concepts(
        config,
        root_dir,
        n_concepts,
        rerun=rerun,
    )

    if len(selected_concepts) != n_concepts:
        def concept_transform(sample):
            if isinstance(sample, list):
                sample = np.array(sample)
            return sample[selected_concepts]
        n_concepts = len(selected_concepts)
    else:
        concept_transform = None

    train_dl = load_data(
        split="train",
        batch_size=batch_size,
        image_size=image_size,
        root_dir=root_dir,
        num_workers=num_workers,
        dataset_transform=training_transform if val_subsample is None else (lambda x: x),
        dataset_size=dataset_size,
        augment_data=augment_data,
        concept_transform=concept_transform,
        additional_sample_transform=train_sample_transform,
        seed=seed,
        config=dataset_opts,
    )

    if config.get("weight_loss", False):
        imbalance = _compute_or_load_imbalance(
            train_dl=train_dl,
            n_concepts=n_concepts,
            config=config,
            root_dir=root_dir,
            selected_concepts=selected_concepts,
        )
    else:
        imbalance = None

    if val_subsample is None:
        val_dl = load_data(
            split="val",
            batch_size=batch_size,
            image_size=image_size,
            root_dir=root_dir,
            num_workers=num_workers,
            dataset_transform=dataset_transform,
            dataset_size=dataset_size,
            augment_data=False,
            concept_transform=concept_transform,
            additional_sample_transform=val_sample_transform,
            seed=seed,
            config=dataset_opts,
        )
    else:
        from sklearn import model_selection

        ys = []
        xs = []
        attributes = []
        fast_loader = torch.utils.data.DataLoader(
            train_dl.dataset,
            batch_size=min(256, len(train_dl.dataset)),
            num_workers=min(2, max(1, num_workers)),
            drop_last=False,
            shuffle=False,
        )
        for x, y, attribute in fast_loader:
            ys.append(y.detach().cpu().numpy())
            xs.append(x.detach().cpu().numpy())
            attributes.append(attribute.detach().cpu().numpy())
        ys = np.concatenate(ys, axis=0)
        xs = np.concatenate(xs, axis=0)
        attributes = np.concatenate(attributes, axis=0)

        x_train, x_val, y_train, y_val, c_train, c_val = model_selection.train_test_split(
            xs,
            ys,
            attributes,
            test_size=val_subsample,
            random_state=seed,
            stratify=ys,
        )
        train_dl = DataLoader(
            training_transform(torch.utils.data.TensorDataset(
                torch.FloatTensor(x_train),
                torch.LongTensor(y_train),
                torch.FloatTensor(c_train),
            )),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
        val_dl = DataLoader(
            dataset_transform(torch.utils.data.TensorDataset(
                torch.FloatTensor(x_val),
                torch.LongTensor(y_val),
                torch.FloatTensor(c_val),
            )),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

    test_dl = load_data(
        split="test",
        batch_size=batch_size,
        image_size=image_size,
        root_dir=root_dir,
        num_workers=num_workers,
        dataset_transform=dataset_transform,
        dataset_size=dataset_size,
        augment_data=False,
        concept_transform=concept_transform,
        additional_sample_transform=test_sample_transform,
        seed=seed,
        config=dataset_opts,
    )

    n_tasks = len(train_dl.dataset.dataset.class_names) if isinstance(train_dl.dataset, Subset) else len(train_dl.dataset.class_names)

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
    if root_dir is None:
        root_dir = DATASET_DIR
    seed_everything(seed)

    concept_names = _load_concept_semantics(root_dir, config)
    selected_concepts, _ = _compute_selected_concepts(
        config,
        root_dir,
        len(concept_names),
        rerun=rerun,
    )
    return list(np.array(concept_names)[selected_concepts])
