import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from collections import defaultdict
import sklearn.model_selection

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from cem.data.CUB200.cub_loader import load_data

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 1
N_CONCEPTS = 13


# IMPORANT NOTE: THIS DATASET NEEDS TO BE DOWNLOADED FIRST BEFORE BEING ABLE
#                TO RUN ANY CUB EXPERIMENTS!!
#                Instructions on how to download it can be found
#                in the original CBM paper's repository
#                found here: https://github.com/yewsiang/ConceptBottleneck
# CAN BE OVERWRITTEN WITH AN ENV VARIABLE DATASET_DIR
DATASET_DIR = os.environ.get("DATASET_DIR", 'cem/data/')



##########################################################
## SIMPLIFIED LOADER FUNCTION FOR STANDARDIZATION
##########################################################


def generate_data(
    config,
    root_dir=DATASET_DIR,
    seed=42,
    output_dataset_vars=False,
    rerun=False,
):
    if root_dir is None:
        root_dir = DATASET_DIR
    base_dir = os.path.join(root_dir, 'CheXpert-v1.0-small/metadata')
    seed_everything(seed)
    train_data_path = os.path.join(base_dir, 'train.pkl')

    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    test_data_path = train_data_path.replace('train.pkl', 'test.pkl')
    sampling_percent = config.get("sampling_percent", 1)
    sampling_groups = config.get("sampling_groups", False)

    n_concepts = N_CONCEPTS
    n_tasks = 1
    concept_group_map = {
        i: [i] for i in range(n_concepts)
    }
    def prev_concept_transform(sample):
        if isinstance(sample, list):
            sample = np.array(sample)
        return np.where(np.logical_or(sample == 1, sample == 0), sample, 0.5)
    if sampling_percent != 1:
        # Do the subsampling
        if sampling_groups:
            new_n_groups = int(np.ceil(len(concept_group_map) * sampling_percent))
            selected_groups_file = os.path.join(
                DATASET_DIR,
                f"selected_groups_sampling_{sampling_percent}.npy",
            )
            if (not rerun) and os.path.exists(selected_groups_file):
                selected_groups = np.load(selected_groups_file)
            else:
                selected_groups = sorted(
                    np.random.permutation(len(concept_group_map))[:new_n_groups]
                )
                np.save(selected_groups_file, selected_groups)
            selected_concepts = []
            group_concepts = [x[1] for x in concept_group_map.items()]
            for group_idx in selected_groups:
                selected_concepts.extend(group_concepts[group_idx])
            selected_concepts = sorted(set(selected_concepts))
        else:
            new_n_concepts = int(np.ceil(n_concepts * sampling_percent))
            selected_concepts_file = os.path.join(
                DATASET_DIR,
                f"selected_concepts_sampling_{sampling_percent}.npy",
            )
            if (not rerun) and os.path.exists(selected_concepts_file):
                selected_concepts = np.load(selected_concepts_file)
            else:
                selected_concepts = sorted(
                    np.random.permutation(n_concepts)[:new_n_concepts]
                )
                np.save(selected_concepts_file, selected_concepts)
        # Then we also have to update the concept group map so that
        # selected concepts that were previously in the same concept
        # group are maintained in the same concept group
        new_concept_group = {}
        remap = dict((y, x) for (x, y) in enumerate(selected_concepts))
        selected_concepts_set = set(selected_concepts)
        for selected_concept in selected_concepts:
            for concept_group_name, group_concepts in concept_group_map.items():
                if selected_concept in group_concepts:
                    if concept_group_name in new_concept_group:
                        # Then we have already added this group
                        continue
                    # Then time to add this group!
                    new_concept_group[concept_group_name] = []
                    for other_concept in group_concepts:
                        if other_concept in selected_concepts_set:
                            # Add the remapped version of this concept
                            # into the concept group
                            new_concept_group[concept_group_name].append(
                                remap[other_concept]
                            )
        # And update the concept group map accordingly
        concept_group_map = new_concept_group
        print("\t\tSelected concepts:", selected_concepts)
        print(f"\t\tUpdated concept group map (with {len(concept_group_map)} groups):")
        for k, v in concept_group_map.items():
            print(f"\t\t\t{k} -> {v}")

        def concept_transform(sample):
            if isinstance(sample, list):
                sample = np.array(sample)
            return prev_concept_transform(sample[selected_concepts])

        # And correct the weight imbalance
        if config.get('weight_loss', False):
            imbalance = np.array(imbalance)[selected_concepts]
        n_concepts = len(selected_concepts)
    else:
        concept_transform = prev_concept_transform

    og_train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=config['batch_size'],
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=config.get("stratified_sampling", False),
        root_dir=root_dir,
        num_workers=config['num_workers'],
        concept_transform=concept_transform,
        path_transform=lambda path: os.path.join(root_dir, path),
        label_transform=lambda x: float(x),
    )
    if not config.get("train_subsampling", 1) in [1, 0, None]:
        percent = config.get("train_subsampling", 1)
        if config.get("expand_test", 0) != 0:
            file_name = os.path.join(root_dir, f"train_idxs_subsample_{percent}_expand_test_{config['expand_test']}.npy")
        else:
            file_name = os.path.join(root_dir, f"train_idxs_subsample_{percent}.npy")
        if os.path.exists(
            os.path.join(root_dir, f"train_initially_selected_{percent}.npy")
        ):
            full_train_idxs = np.load(os.path.join(root_dir, f"train_initially_selected_{percent}.npy"))
        else:
            full_train_idxs = np.random.choice(
                list(range(len(og_train_dl.dataset))),
                size=int(np.ceil(len(og_train_dl.dataset) * percent)),
                replace=False,
            )
            np.save(os.path.join(root_dir, f"train_initially_selected_{percent}.npy"), full_train_idxs)
    else:
        if config.get("expand_test", 0) != 0:
            file_name = os.path.join(root_dir, f"train_idxs_expand_test_{config['expand_test']}.npy")
        else:
            file_name = os.path.join(root_dir, f"train_idxs.npy")
        full_train_idxs = list(range(len(og_train_dl.dataset)))
    if os.path.exists(file_name):
        train_idxs = np.load(file_name)
        if config.get("expand_test", 0) != 0:
            test_idxs = np.load(file_name.replace("train_idxs", "test_idxs"))
        else:
            test_idxs = None
        val_idxs = np.load(file_name.replace("train_idxs", "val_idxs"))
    else:
        if config.get("expand_test", 0) != 0:
            train_idxs, test_idxs = sklearn.model_selection.train_test_split(
                full_train_idxs,
                test_size=config.get("expand_test", 0),
                random_state=42,
            )
        else:
            train_idxs = full_train_idxs
            test_idxs = None
        train_idxs, val_idxs = sklearn.model_selection.train_test_split(
            train_idxs,
            test_size=0.2,
            random_state=42,
        )
        np.save(file_name, train_idxs)
        if test_idxs is not None:
            np.save(file_name.replace("train_idxs", "test_idxs"), test_idxs)
        np.save(file_name.replace("train_idxs", "val_idxs"), val_idxs)

    val_dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(og_train_dl.dataset, val_idxs),
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )
    train_dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(og_train_dl.dataset, train_idxs),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
    )
    pre_merge_test_dl = load_data(
        pkl_paths=[test_data_path],
        use_attr=True,
        no_img=False,
        batch_size=config['batch_size'],
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
        concept_transform=concept_transform,
        path_transform=lambda path: os.path.join(root_dir, path),
        label_transform=lambda x: float(x),
    )
    if test_idxs is not None:
        test_dl = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(
                [
                    pre_merge_test_dl.dataset,
                    torch.utils.data.Subset(og_train_dl.dataset, test_idxs)
                ],
            ),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
        )
    else:
        test_dl = pre_merge_test_dl

    # Finally, determine whether or not we will need to compute the imbalance factors
    if config.get('weight_loss', False):
        attribute_count = np.zeros((n_concepts,))
        samples_seen = 0
        for i, (_, y, c) in enumerate(train_dl):
            c = c.cpu().detach().numpy()
            attribute_count += np.sum(c, axis=0)
            samples_seen += c.shape[0]
        print("Concept distribution is:", attribute_count / samples_seen)
        imbalance = samples_seen / attribute_count - 1
    else:
        imbalance = None

    if not output_dataset_vars:
        return train_dl, val_dl, test_dl, imbalance
    return train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_group_map)
