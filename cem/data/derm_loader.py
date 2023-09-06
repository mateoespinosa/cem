import numpy as np
import os
import pandas as pd
import sklearn.model_selection
import torch
import torchvision.transforms as transforms

from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

NUM_CONCEPTS = 8

# Derm7pt is obtained from : https://derm.cs.sfu.ca/Welcome.html
DATASET_DIR = os.environ.get("DATASET_DIR", 'cem/data/')

# Derm data constants
DATASET_DIR = "/path/to/derm7pt/"

class Derm7ptDataset(object):
    def __init__(
        self,
        fold="train",
        base_dir=DATASET_DIR,
        use_full_concepts=False,
        transform=None,
        image_key="derm",
        unc_value=0.5,
        label_transform=None,
        concept_transform=None,
        label_key="diagnosis",
        label_generating_fn=None,
    ):
        self.use_full_concepts = use_full_concepts
        self.concept_transform = concept_transform
        self.label_transform = label_transform
        self.label_key = label_key
        if fold == "train":
            indexes = list(
                pd.read_csv(
                    os.path.join(base_dir, "meta", "train_indexes.csv")
                )['indexes']
            )
        elif fold == "test":
            indexes =list(
                pd.read_csv(
                    os.path.join(base_dir, "meta", "valid_indexes.csv")
                )['indexes']
            )
        else:
            raise ValueError(f"Invalid fold {fold}")
        self.posible_concept_vals = {}
        self.concept_map = {}
        self.meta = pd.read_csv(os.path.join(base_dir, "meta", "meta.csv"))

        current_index = 0
        self.posible_concept_vals["TypicalPigmentNetwork"] = [0, 1, 2]
        if self.use_full_concepts:
            self.concept_map["TypicalPigmentNetwork"] = \
                [current_index, current_index + 1, current_index + 2]
            current_index += 3
        else:
            self.concept_map["TypicalPigmentNetwork"] = [current_index]
            current_index += 1
        self.meta["TypicalPigmentNetwork"] = self.meta.apply(
            lambda row: {"absent": 0, "typical": 1, "atypical": 2}[row["pigment_network"]],
            axis=1,
        )

        self.posible_concept_vals["RegularStreaks"] = [0, 1, 2]
        if self.use_full_concepts:
            self.concept_map["RegularStreaks"] = [current_index, current_index + 1, current_index + 2]
            current_index += 3
        else:
            self.concept_map["RegularStreaks"] = [current_index]
            current_index += 1
        self.meta["RegularStreaks"] = self.meta.apply(
            lambda row: {"absent": 0, "regular": 1, "irregular": 2}[row["streaks"]],
            axis=1,
        )

        self.posible_concept_vals["RegressionStructures"] = [1]
        self.concept_map["RegressionStructures"] = [current_index]
        current_index += 1
        self.meta["RegressionStructures"] = self.meta.apply(
            lambda row: (1 - int(row["regression_structures"] == "absent")),
            axis=1,
        )


        self.posible_concept_vals["RegularDG"] = [0, 1, 2]
        if self.use_full_concepts:
            self.concept_map["RegularDG"] = [current_index, current_index + 1, current_index + 2]
            current_index += 3
        else:
            self.concept_map["RegularDG"] = [current_index]
            current_index += 1
        self.meta["RegularDG"] = self.meta.apply(
            lambda row: {"absent": 0, "regular": 1, "irregular": 2}[row["dots_and_globules"]],
            axis=1,
        )

        self.posible_concept_vals["BWV"] = [1]
        self.concept_map["BWV"] = [current_index]
        current_index += 1
        self.meta["BWV"] = self.meta.apply(
            lambda row: {"absent": 0, "present": 1}[row["blue_whitish_veil"]],
            axis=1,
        )

        self.n_concepts = current_index
        self.label_map = {}
        if label_generating_fn is not None:
            self.label_col = self.meta.apply(
                lambda row: label_generating_fn(row[self.label_key]),
                axis=1,
            )
        else:
            self.label_col = self.meta[self.label_key]
        for val in self.label_col.unique():
            self.label_map[val] = len(self.label_map)
        self.meta = self.meta.iloc[indexes]
        self.transform = transform
        self.base_dir = base_dir
        self.image_key = image_key
        self.concepts = [
            "BWV",
            "RegularDG",
            # "IrregularDG",
            "RegressionStructures",
            # "IrregularStreaks",
            "RegularStreaks",
            # "AtypicalPigmentNetwork",
            "TypicalPigmentNetwork",
        ]


    def num_classes(self):
        return len(self.label_map) if len(self.label_map) > 2 else 1

    def _get_concepts(self, idx):
        result = []
        if self.use_full_concepts:
            for c_name in self.concepts:
                for posible_value in self.posible_concept_vals[c_name]:
                    if self.meta.iloc[idx][c_name] == posible_value:
                        result.append(1)
                    else:
                        result.append(0)
        else:
            for c_name in self.concepts:
                result.append(self.meta.iloc[idx][c_name])
        return np.array(result)

    def _get_label(self, idx):
        return np.array(
            self.label_map[self.label_col.iloc[idx]]
        )

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.meta.iloc[idx]
        img_path = os.path.join(
            self.base_dir,
            'images/',
            row[self.image_key],
        )
        image = Image.open(img_path).convert('RGB')
        class_label = self._get_label(idx)
        if self.label_transform:
            class_label = self.label_transform(class_label)
        if self.transform:
            image = self.transform(image)
        if self.num_classes() == 1:
            class_label = float(class_label)
        concepts = self._get_concepts(idx)
        if self.concept_transform is not None:
            concepts = self.concept_transform(concepts)
        return image, class_label, torch.FloatTensor(concepts)


def load_data(
    batch_size,
    fold="train",
    root_dir=DATASET_DIR,
    resol=299,
    num_workers=1,
    concept_transform=None,
    label_transform=None,
    unc_value=0.5,
    label_key="diagnosis",
    use_full_concepts=False,
    label_generating_fn=None,
):
    resized_resol = int(resol * 256/224)
    is_training = fold == "train"
    if is_training:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])

    dataset = Derm7ptDataset(
        fold=fold,
        base_dir=root_dir,
        transform=transform,
        unc_value=2 if use_full_concepts else unc_value,
        concept_transform=concept_transform,
        label_transform=label_transform,
        label_key=label_key,
        use_full_concepts=use_full_concepts,
        label_generating_fn=label_generating_fn,
    )
    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return loader

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
    label_generating_fn = None
    if config.get("cancer_binary_label", False):
        label_generating_fn = lambda x: int(
            ("clark nevus" in x)
        )
    seed_everything(seed)
    sampling_percent = config.get("sampling_percent", 1)
    sampling_groups = config.get("sampling_groups", False)
    pre_dl = load_data(
        batch_size=config['batch_size'],
        fold="train",
        root_dir=root_dir,
        resol=299,
        num_workers=config['num_workers'],
        unc_value=config.get('unc_value', 0.5),
        label_key=config.get('label_key', 'diagnosis'),
        use_full_concepts=config.get('use_full_concepts', False),
        label_generating_fn=label_generating_fn,
    )
    n_concepts = pre_dl.dataset.n_concepts
    concept_group_map = pre_dl.dataset.concept_map.copy()
    n_concepts = n_concepts
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
        for k, v in concept_group_map.items():
            print(f"\t\t\t{k} -> {v}")

        def concept_transform(sample):
            if isinstance(sample, list):
                sample = np.array(sample)
            return sample[selected_concepts]

        # And correct the weight imbalance
        if config.get('weight_loss', False):
            imbalance = np.array(imbalance)[selected_concepts]
        n_concepts = len(selected_concepts)
    else:
        concept_transform = None


    og_train_dl = load_data(
        batch_size=config['batch_size'],
        fold="train",
        root_dir=root_dir,
        resol=299,
        num_workers=config['num_workers'],
        concept_transform=concept_transform,
        unc_value=config.get('unc_value', 0.5),
        label_key=config.get('label_key', 'diagnosis'),
        use_full_concepts=config.get('use_full_concepts', False),
        label_generating_fn=label_generating_fn,
    )
    num_classes = og_train_dl.dataset.num_classes()
    val_size = config.get("val_size", 0.1)
    if not config.get("train_subsampling", 1) in [1, 0, None]:
        percent = config.get("train_subsampling", 1)
        file_name = os.path.join(
            root_dir,
            f"train_valsize_{val_size}_idxs_subsample_{percent}.npy",
        )
        if os.path.exists(
            os.path.join(
                root_dir,
                f"train_valsize_{val_size}_initially_selected_{percent}.npy",
            )
        ):
            full_train_idxs = np.load(os.path.join(
                root_dir,
                f"train_valsize_{val_size}_initially_selected_{percent}.npy",
            ))
        else:
            full_train_idxs = np.random.choice(
                list(range(len(og_train_dl.dataset))),
                size=int(np.ceil(len(og_train_dl.dataset) * percent)),
                replace=False,
            )
            np.save(
                os.path.join(
                    root_dir,
                    f"train_valsize_{val_size}_initially_selected_{percent}.npy"
                ),
                full_train_idxs,
            )
    else:
        file_name = os.path.join(root_dir, f"train_valsize_{val_size}_idxs.npy")
        full_train_idxs = list(range(len(og_train_dl.dataset)))
    if os.path.exists(file_name):
        train_idxs = np.load(file_name)
        val_idxs = np.load(file_name.replace(f"train_valsize_{val_size}_idxs", f"val_idxs"))
    else:
        train_idxs = full_train_idxs
        train_idxs, val_idxs = sklearn.model_selection.train_test_split(
            train_idxs,
            test_size=val_size,
            random_state=42,
        )
        np.save(file_name, train_idxs)
        np.save(
            file_name.replace(f"train_valsize_{val_size}_idxs", "val_idxs"),
            val_idxs,
        )

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
    test_dl = load_data(
        batch_size=config['batch_size'],
        fold="test",
        root_dir=root_dir,
        resol=299,
        num_workers=config['num_workers'],
        concept_transform=concept_transform,
        unc_value=config.get('unc_value', 0.5),
        label_key=config.get('label_key', 'diagnosis'),
        use_full_concepts=config.get('use_full_concepts', False),
        label_generating_fn=label_generating_fn,
    )

    # Finally, determine whether or not we will need to compute the imbalance
    # factors
    if config.get('weight_loss', False):
        attribute_count = np.zeros((n_concepts,))
        samples_seen = 0
        for i, (_, y, c) in enumerate(train_dl):
            c = c.cpu().detach().numpy()
            attribute_count += np.sum(c, axis=0)
            samples_seen += c.shape[0]
        imbalance = samples_seen / attribute_count - 1
    else:
        imbalance = None

    if not output_dataset_vars:
        return train_dl, val_dl, test_dl, imbalance
    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        (n_concepts, num_classes, concept_group_map),
    )