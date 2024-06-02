import copy
import logging
import numpy as np
import os
import pytorch_lightning
import random
import sklearn.model_selection
import torch
import torchvision


def _color_digit(
    x,
    color_distr_label,
    rng,
    colors=["red", "green", "blue"],
    digit_color_distribution=None,
    seed=42,
    normalize_samples=True,
):
    if (digit_color_distribution is not None) and (
        color_distr_label in digit_color_distribution
    ):
        color_distr = digit_color_distribution[color_distr_label]
    else:
        # Otherwise we select a color uniformly at random
        color_distr = [1/len(colors) for _ in colors]
    selected_color_idx = rng.choice(
        range(len(colors)),
        p=color_distr,
    )
    group = selected_color_idx
    color = colors[selected_color_idx]
    corresponding_sample = np.transpose(x, (0, 3, 1, 2))
    rest_sample = np.zeros_like(corresponding_sample)
    if normalize_samples:
        corresponding_sample = corresponding_sample.copy() / 255.0

    # Now time to generate the color using RGB maps!
    if color == "red":
        new_digit = np.concatenate(
            [
                corresponding_sample,
                rest_sample,
                rest_sample,
            ],
            axis=1,
        )
    elif color == "green":
        new_digit = np.concatenate(
            [
                rest_sample,
                corresponding_sample,
                rest_sample,
            ],
            axis=1,
        )
    elif color == "blue":
        new_digit = np.concatenate(
            [
                rest_sample,
                rest_sample,
                corresponding_sample,
            ],
            axis=1,
        )
    elif color == "white":
        new_digit = np.concatenate(
            [
                corresponding_sample,
                corresponding_sample,
                corresponding_sample,
            ],
            axis=1,
        )
    elif color == "gray":
        new_map = (
            corresponding_sample / 2 if normalize_samples
            else corresponding_sample // 2
        )
        new_digit = np.concatenate(
            [new_map, new_map, new_map],
            axis=1,
        )
    elif color.startswith("random_"):
        # If it is random, then we randomly generate an RGB color using the
        # seed provided after the word "random_"
        use_color_seed = int(color[len("random_"):])
        rng_2 = np.random.default_rng(seed=use_color_seed)
        selected_color = rng_2.uniform(0, 1, size=3)
        new_digit = np.concatenate(
            [
                corresponding_sample * selected_color[0],
                corresponding_sample * selected_color[1],
                corresponding_sample * selected_color[2],
            ],
            axis=1,
        )
    else:
        raise ValueError(
            f'Unsupported color name {color}'
        )
    return new_digit, group

def produce_addition_set(
    X,
    y,
    dataset_size=30000,
    num_operands=2,
    selected_digits=list(range(10)),
    img_format='channels_first',
    sample_concepts=None,
    normalize_samples=True,
    concat_dim='y',
    even_concepts=False,
    even_labels=False,
    count_labels=False,
    threshold_labels=None,
    concept_transform=None,
    noise_level=0.0,
    low_noise_level=0.0,
    colors=["gray"],
    digit_color_distribution=None,
    seed=42,
    color_by_label=False,
    count_digit=None,
    condition=lambda x: x,
):
    filter_idxs = []
    rng = np.random.default_rng(seed=seed)
    if len(y.shape) == 2 and y.shape[-1] == 1:
        y = y[:, 0]
    if not isinstance(selected_digits[0], list):
        selected_digits = [selected_digits[:] for _ in range(num_operands)]
    elif len(selected_digits) != num_operands:
        raise ValueError(
            "If selected_digits is a list of lists, it must have the same "
            "length as num_operands"
        )

    operand_remaps = [
        dict((dig, idx) for (idx, dig) in enumerate(operand_digits))
        for operand_digits in selected_digits
    ]
    total_operand_samples = []
    total_operand_labels = []
    for allowed_digits in selected_digits:
        filter_idxs = []
        for idx, digit in enumerate(y):
            if digit in allowed_digits:
                filter_idxs.append(idx)
        total_operand_samples.append(X[filter_idxs, :, :, :])
        total_operand_labels.append(y[filter_idxs])

    sum_train_samples = []
    sum_train_concepts = []
    sum_train_labels = []
    sum_train_colors = []
    for i in range(dataset_size):
        operands = []
        digit_identities = []
        concepts = []
        sample_label = 0
        selected = []
        used_colors = []
        sum_train_colors.append(used_colors)
        for operand_digits, remap, total_samples, total_labels in zip(
            selected_digits,
            operand_remaps,
            total_operand_samples,
            total_operand_labels,
        ):
            img_idx = rng.choice(total_samples.shape[0])
            selected.append(total_labels[img_idx])
            img = total_samples[img_idx: img_idx + 1, :, :, :].copy()
            if len(operand_digits) > 2:
                if even_concepts:
                    concept_vals = np.array([[
                        int((remap[total_labels[img_idx]] % 2) == 0)
                    ]])
                else:
                    concept_vals = torch.nn.functional.one_hot(
                        torch.LongTensor([remap[total_labels[img_idx]]]),
                        num_classes=len(operand_digits)
                    ).numpy()
                concepts.append(concept_vals)
            else:
                # Else we will treat it as a simple binary concept (this allows
                # us to train models that do not have mutually exclusive
                # concepts!)
                if even_concepts:
                    concepts.append(np.array([[
                        int((total_labels[img_idx] % 2) == 0)
                    ]]))
                else:
                    max_bound = np.max(operand_digits)
                    val = int(total_labels[img_idx] == max_bound)
                    concepts.append(np.array([[val]]))
            sample_label += total_labels[img_idx]
            operands.append(img)
            digit_identities.append(total_labels[img_idx])
        if even_labels:
            sum_train_labels.append(int(sample_label % 2 == 0))
        elif count_labels:
            sum_train_labels.append(int(np.sum(np.concatenate(concepts, axis=-1))))
        else:
            sum_train_labels.append(sample_label)

        if threshold_labels is not None:
            sum_train_labels[-1] = int(
                sum_train_labels[-1] >= threshold_labels
            )

        for operand_idx, img in enumerate(operands):
            img, color = _color_digit(
                img,
                color_distr_label=(
                    sum_train_labels[-1] if color_by_label
                    else digit_identities[operand_idx]
                ),
                rng=rng,
                colors=colors,
                digit_color_distribution=digit_color_distribution,
                normalize_samples=normalize_samples,
            )
            used_colors.append(color)
            operands[operand_idx] = img

        if concat_dim == 'y':
            sum_train_samples.append(np.concatenate(operands, axis=2))
        elif concat_dim == 'x':
            sum_train_samples.append(np.concatenate(operands, axis=3))
        else:
            raise ValueError(f'Unsupported concat dim {concat_dim}')
        sum_train_concepts.append(np.concatenate(concepts, axis=-1))

    sum_train_samples = np.concatenate(sum_train_samples, axis=0)
    sum_train_concepts = np.concatenate(sum_train_concepts, axis=0)
    sum_train_colors = np.array(sum_train_colors)
    sum_train_labels = np.array(sum_train_labels)
    if img_format == 'channels_last':
        sum_train_samples = np.transpose(sum_train_samples, axes=[0, 2, 3, 1])
    if sample_concepts is not None:
        sum_train_concepts = sum_train_concepts[:, sample_concepts]
    if concept_transform is not None:
        sum_train_concepts = concept_transform(sum_train_concepts)
    if noise_level > 0.0:
        mask = rng.choice(
            [0, 1],
            size=sum_train_samples.shape,
            p=[1 - noise_level, noise_level],
        )
        substitutes = rng.uniform(
            low=0,
            high=low_noise_level,
            size=sum_train_samples.shape,
        )
        sum_train_samples = mask * substitutes + (1 - mask) * sum_train_samples
    return sum_train_samples, sum_train_labels, sum_train_concepts, sum_train_colors

def load_color_mnist_addition(
    cache_dir="mnist",
    seed=42,
    train_dataset_size=30000,
    test_dataset_size=10000,
    num_operands=10,
    selected_digits=list(range(10)),
    val_percent=0.2,
    batch_size=512,
    test_only=False,
    num_workers=-1,
    sample_concepts=None,
    img_format='channels_first',
    even_concepts=False,
    even_labels=False,
    count_labels=False,
    count_digit=None,
    threshold_labels=None,
    concept_transform=None,
    low_noise_level=0.0,
    noise_level=0.0,
    test_noise_level=None,
    test_low_noise_level=None,
    colors=["gray"],
    digit_color_distribution=None,
    test_digit_color_distribution=None,
    spurious_strength=0,
    color_by_label=False,
):
    if digit_color_distribution:
        real_distr_map = copy.deepcopy(digit_color_distribution)
        for digit, distr_vals in real_distr_map.items():
            if "spurious" in distr_vals:
                color_index = distr_vals.index("spurious")
                other_probs = (1 - spurious_strength)/(len(colors) - 1)
                real_distr_map[digit] = [
                    spurious_strength if i == color_index else other_probs
                    for i in range(len(colors))
                ]
    else:
        real_distr_map = {
            digit: [1/len(colors) for _ in colors]
            for digit in selected_digits
        }
    digit_color_distribution = real_distr_map
    if test_digit_color_distribution is None:
        test_digit_color_distribution = digit_color_distribution
    else:
        test_real_distr_map = copy.deepcopy(test_digit_color_distribution)
        for digit, distr_vals in test_real_distr_map.items():
            if "spurious" in distr_vals:
                color_index = distr_vals.index("spurious")
                other_probs = (1 - spurious_strength)/(len(colors) - 1)
                test_real_distr_map[digit] = [
                    spurious_strength if i == color_index else other_probs
                    for i in range(len(colors))
                ]
        test_digit_color_distribution = test_real_distr_map

    test_noise_level = (
        test_noise_level if (test_noise_level is not None) else noise_level
    )
    test_low_noise_level = (
        test_low_noise_level if (test_low_noise_level is not None)
        else low_noise_level
    )
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    pytorch_lightning.utilities.seed.seed_everything(seed)

    concept_groups = []
    for operand_digits in selected_digits:
        concept_groups.append(list(range(
            len(concept_groups),
            len(concept_groups) + len(operand_digits),
        )))

    ds_test = torchvision.datasets.MNIST(
        cache_dir,
        train=False,
        download=True,
    )

    # Put all the images into a single np array for easy
    # manipulation
    x_test = []
    y_test = []

    for x, y in ds_test:
        x_test.append(np.expand_dims(
            np.expand_dims(x, axis=0),
            axis=-1,
        ))
        y_test.append(np.expand_dims(
            np.expand_dims(y, axis=0),
            axis=-1,
        ))
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Wrap them up in dataloaders
    x_test, y_test, c_test, g_test = produce_addition_set(
        X=x_test,
        y=y_test,
        dataset_size=test_dataset_size,
        num_operands=num_operands,
        selected_digits=selected_digits,
        sample_concepts=sample_concepts,
        img_format=img_format,
        concat_dim='y',
        even_concepts=even_concepts,
        even_labels=even_labels,
        count_labels=count_labels,
        count_digit=count_digit,
        threshold_labels=threshold_labels,
        concept_transform=concept_transform,
        noise_level=test_noise_level,
        low_noise_level=test_low_noise_level,
        colors=colors,
        digit_color_distribution=test_digit_color_distribution,
        seed=seed,
        color_by_label=color_by_label,
    )
    x_test = torch.FloatTensor(x_test)
    if even_labels or (threshold_labels is not None) or (
        count_labels and (len(selected_digits) == 2)
    ):
        y_test = torch.FloatTensor(y_test)
    else:
        y_test = torch.LongTensor(y_test)
    c_test = torch.FloatTensor(c_test)
    g_test = torch.LongTensor(g_test)
    test_data = torch.utils.data.TensorDataset(
        x_test,
        y_test,
        c_test,
        g_test,
    )
    test_dl = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if test_only:
        return test_dl



    # Now time to do the same for the train/val datasets!
    ds_train = torchvision.datasets.MNIST(
        cache_dir,
        train=True,
        download=True,
    )


    x_train = []
    y_train = []

    for x, y in ds_train:
        x_train.append(np.expand_dims(
            np.expand_dims(x, axis=0),
            axis=-1,
        ))
        y_train.append(np.expand_dims(
            np.expand_dims(y, axis=0),
            axis=-1,
        ))

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)


    if val_percent:
        x_train, x_val, y_train, y_val = \
            sklearn.model_selection.train_test_split(
                x_train,
                y_train,
                test_size=val_percent,
            )
        x_val, y_val, c_val, g_val = produce_addition_set(
            X=x_val,
            y=y_val,
            dataset_size=int(train_dataset_size*val_percent),
            num_operands=num_operands,
            selected_digits=selected_digits,
            sample_concepts=sample_concepts,
            img_format=img_format,
            concat_dim='y',
            even_concepts=even_concepts,
            even_labels=even_labels,
            count_labels=count_labels,
            count_digit=count_digit,
            threshold_labels=threshold_labels,
            concept_transform=concept_transform,
            noise_level=noise_level,
            low_noise_level=low_noise_level,
            colors=colors,
            digit_color_distribution=digit_color_distribution,
            seed=seed,
            color_by_label=color_by_label,
        )
        x_val = torch.FloatTensor(x_val)
        if even_labels or (threshold_labels is not None) or (
            count_labels and (len(selected_digits) == 2)
        ):
            y_val = torch.FloatTensor(y_val)
        else:
            y_val = torch.LongTensor(y_val)
        g_val = torch.LongTensor(g_val)
        c_val = torch.FloatTensor(c_val)
        val_data = torch.utils.data.TensorDataset(x_val, y_val, c_val, g_val)
        val_dl = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        val_dl = None

    x_train, y_train, c_train, g_train = produce_addition_set(
        X=x_train,
        y=y_train,
        dataset_size=train_dataset_size,
        num_operands=num_operands,
        selected_digits=selected_digits,
        sample_concepts=sample_concepts,
        img_format=img_format,
        concat_dim='y',
        even_concepts=even_concepts,
        even_labels=even_labels,
        count_labels=count_labels,
        count_digit=count_digit,
        threshold_labels=threshold_labels,
        concept_transform=concept_transform,
        noise_level=noise_level,
        low_noise_level=low_noise_level,
        colors=colors,
        digit_color_distribution=digit_color_distribution,
        seed=seed,
        color_by_label=color_by_label,
    )
    x_train = torch.FloatTensor(x_train)
    if even_labels or (threshold_labels is not None) or (
        count_labels and (len(selected_digits) == 2)
    ):
        y_train = torch.FloatTensor(y_train)
    else:
        y_train = torch.LongTensor(y_train)
    c_train = torch.FloatTensor(c_train)
    g_train = torch.LongTensor(g_train)
    train_data = torch.utils.data.TensorDataset(
        x_train,
        y_train,
        c_train,
        g_train,
    )
    train_dl = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if val_dl is not None:
        return train_dl, val_dl, test_dl
    return train_dl, test_dl


def generate_data(
    config,
    root_dir="mnist",
    seed=42,
    output_dataset_vars=False,
    rerun=False,
):
    selected_digits = config.get("selected_digits", list(range(2)))
    num_operands = config.get("num_operands", 32)
    if not isinstance(selected_digits[0], list):
        selected_digits = [selected_digits[:] for _ in range(num_operands)]
    elif len(selected_digits) != num_operands:
        raise ValueError(
            "If selected_digits is a list of lists, it must have the same "
            "length as num_operands"
        )
    even_concepts = config.get("even_concepts", False)
    even_labels = config.get("even_labels", False)
    count_labels = config.get('count_labels', False)
    threshold_labels = config.get("threshold_labels", None)

    if even_concepts:
        num_concepts = num_operands
        concept_group_map = {
            i: [i] for i in range(num_operands)
        }
    else:
        num_concepts = 0
        concept_group_map = {}
        n_tasks = 1 # Zero is always included as a possible sum
        for operand_idx, used_operand_digits in enumerate(selected_digits):
            num_curr_concepts = len(used_operand_digits) if len(used_operand_digits) > 2 else 1
            concept_group_map[operand_idx] = list(range(num_concepts, num_concepts + num_curr_concepts))
            num_concepts += num_curr_concepts
            n_tasks += np.max(used_operand_digits)

    if even_labels or (threshold_labels is not None):
        n_tasks = 1
    elif count_labels:
        n_tasks = num_operands + 1

    sampling_percent = config.get("sampling_percent", 1)
    sampling_groups = config.get("sampling_groups", False)

    if sampling_percent != 1:
        # Do the subsampling
        if sampling_groups:
            new_n_groups = int(np.ceil(len(concept_group_map) * sampling_percent))
            selected_groups_file = os.path.join(
                root_dir,
                f"selected_groups_sampling_{sampling_percent}_operands_{num_operands}.npy",
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
            new_n_concepts = len(selected_concepts)
        else:
            new_n_concepts = int(np.ceil(num_concepts * sampling_percent))
            selected_concepts_file = os.path.join(
                root_dir,
                f"selected_concepts_sampling_{sampling_percent}_operands_{num_operands}.npy",
            )
            if (not rerun) and os.path.exists(selected_concepts_file):
                selected_concepts = np.load(selected_concepts_file)
            else:
                selected_concepts = sorted(
                    np.random.permutation(num_concepts)[:new_n_concepts]
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
        def concept_transform(sample):
            return sample[:, selected_concepts]
        num_concepts = new_n_concepts
        concept_group_map = new_concept_group
        logging.debug("\t\tSelected concepts:", selected_concepts)
        logging.debug(
            f"\t\tUpdated concept group map "
            f"(with {len(concept_group_map)} groups):"
        )
        for k, v in concept_group_map.items():
            logging.debug(f"\t\t\t{k} -> {v}")
    else:
        concept_transform = None
    train_dl, val_dl, test_dl = load_color_mnist_addition(
        cache_dir=root_dir,
        seed=seed,
        train_dataset_size=config.get("train_dataset_size", 30000),
        test_dataset_size=config.get("test_dataset_size", 10000),
        num_operands=num_operands,
        selected_digits=selected_digits,
        spurious_strength=config.get("spurious_strength", 0),
        val_percent=config.get("val_percent", 0.2),
        batch_size=config.get("batch_size", 512),
        test_only=config.get("test_only", False),
        num_workers=config.get("num_workers", -1),
        sample_concepts=config.get("sample_concepts", None),
        img_format=config.get("img_format", 'channels_first'),
        even_labels=even_labels,
        count_labels=count_labels,
        count_digit=config.get('count_digit', None),
        threshold_labels=threshold_labels,
        even_concepts=even_concepts,
        concept_transform=concept_transform,
        noise_level=config.get("noise_level", 0),
        low_noise_level=config.get("low_noise_level", 0),
        test_noise_level=config.get(
            "test_noise_level",
            config.get("noise_level", 0),
        ),
        test_low_noise_level=config.get(
            "test_low_noise_level",
            config.get("low_noise_level", 0),
        ),
        colors=config.get("colors", ['gray']),
        digit_color_distribution=config.get("digit_color_distribution", None),
        test_digit_color_distribution=config.get("test_digit_color_distribution", None),
        color_by_label=config.get('color_by_label', False),
    )

    if config.get('weight_loss', False):
        attribute_count = np.zeros((num_concepts,))
        samples_seen = 0
        for i, data in enumerate(train_dl):
            if len(data) == 2:
                (_, (_, c)) = data
            else:
                c = data[2]
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
        (num_concepts, n_tasks, concept_group_map)
    )