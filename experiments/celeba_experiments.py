import argparse
import copy
import joblib
import numpy as np
import os
import torch
import torchvision

from pathlib import Path
from pytorch_lightning import seed_everything
from torchvision import transforms

import cem.train.training as training
import cem.train.utils as utils

###############################################################################
## GLOBAL VARIABLES
###############################################################################

SELECTED_CONCEPTS = [
    2,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    39,
]

CONCEPT_SEMANTICS = [
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young',
]

# IMPORANT NOTE: THIS DATASET NEEDS TO BE DOWNLOADED FIRST BEFORE BEING ABLE
#                TO RUN ANY CUB EXPERIMENTS!!
#                Instructions on how to download it can be found
#                in https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
CELEBA_ROOT = 'data/celeba'


###############################################################################
## MAIN EXPERIMENT LOOP
###############################################################################


def main(
    rerun=False,
    result_dir='results/celeba/',
    project_name='',
    activation_freq=0,
    num_workers=8,
    single_frequency_epochs=0,
    global_params=None,
    save_model=True,
    data_root=CELEBA_ROOT,
):
    seed_everything(42)
    # parameters for data, model, and training
    og_config = dict(
        cv=5,
        max_epochs=200,
        patience=15,
        batch_size=512,
        num_workers=num_workers,
        emb_size=16,
        extra_dims=0,
        concept_loss_weight=1,
        normalize_loss=False,
        learning_rate=0.005,
        weight_decay=4e-05,
        weight_loss=False,
        pretrain_model=True,
        c_extractor_arch="resnet34",
        optimizer="sgd",
        bool=False,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        early_stopping_delta=0.0,
        image_size=64,
        num_classes=1000,
        top_k_accuracy=[3, 5, 10],
        save_model=True,
        use_imbalance=True,
        use_binary_vector_class=True,
        num_concepts=6,
        label_binary_width=1,
        label_dataset_subsample=12,
        num_hidden_concepts=2,
        selected_concepts=False,

        momentum=0.9,
        shared_prob_gen=False,
        sigmoidal_prob=False,
        sigmoidal_embedding=False,
        training_intervention_prob=0.0,
        embeding_activation=None,
        concat_prob=False,
    )

    utils.extend_with_global_params(og_config, global_params or [])
    use_binary_vector_class = og_config.get('use_binary_vector_class', False)
    if use_binary_vector_class:
        # Now reload by transform the labels accordingly
        width = og_config.get('label_binary_width', 5)
        def _binarize(concepts, selected, width):
            result = []
            binary_repr = []
            concepts = concepts[selected]
            for i in range(0, concepts.shape[-1], width):
                binary_repr.append(
                    str(int(np.sum(concepts[i : i + width]) > 0))
                )
            return int("".join(binary_repr), 2)

        celeba_train_data = torchvision.datasets.CelebA(
            root=data_root,
            split='all',
            download=True,
            target_transform=lambda x: x[0].long() - 1,
            target_type=['attr'],
        )

        concept_freq = np.sum(
            celeba_train_data.attr.cpu().detach().numpy(),
            axis=0
        ) / celeba_train_data.attr.shape[0]
        print("Concept frequency is:", concept_freq)
        sorted_concepts = list(map(
            lambda x: x[0],
            sorted(enumerate(np.abs(concept_freq - 0.5)), key=lambda x: x[1]),
        ))
        num_concepts = og_config.get(
            'num_concepts',
            celeba_train_data.attr.shape[-1],
        )
        concept_idxs = sorted_concepts[:num_concepts]
        concept_idxs = sorted(concept_idxs)
        if og_config.get('num_hidden_concepts', 0):
            num_hidden = og_config.get('num_hidden_concepts', 0)
            hidden_concepts = sorted(
                sorted_concepts[
                    num_concepts:min(
                        (num_concepts + num_hidden),
                        len(sorted_concepts)
                    )
                ]
            )
        else:
            hidden_concepts = []
        print("Selecting concepts:", concept_idxs)
        print("\tAnd hidden concepts:", hidden_concepts)
        celeba_train_data = torchvision.datasets.CelebA(
            root=data_root,
            split='all',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(og_config['image_size']),
                transforms.CenterCrop(og_config['image_size']),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=lambda x: [
                torch.tensor(
                    _binarize(
                        x[1].cpu().detach().numpy(),
                        selected=(concept_idxs + hidden_concepts),
                        width=width,
                    ),
                    dtype=torch.long,
                ),
                x[1][concept_idxs].float(),
            ],
            target_type=['identity', 'attr'],
        )
        label_remap = {}
        vals, counts = np.unique(
            list(map(
                lambda x: _binarize(
                    x.cpu().detach().numpy(),
                    selected=(concept_idxs + hidden_concepts),
                    width=width,
                ),
                celeba_train_data.attr
            )),
            return_counts=True,
        )
        for i, label in enumerate(vals):
            label_remap[label] = i

        celeba_train_data = torchvision.datasets.CelebA(
            root=data_root,
            split='all',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(og_config['image_size']),
                transforms.CenterCrop(og_config['image_size']),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=lambda x: [
                torch.tensor(
                    label_remap[_binarize(
                        x[1].cpu().detach().numpy(),
                        selected=(concept_idxs + hidden_concepts),
                        width=width,
                    )],
                    dtype=torch.long,
                ),
                x[1][concept_idxs].float(),
            ],
            target_type=['identity', 'attr'],
        )
        num_classes = len(label_remap)

        # And subsample to reduce its massive size
        factor = og_config.get('label_dataset_subsample', 1)
        if factor != 1:
            train_idxs = np.random.choice(
                np.arange(0, len(celeba_train_data)),
                replace=False,
                size=len(celeba_train_data)//factor,
            )
            print("Subsampling to", len(train_idxs), "elements.")
            celeba_train_data = torch.utils.data.Subset(
                celeba_train_data,
                train_idxs,
            )
    else:
        concept_selection = list(range(0, len(CONCEPT_SEMANTICS)))
        if og_config.get('selected_concepts', False):
            concept_selection = SELECTED_CONCEPTS
        celeba_train_data = torchvision.datasets.CelebA(
            root=data_root,
            split='all',
            download=True,
            target_transform=lambda x: x[0].long() - 1,
            target_type=['identity'],
        )
        vals, counts = np.unique(
            celeba_train_data.identity,
            return_counts=True,
        )
        sorted_labels = list(map(
            lambda x: x[0],
            sorted(zip(vals, counts), key=lambda x: -x[1])
        ))
        print(
            "Selecting",
            og_config['num_classes'],
            "out of",
            len(vals),
            "classes",
        )
        if result_dir:
            Path(result_dir).mkdir(parents=True, exist_ok=True)
            np.save(
                os.path.join(
                    result_dir,
                    f"selected_top_{og_config['num_classes']}_labels.npy",
                ),
                sorted_labels[:og_config['num_classes']],
            )
        label_remap = {}
        for i, label in enumerate(sorted_labels[:og_config['num_classes']]):
            label_remap[label] = i
        print("len(label_remap) =", len(label_remap))

        # Now reload by transform the labels accordingly
        celeba_train_data = torchvision.datasets.CelebA(
            root=data_root,
            split='all',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(og_config['image_size']),
                transforms.CenterCrop(og_config['image_size']),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=lambda x: [
                torch.tensor(
                    # If it is not in our map, then we make it be the token label
                    # og_config['num_classes'] which will be removed afterwards
                    label_remap.get(
                        x[0].cpu().detach().item() - 1,
                        og_config['num_classes']
                    ),
                    dtype=torch.long,
                ),
                x[1][concept_selection].float(),
            ],
            target_type=['identity', 'attr'],
        )
        num_classes = og_config['num_classes']

        train_idxs = np.where(
            list(map(
                lambda x: x.cpu().detach().item() - 1 in label_remap,
                celeba_train_data.identity
            ))
        )[0]
        celeba_train_data = torch.utils.data.Subset(
            celeba_train_data,
            train_idxs,
        )
    total_samples = len(celeba_train_data)
    train_samples = int(0.7 * total_samples)
    test_samples = int(0.2 * total_samples)
    val_samples = total_samples - test_samples - train_samples
    print(
        f"Data split is: {total_samples} = {train_samples} (train) + "
        f"{test_samples} (test) + {val_samples} (validation)"
    )
    celeba_train_data, celeba_test_data, celeba_val_data = \
        torch.utils.data.random_split(
            celeba_train_data,
            [train_samples, test_samples, val_samples],
        )
    train_dl = torch.utils.data.DataLoader(
        celeba_train_data,
        batch_size=og_config['batch_size'],
        shuffle=True,
        num_workers=og_config['num_workers'],
    )
    test_dl = torch.utils.data.DataLoader(
        celeba_test_data,
        batch_size=og_config['batch_size'],
        shuffle=False,
        num_workers=og_config['num_workers'],
    )
    val_dl = torch.utils.data.DataLoader(
        celeba_val_data,
        batch_size=og_config['batch_size'],
        shuffle=False,
        num_workers=og_config['num_workers'],
    )

    if result_dir and activation_freq:
        # Then let's save the testing data for further analysis later on
        out_acts_save_dir = os.path.join(result_dir, "test_embedding_acts")
        Path(out_acts_save_dir).mkdir(parents=True, exist_ok=True)

        for (ds, name) in [
            (test_dl, "test"),
            (val_dl, "val"),
        ]:
            x_total = []
            y_total = []
            c_total = []
            for x, (y, c) in ds:
                x_total.append(x.cpu().detach())
                y_total.append(y.cpu().detach())
                c_total.append(c.cpu().detach())
            x_inputs = np.concatenate(x_total, axis=0)
            print(f"x_{name}.shape =", x_inputs.shape)
            np.save(os.path.join(out_acts_save_dir, f"x_{name}.npy"), x_inputs)

            y_inputs = np.concatenate(y_total, axis=0)
            print(f"y_{name}.shape =", y_inputs.shape)
            np.save(os.path.join(out_acts_save_dir, f"y_{name}.npy"), y_inputs)

            c_inputs = np.concatenate(c_total, axis=0)
            print(f"c_{name}.shape =", c_inputs.shape)
            np.save(os.path.join(out_acts_save_dir, f"c_{name}.npy"), c_inputs)

    label_set = set()
    sample = next(iter(train_dl))
    real_sample = []
    for derp in sample:
        if isinstance(derp, list):
            real_sample += derp
        else:
            real_sample.append(derp)
    sample = real_sample
    print("Sample has", len(sample), "elements.")
    for i, derp in enumerate(sample):
        print("Element", i, "has shape", derp.shape, "and type", derp.dtype)

    print("Training sample shape is:", sample[0].shape)
    print("Training label shape is:", sample[1].shape)
    print("Training concept shape is:", sample[2].shape)

    n_concepts, n_tasks = sample[2].shape[-1], num_classes

    attribute_count = np.zeros((n_concepts,))
    samples_seen = 0
    for i, (_, (y, c)) in enumerate(train_dl):
        print("\rIn batch", i, "we have seen", len(label_set), "classes")
        c = c.cpu().detach().numpy()
        attribute_count += np.sum(c, axis=0)
        samples_seen += c.shape[0]
        for l in y.reshape(-1).cpu().detach():
            label_set.add(l.item())

    print("Found a total of", len(label_set), "classes")
    if og_config.get("use_imbalance", False):
        imbalance = samples_seen / attribute_count - 1
    else:
        imbalance = None
    print("Imbalance:", imbalance)

    os.makedirs(result_dir, exist_ok=True)

    results = {}
    for split in range(og_config["cv"]):
        print(f'Experiment {split+1}/{og_config["cv"]}')
        results[f'{split}'] = {}

        # Trial period for mixture embedding model
        config = copy.deepcopy(og_config)
        config["architecture"] = "MixtureEmbModel"
        config["extra_name"] = f"SharedProb_AdaptiveDropout_NoProbConcat"
        config["shared_prob_gen"] = True
        config["sigmoidal_prob"] = True
        config["sigmoidal_embedding"] = False
        config['training_intervention_prob'] = 0.25
        config['concat_prob'] = False
        config['emb_size'] = config['emb_size']
        config["embeding_activation"] = "leakyrelu"
        mixed_emb_shared_prob_model,  mixed_emb_shared_prob_test_results = \
            training.train_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                split=split,
                result_dir=result_dir,
                rerun=rerun,
                project_name=project_name,
                seed=split,
                activation_freq=activation_freq,
                single_frequency_epochs=single_frequency_epochs,
                imbalance=imbalance,
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            mixed_emb_shared_prob_model,
            mixed_emb_shared_prob_test_results,
        )

        # Trial period for mixture embedding model
        config = copy.deepcopy(og_config)
        config["architecture"] = "MixtureEmbModel"
        config["extra_name"] = f"SharedProb_Adaptive_NoProbConcat"
        config["shared_prob_gen"] = True
        config["sigmoidal_prob"] = True
        config["sigmoidal_embedding"] = False
        config['training_intervention_prob'] = 0.0
        config['concat_prob'] = False
        config['emb_size'] = config['emb_size']
        config["embeding_activation"] = "leakyrelu"
        mixed_emb_shared_prob_model,  mixed_emb_shared_prob_test_results = \
            training.train_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                split=split,
                result_dir=result_dir,
                rerun=rerun,
                project_name=project_name,
                seed=split,
                activation_freq=activation_freq,
                single_frequency_epochs=single_frequency_epochs,
                imbalance=imbalance,
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            mixed_emb_shared_prob_model,
            mixed_emb_shared_prob_test_results,
        )

        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["bool"] = False
        config["extra_dims"] = (config['emb_size'] - 1) * n_concepts
        config["extra_name"] = f"FuzzyExtraCapacity_Logit"
        config["bottleneck_nonlinear"] = "leakyrelu"
        config["sigmoidal_extra_capacity"] = False
        config["sigmoidal_prob"] = False
        extra_fuzzy_logit_model, extra_fuzzy_logit_test_results = \
            training.train_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                split=split,
                result_dir=result_dir,
                rerun=rerun,
                project_name=project_name,
                seed=split,
                activation_freq=activation_freq,
                single_frequency_epochs=single_frequency_epochs,
                imbalance=imbalance,
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            extra_fuzzy_logit_model,
            extra_fuzzy_logit_test_results,
        )

        # fuzzy model
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["extra_name"] = f"Fuzzy"
        config["bool"] = False
        config["extra_dims"] = 0
        config["sigmoidal_extra_capacity"] = False
        config["sigmoidal_prob"] = True
        extra_fuzzy_logit_model, extra_fuzzy_logit_test_results = \
            training.train_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                split=split,
                result_dir=result_dir,
                rerun=rerun,
                project_name=project_name,
                seed=split,
                activation_freq=activation_freq,
                single_frequency_epochs=single_frequency_epochs,
                imbalance=imbalance,
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            extra_fuzzy_logit_model,
            extra_fuzzy_logit_test_results,
        )

        # sequential and independent models
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["extra_name"] = f""
        config["sigmoidal_prob"] = True
        ind_model, ind_test_results, seq_model, seq_test_results = \
            training.train_independent_and_sequential_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                split=split,
                result_dir=result_dir,
                rerun=rerun,
                project_name=project_name,
                seed=split,
                activation_freq=activation_freq,
                single_frequency_epochs=single_frequency_epochs,
                imbalance=imbalance,
            )
        config["architecture"] = "IndependentConceptBottleneckModel"
        training.update_statistics(
            results[f'{split}'],
            config,
            ind_model,
            ind_test_results,
        )

        config["architecture"] = "SequentialConceptBottleneckModel"
        training.update_statistics(
            results[f'{split}'],
            config,
            seq_model,
            seq_test_results,
        )

        # train model *without* embeddings (concepts are just *Boolean* scalars)
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["extra_name"] = f"Bool"
        config["bool"] = True
        trainingl_model, bool_test_results = training.train_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            split=split,
            imbalance=imbalance,
            result_dir=result_dir,
            rerun=rerun,
            project_name=project_name,
            seed=split,
            activation_freq=activation_freq,
            single_frequency_epochs=single_frequency_epochs,
        )
        training.update_statistics(
            results[f'{split}'],
            config,
            bool_model,
            bool_test_results,
        )

        # train vanilla model with more capacity (i.e., no concept supervision)
        # but with ReLU activation
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["extra_name"] = f"NoConceptSupervisionReLU_ExtraCapacity"
        config["bool"] = False
        config["extra_dims"] = (config['emb_size'] - 1) * n_concepts
        config["bottleneck_nonlinear"] = "leakyrelu"
        config["concept_loss_weight"] = 0
        extra_vanilla_relu_model, extra_vanilla_relu_test_results = \
            training.train_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                split=split,
                result_dir=result_dir,
                rerun=rerun,
                project_name=project_name,
                seed=split,
                activation_freq=activation_freq,
                single_frequency_epochs=single_frequency_epochs,
                imbalance=imbalance,
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            extra_vanilla_relu_model,
            extra_vanilla_relu_test_results,
        )

        # save results
        joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Runs concept embedding experiment in CelebA dataset.'
        ),
    )
    parser.add_argument(
        '--project_name',
        default='',
        help=(
            "Project name used for Weights & Biases monitoring. If not "
            "provided, then we will assume we do not run a w&b project."
        ),
        metavar="name",

    )

    parser.add_argument(
        '--output_dir',
        '-o',
        default='results/celeba/',
        help=(
            "directory where we will dump our experiment's results. If not "
            "given, then we will use ./results/celeba/."
        ),
        metavar="path",

    )
    parser.add_argument(
        '--rerun',
        '-r',
        default=False,
        action="store_true",
        help=(
            "If set, then we will force a rerun of the entire experiment even if "
            "valid results are found in the provided output directory. Note that "
            "this may overwrite and previous results, so use with care."
        ),

    )
    parser.add_argument(
        '--activation_freq',
        default=0,
        help=(
            'how frequently, in terms of epochs, should we store the embedding activations for our '
            'validation set. By default we will not store any activations.'
        ),
        metavar='N',
        type=int,
    )
    parser.add_argument(
        '--single_frequency_epochs',
        default=0,
        help=(
            'how frequently, in terms of epochs, should we store the embedding activations for our '
            'validation set. By default we will not store any activations.'
        ),
        metavar='N',
        type=int,
    )
    parser.add_argument(
        '--num_workers',
        default=12,
        help=(
            'number of workers used for data feeders. Do not use more workers '
            'than cores in the machine.'
        ),
        metavar='N',
        type=int,
    )
    parser.add_argument(
        '--data_root',
        default=CELEBA_ROOT,
        help=(
            'directory containing the CelebA dataset.'
        ),
        metavar='path',
        type=str,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in our program.",
    )
    parser.add_argument(
        "--no_save_model",
        action="store_true",
        default=False,
        help="whether or not we will save the fully trained models.",
    )
    parser.add_argument(
        '-p',
        '--param',
        action='append',
        nargs=2,
        metavar=('param_name=value'),
        help=(
            'Allows the passing of a config param that will overwrite '
            'anything passed as part of the config file itself.'
        ),
        default=[],
    )
    args = parser.parse_args()
    main(
        data_root=args.data_root,
        rerun=args.rerun,
        result_dir=args.output_dir,
        project_name=args.project_name,
        activation_freq=args.activation_freq,
        num_workers=args.num_workers,
        single_frequency_epochs=args.single_frequency_epochs,
        global_params=args.param,
        save_model=(not args.no_save_model),
    )

