import argparse
import copy
import joblib
import numpy as np
import os
import pytorch_lightning as pl
import torch

from pathlib import Path
from pytorch_lightning import seed_everything

import cem.train.training as training
import cem.train.utils as utils

################################################################################
## DATASET GENERATORS
################################################################################


def generate_xor_data(size):
    # sample from normal distribution
    x = np.random.uniform(0, 1, (size, 2))
    c = np.stack([
        x[:, 0] > 0.5,
        x[:, 1] > 0.5,
    ]).T
    y = np.logical_xor(c[:, 0], c[:, 1])

    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)
    return x, c, y


def generate_trig_data(size):
    h = np.random.normal(0, 2, (size, 3))
    x, y, z = h[:, 0], h[:, 1], h[:, 2]

    # raw features
    input_features = np.stack([
        np.sin(x) + x,
        np.cos(x) + x,
        np.sin(y) + y,
        np.cos(y) + y,
        np.sin(z) + z,
        np.cos(z) + z,
        x ** 2 + y ** 2 + z ** 2,
    ]).T

    # concetps
    concetps = np.stack([
        x > 0,
        y > 0,
        z > 0,
    ]).T

    # task
    downstream_task = (x + y + z) > 1

    input_features = torch.FloatTensor(input_features)
    concetps = torch.FloatTensor(concetps)
    downstream_task = torch.FloatTensor(downstream_task)
    return input_features, concetps, downstream_task


def generate_dot_data(size):
    # sample from normal distribution
    emb_size = 2
    v1 = np.random.randn(size, emb_size) * 2
    v2 = np.ones(emb_size)
    v3 = np.random.randn(size, emb_size) * 2
    v4 = -np.ones(emb_size)
    x = np.hstack([v1+v3, v1-v3])
    c = np.stack([
        np.dot(v1, v2).ravel() > 0,
        np.dot(v3, v4).ravel() > 0,
    ]).T
    y = ((v1*v3).sum(axis=-1) > 0).astype(np.int64)

    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.Tensor(y)
    return x, c, y


################################################################################
## MAIN PROGRAM
################################################################################

def main(
    dataset,
    result_dir,
    rerun=False,
    project_name='',
    activation_freq=0,
    single_frequency_epochs=0,
    global_params=None,
):
    seed_everything(42)
    # parameters for data, model, and training
    og_config = dict(
        cv=5,
        dataset_size=3000,
        max_epochs=500,
        patience=15,
        batch_size=256,
        num_workers=8,
        emb_size=128,
        extra_dims=0,
        concept_loss_weight=1,
        learning_rate=0.01,
        weight_decay=0,
        scheduler_step=20,
        weight_loss=False,
        optimizer="adam",
        bool=False,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        early_stopping_delta=0.0,
        masked=False,
        check_val_every_n_epoch=30,
        linear_c2y=True,
        embeding_activation="leakyrelu",

        momentum=0.9,
        shared_prob_gen=False,
        sigmoidal_prob=False,
        sigmoidal_embedding=False,
        training_intervention_prob=0.0,
        concat_prob=False,
    )

    if dataset == "xor":
        generate_data = generate_xor_data
    elif dataset in ["trig", "trigonometry"]:
        generate_data = generate_trig_data
    elif dataset in ["vector", "dot"]:
        generate_data = generate_dot_data
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    utils.extend_with_global_params(og_config, global_params or [])
    dataset_size = og_config['dataset_size']
    batch_size = og_config["batch_size"]
    x, c, y = generate_data(int(dataset_size * 0.7))
    train_data = torch.utils.data.TensorDataset(x, y, c)
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    dataset = dataset.lower()

    x_test, c_test, y_test = generate_data(int(dataset_size * 0.2))
    test_data = torch.utils.data.TensorDataset(x_test, y_test, c_test)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    x_val, c_val, y_val = generate_data(int(dataset_size * 0.1))
    val_data = torch.utils.data.TensorDataset(x_val, y_val, c_val)
    val_dl = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    if result_dir and activation_freq:
        # Then let's save the testing data for further analysis later on
        out_acts_save_dir = os.path.join(result_dir, "test_embedding_acts")
        Path(out_acts_save_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(out_acts_save_dir, "x_test.npy"), x_test)
        np.save(os.path.join(out_acts_save_dir, "y_test.npy"), y_test)
        np.save(os.path.join(out_acts_save_dir, "c_test.npy"), c_test)
        np.save(os.path.join(out_acts_save_dir, "x_val.npy"), x_val)
        np.save(os.path.join(out_acts_save_dir, "y_val.npy"), y_val)
        np.save(os.path.join(out_acts_save_dir, "c_val.npy"), c_val)

    sample = next(iter(train_dl))
    n_features, n_concepts, n_tasks = (
        sample[0].shape[-1],
        sample[2].shape[-1],
        1,
    )

    # And make the concept extractor architecture
    def c_extractor_arch(output_dim):
        return torch.nn.Sequential(*[
            torch.nn.Linear(n_features, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, output_dim),
        ])
    og_config['c_extractor_arch'] = c_extractor_arch

    print("Training sample shape is:", sample[0].shape)
    print("Training label shape is:", sample[2].shape)
    print("Training concept shape is:", sample[1].shape)

    os.makedirs(result_dir, exist_ok=True)

    results = {}
    for split in range(og_config["cv"]):
        print(f'Experiment {split+1}/{og_config["cv"]}')
        results[f'{split}'] = {}

        # train model *without* embeddings (concepts are just *fuzzy* scalars)
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["bool"] = False
        config["extra_name"] = "Fuzzy"
        config["concept_loss_weight"] = config.get(
            "cbm_concept_loss_weight",
            config["concept_loss_weight"],
        )
        fuzzy_model, fuzzy_test_results = training.train_model(
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
        )
        training.update_statistics(
            results[f'{split}'],
            config,
            fuzzy_model,
            fuzzy_test_results,
        )

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
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            mixed_emb_shared_prob_model,
            mixed_emb_shared_prob_test_results,
        )

        # train model *without* embeddings but with extra capacity
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["bool"] = False
        config["extra_dims"] = (config['emb_size'] - 1) * n_concepts
        config["extra_name"] = "FuzzyExtraCapacity_LogitOnlyProb"
        config["bottleneck_nonlinear"] = "leakyrelu"
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
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            extra_fuzzy_logit_model,
            extra_fuzzy_logit_test_results,
        )

        # train vanilla model with more capacity (i.e., no concept supervision)
        # but with ReLU activation
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["bool"] = False
        config["extra_dims"] = (config['emb_size'] - 1) * n_concepts
        config["bottleneck_nonlinear"] = "leakyrelu"
        config["extra_name"] = "NoConceptSupervisionReLU_ExtraCapacity"
        config["concept_loss_weight"] = 0
        config["sigmoidal_extra_capacity"] = False
        config["sigmoidal_prob"] = False
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
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            extra_vanilla_relu_model,
            extra_vanilla_relu_test_results,
        )

        # train model *without* embeddings (concepts are just *Boolean* scalars)
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["extra_name"] = "Bool"
        config["bool"] = True
        if "cbm_bool_concept_loss_weight" in config:
            config["concept_loss_weight"] = config[
                "cbm_bool_concept_loss_weight"
            ]
        else:
            config["concept_loss_weight"] = config.get(
                "cbm_concept_loss_weight",
                config["concept_loss_weight"],
            )
        bool_model, bool_test_results = training.train_model(
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
        )
        training.update_statistics(
            results[f'{split}'],
            config,
            bool_model,
            bool_test_results,
        )

        # save results
        joblib.dump(results, os.path.join(result_dir, f'results.joblib'))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Runs concept embedding experiment in our synthetic datasets.'
        ),
    )
    parser.add_argument(
        'dataset',
        help=(
            "Dataset to be used. One of xor, trig, dot."
        ),
        metavar="name",

    )
    parser.add_argument(
        '--project_name',
        default='',
        help=(
            "Project name used for Weights & Biases monitoring. If not "
            "provided, then we will assume we will not be using wandb "
            "for logging'."
        ),
        metavar="name",

    )

    parser.add_argument(
        '--output_dir',
        '-o',
        default='results/synthetic/',
        help=(
            "directory where we will dump our experiment's results. If not "
            "given, then we will use results/synthetic/."
        ),
        metavar="path",

    )
    parser.add_argument(
        '--rerun',
        '-r',
        default=False,
        action="store_true",
        help=(
            "If set, then we will force a rerun of the entire experiment even "
            "if valid results are found in the provided output directory. "
            "Note that this may overwrite and previous results, so use with "
            "care."
        ),

    )
    parser.add_argument(
        '--activation_freq',
        default=0,
        help=(
            'How frequently, in terms of epochs, should we store the '
            'embedding activations for our validation set. By default we will '
            'not store any activations.'
        ),
        metavar='N',
        type=int,
    )
    parser.add_argument(
        '--single_frequency_epochs',
        default=0,
        help=(
            'how many epochs we will monitor using an equivalent frequency of 1.'
        ),
        metavar='N',
        type=int,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in our program.",
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
        dataset=args.dataset,
        rerun=args.rerun,
        result_dir=args.output_dir,
        project_name=args.project_name,
        activation_freq=args.activation_freq,
        single_frequency_epochs=args.single_frequency_epochs,
        global_params=args.param
    )
