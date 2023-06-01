import argparse
import copy
import joblib
import numpy as np
import os

from pathlib import Path
from pytorch_lightning import seed_everything

import cem.data.celeba_loader as celeba_data_module
import cem.train.training as training
import cem.train.utils as utils


def main(
    rerun=False,
    result_dir='results/cub_emb_size_ablation/',
    project_name='',
    activation_freq=0,
    num_workers=8,
    single_frequency_epochs=0,
    global_params=None,
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
        learning_rate=0.005,
        weight_decay=4e-05,
        weight_loss=False,
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
    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, _) = celeba_data_module.generate_data(
        config=og_config,
        seed=42,
        output_dataset_vars=True,
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
    os.makedirs(result_dir, exist_ok=True)
    results = {}
    for split in range(og_config["cv"]):
        for emb_size in [1, 2, 4, 6, 8, 16, 32, 64]:
            if emb_size not in results:
                results[emb_size] = {}
            if f'{split}' not in results[emb_size]:
                results[emb_size][f'{split}'] = {}
            print(
                f'Experiment {split+1}/{og_config["cv"]} with emb_size',
                emb_size,
            )

            # Trial period for mixture embedding model
            config = copy.deepcopy(og_config)
            config["architecture"] = "MixtureEmbModel"
            config["extra_name"] = (
                f"SharedProb_AdaptiveDropout_NoProbConcat_"
                f"emb_size_{emb_size}"
            )
            config["shared_prob_gen"] = True
            config["sigmoidal_prob"] = False
            config["sigmoidal_embedding"] = False
            config['training_intervention_prob'] = 0.25
            config['concat_prob'] = False
            config['emb_size'] = emb_size
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
                results[emb_size][f'{split}'],
                config,
                mixed_emb_shared_prob_model,
                mixed_emb_shared_prob_test_results,
            )

            # Train fuzzy CBM with extra capacity
            config = copy.deepcopy(og_config)
            config["architecture"] = "ConceptBottleneckModel"
            config["bool"] = False
            config["extra_dims"] = (emb_size - 1) * n_concepts
            config["extra_name"] = (
                f"FuzzyExtraCapacity_Logit_emb_size_{emb_size}"
            )
            config["bottleneck_nonlinear"] = "leakyrelu"
            config["sigmoidal_extra_capacity"] = False
            config["sigmoidal_prob"] = False
            config['emb_size'] = emb_size
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
                results[emb_size][f'{split}'],
                config,
                extra_fuzzy_logit_model,
                extra_fuzzy_logit_test_results,
            )

            # train vanilla model with more capacity (i.e., no concept
            # supervision) but with ReLU activation
            config = copy.deepcopy(og_config)
            config["architecture"] = "ConceptBottleneckModel"
            config["extra_name"] = (
                f"NoConceptSupervisionReLU_ExtraCapacity_emb_size_{emb_size}"
            )
            config["bool"] = False
            config["extra_dims"] = (emb_size - 1) * n_concepts
            config["bottleneck_nonlinear"] = "leakyrelu"
            config["concept_loss_weight"] = 0
            config['emb_size'] = emb_size
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
                results[emb_size][f'{split}'],
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
            'Runs embedding ablation study in CelebA dataset.'
        ),
    )
    parser.add_argument(
        '--project_name',
        default='',
        help=(
            "Project name used for Weights & Biases monitoring. If not "
            "provided, we assume no W&B logging is done."
        ),
        metavar="name",

    )

    parser.add_argument(
        '--output_dir',
        '-o',
        default='results/celeba_emb_size_ablation/',
        help=(
            "directory where we will dump our experiment's results. If not "
            "given, then we will use results/celeba_emb_size_ablation/."
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
            'how frequently, in terms of epochs, should we store the '
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
            'how frequently, in terms of epochs, should we store the '
            'embedding activations for our validation set. By default we '
            'will not store any activations.'
        ),
        metavar='N',
        type=int,
    )
    parser.add_argument(
        '--num_workers',
        default=8,
        help=(
            'number of workers used for data feeders. Do not use more workers '
            'than cores in the machine.'
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
        rerun=args.rerun,
        result_dir=args.output_dir,
        project_name=args.project_name,
        activation_freq=args.activation_freq,
        num_workers=args.num_workers,
        single_frequency_epochs=args.single_frequency_epochs,
        global_params=args.param,
    )
