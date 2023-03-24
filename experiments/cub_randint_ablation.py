import argparse
import copy
import joblib
import numpy as np
import os
import torch

from cem.data.CUB200.cub_loader import load_data, find_class_imbalance
from pathlib import Path
from pytorch_lightning import seed_everything

import cem.experiments.cub_experiments as cub
import cem.train.training as training
import cem.train.utils as utils

def main(
    rerun=False,
    result_dir='results/cub_randint_ablation/',
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
        max_epochs=300,
        patience=15,
        batch_size=128,
        num_workers=num_workers,
        emb_size=16,
        extra_dims=0,
        concept_loss_weight=5,
        learning_rate=0.01,
        weight_decay=4e-05,
        scheduler_step=20,
        weight_loss=True,
        c_extractor_arch="resnet34",
        optimizer="sgd",
        bool=False,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        early_stopping_delta=0.0,
        sampling_percent=1,

        momentum=0.9,
        shared_prob_gen=False,
        sigmoidal_prob=False,
        sigmoidal_embedding=False,
        training_intervention_prob=0.0,
        embeding_activation=None,
        concat_prob=False,
    )

    utils.extend_with_global_params(og_config, global_params or [])
    train_data_path = os.path.join(cub.BASE_DIR, 'train.pkl')
    if og_config['weight_loss']:
        imbalance = find_class_imbalance(train_data_path, True)
    else:
        imbalance = None

    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    test_data_path = train_data_path.replace('train.pkl', 'test.pkl')
    sampling_percent = og_config.get("sampling_percent", 1)
    n_concepts, n_tasks = 112, 200

    if sampling_percent != 1:
        # Do the subsampling
        new_n_concepts = int(np.ceil(n_concepts * sampling_percent))
        selected_concepts_file = os.path.join(
            result_dir,
            f"selected_concepts_sampling_{sampling_percent}.npy",
        )
        if (not rerun) and os.path.exists(selected_concepts_file):
            selected_concepts = np.load(selected_concepts_file)
        else:
            selected_concepts = sorted(
                np.random.permutation(n_concepts)[:new_n_concepts]
            )
            np.save(selected_concepts_file, selected_concepts)
        print("\t\tSelected concepts:", selected_concepts)
        def subsample_transform(sample):
            if isinstance(sample, list):
                sample = np.array(sample)
            return sample[selected_concepts]

        if og_config['weight_loss']:
            imbalance = np.array(imbalance)[selected_concepts]

        train_dl = load_data(
            pkl_paths=[train_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=False,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=cub.CUB_DIR,
            num_workers=og_config['num_workers'],
            concept_transform=subsample_transform,
        )
        val_dl = load_data(
            pkl_paths=[val_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=False,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=cub.CUB_DIR,
            num_workers=og_config['num_workers'],
            concept_transform=subsample_transform,
        )
        test_dl = load_data(
            pkl_paths=[test_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=False,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=cub.CUB_DIR,
            num_workers=og_config['num_workers'],
            concept_transform=subsample_transform,
        )
        # And set the right number of concepts to be used
        n_concepts = new_n_concepts
    else:
        train_dl = load_data(
            pkl_paths=[train_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=False,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=cub.CUB_DIR,
            num_workers=og_config['num_workers'],
        )
        val_dl = load_data(
            pkl_paths=[val_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=False,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=cub.CUB_DIR,
            num_workers=og_config['num_workers'],
        )
        test_dl = load_data(
            pkl_paths=[test_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=False,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=cub.CUB_DIR,
            num_workers=og_config['num_workers'],
        )

    if result_dir and activation_freq:
        # Then let's save the testing data for furter analysis later on
        out_acts_save_dir = os.path.join(result_dir, "test_embedding_acts")
        Path(out_acts_save_dir).mkdir(parents=True, exist_ok=True)
        for (ds, name) in [
            (test_dl, "test"),
            (val_dl, "val"),
        ]:
            x_total = []
            y_total = []
            c_total = []
            for x, y, c in ds:
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

    sample = next(iter(train_dl))
    n_concepts, n_tasks = sample[2].shape[-1], 200

    print("Training sample shape is:", sample[0].shape)
    print("Training label shape is:", sample[1].shape)
    print("Training concept shape is:", sample[2].shape)
    os.makedirs(result_dir, exist_ok=True)
    results = {}

    for prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0]:
        results[prob] = {}
        for split in range(og_config["cv"]):
            print(f'Experiment {split+1}/{og_config["cv"]} with prob', prob)
            results[prob][f'{split}'] = {}

            # Trial period for mixture embedding model
            config = copy.deepcopy(og_config)
            config["architecture"] = "MixtureEmbModel"
            config["extra_name"] = (
                f"SharedProb_AdaptiveDropout_NoProbConcat_prob_{prob}"
            )
            config["shared_prob_gen"] = True
            config["sigmoidal_prob"] = False
            config["sigmoidal_embedding"] = False
            config['training_intervention_prob'] = prob
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
                    imbalance=imbalance,
                )
            training.update_statistics(
                results[prob][f'{split}'],
                config,
                mixed_emb_shared_prob_model,
                mixed_emb_shared_prob_test_results,
            )

            # save results
            joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Runs ablation study for RandInt in CUB dataset.'
        ),
    )
    parser.add_argument(
        '--project_name',
        default='',
        help=(
            "Project name used for Weights & Biases monitoring. If not "
            "provided, then we will assume no W&B logging is used."
        ),
        metavar="name",

    )

    parser.add_argument(
        '--output_dir',
        '-o',
        default='results/cub_randint_ablation/',
        help=(
            "directory where we will dump our experiment's results. If not "
            "given, then we will use ./results/cub_randint_ablation/."
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
            'how frequently, in terms of epochs, should we store the '
            'embedding activations for our validation set. By default we '
            'will not store any activations.'
        ),
        metavar='N',
        type=int,
    )
    parser.add_argument(
        '--single_frequency_epochs',
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
    if args.project_name:
        # Lazy import to avoid importing unless necessary
        import wandb
    main(
        rerun=args.rerun,
        result_dir=args.output_dir,
        project_name=args.project_name,
        activation_freq=args.activation_freq,
        num_workers=args.num_workers,
        single_frequency_epochs=args.single_frequency_epochs,
        global_params=args.param,
    )
