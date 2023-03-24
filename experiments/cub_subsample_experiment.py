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
    result_dir='results/cub_subsample/',
    project_name='',
    save_models=True,
    activation_freq=0,
    single_frequency_epochs=0,
    global_params=None,
    num_workers=8,
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
        corr_thresh=0.5,
        dense_corr_thresh=0.25,
        sampling_percent=1,
        sampling_percents=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1],

        momentum=0.9,
        shared_prob_gen=False,
        sigmoidal_prob=False,
        sigmoidal_embedding=False,
        training_intervention_prob=0.0,
        embeding_activation=None,
        concat_prob=False,
    )

    train_data_path = os.path.join(cub.BASE_DIR, 'train.pkl')
    if og_config['weight_loss']:
        og_imbalance = find_class_imbalance(train_data_path, True)
    else:
        og_imbalance = None
    utils.extend_with_global_params(og_config, global_params or [])

    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    test_data_path = train_data_path.replace('train.pkl', 'test.pkl')
    n_concepts, n_tasks = 112, 200

    os.makedirs(result_dir, exist_ok=True)
    joblib.dump(
        og_config,
        os.path.join(result_dir, f'experiment_config.joblib'),
    )

    if result_dir and activation_freq:
        # Then let's save the testing data for further analysis later on
        out_acts_save_dir = os.path.join(result_dir, "test_embedding_acts")
        Path(out_acts_save_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for sampling_percent in og_config['sampling_percents']:
        print(
            f"Training model by subsampling {sampling_percent *100}% of "
            f"concepts"
        )
        results[sampling_percent] = {}
        new_n_concepts = int(np.ceil(n_concepts * sampling_percent))
        for split in range(og_config["cv"]):
            print(
                f'\tExperiment {split+1}/{og_config["cv"]} with sampling '
                f'rate {sampling_percent *100}% and {new_n_concepts} concepts'
            )
            results[sampling_percent][f'{split}'] = {}

            # Do the subsampling
            selected_concepts_file = os.path.join(
                result_dir,
                (
                    f"selected_concepts_"
                    f"sampling_{sampling_percent}_fold_{split}.npy"
                ),
            )
            if (not rerun) and os.path.exists(selected_concepts_file):
                selected_concepts = np.load(selected_concepts_file)
            else:
                if sampling_percent != 1:
                    selected_concepts = np.random.permutation(
                        n_concepts
                    )[:new_n_concepts]
                else:
                    # Then simply select them all in their original order
                    selected_concepts = np.range(new_n_concepts)
                np.save(selected_concepts_file, selected_concepts)
            print("\t\tSelected concepts:", selected_concepts)
            def subsample_transform(sample):
                if isinstance(sample, list):
                    sample = np.array(sample)
                return sample[selected_concepts]

            if og_config['weight_loss']:
                imbalance = np.array(og_imbalance)[selected_concepts]
            else:
                imbalance = np.array(og_imbalance)[selected_concepts]

            train_dl = load_data(
                pkl_paths=[train_data_path],
                use_attr=True,
                no_img=False,
                batch_size=og_config['batch_size'],
                uncertain_label=False,
                n_class_attr=2,
                image_dir='images',
                resampling=False,
                root_dir=CUB_DIR,
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
                root_dir=CUB_DIR,
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
                root_dir=CUB_DIR,
                num_workers=og_config['num_workers'],
                concept_transform=subsample_transform,
            )

            sample = next(iter(train_dl))
            print("Training sample shape is:", sample[0].shape)
            print("Training label shape is:", sample[1].shape)
            print("Training concept shape is:", sample[2].shape)


            # train vanilla model with more capacity (i.e., no concept
            # supervision) but with ReLU activation
            config = copy.deepcopy(og_config)
            config["architecture"] = "ConceptBottleneckModel"
            config["extra_name"] = (
                f"NoConceptSupervisionReLU_ExtraCapacity_"
                f"subsample_{sampling_percent}"
            )
            config["sampling_percent"] = sampling_percent
            config["bool"] = False
            config["extra_dims"] = config['emb_size'] * new_n_concepts
            config["bottleneck_nonlinear"] = "relu"
            config["concept_loss_weight"] = 0
            extra_vanilla_relu_model, extra_vanilla_relu_test_results = \
                training.train_model(
                    n_concepts=new_n_concepts,
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
                results[sampling_percent][f'{split}'],
                config,
                extra_vanilla_relu_model,
                extra_vanilla_relu_test_results,
            )

            # fuzzy model
            config = copy.deepcopy(og_config)
            config["architecture"] = "ConceptBottleneckModel"
            config["extra_name"] = f"Fuzzy_subsample_{sampling_percent}"
            config["sampling_percent"] = sampling_percent
            config["bool"] = False
            config["extra_dims"] = 0
            config["sigmoidal_extra_capacity"] = False
            config["sigmoidal_prob"] = True
            extra_fuzzy_logit_model, extra_fuzzy_logit_test_results = \
                training.train_model(
                    n_concepts=new_n_concepts,
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
                results[sampling_percent][f'{split}'],
                config,
                extra_fuzzy_logit_model,
                extra_fuzzy_logit_test_results,
            )

            # train model *without* embeddings but with extra capacity.
            config = copy.deepcopy(og_config)
            config["architecture"] = "ConceptBottleneckModel"
            config["bool"] = False
            config["extra_dims"] = config['emb_size'] * new_n_concepts
            config["sampling_percent"] = sampling_percent
            config["extra_name"] = (
                f"FuzzyExtraCapacity_Logit_subsample_{sampling_percent}"
            )
            config["sigmoidal_extra_capacity"] = False
            config["sigmoidal_prob"] = False
            extra_fuzzy_logit_model, extra_fuzzy_logit_test_results = \
                training.train_model(
                    n_concepts=new_n_concepts,
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
                results[sampling_percent][f'{split}'],
                config,
                extra_fuzzy_logit_model,
                extra_fuzzy_logit_test_results,
            )

            # train model *without* embeddings (concepts are just *Boolean*
            # scalars)
            config = copy.deepcopy(og_config)
            config["architecture"] = "ConceptBottleneckModel"
            config["extra_name"] = f"Bool_subsample_{sampling_percent}"
            config["bool"] = True
            config["sampling_percent"] = sampling_percent
            config["selected_concepts"] = selected_concepts
            bool_model, bool_test_results = training.train_model(
                n_concepts=new_n_concepts,
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
                save_model=save_models,
                activation_freq=activation_freq,
                single_frequency_epochs=single_frequency_epochs,
            )
            training.update_statistics(
                results[sampling_percent][f'{split}'],
                config,
                bool_model,
                bool_test_results,
                save_model=save_models,
            )

            config = copy.deepcopy(og_config)
            config["architecture"] = "MixtureEmbModel"
            config["extra_name"] = (
                f"SharedProb_AdaptiveDropout_NoProbConcat_"
                f"subsample_{sampling_percent}"
            )
            config["sampling_percent"] = sampling_percent
            config["shared_prob_gen"] = True
            config["sigmoidal_prob"] = True
            config["sigmoidal_embedding"] = False
            config['training_intervention_prob'] = 0.25
            config['concat_prob'] = False
            config['emb_size'] = config['emb_size']
            config["embeding_activation"] = "leakyrelu"
            mixed_emb_shared_prob_model,  mixed_emb_shared_prob_test_results = \
                training.train_model(
                    n_concepts=new_n_concepts,
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
                results[sampling_percent][f'{split}'],
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
            'Runs concept subsampling experiment in CUB dataset.'
        ),
    )
    parser.add_argument(
        '--project_name',
        default='',
        help=(
            "Project name used for Weights & Biases monitoring. If not "
            "provided, then we will assume no W&B logging is done."
        ),
        metavar="name",

    )

    parser.add_argument(
        '--output_dir',
        '-o',
        default='results/cub_subsample/',
        help=(
            "directory where we will dump our experiment's results. If not "
            "given, then we will use ./results/cub_subsample/."
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
            'embedding activations for our validation set. By default we will '
            'not store any activations.'
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
