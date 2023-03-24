import argparse
import copy
import joblib
import numpy as np
import os

import logging
import torch
from pathlib import Path
from pytorch_lightning import seed_everything

import cem.data.CUB200.cub_loader as cub_data_module
import cem.data.celeba_loader as celeba_data_module

import cem.train.training as training
import cem.train.utils as utils
from intervention_utils import (
    intervene_in_cbm, random_int_policy
)
from run_experiments import CUB_CONFIG, CELEBA_CONFIG


################################################################################
## HELPER FUNCTIONS
################################################################################

def _filter_results(results, full_run_name):
    output = {}
    for key, val in results.items():
        if full_run_name not in key:
            continue
        output[key] = val
    return output


################################################################################
## MAIN FUNCTION
################################################################################


def main(
    data_module,
    rerun=False,
    result_dir='results/interventions/',
    project_name='',
    num_workers=8,
    global_params=None,
    gpu=torch.cuda.is_available(),
    og_config=None,
):
    seed_everything(42)
    # parameters for data, model, and training
    if og_config is None:
        # Then we use the CUB one as the default
        og_config = CUB_CONFIG
    og_config = copy.deepcopy(og_config)
    og_config['num_workers'] = num_workers

    gpu = 1 if gpu else 0
    utils.extend_with_global_params(og_config, global_params or [])

    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks) = \
        data_module.generate_data(
            config=og_config,
            seed=42,
            output_dataset_vars=True,
        )
    concept_map = None
    if hasattr(data_module, 'CONCEPT_MAP'):
        concept_map = data_module.CONCEPT_MAP
        intervened_groups = list(
            range(
                0,
                len(concept_map) + 1,
                og_config.get('intervention_freq', 1),
            )
        )
    else:
        concept_map = None
        intervened_groups = list(
            range(
                0,
                n_concepts + 1,
                og_config.get('intervention_freq', 1),
            )
        )

    sample = next(iter(train_dl))
    real_sample = []
    for x in sample:
        if isinstance(x, list):
            real_sample += x
        else:
            real_sample.append(x)
    sample = real_sample
    logging.info(f"Training sample shape is: {sample[0].shape}")
    logging.info(f"Training label shape is: {sample[1].shape}")
    logging.info(f"\tNumber of output classes: {n_tasks}")
    logging.info(f"Training concept shape is: {sample[2].shape}")
    logging.info(f"\tNumber of training concepts: {n_concepts}")

    os.makedirs(result_dir, exist_ok=True)
    old_results = {}
    if os.path.exists(os.path.join(result_dir, f'results.joblib')):
        old_results = joblib.load(
            os.path.join(result_dir, f'results.joblib')
        )

    results = {}
    for split in range(og_config["cv"]):
        results[f'{split}'] = {}
        logging.info(f'Experiment {split+1}/{og_config["cv"]}')

        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptEmbeddingModel"
        config["extra_name"] = ""
        config["shared_prob_gen"] = True
        config["sigmoidal_prob"] = True
        config["sigmoidal_embedding"] = False
        config['training_intervention_prob'] = 0.25
        config['concat_prob'] = False
        config['emb_size'] = config['emb_size']
        config["embeding_activation"] = "leakyrelu"
        mixed_emb_shared_prob_model,  mixed_emb_shared_prob_test_results = \
            training.train_model(
                gpu=gpu if gpu else 0,
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
                imbalance=imbalance,
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            mixed_emb_shared_prob_model,
            mixed_emb_shared_prob_test_results,
        )
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}"
        )
        results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
            intervene_in_cbm(
                concept_selection_policy=random_int_policy,
                concept_group_map=concept_map,
                intervened_groups=intervened_groups,
                gpu=gpu if gpu else None,
                config=config,
                test_dl=test_dl,
                train_dl=train_dl,
                n_tasks=n_tasks,
                n_concepts=n_concepts,
                result_dir=result_dir,
                imbalance=imbalance,
                split=split,
                adversarial_intervention=False,
                rerun=rerun,
                batch_size=512,
                old_results=old_results.get(str(split), {}).get(
                    f'test_acc_y_ints_{full_run_name}'
                ),
            )

        print(f"\tResults for {full_run_name} in split {split}:")
        for key, val in results[f'{split}'].items():
            print(f"\t\t{key} -> {val}")
        joblib.dump(
            _filter_results(results[f'{split}'], full_run_name),
            os.path.join(
                result_dir,
                f'{full_run_name}_split_{split}_results.joblib'
            ),
        )
        joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

        # train vanilla CBM models with both logits and sigmoidal
        # bottleneck activations
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["bool"] = False
        config["extra_dims"] = 0
        config["extra_name"] = f"Logit"
        config["bottleneck_nonlinear"] = "leakyrelu"
        config["sigmoidal_extra_capacity"] = False
        config["sigmoidal_prob"] = False
        cbm_logit_model, cbm_logit_test_results = \
            training.train_model(
                gpu=gpu if gpu else None,
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
                imbalance=imbalance,
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            cbm_logit_model,
            cbm_logit_test_results,
        )
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}"
        )
        results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
            intervene_in_cbm(
                concept_selection_policy=random_int_policy,
                concept_group_map=concept_map,
                intervened_groups=intervened_groups,
                gpu=gpu if gpu else None,
                config=config,
                test_dl=test_dl,
                train_dl=train_dl,
                n_tasks=n_tasks,
                n_concepts=n_concepts,
                result_dir=result_dir,
                split=split,
                imbalance=imbalance,
                adversarial_intervention=False,
                rerun=rerun,
                batch_size=512,
                old_results=old_results.get(str(split), {}).get(
                    f'test_acc_y_ints_{full_run_name}'
                ),
            )

        print(f"\tResults for {full_run_name} in split {split}:")
        for key, val in results[f'{split}'].items():
            print(f"\t\t{key} -> {val}")
        joblib.dump(
            _filter_results(results[f'{split}'], full_run_name),
            os.path.join(
                result_dir,
                f'{full_run_name}_split_{split}_results.joblib'
            ),
        )
        joblib.dump(results, os.path.join(result_dir, f'results.joblib'))


        # CBM with sigmoidal bottleneck
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["extra_name"] = f"Sigmoid"
        config["bool"] = False
        config["extra_dims"] = 0
        config["sigmoidal_extra_capacity"] = False
        config["sigmoidal_prob"] = True
        cbm_sigmoid_model, cbm_sigmoid_test_results = \
            training.train_model(
                gpu=gpu if gpu else None,
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
                imbalance=imbalance,
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            cbm_sigmoid_model,
            cbm_sigmoid_test_results,
        )
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}"
        )
        results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
            intervene_in_cbm(
                concept_selection_policy=random_int_policy,
                concept_group_map=concept_map,
                intervened_groups=intervened_groups,
                gpu=gpu if gpu else None,
                config=config,
                test_dl=test_dl,
                train_dl=train_dl,
                n_tasks=n_tasks,
                n_concepts=n_concepts,
                result_dir=result_dir,
                split=split,
                imbalance=imbalance,
                adversarial_intervention=False,
                rerun=rerun,
                batch_size=512,
                old_results=old_results.get(str(split), {}).get(
                    f'test_acc_y_ints_{full_run_name}'
                ),
            )

        # save results
        print(f"\tResults for {full_run_name} in split {split}:")
        for key, val in results[f'{split}'].items():
            print(f"\t\t{key} -> {val}")
        joblib.dump(
            _filter_results(results[f'{split}'], full_run_name),
            os.path.join(
                result_dir,
                f'{full_run_name}_split_{split}_results.joblib'
            ),
        )
        joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Runs CEM intervention experiments in a given dataset.'
        ),
    )
    parser.add_argument(
        '--project_name',
        default='',
        help=(
            "Project name used for Weights & Biases monitoring. If not "
            "provided, then we will not log in W&B."
        ),
        metavar="name",

    )
    parser.add_argument(
        'dataset',
        choices=['cub', 'celeba'],
        help=(
            "Dataset to run experiments for. Must be a supported dataset with "
            "a loader."
        ),
        metavar="ds_name",

    )
    parser.add_argument(
        '--output_dir',
        '-o',
        default='results/{ds_name}_interventions/',
        help=(
            "directory where we will dump our experiment's results. If not "
            "given, then we will use ./results/{ds_name}_interventions/."
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
            "Note that this may overwrite and previous results, so use "
            "with care."
        ),

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
        "--force_cpu",
        action="store_true",
        default=False,
        help="forces CPU training.",
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
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    if args.dataset == "cub":
        data_module = cub_data_module
        og_config = CUB_CONFIG
        args.output_dir = args.output_dir.format(ds_name="cub")
        args.project_name = args.project_name.format(ds_name="cub")
    elif args.dataset == "celeba":
        data_module = celeba_data_module
        og_config = CELEBA_CONFIG
        args.output_dir = args.output_dir.format(ds_name="celeba")
        args.project_name = args.project_name.format(ds_name="celeba")
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}!")
    main(
        data_module=data_module,
        rerun=args.rerun,
        result_dir=args.output_dir,
        project_name=args.project_name,
        num_workers=args.num_workers,
        global_params=args.param,
        gpu=(not args.force_cpu) and (torch.cuda.is_available()),
        og_config=og_config,
    )
