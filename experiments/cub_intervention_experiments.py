import argparse
import copy
import joblib
import numpy as np
import os

import logging
import torch
from pathlib import Path
from pytorch_lightning import seed_everything

from cem.data.CUB200.cub_loader import load_data, find_class_imbalance
import cem.train.training as training
import cem.train.utils as utils
from intervention_utils import (
    intervene_in_cbm, CUB_CONCEPT_GROUP_MAP, random_int_policy
)

################################################################################
## GLOBAL CUB VARIABLES
################################################################################

# IMPORANT NOTE: THIS DATASET NEEDS TO BE DOWNLOADED FIRST BEFORE BEING ABLE
#                TO RUN ANY CUB EXPERIMENTS!!
#                Instructions on how to download it can be found
#                in the original CBM paper's repository
#                found here: https://github.com/yewsiang/ConceptBottleneck
CUB_DIR = 'cem/data/CUB200/'
BASE_DIR = os.path.join(CUB_DIR, 'class_attr_data_10')


def _filter_results(results, full_run_name):
    output = {}
    for key, val in results.items():
        if full_run_name not in key:
            continue
        output[key] = val
    return output

def generate_cub_data(config, base_dir=BASE_DIR, root_dir=CUB_DIR, unc_map=None, seed=42):
    seed_everything(seed)
    train_data_path = os.path.join(base_dir, 'train.pkl')
    if config.get('weight_loss', False):
        imbalance = find_class_imbalance(train_data_path, True)
    else:
        imbalance = None

    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    test_data_path = train_data_path.replace('train.pkl', 'test.pkl')
    sampling_percent = config.get("sampling_percent", 1)

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
        def concept_transform(sample):
            if isinstance(sample, list):
                sample = np.array(sample)
            return sample[selected_concepts]

        # And correct the weight imbalance
        if config.get('weight_loss', False):
            imbalance = np.array(imbalance)[selected_concepts]
    else:
        concept_transform = None


    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=config['batch_size'],
        uncertain_label=unc_map is not None,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
        concept_transform=concept_transform,
        unc_map=unc_map,
    )
    val_dl = load_data(
        pkl_paths=[val_data_path],
        use_attr=True,
        no_img=False,
        batch_size=config['batch_size'],
        uncertain_label=unc_map is not None,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
        concept_transform=concept_transform,
        unc_map=unc_map,
    )

    test_dl = load_data(
        pkl_paths=[test_data_path],
        use_attr=True,
        no_img=False,
        batch_size=config['batch_size'],
        uncertain_label=unc_map is not None,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
        concept_transform=concept_transform,
        unc_map=unc_map,
    )
    return train_dl, val_dl, test_dl, imbalance

################################################################################
## MAIN FUNCTION
################################################################################


def main(
    rerun=False,
    result_dir='results/cub/',
    project_name='',
    num_workers=8,
    global_params=None,
    test_uncertain=False,
    include_uncertain_train=False,
    gpu=torch.cuda.is_available(),
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
        normalize_loss=False,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=True,
        pretrain_model=True,
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
    gpu = 1 if gpu else 0
    utils.extend_with_global_params(og_config, global_params or [])

    test_data_path = os.path.join(BASE_DIR, 'test.pkl')
    train_dl, val_dl, test_dl, imbalance = generate_cub_data(
        config=og_config,
        unc_map=None,
        seed=42,
    )

    sample = next(iter(train_dl))
    n_concepts, n_tasks = sample[2].shape[-1], 200
    print("Training sample shape is:", sample[0].shape)
    print("Training label shape is:", sample[1].shape)
    print("Training concept shape is:", sample[2].shape)

    os.makedirs(result_dir, exist_ok=True)
    old_results = {}
    if os.path.exists(os.path.join(result_dir, f'results.joblib')):
        old_results = joblib.load(
            os.path.join(result_dir, f'results.joblib')
        )

    results = {}
	if include_uncertain_train:
        train_uncertain_set = [None, 0.5, 0.7, 0.9]
    else:
        train_uncertain_set = [None]
    for split in range(og_config["cv"]):
        results[f'{split}'] = {}
		for train_uncertain in train_uncertain_set:
            if train_uncertain is not None:
                train_dl_uncertain, val_dl_uncertain, test_dl_uncertain, _ = generate_cub_data(
                    config=og_config,
                    unc_map={0: 0.5, 1: 0.5, 2: 0.5, 3: train_uncertain, 4: 1.0},
                    seed=42,
                )
            used_train_dl = train_dl_uncertain if train_uncertain is not None else train_dl
            used_val_dl = val_dl_uncertain if train_uncertain is not None else val_dl
            print(f'Experiment {split+1}/{og_config["cv"]}')

            config = copy.deepcopy(og_config)
            config["architecture"] = "ConceptEmbeddingModel"
			config["extra_name"] = f"Uncertain{train_uncertain}" if train_uncertain is not None else ""
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
                    train_dl=used_train_dl,
                    val_dl=used_val_dl,
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
                    concept_group_map=CUB_CONCEPT_GROUP_MAP,
                    intervened_groups=list(
                        range(
                            0,
                            len(CUB_CONCEPT_GROUP_MAP) + 1,
                            config.get('intervention_freq', 4),
                        )
                    ),
                    gpu=gpu if gpu else None,
                    config=config,
                    test_dl=test_dl,
                    train_dl=used_train_dl,
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
            if test_uncertain:
                for unc_value in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    unc_map = {0: 0.5, 1: 0.5, 2: 0.5, 3: unc_value, 4: 1.0}
                    test_dl_uncertain_current = load_data(
                        pkl_paths=[test_data_path],
                        use_attr=True,
                        no_img=False,
                        batch_size=og_config['batch_size'],
                        uncertain_label=True,
                        n_class_attr=2,
                        image_dir='images',
                        resampling=False,
                        root_dir=CUB_DIR,
                        num_workers=og_config['num_workers'],
                        unc_map=unc_map
                    )
                    results[f'{split}'][f'test_acc_y_uncert_{unc_value}_ints_{full_run_name}'] = \
                        intervene_in_cbm(
                            concept_selection_policy=random_int_policy,
                            concept_group_map=CUB_CONCEPT_GROUP_MAP,
                            intervened_groups=list(
                                range(
                                    0,
                                    len(CUB_CONCEPT_GROUP_MAP) + 1,
                                    config.get('intervention_freq', 4),
                                )
                            ),
                            gpu=gpu if gpu else None,
                            config=config,
                            test_dl=test_dl_uncertain_current,
                            train_dl=used_train_dl,
                            n_tasks=n_tasks,
                            n_concepts=n_concepts,
                            result_dir=result_dir,
                            imbalance=imbalance,
                            adversarial_intervention=False,
                            rerun=rerun,
                            old_results=old_results.get(str(split), {}).get(
                                f'test_acc_y_uncert_{unc_value}_ints_{full_run_name}'
                            ),
                        )
            print(f"\tResults for {full_run_name} in split {split}:")
            for key, val in results[f'{split}'].items():
                print(f"\t\t{key} -> {val}")
            joblib.dump(
                _filter_results(results[f'{split}'], full_run_name),
                os.path.join(result_dir, f'{full_run_name}_split_{split}_results.joblib'),
            )
            joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

            # train vanilla CBM models with both logits and sigmoidal
            # bottleneck activations
            config = copy.deepcopy(og_config)
            config["architecture"] = "ConceptBottleneckModel"
            config["bool"] = False
            config["extra_dims"] = 0
            config["extra_name"] = (
				f"Uncertain_Logit{train_uncertain}" if train_uncertain is not None else f"Logit"
            )
            config["bottleneck_nonlinear"] = "leakyrelu"
            config["sigmoidal_extra_capacity"] = False
            config["sigmoidal_prob"] = False
            cbm_logit_model, cbm_logit_test_results = \
                training.train_model(
                    gpu=gpu if gpu else None,
                    n_concepts=n_concepts,
                    n_tasks=n_tasks,
                    config=config,
                    train_dl=used_train_dl,
                    val_dl=used_val_dl,
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
                    concept_group_map=CUB_CONCEPT_GROUP_MAP,
                    intervened_groups=list(
                        range(
                            0,
                            len(CUB_CONCEPT_GROUP_MAP) + 1,
                            config.get('intervention_freq', 4),
                        )
                    ),
                    gpu=gpu if gpu else None,
                    config=config,
                    test_dl=test_dl,
                    train_dl=used_train_dl,
                    n_tasks=n_tasks,
                    n_concepts=n_concepts,
                    result_dir=result_dir,
                    split=split,
                    imbalance=imbalance,
                    adversarial_intervention=False,
                    rerun=rerun,
					batch_size=512,
                    old_results=old_results.get(int(split), {}).get(
                        f'test_acc_y_ints_{full_run_name}'
                    ),
                )

            # No uncertain interventions here as it is unclear how to do that
            # when the bottleneck's activations are unconstrained
            if test_uncertain:
                for unc_value in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    unc_map = {0: 0.5, 1: 0.5, 2: 0.5, 3: unc_value, 4: 1.0}
                    test_dl_uncertain_current = load_data(
                        pkl_paths=[test_data_path],
                        use_attr=True,
                        no_img=False,
                        batch_size=og_config['batch_size'],
                        uncertain_label=True,
                        n_class_attr=2,
                        image_dir='images',
                        resampling=False,
                        root_dir=CUB_DIR,
                        num_workers=og_config['num_workers'],
                        unc_map=unc_map
                    )
                    results[f'{split}'][f'test_acc_y_uncert_{unc_value}_ints_{full_run_name}'] = \
                        intervene_in_cbm(
                            concept_selection_policy=random_int_policy,
                            concept_group_map=CUB_CONCEPT_GROUP_MAP,
                            intervened_groups=list(
                                range(
                                    0,
                                    len(CUB_CONCEPT_GROUP_MAP) + 1,
                                    config.get('intervention_freq', 4),
                                )
                            ),
                            gpu=gpu if gpu else None,
                            config=config,
                            test_dl=test_dl_uncertain_current,
                            train_dl=used_train_dl,
                            n_tasks=n_tasks,
                            n_concepts=n_concepts,
                            result_dir=result_dir,
                            imbalance=imbalance,
                            adversarial_intervention=False,
                            rerun=rerun,
							batch_size=512,
                            old_results=old_results.get(str(split), {}).get(
                                f'test_acc_y_uncert_{unc_value}_ints_{full_run_name}'
                            ),
                        )
            print(f"\tResults for {full_run_name} in split {split}:")
            for key, val in results[f'{split}'].items():
                print(f"\t\t{key} -> {val}")
            joblib.dump(
                _filter_results(results[f'{split}'], full_run_name),
                os.path.join(result_dir, f'{full_run_name}_split_{split}_results.joblib'),
            )
            joblib.dump(results, os.path.join(result_dir, f'results.joblib'))


            # CBM with sigmoidal bottleneck
            config = copy.deepcopy(og_config)
            config["architecture"] = "ConceptBottleneckModel"
            config["extra_name"] = (
				f"Uncertain_Sigmoid{train_uncertain}" if train_uncertain is not None else f"Sigmoid"
            )
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
                    train_dl=used_train_dl,
                    val_dl=used_val_dl,
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
                    concept_group_map=CUB_CONCEPT_GROUP_MAP,
                    intervened_groups=list(
                        range(
                            0,
                            len(CUB_CONCEPT_GROUP_MAP) + 1,
                            config.get('intervention_freq', 4),
                        )
                    ),
                    gpu=gpu if gpu else None,
                    config=config,
                    test_dl=test_dl,
                    train_dl=used_train_dl,
                    n_tasks=n_tasks,
                    n_concepts=n_concepts,
                    result_dir=result_dir,
                    split=split,
                    imbalance=imbalance,
                    adversarial_intervention=False,
                    rerun=rerun,
					batch_size=512,
                    old_results=old_results.get(int(split), {}).get(
                        f'test_acc_y_ints_{full_run_name}'
                    ),
                )
            if test_uncertain:
                for unc_value in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    unc_map = {0: 0.5, 1: 0.5, 2: 0.5, 3: unc_value, 4: 1.0}
                    test_dl_uncertain_current = load_data(
                        pkl_paths=[test_data_path],
                        use_attr=True,
                        no_img=False,
                        batch_size=og_config['batch_size'],
                        uncertain_label=True,
                        n_class_attr=2,
                        image_dir='images',
                        resampling=False,
                        root_dir=CUB_DIR,
                        num_workers=og_config['num_workers'],
                        unc_map=unc_map
                    )
                    results[f'{split}'][f'test_acc_y_uncert_{unc_value}_ints_{full_run_name}'] = \
                        intervene_in_cbm(
                            concept_selection_policy=random_int_policy,
                            concept_group_map=CUB_CONCEPT_GROUP_MAP,
                            intervened_groups=list(
                                range(
                                    0,
                                    len(CUB_CONCEPT_GROUP_MAP) + 1,
                                    config.get('intervention_freq', 4),
                                )
                            ),
                            gpu=gpu if gpu else None,
                            config=config,
                            test_dl=test_dl_uncertain_current,
                            train_dl=used_train_dl,
                            n_tasks=n_tasks,
                            n_concepts=n_concepts,
                            result_dir=result_dir,
                            imbalance=imbalance,
                            adversarial_intervention=False,
                            rerun=rerun,
							batch_size=512,
                            old_results=old_results.get(str(split), {}).get(
                                f'test_acc_y_uncert_{unc_value}_ints_{full_run_name}'
                            ),
                        )

            # save results
            print(f"\tResults for {full_run_name} in split {split}:")
            for key, val in results[f'{split}'].items():
                print(f"\t\t{key} -> {val}")
            joblib.dump(
                _filter_results(results[f'{split}'], full_run_name),
                os.path.join(result_dir, f'{full_run_name}_split_{split}_results.joblib'),
            )
            joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Runs CEM intervention experiments in CUB dataset.'
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
        '--output_dir',
        '-o',
        default='results/cub_intervention_experiments/',
        help=(
            "directory where we will dump our experiment's results. If not "
            "given, then we will use ./results/cub/."
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
        "--test_uncertain",
        action="store_true",
        default=False,
        help="test uncertain concept labels during interventions.",
    )
    parser.add_argument(
        "--include_uncertain_train",
        action="store_true",
        default=False,
        help="includes uncertainty in concept labels at training time.",
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
    main(
        rerun=args.rerun,
        result_dir=args.output_dir,
        project_name=args.project_name,
        num_workers=args.num_workers,
        global_params=args.param,
        test_uncertain=args.test_uncertain,
        include_uncertain_train=args.include_uncertain_train,
        gpu=(not args.force_cpu) and (torch.cuda.is_available()),
    )
