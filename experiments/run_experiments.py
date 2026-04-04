"""
usage: run_experiments.py [-h] [--config config.yaml] [--project_name name]
                          [--output_dir path] [--rerun] [--num_workers N] [-d]
                          [--force_cpu] [-p param_name value]
                          [--filter_out regex] [--filter_in regex]
                          [--filter_in_file model_selection_file.joblib]
                          [--extra_datasets_filter_in_file model_selection_file.joblib]
                          [--only_previously_selected] [--model_selection_metrics metric_name]
                          [--summary_table_metrics metric_name pretty_name metric_name pretty_name]
                          [-m group_pattern_regex group_name group_pattern_regex group_name]
                          [--fast_run] [--no_new_runs] [--use_auc]
                          [--reverse] [--use_dataset_cache]
                          [--out_selected_config out_config.yaml]

Runs the set of experiments of CBM-like models in the provided configuration file.

optional arguments:
  -h, --help            show this help message and exit
  --config config.yaml, -c config.yaml
                        YAML file with the configuration for the set of
                        experiments to run.
  --project_name name   Project name used for Weights & Biases monitoring. If
                        not provided, then we will not log in W&B.
  --output_dir path, -o path
                        directory where we will dump our experiment's results.
  --rerun, -r           If set, then we will force a rerun of the entire
                        experiment even if valid results are found in the
                        provided output directory. Note that this may overwrite
                        and previous results, so use with care.
  --num_workers N       number of workers used for data feeders. Do not use more
                        workers than cores in the machine.
  -d, --debug           starts debug mode in our program.
  --force_cpu           forces CPU training.
  -p param_name value, --param param_name value
                        Allows the passing of a config param that will overwrite
                        anything passed as part of the config file itself.
  --filter_out regex    skips runs whose names match the regexes provided via
                        this argument. These regexes must follow Python's regex
                        syntax.
  --filter_in regex     includes only runs whose names match the regexes provided
                        with this argument. These regexes must follow Python's
                        regex syntax.
  --filter_in_file model_selection_file.joblib
                        includes only runs whose names are in the joblib file
                        outputed from a previous model selection run.
  --extra_datasets_filter_in_file model_selection_file.joblib
                        includes for extra dataset evaluation only runs whose
                        names are in the joblib file outputed from a previous
                        model selection run.
  --only_previously_selected
                        it runs the models that were only previously selected by
                        the model selection ran on a previous iteration of this
                        experiment
  --model_selection_metrics metric_name
                        metrics to be used to make a summary table by selecting
                        models based on some (validation) metric. If provided,
                        the one must also provide groups via the
                        model_selection_groups argument.
  --summary_table_metrics metric_name pretty_name metric_name pretty_name
                        List of metrics to be included as part of the final
                        summary table of this run.
  -m group_pattern_regex group_name group_pattern_regex group_name,
    --model_selection_groups group_pattern_regex group_name group_pattern_regex group_name
                        Performs model selection based on the requested model
                        selection metrics by grouping methods that match the
                        Python regex `group_pattern_regex` into a single group
                        with name `group_name`.
  --fast_run            does not perform a check on expected result keys on
                        previously found results. Only use if you are certain
                        old results are not stale and are complete!
  --no_new_runs         does excecute any new runs whose results were not
                        previously computed/cached.
  --use_auc             use ROC-AUC as the main performance metric rather than
                        accuracy.
  --reverse             Reverses the order in which we process experiments
                        (useful for multiprocessing).
  --use_dataset_cache   we will avoid re-generating dataset loaders if they have
                        already been generated for previous runs.
  --out_selected_config out_config.yaml
                        outputs a config that will kick-start the same
                        experiments with only the selected models after running
                        the entire model selection process.
"""
import argparse
import copy
import gc
import hashlib
import joblib
import json
import logging
import multiprocessing
import numpy as np
import os
import re
import sys
import torch
import yaml



torch.multiprocessing.set_sharing_strategy('file_system')
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pytorch_lightning import seed_everything

import cem.data.awa2_loader as awa2_data_module
import cem.data.celeba_loader as celeba_data_module
import cem.data.cifar_10_loader as cifar_10_data_module
import cem.data.color_mnist_add as color_mnist_data_module
import cem.data.CUB200.cub_loader as cub_data_module
import cem.data.mnist_add as mnist_data_module
try:
    import cem.data.siim_arc_loader as siim_arc_data_module
except ImportError:
    siim_arc_data_module = None
import cem.data.sun_loader as sun_data_module
import cem.data.sun_loader_v2 as sun_data_module_v2
import cem.data.traffic_loader as traffic_data_module
import cem.data.waterbirds_loader as waterbirds_data_module
import cem.train.train_blackbox as train_blackbox
import cem.train.train_fixed_cem as train_fixed_cem
import cem.train.train_glancenet as train_glancenet
import cem.train.train_mixcem as train_mixcem
import cem.train.train_pcbm as train_pcbm
import cem.train.train_prob_cbm as train_prob_cbm
import cem.train.training as training
import cem.train.utils as utils

import experiments.evaluate_models as evaluate_models
import experiments.experiment_utils as experiment_utils

from cem.data.synthetic_loaders import (
    get_synthetic_data_loader, get_synthetic_num_features
)
from cem.data.utils import LambdaDataset, transform_from_config


################################################################################
## Global Register
################################################################################

DATASET_CACHE = {}

################################################################################
## HELPER FUNCTIONS
################################################################################

def update_statistics(
    aggregate_results,
    test_results,
    prefix='',
):
    for key, val in test_results.items():
        aggregate_results[prefix + key] = val


def hash_function(func):
    if func is None:
        return hash(None)
    bytecode = func.__code__.co_code
    return hashlib.sha256(bytecode).hexdigest()

def _apply_transformation(dl, transformation_config):
    new_ds = LambdaDataset(
        dl.dataset,
        transform_from_config(transformation_config),
    )
    return torch.utils.data.DataLoader(
        new_ds,
        batch_size=dl.batch_size,
        num_workers=dl.num_workers,
    )

def _update_config_with_dataset(
    config,
    train_dl,
    n_concepts,
    n_tasks,
    concept_map,
    data_module=None,
):
    config["n_concepts"] = config.get(
        "n_concepts",
        n_concepts,
    )
    config["n_tasks"] = config.get(
        "n_tasks",
        n_tasks,
    )
    config["concept_map"] = config.get(
        "concept_map",
        concept_map,
    )
    for data in train_dl:
        config['input_shape'] = tuple(data[0].shape)
        break

    task_class_weights = None

    if config.get('use_task_class_weights', False):
        logging.info(
            f"Computing task class weights in the training dataset with "
            f"size {len(train_dl)}..."
        )
        attribute_count = np.zeros((max(n_tasks, 2),))
        samples_seen = 0
        for i, data in enumerate(train_dl):
            if len(data) == 2:
                (_, (y, _)) = data
            else:
                y = data[1]
            if n_tasks > 1:
                y = torch.nn.functional.one_hot(
                    y,
                    num_classes=n_tasks,
                ).cpu().detach().numpy()
            else:
                y = torch.cat(
                    [torch.unsqueeze(1 - y, dim=-1), torch.unsqueeze(y, dim=-1)],
                    dim=-1,
                ).cpu().detach().numpy()
            attribute_count += np.sum(y, axis=0)
            samples_seen += y.shape[0]
        print("Class distribution is:", attribute_count / samples_seen)
        if n_tasks > 1:
            task_class_weights = samples_seen / attribute_count - 1
        else:
            task_class_weights = np.array(
                [attribute_count[0]/attribute_count[1]]
            )
    return task_class_weights

def _generate_dataset_and_update_config(
    experiment_config,
    use_dataset_cache=False,
):
    if experiment_config.get("dataset_config", None) is None:
        raise ValueError(
            "A dataset_config must be provided for each experiment run!"
        )

    dataset_config = experiment_config['dataset_config']
    dataset_config['num_workers'] = experiment_config.get(
        'num_workers',
        dataset_config.get('num_workers', 8),
    )

    ds_name = dataset_config["dataset"].strip().lower()
    logging.debug(
        f"The dataset's root directory is {dataset_config.get('root_dir')}"
    )
    intervention_config = experiment_config.get('intervention_config', {})
    if ds_name == "cub":
        data_module = cub_data_module
    elif ds_name == "awa2":
        data_module = awa2_data_module
    elif ds_name == "waterbirds":
        data_module = waterbirds_data_module
    elif ds_name == "celeba":
        data_module = celeba_data_module
    elif ds_name in ["xor", "vector", "dot", "trig"]:
        data_module = get_synthetic_data_loader(ds_name)
    elif ds_name == "mnist_add":
        data_module = mnist_data_module
    elif ds_name == "traffic":
        data_module = traffic_data_module
    elif ds_name in ["sun", "sun397", "sun_v2", "sun397_v2"]:
        use_sun_v2 = dataset_config.get("use_sun_loader_v2", False) or (ds_name in ["sun_v2", "sun397_v2"])
        data_module = sun_data_module_v2 if use_sun_v2 else sun_data_module
    elif ds_name in ["siim_arc", "siim-arc", "siim"]:
        if siim_arc_data_module is None:
            raise ValueError(
                "SIIM dataset support is not available in this checkout "
                "(missing cem.data.siim_arc_loader)."
            )
        data_module = siim_arc_data_module
    elif ds_name in ['cifar10', 'cifar_10']:
        data_module = cifar_10_data_module
    elif ds_name == "color_mnist_add":
        data_module = color_mnist_data_module
    else:
        raise ValueError(f"Unsupported dataset {dataset_config['dataset']}!")


    transformation_config = dataset_config.get("transformation_config", {})
    train_transform_config = dataset_config.get(
        "train_transformation_config",
        transformation_config,
    )
    if not train_transform_config.get('post_generation', True):
        train_transform_fn = transform_from_config(train_transform_config)
    else:
        train_transform_fn = None

    test_transform_config = dataset_config.get(
        "test_transformation_config",
        transformation_config,
    )
    if not test_transform_config.get('post_generation', True):
        test_transform_fn = transform_from_config(test_transform_config)
    else:
        test_transform_fn = None

    val_transform_config = dataset_config.get(
        "val_transformation_config",
        transformation_config,
    )
    if not val_transform_config.get('post_generation', True):
        val_transform_fn = transform_from_config(val_transform_config)
    else:
        val_transform_fn = None

    if use_dataset_cache:
        key_dict = copy.deepcopy(dataset_config)
        key_dict['seed'] = 42
        key_dict['train_transform_fn'] = hash_function(train_transform_fn)
        key_dict['test_transform_fn'] = hash_function(test_transform_fn)
        key_dict['val_transform_fn'] = hash_function(val_transform_fn)
        config_key = hashlib.sha256(
            json.dumps(key_dict, sort_keys=True).encode()
        ).hexdigest()
        if config_key not in DATASET_CACHE:
            print(
                "First time we find dataset config, so adding it into our cache"
            )
            DATASET_CACHE[config_key] = data_module.generate_data(
                config=dataset_config,
                seed=42,
                output_dataset_vars=True,
                root_dir=dataset_config.get('root_dir', None),
                train_sample_transform=train_transform_fn,
                test_sample_transform=test_transform_fn,
                val_sample_transform=val_transform_fn,
            )

        # Now we can safely fetch it from the dataset cache
        (
            train_dl,
            val_dl,
            test_dl,
            imbalance,
            (n_concepts, n_tasks, concept_map),
        ) = DATASET_CACHE[config_key]

    else:
        train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = \
            data_module.generate_data(
                config=dataset_config,
                seed=42,
                output_dataset_vars=True,
                root_dir=dataset_config.get('root_dir', None),
                train_sample_transform=train_transform_fn,
                test_sample_transform=test_transform_fn,
                val_sample_transform=val_transform_fn,
            )

    # For now, we assume that all concepts have the same
    # aquisition cost
    acquisition_costs = None


    task_class_weights = _update_config_with_dataset(
        config=experiment_config,
        train_dl=train_dl,
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        concept_map=concept_map,
    )

    if experiment_config['c_extractor_arch'] == "mnist_extractor":
        num_operands = dataset_config.get('num_operands', 32)
        experiment_config["c_extractor_arch"] = \
            experiment_utils.get_mnist_extractor_arch(
                input_shape=(
                    dataset_config.get('batch_size', 512),
                    num_operands,
                    28,
                    28,
                ),
                in_channels=num_operands,
            )
    if experiment_config['c_extractor_arch'] == "color_mnist_extractor":
        num_operands = dataset_config.get('num_operands', 32)
        in_channels = 3 if (dataset_config.get('concat_dim', 'y') != 'c') else (
                num_operands * (
                1 if len(dataset_config.get('colors', ['gray'])) == 0 else 3
            )
        )
        experiment_config["c_extractor_arch"] = \
            experiment_utils.get_mnist_extractor_arch(
                input_shape=(
                    dataset_config.get('batch_size', 512),
                    in_channels,
                    28*num_operands if (dataset_config.get('concat_dim', 'y') == 'y') else 28,
                    28*num_operands if (dataset_config.get('concat_dim', 'y') == 'x') else 28,
                ),
                in_channels=in_channels,
            )
    elif experiment_config['c_extractor_arch'] == 'synth_extractor':
        input_features = get_synthetic_num_features(ds_name)
        def synth_c_extractor_arch(
            output_dim,
            pretrained=False,
        ):
            if output_dim is None:
                output_dim = 128
            return torch.nn.Sequential(*[
                torch.nn.Linear(input_features, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, output_dim),
            ])
        experiment_config["c_extractor_arch"] = synth_c_extractor_arch

    elif isinstance(experiment_config['c_extractor_arch'], str) and (
        experiment_config['c_extractor_arch'].lower() == 'mlp'
    ):
        input_features = experiment_config["input_shape"][-1]
        def c_extractor_arch(
            output_dim,
        ):
            if output_dim is None:
                output_dim = 128
            layer_units = [input_features] + experiment_config.get('c_extractor_arch_layers', []) + [output_dim]
            used_layers = []
            for i in range(1, len(layer_units)):
                num_units = layer_units[i]
                used_layers.append(torch.nn.Linear(layer_units[i - 1], num_units))
                if i != len(layer_units) - 1:
                    used_layers.append(torch.nn.LeakyReLU())
                if i in experiment_config.get('batch_norm_layers', []) and (
                    i != len(layer_units) - 1
                ):
                    used_layers.append(
                        torch.nn.BatchNorm1d(num_units)
                    )
            return torch.nn.Sequential(*used_layers)

        experiment_config["c_extractor_arch"] = c_extractor_arch

    elif isinstance(experiment_config['c_extractor_arch'], str) and (
        experiment_config['c_extractor_arch'].lower() == 'cnn'
    ):
        input_shape = experiment_config["input_shape"]
        def c_extractor_arch(output_dim=n_tasks):
            if output_dim is None:
                output_dim = 128
            layers = []
            current_shape = input_shape[1:]
            extractor_config = experiment_config.get(
                'c_extractor_config',
                {},
            )
            for layer in extractor_config.get("cnn_layers", []):
                layers.append(torch.nn.Conv2d(
                    in_channels=current_shape[0],
                    out_channels=layer['out_channels'],
                    kernel_size=layer['filter'],
                    padding='same',
                ))
                current_shape = (layer['out_channels'], *current_shape[1:])
                act_fn = layer.get(
                    'activation_fn',
                    extractor_config.get('activation_fn', 'relu'),
                )
                if act_fn.lower() == 'relu':
                    layers.append(torch.nn.ReLU())
                elif act_fn.lower() == 'leakyrelu':
                    layers.append(torch.nn.LeakyReLU())
                else:
                    raise ValueError(
                        f'Unsupported activation {act_fn}'
                    )
                if layer.get('batchnorm', False):
                    layers.append(
                        torch.nn.BatchNorm2d(num_features=layer['out_channels'])
                    )
                if layer.get('max_pool_kernel', False):
                    kernel = layer['max_pool_kernel']
                    layers.append(
                        torch.nn.MaxPool2d(kernel)
                    )
                    current_shape = (
                        current_shape[0],
                        int(np.floor(1 + (current_shape[1] - (kernel[0] - 1) - 1) / kernel[0])),
                        int(np.floor(1 + (current_shape[2] - (kernel[1] - 1) - 1) / kernel[1])),
                    )
                elif layer.get('avg_pool_kernel', False):
                    kernel = layer['avg_pool_kernel']
                    layers.append(
                        torch.nn.AvgPool2d(kernel)
                    )
                    current_shape = (
                        current_shape[0],
                        int(np.floor(1 + (current_shape[1] - (kernel[0] - 1) - 1) / kernel[0])),
                        int(np.floor(1 + (current_shape[2] - (kernel[1] - 1) - 1) / kernel[1])),
                    )
            layers.append(torch.nn.Flatten())
            current_shape = np.prod(current_shape)
            for num_acts in extractor_config.get('mlp_layers', []):
                layers.append(torch.nn.Linear(
                    current_shape,
                    num_acts,
                ))
                current_shape = num_acts
                act_fn = extractor_config.get(
                    'activation_fn',
                    'relu',
                )
                if act_fn.lower() == 'relu':
                    layers.append(torch.nn.ReLU())
                elif act_fn.lower() == 'leakyrelu':
                    layers.append(torch.nn.LeakyReLU())
                else:
                    raise ValueError(
                        f'Unsupported activation {extractor_config["activation_fn"]}'
                    )
            layers.append(torch.nn.Linear(
                current_shape,
                output_dim,
            ))
            out_model = torch.nn.Sequential(*layers)
            return out_model

        experiment_config["c_extractor_arch"] = c_extractor_arch

    if train_transform_config.get('post_generation', True):
        train_dl = _apply_transformation(
            train_dl,
            transformation_config=train_transform_config,
        )
    if test_transform_config.get('post_generation', True):
        test_dl = _apply_transformation(
            test_dl,
            transformation_config=test_transform_config,
        )
    if not (val_dl is None):
        if val_transform_config.get('post_generation', True):
            val_dl = _apply_transformation(
                val_dl,
                transformation_config=val_transform_config,
            )

    if (data_module is not None) and (experiment_config.get(
        'initial_concept_embeddings',
        None,
    ) is not None):
        experiment_config['initial_concept_embeddings'] = \
            data_module.get_concept_embeddings(
                config=experiment_config,
                root_dir=dataset_config.get('root_dir', None),
            )
    if (data_module is not None) and hasattr(
        data_module,
        'get_concept_descriptions',
    ):
        experiment_config['concept_descriptions'] = \
            data_module.get_concept_descriptions(
                config=experiment_config,
                root_dir=dataset_config.get('root_dir', None),
            )

    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        concept_map,
        task_class_weights,
        acquisition_costs,
    )

def _perform_model_selection(
    model_selection_groups,
    model_selection_metrics,
    results,
    result_dir,
    split,
    summary_table_metrics=None,
    config=None,
    included_models=None,
    use_auc=False,
):
    ############################################################################
    ## Automatic Model Selection
    ############################################################################
    prev_selected_results = None
    if (model_selection_groups is not None) and (
        model_selection_metrics is not None
    ):
        prev_selected_results = []
        for idx, model_selection_metric in enumerate(model_selection_metrics):
            model_selection_results, selection_map = \
                experiment_utils.perform_model_selection(
                    results=results,
                    selection_metric=model_selection_metric,
                    model_groupings=model_selection_groups,
                    included_models=(
                        included_models[idx] if included_models else None
                    ),
                )
            print(
                f"********** Results after model selection "
                f"with {model_selection_metric} **********"
            )
            experiment_utils.print_table(
                summary_table_metrics=summary_table_metrics,
                config=config,
                results=model_selection_results,
                result_dir=result_dir,
                split=split,
                save_name=f"output_table_{model_selection_metric}",
                use_auc=(use_auc or config.get("n_tasks", 3) <= 2),
                use_int_auc=use_auc or config.get('intervention_config', {}).get(
                    'use_auc',
                    False,
                ),
            )
            with open(
                os.path.join(
                    result_dir,
                    f'results_selection_via_{model_selection_metric}.joblib'
                ),
                'wb',
            ) as f:
                joblib.dump(model_selection_results, f)
            with open(
                os.path.join(
                    result_dir,
                    f'selected_models_{model_selection_metric}.joblib'
                ),
                'wb',
            ) as f:
                joblib.dump(selection_map, f)
            prev_selected_results.append(
                (model_selection_results, selection_map, model_selection_metric)
            )
    return prev_selected_results



def _multiprocess_run_trial(
    trial_results,
    run_name,
    run_config,
    accelerator,
    devices,
    result_dir,
    split,
    project_name,
    current_rerun,
    old_results,
    use_dataset_cache=False,
    extra_datasets_filter_in_file=None,
):
    config = run_config #copy.deepcopy(run_config)
    (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        concept_map,
        task_class_weights,
        acquisition_costs
    ) = _generate_dataset_and_update_config(
        config,
        use_dataset_cache=use_dataset_cache,
    )
    experiment_utils.evaluate_expressions(config)

    # Get the appropiate training function
    if config["architecture"] == \
            "IndependentConceptBottleneckModel":
        # Special case for now for independent CBMs
        # config["architecture"] = "ConceptBottleneckModel"
        config["sigmoidal_prob"] = True
        train_fn = training.train_independent_model
    elif config["architecture"] == \
            "SequentialConceptBottleneckModel":
        # Special case for now for sequential CBMs
        # config["architecture"] = "ConceptBottleneckModel"
        config["sigmoidal_prob"] = True
        train_fn = training.train_sequential_model
    elif config["architecture"] in [
        "ProbCBM",
        "ProbabilisticCBM",
        "ProbabilisticConceptBottleneckModel",
    ]:
        train_fn = train_prob_cbm.train_prob_cbm

    elif config['architecture'] in [
        'GlanceNet',
    ]:
        train_fn = train_glancenet.train_glancenet

    elif config['architecture'] in [
        'MixtureOfConceptEmbeddingsModel',
        'MixCEM',
        'MCIntCEM', # Legacy
    ]:
        train_fn = train_mixcem.train_mixcem

    elif config["architecture"] in [
        "PosthocCBM",
        "PCBM",
        "PosthocConceptBottleneckModel",
        "Post-hocConceptBottleneckModel",
    ]:
        train_fn = train_pcbm.train_pcbm

    elif config["architecture"] in [
        "FixedEmbConceptEmbeddingModel",
        "FixedConceptEmbeddingModel",
        "FixedCEM",
    ]:
        train_fn = train_fixed_cem.train_fixed_cem

    elif config["architecture"] in [
        "BBModel",
        "BlackBoxModel",
        "BlackBox",
    ]:
        train_fn = train_blackbox.train_blackbox
    else:
        train_fn = training.train_end_to_end_model

    # Train the model and get testing and validation results
    model, model_results = train_fn(
        run_name=run_name,
        task_class_weights=task_class_weights,
        accelerator=accelerator,
        devices=devices,
        n_concepts=config['n_concepts'],
        n_tasks=config['n_tasks'],
        input_shape=config['input_shape'],
        config=config,
        train_dl=train_dl,
        val_dl=val_dl,
        split=split,
        result_dir=result_dir,
        rerun=current_rerun,
        project_name=project_name,
        seed=(42 + split),
        imbalance=imbalance,
        old_results=old_results,
        gradient_clip_val=config.get(
            'gradient_clip_val',
            0,
        ),
    )
    update_statistics(
        aggregate_results=trial_results,
        test_results=model_results,
    )
    test_datasets = [
        (val_dl, "val"),
        (test_dl, "test"),
    ]
    # Change the model to eval mode
    was_training = model.training
    model.eval()
    # Add the no grad guard to save memory and speed things up
    with torch.no_grad():
        eval_results = evaluate_models.evaluate_model(
            model,
            config,
            test_datasets,
            train_dl,
            val_dl=val_dl,
            run_name=run_name,
            task_class_weights=task_class_weights,
            imbalance=imbalance,
            acquisition_costs=acquisition_costs,
            result_dir=result_dir,
            concept_map=concept_map,
            accelerator=accelerator,
            devices=devices,
            split=split,
            rerun=current_rerun,
            old_results=old_results,
        )
        update_statistics(
            aggregate_results=trial_results,
            test_results=eval_results,
        )

        eval_config = config.get('eval_config', {})
        run_additional = True
        if extra_datasets_filter_in_file is not None:
            run_additional = False
            for reg in extra_datasets_filter_in_file:
                if re.search(reg, f'{run_name}_split_{split}'):
                    logging.info(
                        f'Including additional dataset runs for '
                        f'{f"{run_name}_split_{split}"} as it '
                        f'matched filter-out regex {reg}'
                    )
                    run_additional = True
                    break
        config_copy = copy.deepcopy(config)
        if run_additional:
            for new_test_dl_configs in eval_config.get(
                'additional_test_sets',
                [],
            ):
                skip = False
                for skip_regex in new_test_dl_configs.get('skip_list', []):
                    if re.search(skip_regex, f'{run_name}_split_{split}'):
                        logging.info(
                            f"Skipping evaluation of dataset "
                            f"{new_test_dl_configs['name'] + '_test'} as it "
                            f"matched skip regex {skip_regex}"
                        )
                        skip = True
                        break
                if skip:
                    continue

                config_copy = copy.deepcopy(config)
                if new_test_dl_configs.get('update_previous', False):
                    config_copy['dataset_config'].update(
                        new_test_dl_configs['dataset_config']
                    )
                else:
                    config_copy['dataset_config'] = (
                        new_test_dl_configs['dataset_config']
                    )
                (
                    _,
                    _,
                    new_test_dl,
                    new_imbalance,
                    new_concept_map,
                    new_task_class_weights,
                    new_acquisition_costs
                ) = _generate_dataset_and_update_config(
                    config_copy,
                    use_dataset_cache=use_dataset_cache,
                )
                eval_results = evaluate_models.evaluate_model(
                    model,
                    config_copy,
                    [(new_test_dl, (new_test_dl_configs['name'] + "_test"))],
                    train_dl,
                    val_dl=val_dl,
                    run_name=run_name,
                    task_class_weights=new_task_class_weights,
                    imbalance=new_imbalance,
                    acquisition_costs=new_acquisition_costs,
                    result_dir=result_dir,
                    concept_map=new_concept_map,
                    accelerator=accelerator,
                    devices=devices,
                    split=split,
                    rerun=current_rerun,
                    old_results=old_results,
                )
                update_statistics(
                    aggregate_results=trial_results,
                    test_results=eval_results,
                )
    # And restore training mode if the model was previously training
    if was_training:
        model.train()
    return config_copy


################################################################################
## MAIN FUNCTION
################################################################################


def main(
    result_dir,
    experiment_config,
    rerun=False,
    project_name='',
    num_workers=8,
    global_params=None,
    accelerator="auto",
    devices="auto",
    summary_table_metrics=None,
    sort_key="Task Accuracy",
    filter_out_regex=None,
    filter_in_regex=None,
    model_selection_metrics=None,
    model_selection_groups=None,
    fast_run=False,
    no_new_runs=False,
    use_auc=False,
    use_dataset_cache=False,
    extra_datasets_filter_in_file=None,
    reverse_experiments=False,
    results_fname=None,
):
    seed_everything(42)
    # parameters for data, model, and training
    experiment_config = copy.deepcopy(experiment_config)
    if 'shared_params' not in experiment_config:
        experiment_config['shared_params'] = {}
    # Move all global things into the shared params
    shared_params = experiment_config['shared_params']
    for key, vals in experiment_config.items():
        if key not in ['runs', 'shared_params']:
            shared_params[key] = vals
    shared_params['num_workers'] = num_workers
    shared_params["trials"] = shared_params.get(
        "trials",
        1,
    )

    utils.extend_with_global_params(
        shared_params, global_params or []
    )

    # Set log level in env variable as this will be necessary for
    # subprocessing
    os.environ['LOGLEVEL'] = os.environ.get(
        'LOGLEVEL',
        logging.getLevelName(logging.getLogger().getEffectiveLevel()),
    )
    loglevel = os.environ['LOGLEVEL']
    logging.info(f'Setting log level to: "{loglevel}"')

    # Things regarding model selection
    model_selection_trials = shared_params.get(
        'model_selection_trials',
        shared_params["trials"],
    )
    models_selected_to_continue = None
    included_models = None
    results_fname = shared_params.get('results_fname', 'results')

    os.makedirs(result_dir, exist_ok=True)
    results = {}
    for split in range(
        shared_params.get("start_split", 0),
        shared_params["trials"],
    ):
        results[f'{split}'] = defaultdict(dict)
        now = datetime.now()
        print(
            f"[TRIAL "
            f"{split + 1}/{shared_params['trials']} "
            f"BEGINS AT {now.strftime('%d/%m/%Y at %H:%M:%S')}"
        )
        # And then over all runs in a given trial
        runs = experiment_config['runs']
        if reverse_experiments:
            runs.reverse()
        for current_config in runs:
            # Construct the config for this particular trial
            trial_config = copy.deepcopy(shared_params)
            trial_config.update(current_config)
            # Time to try as many seeds as requested
            for run_config in experiment_utils.generate_hyperparameter_configs(
                trial_config
            ):
                torch.cuda.empty_cache()
                run_config = copy.deepcopy(run_config)
                run_config['result_dir'] = result_dir
                run_config['split'] = split
                experiment_utils.evaluate_expressions(run_config, soft=True)

                old_results = None
                if "run_name" not in run_config:
                    run_name = (
                        f"{run_config['architecture']}"
                        f"{run_config.get('extra_name', '')}"
                    )
                    logging.warning(
                        f'Did not find a run name so using the '
                        f'name "{run_name}" by default'
                    )
                    run_config["run_name"] = run_name
                run_name = run_config["run_name"]

                # Determine filtering in and filtering out of run
                if filter_out_regex:
                    skip = False
                    for reg in filter_out_regex:
                        if re.search(reg, f'{run_name}_split_{split}'):
                            logging.info(
                                f'Skipping run '
                                f'{f"{run_name}_split_{split}"} as it '
                                f'matched filter-out regex {reg}'
                            )
                            skip = True
                            break
                    if skip:
                        continue
                if filter_in_regex:
                    found = False
                    for reg in filter_in_regex:
                        if re.search(reg, f'{run_name}_split_{split}'):
                            found = True
                            logging.info(
                                f'Including run '
                                f'{f"{run_name}_split_{split}"} as it '
                                f'did matched filter-in regex {reg}'
                            )
                            break
                    if not found:
                        logging.info(
                            f'Skipping run {f"{run_name}_split_{split}"} as it '
                            f'did not match any filter-in regexes'
                        )
                        continue

                if models_selected_to_continue and (
                    run_name not in models_selected_to_continue
                ):
                    logging.info(
                        f'Skipping run {f"{run_name}_split_{split}"} it was '
                        f'not selected based on any of the model-selection '
                        f'metrics measured after fold {model_selection_trials}'
                    )
                    continue

                # Determine training rerun or not
                current_results_path = os.path.join(
                    result_dir,
                    f'{run_name}_split_{split}_{results_fname}.joblib'
                )
                current_rerun = experiment_utils.determine_rerun(
                    config=run_config,
                    rerun=rerun,
                    split=split,
                    run_name=run_name,
                )
                if current_rerun:
                    logging.warning(
                        f"We will rerun model {run_name}_split_{split} "
                        f"as requested by the config"
                    )
                if (not current_rerun) and os.path.exists(current_results_path):
                    try:
                        with open(current_results_path, 'rb') as f:
                            old_results = joblib.load(f)
                    except Exception as e:
                        logging.info(
                            f'\t\t[IMPORTANT] We found previous results for '
                            f'run {run_name} at trial {split + 1} but we were '
                            f'unable to properly open them after encountering '
                            f'exception {e}'
                        )
                        old_results = None
                if fast_run and (old_results is not None):
                    logging.info(
                        f'\t\t[IMPORTANT] We found previous results for '
                        f'run {run_name} at trial {split + 1} and WILL '
                        f'use them without verifying all expected evaluation '
                        f'keys are present as we are running in FAST RUN mode.'
                    )
                    results[f'{split}'][run_name].update(old_results)
                    continue
                if no_new_runs:
                    if old_results is not None:
                        logging.info(
                            f'\t\t[IMPORTANT] We found previous results for '
                            f'run {run_name} at trial {split + 1} and WILL '
                            f'use them without verifying all expected evaluation '
                            f'keys are present as we are running in NO NEW RUNS mode.'
                        )
                        results[f'{split}'][run_name].update(old_results)
                        continue
                    else:
                        logging.info(
                            f'\t\t[IMPORTANT] Skipping '
                            f'run {run_name} at trial {split + 1} as no '
                            f'previous results were found and we are running '
                            f'in NO NEW RUNS mode.'
                        )
                        continue

                if int(os.environ.get("MULTIPROCESS", "0")):
                    manager = multiprocessing.Manager()
                    trial_results = manager.dict()
                    context = multiprocessing.get_context('spawn')
                    p = context.Process(
                        target=_multiprocess_run_trial,
                        kwargs=dict(
                            trial_results=trial_results,
                            run_name=run_name,
                            run_config=run_config,
                            accelerator=accelerator,
                            devices=devices,
                            result_dir=result_dir,
                            split=split,
                            project_name=project_name,
                            current_rerun=current_rerun,
                            old_results=old_results,
                            use_dataset_cache=use_dataset_cache,
                            extra_datasets_filter_in_file=extra_datasets_filter_in_file,
                        ),
                    )
                    p.start()
                    logging.debug(f"\t\tStarting run in subprocess {p.pid}")
                    p.join()
                    if p.exitcode:
                        raise ValueError(
                            f'Subprocess for trial {split + 1} of {run_name} failed!'
                        )
                    p.kill()
                else:
                    trial_results = {}
                    _multiprocess_run_trial(
                        trial_results=trial_results,
                        run_name=run_name,
                        run_config=run_config,
                        accelerator=accelerator,
                        devices=devices,
                        result_dir=result_dir,
                        split=split,
                        project_name=project_name,
                        current_rerun=current_rerun,
                        old_results=old_results,
                        use_dataset_cache=use_dataset_cache,
                        extra_datasets_filter_in_file=extra_datasets_filter_in_file,
                    )
                update_statistics(
                    aggregate_results=results[f'{split}'][run_name],
                    test_results=trial_results,
                )

                logging.debug(
                    f"\tResults for {run_name} in split {split}:"
                )
                for key, val in results[f'{split}'][run_name].items():
                    logging.debug(f"\t\t{key} -> {val}")
                with open(current_results_path, 'wb') as f:
                    joblib.dump(
                        results[f'{split}'][run_name],
                        f,
                    )

                # Clean-up memory to setup everything for the next experiment
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


            # Save this run's results
            if run_config.get("start_split", 0) == 0:
                attempt = 0
                # We will try and dump things a few times in case there
                # are other threads/processes currently modifying or
                # writing this same file
                while attempt < 5:
                    try:
                        with open(
                            os.path.join(result_dir, f'{results_fname}.joblib'),
                            'wb',
                        ) as f:
                            joblib.dump(results, f)
                        break
                    except Exception as e:
                        print(e)
                        print(
                            "FAILED TO SERIALIZE RESULTS TO",
                            os.path.join(result_dir, f'{results_fname}.joblib')
                        )
                        attempt += 1
                if attempt == 5:
                    raise ValueError(
                        "Could not serialize " +
                        os.path.join(result_dir, f'{results_fname}.joblib') +
                        " to disk"
                    )
                then = datetime.now()
                diff = then - now
                diff_minutes = diff.total_seconds() / 60
                logging.debug(
                    f"\tTrial {split + 1} COMPLETED for {run_name} ending "
                    f"at {then.strftime('%d/%m/%Y at %H:%M:%S')} "
                    f"({diff_minutes:.4f} minutes):"
                )

            # And print table after the entire trial has been completed
            print(f"********** Results in between trial {split + 1} **********")
            experiment_utils.print_table(
                config=experiment_config,
                results=results,
                summary_table_metrics=summary_table_metrics,
                sort_key=sort_key,
                result_dir=None,
                split=split,
                use_auc=(use_auc or run_config.get("n_tasks", 3) <= 2),
                use_int_auc=use_auc or run_config.get('intervention_config', {}).get(
                    'use_auc',
                    False,
                ),
            )
        if (model_selection_groups is not None) and (
            model_selection_metrics is not None
        ) and (
            models_selected_to_continue is None
        ) and (
            (split + 1) >= model_selection_trials
        ):
            # Then time to do model selection to avoid running every
            # iteration on every seed
            model_selection_results = _perform_model_selection(
                model_selection_groups=model_selection_groups,
                model_selection_metrics=model_selection_metrics,
                results=results,
                result_dir=result_dir,
                split=split,
                summary_table_metrics=summary_table_metrics,
                config=experiment_config,
                use_auc=use_auc,
            )
            models_selected_to_continue = set()
            included_models = []
            for _, selection_map, _ in model_selection_results:
                included_models.append(set())
                for _, group_selected_method in selection_map.items():
                    included_models[-1].add(group_selected_method)
                    models_selected_to_continue.add(group_selected_method)

            logging.debug(f"\t\tDone with trial {split + 1}")
    print(f"********** End Experiment Results **********")
    experiment_utils.print_table(
        config=experiment_config,
        results=results,
        summary_table_metrics=summary_table_metrics,
        sort_key=sort_key,
        result_dir=result_dir,
        split=split,
        use_auc=(use_auc or run_config.get("n_tasks", 3) <= 2),
        use_int_auc=use_auc or run_config.get('intervention_config', {}).get(
            'use_auc',
            False,
        ),
    )
    # And repring the final model selection table after all other folds have
    # been computed
    _perform_model_selection(
        model_selection_groups=model_selection_groups,
        model_selection_metrics=model_selection_metrics,
        results=results,
        result_dir=result_dir,
        split=split,
        summary_table_metrics=summary_table_metrics,
        config=experiment_config,
        included_models=included_models,
        use_auc=use_auc,
    )
    logging.debug(f"\t\tDone complete experiment after {split + 1} trials")
    return results


################################################################################
## Arg Parser
################################################################################


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Runs the set of experiments of CBM-like models in the provided '
            'configuration file.'
        ),
    )
    parser.add_argument(
        '--config',
        '-c',
        help=(
            "YAML file with the configuration for the set of experiments to "
            "run."
        ),
        metavar="config.yaml",
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
        default=None,
        help=(
            "directory where we will dump our experiment's results."
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
        metavar=('param_name', 'value'),
        help=(
            'Allows the passing of a config param that will overwrite '
            'anything passed as part of the config file itself.'
        ),
        default=[],
    )
    parser.add_argument(
        "--filter_out",
        action='append',
        metavar=('regex'),
        default=None,
        help=(
            "skips runs whose names match the regexes provided via this "
            "argument. These regexes must follow Python's regex syntax."
        ),
    )
    parser.add_argument(
        "--filter_in",
        action='append',
        metavar=('regex'),
        default=None,
        help=(
            "includes only runs whose names match the regexes provided with "
            "this argument. These regexes must follow Python's regex syntax."
        ),
    )
    parser.add_argument(
        "--filter_in_file",
        action='append',
        metavar=('model_selection_file.joblib'),
        default=None,
        help=(
            "includes only runs whose names are in the joblib file outputed "
            "from a previous model selection run."
        ),
    )
    parser.add_argument(
        "--extra_datasets_filter_in_file",
        action='append',
        metavar=('model_selection_file.joblib'),
        default=None,
        help=(
            "includes for extra dataset evaluation only runs whose names are "
            "in the joblib file outputed from a previous model selection run."
        ),
    )
    parser.add_argument(
        "--only_previously_selected",
        action="store_true",
        default=False,
        help=(
            "it runs the models that were only previously selected by the "
            "model selection ran on a previous iteration of this experiment"
        ),
    )
    parser.add_argument(
        "--model_selection_metrics",
        action='append',
        metavar=('metric_name'),
        default=None,
        help=(
            "metrics to be used to make a summary table by selecting models "
            "based on some (validation) metric. If provided, the one must "
            "also provide groups via the model_selection_groups argument."
        ),
    )
    parser.add_argument(
        "--summary_table_metrics",
        action='append',
        nargs=2,
        metavar=('metric_name pretty_name'),
        help=(
            'List of metrics to be included as part of the final summary '
            'table of this run.'
        ),
        default=None,
    )

    parser.add_argument(
        "-m",
        "--model_selection_groups",
        action='append',
        nargs=2,
        metavar=('group_pattern_regex group_name'),
        help=(
            'Performs model selection based on the requested model selection '
            'metrics by grouping methods that match the Python regex '
            '`group_pattern_regex` into a single group with name '
            '`group_name`.'
        ),
        default=[],
    )

    parser.add_argument(
         "--fast_run",
         action="store_true",
         default=False,
         help=(
             "does not perform a check on expected result keys on previously "
             "found results. Only use if you are certain old results are not "
             "stale and are complete!"
         ),
     )
    parser.add_argument(
        '--no_new_runs',
        action="store_true",
         default=False,
         help=(
             "does excecute any new runs whose results were not previously "
             "computed/cached."
         ),
    )
    parser.add_argument(
        '--use_auc',
        action="store_true",
         default=False,
         help=(
             "use ROC-AUC as the main performance metric rather than accuracy."
         ),
    )
    parser.add_argument(
        '--reverse',
        action="store_true",
         default=False,
         help=(
             "Reverses the order in which we process experiments (useful for multiprocessing)."
         ),
    )

    parser.add_argument(
        '--use_dataset_cache',
        action="store_true",
         default=False,
         help=(
            "we will avoid re-generating dataset loaders if they have already "
            "been generated for previous runs."
         ),
    )
    parser.add_argument(
        '--out_selected_config',
        default=None,
        help=(
            'outputs a config that will kick-start the same experiments with '
            'only the selected models after running the entire model selection '
            'process.'
        ),
        metavar='out_config.yaml',
        type=str,
    )
    return parser


################################################################################
## Main Entry Point
################################################################################

if __name__ == '__main__':
    # Build our arg parser first
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.project_name:
        # Lazy import to avoid importing unless necessary
        pass #import wandb
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    if args.config:
        with open(args.config, "r") as f:
            loaded_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        loaded_config = {}
    if "shared_params" not in loaded_config:
        loaded_config["shared_params"] = {}
    if "runs" not in loaded_config:
        loaded_config["runs"] = []

    if args.output_dir is not None:
        loaded_config['results_dir'] = args.output_dir
    if args.debug:
        print(json.dumps(loaded_config, sort_keys=True, indent=4))
    results_dir = loaded_config.get(
        'results_dir',
        loaded_config.get('shared_params', {}).get('results_dir', 'results')
    )
    logging.info(f"Results will be dumped in {results_dir}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    # Write down the actual command executed
    # And the configuration file
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    loaded_config["time_last_called"] = now.strftime("%Y/%m/%d at %H:%M:%S")
    with open(
        os.path.join(results_dir, f"command_{dt_string}.txt"),
        "w",
    ) as f:
        command_args = [
            arg if " " not in arg else f'"{arg}"' for arg in sys.argv
        ]
        command = "python " + " ".join(command_args) + "\n"
        f.write(command)
    with open(
        os.path.join(results_dir, f"last_run_command.txt"),
        "w",
    ) as f:
        f.write(command)

    # Also save the current experiment configuration
    with open(
        os.path.join(
            results_dir,
            f"experiment_{dt_string}_config.yaml")
        ,
        "w"
    ) as f:
        yaml.dump(loaded_config, f)

    with open(
        os.path.join(
            results_dir,
            f"last_run_experiment_config.yaml")
        ,
        "w"
    ) as f:
        yaml.dump(loaded_config, f)

    # Finally, time to actually call our main function!
    model_selection_groups = loaded_config.get("model_selection_groups", None)
    if args.model_selection_groups:
        model_selection_groups = args.model_selection_groups

    summary_table_metrics = loaded_config.get("summary_table_metrics", None)
    if args.summary_table_metrics:
        summary_table_metrics = args.summary_table_metrics

    model_selection_metrics = loaded_config.get("model_selection_metrics", None)
    if args.model_selection_metrics:
        model_selection_metrics = args.model_selection_metrics

    given_filter_in_file = args.filter_in_file
    if args.only_previously_selected and model_selection_metrics:
        if given_filter_in_file is None:
            given_filter_in_file = []
        for model_selection_metric in model_selection_metrics:
            given_filter_in_file.append(
                os.path.join(
                    results_dir,
                    f'selected_models_{model_selection_metric}.joblib'
                )
            )

    if given_filter_in_file is not None:
        if args.filter_in is None:
            args.filter_in = []
        for file_path in given_filter_in_file:
            if not os.path.exists(file_path):
                raise ValueError(
                    f'Path for filter-in file {file_path} is not a valid path'
                )
            loaded_selection = joblib.load(file_path)
            for _, method_name in loaded_selection.items():
                args.filter_in.append(method_name)

    extra_datasets_filter_in_file = None
    if args.extra_datasets_filter_in_file is not None:
        extra_datasets_filter_in_file = []
        if args.filter_in is None:
            args.filter_in = []
        for file_path in args.extra_datasets_filter_in_file:
            if not os.path.exists(file_path):
                raise ValueError(
                    f'Path for extra dataset filter-in file {file_path} is '
                    f'not a valid path'
                )
            loaded_selection = joblib.load(file_path)
            for _, method_name in loaded_selection.items():
                extra_datasets_filter_in_file.append(method_name)
    main(
        rerun=args.rerun,
        result_dir=results_dir,
        project_name=args.project_name,
        num_workers=args.num_workers,
        global_params=args.param,
        accelerator=(
            "gpu" if (not args.force_cpu) and (torch.cuda.is_available())
            else "cpu"
        ),
        experiment_config=loaded_config,
        filter_out_regex=args.filter_out,
        filter_in_regex=args.filter_in,
        model_selection_metrics=model_selection_metrics,
        model_selection_groups=model_selection_groups,
        summary_table_metrics=summary_table_metrics,
        fast_run=args.fast_run,
        no_new_runs=args.no_new_runs,
        use_auc=args.use_auc,
        use_dataset_cache=args.use_dataset_cache,
        extra_datasets_filter_in_file=extra_datasets_filter_in_file,
        reverse_experiments=args.reverse,
    )
