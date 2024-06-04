"""
$ python run_experiments.py --help
usage: run_experiments.py [-h] [--config config.yaml] [--project_name name]
                          [--output_dir path] [--rerun] [--num_workers N] [-d]
                          [--force_cpu] [-p param_name=value param_name=value]
                          [--activation_freq N] [--filter_out regex]
                          [--filter_in regex] [--model_selection_metrics metric_name]
                          [--summary_table_metrics metric_name pretty_name metric_name pretty_name]
                          [-m group_pattern_regex group_name group_pattern_regex group_name]
                          [--single_frequency_epochs N] [--fast_run]

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
  -p param_name=value param_name=value, --param param_name=value param_name=value
                        Allows the passing of a config param that will overwrite
                        anything passed as part of the config file itself.
  --activation_freq N   how frequently, in terms of epochs, should we store the
                        embedding activations for our validation set. By default
                        we will not store any activations.
  --filter_out regex    skips runs whose names match the regexes provided via
                        this argument. These regexes must follow Python's regex
                        syntax.
  --filter_in regex     includes only runs whose names match the regexes provided
                        with this argument. These regexes must follow Python's
                        regex syntax.
  --model_selection_metrics metric_name
                        metrics to be used to make a summary table by selecting
                        models based on some (validation) metric. If provided,
                        the one must also provide groups via the
                        model_selection_groups argument.
  --summary_table_metrics metric_name pretty_name metric_name pretty_name
                        List of metrics to be included as part of the final
                        summary table of this run.
  -m group_pattern_regex group_name group_pattern_regex group_name, --model_selection_groups group_pattern_regex group_name group_pattern_regex group_name
                        Performs model selection based on the requested model
                        selection metrics by grouping methods that match the
                        Python regex `group_pattern_regex` into a single group
                        with name `group_name`.
  --single_frequency_epochs N
                        how frequently, in terms of epochs, should we store the
                        embedding activations for our validation set. By default
                        we will not store any activations.
  --fast_run            does not perform a check on expected result keys on
                        previously found results. Only use if you are certain
                        old results are not stale and are complete!
"""
import argparse
import copy
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

from cem.data.synthetic_loaders import (
    get_synthetic_data_loader, get_synthetic_num_features
)
import cem.data.celeba_loader as celeba_data_module
import cem.data.color_mnist_add as color_mnist_data_module
import cem.data.CUB200.cub_loader as cub_data_module
import cem.data.mnist_add as mnist_data_module
import cem.interventions.utils as intervention_utils
import cem.train.evaluate as evaluation
import cem.train.training as training
import cem.train.utils as utils

import evaluate_models
import experiment_utils

################################################################################
## HELPER FUNCTIONS
################################################################################


def _debug_ds(train_dl, n_tasks, n_concepts):

    real_sample = []
    for x in sample:
        if isinstance(x, list):
            real_sample += x
        else:
            real_sample.append(x)
    sample = real_sample
    logging.info(
        f"Training sample shape is: {sample[0].shape} with "
        f"type {sample[0].type()}"
    )
    logging.info(
        f"Training label shape is: {sample[1].shape} with "
        f"type {sample[1].type()}"
    )
    logging.info(
        f"\tNumber of output classes: {n_tasks}"
    )
    logging.info(
        f"Training concept shape is: {sample[2].shape} with "
        f"type {sample[2].type()}"
    )
    logging.info(
        f"\tNumber of training concepts: {n_concepts}"
    )

def _update_config_with_dataset(
    config,
    train_dl,
    n_concepts,
    n_tasks,
    concept_map,
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
    experiment_config
):
    if experiment_config.get("dataset_config", None) is None:
        raise ValueError(
            "A dataset_config must be provided for each experiment run!"
        )

    dataset_config = experiment_config['dataset_config']
    logging.debug(
        f"The dataset's root directory is {dataset_config.get('root_dir')}"
    )
    intervention_config = experiment_config.get('intervention_config', {})
    if dataset_config["dataset"] == "cub":
        data_module = cub_data_module
    elif dataset_config["dataset"] == "celeba":
        data_module = celeba_data_module
    elif dataset_config["dataset"] in ["xor", "vector", "dot", "trig"]:
        data_module = get_synthetic_data_loader(dataset_config["dataset"])
    elif dataset_config["dataset"] == "mnist_add":
        data_module = mnist_data_module
    elif dataset_config["dataset"] == "color_mnist_add":
        data_module = color_mnist_data_module
    else:
        raise ValueError(f"Unsupported dataset {dataset_config['dataset']}!")

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
        experiment_config["c_extractor_arch"] = \
            experiment_utils.get_mnist_extractor_arch(
                input_shape=(
                    dataset_config.get('batch_size', 512),
                    3,
                    28*num_operands,
                    28,
                ),
                in_channels=3,
            )
    elif experiment_config['c_extractor_arch'] == 'synth_extractor':
        input_features = get_synthetic_num_features(dataset_config["dataset"])
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

    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = \
        data_module.generate_data(
            config=dataset_config,
            seed=42,
            output_dataset_vars=True,
            root_dir=dataset_config.get('root_dir', None),
        )
    # For now, we assume that all concepts have the same
    # aquisition cost
    acquisition_costs = None
    if concept_map is not None:
        intervened_groups = list(
            range(
                0,
                len(concept_map) + 1,
                intervention_config.get('intervention_freq', 1),
            )
        )
    else:
        intervened_groups = list(
            range(
                0,
                n_concepts + 1,
                intervention_config.get('intervention_freq', 1),
            )
        )

    task_class_weights = _update_config_with_dataset(
        config=experiment_config,
        train_dl=train_dl,
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        concept_map=concept_map,
    )
    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        concept_map,
        intervened_groups,
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
):
    ############################################################################
    ## Automatic Model Selection
    ############################################################################
    prev_selected_results = None
    if (model_selection_groups is not None) and (
        model_selection_metrics is not None
    ):
        prev_selected_results = []
        for model_selection_metric in model_selection_metrics:
            model_selection_results, selection_map = \
                experiment_utils.perform_model_selection(
                    results=results,
                    selection_metric=model_selection_metric,
                    model_groupings=model_selection_groups,
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
                (model_selection_results, model_selection_metric)
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
    single_frequency_epochs,
    activation_freq,
):
    config = copy.deepcopy(run_config)
    (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        concept_map,
        intervened_groups,
        task_class_weights,
        acquisition_costs
    ) = _generate_dataset_and_update_config(
        config,
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
        train_fn = training.train_prob_cbm
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
        config=config,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
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
        single_frequency_epochs=single_frequency_epochs,
        activation_freq=activation_freq,
    )
    training.update_statistics(
        aggregate_results=trial_results,
        run_config=config,
        model=model,
        test_results=model_results,
        run_name=run_name,
        prefix="",
    )
    test_datasets = [
        (val_dl, "val"),
        (test_dl, "test"),
    ]
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
        intervened_groups=intervened_groups,
        accelerator=accelerator,
        devices=devices,
        split=split,
        rerun=current_rerun,
        old_results=old_results,
    )
    training.update_statistics(
        aggregate_results=trial_results,
        run_config=config,
        model=model,
        test_results=eval_results,
        run_name=run_name,
        prefix="",
    )

    eval_config = config.get('eval_config', {})
    for new_test_dl_configs in eval_config.get('additional_test_sets'):
        config_copy = copy.deepcopy(config)
        config_copy['dataset_config'] = new_test_dl_configs['dataset_config']
        (
            _,
            _,
            new_test_dl,
            new_imbalance,
            new_concept_map,
            new_intervened_groups,
            new_task_class_weights,
            new_acquisition_costs
        ) = _generate_dataset_and_update_config(
            config_copy,
        )
        eval_results = evaluate_models.evaluate_model(
            model,
            config_copy,
            test_datasets,
            train_dl,
            val_dl=val_dl,
            run_name=run_name,
            task_class_weights=new_task_class_weights,
            imbalance=new_imbalance,
            acquisition_costs=new_acquisition_costs,
            result_dir=result_dir,
            concept_map=new_concept_map,
            intervened_groups=new_intervened_groups,
            accelerator=accelerator,
            devices=devices,
            split=split,
            rerun=current_rerun,
            old_results=old_results,
        )
        training.update_statistics(
            aggregate_results=trial_results,
            run_config=config,
            model=model,
            test_results=eval_results,
            run_name=run_name,
            prefix=(new_test_dl_configs.name + "_"),
        )


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
    single_frequency_epochs=0,
    activation_freq=0,
    filter_out_regex=None,
    filter_in_regex=None,
    model_selection_metrics=None,
    model_selection_groups=None,
    fast_run=False,
    no_new_runs=False,
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
        for current_config in experiment_config['runs']:
            # Construct the config for this particular trial
            trial_config = copy.deepcopy(shared_params)
            trial_config.update(current_config)
            # Time to try as many seeds as requested
            for run_config in experiment_utils.generate_hyperparameter_configs(trial_config):
                run_config = copy.deepcopy(run_config)
                run_config['result_dir'] = result_dir
                experiment_utils.evaluate_expressions(run_config, soft=True)
                run_config['split'] = split

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
                if filter_out_regex is not None:
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
                if filter_in_regex is not None:
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

                # Determine training rerun or not
                current_results_path = os.path.join(
                    result_dir,
                    f'{run_name}_split_{split}_results.joblib'
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
                    with open(current_results_path, 'rb') as f:
                        old_results = joblib.load(f)
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
                            single_frequency_epochs=single_frequency_epochs,
                            activation_freq=activation_freq,
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
                        single_frequency_epochs=single_frequency_epochs,
                        activation_freq=activation_freq,
                    )
                training.update_statistics(
                    aggregate_results=results[f'{split}'][run_name],
                    run_config=run_config,
                    test_results=trial_results,
                    run_name=run_name,
                    prefix="",
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


            # Save this run's results
            if run_config.get("start_split", 0) == 0:
                attempt = 0
                # We will try and dump things a few times in case there
                # are other threads/processes currently modifying or
                # writing this same file
                while attempt < 5:
                    try:
                        with open(
                            os.path.join(result_dir, f'results.joblib'),
                            'wb',
                        ) as f:
                            joblib.dump(results, f)
                        break
                    except Exception as e:
                        print(e)
                        print(
                            "FAILED TO SERIALIZE RESULTS TO",
                            os.path.join(result_dir, f'results.joblib')
                        )
                        attempt += 1
                if attempt == 5:
                    raise ValueError(
                        "Could not serialize " +
                        os.path.join(result_dir, f'results.joblib') +
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
            )
            logging.debug(f"\t\tDone with trial {split + 1}")
    print(f"********** Results after trial {split + 1} **********")
    experiment_utils.print_table(
        config=experiment_config,
        results=results,
        summary_table_metrics=summary_table_metrics,
        sort_key=sort_key,
        result_dir=result_dir,
        split=split,
    )
    logging.debug(f"\t\tDone with trial {split + 1}")
    _perform_model_selection(
        model_selection_groups=model_selection_groups,
        model_selection_metrics=model_selection_metrics,
        results=results,
        result_dir=result_dir,
        split=split,
        summary_table_metrics=summary_table_metrics,
        config=experiment_config,
    )
        # Locally serialize the results of this trial
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
        metavar=('param_name=value'),
        help=(
            'Allows the passing of a config param that will overwrite '
            'anything passed as part of the config file itself.'
        ),
        default=[],
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
    logging.info(f"Results will be dumped in {loaded_config['results_dir']}")
    Path(loaded_config['results_dir']).mkdir(parents=True, exist_ok=True)
    # Write down the actual command executed
    # And the configuration file
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    loaded_config["time_last_called"] = now.strftime("%Y/%m/%d at %H:%M:%S")
    with open(
        os.path.join(loaded_config['results_dir'], f"command_{dt_string}.txt"),
        "w",
    ) as f:
        command_args = [
            arg if " " not in arg else f'"{arg}"' for arg in sys.argv
        ]
        f.write("python " + " ".join(command_args))

    # Also save the current experiment configuration
    with open(
        os.path.join(
            loaded_config['results_dir'],
            f"experiment_{dt_string}_config.yaml")
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

    main(
        rerun=args.rerun,
        result_dir=(
            args.output_dir if args.output_dir
            else loaded_config['results_dir']
        ),
        project_name=args.project_name,
        num_workers=args.num_workers,
        global_params=args.param,
        accelerator=(
            "gpu" if (not args.force_cpu) and (torch.cuda.is_available())
            else "cpu"
        ),
        experiment_config=loaded_config,
        activation_freq=args.activation_freq,
        single_frequency_epochs=args.single_frequency_epochs,
        filter_out_regex=args.filter_out,
        filter_in_regex=args.filter_in,
        model_selection_metrics=model_selection_metrics,
        model_selection_groups=model_selection_groups,
        summary_table_metrics=summary_table_metrics,
        fast_run=args.fast_run,
        no_new_runs=args.no_new_runs,
    )
