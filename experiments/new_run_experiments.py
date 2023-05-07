import argparse
import copy
import joblib
import numpy as np
import os

import logging
import torch
from pytorch_lightning import seed_everything
import json
from prettytable import PrettyTable
from collections import defaultdict

import cem.data.CUB200.cub_loader as cub_data_module
import cem.data.mnist_add as mnist_data_module
import cem.data.celeba_loader as celeba_data_module
import cem.data.chexpert_loader as chexpert_data_module
from cem.data.synthetic_loaders import (
    get_synthetic_data_loader,
    get_synthetic_num_features,
)
import yaml
import sys
from pathlib import Path
from datetime import datetime

import cem.train.training as training
import cem.train.utils as utils
import cem.interventions.utils as intervention_utils
from run_experiments import CUB_CONFIG, CELEBA_CONFIG, SYNTH_CONFIG
from experiment_utils import (
    evaluate_expressions,
    generate_hyperatemer_configs, filter_results,
)

################################################################################
## HELPER FUNCTIONS
################################################################################

def _get_mnist_extractor_arch(input_shape, num_operands):
    def c_extractor_arch(output_dim):
        intermediate_maps = 16
        output_dim = output_dim or 128
        return torch.nn.Sequential(*[
            torch.nn.Conv2d(
                in_channels=num_operands,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=intermediate_maps,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=intermediate_maps,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=intermediate_maps,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(int(np.prod(input_shape[2:]))*intermediate_maps, output_dim),
        ])
    return c_extractor_arch

def _print_table(results, result_dir, split=0, result_table_fields=None, sort_key="model"):
    # Initialise output table
    results_table = PrettyTable()
    field_names = [
        "Method",
        "Task Accuracy",
        "Task AUC",
        "Concept Accuracy",
        "Concept AUC",
        "CAS",
        "NIS",
        "OIS",
    ]
    result_table_fields_keys = [
        "test_acc_y",
        "test_auc_y",
        "test_acc_c",
        "test_auc_c",
        "test_cas",
        "test_nis",
        "test_ois"
    ]
    if result_table_fields is not None:
        for field in result_table_fields:
            if not isinstance(field, (tuple, list)):
                field = field, field
            field_name, field_pretty_name = field
            result_table_fields_keys.append(field_name)
            field_names.append(field_pretty_name)
    results_table.field_names = field_names
    table_rows_inds = {name: i for (i, name) in enumerate(result_table_fields_keys)}
    table_rows = {}
    end_results = defaultdict(lambda: defaultdict(list))
    for fold_idx, metric_keys in results.items():
        for metric_name, vals in metric_keys.items():
            for desired_metric in result_table_fields_keys:
                if metric_name.startswith(desired_metric + "_") and (
                    metric_name[len(desired_metric) + 1 : len(desired_metric) + 2].isupper()
                ):
                    method_name = metric_name[len(desired_metric) + 1:]
                    end_results[desired_metric][method_name].append(vals)
        vals = np.array(vals)
    for metric_name, runs in end_results.items():
        for method_name, trial_results in runs.items():
            if method_name not in table_rows:
                table_rows[method_name] = [(None, None) for _ in result_table_fields_keys]
            try:
                (mean, std) = np.mean(trial_results), np.std(trial_results)
                if metric_name in table_rows_inds:
                    table_rows[method_name][table_rows_inds[metric_name]] = (mean, std)
            except:
                logging.warning(
                    f"\tWe could not average results for {metric_name} in model {method_name}"
                )
    table_rows = list(table_rows.items())
    if sort_key == "model":
        # Then sort based on method name
        table_rows.sort(key=lambda x: x[0], reverse=True)
    elif sort_key in table_rows_inds:
        # Else sort based on the requested parameter
        table_rows.sort(
            key=lambda x: (
                x[1][table_rows_inds[sort_key]][0]
                if x[1][table_rows_inds[sort_key]][0] is not None else -float("inf")
            ),
            reverse=True,
    )
    for aggr_key, row in table_rows:
        for i, (mean, std) in enumerate(row):
            if mean is None or std is None:
                row[i] = "N/A"
            elif int(mean) == float(mean):
                row[i] = f'{mean} ± {std:}'
            else:
                row[i] = f'{mean:.4f} ± {std:.4f}'
        results_table.add_row([str(aggr_key)] + row)
    print("\t", "*" * 30)
    print(results_table)
    print("\n\n")

    # Also serialize the results
    with open(os.path.join(result_dir, f"output_table_fold_{split + 1}.txt"), "w") as f:
        f.write(str(results_table))
################################################################################
## MAIN FUNCTION
################################################################################


def main(
    data_module,
    result_dir,
    experiment_config,
    rerun=False,
    project_name='',
    num_workers=8,
    global_params=None,
    gpu=torch.cuda.is_available(),
    result_table_fields=None,
    sort_key="Task Accuracy",
):
    seed_everything(42)
    # parameters for data, model, and training
    experiment_config = copy.deepcopy(experiment_config)
    if 'shared_params' not in experiment_config:
        experiment_config['shared_params'] = {}
    # Move all global things into the shared params
    for key, vals in experiment_config.items():
        if key not in ['runs', 'shared_params']:
            experiment_config['shared_params'][key] = vals
    experiment_config['shared_params']['num_workers'] = num_workers

    gpu = 1 if gpu else 0
    utils.extend_with_global_params(experiment_config['shared_params'], global_params or [])



    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = \
        data_module.generate_data(
            config=experiment_config['shared_params'],
            seed=42,
            output_dataset_vars=True,
            root_dir=experiment_config.get('root_dir', None),
        )
    # For now, we assume that all concepts have the same
    # aquisition cost
    acquisition_costs = None
    if concept_map is not None:
        intervened_groups = list(
            range(
                0,
                len(concept_map) + 1,
                experiment_config['shared_params'].get('intervention_freq', 1),
            )
        )
    else:
        intervened_groups = list(
            range(
                0,
                n_concepts + 1,
                experiment_config['shared_params'].get('intervention_freq', 1),
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
    logging.info(f"Training sample shape is: {sample[0].shape} with type {sample[0].type()}")
    logging.info(f"Training label shape is: {sample[1].shape} with type {sample[1].type()}")
    logging.info(f"\tNumber of output classes: {n_tasks}")
    logging.info(f"Training concept shape is: {sample[2].shape} with type {sample[2].type()}")
    logging.info(f"\tNumber of training concepts: {n_concepts}")

    task_class_weights = None

    if experiment_config['shared_params'].get('use_task_class_weights', False):
        attribute_count = np.zeros((max(n_tasks, 2),))
        samples_seen = 0
        for i, data in enumerate(train_dl):
            if len(data) == 2:
                (_, (y, _)) = data
            else:
                (_, y, _) = data
            if n_tasks > 1:
                y = torch.nn.functional.one_hot(y, num_classes=n_tasks).cpu().detach().numpy()
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
            task_class_weights = np.array([attribute_count[0]/attribute_count[1]])


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
        experiment_config['shared_params'].get("start_split", 0),
        experiment_config['shared_params']["trials"],
    ):
        results[f'{split}'] = {}
        now = datetime.now()
        print(f"[TRIAL {split + 1}/{experiment_config['shared_params']['trials']} BEGINS AT {now.strftime('%d/%m/%Y at %H:%M:%S')}")
        # And then over all runs in a given trial
        for current_config in experiment_config['runs']:
            # Construct the config for this particular trial
            trial_config = copy.deepcopy(experiment_config.get('shared_params', {}))

            trial_config.update(current_config)
            trial_config["concept_map"] = concept_map
            # Now time to iterate 5
            # over all hyperparameters that were given as part
            for run_config in generate_hyperatemer_configs(trial_config):
                now = datetime.now()
                run_config = copy.deepcopy(run_config)
                evaluate_expressions(run_config)
                run_config["extra_name"] = run_config.get("extra_name", "").format(
                    **run_config
                )
                old_results = None
                full_run_name = (
                    f"{run_config['architecture']}{run_config.get('extra_name', '')}"
                )
                current_results_path = os.path.join(
                    result_dir,
                    f'{full_run_name}_split_{split}_results.joblib'
                )
                if os.path.exists(current_results_path):
                    old_results = joblib.load(current_results_path)

                if run_config["architecture"] in [
                    "IndependentConceptBottleneckModel",
                    "SequentialConceptBottleneckModel",
                ]:
                    # Special case for now for sequential and independent CBMs
                    config = copy.deepcopy(run_config)
                    config["architecture"] = "ConceptBottleneckModel"
                    config["sigmoidal_prob"] = True
                    full_run_name = (
                        f"{config['architecture']}{config.get('extra_name', '')}"
                    )
                    seq_old_results = None
                    seq_current_results_path = os.path.join(
                        result_dir,
                        f'Sequential{full_run_name}_split_{split}_results.joblib'
                    )
                    if os.path.exists(seq_current_results_path):
                        seq_old_results = joblib.load(seq_current_results_path)

                    ind_old_results = None
                    ind_current_results_path = os.path.join(
                        result_dir,
                        f'Sequential{full_run_name}_split_{split}_results.joblib'
                    )
                    if os.path.exists(ind_current_results_path):
                        ind_old_results = joblib.load(ind_current_results_path)
                    ind_model, ind_test_results, seq_model, seq_test_results = \
                        training.train_independent_and_sequential_model(
                            task_class_weights=task_class_weights,
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
                            seed=(42 + split),
                            imbalance=imbalance,
                            ind_old_results=ind_old_results,
                            seq_old_results=seq_old_results,
                        )
                    config["architecture"] = "IndependentConceptBottleneckModel"
                    training.update_statistics(
                        results[f'{split}'],
                        config,
                        ind_model,
                        ind_test_results,
                    )
                    full_run_name = (
                        f"{config['architecture']}{config.get('extra_name', '')}"
                    )
                    results[f'{split}'].update(intervention_utils.test_interventions(
                        task_class_weights=task_class_weights,
                        full_run_name=full_run_name,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        test_dl=test_dl,
                        imbalance=imbalance,
                        config=config,
                        n_tasks=n_tasks,
                        n_concepts=n_concepts,
                        acquisition_costs=acquisition_costs,
                        result_dir=result_dir,
                        concept_map=concept_map,
                        intervened_groups=intervened_groups,
                        gpu=gpu,
                        split=split,
                        rerun=rerun,
                        old_results=ind_old_results,
                        independent=True,
                    ))
                    logging.debug(f"\tResults for {full_run_name} in split {split}:")
                    for key, val in filter_results(results[f'{split}'], full_run_name, cut=True).items():
                        logging.debug(f"\t\t{key} -> {val}")
                    joblib.dump(
                        filter_results(results[f'{split}'], full_run_name),
                        ind_current_results_path,
                    )

                    config["architecture"] = "SequentialConceptBottleneckModel"
                    training.update_statistics(
                        results[f'{split}'],
                        config,
                        seq_model,
                        seq_test_results,
                    )
                    full_run_name = (
                        f"{config['architecture']}{config.get('extra_name', '')}"
                    )
                    results[f'{split}'].update(intervention_utils.test_interventions(
                        task_class_weights=task_class_weights,
                        full_run_name=full_run_name,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        test_dl=test_dl,
                        imbalance=imbalance,
                        config=config,
                        n_tasks=n_tasks,
                        n_concepts=n_concepts,
                        acquisition_costs=acquisition_costs,
                        result_dir=result_dir,
                        concept_map=concept_map,
                        intervened_groups=intervened_groups,
                        gpu=gpu,
                        split=split,
                        rerun=rerun,
                        old_results=seq_old_results,
                        sequential=True,
                    ))
                    logging.debug(f"\tResults for {full_run_name} in split {split}:")
                    for key, val in filter_results(results[f'{split}'], full_run_name, cut=True).items():
                        logging.debug(f"\t\t{key} -> {val}")
                    joblib.dump(
                        filter_results(results[f'{split}'], full_run_name),
                        seq_current_results_path,
                    )
                    if experiment_config['shared_params'].get("start_split", 0) == 0:
                        joblib.dump(results, os.path.join(result_dir, f'results.joblib'))
                else:
                    model,  model_results = \
                        training.train_model(
                            task_class_weights=task_class_weights,
                            gpu=gpu if gpu else 0,
                            n_concepts=n_concepts,
                            n_tasks=n_tasks,
                            config=run_config,
                            train_dl=train_dl,
                            val_dl=val_dl,
                            test_dl=test_dl,
                            split=split,
                            result_dir=result_dir,
                            rerun=rerun,
                            project_name=project_name,
                            seed=(42 + split),
                            imbalance=imbalance,
                            old_results=old_results,
                            gradient_clip_val=run_config.get('gradient_clip_val', 0),
                        )
                    training.update_statistics(
                        results[f'{split}'],
                        run_config,
                        model,
                        model_results,
                    )
                    results[f'{split}'].update(intervention_utils.test_interventions(
                        task_class_weights=task_class_weights,
                        full_run_name=full_run_name,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        test_dl=test_dl,
                        imbalance=imbalance,
                        config=run_config,
                        n_tasks=n_tasks,
                        n_concepts=n_concepts,
                        acquisition_costs=acquisition_costs,
                        result_dir=result_dir,
                        concept_map=concept_map,
                        intervened_groups=intervened_groups,
                        gpu=gpu,
                        split=split,
                        rerun=rerun,
                        old_results=old_results,
                    ))
                    results[f'{split}'].update(training.evaluate_representation_metrics(
                        config=run_config,
                        n_concepts=n_concepts,
                        n_tasks=n_tasks,
                        test_dl=test_dl,
                        full_run_name=full_run_name,
                        split=split,
                        imbalance=imbalance,
                        result_dir=result_dir,
                        sequential=False,
                        independent=False,
                        task_class_weights=task_class_weights,
                        gpu=gpu,
                        rerun=rerun,
                        seed=42,
                        old_results=old_results,
                    ))

                    logging.debug(f"\tResults for {full_run_name} in split {split}:")
                    for key, val in filter_results(
                        results[f'{split}'],
                        full_run_name,
                        cut=True,
                    ).items():
                        logging.debug(f"\t\t{key} -> {val}")
                    joblib.dump(
                        filter_results(results[f'{split}'], full_run_name),
                        current_results_path,
                    )
                if run_config.get("start_split", 0) == 0:
                    joblib.dump(results, os.path.join(result_dir, f'results.joblib'))
                extr_name = run_config['c_extractor_arch']
                if not isinstance(extr_name, str):
                    extr_name = "lambda"
                then = datetime.now()
                diff = then - now
                diff_minutes = diff.total_seconds() / 60
                logging.debug(
                    f"\tTrial {split + 1} COMPLETED for {full_run_name} ending at "
                    f"{then.strftime('%d/%m/%Y at %H:%M:%S')} ({diff_minutes:.4f} "
                    f"minutes):"
                )
            print(f"************ Results after trial {split + 1} ************")
            _print_table(
                results=results,
                result_table_fields=result_table_fields,
                sort_key=sort_key,
                result_dir=result_dir,
                split=split,
            )
            logging.debug(f"\t\tDone with trial {split + 1}")
    print(f"************ Results after trial {split + 1} ************")
    _print_table(
        results=results,
        result_table_fields=result_table_fields,
        sort_key=sort_key,
        result_dir=result_dir,
        split=split,
    )
    logging.debug(f"\t\tDone with trial {split + 1}")
        # Locally serialize the results of this trial
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Runs CEM intervention experiments in a given dataset.'
        ),
    )
    parser.add_argument(
        '--config',
        '-c',
        default=None,
        help=(
            "YAML file with the configuration for the experiment. If not "
            "provided, then we will use the default configuration for the "
            "dataset."
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
        '--dataset',
        choices=['cub', 'celeba', 'xor', 'vector', 'dot', 'trig', 'mnist_add', 'chexpert'],
        help=(
            "Dataset to run experiments for. Must be a supported dataset with "
            "a loader."
        ),
        metavar="ds_name",
        default=None,
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
    args = parser.parse_args()

    if args.project_name:
        # Lazy import to avoid importing unless necessary
        import wandb
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

    if args.dataset is not None:
        loaded_config["dataset"] = args.dataset
    if loaded_config.get("dataset", None) is None:
        raise ValueError(
            "A dataset must be provided either as part of the "
            "configuration file or as a command line argument."
        )
    if loaded_config["dataset"] == "cub":
        data_module = cub_data_module
        args.project_name = args.project_name.format(ds_name="cub")
    elif loaded_config["dataset"] == "celeba":
        data_module = celeba_data_module
        args.project_name = args.project_name.format(ds_name="celeba")
    elif loaded_config["dataset"] == "chexpert":
        data_module = chexpert_data_module
        args.project_name = args.project_name.format(ds_name="chexpert")
    elif loaded_config["dataset"] in ["xor", "vector", "dot", "trig"]:
        data_module = get_synthetic_data_loader(loaded_config["dataset"])
        args.project_name = args.project_name.format(ds_name=loaded_config["dataset"])
        input_features = get_synthetic_num_features(loaded_config["dataset"])
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
        loaded_config["c_extractor_arch"] = synth_c_extractor_arch
    elif args.dataset == "mnist_add":
        data_module = mnist_data_module
        args.project_name = args.project_name.format(ds_name=args.dataset)
        utils.extend_with_global_params(
            loaded_config,
            args.param or []
        )
        num_operands = loaded_config.get('num_operands', 32)
        loaded_config["c_extractor_arch"] = _get_mnist_extractor_arch(
            input_shape=(loaded_config.get('batch_size', 512), num_operands, 28, 28),
            num_operands=num_operands,
        )
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}!")

    if args.output_dir is not None:
        loaded_config['results_dir'] = args.output_dir
    if args.debug:
        print(json.dumps(loaded_config, sort_keys=True, indent=4))
    logging.info(f"Results will be dumped in {loaded_config['results_dir']}")
    logging.debug(f"And the dataset's root directory is {loaded_config.get('root_dir')}")
    Path(loaded_config['results_dir']).mkdir(parents=True, exist_ok=True)
    # Write down the actual command executed
    # And the configuration file
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    loaded_config["time_last_called"] = now.strftime("%Y/%m/%d at %H:%M:%S")
    with open(os.path.join(loaded_config['results_dir'], f"command_{dt_string}.txt"), "w") as f:
        command_args = [arg if " " not in arg else f'"{arg}"' for arg in sys.argv]
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

    main(
        data_module=data_module,
        rerun=args.rerun,
        result_dir=args.output_dir,
        project_name=args.project_name,
        num_workers=args.num_workers,
        global_params=args.param,
        gpu=(not args.force_cpu) and (torch.cuda.is_available()),
        experiment_config=loaded_config,
    )
