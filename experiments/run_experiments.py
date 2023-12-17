import argparse
import copy
import joblib
import json
import logging
import numpy as np
import os
import sys
import torch
import yaml


from datetime import datetime
from pathlib import Path
from pytorch_lightning import seed_everything

from cem.data.synthetic_loaders import (
    get_synthetic_data_loader, get_synthetic_num_features
)
import cem.data.celeba_loader as celeba_data_module
import cem.data.chexpert_loader as chexpert_data_module
import cem.data.CUB200.cub_loader as cub_data_module
import cem.data.derm_loader as derm_data_module
import cem.data.mnist_add as mnist_data_module
import cem.interventions.utils as intervention_utils
import cem.train.training as training
import cem.train.utils as utils

from experiment_utils import (
    evaluate_expressions, determine_rerun,
    generate_hyperatemer_configs, filter_results,
    print_table, get_mnist_extractor_arch
)

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
    accelerator="auto",
    devices="auto",
    result_table_fields=None,
    sort_key="Task Accuracy",
    single_frequency_epochs=0,
    activation_freq=0,
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

    utils.extend_with_global_params(
        experiment_config['shared_params'], global_params or []
    )



    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = \
        data_module.generate_data(
            config=experiment_config['shared_params'],
            seed=42,
            output_dataset_vars=True,
            root_dir=experiment_config['shared_params'].get('root_dir', None),
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
    experiment_config["shared_params"]["n_concepts"] = \
        experiment_config["shared_params"].get(
            "n_concepts",
            n_concepts,
        )
    experiment_config["shared_params"]["n_tasks"] = \
        experiment_config["shared_params"].get(
            "n_tasks",
            n_tasks,
        )
    experiment_config["shared_params"]["concept_map"] = \
        experiment_config["shared_params"].get(
            "concept_map",
            concept_map,
        )

    sample = next(iter(train_dl))
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

    task_class_weights = None

    if experiment_config['shared_params'].get('use_task_class_weights', False):
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
                (_, y, _) = data
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
        print(
            f"[TRIAL "
            f"{split + 1}/{experiment_config['shared_params']['trials']} "
            f"BEGINS AT {now.strftime('%d/%m/%Y at %H:%M:%S')}"
        )
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
                current_rerun = determine_rerun(
                    config=run_config,
                    rerun=rerun,
                    split=split,
                    full_run_name=full_run_name,
                )
                if current_rerun:
                    logging.warning(
                        f"We will rerun model {full_run_name}_split_{split} "
                        f"as requested by the config"
                    )
                if (not current_rerun) and os.path.exists(current_results_path):
                    with open(current_results_path, 'rb') as f:
                        old_results = joblib.load(f)

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
                        with open(seq_current_results_path, 'rb') as f:
                            seq_old_results = joblib.load(f)

                    ind_old_results = None
                    ind_current_results_path = os.path.join(
                        result_dir,
                        f'Sequential{full_run_name}_split_{split}_results.joblib'
                    )
                    if os.path.exists(ind_current_results_path):
                        with open(ind_current_results_path, 'rb') as f:
                            ind_old_results = joblib.load(f)
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
                            rerun=current_rerun,
                            project_name=project_name,
                            seed=(42 + split),
                            imbalance=imbalance,
                            ind_old_results=ind_old_results,
                            seq_old_results=seq_old_results,
                            single_frequency_epochs=single_frequency_epochs,
                            activation_freq=activation_freq,
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
                    results[f'{split}'].update(
                        intervention_utils.test_interventions(
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
                            accelerator=accelerator,
                            devices=devices,
                            split=split,
                            rerun=current_rerun,
                            old_results=ind_old_results,
                            independent=True,
                            competence_levels=config.get(
                            'competence_levels',
                            [1],
                        ),
                        )
                    )
                    logging.debug(
                        f"\tResults for {full_run_name} in split {split}:"
                    )
                    for key, val in filter_results(
                        results[f'{split}'],
                        full_run_name,
                        cut=True,
                    ).items():
                        logging.debug(f"\t\t{key} -> {val}")
                    with open(ind_current_results_path, 'wb') as f:
                        joblib.dump(
                            filter_results(results[f'{split}'], full_run_name),
                            f,
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
                    results[f'{split}'].update(
                        intervention_utils.test_interventions(
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
                            accelerator=accelerator,
                            devices=devices,
                            split=split,
                            rerun=current_rerun,
                            old_results=seq_old_results,
                            sequential=True,
                            competence_levels=config.get('competence_levels', [1]),
                        )
                    )
                    logging.debug(
                        f"\tResults for {full_run_name} in split {split}:"
                    )
                    for key, val in filter_results(
                        results[f'{split}'],
                        full_run_name,
                        cut=True,
                    ).items():
                        logging.debug(f"\t\t{key} -> {val}")
                    with open(seq_current_results_path, 'wb') as f:
                        joblib.dump(
                            filter_results(results[f'{split}'], full_run_name),
                            f,
                        )
                    if experiment_config['shared_params'].get("start_split", 0) == 0:
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
                else:
                    model, model_results = \
                        training.train_model(
                            task_class_weights=task_class_weights,
                            accelerator=accelerator,
                            devices=devices,
                            n_concepts=n_concepts,
                            n_tasks=n_tasks,
                            config=run_config,
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
                            gradient_clip_val=run_config.get(
                                'gradient_clip_val',
                                0,
                            ),
                            single_frequency_epochs=single_frequency_epochs,
                            activation_freq=activation_freq,
                        )
                    training.update_statistics(
                        results[f'{split}'],
                        run_config,
                        model,
                        model_results,
                    )
                    results[f'{split}'].update(
                        intervention_utils.test_interventions(
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
                            accelerator=accelerator,
                            devices=devices,
                            split=split,
                            rerun=current_rerun,
                            old_results=old_results,
                            competence_levels=run_config.get(
                                'competence_levels',
                                [1],
                            ),
                        )
                    )
                    results[f'{split}'].update(
                        training.evaluate_representation_metrics(
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
                            accelerator=accelerator,
                            devices=devices,
                            rerun=current_rerun,
                            seed=42,
                            old_results=old_results,
                        )
                    )

                    logging.debug(
                        f"\tResults for {full_run_name} in split {split}:"
                    )
                    for key, val in filter_results(
                        results[f'{split}'],
                        full_run_name,
                        cut=True,
                    ).items():
                        logging.debug(f"\t\t{key} -> {val}")
                    with open(current_results_path, 'wb') as f:
                        joblib.dump(
                            filter_results(results[f'{split}'], full_run_name),
                            f,
                        )
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
                extr_name = run_config['c_extractor_arch']
                if not isinstance(extr_name, str):
                    extr_name = "lambda"
                then = datetime.now()
                diff = then - now
                diff_minutes = diff.total_seconds() / 60
                logging.debug(
                    f"\tTrial {split + 1} COMPLETED for {full_run_name} ending "
                    f"at {then.strftime('%d/%m/%Y at %H:%M:%S')} "
                    f"({diff_minutes:.4f} minutes):"
                )
            print(f"********** Results in between trial {split + 1} **********")
            print_table(
                config=experiment_config,
                results=results,
                result_table_fields=result_table_fields,
                sort_key=sort_key,
                result_dir=None,
                split=split,
            )
            logging.debug(f"\t\tDone with trial {split + 1}")
    print(f"********** Results after trial {split + 1} **********")
    print_table(
        config=experiment_config,
        results=results,
        result_table_fields=result_table_fields,
        sort_key=sort_key,
        result_dir=result_dir,
        split=split,
    )
    logging.debug(f"\t\tDone with trial {split + 1}")
        # Locally serialize the results of this trial
    return results


################################################################################
## Arg Parser
################################################################################


def _build_arg_parser():
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
        choices=[
            'cub',
            'celeba',
            'xor',
            'vector',
            'dot',
            'trig',
            'mnist_add',
            'chexpert',
            'derma',
        ],
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
    elif loaded_config["dataset"] == "derm":
        data_module = derm_data_module
        args.project_name = args.project_name.format(ds_name="derma")
    elif loaded_config["dataset"] == "celeba":
        data_module = celeba_data_module
        args.project_name = args.project_name.format(ds_name="celeba")
    elif loaded_config["dataset"] == "chexpert":
        data_module = chexpert_data_module
        args.project_name = args.project_name.format(ds_name="chexpert")
    elif loaded_config["dataset"] in ["xor", "vector", "dot", "trig"]:
        data_module = get_synthetic_data_loader(loaded_config["dataset"])
        args.project_name = args.project_name.format(
            ds_name=loaded_config["dataset"]
        )
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
    elif loaded_config["dataset"] == "mnist_add":
        data_module = mnist_data_module
        args.project_name = args.project_name.format(ds_name=args.dataset)
        utils.extend_with_global_params(
            loaded_config,
            args.param or []
        )
        num_operands = loaded_config.get('num_operands', 32)
        loaded_config["c_extractor_arch"] = get_mnist_extractor_arch(
            input_shape=(
                loaded_config.get('batch_size', 512),
                num_operands,
                28,
                28,
            ),
            num_operands=num_operands,
        )
    else:
        raise ValueError(f"Unsupported dataset {loaded_config['dataset']}!")

    if args.output_dir is not None:
        loaded_config['results_dir'] = args.output_dir
    if args.debug:
        def serialize_function(obj):
            """Custom serialization for function objects."""
            if callable(obj):
                return f"Function: {obj.__name__}"
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
        print(json.dumps(loaded_config, sort_keys=True, indent=4, default=serialize_function))
    logging.info(f"Results will be dumped in {loaded_config['results_dir']}")
    logging.debug(
        f"And the dataset's root directory is {loaded_config.get('root_dir')}"
    )
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

    main(
        data_module=data_module,
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
    )
