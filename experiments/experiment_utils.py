import copy
import itertools
import logging
import numpy as np
import os
import torch

from collections import defaultdict
from pathlib import Path
from prettytable import PrettyTable

################################################################################
## HELPER FUNCTIONS
################################################################################

def determine_rerun(
    config,
    rerun,
    full_run_name,
    split,
):
    if rerun:
        return True
    reruns = config.get('reruns', [])
    if "RERUNS" in os.environ:
        reruns += os.environ['RERUNS'].split(",")
    for variant in [
        full_run_name,
        full_run_name + f"_split_{split}",
        full_run_name + f"_fold_{split}",
    ]:
        if variant in reruns:
            return True
    return False

def get_mnist_extractor_arch(input_shape, num_operands):
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
            torch.nn.Linear(
                int(np.prod(input_shape[2:]))*intermediate_maps,
                output_dim,
            ),
        ])
    return c_extractor_arch

def print_table(
    results,
    result_dir,
    split=0,
    result_table_fields=None,
    sort_key="model",
    config=None,
):
    config = config or {}
    # Initialise output table
    results_table = PrettyTable()
    field_names = [
        "Method",
        "Task Accuracy",

    ]
    result_table_fields_keys = [
        "test_acc_y",
    ]

    # Add AUC only when it is a binary class
    shared_params = config.get("shared_params", {})
    if shared_params.get("n_tasks", 3) <= 2:
        field_names.append("Task AUC")
        result_table_fields_keys.append("test_auc_y")

    # Now add concept evaluation metrics
    field_names.extend([
        "Concept Accuracy",
        "Concept AUC",
    ])
    result_table_fields_keys.extend([
        "test_acc_c",
        "test_auc_c",
    ])

    # CAS, if we chose to compute it (off by default as it may be
    # computationally expensive)
    if (
        (not shared_params.get("skip_repr_evaluation", False)) and
        shared_params.get("run_cas", True)
    ):
        field_names.append("CAS")
        result_table_fields_keys.append("test_cas")

    # And intervention summaries if we chose to also include them
    if len(shared_params.get("intervention_policies", [])) > 0:
        field_names.extend([
            "25% Int Acc",
            "50% Int Acc",
            "75% Int Acc",
            "100% Int Acc",
        ])
        result_table_fields_keys.extend([
            "test_acc_y_group_random_no_prior_ints_25%",
            "test_acc_y_group_random_no_prior_ints_50%",
            "test_acc_y_group_random_no_prior_ints_75%",
            "test_acc_y_group_random_no_prior_ints_100%",
        ])

    if result_table_fields is not None:
        for field in result_table_fields:
            if not isinstance(field, (tuple, list)):
                field = field, field
            field_name, field_pretty_name = field
            result_table_fields_keys.append(field_name)
            field_names.append(field_pretty_name)
    results_table.field_names = field_names
    table_rows_inds = {
        name: i for (i, name) in enumerate(result_table_fields_keys)
    }
    table_rows = {}
    end_results = defaultdict(lambda: defaultdict(list))
    for fold_idx, metric_keys in results.items():
        for metric_name, vals in metric_keys.items():
            for desired_metric in result_table_fields_keys:
                real_name = desired_metric
                if desired_metric.startswith("test_acc_y_") and (
                    ("_ints_" in desired_metric) and
                    (desired_metric[-1] == "%")
                ):
                    # Then we are dealing with some interventions we wish
                    # to log
                    percent = int(
                        desired_metric[desired_metric.rfind("_") + 1 : -1]
                    )
                    desired_metric = desired_metric[:desired_metric.rfind("_")]
                else:
                    percent = None

                if metric_name.startswith(desired_metric + "_") and (
                    metric_name[
                        len(desired_metric) + 1 : len(desired_metric) + 2
                    ].isupper()
                ):
                    method_name = metric_name[len(desired_metric) + 1:]
                    if percent is None:
                        end_results[real_name][method_name].append(vals)
                    else:
                        end_results[real_name][method_name].append(
                            vals[int((len(vals) - 1) * percent/100)]
                        )

    for metric_name, runs in end_results.items():
        for method_name, trial_results in runs.items():
            if method_name not in table_rows:
                table_rows[method_name] = [
                    (None, None) for _ in result_table_fields_keys
                ]
            try:
                (mean, std) = np.mean(trial_results), np.std(trial_results)
                if metric_name in table_rows_inds:
                    table_rows[method_name][table_rows_inds[metric_name]] = \
                        (mean, std)
            except:
                logging.warning(
                    f"\tWe could not average results "
                    f"for {metric_name} in model {method_name}"
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
                if x[1][table_rows_inds[sort_key]][0] is not None
                else -float("inf")
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
    if result_dir:
        with open(
            os.path.join(result_dir, f"output_table_fold_{split + 1}.txt"),
            "w",
        ) as f:
            f.write(str(results_table))

def filter_results(results, full_run_name, cut=False):
    output = {}
    for key, val in results.items():
        if full_run_name not in key:
            continue
        if cut:
            key = key[: -len("_" + full_run_name)]
        output[key] = val
    return output

def evaluate_expressions(config):
    for key, val in config.items():
        if isinstance(val, (str,)) and len(val) >= 4 and (
            val[0:2] == "{{" and val[-2:] == "}}"
        ):
            # Then do a simple substitution here
            config[key] = val[2:-2].format(**config)
            config[key] = eval(config[key])


def initialize_result_directory(results_dir):
    Path(
        os.path.join(
            results_dir,
            "models",
        )
    ).mkdir(parents=True, exist_ok=True)

    Path(
        os.path.join(
            results_dir,
            "history",
        )
    ).mkdir(parents=True, exist_ok=True)


def generate_hyperatemer_configs(config):
    if "grid_variables" not in config:
        # Then nothing to see here so we will return
        # a singleton set with this config in it
        return [config]
    # Else time to do some hyperparameter search in here!
    vars = config["grid_variables"]
    options = []
    for var in vars:
        if var not in config:
            raise ValueError(
                f'All variable names in "grid_variables" must be exhisting '
                f'fields in the config. However, we could not find any field '
                f'with name "{var}".'
            )
        if not isinstance(config[var], list):
            raise ValueError(
                f'If we are doing a hyperparamter search over variable '
                f'"{var}", we expect it to be a list of values. Instead '
                f'we got {config[var]}.'
            )
        options.append(config[var])
    mode = config.get('grid_search_mode', "exhaustive").lower().strip()
    if mode in ["grid", "exhaustive"]:
        iterator = itertools.product(*options)
    elif mode in ["paired"]:
        iterator = zip(*options)
    else:
        raise ValueError(
            f'The only supported values for grid_search_mode '
            f'are "paired" and "exhaustive". We got {mode} '
            f'instead.'
        )
    result = []
    for specific_vals in iterator:
        current = copy.deepcopy(config)
        for var_name, new_val in zip(vars, specific_vals):
            current[var_name] = new_val
        result.append(current)
    return result