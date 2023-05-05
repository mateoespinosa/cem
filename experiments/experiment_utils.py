import itertools
from pathlib import Path
import os
import copy

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
            config[key] = val[1:-1].format(**config)
            # And then try and convert it into its
            # possibly numerical value. We first try
            # ints then floats
            try:
                config[key] = int(config[key])
            except:
                try:
                    config[key] = float(config[key])
                except:
                    pass


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
                f'fields in the config. However, we could not find any field with '
                f'name "{var}".'
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