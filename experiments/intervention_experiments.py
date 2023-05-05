import argparse
import copy
import joblib
import numpy as np
import os

import logging
import torch
import scipy.special
from pytorch_lightning import seed_everything

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

import cem.train.training as training
import cem.train.utils as utils
import cem.interventions.utils as intervention_utils
from cem.interventions.random import IndependentRandomMaskIntPolicy
from cem.interventions.uncertainty import UncertaintyMaximizerPolicy
from cem.interventions.coop import CooPEntropy, CooP,CompetenceCooPEntropy
from cem.interventions.optimal import GreedyOptimal, TrueOptimal
from cem.interventions.behavioural_learning import BehavioralLearningPolicy
from cem.interventions.intcem_policy import IntCemInterventionPolicy
from cem.interventions.global_policies import (
    GlobalValidationPolicy,
    GlobalValidationImprovementPolicy,
)
from cem.interventions.arbitrary_conditionals import (
    LearntExpectedInfoGainPolicy,
    ExpectedLossImprovement,
    NewExpectedLossImprovement,
)
from run_experiments import CUB_CONFIG, CELEBA_CONFIG, SYNTH_CONFIG


INT_BATCH_SIZE = int(os.environ.get(f"INT_BATCH_SIZE", "1024"))
MAX_COMB_BOUND = 5000
POLICY_NAMES = [
    "intcem_policy",
    "group_random",
    "group_random_no_prior",
    "group_coop_no_prior",
    "behavioural_cloning_no_prior",
#     "group_coop",
#     "new_expected_loss_improvement",
#     "new_expected_loss_improvement_no_prior",
#     "group_competence_coop_entropy",
    "group_uncertainty_no_prior",
    "optimal_greedy_no_prior",
    "global_val_error_no_prior",
    "global_val_improvement_no_prior",
#     "group_coop_entropy",

#     "new_non_greedy_expected_loss_improvement",
#     "expected_loss_improvement",
#     "learnt_expected_gain",
#     "optimal_global",
#     "individual_uncertainty",
#     "individual_random",
#     "individual_coop",
#     "individual_coop_entropy",
]


################################################################################
## HELPER FUNCTIONS
################################################################################

def _get_mnist_extractor_arch(input_shape, num_operands):
    def c_extractor_arch(output_dim):
        intermediate_maps = 16
        output_dim = output_dim or 128 #int(np.prod(input_shape[2:]))*intermediate_maps
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

def _filter_results(results, full_run_name, cut=False):
    output = {}
    for key, val in results.items():
        if full_run_name not in key:
            continue
        if cut:
            key = key[: -len("_" + full_run_name)]
        output[key] = val
    return output

def _get_int_policy(
    policy_name,
    n_tasks,
    n_concepts,
    config,
    acquisition_costs=None,
    result_dir='results/interventions/',
    tune_params=True,
    concept_group_map=None,
    intervened_groups=None,
    val_dl=None,
    train_dl=None,
    imbalance=None,
    task_class_weights=None,
    split=0,
    rerun=False,
    sequential=False,
    independent=False,
    gpu=torch.cuda.is_available(),
):
    og_policy_name = policy_name
    policy_name = policy_name.lower()

    if "random" in policy_name:
        concept_selection_policy = IndependentRandomMaskIntPolicy
    elif "intcem_policy" in policy_name:
        concept_selection_policy = IntCemInterventionPolicy
    elif "global_val_improvement" in policy_name:
        concept_selection_policy = GlobalValidationImprovementPolicy
    elif "uncertainty" in policy_name:
        concept_selection_policy = UncertaintyMaximizerPolicy
    elif "global_val_error" in policy_name:
        concept_selection_policy = GlobalValidationPolicy
    elif "learnt_expected_gain" in policy_name:
        concept_selection_policy = LearntExpectedInfoGainPolicy
    elif "expected_loss_improvement" in policy_name:
        if "new" in policy_name:
            concept_selection_policy = NewExpectedLossImprovement
        else:
            concept_selection_policy = ExpectedLossImprovement
    elif "coop" in policy_name:
        concept_selection_policy = (
            CooPEntropy if "entropy" in policy_name
            else CooP
        )
        if "competence" in policy_name:
            concept_selection_policy = CompetenceCooPEntropy
    elif "behavioural_cloning" in policy_name:
        concept_selection_policy = BehavioralLearningPolicy

    elif "optimal_greedy" in policy_name:
        concept_selection_policy = GreedyOptimal
    elif "optimal_global" in policy_name:
        concept_selection_policy = TrueOptimal
    else:
        raise ValueError(f'Unsupported policy name "{og_policy_name}"')

    def _params_fn(
        intervened_groups=intervened_groups,
        concept_group_map=concept_group_map,
        tune_params=tune_params,
        rerun=rerun,
    ):
        policy_params = {}
        policy_params["include_prior"] = not ("no_prior" in policy_name)
        if "random" in policy_name:
            policy_params["group_based"] = not (
                "individual" in policy_name
            )
        elif "intcem_policy" in policy_name:
            policy_params["group_based"] = not (
                "individual" in policy_name
            )
            policy_params["n_tasks"] = n_tasks
            policy_params["importance_weight"] = config.get("importance_weight", 1)
            policy_params["acquisition_weight"] = config.get("acquisition_weight", 1)
            policy_params["acquisition_costs"] = acquisition_costs
            policy_params["n_tasks"] = n_tasks
            policy_params["n_concepts"] = n_concepts
            policy_params["eps"] = config.get("eps", 1e-8)
        elif "global_val_improvement" in policy_name:
            policy_params['n_concepts'] = n_concepts
            policy_params['val_ds'] = val_dl
            policy_params["group_based"] = not (
                "individual" in policy_name
            )
        elif "uncertainty" in policy_name:
            policy_params["eps"] = config.get("eps", 1e-8)
            policy_params["group_based"] = (
                policy_name == "uncertainty" or
                ("group" in policy_name)
            )
        elif "global_val_error" in policy_name:
            policy_params["eps"] = config.get("eps", 1e-8)
            policy_params["group_based"] = not (
                "individual" in policy_name
            )
            _, _, _, _, _, _, _, val_c_aucs = intervention_utils.generate_arb_conds_training_data(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                split=split,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                train_dl=train_dl,
                val_dl=val_dl,
                result_dir=result_dir,
                config=config,
                sequential=sequential,
                independent=independent,
                rerun=rerun,
                gpu=gpu,
                seed=(42 + split),
            )
            policy_params["val_c_aucs"] = val_c_aucs
        elif "learnt_expected_gain" in policy_name:
            policy_params["importance_weight"] = config.get("importance_weight", 1)
            policy_params["acquisition_weight"] = config.get("acquisition_weight", 1)
            policy_params["concept_entropy_weight"] = config.get("concept_entropy_weight", 0.1)
            policy_params["acquisition_costs"] = acquisition_costs
            policy_params["n_tasks"] = n_tasks
            policy_params["n_concepts"] = n_concepts
            policy_params["eps"] = config.get("eps", 1e-8)
            policy_params["group_based"] = not (
                "individual" in policy_name
            )
            full_run_name = (
                f"{config['architecture']}{config.get('extra_name', '')}"
            )
            key_name = (
                f'individual_coop_{full_run_name}'
                if "individual" in policy_name else f'group_coop_{full_run_name}'
            )

            if concept_group_map is None:
                concept_group_map = dict(
                    [(i, [i]) for i in range(n_concepts)]
                )
            if intervened_groups is None:
                # Then intervene on 1% 5% 25% 50% of all concepts
                if policy_params["group_based"]:
                    intervened_groups = set([
                        int(np.ceil(p * len(concept_group_map)))
                        for p in [0.01, 0.05, 0.25, 0.5]
                    ])
                else:
                    intervened_groups = set([
                        int(np.ceil(p * n_concepts))
                        for p in [0.01, 0.05, 0.25, 0.5]
                    ])
                # We do this to avoid running the same twice if, say,
                # 1% of the groups and 5% of the groups gives use the
                # same whole number once the ceiling is applied
                intervened_groups = sorted(intervened_groups)
            best_params = intervention_utils.fine_tune_coop(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                split=split,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                val_dl=val_dl,
                train_dl=train_dl,
                result_dir=result_dir,
                config=config,
                intervened_groups=intervened_groups,
                concept_group_map=concept_group_map,
                concept_entropy_weight_range=config.get(
                    'concept_entropy_weight_range',
                    None,
                ),
                importance_weight_range=config.get(
                    'importance_weight_range',
                    None,
                ),
                acquisition_weight_range=config.get(
                    'acquisition_weight_range',
                    None,
                ),
                acquisition_costs=acquisition_costs,
                group_based=policy_params["group_based"],
                eps=policy_params["eps"],
                key_name=key_name,
                coop_variant=CooP,
                sequential=sequential,
                independent=independent,
                rerun=rerun,
                batch_size=INT_BATCH_SIZE,
                seed=(42 + split),
                include_prior=policy_params["include_prior"],
            )
            print("Best params found for", policy_name, "are:")
            for param_name, param_value in best_params.items():
                policy_params[param_name] = param_value
                print(f"\t{param_name} = {param_value}")


            x_train, y_train, c_train, c_sem, c_pred, y_pred, ground_truth_embs_train, val_c_aucs = intervention_utils.generate_arb_conds_training_data(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                split=split,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                train_dl=train_dl,
                val_dl=val_dl,
                result_dir=result_dir,
                config=config,
                sequential=sequential,
                independent=independent,
                rerun=rerun,
                gpu=gpu,
                seed=(42 + split),
            )
            policy_params["x_train"] = x_train
            policy_params["c_sem"] = c_sem
            policy_params["c_embs_train"] = c_pred
            policy_params["y_pred_train"] = y_pred
            policy_params["result_dir"] = result_dir
            policy_params["include_inputs"] = config.get('include_inputs', False)
            policy_params["vae_latent_dim"] = config.get('vae_latent_dim', 32)
            policy_params["freeze_encoder"] = config.get('freeze_encoder', False)
            policy_params["weight_decay"] = config.get('weight_decay', 0.00001)
            policy_params["batch_size"] = config.get('batch_size', 512)
            policy_params["train_epochs"] = config.get('vae_train_epochs', 500)
            policy_params["lookahead_train_epochs"] = config.get('lookahead_train_epochs', 250)
            policy_params["info_gains_samples"] = config.get('info_gains_samples', 100)
            policy_params["matching_coef"] = config.get('matching_coef', 1)
            policy_params["seed"] = config.get('seed', 42) + split
            policy_params["binary_inputs"] = False
            if (
                (config["architecture"] == "ConceptBottleneckModel") and
                config.get("sigmoidal_prob", True)
            ):
                policy_params["binary_inputs"] = True
            policy_params["emb_size"] = (
                config["emb_size"] if config["architecture"] in [
                    "CEM",
                    "ConceptEmbeddingModel",
                    "IntAwareConceptEmbeddingModel",
                    "IntCEM",
                ]
                else 1
            )
            policy_params["full_run_name"] = f"{full_run_name}_fold_{split}"


        elif "expected_loss_improvement" in policy_name:
            policy_params["non_greedy"] = "non_greedy" in policy_name
            policy_params["importance_weight"] = config.get("importance_weight", 1)
            policy_params["acquisition_weight"] = config.get("acquisition_weight", 1)
            policy_params["acquisition_costs"] = acquisition_costs
            policy_params["n_tasks"] = n_tasks
            policy_params["n_concepts"] = n_concepts
            policy_params["eps"] = config.get("eps", 1e-8)
            policy_params["group_based"] = not (
                "individual" in policy_name
            )
            full_run_name = (
                f"{config['architecture']}{config.get('extra_name', '')}"
            )



            x_train, y_train, c_train, c_sem, c_pred, y_pred, ground_truth_embs_train, val_c_aucs = intervention_utils.generate_arb_conds_training_data(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                split=split,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                train_dl=train_dl,
                val_dl=val_dl,
                result_dir=result_dir,
                config=config,
                sequential=sequential,
                independent=independent,
                rerun=rerun,
                gpu=gpu,
                seed=(42 + split),
            )

            policy_params["x_train"] = x_train
            policy_params["y_train"] = y_train
            policy_params["c_train"] = c_train
            policy_params["val_c_aucs"] = val_c_aucs
            policy_params["c_embs_train"] = c_pred
            policy_params["y_pred_train"] = y_pred
            policy_params["ground_truth_embs_train"] = ground_truth_embs_train
            policy_params["result_dir"] = result_dir
            policy_params["vae_latent_dim"] = config.get('vae_latent_dim', 32)
            policy_params["freeze_encoder"] = config.get('freeze_encoder', False)
            policy_params["weight_decay"] = config.get('weight_decay', 0.00001)
            policy_params["batch_size"] = config.get('batch_size', 512)
            policy_params["train_epochs"] = config.get('vae_train_epochs', 500)
            policy_params["matching_coef"] = config.get('matching_coef', 1)
            policy_params["seed"] = config.get('seed', 42) + split
            if (
                (config["architecture"] == "ConceptBottleneckModel") and
                config.get("sigmoidal_prob", True)
            ):
                policy_params["binary_inputs"] = True
            policy_params["emb_size"] = (
                config["emb_size"] if config["architecture"] in [
                    "CEM",
                    "ConceptEmbeddingModel",
                    "IntAwareConceptEmbeddingModel",
                    "IntCEM",
                ]
                else 1
            )
            policy_params["full_run_name"] = f"{full_run_name}_fold_{split}"
        elif "coop" in policy_name:
            policy_params["concept_entropy_weight"] = config.get("concept_entropy_weight", 1)
            policy_params["importance_weight"] = config.get("importance_weight", 1)
            policy_params["acquisition_weight"] = config.get("acquisition_weight", 1)
            policy_params["acquisition_costs"] = acquisition_costs
            policy_params["n_tasks"] = n_tasks
            policy_params["eps"] = config.get("eps", 1e-8)
            policy_params["group_based"] = (
                not ("individual" in policy_name)
            )
            if "competence" in policy_name:
                tune_params = False

            # Then also run our hyperparameter search using the validation data, if given
            if tune_params and (val_dl is not None):
                full_run_name = (
                    f"{config['architecture']}{config.get('extra_name', '')}"
                )
                key_name = f'group_coop_{full_run_name}'
                if concept_group_map is None:
                    concept_group_map = dict(
                        [(i, [i]) for i in range(n_concepts)]
                    )
                if intervened_groups is None:
                    # Then intervene on 1% 5% 25% 50% of all concepts
                    if policy_params["group_based"]:
                        intervened_groups = set([
                            int(np.ceil(p * len(concept_group_map)))
                            for p in [0.01, 0.05, 0.25, 0.5]
                        ])
                    else:
                        intervened_groups = set([
                            int(np.ceil(p * n_concepts))
                            for p in [0.01, 0.05, 0.25, 0.5]
                        ])
                    # We do this to avoid running the same twice if, say,
                    # 1% of the groups and 5% of the groups gives use the
                    # same whole number once the ceiling is applied
                    intervened_groups = sorted(intervened_groups)
                best_params = intervention_utils.fine_tune_coop(
                    n_concepts=n_concepts,
                    n_tasks=n_tasks,
                    split=split,
                    imbalance=imbalance,
                    task_class_weights=task_class_weights,
                    val_dl=val_dl,
                    train_dl=train_dl,
                    result_dir=result_dir,
                    config=config,
                    intervened_groups=intervened_groups,
                    concept_group_map=concept_group_map,
                    concept_entropy_weight_range=config.get(
                        'concept_entropy_weight_range',
                        None,
                    ),
                    importance_weight_range=config.get(
                        'importance_weight_range',
                        None,
                    ),
                    acquisition_weight_range=config.get(
                        'acquisition_weight_range',
                        None,
                    ),
                    acquisition_costs=acquisition_costs,
                    group_based=policy_params["group_based"],
                    eps=policy_params["eps"],
                    key_name=key_name,
                    coop_variant=concept_selection_policy,
                    sequential=sequential,
                    independent=independent,
                    rerun=rerun,
                    batch_size=INT_BATCH_SIZE,
                    seed=(42 + split),
                    include_prior=policy_params["include_prior"],
                )
                print("Best params found for", policy_name, "are:")
                for param_name, param_value in best_params.items():
                    policy_params[param_name] = param_value
                    print(f"\t{param_name} = {param_value}")
        elif "behavioural_cloning" in policy_name:
            policy_params["n_tasks"] = n_tasks
            policy_params["n_concepts"] = n_concepts
            policy_params["group_based"] = not (
                "individual" in policy_name
            )
            full_run_name = (
                f"{config['architecture']}{config.get('extra_name', '')}"
            )

            x_train, y_train, c_train, _, _, _, _, _ = intervention_utils.generate_arb_conds_training_data(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                split=split,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                train_dl=train_dl,
                val_dl=val_dl,
                result_dir=result_dir,
                config=config,
                sequential=sequential,
                independent=independent,
                rerun=rerun,
                gpu=gpu,
                seed=(42 + split),
            )
            policy_params["x_train"] = x_train
            policy_params["y_train"] = y_train
            policy_params["c_train"] = c_train
            policy_params["emb_size"] = (
                config["emb_size"] if config["architecture"] in [
                    "CEM",
                    "ConceptEmbeddingModel",
                    "IntAwareConceptEmbeddingModel",
                    "IntCEM",
                ]
                else 1
            )
            policy_params["result_dir"] = result_dir
            policy_params["batch_size"] = config.get('batch_size', 512)
            policy_params["dataset_size"] = config.get('bc_dataset_size', 5000)
            policy_params["train_epochs"] = config.get('bc_train_epochs', 100)
            policy_params["seed"] = config.get('seed', 42) + split
            policy_params["full_run_name"] = f"{full_run_name}_fold_{split}"
            policy_params["rerun"] = rerun

        elif "optimal_greedy" in policy_name:
            policy_params["acquisition_costs"] = acquisition_costs
            policy_params["acquisition_weight"] = config.get("acquisition_weight", 1)
            policy_params["importance_weight"] = config.get("importance_weight", 1)
            policy_params["n_tasks"] = n_tasks
            policy_params["group_based"] = not (
                "individual" in policy_name
            )
        elif "optimal_global" in policy_name:
            policy_params["acquisition_costs"] = acquisition_costs
            policy_params["acquisition_weight"] = config.get("acquisition_weight", 1)
            policy_params["importance_weight"] = config.get("importance_weight", 1)
            policy_params["group_based"] = not (
                "individual" in policy_name
            )
        else:
            raise ValueError(f'Unsupported policy name "{og_policy_name}"')

        return policy_params
    return _params_fn, concept_selection_policy

def _rerun_policy(rerun, policy_name, config, split):
    if rerun:
        return True
    full_run_name = (
        f"{config['architecture']}{config.get('extra_name', '')}"
    )
    if os.environ.get(f"RERUN_INTERVENTIONS", "0") == "1":
        return True
    if "coop" in policy_name.lower() and (
        (os.environ.get(f"RERUN_COOP_TUNING", "0") == "1")
    ):
        return True
    if os.environ.get(f"RERUN_INTERVENTION_{policy_name.upper()}", "0") == "1":
        rerun_list = os.environ.get(f"RERUN_INTERVENTION_MODELS", "")
        if rerun_list:
            rerun_list = rerun_list.split(",")
        if len(rerun_list) == 0:
            # Then we always rerun this guy
            return True
        # Else, check if one of the models we are asking to rerun corresponds to this guy
        for model_to_rerun in rerun_list:
            if model_to_rerun in full_run_name:
                return True
    return False
def test_interventions(
    full_run_name,
    train_dl,
    val_dl,
    test_dl,
    imbalance,
    config,
    n_tasks,
    n_concepts,
    acquisition_costs,
    result_dir,
    concept_map,
    intervened_groups,
    used_policies=POLICY_NAMES,
    intervention_batch_size=INT_BATCH_SIZE,
    competence_levels=[1], #, 0.9, 0.75, 0.6, 0.5, "unif"], #[0.5, 0.6, 0.75, 0.9, 1], #0.25, 1, 0, 0.5, 0.75],
    gpu=1,
    split=0,
    rerun=False,
    sequential=False,
    independent=False,
    old_results=None,
    task_class_weights=None,
):
    results = {}
    if hasattr(test_dl.dataset, 'tensors'):
        x_test, y_test, c_test = test_dl.dataset.tensors
    else:
        x_test, y_test, c_test = [], [], []
        for ds_data in test_dl:
            if len(ds_data) == 2:
                x, (y, c) = ds_data
            else:
                (x, y, c) = ds_data
            x_type = x.type()
            y_type = y.type()
            c_type = c.type()
            x_test.append(x)
            y_test.append(y)
            c_test.append(c)
        x_test = torch.FloatTensor(
            np.concatenate(x_test, axis=0)
        ).type(x_type)
        y_test = torch.FloatTensor(
            np.concatenate(y_test, axis=0)
        ).type(y_type)
        c_test = torch.FloatTensor(
            np.concatenate(c_test, axis=0)
        ).type(c_type)


    for competence_level in competence_levels:
        def competence_generator(
            x,
            y,
            c,
            concept_group_map,
        ):
            if competence_level == "unif":
                 # When using uniform competence, we will assign the same competence
                # level to the same batch index
                # The same competence is assigned to all concepts within the same
                # group
                np.random.seed(42)
                batch_group_level_competencies = np.random.uniform(
                    0.5,
                    1,
                    size=(c.shape[0], len(concept_group_map)),
                )
                batch_concept_level_competencies = np.ones((c.shape[0], c.shape[1]))
                for group_idx, (_, group_concepts) in enumerate(concept_group_map.items()):
                    batch_concept_level_competencies[:, group_concepts] = np.expand_dims(
                        batch_group_level_competencies[:, group_idx],
                        axis=-1,
                    )
                return batch_concept_level_competencies
            return np.ones(c.shape) * competence_level

        for policy in used_policies:
            if os.environ.get(f"IGNORE_INTERVENTION_{policy.upper()}", "0") == "1":
                continue
            if "optimal_global" in policy:
                eff_n_concepts = len(concept_map) if (
                    "group" in policy or "optimal_global" == policy
                ) else n_concepts
                used_intervened_groups = [
                    x if int(scipy.special.comb(eff_n_concepts, x)) <= MAX_COMB_BOUND else None
                    for x in intervened_groups

                ]
                print("\tUsing intervened groups", used_intervened_groups, "with optimal global policy")
            else:
                used_intervened_groups = intervened_groups
            policy_params_fn, concept_selection_policy = _get_int_policy(
                policy_name=policy,
                config=config,
                n_tasks=n_tasks,
                n_concepts=n_concepts,
                acquisition_costs=acquisition_costs,
                result_dir=result_dir,
                tune_params=config.get('tune_params', True),
                concept_group_map=concept_map,
                intervened_groups=config.get('tune_intervened_groups', None),
                val_dl=val_dl,
                train_dl=train_dl,
                gpu=gpu if gpu else None,
                imbalance=imbalance,
                split=split,
                rerun=_rerun_policy(rerun, policy, config, split),
                sequential=sequential,
                independent=independent,
                task_class_weights=task_class_weights,
            )
            print(f"\tIntervening in {full_run_name} with policy {policy} and competence {competence_level}")
            if competence_level == 1:
                key = f'test_acc_y_{policy}_ints_{full_run_name}'
                int_time_key = f'avg_int_time_{policy}_ints_{full_run_name}'
                construction_times_key = f'construction_time_{policy}_ints_{full_run_name}'
            else:
                key = f'test_acc_y_{policy}_ints_competence_{competence_level}_{full_run_name}'
                int_time_key = f'avg_int_time_{policy}_ints_competence_{competence_level}_{full_run_name}'
                construction_times_key = f'construction_time_{policy}_ints_competence_{competence_level}_{full_run_name}'

            (int_results, avg_time, constr_time), loaded = training.load_call(
                function=intervention_utils.intervene_in_cbm,
                keys=(key, int_time_key, construction_times_key),
                old_results=old_results,
                full_run_name=full_run_name,
                rerun=_rerun_policy(rerun, policy, config, split),
                kwargs=dict(
                    concept_selection_policy=concept_selection_policy,
                    policy_params=policy_params_fn,
                    concept_group_map=concept_map,
                    intervened_groups=used_intervened_groups,
                    gpu=gpu if gpu else None,
                    config=config,
                    test_dl=test_dl,
                    train_dl=train_dl,
                    n_tasks=n_tasks,
                    n_concepts=n_concepts,
                    result_dir=result_dir,
                    imbalance=imbalance,
                    split=split,
                    rerun=_rerun_policy(rerun, policy, config, split),
                    batch_size=intervention_batch_size,
                    key_name=key,
                    competence_generator=competence_generator,
                    x_test=x_test,
                    y_test=y_test,
                    c_test=c_test,
                    test_subsampling=config.get('test_subsampling', 1),
                    sequential=sequential,
                    independent=independent,
                    seed=(42 + split),
                    task_class_weights=task_class_weights,
                ),
            )
            results[key] = int_results
            results[int_time_key] = avg_time
            results[construction_times_key] = constr_time
            if loaded:
                if avg_time:
                    extra = f" (avg int time is {avg_time:.5f}s and construction time is {constr_time:.5f}s)"
                else:
                    extra = ""
                for num_groups_intervened, val in enumerate(int_results):
                    if n_tasks > 1:
                        print(
                            f"\t\tTest accuracy when intervening with {num_groups_intervened} "
                            f"concept groups is {val * 100:.2f}%{extra}."
                        )
                    else:
                        print(
                            f"\t\tTest AUC when intervening with {num_groups_intervened} "
                            f"concept groups is {val * 100:.2f}%{extra}."
                        )

            if policy in ["group_random", "individual_random"] and (
                "IntAware" in config["architecture"]
            ):
                # Then we will also attempt some rollouts where we give the model an actual boundary
                # budget
                budget_intervened_groups = [
                    int(np.ceil(len(used_intervened_groups) * percent))
                    for percent in [0.25] #, 0.5, 0.75, 1]
                ]
                for budget_limit in budget_intervened_groups:
                    if competence_level == 1:
                        key = f'test_acc_y_{policy}_budgeted_{budget_limit}_ints_{full_run_name}'
                        int_time_key = f'avg_int_time_{policy}_budgeted_{budget_limit}_ints_{full_run_name}'
                        construction_times_key = f'construction_time_{policy}_budgeted_{budget_limit}_ints_{full_run_name}'
                    else:
                        key = f'test_acc_y_{policy}_budgeted_{budget_limit}_ints_competence_{competence_level}_{full_run_name}'
                        int_time_key = f'avg_int_time_{policy}_budgeted_i{budget_limit}_nts_competence_{competence_level}_{full_run_name}'
                        construction_times_key = f'construction_time_{policy}_budgeted_{budget_limit}_ints_competence_{competence_level}_{full_run_name}'

                    current_budget_intervened_groups = list(range(0, budget_limit + 1, 1))
                    (int_results, avg_time, constr_time), loaded = training.load_call(
                        function=intervention_utils.intervene_in_cbm,
                        keys=(key, int_time_key, construction_times_key),
                        old_results=old_results,
                        full_run_name=full_run_name,
                        rerun=_rerun_policy(rerun, policy, config, split),
                        kwargs=dict(
                            concept_selection_policy=concept_selection_policy,
                            policy_params=policy_params_fn,
                            concept_group_map=concept_map,
                            intervened_groups=current_budget_intervened_groups,
                            gpu=gpu if gpu else None,
                            config=config,
                            test_dl=test_dl,
                            train_dl=train_dl,
                            n_tasks=n_tasks,
                            n_concepts=n_concepts,
                            result_dir=result_dir,
                            imbalance=imbalance,
                            split=split,
                            rerun=_rerun_policy(rerun, policy, config, split),
                            batch_size=intervention_batch_size,
                            key_name=key,
                            competence_generator=competence_generator,
                            x_test=x_test,
                            y_test=y_test,
                            c_test=c_test,
                            test_subsampling=config.get('test_subsampling', 1),
                            sequential=sequential,
                            independent=independent,
                            seed=(42 + split),
                            budgeted=True,
                            task_class_weights=task_class_weights,
                        ),
                    )
                    results[key] = int_results
                    results[int_time_key] = avg_time
                    results[construction_times_key] = constr_time
                    if loaded:
                        if avg_time:
                            extra = f" (avg int time is {avg_time:.5f}s and construction time is {constr_time:.5f}s)"
                        else:
                            extra = ""
                        for num_groups_intervened, val in enumerate(int_results):
                            print(
                                f"\t\tTest accuracy when intervening, using explicit BUDGET {budget_limit}, with {num_groups_intervened} "
                                f"concept groups is {val * 100:.2f}%{extra}."
                            )
    return results


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

    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = \
        data_module.generate_data(
            config=og_config,
            seed=42,
            output_dataset_vars=True,
        )
    # For now, we assume that all concepts have the same
    # aquisition cost
    acquisition_costs = None
    if concept_map is not None:
        intervened_groups = list(
            range(
                0,
                len(concept_map) + 1,
                og_config.get('intervention_freq', 1),
            )
        )
    else:
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
    logging.info(f"Training sample shape is: {sample[0].shape} with type {sample[0].type()}")
    logging.info(f"Training label shape is: {sample[1].shape} with type {sample[1].type()}")
    logging.info(f"\tNumber of output classes: {n_tasks}")
    logging.info(f"Training concept shape is: {sample[2].shape} with type {sample[2].type()}")
    logging.info(f"\tNumber of training concepts: {n_concepts}")

    task_class_weights = None

    if og_config.get('use_task_class_weights', False):
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

    os.makedirs(result_dir, exist_ok=True)
    results = {}
    for split in range(og_config.get("start_split", 0), og_config["cv"]):
        results[f'{split}'] = {}
        logging.info(f'Experiment {split+1}/{og_config["cv"]}')

        # CEM Training
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptEmbeddingModel"
        config["extra_name"] = ""
        config["shared_prob_gen"] = True
        config["sigmoidal_prob"] = True
        config["sigmoidal_embedding"] = False
        config['training_intervention_prob'] = 0.25
        config['concat_prob'] = False
        config['emb_size'] = config['emb_size']
        config["embedding_activation"] = "leakyrelu"
        config["concept_loss_weight"] = config.get(
            "cem_concept_loss_weight",
            config.get("concept_loss_weight", 5),
        )
        old_results = None
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}"
        )
        current_results_path = os.path.join(
            result_dir,
            f'{full_run_name}_split_{split}_results.joblib'
        )
        if os.path.exists(current_results_path):
            old_results = joblib.load(current_results_path)
        cem,  cem_results = \
            training.train_model(
                task_class_weights=task_class_weights,
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
                seed=(42 + split),
                imbalance=imbalance,
                old_results=old_results,
                gradient_clip_val=config.get('gradient_clip_val', 0),
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            cem,
            cem_results,
        )
        results[f'{split}'].update(test_interventions(
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
            old_results=old_results,
        ))
        results[f'{split}'].update(training.evaluate_representation_metrics(
            config=config,
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

        print(f"\tResults for {full_run_name} in split {split}:")
        for key, val in _filter_results(results[f'{split}'], full_run_name, cut=True).items():
            print(f"\t\t{key} -> {val}")
        joblib.dump(
            _filter_results(results[f'{split}'], full_run_name),
            current_results_path,
        )
        if og_config.get("start_split", 0) == 0:
            joblib.dump(results, os.path.join(result_dir, f'results.joblib'))
        extr_name = config['c_extractor_arch']
        if not isinstance(extr_name, str):
            extr_name = "lambda"
        cem_train_path = os.path.join(
            result_dir or ".",
            f'{full_run_name}_{extr_name}_fold_{split + 1}.pt'
        )

        # IntAwareCEM Training with the final task loss of the trajectory also included
        # in the minimization
        for intervention_weight in og_config.get(
            "intervention_weight_experiment",
            [og_config.get('intervention_weight', 5)]
        ): #, 1]:
            for intervention_task_discount in og_config.get(
                "intervention_task_discount_experiment",
                [og_config.get('intervention_task_discount', 1.1)],
            ): #, 0.9]:
                for horizon_rate in og_config.get(
                    "horizon_rate_experiment",
                    [og_config.get('horizon_rate', 1.005)],
                ):
                    for intervention_discount in [og_config.get('intervention_discount', 1)]: #[0.95, 0.99]:
                        for horizon_uniform_distr in [True]:
                            for use_concept_groups in og_config.get('use_concept_groups_experiment', [True]): #, False]):
                                for max_horizon in [
                                    len(concept_map) if concept_map else n_concepts,
    #                                 (len(concept_map) if concept_map else n_concepts)//2,
    #                                 (len(concept_map) if concept_map else n_concepts)//4,
                                ]:
                                    config = copy.deepcopy(og_config)
                                    config["max_epochs"] = config.get("intcem_max_epochs", config.get("max_epochs", 100))
                                    config["architecture"] = "IntAwareConceptEmbeddingModel"
                                    config['training_intervention_prob'] = config.get(
                                        "cem_training_intervention_prob",
                                        config.get('training_intervention_prob', 0.25),
                                    )
                                    config['horizon_binary_representation'] =  config.get('horizon_binary_representation', True) # BINARY REPR FOR HORIZON
                                    config['include_task_trajectory_loss'] = config.get('include_task_trajectory_loss', True) # BUT REQUEST TO INCLUDE TASK LOSS IN TRAJECTORY!
                                    config['include_only_last_trajectory_loss'] = config.get('include_only_last_trajectory_loss', True) # AND ONLY INCLUDE THE LAST ONE HERE
                                    config['task_loss_weight'] = config.get("intcem_task_loss_weight", 0)  # Zeroing out to include only task loss from after interventions!
                                    config['intervention_weight'] = intervention_weight
                                    config['intervention_task_loss_weight'] = config.get("intervention_task_loss_weight", 1)
                                    config['initial_horizon'] = config.get('initial_horizon', 2)  # Make sure we at least intervene with one concept at all times!
                                    config["use_concept_groups"] = use_concept_groups
                                    config['emb_size'] = config['emb_size']
                                    prev_value = config.get('concept_loss_weight', 5)
                                    config['concept_loss_weight'] = config.get(
                                        'intcem_concept_loss_weight',
                                        config.get('concept_loss_weight', 5),
                                    )
                                    config["embedding_activation"] = "leakyrelu"
                                    config["concept_map"] = concept_map
                                    config["tau"] = config.get('tau', 1)
                                    config["max_horizon"] = max_horizon
                                    config["horizon_uniform_distr"] = horizon_uniform_distr
                                    config["beta_a"] = og_config.get('beta_a', 1)
                                    config["beta_b"] = og_config.get('beta_b', 3)
                                    config["intervention_task_discount"] = intervention_task_discount
                                    config['average_trajectory'] = config.get('average_trajectory', True)
                                    config["use_horizon"] = config.get('use_horizon', True)
                                    config["model_pretrain_path"] = cem_train_path if config.get("pretrain_intcem", False) else None

                                    config['horizon_rate'] = horizon_rate
                                    config['intervention_discount'] = intervention_discount

                                    config["extra_name"] = (
                                        f"LastOnly_intervention_weight_{config['intervention_weight']}_"
                                        f"horizon_rate_{config['horizon_rate']}_"
                                        f"intervention_discount_{config['intervention_discount']}_"
                                        f"tau_{config['tau']}_"
                                        f"max_horizon_{config['max_horizon']}_"
                                        f"task_discount_{config['intervention_task_discount']}_"
                                        f"uniform_distr_{config['horizon_uniform_distr']}"
                                    )
                                    if not config['use_concept_groups']:
                                        config["extra_name"] += f"_use_cg_{config['use_concept_groups']}"
                                    if config['concept_loss_weight'] != prev_value:
                                        config["extra_name"] += f"_c_weight_{config['concept_loss_weight']}"
                                    if not config["use_horizon"]:
                                        config["extra_name"] += f"_no_horizon"
                                    if config.get("rollout_init_steps", 0) > 0:
                                        config["extra_name"] += f"_init_steps_{config['rollout_init_steps']}"


                                    old_results = None
                                    full_run_name = (
                                        f"{config['architecture']}{config.get('extra_name', '')}"
                                    )
                                    current_results_path = os.path.join(
                                        result_dir,
                                        f'{full_run_name}_split_{split}_results.joblib'
                                    )
                                    if os.path.exists(current_results_path):
                                        old_results = joblib.load(current_results_path)
                                    intcem,  intcem_results = \
                                        training.train_model(
                                            task_class_weights=task_class_weights,
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
                                            seed=(42 + split),
                                            imbalance=imbalance,
                                            gradient_clip_val=config.get('gradient_clip_val', 1000),
                                            old_results=old_results,
                                        )
                                    training.update_statistics(
                                        results[f'{split}'],
                                        config,
                                        intcem,
                                        intcem_results,
                                    )
                                    results[f'{split}'].update(test_interventions(
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
                                        old_results=old_results,
                                    ))

                                    results[f'{split}'].update(training.evaluate_representation_metrics(
                                        config=config,
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

                                    print(f"\tResults for {full_run_name} in split {split}:")
                                    for key, val in _filter_results(results[f'{split}'], full_run_name, cut=True).items():
                                        print(f"\t\t{key} -> {val}")
                                    joblib.dump(
                                        _filter_results(results[f'{split}'], full_run_name),
                                        current_results_path,
                                    )
                                    if og_config.get("start_split", 0) == 0:
                                        joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

        ######################################### LEGACY #########################################
        if int(os.environ.get(f"RUN_LEGACY", "0")):
            # IntAwareCEM Training
            for intervention_weight in og_config.get('intervention_weights', [5, 1]):
                for horizon_rate in og_config.get('horizon_rate', [1.005]):
                    for intervention_discount in [0.95]:
                        for max_horizon in og_config.get('max_horizons', [15, 10, 5]):
                            config = copy.deepcopy(og_config)
                            config["architecture"] = "IntAwareConceptEmbeddingModel"
                            config["shared_prob_gen"] = True
                            config["sigmoidal_prob"] = True
                            config["sigmoidal_embedding"] = False
                            config['training_intervention_prob'] = config.get('training_intervention_prob', 0.25)
                            config['concat_prob'] = False
                            config['emb_size'] = config['emb_size']
                            config['concept_loss_weight'] = 5
                            config["embedding_activation"] = "leakyrelu"
                            config["concept_map"] = concept_map
                            config["tau"] = 1
                            config["max_horizon"] = max_horizon
                            config["max_epochs"] = 100

                            config['intervention_weight'] = intervention_weight
                            config['horizon_rate'] = horizon_rate
                            config['intervention_discount'] = intervention_discount
                            config['average_trajectory'] = config.get('average_trajectory', False)
                            config["extra_name"] = (
                                f"_intervention_weight_{config['intervention_weight']}_"
                                f"horizon_rate_{config['horizon_rate']}_"
                                f"intervention_discount_{config['intervention_discount']}_"
                                f"average_trajectory_{config['average_trajectory']}_"
                                f"tau_{config['tau']}"
                            )
                            if max_horizon != 5:
                                config["extra_name"] += f"_max_horizon_{config['max_horizon']}"
                            old_results = None
                            full_run_name = (
                                f"{config['architecture']}{config.get('extra_name', '')}"
                            )
                            current_results_path = os.path.join(
                                result_dir,
                                f'{full_run_name}_split_{split}_results.joblib'
                            )
                            if os.path.exists(current_results_path):
                                old_results = joblib.load(current_results_path)
                            intcem,  intcem_results = \
                                training.train_model(
                                    task_class_weights=task_class_weights,
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
                                    seed=(42 + split),
                                    imbalance=imbalance,
                                    gradient_clip_val=1000,
                                    old_results=old_results,
                                )
                            training.update_statistics(
                                results[f'{split}'],
                                config,
                                intcem,
                                intcem_results,
                            )
                            results[f'{split}'].update(test_interventions(
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
                                old_results=old_results,
                            ))

                            print(f"\tResults for {full_run_name} in split {split}:")
                            for key, val in _filter_results(results[f'{split}'], full_run_name, cut=True).items():
                                print(f"\t\t{key} -> {val}")
                            joblib.dump(
                                _filter_results(results[f'{split}'], full_run_name),
                                current_results_path,
                            )
                            if og_config.get("start_split", 0) == 0:
                                joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

#         # IntAwareCBMLogit Training
#         for intervention_weight in og_config.get('intervention_weights', [5, 1]):
#             for horizon_rate in og_config.get('horizon_rate', [1.005]):
#                 for intervention_discount in [0.95]:
#                     for max_horizon in og_config.get('max_horizons', [15, 10, 5]):
#                         config = copy.deepcopy(og_config)
#                         config["architecture"] = "IntAwareConceptBottleneckModel"
#                         config['concept_loss_weight'] = 5
#                         config["bool"] = False
#                         config["extra_dims"] = 0
#                         config["sigmoidal_extra_capacity"] = False
#                         config["sigmoidal_prob"] = False

#                         config["concept_map"] = concept_map
#                         config["tau"] = 1
#                         config["max_horizon"] = max_horizon
#                         config['intervention_weight'] = intervention_weight
#                         config['horizon_rate'] = horizon_rate
#                         config['intervention_discount'] = intervention_discount
#                         config['average_trajectory'] = config.get('average_trajectory', False)
#                         config["extra_name"] = (
#                             f"Logit_intervention_weight_{config['intervention_weight']}_"
#                             f"horizon_rate_{config['horizon_rate']}_"
#                             f"intervention_discount_{config['intervention_discount']}_"
#                             f"average_trajectory_{config['average_trajectory']}_"
#                             f"tau_{config['tau']}"
#                         )
#                         if max_horizon != 5:
#                             config["extra_name"] += f"_max_horizon_{config['max_horizon']}"

#                         old_results = None
#                         full_run_name = (
#                             f"{config['architecture']}{config.get('extra_name', '')}"
#                         )
#                         current_results_path = os.path.join(
#                             result_dir,
#                             f'{full_run_name}_split_{split}_results.joblib'
#                         )
#                         if os.path.exists(current_results_path):
#                             old_results = joblib.load(current_results_path)
#                         intcem,  intcem_results = \
#                             training.train_model(
#                                 task_class_weights=task_class_weights,
#                                 gpu=gpu if gpu else 0,
#                                 n_concepts=n_concepts,
#                                 n_tasks=n_tasks,
#                                 config=config,
#                                 train_dl=train_dl,
#                                 val_dl=val_dl,
#                                 test_dl=test_dl,
#                                 split=split,
#                                 result_dir=result_dir,
#                                 rerun=rerun,
#                                 project_name=project_name,
#                                 seed=split,
#                                 imbalance=imbalance,
#                                 gradient_clip_val=1000,
#                                 old_results=old_results,
#                             )
#                         training.update_statistics(
#                             results[f'{split}'],
#                             config,
#                             intcem,
#                             intcem_results,
#                         )
#                         current_results_path = os.path.join(
#                             result_dir,
#                             f'{full_run_name}_split_{split}_results.joblib'
#                         )
#                         results[f'{split}'].update(test_interventions(
#                             task_class_weights=task_class_weights,
#                             full_run_name=full_run_name,
#                             train_dl=train_dl,
#                             val_dl=val_dl,
#                             test_dl=test_dl,
#                             imbalance=imbalance,
#                             config=config,
#                             n_tasks=n_tasks,
#                             n_concepts=n_concepts,
#                             acquisition_costs=acquisition_costs,
#                             result_dir=result_dir,
#                             concept_map=concept_map,
#                             intervened_groups=intervened_groups,
#                             gpu=gpu,
#                             split=split,
#                             rerun=rerun,
#                             old_results=old_results,
#                         ))
#                         print(f"\tResults for {full_run_name} in split {split}:")
#                         for key, val in _filter_results(results[f'{split}'], full_run_name, cut=True).items():
#                             print(f"\t\t{key} -> {val}")
#                         joblib.dump(
#                             _filter_results(results[f'{split}'], full_run_name),
#                             current_results_path,
#                         )
#                         if og_config.get("start_split", 0) == 0:
#                             joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

        ######################################### LEGACY #########################################
        if int(os.environ.get(f"RUN_LEGACY", "0")):
            # IntAwareCEM but without RandInt to see if it helps at all or not!
            for intervention_weight in [1]:
                for horizon_rate in [1.005]:
                    for intervention_discount in [0.95]:
                        for max_horizon in og_config.get('max_horizons', [15]):
                            config = copy.deepcopy(og_config)
                            config["architecture"] = "IntAwareConceptEmbeddingModel"
                            config["shared_prob_gen"] = True
                            config["sigmoidal_prob"] = True
                            config["sigmoidal_embedding"] = False
                            config['training_intervention_prob'] = 0 # NO RANDINT!!!!!!!!!
                            config['include_task_trajectory_loss'] = True # BUT REQUEST TO INCLUDE TASK LOSS IN TRAJECTORY!
                            config['concat_prob'] = False
                            config['emb_size'] = config['emb_size']
                            config['concept_loss_weight'] = 5
                            config["embedding_activation"] = "leakyrelu"
                            config["concept_map"] = concept_map
                            config["tau"] = 1
                            config["max_horizon"] = max_horizon

                            config['intervention_weight'] = intervention_weight
                            config['horizon_rate'] = horizon_rate
                            config['intervention_discount'] = intervention_discount
                            config['average_trajectory'] = config.get('average_trajectory', False)
                            config["extra_name"] = (
                                f"NoRandInt_TaskLossTrajectory_intervention_weight_{config['intervention_weight']}_"
                                f"horizon_rate_{config['horizon_rate']}_"
                                f"intervention_discount_{config['intervention_discount']}_"
                                f"average_trajectory_{config['average_trajectory']}_"
                                f"tau_{config['tau']}"
                            )
                            if max_horizon != 5:
                                config["extra_name"] += f"_max_horizon_{config['max_horizon']}"
                            old_results = None
                            full_run_name = (
                                f"{config['architecture']}{config.get('extra_name', '')}"
                            )
                            current_results_path = os.path.join(
                                result_dir,
                                f'{full_run_name}_split_{split}_results.joblib'
                            )
                            if os.path.exists(current_results_path):
                                old_results = joblib.load(current_results_path)
                            intcem, intcem_results = \
                                training.train_model(
                                    task_class_weights=task_class_weights,
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
                                    seed=(42 + split),
                                    imbalance=imbalance,
                                    gradient_clip_val=1000,
                                    old_results=old_results,
                                )
                            training.update_statistics(
                                results[f'{split}'],
                                config,
                                intcem,
                                intcem_results,
                            )
                            results[f'{split}'].update(test_interventions(
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
                                old_results=old_results,
                            ))

                            print(f"\tResults for {full_run_name} in split {split}:")
                            for key, val in _filter_results(results[f'{split}'], full_run_name, cut=True).items():
                                print(f"\t\t{key} -> {val}")
                            joblib.dump(
                                _filter_results(results[f'{split}'], full_run_name),
                                current_results_path,
                            )
                            if og_config.get("start_split", 0) == 0:
                                joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

            # IntAwareCBMSigmoid Training
            for intervention_weight in og_config.get('intervention_weights', [5, 1]):
                for horizon_rate in og_config.get('horizon_rate', [1.005]):
                    for intervention_discount in [0.95]:
                        for max_horizon in og_config.get('max_horizons', [15, 5]):
                            config = copy.deepcopy(og_config)
                            config["architecture"] = "IntAwareConceptBottleneckModel"
                            config['concept_loss_weight'] = 5
                            config["bool"] = False
                            config["extra_dims"] = 0
                            config["sigmoidal_extra_capacity"] = False
                            config["sigmoidal_prob"] = True

                            config["concept_map"] = concept_map
                            config["tau"] = 1
                            config["max_horizon"] = max_horizon
                            config['intervention_weight'] = intervention_weight
                            config['horizon_rate'] = horizon_rate
                            config['intervention_discount'] = intervention_discount
                            config['average_trajectory'] = config.get('average_trajectory', False)
                            config["extra_name"] = (
                                f"Sigmoid_intervention_weight_{config['intervention_weight']}_"
                                f"horizon_rate_{config['horizon_rate']}_"
                                f"intervention_discount_{config['intervention_discount']}_"
                                f"average_trajectory_{config['average_trajectory']}_"
                                f"tau_{config['tau']}"
                            )
                            if max_horizon != 5:
                                config["extra_name"] += f"_max_horizon_{config['max_horizon']}"
                            old_results = None
                            full_run_name = (
                                f"{config['architecture']}{config.get('extra_name', '')}"
                            )
                            current_results_path = os.path.join(
                                result_dir,
                                f'{full_run_name}_split_{split}_results.joblib'
                            )
                            if os.path.exists(current_results_path):
                                old_results = joblib.load(current_results_path)
                            intcem,  intcem_results = \
                                training.train_model(
                                    task_class_weights=task_class_weights,
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
                                    seed=(42 + split),
                                    imbalance=imbalance,
                                    gradient_clip_val=1000,
                                    old_results=old_results,
                                )
                            training.update_statistics(
                                results[f'{split}'],
                                config,
                                intcem,
                                intcem_results,
                            )
                            results[f'{split}'].update(test_interventions(
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
                                old_results=old_results,
                            ))

                            print(f"\tResults for {full_run_name} in split {split}:")
                            for key, val in _filter_results(results[f'{split}'], full_run_name, cut=True).items():
                                print(f"\t\t{key} -> {val}")
                            joblib.dump(
                                _filter_results(results[f'{split}'], full_run_name),
                                current_results_path,
                            )
                            if og_config.get("start_split", 0) == 0:
                                joblib.dump(results, os.path.join(result_dir, f'results.joblib'))


#         # IntAwareCEM but without RandInt to see if it helps at all or not!
#         for intervention_weight in [5]:
#             for horizon_rate in [1.005]:
#                 for intervention_discount in [0.95]:
#                     for max_horizon in og_config.get('max_horizons', [15, 10, 5]):
#                         config = copy.deepcopy(og_config)
#                         config["architecture"] = "IntAwareConceptEmbeddingModel"
#                         config["shared_prob_gen"] = True
#                         config["sigmoidal_prob"] = True
#                         config["sigmoidal_embedding"] = False
#                         config['training_intervention_prob'] = 0 # NO RANDINT!!!!!!!!!
#                         config['concat_prob'] = False
#                         config['emb_size'] = config['emb_size']
#                         config['concept_loss_weight'] = 5
#                         config["embedding_activation"] = "leakyrelu"
#                         config["concept_map"] = concept_map
#                         config["tau"] = 1
#                         config["max_horizon"] = max_horizon

#                         config['intervention_weight'] = intervention_weight
#                         config['horizon_rate'] = horizon_rate
#                         config['intervention_discount'] = intervention_discount
#                         config['average_trajectory'] = config.get('average_trajectory', False)
#                         config["extra_name"] = (
#                             f"NoRandInt_intervention_weight_{config['intervention_weight']}_"
#                             f"horizon_rate_{config['horizon_rate']}_"
#                             f"intervention_discount_{config['intervention_discount']}_"
#                             f"average_trajectory_{config['average_trajectory']}_"
#                             f"tau_{config['tau']}"
#                         )
#                         if max_horizon != 5:
#                             config["extra_name"] += f"_max_horizon_{config['max_horizon']}"
#                         old_results = None
#                         full_run_name = (
#                             f"{config['architecture']}{config.get('extra_name', '')}"
#                         )
#                         current_results_path = os.path.join(
#                             result_dir,
#                             f'{full_run_name}_split_{split}_results.joblib'
#                         )
#                         if os.path.exists(current_results_path):
#                             old_results = joblib.load(current_results_path)
#                         intcem, intcem_results = \
#                             training.train_model(
#                                 task_class_weights=task_class_weights,
#                                 gpu=gpu if gpu else 0,
#                                 n_concepts=n_concepts,
#                                 n_tasks=n_tasks,
#                                 config=config,
#                                 train_dl=train_dl,
#                                 val_dl=val_dl,
#                                 test_dl=test_dl,
#                                 split=split,
#                                 result_dir=result_dir,
#                                 rerun=rerun,
#                                 project_name=project_name,
#                                 seed=(42 + split),
#                                 imbalance=imbalance,
#                                 gradient_clip_val=1000,
#                                 old_results=old_results,
#                             )
#                         training.update_statistics(
#                             results[f'{split}'],
#                             config,
#                             intcem,
#                             intcem_results,
#                         )
#                         results[f'{split}'].update(test_interventions(
#                             task_class_weights=task_class_weights,
#                             full_run_name=full_run_name,
#                             train_dl=train_dl,
#                             val_dl=val_dl,
#                             test_dl=test_dl,
#                             imbalance=imbalance,
#                             config=config,
#                             n_tasks=n_tasks,
#                             n_concepts=n_concepts,
#                             acquisition_costs=acquisition_costs,
#                             result_dir=result_dir,
#                             concept_map=concept_map,
#                             intervened_groups=intervened_groups,
#                             gpu=gpu,
#                             split=split,
#                             rerun=rerun,
#                             old_results=old_results,
#                         ))

#                         print(f"\tResults for {full_run_name} in split {split}:")
#                         for key, val in _filter_results(results[f'{split}'], full_run_name, cut=True).items():
#                             print(f"\t\t{key} -> {val}")
#                         joblib.dump(
#                             _filter_results(results[f'{split}'], full_run_name),
#                             current_results_path,
#                         )
#                         if og_config.get("start_split", 0) == 0:
#                             joblib.dump(results, os.path.join(result_dir, f'results.joblib'))


        # Train vanilla CBM models with both logits and sigmoidal
        # bottleneck activations
        # CBM with sigmoidal bottleneck
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["extra_name"] = f"Sigmoid"
        config["bool"] = False
        config["extra_dims"] = 0
        config["sigmoidal_extra_capacity"] = False
        config["sigmoidal_prob"] = True
        config["concept_loss_weight"] = config.get(
            "cbm_concept_loss_weight",
            config.get("concept_loss_weight", 5),
        )
        old_results = None
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}"
        )
        current_results_path = os.path.join(
            result_dir,
            f'{full_run_name}_split_{split}_results.joblib'
        )
        if os.path.exists(current_results_path):
            old_results = joblib.load(current_results_path)
        cbm_sigmoid_model, cbm_sigmoid_test_results = \
            training.train_model(
                task_class_weights=task_class_weights,
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
                seed=(42 + split),
                imbalance=imbalance,
                old_results=old_results,
                gradient_clip_val=config.get('gradient_clip_val', 0),
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            cbm_sigmoid_model,
            cbm_sigmoid_test_results,
        )
        results[f'{split}'].update(test_interventions(
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
            old_results=old_results,
        ))
        results[f'{split}'].update(training.evaluate_representation_metrics(
            config=config,
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

        # save results
        print(f"\tResults for {full_run_name} in split {split}:")
        for key, val in _filter_results(results[f'{split}'], full_run_name, cut=True).items():
            print(f"\t\t{key} -> {val}")
        joblib.dump(
            _filter_results(results[f'{split}'], full_run_name),
            current_results_path,
        )
        if og_config.get("start_split", 0) == 0:
            joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

        # CBM with logits
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["bool"] = False
        config["extra_dims"] = 0
        config["extra_name"] = f"Logit"
        config["bottleneck_nonlinear"] = "leakyrelu"
        config["sigmoidal_extra_capacity"] = False
        config["sigmoidal_prob"] = False
        config["concept_loss_weight"] = config.get(
            "cbm_concept_loss_weight",
            config.get("concept_loss_weight", 5),
        )
        old_results = None
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}"
        )
        current_results_path = os.path.join(
            result_dir,
            f'{full_run_name}_split_{split}_results.joblib'
        )
        if os.path.exists(current_results_path):
            old_results = joblib.load(current_results_path)
        cbm_logit_model, cbm_logit_test_results = \
            training.train_model(
                task_class_weights=task_class_weights,
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
                seed=(42 + split),
                imbalance=imbalance,
                old_results=old_results,
                gradient_clip_val=config.get('gradient_clip_val', 1000),
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            cbm_logit_model,
            cbm_logit_test_results,
        )
        results[f'{split}'].update(test_interventions(
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
            old_results=old_results,
        ))
        results[f'{split}'].update(training.evaluate_representation_metrics(
            config=config,
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

        print(f"\tResults for {full_run_name} in split {split}:")
        for key, val in _filter_results(results[f'{split}'], full_run_name, cut=True).items():
            print(f"\t\t{key} -> {val}")
        joblib.dump(
            _filter_results(results[f'{split}'], full_run_name),
            current_results_path,
        )
        if og_config.get("start_split", 0) == 0:
            joblib.dump(results, os.path.join(result_dir, f'results.joblib'))


        # sequential and independent CBMs
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["extra_name"] = f""
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
        results[f'{split}'].update(test_interventions(
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
        print(f"\tResults for {full_run_name} in split {split}:")
        for key, val in _filter_results(results[f'{split}'], full_run_name, cut=True).items():
            print(f"\t\t{key} -> {val}")
        joblib.dump(
            _filter_results(results[f'{split}'], full_run_name),
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
        results[f'{split}'].update(test_interventions(
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
        print(f"\tResults for {full_run_name} in split {split}:")
        for key, val in _filter_results(results[f'{split}'], full_run_name, cut=True).items():
            print(f"\t\t{key} -> {val}")
        joblib.dump(
            _filter_results(results[f'{split}'], full_run_name),
            seq_current_results_path,
        )
        if og_config.get("start_split", 0) == 0:
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
        'dataset',
        choices=['cub', 'celeba', 'xor', 'vector', 'dot', 'trig', 'mnist_add', 'chexpert'],
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

    if args.config:
        loaded_config = yaml.load(args.config)
    else:
        loaded_config = None

    if args.dataset == "cub":
        data_module = cub_data_module
        og_config = loaded_config or CUB_CONFIG
        args.output_dir = args.output_dir.format(ds_name="cub")
        args.project_name = args.project_name.format(ds_name="cub")
    elif args.dataset == "celeba":
        data_module = celeba_data_module
        og_config = loaded_config or CELEBA_CONFIG
        args.output_dir = args.output_dir.format(ds_name="celeba")
        args.project_name = args.project_name.format(ds_name="celeba")
    elif args.dataset == "chexpert":
        data_module = chexpert_data_module
        og_config = loaded_config or CUB_CONFIG
        args.output_dir = args.output_dir.format(ds_name="chexpert")
        args.project_name = args.project_name.format(ds_name="chexpert")
    elif args.dataset in ["xor", "vector", "dot", "trig"]:
        data_module = get_synthetic_data_loader(args.dataset)
        og_config = loaded_config or SYNTH_CONFIG.copy()
        args.output_dir = args.output_dir.format(ds_name=args.dataset)
        args.project_name = args.project_name.format(ds_name=args.dataset)
        input_features = get_synthetic_num_features(args.dataset)
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
        og_config["c_extractor_arch"] = synth_c_extractor_arch
    elif args.dataset == "mnist_add":
        data_module = mnist_data_module
        og_config = loaded_config or CUB_CONFIG.copy()
        args.output_dir = args.output_dir.format(ds_name=args.dataset)
        args.project_name = args.project_name.format(ds_name=args.dataset)
        utils.extend_with_global_params(
            og_config,
            args.param or []
        )
        num_operands = og_config.get('num_operands', 32)
        og_config["c_extractor_arch"] = _get_mnist_extractor_arch(
            input_shape=(og_config.get('batch_size', 512), num_operands, 28, 28),
            num_operands=num_operands,
        )

    else:
        raise ValueError(f"Unsupported dataset {args.dataset}!")

    og_config['results_dir'] = args.output_dir
    logging.info(f"Results will be dumped in {og_config['results_dir']}")
    Path(og_config['results_dir']).mkdir(parents=True, exist_ok=True)
    # Write down the actual command executed
    with open(os.path.join(og_config['results_dir'], "command.txt"), "w") as f:
        program_args = [arg if " " not in arg else f'"{arg}"' for arg in sys.argv]
        f.write("python " + " ".join(program_args))

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
