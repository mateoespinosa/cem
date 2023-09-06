import argparse
import copy
import joblib
import numpy as np
import os

import logging
import torch
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
from run_experiments import CUB_CONFIG, CELEBA_CONFIG, SYNTH_CONFIG
from experiment_utils import filter_results


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
    accelerator="auto",
    devices="auto",
    og_config=None,
):
    seed_everything(42)
    # parameters for data, model, and training
    if og_config is None:
        # Then we use the CUB one as the default
        og_config = CUB_CONFIG
    og_config = copy.deepcopy(og_config)
    og_config['num_workers'] = num_workers

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
                accelerator=accelerator,
                devices=devices,
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
            accelerator=accelerator,
            devices=devices,
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
            accelerator=accelerator,
            devices=devices,
            rerun=rerun,
            seed=42,
            old_results=old_results,
        ))

        print(f"\tResults for {full_run_name} in split {split}:")
        for key, val in filter_results(results[f'{split}'], full_run_name, cut=True).items():
            print(f"\t\t{key} -> {val}")
        joblib.dump(
            filter_results(results[f'{split}'], full_run_name),
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
                                            accelerator=accelerator,
                                            devices=devices,
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
                                        accelerator=accelerator,
                                        devices=devices,
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
                                            devices=devices,
                                            devices=devices,
                                        rerun=rerun,
                                        seed=42,
                                        old_results=old_results,
                                    ))

                                    print(f"\tResults for {full_run_name} in split {split}:")
                                    for key, val in filter_results(results[f'{split}'], full_run_name, cut=True).items():
                                        print(f"\t\t{key} -> {val}")
                                    joblib.dump(
                                        filter_results(results[f'{split}'], full_run_name),
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
                                    accelerator=accelerator,
                                    devices=devices,
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
                                accelerator=accelerator,
                                devices=devices,
                                split=split,
                                rerun=rerun,
                                old_results=old_results,
                            ))

                            print(f"\tResults for {full_run_name} in split {split}:")
                            for key, val in filter_results(results[f'{split}'], full_run_name, cut=True).items():
                                print(f"\t\t{key} -> {val}")
                            joblib.dump(
                                filter_results(results[f'{split}'], full_run_name),
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
#                                 accelerator=accelerator,
#                                 devices=devices,
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
#                         results[f'{split}'].update(intervention_utils.test_interventions(
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
#                             accelerator=accelerator,
#                             devices=devices,
#                             split=split,
#                             rerun=rerun,
#                             old_results=old_results,
#                         ))
#                         print(f"\tResults for {full_run_name} in split {split}:")
#                         for key, val in filter_results(results[f'{split}'], full_run_name, cut=True).items():
#                             print(f"\t\t{key} -> {val}")
#                         joblib.dump(
#                             filter_results(results[f'{split}'], full_run_name),
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
                                    accelerator=accelerator,
                                    devices=devices,
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
                                accelerator=accelerator,
                                devices=devices,
                                split=split,
                                rerun=rerun,
                                old_results=old_results,
                            ))

                            print(f"\tResults for {full_run_name} in split {split}:")
                            for key, val in filter_results(results[f'{split}'], full_run_name, cut=True).items():
                                print(f"\t\t{key} -> {val}")
                            joblib.dump(
                                filter_results(results[f'{split}'], full_run_name),
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
                                    accelerator=accelerator,
                                    devices=devices,
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
                                accelerator=accelerator,
                                devices=devices,
                                split=split,
                                rerun=rerun,
                                old_results=old_results,
                            ))

                            print(f"\tResults for {full_run_name} in split {split}:")
                            for key, val in filter_results(results[f'{split}'], full_run_name, cut=True).items():
                                print(f"\t\t{key} -> {val}")
                            joblib.dump(
                                filter_results(results[f'{split}'], full_run_name),
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
#                                 accelerator=accelerator,
#                                 devices=devices,
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
#                         results[f'{split}'].update(intervention_utils.test_interventions(
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
#                             accelerator=accelerator,
#                             devices=devices,
#                             split=split,
#                             rerun=rerun,
#                             old_results=old_results,
#                         ))

#                         print(f"\tResults for {full_run_name} in split {split}:")
#                         for key, val in filter_results(results[f'{split}'], full_run_name, cut=True).items():
#                             print(f"\t\t{key} -> {val}")
#                         joblib.dump(
#                             filter_results(results[f'{split}'], full_run_name),
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
                accelerator=accelerator,
                devices=devices,
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
            accelerator=accelerator,
            devices=devices,
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
            accelerator=accelerator,
            devices=devices,
            rerun=rerun,
            seed=42,
            old_results=old_results,
        ))

        # save results
        print(f"\tResults for {full_run_name} in split {split}:")
        for key, val in filter_results(results[f'{split}'], full_run_name, cut=True).items():
            print(f"\t\t{key} -> {val}")
        joblib.dump(
            filter_results(results[f'{split}'], full_run_name),
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
                accelerator=accelerator,
                devices=devices,
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
            accelerator=accelerator,
            devices=devices,
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
            accelerator=accelerator,
            devices=devices,
            rerun=rerun,
            seed=42,
            old_results=old_results,
        ))

        print(f"\tResults for {full_run_name} in split {split}:")
        for key, val in filter_results(results[f'{split}'], full_run_name, cut=True).items():
            print(f"\t\t{key} -> {val}")
        joblib.dump(
            filter_results(results[f'{split}'], full_run_name),
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
            accelerator=accelerator,
            devices=devices,
            split=split,
            rerun=rerun,
            old_results=ind_old_results,
            independent=True,
        ))
        print(f"\tResults for {full_run_name} in split {split}:")
        for key, val in filter_results(results[f'{split}'], full_run_name, cut=True).items():
            print(f"\t\t{key} -> {val}")
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
            accelerator=accelerator,
            devices=devices,
            split=split,
            rerun=rerun,
            old_results=seq_old_results,
            sequential=True,
        ))
        print(f"\tResults for {full_run_name} in split {split}:")
        for key, val in filter_results(results[f'{split}'], full_run_name, cut=True).items():
            print(f"\t\t{key} -> {val}")
        joblib.dump(
            filter_results(results[f'{split}'], full_run_name),
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
        accelerator=(
            "gpu" if (not args.force_cpu) and (torch.cuda.is_available())
            else "cpu"
        ),
        og_config=og_config,
    )
