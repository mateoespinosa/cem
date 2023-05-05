import os
import numpy as np
import torch
import pytorch_lightning as pl
import logging
import io
from contextlib import redirect_stdout

import cem.train.training as cem_train
from cem.interventions.random import IndependentRandomMaskIntPolicy

################################################################################
## MAIN INTERVENTION FUNCTION
################################################################################

def intervene_in_cbm(
    config,
    test_dl,
    n_tasks,
    n_concepts,
    result_dir,
    imbalance=None,
    task_class_weights=None,
    adversarial_intervention=False,
    train_dl=None,
    sequential=False,
    independent=False,
    concept_group_map=None,
    intervened_groups=None,
    gpu=int(torch.cuda.is_available()),
    split=0,
    concept_selection_policy=IndependentRandomMaskIntPolicy,
    rerun=False,
    old_results=None,
    batch_size=None,
    policy_params=None,
    key_name="",
):
    if batch_size is not None:
        # Then overwrite the config's batch size
        test_dl = torch.utils.data.DataLoader(
            dataset=test_dl.dataset,
            batch_size=batch_size,
            num_workers=test_dl.num_workers,
        )
    intervention_accs = []
    # If no concept groups are given, then we assume that all concepts
    # represent a unitary group themselves
    concept_group_map = concept_group_map or dict(
        [(i, [i]) for i in range(n_concepts)]
    )
    groups = intervened_groups or list(range(0, len(concept_group_map) + 1, 1))

    if (not rerun) and key_name:
        result_file = os.path.join(
            result_dir,
            key_name + f"_fold_{split}.npy",
        )
        if os.path.exists(result_file):
            result = np.load(result_file)
            for j, num_groups_intervened in enumerate(groups):
                print(
                    f"\tTest accuracy when intervening with {num_groups_intervened} "
                    f"concept groups is {result[j] * 100:.2f}%."
                )
            return result
    config["shared_prob_gen"] = config.get("shared_prob_gen", False)
    config["per_concept_weight"] = config.get(
        "per_concept_weight",
        False,
    )

    model = cem_train.load_trained_model(
        config=config,
        n_tasks=n_tasks,
        n_concepts=n_concepts,
        result_dir=result_dir,
        split=split,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
        intervene=True,
        train_dl=train_dl,
        sequential=sequential,
        independent=independent,
        output_latent=True,
    )

    for j, num_groups_intervened in enumerate(groups):
        logging.debug(
            f"Intervening with {num_groups_intervened} out of "
            f"{len(concept_group_map)} concept groups"
        )
        n_trials = config.get('intervention_trials', 1)
        avg = []
        for trial in range(n_trials):

            logging.debug(
                f"\tFor trial {trial + 1}/{n_trials} for split {split} with "
                f"{num_groups_intervened} groups intervened"
            )

            ####
            # Set the model's intervention policy
            ####

            # Example of how to use a mask-generating policy!
            model.intervention_policy = concept_selection_policy(
                num_groups_intervened=num_groups_intervened,
                concept_group_map=concept_group_map,
                cbm=model,
                **(policy_params or {}),
            )

            trainer = pl.Trainer(
                gpus=gpu,
            )
            f = io.StringIO()
            with redirect_stdout(f):
                [test_results] = trainer.test(model, test_dl, verbose=False)
            acc = test_results['test_y_accuracy']
            avg.append(acc)
            logging.debug(
                f"\tFor model at split {split}, intervening with "
                f"{num_groups_intervened} groups (trial {trial + 1}) gives "
                f"test task accuracy {acc * 100:.2f}%."
            )
        print(
            f"\tTest accuracy when intervening with {num_groups_intervened} "
            f"concept groups is "
            f"{np.mean(avg) * 100:.2f}% ± {np.std(avg)* 100:.2f}%."
        )
        intervention_accs.append(np.mean(avg))
    if key_name:
        result_file = os.path.join(
            result_dir,
            key_name + f"_fold_{split}.npy",
        )
        np.save(result_file, intervention_accs)
    return intervention_accs
