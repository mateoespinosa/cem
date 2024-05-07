import os
import numpy as np
import torch
import pytorch_lightning as pl
import logging
import joblib
import io
from contextlib import redirect_stdout
import scipy.special
import sklearn.metrics
from scipy.special import expit
import time
from pytorch_lightning import seed_everything
from typing import Callable

import cem.utils.data as data_utils

from cem.train.utils import load_call
from cem.models.construction import load_trained_model
from cem.interventions.random import IndependentRandomMaskIntPolicy
from cem.interventions.random import IndependentRandomMaskIntPolicy
from cem.interventions.uncertainty import UncertaintyMaximizerPolicy
from cem.interventions.coop import CooP
from cem.interventions.optimal import GreedyOptimal
from cem.interventions.behavioural_learning import BehavioralLearningPolicy
from cem.interventions.global_policies import (
    GlobalValidationPolicy,
    GlobalValidationImprovementPolicy,
)

################################################################################
## Global Variables
################################################################################


MAX_COMB_BOUND = 5000
DEFAULT_POLICIES = [
    dict(
        policy="random",
        use_prior=True,  # This will use IntCEM's learnt prior intervention
                         # policy to sample next interventions
        group_level=True,
    ),
    dict(
        policy="random",
        use_prior=False,
        group_level=True,
    ),
    dict(
        policy="coop",
        use_prior=False,
        group_level=True,
    ),
    dict(
        policy="behavioural_cloning",
        use_prior=False,
        group_level=True,
    ),
    dict(
        policy="group_uncertainty",
        use_prior=False,
        group_level=True,
    ),
    dict(
        policy="optimal_greedy",
        use_prior=False,
        group_level=True,
    ),
    dict(
        policy="global_val_error",
        use_prior=False,
        group_level=True,
    ),
    dict(
        policy="global_val_improvement",
        use_prior=False,
        group_level=True,
    ),
]


################################################################################
## CONCEPT INTERVENTION SELECTION POLICIES
################################################################################

class InterventionPolicyWrapper(object):

    def __init__(
            self,
            policy_fn,
            concept_group_map,
            num_groups_intervened=0,
            include_prior=True,
        ):
        self.policy_fn = policy_fn
        self.num_groups_intervened = num_groups_intervened
        self.concept_group_map = concept_group_map
        self.include_prior = include_prior

    def __call__(
        self,
        x,
        pred_c,
        c,
        y=None,
        competencies=None,
        prev_interventions=None,
        prior_distribution=None,
    ):
        intervention_idxs = self.policy_fn(
            num_groups_intervened=self.num_groups_intervened,
            concept_group_map=self.concept_group_map,
        )
        return intervention_idxs, c


class AllInterventionPolicy(object):

    def __init__(self, value=None, include_prior=True):
        self.value = value
        self.num_groups_intervened = 1
    def __call__(
        self,
        x,
        pred_c,
        c,
        y=None,
        competencies=None,
        prev_interventions=None,
        prior_distribution=None,
    ):
        if self.value is not None:
            c = torch.ones(c.shape, device=c.device) * self.value
        mask = np.ones(c.shape, dtype=np.int32)
        return mask, c

def concepts_from_competencies(
    c,
    competencies,
    use_concept_groups=False,
    concept_map=None,
    assume_mutually_exclusive=False,
):
    if concept_map is None:
        concept_map = {i : [i] for i in range(c.shape[-1])}

    if use_concept_groups:
        # Then we will have to generate one-hot labels for concept groups
        # one concept at a time
        c_updated = c.clone()
        for batch_idx in range(c.shape[0]):
            for group_idx, (_, group_concepts) in enumerate(
                concept_map.items()
            ):
                if assume_mutually_exclusive:
                    group_size = len(group_concepts)
                    wrong_concept_probs = (
                        (1 - competencies[batch_idx, group_idx]) / (
                            group_size - 1
                        )
                    ) if group_size > 1 else 1 - competencies[batch_idx, group_idx]
                    sample_probs = [
                        competencies[batch_idx, group_idx]
                        if c[batch_idx, concept_idx] == 1 else
                        wrong_concept_probs for concept_idx in group_concepts
                    ]
                    selected = np.random.choice(
                        list(range(group_size)),
                        replace=False,
                        p=sample_probs,
                    )
                    selected_group_label = torch.nn.functional.one_hot(
                        torch.LongTensor([selected]),
                        num_classes=group_size,
                    ).type(c_updated.type()).to(c_updated.device)
                    c_updated[batch_idx, group_concepts] = selected_group_label
                else:
                    # This would all be easy to do if concepts within a group
                    # are all mutually exclusive. However, as seen in CUB, this
                    # is not always the case.
                    # Because of this, for now we will assume that the
                    # competence of an entire group is the same of its
                    # constituent concepts.
                    sample_probs = [
                        competencies[batch_idx, group_idx]
                        if c[batch_idx, concept_idx] == 1 else
                        1 - competencies[batch_idx, group_idx]
                        for concept_idx in group_concepts
                    ]
                    c_updated[batch_idx, group_concepts] = torch.bernoulli(
                        torch.FloatTensor(sample_probs),
                    ).to(c.device)

                    # The solution below can also be a candidate
                    # # Because of this, for now we will assume that the
                    # # if a mistake is made on the group, a mistake is made on
                    # # all of its constituent concepts.
                    # if np.random.uniform(0, 1) > competencies[batch_idx, group_idx]:
                    #     # Then we will assume the entire group was incorrectly
                    #     # intervened on
                    #     c_updated[batch_idx, group_concepts] = (
                    #         1 - c[batch_idx, group_concepts]
                    #     )

    else:
        # Else we assume we are given binary competencies
        if isinstance(competencies, np.ndarray):
            competencies = torch.Tensor(competencies).to(c.device).type(
                c.type()
            )
        correctly_selected = torch.bernoulli(competencies).to(c.device)
        c_updated = (
            c * correctly_selected +
            (1 - c) * (1 - correctly_selected)
        )
    return c_updated.type(torch.FloatTensor).to(c.device)

def _default_competence_generator(
    x,
    y,
    c,
    concept_group_map,
):
    return np.ones(c.shape)


def _random_uniform_competence(
    x,
    y,
    c,
    concept_group_map,
):
    return np.random.uniform(low=0.5, high=1.0, size=c.shape)

################################################################################
## MAIN INTERVENTION FUNCTION
################################################################################

def adversarial_intervene_in_cbm(
    config,
    test_dl,
    n_tasks,
    n_concepts,
    result_dir,
    run_name,
    imbalance=None,
    task_class_weights=None,
    train_dl=None,
    concept_group_map=None,
    intervened_groups=None,
    accelerator="auto",
    devices="auto",
    split=0,
    concept_selection_policy=IndependentRandomMaskIntPolicy,
    rerun=False,
    batch_size=None,
    policy_params=None,
    key_name="",
    test_subsampling=1,
    x_test=None,
    y_test=None,
    c_test=None,
    g_test=None,
    seed=None,
):
    def competence_generator(
        x,
        y,
        c,
        concept_group_map,
    ):
        return np.zeros(c.shape)
    return intervene_in_cbm(
        run_name=run_name,
        config=config,
        test_dl=test_dl,
        n_tasks=n_tasks,
        n_concepts=n_concepts,
        result_dir=result_dir,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
        competence_generator=competence_generator,
        train_dl=train_dl,
        concept_group_map=concept_group_map,
        intervened_groups=intervened_groups,
        accelerator=accelerator,
        devices=devices,
        split=split,
        concept_selection_policy=concept_selection_policy,
        rerun=rerun,
        batch_size=batch_size,
        policy_params=policy_params,
        key_name=key_name,
        test_subsampling=test_subsampling,
        x_test=x_test,
        y_test=y_test,
        c_test=c_test,
        g_test=g_test,
        seed=seed,
    )

def intervene_in_cbm(
    config,
    test_dl,
    n_tasks,
    n_concepts,
    result_dir,
    run_name=None,
    imbalance=None,
    task_class_weights=None,
    competence_generator=_default_competence_generator,
    real_competence_generator=None,
    group_level_competencies=False,
    train_dl=None,
    concept_group_map=None,
    intervened_groups=None,
    accelerator="auto",
    devices="auto",
    split=0,
    concept_selection_policy=IndependentRandomMaskIntPolicy,
    rerun=False,
    batch_size=None,
    policy_params=None,
    key_name="",
    test_subsampling=1,
    x_test=None,
    y_test=None,
    c_test=None,
    g_test=None,
    seed=None,
):
    run_name = run_name or config.get('run_name', config['architecture'])
    if real_competence_generator is None:
        real_competence_generator = lambda x: x
    if seed is not None:
        seed_everything(seed)
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
            total_time_file = os.path.join(
                result_dir,
                key_name + f"_avg_int_time_{run_name}_fold_{split}.npy",
            )
            if os.path.exists(total_time_file):
                avg_time = np.load(total_time_file)
                avg_time = avg_time[0]
            else:
                avg_time = 0

            construct_time_file = os.path.join(
                result_dir,
                key_name + f"_construct_time_{run_name}_fold_{split}.npy",
            )
            if os.path.exists(construct_time_file):
                construct_time = np.load(construct_time_file)
                construct_time = construct_time[0]
            else:
                construct_time = 0
            return result, avg_time, construct_time

    model = load_trained_model(
        config=config,
        n_tasks=n_tasks,
        n_concepts=n_concepts,
        result_dir=result_dir,
        split=split,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
        intervene=True,
        train_dl=train_dl,
        output_latent=True,
        output_interventions=True,
    )
    construct_time = time.time()
    if isinstance(policy_params, Callable):
        # Then we were given some lazy-execution parameters which
        # we will now generate as it seems like we will have to
        # run this after all
        policy_params = policy_params()
    model.intervention_policy = concept_selection_policy(
        concept_group_map=concept_group_map,
        cbm=model,
        **(policy_params or {}),
    )
    construct_time = time.time() - construct_time

    # Now include the competence that we will assume
    # for all concepts
    if (
        (x_test is None) or
        (y_test is None) or
        (c_test is None) or
        (g_test is None)
    ):
        x_test, y_test, c_test, g_test = data_utils.daloader_to_memory(
            test_dl,
            as_torch=True,
            output_groups=True,
        )
    np.random.seed(42)
    indices = np.random.permutation(x_test.shape[0])[
        :int(np.ceil(x_test.shape[0]*test_subsampling))
    ]
    x_test = x_test[indices]
    c_test = c_test[indices]
    y_test = y_test[indices]
    competencies_test = competence_generator(
        x=x_test,
        y=y_test,
        c=c_test,
        concept_group_map=concept_group_map,
    )
    c_test = concepts_from_competencies(
        c=c_test,
        competencies=real_competence_generator(competencies_test),
        use_concept_groups=group_level_competencies,
        concept_map=concept_group_map,
    )
    competencies_test = torch.FloatTensor(competencies_test)
    test_dl = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(
            x_test,
            y_test,
            c_test,
            g_test,
            competencies_test,
        ),
        batch_size=test_dl.batch_size,
        num_workers=test_dl.num_workers,
    )
    prev_num_groups_intervened = 0
    avg_times = []
    for j, num_groups_intervened in enumerate(groups):
        if num_groups_intervened is None:
            # Then this is the case where it is ignored
            intervention_accs.append(0)
            continue
        logging.debug(
            f"Intervening with {num_groups_intervened} out of "
            f"{len(concept_group_map)} concept groups"
        )
        logging.debug(
            f"\tFor split {split} with "
            f"{num_groups_intervened} groups intervened"
        )

        ####
        # Set the model's intervention policy
        ####
        model.intervention_policy.num_groups_intervened = (
            num_groups_intervened - prev_num_groups_intervened
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=False,
        )
        if int(os.environ.get("VERBOSE_INTERVENTIONS", "0")):
            start_time = time.time()
            test_batch_results = trainer.predict(
                model,
                test_dl,
            )

        else:
            f = io.StringIO()
            with redirect_stdout(f):
                start_time = time.time()
                test_batch_results = trainer.predict(
                    model,
                    test_dl,
                )
        coeff = (num_groups_intervened - prev_num_groups_intervened)
        avg_times.append(
            (time.time() - start_time)/(
                x_test.shape[0] * (coeff if coeff != 0 else 1)
            )
        )
        y_pred = np.concatenate(
            list(map(lambda x: x[2].detach().cpu().numpy(), test_batch_results)),
            axis=0,
        )
        if y_pred.shape[-1] > 1:
            y_pred = np.argmax(y_pred, axis=-1)
        else:
            y_pred = np.squeeze((expit(y_pred) >= 0.5).astype(np.int32), axis=-1)
        prev_interventions = np.concatenate(
            list(map(lambda x: x[3].detach().cpu().numpy(), test_batch_results)),
            axis=0,
        )
        if n_tasks > 1:
            acc = np.mean(y_pred == y_test.detach().cpu().numpy())
            logging.debug(
                f"\tTest accuracy when intervening "
                f"with {num_groups_intervened} "
                f"concept groups is {acc * 100:.2f}%."
            )
        else:
            if int(os.environ.get("VERBOSE_INTERVENTIONS", "0")):
                [test_results] = trainer.test(model, test_dl)
            else:
                f = io.StringIO()
                with redirect_stdout(f):
                    [test_results] = trainer.test(model, test_dl)
            acc = test_results['test_y_auc']
            logging.debug(
                f"\tTest AUC when intervening with {num_groups_intervened} "
                f"concept groups is {acc * 100:.2f}% (accuracy "
                f"is {np.mean(y_pred == y_test.detach().cpu().numpy()) * 100:.2f}%)."
            )
        intervention_accs.append(acc)

        # And generate the next dataset so that we can reuse previous
        # interventions on the same samples in the future to save time
        prev_num_groups_intervened = num_groups_intervened
        test_dl = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                x_test,
                y_test,
                c_test,
                g_test,
                competencies_test,
                torch.IntTensor(prev_interventions),
            ),
            batch_size=test_dl.batch_size,
            num_workers=test_dl.num_workers,
        )
    avg_time = np.mean(avg_times)
    print(
        f"\tAverage intervention took {avg_time:.5f} seconds and "
        f"construction took {construct_time:.5f} seconds."
    )
    if key_name:
        result_file = os.path.join(
            result_dir,
            key_name + f"_{run_name}_fold_{split}.npy",
        )
        np.save(result_file, intervention_accs)

        result_file = os.path.join(
            result_dir,
            key_name + f"_avg_int_time_{run_name}_fold_{split}.npy",
        )
        np.save(result_file, np.array([avg_time]))

        result_file = os.path.join(
            result_dir,
            key_name + f"_construct_time_{run_name}_fold_{split}.npy",
        )
        np.save(result_file, np.array([construct_time]))
    return intervention_accs, avg_time, construct_time


##########################
## CooP Fine-tuning
##########################

def fine_tune_coop(
    config,
    coop_variant,
    val_dl,
    n_concepts,
    n_tasks,
    train_dl=None,
    split=0,
    imbalance=None,
    task_class_weights=None,
    result_dir=None,
    intervened_groups=None,
    concept_group_map=None,
    concept_entropy_weight_range=None,
    importance_weight_range=None,
    acquisition_weight_range=None,
    acquisition_costs=None,
    group_based=True,
    eps=1e-8,
    key_name="",
    run_name=None,
    accelerator="auto",
    devices="auto",
    rerun=False,
    batch_size=None,
    include_prior=False,
    seed=None,
):
    run_name = run_name or config.get('run_name', config['architecture'])
    if int(os.environ.get("NO_COOP_FINETUNE", "0")):
        return {
            "concept_entropy_weight": 1,
            "importance_weight": 10,
            "acquisition_weight": 0,
        }
    if seed is not None:
        seed_everything(seed)
    cbm = load_trained_model(
        config=config,
        n_tasks=n_tasks,
        n_concepts=n_concepts,
        result_dir=result_dir,
        split=split,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
        intervene=True,
        train_dl=train_dl,
        output_latent=True,
        output_interventions=True,
    )
    if batch_size is not None:
        # Then overwrite the config's batch size
        val_dl = torch.utils.data.DataLoader(
            dataset=val_dl.dataset,
            batch_size=batch_size,
            num_workers=val_dl.num_workers,
        )
    if (not rerun) and (result_dir is not None):
        result_file = os.path.join(
            result_dir,
            (
                f"coop_best_params{'_' + key_name if key_name else key_name}_"
                f"{run_name}_fold_{split}.joblib"
            ),
        )
        if os.path.exists(result_file):
            return joblib.load(result_file)
    intervention_accs = []
    # If no concept groups are given, then we assume that all concepts
    # represent a unitary group themselves
    concept_group_map = concept_group_map or dict(
        [(i, [i]) for i in range(n_concepts)]
    )
    groups = intervened_groups or list(range(0, len(concept_group_map) + 1, 1))

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
    )
    f = io.StringIO()
    grid_search_results = []
    if concept_entropy_weight_range is None:
        concept_entropy_weight_range = [0.1, 1, 10, 100]
    seen_ratios = set()
    for alpha in concept_entropy_weight_range:
        if importance_weight_range is None:
            importance_weights = [0.1, 1, 10, 100]
        else:
            importance_weights = importance_weight_range
        for beta in importance_weights:
            # If now acquisition weights are given for the search space,
            # then we assume it is always 0
            if acquisition_weight_range is None:
                acquisition_weights = [0]
            else:
                acquisition_weights = acquisition_weight_range
            for gamma in acquisition_weights:
                # Now time to compute the accuracy of intervening on
                # the validation set on a bunch of concepts!
                if gamma == 0 and (beta != 0) and (
                    alpha/beta in np.array(list(seen_ratios))
                ):
                    # Then let's skip it as there is no point checking
                    # this as the ratio between alpha and beta have
                    # already been explored
                    continue
                if beta != 0:
                    seen_ratios.add(alpha/beta)

                intervention_accs = []
                used_params = {
                    "concept_entropy_weight": alpha,
                    "importance_weight": beta,
                    "acquisition_weight": gamma,
                }
                print("Attempting CooP parameters:")
                for k, v in used_params.items():
                    print(f"\t{k} -> {v}")
                for j, num_groups_intervened in enumerate(groups):
                    cbm.intervention_policy = coop_variant(
                        num_groups_intervened=num_groups_intervened,
                        concept_group_map=concept_group_map,
                        cbm=cbm,
                        concept_entropy_weight=alpha,
                        importance_weight=beta,
                        acquisition_weight=gamma,
                        acquisition_costs=acquisition_costs,
                        group_based=group_based,
                        eps=eps,
                        n_tasks=n_tasks,
                        n_concepts=n_concepts,
                        include_prior=include_prior,
                    )
                    with redirect_stdout(f):
                        [test_results] = trainer.test(
                            cbm,
                            val_dl,
                            verbose=False,
                        )
                    intervention_accs.append(test_results['test_y_accuracy'])
                print("\tValidation accuracies are:", intervention_accs)
                grid_search_results.append((used_params, intervention_accs))

    # Sort the results in descending order of their weighted accuracies over
    # all the interventions (weighted by how many concepts we intervened over
    # all concepts)
    acc_weights = 1 - (np.array(intervened_groups) / len(concept_group_map))
    grid_search_results = sorted(
        grid_search_results,
        key=lambda x: -np.sum(x[1] * acc_weights),
    )
    best_params = grid_search_results[0][0]
    if result_dir is not None:
        result_file = os.path.join(
            result_dir,
            (
                f"coop_best_params{'_' + key_name if key_name else key_name}_"
                f"{run_name}_fold_{split}.joblib"
            ),
        )
        joblib.dump(best_params, result_file)

        grid_search_results_file = os.path.join(
            result_dir,
            (
                f"coop_grid_search{'_' + key_name if key_name else key_name}_"
                f"{run_name}_fold_{split}.joblib"
            ),
        )
        joblib.dump(grid_search_results, grid_search_results_file)
    return best_params

def generate_policy_training_data(
    config,
    n_concepts,
    n_tasks,
    train_dl,
    split=0,
    val_dl=None,
    imbalance=None,
    result_dir=None,
    task_class_weights=None,
    accelerator="auto",
    devices="auto",
    rerun=False,
    batch_size=None,
    seed=None,
):
    if seed is not None:
        seed_everything(seed)
    cbm = load_trained_model(
        config=config,
        n_tasks=n_tasks,
        n_concepts=n_concepts,
        result_dir=result_dir,
        split=split,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
        intervene=True,
        train_dl=train_dl,
    )
    batch_size = batch_size or train_dl.batch_size
    x_train, y_train, c_train = data_utils.daloader_to_memory(
        train_dl,
        as_torch=True,
    )
    unshuffle_dl = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(x_train, y_train, c_train),
        batch_size=batch_size,
        num_workers=train_dl.num_workers,
        shuffle=False,
    )
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
    )
    batch_results = trainer.predict(cbm, unshuffle_dl)
    c_sem = np.concatenate(
        list(map(lambda x: x[0].detach().cpu().numpy(), batch_results)),
        axis=0,
    )
    c_pred = np.concatenate(
        list(map(lambda x: x[1].detach().cpu().numpy(), batch_results)),
        axis=0,
    )
    y_pred = np.concatenate(
        list(map(lambda x: x[2].detach().cpu().numpy(), batch_results)),
        axis=0,
    )
    y_pred = scipy.special.softmax(y_pred, axis=-1)

    # Finally, let's compute the ground truth embeddings
    cbm.intervention_policy = AllInterventionPolicy()
    batch_results = trainer.predict(cbm, unshuffle_dl)
    ground_truth_embs = np.concatenate(
        list(map(lambda x: x[1], batch_results)),
        axis=0,
    )
    if (val_dl is not None):
        val_c_aucs = np.zeros((n_concepts,))
        val_batch_results = trainer.predict(cbm, val_dl)
        c_sem = np.concatenate(
            list(map(lambda x: x[0].detach().cpu().numpy(), val_batch_results)),
            axis=0,
        )
        _, _, val_c_true = data_utils.daloader_to_memory(val_dl)
        for concept_idx in range(n_concepts):
            if (
                (len(np.unique(val_c_true[:, concept_idx] >= 0.5)) == 1) or
                (len(np.unique(c_sem[:, concept_idx] >= 0.5)) == 1)
            ):
                val_c_aucs[concept_idx] = sklearn.metrics.accuracy_score(
                    val_c_true[:, concept_idx] >= 0.5,
                    c_sem[:, concept_idx] >= 0.5,
                )
            else:
                val_c_aucs[concept_idx] = sklearn.metrics.roc_auc_score(
                    val_c_true[:, concept_idx] >= 0.5,
                    c_sem[:, concept_idx] >= 0.5,
                )
    else:
        val_c_aucs = None
    return (
        x_train.detach().cpu().numpy(),
        y_train.detach().cpu().numpy().astype(np.int32),
        c_train.detach().cpu().numpy(),
        c_sem,
        c_pred,
        y_pred,
        ground_truth_embs,
        val_c_aucs
    )


def get_int_policy(
    policy_args,
    n_tasks,
    n_concepts,
    config,
    run_name,
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
    accelerator="auto",
    devices="auto",
    intervention_batch_size=1024,
):
    intervention_config = config.get('intervention_config', {})
    intervention_batch_size = intervention_config.get(
        "intervention_batch_size",
        int(os.environ.get(f"INT_BATCH_SIZE", intervention_batch_size)),
    )
    og_policy_name = policy_args['policy']
    policy_name = policy_args['policy'].lower()

    if policy_name == "random":
        concept_selection_policy = IndependentRandomMaskIntPolicy
    elif policy_name == "global_val_improvement":
        concept_selection_policy = GlobalValidationImprovementPolicy
    elif policy_name == "uncertainty":
        concept_selection_policy = UncertaintyMaximizerPolicy
    elif policy_name == "global_val_error":
        concept_selection_policy = GlobalValidationPolicy
    elif policy_name == "coop":
        concept_selection_policy = CooP
    elif policy_name == "behavioural_cloning":
        concept_selection_policy = BehavioralLearningPolicy
    elif policy_name == "optimal_greedy":
        concept_selection_policy = GreedyOptimal
    else:
        raise ValueError(f'Unsupported policy name "{og_policy_name}"')

    def _params_fn(
        intervened_groups=intervened_groups,
        concept_group_map=concept_group_map,
        tune_params=tune_params,
        rerun=rerun,
    ):
        policy_params = {}
        policy_params["include_prior"] = policy_args.get('use_prior', False)
        if policy_name == "random":
            policy_params["group_based"] = policy_args.get(
                'group_level',
                True,
            )
        elif policy_name == "global_val_improvement":
            policy_params['n_concepts'] = n_concepts
            policy_params['val_ds'] = val_dl
            policy_params["group_based"] = policy_args.get(
                'group_level',
                True,
            )
        elif policy_name == "uncertainty":
            policy_params["eps"] = policy_args.get("eps", 1e-8)
            policy_params["group_based"] = policy_args.get(
                'group_level',
                True,
            )
        elif policy_name ==  "global_val_error":
            policy_params["eps"] = policy_args.get("eps", 1e-8)
            policy_params["group_based"] = policy_args.get(
                'group_level',
                True,
            )
            _, _, _, _, _, _, _, val_c_aucs = generate_policy_training_data(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                split=split,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                train_dl=train_dl,
                val_dl=val_dl,
                result_dir=result_dir,
                config=config,
                rerun=rerun,
                accelerator=accelerator,
                devices=devices,
                seed=(42 + split),
            )
            policy_params["val_c_aucs"] = val_c_aucs
        elif policy_name == "coop":
            policy_params["concept_entropy_weight"] = policy_args.get(
                "concept_entropy_weight",
                1,
            )
            policy_params["importance_weight"] = policy_args.get(
                "importance_weight",
                1,
            )
            policy_params["acquisition_weight"] = policy_args.get(
                "acquisition_weight",
                1,
            )
            policy_params["acquisition_costs"] = acquisition_costs
            policy_params["n_tasks"] = n_tasks
            policy_params["eps"] = policy_args.get("eps", 1e-8)
            policy_params["group_based"] = policy_args.get(
                'group_level',
                True,
            )

            # Then also run our hyperparameter search using the validation data,
            # if given
            if tune_params and (val_dl is not None):
                key_name = f'coop'
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
                best_params = fine_tune_coop(
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
                    concept_entropy_weight_range=policy_args.get(
                        'concept_entropy_weight_range',
                        None,
                    ),
                    importance_weight_range=policy_args.get(
                        'importance_weight_range',
                        None,
                    ),
                    acquisition_weight_range=policy_args.get(
                        'acquisition_weight_range',
                        None,
                    ),
                    acquisition_costs=acquisition_costs,
                    group_based=policy_params["group_based"],
                    eps=policy_params["eps"],
                    key_name=key_name,
                    run_name=run_name,
                    coop_variant=concept_selection_policy,
                    rerun=rerun,
                    batch_size=intervention_batch_size,
                    seed=(42 + split),
                    include_prior=policy_params["include_prior"],
                )
                print("Best params found for", policy_name, "are:")
                for param_name, param_value in best_params.items():
                    policy_params[param_name] = param_value
                    print(f"\t{param_name} = {param_value}")
        elif policy_name == "behavioural_cloning":
            policy_params["n_tasks"] = n_tasks
            policy_params["n_concepts"] = n_concepts
            policy_params["group_based"] = policy_args.get(
                'group_level',
                True,
            )
            x_train, y_train, c_train, _, _, _, _, _ = \
                generate_policy_training_data(
                    n_concepts=n_concepts,
                    n_tasks=n_tasks,
                    split=split,
                    imbalance=imbalance,
                    task_class_weights=task_class_weights,
                    train_dl=train_dl,
                    val_dl=val_dl,
                    result_dir=result_dir,
                    config=config,
                    rerun=rerun,
                    accelerator=accelerator,
                    devices=devices,
                    seed=(42 + split),
                )
            policy_params["x_train"] = x_train
            policy_params["y_train"] = y_train
            policy_params["c_train"] = c_train
            policy_params["emb_size"] = (
                config["emb_size"] if config["architecture"] in [
                    "CEM",
                    "ConceptEmbeddingModel",
                    "IntCEM",
                    "IntAwareConceptEmbeddingModel",
                    "H-CEM",
                    "HybridConceptEmbeddingModel",
                ]
                else 1
            )
            policy_params["result_dir"] = result_dir
            policy_params["batch_size"] = policy_args.get('batch_size', 512)
            policy_params["dataset_size"] = policy_args.get(
                'bc_dataset_size',
                5000,
            )
            policy_params["train_epochs"] = policy_args.get(
                'bc_train_epochs',
                100,
            )
            policy_params["seed"] = policy_args.get('seed', 42) + split
            policy_params["full_run_name"] = f"{run_name}_fold_{split + 1}"
            policy_params["rerun"] = rerun

        elif policy_name == "optimal_greedy":
            policy_params["acquisition_costs"] = acquisition_costs
            policy_params["acquisition_weight"] = intervention_config.get(
                "acquisition_weight",
                1,
            )
            policy_params["importance_weight"] = intervention_config.get(
                "importance_weight",
                1,
            )
            policy_params["n_tasks"] = n_tasks
            policy_params["group_based"] = policy_args.get(
                'group_level',
                True,
            )
        else:
            raise ValueError(f'Unsupported policy name "{og_policy_name}"')

        return policy_params
    return _params_fn, concept_selection_policy


def _rerun_policy(
    rerun,
    key_policy_name,
    config,
    split,
    run_name,
):
    if split is not None:
        full_run_name = (
            f"{run_name}_fold_{split + 1}"
        )
    else:
        full_run_name = run_name
    if rerun:
        return True
    if config.get(
        'rerun_interventions',
        os.environ.get(f"RERUN_INTERVENTIONS", "0") == "1"
    ):
        return True
    if (key_policy_name.lower().startswith("coop")) and (
        config.get(
        'rerun_coop_tuning',
        (os.environ.get(f"RERUN_COOP_TUNING", "0") == "1"),
    )
    ):
        return True
    if config.get(
        f'rerun_intervention_{key_policy_name}',
        os.environ.get(f"RERUN_INTERVENTION_{key_policy_name.upper()}", "0") == "1"
    ):
        rerun_list = config.get(
            'rerun_intervention_models',
            os.environ.get(f"RERUN_INTERVENTION_MODELS", ""),
        )
        if rerun_list and isinstance(rerun_list, str):
            rerun_list = rerun_list.split(",")
        if len(rerun_list) == 0:
            # Then we always rerun this guy
            return True
        # Else, check if one of the models we are asking to rerun corresponds to
        # this guy
        for model_to_rerun in rerun_list:
            if model_to_rerun in full_run_name:
                return True
    return False

def test_interventions(
    run_name,
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
    used_policies=None,
    intervention_batch_size=1024,
    competence_levels=[1],
    real_competence_generator=None,
    extra_suffix="",
    real_competence_level="same",
    group_level_competencies=False,
    accelerator="auto",
    devices="auto",
    split=0,
    rerun=False,
    old_results=None,
    task_class_weights=None,
):
    intervention_config = config.get('intervention_config', {})
    used_policies = intervention_config.get(
        'intervention_policies',
        DEFAULT_POLICIES,
    )
    intervention_batch_size = intervention_config.get(
        "intervention_batch_size",
        int(os.environ.get(f"INT_BATCH_SIZE", intervention_batch_size)),
    )
    results = {}
    x_test, y_test, c_test, g_test = data_utils.daloader_to_memory(
        test_dl,
        as_torch=True,
        output_groups=True,
    )

    for competence_level in competence_levels:
        def competence_generator(
            x,
            y,
            c,
            concept_group_map,
        ):
            if group_level_competencies:
                # Then we will operate at a group level, so we need to make
                # sure we distribute competence correctly across different
                # members of a given group!
                if competence_level == "unif":
                    # When using uniform competence, we will assign the same
                    # competencies to all concepts within the same group based
                    # on the cardinallity of the groups (assuming all groups
                    # correspond to mutually exclusive concepts!)
                    batch_group_level_competencies = np.zeros(
                        (c.shape[0], len(concept_group_map))
                    )
                    for batch_idx in range(c.shape[0]):
                        for group_idx, (_, concept_members) in enumerate(
                            concept_map.items()
                        ):
                            batch_group_level_competencies[
                                batch_idx,
                                group_idx,
                            ] = np.random.uniform(1/len(concept_members), 1)
                else:
                    batch_group_level_competencies = np.ones(
                        (c.shape[0], len(concept_group_map))
                    ) * competence_level
                return batch_group_level_competencies

            if competence_level == "unif":
                # Then we will sample from a uniform distribution all concepts
                # regardless of their group!
                return np.random.uniform(
                    0.5,
                    1,
                    size=c.shape,
                )
            # Else we simply assign the same competency to all concepts in
            # all groups!
            return np.ones(c.shape) * competence_level

        if competence_level == 1:
            currently_used_policies = used_policies
        else:
            currently_used_policies = intervention_config.get(
                'incompetence_intervention_policies',
                used_policies,
            )
        for policy_args in currently_used_policies:
            key_policy_name = policy_args["policy"] + "_" + "_".join([
                f'{key}_{policy_args[key]}'
                for key in sorted(policy_args.keys())
                if key != 'policy'
            ])
            if (
                os.environ.get(
                    f"IGNORE_INTERVENTION_{key_policy_name.upper()}",
                    "0"
                ) == "1"
            ):
                continue
            policy_params_fn, concept_selection_policy = get_int_policy(
                policy_args=policy_args,
                config=config,
                n_tasks=n_tasks,
                n_concepts=n_concepts,
                acquisition_costs=acquisition_costs,
                run_name=run_name,
                result_dir=result_dir,
                tune_params=intervention_config.get('tune_params', True),
                concept_group_map=concept_map,
                intervened_groups=intervention_config.get(
                    'tune_intervened_groups',
                    None,
                ),
                val_dl=val_dl,
                train_dl=train_dl,
                accelerator=accelerator,
                devices=devices,
                imbalance=imbalance,
                split=split,
                rerun=_rerun_policy(
                    rerun=rerun,
                    key_policy_name=key_policy_name,
                    config=config,
                    split=split,
                    run_name=run_name,
                ),
                task_class_weights=task_class_weights,
            )
            print(
                f"\tIntervening in {run_name} with policy {key_policy_name} and "
                f"competence {competence_level}"
            )
            if competence_level == 1:
                key = f'test_acc_y_{key_policy_name}_ints'
                int_time_key = f'avg_int_time_{key_policy_name}_ints'
                construction_times_key = (
                    f'construction_time_{key_policy_name}_ints'
                )
            else:
                extra_suffix = (
                    ("_"  + extra_suffix) if extra_suffix else extra_suffix
                )
                if group_level_competencies:
                    key = (
                        f'test_acc_y_{key_policy_name}_ints_co_{competence_level}'
                        f'_gl{extra_suffix}'
                    )
                    int_time_key = (
                        f'avg_int_time_{key_policy_name}_ints_co_{competence_level}'
                        f'_gl{extra_suffix}'
                    )
                    construction_times_key = (
                        f'construction_time_{key_policy_name}_ints_'
                        f'co_{competence_level}_gl_{extra_suffix}'
                    )
                else:
                    key = (
                        f'test_acc_y_{key_policy_name}_ints_co_{competence_level}'
                        f'{extra_suffix}'
                    )
                    int_time_key = (
                        f'avg_int_time_{key_policy_name}_ints_co_{competence_level}'
                        f'{extra_suffix}'
                    )
                    construction_times_key = (
                        f'construction_time_{key_policy_name}_ints_co_{competence_level}'
                        f'{extra_suffix}'
                    )
            dataset_config = config['dataset_config']
            (int_results, avg_time, constr_time), loaded = load_call(
                function=intervene_in_cbm,
                keys=(key, int_time_key, construction_times_key),
                old_results=old_results,
                run_name=run_name,
                rerun=_rerun_policy(
                    rerun=rerun,
                    key_policy_name=key_policy_name,
                    config=config,
                    split=split,
                    run_name=run_name,
                ),
                kwargs=dict(
                    run_name=run_name,
                    concept_selection_policy=concept_selection_policy,
                    policy_params=policy_params_fn,
                    concept_group_map=concept_map,
                    intervened_groups=intervened_groups,
                    accelerator=accelerator,
                    devices=devices,
                    config=config,
                    test_dl=test_dl,
                    train_dl=train_dl,
                    n_tasks=n_tasks,
                    n_concepts=n_concepts,
                    result_dir=result_dir,
                    imbalance=imbalance,
                    split=split,
                    rerun=_rerun_policy(
                        rerun=rerun,
                        key_policy_name=key_policy_name,
                        config=config,
                        split=split,
                        run_name=run_name,
                    ),
                    batch_size=intervention_batch_size,
                    key_name=key,
                    competence_generator=competence_generator,
                    real_competence_generator=real_competence_generator,
                    group_level_competencies=group_level_competencies,
                    x_test=x_test,
                    y_test=y_test,
                    c_test=c_test,
                    g_test=g_test,
                    test_subsampling=dataset_config.get('test_subsampling', 1),
                    seed=(42 + split),
                    task_class_weights=task_class_weights,
                ),
            )
            results[key] = int_results
            results[int_time_key] = avg_time
            results[construction_times_key] = constr_time
            if avg_time:
                extra = (
                    f" (avg int time is {avg_time:.5f}s and construction "
                    f"time is {constr_time:.5f}s)"
                )
            else:
                extra = ""
            for num_groups_intervened, val in enumerate(int_results):
                if n_tasks > 1:
                    logging.info(
                        f"\t\tTest accuracy when intervening "
                        f"with {num_groups_intervened} "
                        f"concept groups with claimed competence "
                        f"{competence_level} and real competence "
                        f"{real_competence_level} is {val * 100:.2f}%{extra}."
                    )
                else:
                    logging.info(
                        f"\t\tTest AUC when intervening "
                        f"with {num_groups_intervened} "
                        f"concept groups with claimed competence "
                        f"{competence_level} and real competence "
                        f"{real_competence_level} is {val * 100:.2f}%{extra}."
                    )
    return results
