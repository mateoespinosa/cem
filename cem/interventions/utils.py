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

from cem.train.utils import load_call
from cem.models.construction import load_trained_model
from cem.interventions.random import IndependentRandomMaskIntPolicy
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


################################################################################
## Global Variables
################################################################################


MAX_COMB_BOUND = 500000
POLICY_NAMES = [
    "intcem_policy",
    "group_random",
    "group_random_no_prior",
    "group_coop_no_prior",
    "behavioural_cloning_no_prior",
    "group_uncertainty_no_prior",
    "optimal_greedy_no_prior",
    "global_val_error_no_prior",
    "global_val_improvement_no_prior",
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

def concepts_from_competencies(c, competencies):
    correct_interventions = np.random.binomial(
        n=1,
        p=competencies,
        size=c.shape,
    )
    return (
        c * correct_interventions + (1 - c) * (1 - correct_interventions)
    ).type(torch.FloatTensor)

def _default_competence_generator(
    x,
    y,
    c,
    concept_group_map,
):
    return np.ones(c.shape)

################################################################################
## MAIN INTERVENTION FUNCTION
################################################################################

def adversarial_intervene_in_cbm(
    config,
    test_dl,
    n_tasks,
    n_concepts,
    result_dir,
    imbalance=None,
    task_class_weights=None,
    train_dl=None,
    sequential=False,
    independent=False,
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
        config=config,
        test_dl=test_dl,
        n_tasks=n_tasks,
        n_concepts=n_concepts,
        result_dir=result_dir,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
        competence_generator=competence_generator,
        train_dl=train_dl,
        sequential=sequential,
        independent=independent,
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
        seed=seed,
    )

def intervene_in_cbm(
    config,
    test_dl,
    n_tasks,
    n_concepts,
    result_dir,
    imbalance=None,
    task_class_weights=None,
    competence_generator=_default_competence_generator,
    train_dl=None,
    sequential=False,
    independent=False,
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
    seed=None,
):
    if seed is not None:
        seed_everything(seed)
    if batch_size is not None:
        # Then overwrite the config's batch size
        try:
            test_dl = torch.utils.data.DataLoader(
                dataset=test_dl.dataset,
                batch_size=batch_size,
                num_workers=test_dl.num_workers,
            )
        except:
            import pdb
            pdb.set_trace()
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
                key_name + f"avg_int_time_fold_{split}.npy",
            )
            if os.path.exists(total_time_file):
                avg_time = np.load(total_time_file)
                avg_time = avg_time[0]
            else:
                avg_time = 0

            construct_time_file = os.path.join(
                result_dir,
                key_name + f"construct_time_fold_{split}.npy",
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
        sequential=sequential,
        independent=independent,
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
        (c_test is None)
    ):
        if hasattr(test_dl.dataset, 'tensors'):
            x_test, y_test, c_test = test_dl.dataset.tensors
        else:
            x_test, y_test, c_test = [], [], []
            for data in test_dl:
                if len(data) == 2:
                    x, (y, c) = data
                else:
                    (x, y, c) = data
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
    c_test = concepts_from_competencies(c_test, competencies_test)
    competencies_test = torch.FloatTensor(competencies_test)
    test_dl = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(
            x_test,
            y_test,
            c_test,
            competencies_test,
        ),
        batch_size=test_dl.batch_size,
        num_workers=test_dl.num_workers,
    )
    prev_num_groups_intervened = 0
    avg_times = []
    total_times = []
    logging.debug(
        f"Intervention groups: {groups}"
    )
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
        time_diff = time.time() - start_time
        avg_times.append(
            (time_diff)/(
                x_test.shape[0] * (coeff if coeff != 0 else 1)
            )
        )
        total_times.append(time_diff)
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
        if model.intervention_policy.greedy:
            prev_num_groups_intervened = num_groups_intervened
        else:
            prev_num_groups_intervened = 0
        test_dl = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                x_test,
                y_test,
                c_test,
                competencies_test,
                torch.IntTensor(prev_interventions),
            ),
            batch_size=test_dl.batch_size,
            num_workers=test_dl.num_workers,
        )
    avg_time = np.mean(avg_times)
    total_time = np.sum(total_times)
    logging.debug(
        f"\tAverage intervention took {avg_time:.5f} seconds and "
        f"construction took {construct_time:.5f} seconds.\n"
        f"\tIn total interventions took {total_time:.5f} seconds"
    )
    if key_name:
        result_file = os.path.join(
            result_dir,
            key_name + f"_fold_{split}.npy",
        )
        np.save(result_file, intervention_accs)

        result_file = os.path.join(
            result_dir,
            key_name + f"avg_int_time_fold_{split}.npy",
        )
        np.save(result_file, np.array([avg_time]))

        result_file = os.path.join(
            result_dir,
            key_name + f"construct_time_fold_{split}.npy",
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
    sequential=False,
    independent=False,
    accelerator="auto",
    devices="auto",
    rerun=False,
    batch_size=None,
    include_prior=False,
    seed=None,
):
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
        sequential=sequential,
        independent=independent,
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
                f"fold_{split}.joblib"
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
                f"fold_{split}.joblib"
            ),
        )
        joblib.dump(best_params, result_file)

        grid_search_results_file = os.path.join(
            result_dir,
            (
                f"coop_grid_search{'_' + key_name if key_name else key_name}_"
                f"fold_{split}.joblib"
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
    sequential=False,
    independent=False,
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
        sequential=sequential,
        independent=independent,
    )
    batch_size = batch_size or train_dl.batch_size
    x_train, y_train, c_train = [], [], []
    for ds_data in train_dl:
        if len(ds_data) == 2:
            x, (y, c) = ds_data
        else:
            (x, y, c) = ds_data
        x_train.append(x)
        y_train.append(y)
        c_train.append(c)
    x_train = torch.FloatTensor(np.concatenate(x_train, axis=0))
    y_train = torch.FloatTensor(np.concatenate(y_train, axis=0))
    c_train = torch.FloatTensor(np.concatenate(c_train, axis=0))
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
        val_c_true = []
        for data in val_dl:
            if len(data) == 2:
                x, (y, c) = data
            else:
                (x, y, c) = data
            val_c_true.append(c)
        val_c_true = np.concatenate(val_c_true, axis=0)
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
    accelerator="auto",
    devices="auto",
    intervention_batch_size=1024,
):
    intervention_batch_size = config.get(
        "intervention_batch_size",
        int(os.environ.get(f"INT_BATCH_SIZE", intervention_batch_size)),
    )
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
            policy_params["importance_weight"] = config.get(
                "importance_weight",
                1,
            )
            policy_params["acquisition_weight"] = config.get(
                "acquisition_weight",
                1,
            )
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
                sequential=sequential,
                independent=independent,
                rerun=rerun,
                accelerator=accelerator,
                devices=devices,
                seed=(42 + split),
            )
            policy_params["val_c_aucs"] = val_c_aucs
        elif "coop" in policy_name:
            policy_params["concept_entropy_weight"] = config.get(
                "concept_entropy_weight",
                1,
            )
            policy_params["importance_weight"] = config.get(
                "importance_weight",
                1,
            )
            policy_params["acquisition_weight"] = config.get(
                "acquisition_weight",
                1,
            )
            policy_params["acquisition_costs"] = acquisition_costs
            policy_params["n_tasks"] = n_tasks
            policy_params["eps"] = config.get("eps", 1e-8)
            policy_params["group_based"] = (
                not ("individual" in policy_name)
            )
            if "competence" in policy_name:
                tune_params = False

            # Then also run our hyperparameter search using the validation data,
            # if given
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
                    batch_size=intervention_batch_size,
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
                    sequential=sequential,
                    independent=independent,
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
                    "IntAwareConceptEmbeddingModel",
                    "IntCEM",
                    "ACFlowConceptEmbeddingModel",
                    "ACFCEM",
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
            policy_params["acquisition_weight"] = config.get(
                "acquisition_weight",
                1,
            )
            policy_params["importance_weight"] = config.get(
                "importance_weight",
                1,
            )
            policy_params["n_tasks"] = n_tasks
            policy_params["group_based"] = not (
                "individual" in policy_name
            )
        elif "optimal_global" in policy_name:
            policy_params["acquisition_costs"] = acquisition_costs
            policy_params["acquisition_weight"] = config.get(
                "acquisition_weight",
                1,
            )
            policy_params["importance_weight"] = config.get(
                "importance_weight",
                1,
            )
            policy_params["group_based"] = not (
                "individual" in policy_name
            )
            policy_params["n_tasks"] = n_tasks
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
    if config.get(
        'rerun_interventions',
        os.environ.get(f"RERUN_INTERVENTIONS", "0") == "1"
    ):
        return True
    if "coop" in policy_name.lower() and (
        config.get(
        'rerun_coop_tuning',
        (os.environ.get(f"RERUN_COOP_TUNING", "0") == "1"),
    )
    ):
        return True
    if config.get(
        f'rerun_intervention_{policy_name}',
        os.environ.get(f"RERUN_INTERVENTION_{policy_name.upper()}", "0") == "1"
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
    used_policies=None,
    intervention_batch_size=1024,
    competence_levels=[1],
    accelerator="auto",
    devices="auto",
    split=0,
    rerun=False,
    sequential=False,
    independent=False,
    old_results=None,
    task_class_weights=None,
):
    used_policies = config.get('intervention_policies', POLICY_NAMES)
    intervention_batch_size = config.get(
        "intervention_batch_size",
        int(os.environ.get(f"INT_BATCH_SIZE", intervention_batch_size)),
    )
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
                # When using uniform competence, we will assign the same
                # competence level to the same batch index
                # The same competence is assigned to all concepts within the same
                # group
                np.random.seed(42)
                batch_group_level_competencies = np.random.uniform(
                    0.5,
                    1,
                    size=(c.shape[0], len(concept_group_map)),
                )
                batch_concept_level_competencies = np.ones(
                    (c.shape[0], c.shape[1])
                )
                for group_idx, (_, group_concepts) in enumerate(
                    concept_group_map.items()
                ):
                    batch_concept_level_competencies[:, group_concepts] = \
                        np.expand_dims(
                            batch_group_level_competencies[:, group_idx],
                            axis=-1,
                        )
                return batch_concept_level_competencies
            return np.ones(c.shape) * competence_level
        if competence_level == 1:
            currently_used_policies = used_policies
        else:
            currently_used_policies = config.get(
                'incompetence_intervention_policies',
                used_policies,
            )
        for policy in currently_used_policies:
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
                logging.debug(
                    f"effective number of concepts is {eff_n_concepts} and "
                    f"intervened groups are {intervened_groups}"
                    f"used intervened groups are {used_intervened_groups}"
                )
            else:
                used_intervened_groups = intervened_groups
            policy_params_fn, concept_selection_policy = get_int_policy(
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
                accelerator=accelerator,
                devices=devices,
                imbalance=imbalance,
                split=split,
                rerun=_rerun_policy(rerun, policy, config, split),
                sequential=sequential,
                independent=independent,
                task_class_weights=task_class_weights,
            )
            print(
                f"\tIntervening in {full_run_name} with policy {policy} and "
                f"competence {competence_level}"
            )
            if competence_level == 1:
                key = f'test_acc_y_{policy}_ints_{full_run_name}'
                int_time_key = f'avg_int_time_{policy}_ints_{full_run_name}'
                construction_times_key = (
                    f'construction_time_{policy}_ints_{full_run_name}'
                )
            else:
                key = (
                    f'test_acc_y_{policy}_ints_co_{competence_level}_'
                    f'{full_run_name}'
                )
                int_time_key = (
                    f'avg_int_time_{policy}_ints_co_{competence_level}_'
                    f'{full_run_name}'
                )
                construction_times_key = (
                    f'construction_time_{policy}_ints_co_{competence_level}_'
                    f'{full_run_name}'
                )
            (int_results, avg_time, constr_time), loaded = load_call(
                function=intervene_in_cbm,
                keys=(key, int_time_key, construction_times_key),
                old_results=old_results,
                full_run_name=full_run_name,
                rerun=_rerun_policy(rerun, policy, config, split),
                kwargs=dict(
                    concept_selection_policy=concept_selection_policy,
                    policy_params=policy_params_fn,
                    concept_group_map=concept_map,
                    intervened_groups=used_intervened_groups,
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
            if avg_time:
                extra = (
                    f" (avg int time is {avg_time:.5f}s and construction "
                    f"time is {constr_time:.5f}s)"
                )
            else:
                extra = ""
            for num_groups_intervened, val in enumerate(int_results):
                if n_tasks > 1:
                    logging.debug(
                        f"\t\tTest accuracy when intervening "
                        f"with {num_groups_intervened} "
                        f"concept groups is {val * 100:.2f}%{extra}."
                    )
                else:
                    logging.debug(
                        f"\t\tTest AUC when intervening "
                        f"with {num_groups_intervened} "
                        f"concept groups is {val * 100:.2f}%{extra}."
                    )
    return results
