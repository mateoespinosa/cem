import os
import numpy as np
import torch
import pytorch_lightning as pl
from collections import defaultdict
import cem.train.training as cem_train
import logging
import io
from contextlib import redirect_stdout


# Set of CUB attributes selected by original CBM paper
CUB_SEL_ATTRS = [
    1,
    4,
    6,
    7,
    10,
    14,
    15,
    20,
    21,
    23,
    25,
    29,
    30,
    35,
    36,
    38,
    40,
    44,
    45,
    50,
    51,
    53,
    54,
    56,
    57,
    59,
    63,
    64,
    69,
    70,
    72,
    75,
    80,
    84,
    90,
    91,
    93,
    99,
    101,
    106,
    110,
    111,
    116,
    117,
    119,
    125,
    126,
    131,
    132,
    134,
    145,
    149,
    151,
    152,
    153,
    157,
    158,
    163,
    164,
    168,
    172,
    178,
    179,
    181,
    183,
    187,
    188,
    193,
    194,
    196,
    198,
    202,
    203,
    208,
    209,
    211,
    212,
    213,
    218,
    220,
    221,
    225,
    235,
    236,
    238,
    239,
    240,
    242,
    243,
    244,
    249,
    253,
    254,
    259,
    260,
    262,
    268,
    274,
    277,
    283,
    289,
    292,
    293,
    294,
    298,
    299,
    304,
    305,
    308,
    309,
    310,
    311,
]

# Names of all CUB attributes
CUB_ATTRS = [
    "has_bill_shape::curved_(up_or_down)",
    "has_bill_shape::dagger",
    "has_bill_shape::hooked",
    "has_bill_shape::needle",
    "has_bill_shape::hooked_seabird",
    "has_bill_shape::spatulate",
    "has_bill_shape::all-purpose",
    "has_bill_shape::cone",
    "has_bill_shape::specialized",
    "has_wing_color::blue",
    "has_wing_color::brown",
    "has_wing_color::iridescent",
    "has_wing_color::purple",
    "has_wing_color::rufous",
    "has_wing_color::grey",
    "has_wing_color::yellow",
    "has_wing_color::olive",
    "has_wing_color::green",
    "has_wing_color::pink",
    "has_wing_color::orange",
    "has_wing_color::black",
    "has_wing_color::white",
    "has_wing_color::red",
    "has_wing_color::buff",
    "has_upperparts_color::blue",
    "has_upperparts_color::brown",
    "has_upperparts_color::iridescent",
    "has_upperparts_color::purple",
    "has_upperparts_color::rufous",
    "has_upperparts_color::grey",
    "has_upperparts_color::yellow",
    "has_upperparts_color::olive",
    "has_upperparts_color::green",
    "has_upperparts_color::pink",
    "has_upperparts_color::orange",
    "has_upperparts_color::black",
    "has_upperparts_color::white",
    "has_upperparts_color::red",
    "has_upperparts_color::buff",
    "has_underparts_color::blue",
    "has_underparts_color::brown",
    "has_underparts_color::iridescent",
    "has_underparts_color::purple",
    "has_underparts_color::rufous",
    "has_underparts_color::grey",
    "has_underparts_color::yellow",
    "has_underparts_color::olive",
    "has_underparts_color::green",
    "has_underparts_color::pink",
    "has_underparts_color::orange",
    "has_underparts_color::black",
    "has_underparts_color::white",
    "has_underparts_color::red",
    "has_underparts_color::buff",
    "has_breast_pattern::solid",
    "has_breast_pattern::spotted",
    "has_breast_pattern::striped",
    "has_breast_pattern::multi-colored",
    "has_back_color::blue",
    "has_back_color::brown",
    "has_back_color::iridescent",
    "has_back_color::purple",
    "has_back_color::rufous",
    "has_back_color::grey",
    "has_back_color::yellow",
    "has_back_color::olive",
    "has_back_color::green",
    "has_back_color::pink",
    "has_back_color::orange",
    "has_back_color::black",
    "has_back_color::white",
    "has_back_color::red",
    "has_back_color::buff",
    "has_tail_shape::forked_tail",
    "has_tail_shape::rounded_tail",
    "has_tail_shape::notched_tail",
    "has_tail_shape::fan-shaped_tail",
    "has_tail_shape::pointed_tail",
    "has_tail_shape::squared_tail",
    "has_upper_tail_color::blue",
    "has_upper_tail_color::brown",
    "has_upper_tail_color::iridescent",
    "has_upper_tail_color::purple",
    "has_upper_tail_color::rufous",
    "has_upper_tail_color::grey",
    "has_upper_tail_color::yellow",
    "has_upper_tail_color::olive",
    "has_upper_tail_color::green",
    "has_upper_tail_color::pink",
    "has_upper_tail_color::orange",
    "has_upper_tail_color::black",
    "has_upper_tail_color::white",
    "has_upper_tail_color::red",
    "has_upper_tail_color::buff",
    "has_head_pattern::spotted",
    "has_head_pattern::malar",
    "has_head_pattern::crested",
    "has_head_pattern::masked",
    "has_head_pattern::unique_pattern",
    "has_head_pattern::eyebrow",
    "has_head_pattern::eyering",
    "has_head_pattern::plain",
    "has_head_pattern::eyeline",
    "has_head_pattern::striped",
    "has_head_pattern::capped",
    "has_breast_color::blue",
    "has_breast_color::brown",
    "has_breast_color::iridescent",
    "has_breast_color::purple",
    "has_breast_color::rufous",
    "has_breast_color::grey",
    "has_breast_color::yellow",
    "has_breast_color::olive",
    "has_breast_color::green",
    "has_breast_color::pink",
    "has_breast_color::orange",
    "has_breast_color::black",
    "has_breast_color::white",
    "has_breast_color::red",
    "has_breast_color::buff",
    "has_throat_color::blue",
    "has_throat_color::brown",
    "has_throat_color::iridescent",
    "has_throat_color::purple",
    "has_throat_color::rufous",
    "has_throat_color::grey",
    "has_throat_color::yellow",
    "has_throat_color::olive",
    "has_throat_color::green",
    "has_throat_color::pink",
    "has_throat_color::orange",
    "has_throat_color::black",
    "has_throat_color::white",
    "has_throat_color::red",
    "has_throat_color::buff",
    "has_eye_color::blue",
    "has_eye_color::brown",
    "has_eye_color::purple",
    "has_eye_color::rufous",
    "has_eye_color::grey",
    "has_eye_color::yellow",
    "has_eye_color::olive",
    "has_eye_color::green",
    "has_eye_color::pink",
    "has_eye_color::orange",
    "has_eye_color::black",
    "has_eye_color::white",
    "has_eye_color::red",
    "has_eye_color::buff",
    "has_bill_length::about_the_same_as_head",
    "has_bill_length::longer_than_head",
    "has_bill_length::shorter_than_head",
    "has_forehead_color::blue",
    "has_forehead_color::brown",
    "has_forehead_color::iridescent",
    "has_forehead_color::purple",
    "has_forehead_color::rufous",
    "has_forehead_color::grey",
    "has_forehead_color::yellow",
    "has_forehead_color::olive",
    "has_forehead_color::green",
    "has_forehead_color::pink",
    "has_forehead_color::orange",
    "has_forehead_color::black",
    "has_forehead_color::white",
    "has_forehead_color::red",
    "has_forehead_color::buff",
    "has_under_tail_color::blue",
    "has_under_tail_color::brown",
    "has_under_tail_color::iridescent",
    "has_under_tail_color::purple",
    "has_under_tail_color::rufous",
    "has_under_tail_color::grey",
    "has_under_tail_color::yellow",
    "has_under_tail_color::olive",
    "has_under_tail_color::green",
    "has_under_tail_color::pink",
    "has_under_tail_color::orange",
    "has_under_tail_color::black",
    "has_under_tail_color::white",
    "has_under_tail_color::red",
    "has_under_tail_color::buff",
    "has_nape_color::blue",
    "has_nape_color::brown",
    "has_nape_color::iridescent",
    "has_nape_color::purple",
    "has_nape_color::rufous",
    "has_nape_color::grey",
    "has_nape_color::yellow",
    "has_nape_color::olive",
    "has_nape_color::green",
    "has_nape_color::pink",
    "has_nape_color::orange",
    "has_nape_color::black",
    "has_nape_color::white",
    "has_nape_color::red",
    "has_nape_color::buff",
    "has_belly_color::blue",
    "has_belly_color::brown",
    "has_belly_color::iridescent",
    "has_belly_color::purple",
    "has_belly_color::rufous",
    "has_belly_color::grey",
    "has_belly_color::yellow",
    "has_belly_color::olive",
    "has_belly_color::green",
    "has_belly_color::pink",
    "has_belly_color::orange",
    "has_belly_color::black",
    "has_belly_color::white",
    "has_belly_color::red",
    "has_belly_color::buff",
    "has_wing_shape::rounded-wings",
    "has_wing_shape::pointed-wings",
    "has_wing_shape::broad-wings",
    "has_wing_shape::tapered-wings",
    "has_wing_shape::long-wings",
    "has_size::large_(16_-_32_in)",
    "has_size::small_(5_-_9_in)",
    "has_size::very_large_(32_-_72_in)",
    "has_size::medium_(9_-_16_in)",
    "has_size::very_small_(3_-_5_in)",
    "has_shape::upright-perching_water-like",
    "has_shape::chicken-like-marsh",
    "has_shape::long-legged-like",
    "has_shape::duck-like",
    "has_shape::owl-like",
    "has_shape::gull-like",
    "has_shape::hummingbird-like",
    "has_shape::pigeon-like",
    "has_shape::tree-clinging-like",
    "has_shape::hawk-like",
    "has_shape::sandpiper-like",
    "has_shape::upland-ground-like",
    "has_shape::swallow-like",
    "has_shape::perching-like",
    "has_back_pattern::solid",
    "has_back_pattern::spotted",
    "has_back_pattern::striped",
    "has_back_pattern::multi-colored",
    "has_tail_pattern::solid",
    "has_tail_pattern::spotted",
    "has_tail_pattern::striped",
    "has_tail_pattern::multi-colored",
    "has_belly_pattern::solid",
    "has_belly_pattern::spotted",
    "has_belly_pattern::striped",
    "has_belly_pattern::multi-colored",
    "has_primary_color::blue",
    "has_primary_color::brown",
    "has_primary_color::iridescent",
    "has_primary_color::purple",
    "has_primary_color::rufous",
    "has_primary_color::grey",
    "has_primary_color::yellow",
    "has_primary_color::olive",
    "has_primary_color::green",
    "has_primary_color::pink",
    "has_primary_color::orange",
    "has_primary_color::black",
    "has_primary_color::white",
    "has_primary_color::red",
    "has_primary_color::buff",
    "has_leg_color::blue",
    "has_leg_color::brown",
    "has_leg_color::iridescent",
    "has_leg_color::purple",
    "has_leg_color::rufous",
    "has_leg_color::grey",
    "has_leg_color::yellow",
    "has_leg_color::olive",
    "has_leg_color::green",
    "has_leg_color::pink",
    "has_leg_color::orange",
    "has_leg_color::black",
    "has_leg_color::white",
    "has_leg_color::red",
    "has_leg_color::buff",
    "has_bill_color::blue",
    "has_bill_color::brown",
    "has_bill_color::iridescent",
    "has_bill_color::purple",
    "has_bill_color::rufous",
    "has_bill_color::grey",
    "has_bill_color::yellow",
    "has_bill_color::olive",
    "has_bill_color::green",
    "has_bill_color::pink",
    "has_bill_color::orange",
    "has_bill_color::black",
    "has_bill_color::white",
    "has_bill_color::red",
    "has_bill_color::buff",
    "has_crown_color::blue",
    "has_crown_color::brown",
    "has_crown_color::iridescent",
    "has_crown_color::purple",
    "has_crown_color::rufous",
    "has_crown_color::grey",
    "has_crown_color::yellow",
    "has_crown_color::olive",
    "has_crown_color::green",
    "has_crown_color::pink",
    "has_crown_color::orange",
    "has_crown_color::black",
    "has_crown_color::white",
    "has_crown_color::red",
    "has_crown_color::buff",
    "has_wing_pattern::solid",
    "has_wing_pattern::spotted",
    "has_wing_pattern::striped",
    "has_wing_pattern::multi-colored",
]

# Generate a mapping containing all concept groups in CUB generated
# using a simple prefix tree
CUB_CONCEPT_GROUP_MAP = defaultdict(list)
for i, concept_name in enumerate(list(np.array(CUB_ATTRS)[CUB_SEL_ATTRS])):
    group = concept_name[:concept_name.find("::")]
    CUB_CONCEPT_GROUP_MAP[group].append(i)


################################################################################
## HELPER FUNCTIONS
################################################################################

def load_trained_model(
    config,
    n_tasks,
    result_dir,
    n_concepts,
    split=0,
    imbalance=None,
    intervention_idxs=None,
    adversarial_intervention=False,
    train_dl=None,
    sequential=False,
    independent=False,
    gpu=int(torch.cuda.is_available()),
):
    arch_name = config['c_extractor_arch']
    if not isinstance(arch_name, str):
        arch_name = "lambda"
    if split is not None:
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}_"
            f"{arch_name}_fold_{split + 1}"
        )
    else:
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}_"
            f"{arch_name}"
        )
    selected_concepts = np.arange(n_concepts)
    if sequential:
        extra = "Sequential"
    elif independent:
        extra = "Independent"
    else:
        extra = ""
    model_saved_path = os.path.join(
        result_dir or ".",
        f'{extra}{full_run_name}.pt'
    )

    if (
        (intervention_idxs is not None) and
        (train_dl is not None) and
        (config['architecture'] == "ConceptBottleneckModel") and
        (not config.get('sigmoidal_prob', True))
    ):
        # Then let's look at the empirical distribution of the logits in order
        # to be able to intervene
        model = cem_train.construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            imbalance=imbalance,
        )
        model.load_state_dict(torch.load(model_saved_path))
        trainer = pl.Trainer(
            gpus=gpu,
        )
        batch_results = trainer.predict(model, train_dl)
        out_embs = np.concatenate(
            list(map(lambda x: x[1], batch_results)),
            axis=0,
        )
        active_intervention_values = []
        inactive_intervention_values = []
        for idx in range(n_concepts):
            active_intervention_values.append(
                np.percentile(out_embs[:, idx], 95)
            )
            inactive_intervention_values.append(
                np.percentile(out_embs[:, idx], 5)
            )
        if gpu:
            active_intervention_values = torch.cuda.FloatTensor(
                active_intervention_values
            )
            inactive_intervention_values = torch.cuda.FloatTensor(
                inactive_intervention_values
            )
        else:
            active_intervention_values = torch.FloatTensor(
                active_intervention_values
            )
            inactive_intervention_values = torch.FloatTensor(
                inactive_intervention_values
            )
    else:
        active_intervention_values = inactive_intervention_values = None

    if sequential:
        _, c2y_model = cem_train.construct_sequential_models(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            imbalance=imbalance,
        )
    elif independent:
        _, c2y_model = cem_train.construct_sequential_models(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            imbalance=imbalance,
        )
    else:
        c2y_model = None
    model = cem_train.construct_model(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        config=config,
        imbalance=imbalance,
        intervention_idxs=intervention_idxs,
        adversarial_intervention=adversarial_intervention,
        active_intervention_values=active_intervention_values,
        inactive_intervention_values=inactive_intervention_values,
        c2y_model=c2y_model,
    )

    model.load_state_dict(torch.load(model_saved_path))
    return model


################################################################################
## CONCEPT INTERVENTION SELECTION POLICIES
################################################################################

def random_int_policy(num_groups_intervened, concept_group_map, config=None):
    selected_groups_for_trial = np.random.choice(
        list(concept_group_map.keys()),
        size=num_groups_intervened,
        replace=False,
    )
    intervention_idxs = []
    for selected_group in selected_groups_for_trial:
        intervention_idxs.extend(concept_group_map[selected_group])
    return sorted(intervention_idxs)


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
    adversarial_intervention=False,
    train_dl=None,
    sequential=False,
    independent=False,
    concept_group_map=None,
    intervened_groups=None,
    gpu=int(torch.cuda.is_available()),
    split=0,
    concept_selection_policy=random_int_policy,
    rerun=False,
    old_results=None,
):
    intervention_accs = []
    # If no concept groups are given, then we assume that all concepts
    # represent a unitary group themselves
    concept_group_map = concept_group_map or dict(
        [(i, [i]) for i in range(n_concepts)]
    )
    groups = intervened_groups or list(range(0, len(concept_group_map) + 1, 1))

    if (not rerun) and (old_results is not None):
        for j, num_groups_intervened in enumerate(groups):
            print(
                f"\tTest accuracy when intervening with {num_groups_intervened} "
                f"concept groups is {old_results[j] * 100:.2f}%."
            )
        return old_results

    for j, num_groups_intervened in enumerate(groups):
        logging.debug(
            f"Intervening with {num_groups_intervened} out of "
            f"{len(concept_group_map)} concept groups"
        )
        n_trials = config.get('intervention_trials', 3)
        avg = []
        for trial in range(n_trials):
            intervention_idxs = concept_selection_policy(
                num_groups_intervened=num_groups_intervened,
                concept_group_map=concept_group_map,
                config=config,
            )
            logging.debug(
                f"\tFor trial {trial + 1}/{n_trials} for split {split} with "
                f"{num_groups_intervened} groups intervened"
            )
            logging.debug(
                f"\t\tThis results in concepts {intervention_idxs} being "
                f"modified"
            )
            config["shared_prob_gen"] = config.get("shared_prob_gen", False)
            config["per_concept_weight"] = config.get(
                "per_concept_weight",
                False,
            )
            model = load_trained_model(
                config=config,
                n_tasks=n_tasks,
                n_concepts=n_concepts,
                result_dir=result_dir,
                split=split,
                imbalance=imbalance,
                intervention_idxs=intervention_idxs,
                train_dl=train_dl,
                sequential=sequential,
                independent=independent,
            )
            trainer = pl.Trainer(
                gpus=gpu,
            )
            f = io.StringIO()
            with redirect_stdout(f):
                [test_results] = trainer.test(model, test_dl, verbose=False,)
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
    return intervention_accs
