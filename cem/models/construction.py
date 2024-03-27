import copy
import numpy as np
import os
import pytorch_lightning as pl
import torch

from torchvision.models import resnet18, resnet34, resnet50, densenet121

import cem.models.cbm as models_cbm
import cem.models.cem as models_cem
import cem.models.intcbm as models_intcbm
import cem.models.probcbm as models_probcbm
import cem.train.utils as utils


################################################################################
## HELPER LAYERS
################################################################################


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

################################################################################
## MODEL CONSTRUCTION
################################################################################


def construct_model(
    n_concepts,
    n_tasks,
    config,
    c2y_model=None,
    x2c_model=None,
    imbalance=None,
    task_class_weights=None,
    intervention_policy=None,
    active_intervention_values=None,
    inactive_intervention_values=None,
    output_latent=False,
    output_interventions=False,
):
    task_loss_weight = config.get('task_loss_weight', 1.0)
    if config["architecture"] in ["ConceptEmbeddingModel", "MixtureEmbModel"]:
        model_cls = models_cem.ConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "shared_prob_gen": config.get("shared_prob_gen", True),
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                "leakyrelu"
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
        }
        if "embeding_activation" in config:
            # Legacy support for typo in argument
            extra_params["embedding_activation"] = config["embeding_activation"]
    elif config["architecture"] in [
        "ProbCBM",
        "ProbabilisticConceptBottleneckModel",
        "ProbabilisticCBM",
    ]:
        model_cls = models_probcbm.ProbCBM
        extra_params = dict(
            lr_ratio=config.get(
                'lr_ratio',
                10,
            ),
            hidden_dim=config.get(
                'hidden_dim',
                16,
            ),
            class_hidden_dim=config.get(
                'class_hidden_dim',
                128,
            ),
            intervention_prob=config.get(
                'intervention_prob',
                0.5,
            ),
            use_class_emb_from_concept=config.get(
                'use_class_emb_from_concept',
                False,
            ),
            use_probabilistic_concept=config.get(
                'use_probabilistic_concept',
                True,
            ),
            pretrained=config.get(
                'pretrained',
                True,
            ),
            n_samples_inference=config.get(
                'n_samples_inference',
                50,
            ),
            use_neg_concept=config.get(
                'use_neg_concept',
                True,
            ),
            pred_class=config.get(
                'pred_class',
                True,
            ),
            use_scale=config.get(
                'use_scale',
                True,
            ),
            activation_concept2class=config.get(
                'activation_concept2class',
                'prob',
            ),
            token2concept=config.get(
                'token2concept',
                None,
            ),
            train_class_mode=config.get(
                'train_class_mode',
                'sequential',
            ),
            init_negative_scale=config.get(
                'init_negative_scale',
                5,
            ),
            init_shift=config.get(
                'init_shift',
                5,
            ),
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            use_concept_groups=config.get(
                'use_concept_groups',
                False,
            ),
            vib_beta=config.get(
                'vib_beta',
                0.00005,
            )
        )
    elif config["architecture"] in ["IntAwareConceptBottleneckModel", "IntCBM"]:
        task_loss_weight = config.get('task_loss_weight', 0.0)
        model_cls = models_intcbm.IntAwareConceptBottleneckModel
        extra_params = {
            "bool": config["bool"],
            "extra_dims": config["extra_dims"],
            "sigmoidal_extra_capacity": config.get(
                "sigmoidal_extra_capacity",
                True,
            ),
            "sigmoidal_prob": config.get("sigmoidal_prob", True),
            "intervention_policy": intervention_policy,
            "bottleneck_nonlinear": config.get("bottleneck_nonlinear", None),
            "active_intervention_values": active_intervention_values,
            "inactive_intervention_values": inactive_intervention_values,
            "x2c_model": x2c_model,
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "intervention_weight": config.get("intervention_weight", 5),
            "horizon_rate": config.get("horizon_rate", 1.005),
            "concept_map": config.get("concept_map", None),
            "max_horizon": config.get("max_horizon", 6),
            "include_only_last_trajectory_loss": config.get(
                "include_only_last_trajectory_loss",
                True,
            ),
            "intervention_task_loss_weight": config.get(
                "intervention_task_loss_weight",
                1,
            ),
            "initial_horizon": config.get("initial_horizon", 2),
            "use_concept_groups": config.get("use_concept_groups", True),
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                config.get("intervention_task_discount", 1.1),
            ),
            "rollout_init_steps": config.get('rollout_init_steps', 0),
            "int_model_layers": config.get("int_model_layers", None),
            "int_model_use_bn": config.get("int_model_use_bn", True),
            "num_rollouts": config.get("num_rollouts", 1),
        }
    elif config["architecture"] in ["IntAwareConceptEmbeddingModel", "IntCEM"]:
        task_loss_weight = config.get('task_loss_weight', 0.0)
        model_cls = models_intcbm.IntAwareConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                "leakyrelu",
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "intervention_weight": config.get("intervention_weight", 5),
            "horizon_rate": config.get("horizon_rate", 1.005),
            "concept_map": config.get("concept_map", None),
            "max_horizon": config.get("max_horizon", 6),
            "include_only_last_trajectory_loss": config.get(
                "include_only_last_trajectory_loss",
                True,
            ),
            "intervention_task_loss_weight": config.get(
                "intervention_task_loss_weight",
                1,
            ),
            "initial_horizon": config.get("initial_horizon", 2),
            "use_concept_groups": config.get("use_concept_groups", False),
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                config.get("intervention_task_discount", 1.1),
            ),
            "rollout_init_steps": config.get('rollout_init_steps', 0),
            "int_model_layers": config.get("int_model_layers", None),
            "int_model_use_bn": config.get("int_model_use_bn", False),
            "num_rollouts": config.get("num_rollouts", 1),
        }
    elif "ConceptBottleneckModel" in config["architecture"]:
        model_cls = models_cbm.ConceptBottleneckModel
        extra_params = {
            "bool": config["bool"],
            "extra_dims": config["extra_dims"],
            "sigmoidal_extra_capacity": config.get(
                "sigmoidal_extra_capacity",
                True,
            ),
            "sigmoidal_prob": config.get("sigmoidal_prob", True),
            "intervention_policy": intervention_policy,
            "bottleneck_nonlinear": config.get("bottleneck_nonlinear", None),
            "active_intervention_values": active_intervention_values,
            "inactive_intervention_values": inactive_intervention_values,
            "x2c_model": x2c_model,
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
        }
    else:
        raise ValueError(f'Invalid architecture "{config["architecture"]}"')

    if isinstance(config["c_extractor_arch"], str):
        if config["c_extractor_arch"] == "resnet18":
            c_extractor_arch = resnet18
        elif config["c_extractor_arch"] == "resnet34":
            c_extractor_arch = resnet34
        elif config["c_extractor_arch"] == "resnet50":
            c_extractor_arch = resnet50
        elif config["c_extractor_arch"] == "densenet121":
            c_extractor_arch = densenet121
        else:
            raise ValueError(f'Invalid model_to_use "{config["model_to_use"]}"')
    else:
        c_extractor_arch = config["c_extractor_arch"]

    # Create model
    return model_cls(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        weight_loss=(
            torch.FloatTensor(imbalance)
            if config['weight_loss'] and (imbalance is not None)
            else None
        ),
        task_class_weights=(
            torch.FloatTensor(task_class_weights)
            if (task_class_weights is not None)
            else None
        ),
        concept_loss_weight=config['concept_loss_weight'],
        task_loss_weight=task_loss_weight,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        c_extractor_arch=utils.wrap_pretrained_model(c_extractor_arch),
        optimizer=config['optimizer'],
        top_k_accuracy=config.get('top_k_accuracy'),
        output_latent=output_latent,
        output_interventions=output_interventions,
        **extra_params,
    )


def construct_sequential_models(
    n_concepts,
    n_tasks,
    config,
    imbalance=None,
    task_class_weights=None,
):
    assert config.get('extra_dims', 0) == 0, (
        "We can only train sequential/joint models if the extra "
        "dimensions are 0!"
    )
    if isinstance(config["c_extractor_arch"], str):
        if config["c_extractor_arch"] == "resnet18":
            c_extractor_arch = resnet18
        elif config["c_extractor_arch"] == "resnet34":
            c_extractor_arch = resnet34
        elif config["c_extractor_arch"] == "resnet50":
            c_extractor_arch = resnet50
        elif config["c_extractor_arch"] == "densenet121":
            c_extractor_arch = densenet121
        else:
            raise ValueError(
                f'Invalid model_to_use "{config["model_to_use"]}"'
            )
    else:
        c_extractor_arch = config["c_extractor_arch"]
    # Else we assume that it is a callable function which we will
    # need to instantiate here
    try:
        x2c_model = c_extractor_arch(
            pretrained=config.get('pretrain_model', True),
        )
        if c_extractor_arch == densenet121:
            x2c_model.classifier = torch.nn.Linear(1024, n_concepts)
        elif hasattr(x2c_model, 'fc'):
            x2c_model.fc = torch.nn.Linear(512, n_concepts)
    except Exception as e:
        x2c_model = c_extractor_arch(output_dim=n_concepts)
    x2c_model = utils.WrapperModule(
        n_tasks=n_concepts,
        model=x2c_model,
        weight_loss=(
            torch.FloatTensor(imbalance)
            if config['weight_loss'] and (imbalance is not None)
            else None
        ),
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        optimizer=config['optimizer'],
        binary_output=True,
        sigmoidal_output=True,
    )

    # Now construct the label prediction model
    # Else we construct it here directly
    c2y_layers = config.get('c2y_layers', [])
    units = [n_concepts] + (c2y_layers or []) + [n_tasks]
    layers = [
        torch.nn.Linear(units[i-1], units[i])
        for i in range(1, len(units))
    ]
    c2y_model = utils.WrapperModule(
        n_tasks=n_tasks,
        model=torch.nn.Sequential(*layers),
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        optimizer=config['optimizer'],
        top_k_accuracy=config.get('top_k_accuracy'),
        binary_output=False,
        sigmoidal_output=False,
        weight_loss=(
            torch.FloatTensor(task_class_weights)
            if (task_class_weights is not None)
            else None
        ),
    )
    return x2c_model, c2y_model


################################################################################
## MODEL LOADING
################################################################################


def load_trained_model(
    config,
    n_tasks,
    result_dir,
    n_concepts,
    split=0,
    imbalance=None,
    task_class_weights=None,
    train_dl=None,
    logger=False,
    accelerator="auto",
    devices="auto",
    intervention_policy=None,
    intervene=False,
    output_latent=False,
    output_interventions=False,
    enable_checkpointing=False,
):
    if "run_name" in config:
        run_name = config["run_name"]
    else:
        run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}"
        )
    if split is not None:
        full_run_name = (
            f"{run_name}_fold_{split + 1}"
        )
    else:
        full_run_name = run_name
    independent = False
    sequential = False
    if config['architecture'].startswith("Sequential"):
        sequential = True
    elif config['architecture'].startswith("Independent"):
        independent = True
    model_saved_path = os.path.join(
        result_dir or ".",
        f'{full_run_name}.pt'
    )

    if (
        ((intervention_policy is not None) or intervene) and
        (train_dl is not None) and
        (config['architecture'] == "ConceptBottleneckModel") and
        (not config.get('sigmoidal_prob', True))
    ):
        # Then let's look at the empirical distribution of the logits in order
        # to be able to intervene
        model = construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            output_latent=output_latent,
            output_interventions=output_interventions,
        )
        model.load_state_dict(torch.load(model_saved_path))
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            enable_checkpointing=enable_checkpointing,
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

        active_intervention_values = torch.FloatTensor(
            active_intervention_values
        )
        inactive_intervention_values = torch.FloatTensor(
            inactive_intervention_values
        )
    else:
        active_intervention_values = inactive_intervention_values = None
    if independent or sequential:
        _, c2y_model = construct_sequential_models(
            n_concepts,
            n_tasks,
            config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
        )


        # As well as the wrapper CBM model we will use for serialization
        # and testing
        # We will be a bit cheeky and use the model with the task loss
        # weight set to 0 for training with the same dataset
        model_config = copy.deepcopy(config)
        model_config['concept_loss_weight'] = 1
        model_config['task_loss_weight'] = 0
        base_model = construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=model_config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_latent=output_latent,
            output_interventions=output_interventions,
        )
        model = construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=model_config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_latent=output_latent,
            output_interventions=output_interventions,
            x2c_model=base_model.x2c_model,
            c2y_model=c2y_model,
        )


    else:
        model = construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_latent=output_latent,
            output_interventions=output_interventions,
        )

    model.load_state_dict(torch.load(model_saved_path))
    return model
