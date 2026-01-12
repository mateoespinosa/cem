import copy
import numpy as np
import os
import pytorch_lightning as pl
import torch

from torchvision.models import resnet18, resnet34, resnet50, densenet121

import cem.models.cbm as models_cbm
import cem.models.cbm_bert as models_cbm_bert
import cem.models.cem as models_cem
import cem.models.concept_to_label as models_c2l
import cem.models.glancenet as models_glancenet
import cem.models.intcbm as models_intcbm
import cem.models.mixcem as mixcem
import cem.models.posthoc_cbm as models_pcbm
import cem.models.probcbm as models_probcbm
import cem.models.standard as standard_models
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
    train=False,
):
    task_loss_weight = config.get('task_loss_weight', 1.0)

    if isinstance(config["c_extractor_arch"], str):
        if config["c_extractor_arch"] == "resnet18":
            c_extractor_arch = resnet18
        elif config["c_extractor_arch"] == "resnet34":
            c_extractor_arch = resnet34
        elif config["c_extractor_arch"] == "resnet50":
            c_extractor_arch = resnet50
        elif config["c_extractor_arch"] == "densenet121":
            c_extractor_arch = densenet121
        elif  config["c_extractor_arch"] == "identity":
            c_extractor_arch = "identity"
        elif  config["c_extractor_arch"] == "clip":
            c_extractor_arch = "clip"
        else:
            raise ValueError(f'Invalid model_to_use "{config["model_to_use"]}"')
    else:
        c_extractor_arch = config["c_extractor_arch"]
    if c_extractor_arch != "clip":
        c_extractor_arch = utils.wrap_pretrained_model(c_extractor_arch)

    if config["architecture"] in ["ConceptEmbeddingModel", "CEM"]:
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
            "sim_penalty": config.get("sim_penalty", 0.0),
            "prob_training_thresholding": config.get("prob_training_thresholding", 0),
            "l2_penalty": config.get("l2_penalty", 0.0),
            "emb_pred_loss": config.get("emb_pred_loss", 0.0),
            "cbm_mode": config.get("cbm_mode", False),
        }
        if "embeding_activation" in config:
            # Legacy support for typo in argument
            extra_params["embedding_activation"] = config["embeding_activation"]

    elif config["architecture"] in ["EntangledHybridCBM", "EHCBM"]:
        model_cls = models_cbm.EntangledHybridCBM
        extra_params = {
            "bool": config.get("bool", False),
            "extra_dims": config.get("extra_dims", 0),
            "sigmoidal_prob": config.get("sigmoidal_prob", True),
            "intervention_policy": intervention_policy,
            "bottleneck_nonlinear": config.get("bottleneck_nonlinear", None),
            "active_intervention_values": active_intervention_values,
            "inactive_intervention_values": inactive_intervention_values,
            "x2c_model": x2c_model,
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
            "training_intervention_prob": config.get(
                "training_intervention_prob",
                0.0,
            ),
            "entanglement_mode": config.get("entanglement_mode", "additive"),
            "l2_reg_residual_weights": config.get("l2_reg_residual_weights", 0.0),
            "prior_loss_term": config.get("prior_loss_term", 0.0),
            "prior_only_concepts": config.get("prior_only_concepts", False),
        }

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
        task_loss_weight = config.get('task_loss_weight', 1)
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

    elif config['architecture'] in [
        'MixtureOfConceptEmbeddingsModel',
        'MixCEM',
        'MCIntCEM', # Legacy
    ]:
        task_loss_weight = config.get('task_loss_weight', 1)
        model_cls = mixcem.MixCEM
        extra_params = dict(
            emb_size=config["emb_size"],
            intervention_policy=intervention_policy,
            training_intervention_prob=config.get(
                'training_intervention_prob',
                0.25,
            ),
            embedding_activation=config.get(
                "embedding_activation",
                "leakyrelu",
            ),
            c2y_model=c2y_model,
            c2y_layers=config.get("c2y_layers", []),

            # Experimental/debugging arguments
            intervention_discount=config.get(
                'intervention_discount',
                1,
            ),
            include_only_last_trajectory_loss=config.get(
                'include_only_last_trajectory_loss',
                True,
            ),
            intervention_task_loss_weight=config.get(
                'intervention_task_loss_weight',
                1,
            ),

            ##################################
            # New MixCEM-specific arguments
            #################################
            ood_dropout_prob=config.get(
                'ood_dropout_prob',
                0.5,
            ),
            all_intervened_loss_weight=config.get(
                'all_intervened_loss_weight',
                1,
            ),
            initial_concept_embeddings=config.get(
                'initial_concept_embeddings',
                None,
            ),
            fixed_embeddings=config.get(
                'fixed_embeddings',
                False,
            ),
            temperature=config.get(
                'temperature',
                1,
            ),

            # Monte carlo stuff
            deterministic=config.get(
                'deterministic',
                False,
            ),
            montecarlo_train_tries=config.get(
                'montecarlo_train_tries',
                1,
            ),
            montecarlo_test_tries=config.get(
                'montecarlo_test_tries',
                50,
            ),
            output_uncertainty=config.get(
                'output_uncertainty',
                False,
            ),
            hard_selection_value=config.get(
                'hard_selection_value',
                None,
            ),

            ##################################
            # IntCEM-specific arguments
            #################################

            # Intervention-aware hyperparameters (NO intervention-aware loss
            # is used by default)
            intervention_task_discount=config.get(
                'intervention_task_discount',
                1,
            ),
            intervention_weight=config.get(
                'intervention_weight',
                0,
            ),
            concept_map=config.get(
                'concept_map',
                None,
            ),
            use_concept_groups=config.get(
                'use_concept_groups',
                True,
            ),
            rollout_init_steps=config.get(
                'rollout_init_steps',
                0,
            ),
            int_model_layers=config.get(
                'int_model_layers',
                None,
            ),
            int_model_use_bn=config.get(
                'int_model_use_bn',
                False,
            ),
            num_rollouts=config.get(
                'num_rollouts',
                1,
            ),
            max_horizon=config.get(
                'max_horizon',
                1,
            ),
            initial_horizon=config.get(
                'initial_horizon',
                2,
            ),
            horizon_rate=config.get(
                'horizon_rate',
                1,
            ),
        )

    elif config["architecture"] in [
        "PCBM",
        "PosthocCBM",
        "Post-hocCBM",
        "PosthocConceptBottleneckModel",
        "Post-hocConceptBottleneckModel",
    ]:
        if config.get('pretrained_model', None) is None:
            blackbox_model_config = config["blackbox_model_config"]
            bbox, _ = standard_models.construct_standard_model(
                architecture=blackbox_model_config['name'],
                input_shape=config.get('input_shape'),
                n_labels=n_tasks,
                **blackbox_model_config,
            )
            _, latent_name = standard_models.get_out_layer_name_from_config(
                blackbox_model_config["name"],
                add_linear_layers=blackbox_model_config.get('add_linear_layers'),
            )
            embedding_generator = standard_models.create_feature_extractor(
                bbox,
                return_nodes={latent_name: latent_name},
            )
            config['pretrained_model'] = embedding_generator
        run_name = config["run_name"]
        split = config.get("split", 0)
        if split is not None:
            full_run_name = (
                f"{run_name}_fold_{split + 1}"
            )
        else:
            full_run_name = (
                f"{run_name}"
            )
        result_dir = config.get("result_dir", ".")
        cav_path = os.path.join(
            result_dir,
            f'{full_run_name}_CAVs.npy'
        )
        cav_path = config.get('cav_path', cav_path)
        intercept_path = os.path.join(
            result_dir,
            f'{full_run_name}_Intercepts.npy'
        )
        intercept_path = config.get('intercept_path', intercept_path)
        if (config.get('concept_vectors', None) is None) and (
            os.path.exists(cav_path)
        ):
            config['concept_vectors'] = torch.FloatTensor(np.load(cav_path))


        if (config.get('concept_vector_intercepts', None) is None) and (
            os.path.exists(intercept_path)
        ):
            config['concept_vector_intercepts'] = torch.FloatTensor(
                np.load(intercept_path)
            )

        if not config.get('emb_size'):
            config['emb_size'] = config['concept_vectors'].shape[-1]
        return models_pcbm.PCBM(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            emb_size=config['emb_size'],
            task_class_weights=(
                torch.FloatTensor(task_class_weights)
                if (task_class_weights is not None)
                else None
            ),
            learning_rate=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0),
            optimizer=config['optimizer'],
            top_k_accuracy=config.get('top_k_accuracy'),
            output_latent=output_latent,
            output_interventions=output_interventions,
            concept_vectors=config.get('concept_vectors'),
            concept_vector_intercepts=config.get('concept_vector_intercepts'),
            pretrained_model=config['pretrained_model'],
            c2y_model=config.get('c2y_model'),
            residual=config.get('residual', False),
            residual_model=config.get('residual_model', None),
            reg_strength=config.get('reg_strength', 1e-5),
            l1_ratio=config.get('reg_strength', 0.99),
            lr_scheduler_factor=config.get('lr_scheduler_factor', 0.1),
            lr_scheduler_patience=config.get('lr_scheduler_patience', 10),
            freeze_pretrained_model=config.get('freeze_pretrained_model', True),
            freeze_concept_embeddings=config.get('freeze_concept_embeddings', True),
            training_intervention_prob=config.get(
                'training_intervention_prob',
                0.0,
            ),
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
        )

    elif (
        "BertCBM" in config["architecture"]
    ):
        model_cls = models_cbm_bert.BertCBM
        extra_params = {
            "intervention_policy": intervention_policy,
            "active_intervention_values": active_intervention_values,
            "inactive_intervention_values": inactive_intervention_values,
            "concept_descriptions": config.pop("concept_descriptions", None),
            "d_model": config.get("d_model", 512),
            "n_heads": config.get("n_heads", 4),
            "n_layers": config.get("n_layers", 2),
            "mask_prob": config.get("mask_prob", 0.3),
            "train_steps": config.get("train_steps", 2),
            "infer_steps": config.get("infer_steps", 3),
            "pretrained_clip": config.get("pretrained_clip", "ViT-B-32"),
            "clip_weights": config.get("clip_weights", "laion2b_s34b_b79k"),
            "mask_id": config.get("mask_id", -1),
            "task_mask_prob": config.get('task_mask_prob', None),
            "freeze_backbone": config.get("freeze_backbone", True),
            "unmasked_loss_weight": config.get("unmasked_loss_weight", 0.1),
            "concept_head": config.get("concept_head", "linear"),
            "mask_rate": config.get("mask_rate", None),
            "max_prob": config.get("max_prob", 1),
        }

    elif (
        "ConceptBottleneckModel" in config["architecture"] or
        "CBM" in config["architecture"]
    ):
        model_cls = models_cbm.ConceptBottleneckModel
        extra_params = {
            "bool": config.get("bool", False),
            "extra_dims": config.get("extra_dims", 0),
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
            "training_intervention_prob": config.get(
                "training_intervention_prob",
                0.0,
            ),
            "prior_loss_term": config.get("prior_loss_term", 0.0),
            "prior_only_concepts": config.get("prior_only_concepts", False),
        }

    elif (
        config["architecture"] in ["GlanceNet"]
    ):
        model_cls = models_glancenet.GlanceNet
        if config.get("decoder_output_activation", None) is None:
            decoder_output_activation = None
        elif config.get("decoder_output_activation", None) == "sigmoid":
            decoder_output_activation = torch.nn.Sigmoid()
        else:
            raise ValueError(
                f'Unsupported decoder_output_activation: "'
                f'{decoder_output_activation}"'
            )
        extra_params = {
            "input_shape": config.get('input_shape'),
            "decoder_layers": config.get('decoder_layers', [512, 256, 128, 64, 3]),
            "decoder_output_activation": decoder_output_activation,
            "hidden_dim": config.get('hidden_dim', 64),
            "extra_dims": config.get("extra_dims", 0),
            "beta": config.get('beta', 1),
            "recon_weight": config.get('recon_weight', 1),
            "intervention_policy": intervention_policy,
            "active_intervention_values": active_intervention_values,
            "inactive_intervention_values": inactive_intervention_values,
            "x2c_model": x2c_model,
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
            "conditional_prior": config.get("conditional_prior", True),
            "prior_loss_weight": config.get("prior_loss_weight", 1),
        }

    elif (
        "ConceptToLabelModel" == config["architecture"] or
        "C2L" == config["architecture"]
    ):
        model_cls = models_c2l.ConceptToLabelModel
        extra_params = {
            "intervention_policy": intervention_policy,
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
            "feature_drop_out": config.get('feature_drop_out', 0.25),
            "intervene_concept_vals": config.get(
                'intervene_concept_vals',
                0.5,
            ),
        }

    elif config["architecture"] in [
        "FixedEmbConceptEmbeddingModel",
        "FixedConceptEmbeddingModel",
        "FixedCEM",
    ]:
        model_cls = models_cem.FixedEmbConceptEmbeddingModel
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
            "fixed_embeddings": config.get("fixed_embeddings", True),
            "initial_concept_embeddings": config.get(
                "initial_concept_embeddings",
                None,
            ),
            "fixed_embeddings_always": config.get('fixed_embeddings_always', True),
        }
        if "embeding_activation" in config:
            # Legacy support for typo in argument
            extra_params["embedding_activation"] = config["embeding_activation"]
    else:
        raise ValueError(f'Invalid architecture "{config["architecture"]}"')

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
        concept_loss_weight=config.get('concept_loss_weight', 1),
        task_loss_weight=task_loss_weight,
        learning_rate=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 0),
        c_extractor_arch=c_extractor_arch,
        optimizer=config.get('optimizer', 'sgd'),
        lr_scheduler_factor=config.get('lr_scheduler_factor', 0.1),
        lr_scheduler_patience=config.get('lr_scheduler_patience', 10),
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
        weight_decay=config.get('weight_decay', 0),
        optimizer=config['optimizer'],
        lr_scheduler_factor=config.get('lr_scheduler_factor', 0.1),
        lr_scheduler_patience=config.get('lr_scheduler_patience', 10),
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
        weight_decay=config.get('weight_decay', 0),
        optimizer=config['optimizer'],
        lr_scheduler_factor=config.get('lr_scheduler_factor', 0.1),
        lr_scheduler_patience=config.get('lr_scheduler_patience', 10),
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
        (train_dl is not None) and (
            (
                (
                    config['architecture'] in ["ConceptBottleneckModel", "CBM"]
                ) and
                (
                    not config.get('sigmoidal_prob', True)
                )
            ) or
            (
                config['architecture'] in [
                    "PosthocConceptBottleneckModel",
                    "PCBM",
                    "PosthocCBM",
                ]
            )
        )
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

        if config['architecture'] in [
            "FixedEmbConceptEmbeddingModel",
            "FixedConceptEmbeddingModel",
            "FixedCEM",
        ]:
            # Then the intervention valyes depend on the actual mean embeddings
            # of each concept!
            model.output_embeddings = True
            batch_results = trainer.predict(model, train_dl)
            pos_embs = np.concatenate(
                list(map(lambda x: x[-2], batch_results)),
                axis=0,
            )
            neg_embs = np.concatenate(
                list(map(lambda x: x[-1], batch_results)),
                axis=0,
            )
            active_intervention_values = []
            inactive_intervention_values = []
            out_embs = np.reshape(out_embs, (out_embs.shape[0], n_concepts, -1))
            active_intervention_values = np.mean(
                pos_embs,
                axis=0,
            )
            inactive_intervention_values = np.mean(
                neg_embs,
                axis=0,
            )
        else:

            # Else this is the unbounded concept activation instance
            batch_results = trainer.predict(model, train_dl)
            out_embs = np.concatenate(
                list(map(lambda x: x[1], batch_results)),
                axis=0,
            )
            active_intervention_values = []
            inactive_intervention_values = []
            for idx in range(n_concepts):
                active_intervention_values.append(
                    np.percentile(
                        out_embs[:, idx],
                        config.get('active_top_percentile', 95),
                    )
                )
                inactive_intervention_values.append(
                    np.percentile(
                        out_embs[:, idx],
                        config.get('bottom_top_percentile', 5),
                    )
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
