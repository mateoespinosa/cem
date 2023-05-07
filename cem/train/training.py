import copy
import joblib
import numpy as np
import os
import pytorch_lightning as pl
import torch
import logging

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchvision.models import resnet18, resnet34, resnet50, densenet121
import multiprocessing

import cem.models.cem as models_cem
import cem.models.cbm as models_cbm
import cem.models.intcbm as models_intcbm
import cem.train.utils as utils
import cem.metrics.homogeneity as homogeneity
import cem.metrics.oracle as oracle
import cem.metrics.niching as niching

def _save_result(fun, kwargs, output_filepath):
    result = fun(**kwargs)
    joblib.dump(result, output_filepath)
    return result

def _execute_and_save(
    fun,
    kwargs,
    result_dir,
    filename,
    rerun=False,
):
    output_filepath = os.path.join(
        result_dir,
        filename,
    )
    if (not rerun) and os.path.exists(output_filepath):
        return joblib.load(output_filepath)
    context = multiprocessing.get_context('spawn')
    p = context.Process(
        target=_save_result,
        kwargs=dict(
            fun=fun,
            kwargs=kwargs,
            output_filepath=output_filepath,
        ),
    )
    p.start()
    p.join()
    if p.exitcode:
        raise ValueError(
            f'Subprocess failed!'
        )
    p.kill()
    return joblib.load(output_filepath)

def load_call(
    function,
    keys,
    full_run_name,
    old_results=None,
    rerun=False,
    kwargs=None,
):
    old_results = old_results or {}
    kwargs = kwargs or {}
    if not isinstance(keys, (tuple, list)):
        keys = [keys]

    outputs = []
    for key in keys:
        if key.endswith("_" + full_run_name):
            real_key = key[:len(full_run_name) + 1]
            search_key = key
        else:
            real_key = key
            search_key = key + "_" + full_run_name
        rerun = rerun or (
            os.environ.get(f"RERUN_METRIC_{real_key.upper()}", "0") == "1"
        )
        if search_key in old_results:
            outputs.append(old_results[search_key])
        else:
            rerun = True
            logging.debug(
                f"Restarting run because we could not find {search_key} in old results."
            )
            break
    if not rerun:
        return outputs, True

    return function(**kwargs), False


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
            "embedding_activation": config.get("embedding_activation", "leakyrelu"),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
            "include_certainty": config.get("include_certainty", True),
        }
        if "embeding_activation" in config:
            # Legacy support for typo in argument
            extra_params["embedding_activation"] = config["embeding_activation"]
    elif config["architecture"] in ["IntAwareConceptBottleneckModel", "IntCBM"]:
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

            "intervention_discount": config.get("intervention_discount", 0.9),
            "intervention_weight": config.get("intervention_weight", 5),
            "horizon_rate": config.get("horizon_rate", 1.005),
            "average_trajectory": config.get("average_trajectory", True),
            "concept_map": config.get("concept_map", None),
            "tau": config.get("tau", 1),
            "max_horizon": config.get("max_horizon", 5),
            "include_task_trajectory_loss": config.get("include_task_trajectory_loss", False),
            "horizon_binary_representation": config.get("horizon_binary_representation", False),
            "include_only_last_trajectory_loss": config.get("include_only_last_trajectory_loss", False),
            "intervention_task_loss_weight": config.get("intervention_task_loss_weight", 1),
            "initial_horizon": config.get("initial_horizon", 1),
            "use_concept_groups": config.get("use_concept_groups", False),
            "horizon_uniform_distr": config.get("horizon_uniform_distr", True),
            "beta_a": config.get("beta_a", 1),
            "beta_b": config.get("beta_b", 3),
            "intervention_task_discount": config.get("intervention_task_discount", config.get("intervention_discount", 0.9)),
            "use_horizon": config.get("use_horizon", True),
            "rollout_init_steps": config.get('rollout_init_steps', 0),
            "use_full_mask_distr": config.get("use_full_mask_distr", False),
            "int_model_layers": config.get("int_model_layers", None),
            "int_model_use_bn": config.get("int_model_use_bn", False),
            "initialize_discount": config.get("initialize_discount", False),
            "include_probs": config.get("include_probs", False),
            "propagate_target_gradients": config.get("propagate_target_gradients", False),
            "num_rollouts": config.get("num_rollouts", 1),
            "legacy_mode": config.get("legacy_mode", False),
            "include_certainty": config.get("include_certainty", True),
        }
    elif config["architecture"] in ["IntAwareConceptEmbeddingModel", "IntCEM"]:
        model_cls = models_intcbm.IntAwareConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get("embedding_activation", "leakyrelu"),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "intervention_discount": config.get("intervention_discount", 0.9),
            "intervention_weight": config.get("intervention_weight", 5),
            "horizon_rate": config.get("horizon_rate", 1.005),
            "average_trajectory": config.get("average_trajectory", True),
            "concept_map": config.get("concept_map", None),
            "tau": config.get("tau", 1),
            "max_horizon": config.get("max_horizon", 5),
            "include_task_trajectory_loss": config.get("include_task_trajectory_loss", False),
            "horizon_binary_representation": config.get("horizon_binary_representation", False),
            "include_only_last_trajectory_loss": config.get("include_only_last_trajectory_loss", False),
            "intervention_task_loss_weight": config.get("intervention_task_loss_weight", 1),
            "initial_horizon": config.get("initial_horizon", 1),
            "use_concept_groups": config.get("use_concept_groups", False),
            "horizon_uniform_distr": config.get("horizon_uniform_distr", True),
            "beta_a": config.get("beta_a", 1),
            "beta_b": config.get("beta_b", 3),
            "intervention_task_discount": config.get("intervention_task_discount",config.get("intervention_discount", 0.9)),
            "use_horizon": config.get("use_horizon", True),
            "rollout_init_steps": config.get('rollout_init_steps', 0),
            "use_full_mask_distr": config.get("use_full_mask_distr", False),
            "int_model_layers": config.get("int_model_layers", None),
            "int_model_use_bn": config.get("int_model_use_bn", False),
            "initialize_discount": config.get("initialize_discount", False),
            "include_probs": config.get("include_probs", False),
            "propagate_target_gradients": config.get("propagate_target_gradients", False),
            "num_rollouts": config.get("num_rollouts", 1),
            "legacy_mode": config.get("legacy_mode", False),
            "include_certainty": config.get("include_certainty", True),
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
            "include_certainty": config.get("include_certainty", True),
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
        task_loss_weight=config.get('task_loss_weight', 1.0),
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
    sequential=False,
    logger=False,
    independent=False,
    gpu=int(torch.cuda.is_available()),
    intervention_policy=None,
    intervene=False,
    output_latent=False,
    output_interventions=False,
    enable_checkpointing=False,
):
    arch_name = config.get('c_extractor_arch', "")
    if not isinstance(arch_name, str):
        arch_name = "lambda"
    key_full_run_name = (
        f"{config['architecture']}{config.get('extra_name', '')}"
    )
    if split is not None:
        full_run_name = (
            f"{key_full_run_name}_{arch_name}_fold_{split + 1}"
        )
    else:
        full_run_name = (
            f"{key_full_run_name}_{arch_name}"
        )
    selected_concepts = np.arange(n_concepts)
    if sequential and not (config['architecture'].startswith("Sequential")):
        extra = "Sequential"
    elif independent and not (config['architecture'].startswith("Independent")):
        extra = "Independent"
    else:
        extra = ""
    model_saved_path = os.path.join(
        result_dir or ".",
        f'{extra}{full_run_name}.pt'
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
            gpus=gpu,
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



################################################################################
## MODEL TRAINING
################################################################################

def train_model(
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl,
    result_dir=None,
    test_dl=None,
    split=None,
    imbalance=None,
    task_class_weights=None,
    rerun=False,
    logger=False,
    project_name='',
    seed=None,
    save_model=True,
    activation_freq=0,
    single_frequency_epochs=0,
    gpu=int(torch.cuda.is_available()),
    gradient_clip_val=0,
    old_results=None,
    enable_checkpointing=False,
):
    if config['architecture'] in [
        "SequentialConceptBottleneckModel",
        "IndependentConceptBottleneckModel",
    ]:
        return train_independent_and_sequential_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            result_dir=result_dir,
            test_dl=test_dl,
            split=split,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            rerun=rerun,
            logger=logger,
            project_name=project_name,
            seed=seed,
            save_model=save_model,
            activation_freq=activation_freq,
            single_frequency_epochs=single_frequency_epochs,
            enable_checkpointing=enable_checkpointing,
            independent=("Independent" in config['architecture']),
        )
    if seed is not None:
        seed_everything(seed)

    extr_name = config['c_extractor_arch']
    if not isinstance(extr_name, str):
        extr_name = "lambda"
    key_full_run_name = (
        f"{config['architecture']}{config.get('extra_name', '')}"
    )
    if split is not None:
        full_run_name = (
            f"{key_full_run_name}_{extr_name}_fold_{split + 1}"
        )
    else:
        full_run_name = (
            f"{key_full_run_name}_{extr_name}"
        )
    print(f"[Training {full_run_name}]")
    print("config:")
    for key, val in config.items():
        print(f"\t{key} -> {val}")

    # create model
    model = construct_model(
        n_concepts,
        n_tasks,
        config,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
    )
    print(
        "[Number of parameters in model",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        "]"
    )
    print(
        "[Number of non-trainable parameters in model",
        sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "]",
    )
    if config.get("model_pretrain_path"):
        if os.path.exists(config.get("model_pretrain_path")):
            # Then we simply load the model and proceed
            print("\tFound pretrained model to load the initial weights from!")
            model.load_state_dict(torch.load(config.get("model_pretrain_path")), strict=False)

    if (project_name) and result_dir and (
        not os.path.exists(os.path.join(result_dir, f'{full_run_name}.pt'))
    ):
        # Lazy import to avoid importing unless necessary
        import wandb
        with wandb.init(
            project=project_name,
            name=full_run_name,
            config=config,
            reinit=True
        ) as run:
            model_saved_path = os.path.join(
                result_dir,
                f'{full_run_name}.pt'
            )
            trainer = pl.Trainer(
                gpus=gpu,
                max_epochs=config['max_epochs'],
                check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                callbacks=[
                    EarlyStopping(
                        monitor=config["early_stopping_monitor"],
                        min_delta=config.get("early_stopping_delta", 0.00),
                        patience=config['patience'],
                        verbose=config.get("verbose", False),
                        mode=config["early_stopping_mode"],
                    ),
                ],
                enable_checkpointing=enable_checkpointing,
                gradient_clip_val=gradient_clip_val,
#                 track_grad_norm=2,
                # Only use the wandb logger when it is a fresh run
                logger=(
                    logger or
                    (WandbLogger(
                        name=full_run_name,
                        project=project_name,
                        save_dir=os.path.join(result_dir, "logs"),
                    ) if rerun or (not os.path.exists(model_saved_path)) else False)
                ),
            )
            if activation_freq:
                fit_trainer = utils.ActivationMonitorWrapper(
                    model=model,
                    trainer=trainer,
                    activation_freq=activation_freq,
                    single_frequency_epochs=single_frequency_epochs,
                    output_dir=os.path.join(
                        result_dir,
                        f"test_embedding_acts/{full_run_name}",
                    ),
                    # YES, we pass the validation data intentionally to avoid
                    # explosion of memory
                    # usage
                    test_dl=val_dl,
                )
            else:
                fit_trainer = trainer
            if (not rerun) and os.path.exists(model_saved_path):
                # Then we simply load the model and proceed
                print("\tFound cached model... loading it")
                model.load_state_dict(torch.load(model_saved_path))
            else:
                # Else it is time to train it
                fit_trainer.fit(model, train_dl, val_dl)
                config_copy = copy.deepcopy(config)
                if "c_extractor_arch" in config_copy and (
                    not isinstance(config_copy["c_extractor_arch"], str)
                ):
                    del config_copy["c_extractor_arch"]
                joblib.dump(
                    config_copy,
                    os.path.join(
                        result_dir,
                        f'{full_run_name}_experiment_config.joblib',
                    ),
                )
                if save_model:
                    torch.save(
                        model.state_dict(),
                        model_saved_path,
                    )
            # freeze model and compute test accuracy
            if test_dl is not None:
                model.freeze()
                def _inner_call():
                    [test_results] = trainer.test(model, test_dl)
                    output = [
                        test_results["test_c_accuracy"],
                        test_results["test_y_accuracy"],
                        test_results["test_c_auc"],
                        test_results["test_y_auc"],
                        test_results["test_c_f1"],
                        test_results["test_y_f1"],
                    ]
                    top_k_vals = []
                    for key, val in test_results.items():
                        if "test_y_top" in key:
                            top_k = int(key[len("test_y_top_"):-len("_accuracy")])
                            top_k_vals.append((top_k, val))
                    output += list(map(
                        lambda x: x[1],
                        sorted(top_k_vals, key=lambda x: x[0]),
                    ))
                    return output

                keys = [
                    "test_acc_c",
                    "test_acc_y",
                    "test_auc_c",
                    "test_auc_y",
                    "test_f1_c",
                    "test_f1_y",
                ]
                if 'top_k_accuracy' in config:
                    top_k_args = config['top_k_accuracy']
                    if top_k_args is None:
                        top_k_args = []
                    if not isinstance(top_k_args, list):
                        top_k_args = [top_k_args]
                    for top_k in sorted(top_k_args):
                        keys.append(f'test_top_{top_k}_acc_y')
                values, _ = load_call(
                    function=_inner_call,
                    keys=keys,
                    full_run_name=key_full_run_name,
                    old_results=old_results,
                    rerun=rerun,
                    kwargs={},
                )
                test_results = {
                    key: val
                    for (key, val) in zip(keys, values)
                }
                print(
                    f'c_acc: {test_results["test_acc_c"]*100:.2f}%, '
                    f'y_acc: {test_results["test_acc_y"]*100:.2f}%, '
                    f'c_auc: {test_results["test_auc_c"]*100:.2f}%, '
                    f'y_auc: {test_results["test_auc_y"]*100:.2f}%'
                )
            else:
                test_results = None
    else:
        callbacks = [
            EarlyStopping(
                monitor=config["early_stopping_monitor"],
                min_delta=config.get("early_stopping_delta", 0.00),
                patience=config['patience'],
                verbose=config.get("verbose", False),
                mode=config["early_stopping_mode"],
            ),
        ]

        trainer = pl.Trainer(
            gpus=gpu,
            max_epochs=config['max_epochs'],
            check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
            callbacks=callbacks,
            logger=logger or False,
            gradient_clip_val=gradient_clip_val,
            enable_checkpointing=enable_checkpointing,
        )

        if result_dir:
            if activation_freq:
                fit_trainer = utils.ActivationMonitorWrapper(
                    model=model,
                    trainer=trainer,
                    activation_freq=activation_freq,
                    single_frequency_epochs=single_frequency_epochs,
                    output_dir=os.path.join(
                        result_dir,
                        f"test_embedding_acts/{full_run_name}",
                    ),
                    # YES, we pass the validation data intentionally to avoid
                    # explosion of memory usage
                    test_dl=val_dl,
                )
            else:
                fit_trainer = trainer
        else:
            fit_trainer = trainer

        # Else it is time to train it
        model_saved_path = os.path.join(
            result_dir or ".",
            f'{full_run_name}.pt'
        )
        if (not rerun) and os.path.exists(model_saved_path):
            # Then we simply load the model and proceed
            print("\tFound cached model... loading it")
            model.load_state_dict(torch.load(model_saved_path))
        else:
            # Else it is time to train it
            fit_trainer.fit(model, train_dl, val_dl)
            if save_model and (result_dir is not None):
                torch.save(
                    model.state_dict(),
                    model_saved_path,
                )

        if not os.path.exists(os.path.join(
            result_dir,
            f'{full_run_name}_experiment_config.joblib'
        )):
            # Then let's serialize the experiment config for this run
            config_copy = copy.deepcopy(config)
            if "c_extractor_arch" in config_copy and (
                not isinstance(config_copy["c_extractor_arch"], str)
            ):
                del config_copy["c_extractor_arch"]
            joblib.dump(config_copy, os.path.join(
                result_dir,
                f'{full_run_name}_experiment_config.joblib'
            ))
        if test_dl is not None:
            model.freeze()
            def _inner_call():
                [test_results] = trainer.test(model, test_dl)
                output = [
                    test_results["test_c_accuracy"],
                    test_results["test_y_accuracy"],
                    test_results["test_c_auc"],
                    test_results["test_y_auc"],
                    test_results["test_c_f1"],
                    test_results["test_y_f1"],
                ]
                top_k_vals = []
                for key, val in test_results.items():
                    if "test_y_top" in key:
                        top_k = int(key[len("test_y_top_"):-len("_accuracy")])
                        top_k_vals.append((top_k, val))
                output += list(map(
                    lambda x: x[1],
                    sorted(top_k_vals, key=lambda x: x[0]),
                ))
                return output

            keys = [
                "test_acc_c",
                "test_acc_y",
                "test_auc_c",
                "test_auc_y",
                "test_f1_c",
                "test_f1_y",
            ]
            if 'top_k_accuracy' in config:
                top_k_args = config['top_k_accuracy']
                if top_k_args is None:
                    top_k_args = []
                if not isinstance(top_k_args, list):
                    top_k_args = [top_k_args]
                for top_k in sorted(top_k_args):
                    keys.append(f'test_top_{top_k}_acc_y')
            values, _ = load_call(
                function=_inner_call,
                keys=keys,
                full_run_name=key_full_run_name,
                old_results=old_results,
                rerun=rerun,
                kwargs={},
            )
            test_results = {
                key: val
                for (key, val) in zip(keys, values)
            }
            print(
                f'c_acc: {test_results["test_acc_c"]*100:.2f}%, '
                f'y_acc: {test_results["test_acc_y"]*100:.2f}%, '
                f'c_auc: {test_results["test_auc_c"]*100:.2f}%, '
                f'y_auc: {test_results["test_auc_y"]*100:.2f}%'
            )
        else:
            test_results = None
    return model, test_results


def train_independent_and_sequential_model(
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl,
    result_dir=None,
    test_dl=None,
    split=None,
    imbalance=None,
    task_class_weights=None,
    rerun=False,
    logger=False,
    project_name='cub_concept_training',
    seed=None,
    save_model=True,
    activation_freq=0,
    single_frequency_epochs=0,
    gpu=int(torch.cuda.is_available()),
    ind_old_results=None,
    seq_old_results=None,
    enable_checkpointing=False,
):
    if seed is not None:
        seed_everything(seed)

    extr_name = config['c_extractor_arch']
    if not isinstance(extr_name, str):
        extr_name = "lambda"
    if split is not None:
        ind_full_run_name = (
            f"IndependentConceptBottleneckModel"
            f"{config.get('extra_name', '')}_{extr_name}_fold_{split + 1}"
        )
        seq_full_run_name = (
            f"SequentialConceptBottleneckModel"
            f"{config.get('extra_name', '')}_{extr_name}_fold_{split + 1}"
        )
    else:
        ind_full_run_name = (
            f"IndependentConceptBottleneckModel"
            f"{config.get('extra_name', '')}_{extr_name}"
        )
        seq_full_run_name = (
            f"SequentialConceptBottleneckModel"
            f"{config.get('extra_name', '')}_{extr_name}"
        )
    print(f"[Training {ind_full_run_name} and {seq_full_run_name}]")
    print("config:")
    for key, val in config.items():
        print(f"\t{key} -> {val}")

    # Create the two models we will manipulate
    # Else, let's construct the two models we will need for this
    _, ind_c2y_model = construct_sequential_models(
        n_concepts,
        n_tasks,
        config,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
    )

    _, seq_c2y_model = construct_sequential_models(
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
    model = construct_model(
        n_concepts,
        n_tasks,
        config=model_config,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
    )
    print(
        "[Number of parameters in model",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        "]",
    )
    print(
        "[Number of non-trainable parameters in model",
        sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "]",
    )
    seq_model_saved_path = os.path.join(
        result_dir,
        f'{seq_full_run_name}.pt'
    )
    ind_model_saved_path = os.path.join(
        result_dir,
        f'{ind_full_run_name}.pt'
    )
    chpt_exists = (
        os.path.exists(ind_model_saved_path) and
        os.path.exists(seq_model_saved_path)
    )
    # Construct the datasets we will need for training if the model
    # has not been found
    if rerun or (not chpt_exists):
        x_train = []
        y_train = []
        c_train = []
        for elems in train_dl:
            if len(elems) == 2:
                (x, (y, c)) = elems
            else:
                (x, y, c) = elems
            x_train.append(x.cpu().detach())
            y_train.append(y.cpu().detach())
            c_train.append(c.cpu().detach())
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        c_train = np.concatenate(c_train, axis=0)

        if test_dl:
            x_test = []
            y_test = []
            c_test = []
            for elems in test_dl:
                if len(elems) == 2:
                    (x, (y, c)) = elems
                else:
                    (x, y, c) = elems
                x_test.append(x.cpu().detach())
                y_test.append(y.cpu().detach())
                c_test.append(c.cpu().detach())
            x_test = np.concatenate(x_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)
            c_test = np.concatenate(c_test, axis=0)
        if val_dl is not None:
            x_val = []
            y_val = []
            c_val = []
            for elems in val_dl:
                if len(elems) == 2:
                    (x, (y, c)) = elems
                else:
                    (x, y, c) = elems
                x_val.append(x.cpu().detach())
                y_val.append(y.cpu().detach())
                c_val.append(c.cpu().detach())
            x_val = np.concatenate(x_val, axis=0)
            y_val = np.concatenate(y_val, axis=0)
            c_val = np.concatenate(c_val, axis=0)
        else:
            c2y_val_dl = None


    if (project_name) and result_dir and (not chpt_exists):
        # Lazy import to avoid importing unless necessary
        import wandb
        enter_obj = wandb.init(
            project=project_name,
            name=ind_full_run_name,
            config=config,
            reinit=True
        )
    else:
        enter_obj = utils.EmptyEnter()
    with enter_obj as run:
        trainer = pl.Trainer(
            gpus=gpu,
            # We will distribute half epochs in one model and half on the other
            max_epochs=config['max_epochs'],
            check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
            callbacks=[
                EarlyStopping(
                    monitor=config["early_stopping_monitor"],
                    min_delta=config.get("early_stopping_delta", 0.00),
                    patience=config['patience'],
                    verbose=config.get("verbose", False),
                    mode=config["early_stopping_mode"],
                ),
            ],
            # Only use the wandb logger when it is a fresh run
            logger=(
                logger or
                (WandbLogger(
                    name=ind_full_run_name,
                    project=project_name,
                    save_dir=os.path.join(result_dir, "logs"),
                ) if project_name and (rerun or (not chpt_exists)) else False)
            ),
        )
        if activation_freq:
            raise ValueError(
                "Activation drop has not yet been tested for "
                "joint/sequential models!"
            )
        else:
            x2c_trainer = trainer
        if (not rerun) and chpt_exists:
            # Then we simply load the model and proceed
            print("\tFound cached model... loading it")
            ind_model = construct_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                x2c_model=model.x2c_model,
                c2y_model=ind_c2y_model,
            )
            ind_model.load_state_dict(torch.load(ind_model_saved_path))

            seq_model = construct_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                x2c_model=model.x2c_model,
                c2y_model=seq_c2y_model,
            )
            seq_model.load_state_dict(torch.load(seq_model_saved_path))
        else:
            # First train the input to concept model
            print("[Training input to concept model]")
            x2c_trainer.fit(model, train_dl, val_dl)
            if val_dl is not None:
                print(
                    "Validation results for x2c model:",
                    x2c_trainer.test(model, val_dl),
                )

            # Time to construct intermediate dataset for independent model!
            print(
                "[Constructing dataset for independent concept to label model]"
            )
            ind_c2y_train_dl = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.from_numpy(
                        c_train
                    ),
                    torch.from_numpy(y_train),
                ),
                shuffle=True,
                batch_size=config['batch_size'],
                num_workers=config.get('num_workers', 5),
            )
            if val_dl is not None:
                ind_c2y_val_dl = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(
                        torch.from_numpy(
                            c_val
                        ),
                        torch.from_numpy(y_val),
                    ),
                    batch_size=config['batch_size'],
                    num_workers=config.get('num_workers', 5),
                )
            else:
                ind_c2y_val_dl = None

            print(
                "[Constructing dataset for sequential concept to label model]"
            )
            train_batch_concepts = trainer.predict(
                model,
                torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(
                        torch.from_numpy(x_train),
                        torch.from_numpy(y_train),
                        torch.from_numpy(c_train),
                    ),
                    batch_size=1,
                    num_workers=config.get('num_workers', 5),
                ),
            )
            train_complete_concepts = np.concatenate(
                list(map(lambda x: x[1], train_batch_concepts)),
                axis=0,
            )
            seq_c2y_train_dl = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.from_numpy(
                        train_complete_concepts
                    ),
                    torch.from_numpy(y_train),
                ),
                shuffle=True,
                batch_size=config['batch_size'],
                num_workers=config.get('num_workers', 5),
            )

            if val_dl is not None:
                val_batch_concepts = trainer.predict(
                    model,
                    torch.utils.data.DataLoader(
                        torch.utils.data.TensorDataset(
                            torch.from_numpy(x_val),
                            torch.from_numpy(y_val),
                            torch.from_numpy(c_val),
                        ),
                        batch_size=1,
                        num_workers=config.get('num_workers', 5),
                    ),
                )
                val_complete_concepts = np.concatenate(
                    list(map(lambda x: x[1], val_batch_concepts)),
                    axis=0,
                )
                seq_c2y_val_dl = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(
                        torch.from_numpy(
                            val_complete_concepts
                        ),
                        torch.from_numpy(y_val),
                    ),
                    batch_size=config['batch_size'],
                    num_workers=config.get('num_workers', 5),
                )
            else:
                seq_c2y_val_dl = None

            # Train the independent concept to label model
            print("[Training independent concept to label model]")
            ind_c2y_trainer = pl.Trainer(
                gpus=gpu,
                # We will distribute half epochs in one model and half on the
                # other
                max_epochs=config.get('c2y_max_epochs', 50),
                enable_checkpointing=enable_checkpointing,
                check_val_every_n_epoch=config.get(
                    "check_val_every_n_epoch",
                    5,
                ),
                callbacks=[
                    EarlyStopping(
                        monitor=config["early_stopping_monitor"],
                        min_delta=config.get("early_stopping_delta", 0.00),
                        patience=config['patience'],
                        verbose=config.get("verbose", False),
                        mode=config["early_stopping_mode"],
                    ),
                ],
                # Only use the wandb logger when it is a fresh run
                logger=(
                    logger or
                    (WandbLogger(
                        name=ind_full_run_name,
                        project=project_name,
                        save_dir=os.path.join(result_dir, "logs"),
                    ) if project_name and (rerun or (not chpt_exists)) else False)
                ),
            )
            ind_c2y_trainer.fit(
                ind_c2y_model,
                ind_c2y_train_dl,
                ind_c2y_val_dl,
            )
            if ind_c2y_val_dl is not None:
                print(
                    "Independent validation results for c2y model:",
                    ind_c2y_trainer.test(ind_c2y_model, ind_c2y_val_dl),
                )

            # Train the sequential concept to label model
            print("[Training sequential concept to label model]")
            seq_c2y_trainer = pl.Trainer(
                gpus=gpu,
                # We will distribute half epochs in one model and half on the
                # other
                max_epochs=config.get('c2y_max_epochs', 50),
                enable_checkpointing=enable_checkpointing,
                check_val_every_n_epoch=config.get(
                    "check_val_every_n_epoch",
                    5,
                ),
                callbacks=[
                    EarlyStopping(
                        monitor=config["early_stopping_monitor"],
                        min_delta=config.get("early_stopping_delta", 0.00),
                        patience=config['patience'],
                        verbose=config.get("verbose", False),
                        mode=config["early_stopping_mode"],
                    ),
                ],
                # Only use the wandb logger when it is a fresh run
                logger=(
                    logger or
                    (WandbLogger(
                        name=seq_full_run_name,
                        project=project_name,
                        save_dir=os.path.join(result_dir, "logs"),
                    ) if project_name and (rerun or (not chpt_exists)) else False)
                ),
            )
            seq_c2y_trainer.fit(
                seq_c2y_model,
                seq_c2y_train_dl,
                seq_c2y_val_dl,
            )
            if seq_c2y_val_dl is not None:
                print(
                    "Sequential validation results for c2y model:",
                    seq_c2y_trainer.test(seq_c2y_model, seq_c2y_val_dl),
                )

            # Dump the config file
            config_copy = copy.deepcopy(config)
            if "c_extractor_arch" in config_copy and (
                not isinstance(config_copy["c_extractor_arch"], str)
            ):
                del config_copy["c_extractor_arch"]
            joblib.dump(
                config_copy,
                os.path.join(
                    result_dir,
                    f'{ind_full_run_name}_experiment_config.joblib',
                ),
            )
            joblib.dump(
                config_copy,
                os.path.join(
                    result_dir,
                    f'{seq_full_run_name}_experiment_config.joblib',
                ),
            )

            # And serialize the end models
            ind_model = construct_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                x2c_model=model.x2c_model,
                c2y_model=ind_c2y_model,
            )
            if save_model:
                torch.save(
                    ind_model.state_dict(),
                    ind_model_saved_path,
                )
            seq_model = construct_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                x2c_model=model.x2c_model,
                c2y_model=seq_c2y_model,
            )
            if save_model:
                torch.save(
                    seq_model.state_dict(),
                    seq_model_saved_path,
                )

    if test_dl is not None:
        ind_model.freeze()
        ind_trainer = pl.Trainer(
            gpus=gpu,
            logger=(
                logger or
                (WandbLogger(
                    name=ind_full_run_name,
                    project=project_name,
                    save_dir=os.path.join(result_dir, "logs"),
                ) if project_name and (rerun or (not chpt_exists)) else False)
            ),
        )

        def _inner_call(trainer, model):
            [test_results] = trainer.test(model, test_dl)
            output = [
                test_results["test_c_accuracy"],
                test_results["test_y_accuracy"],
                test_results["test_c_auc"],
                test_results["test_y_auc"],
                test_results["test_c_f1"],
                test_results["test_y_f1"],
            ]
            top_k_vals = []
            for key, val in test_results.items():
                if "test_y_top" in key:
                    top_k = int(key[len("test_y_top_"):-len("_accuracy")])
                    top_k_vals.append((top_k, val))
            output += list(map(
                lambda x: x[1],
                sorted(top_k_vals, key=lambda x: x[0]),
            ))
            return output

        keys = [
            "test_acc_c",
            "test_acc_y",
            "test_auc_c",
            "test_auc_y",
            "test_f1_c",
            "test_f1_y",
        ]
        if config.get('top_k_accuracy', None):
            top_k_args = config['top_k_accuracy']
            if top_k_args is None:
                top_k_args = []
            if not isinstance(top_k_args, list):
                top_k_args = [top_k_args]
            for top_k in sorted(top_k_args):
                keys.append(f'test_top_{top_k}_acc_y')
        values, _ = load_call(
            function=_inner_call,
            keys=keys,
            full_run_name=f"IndependentConceptBottleneckModel{config.get('extra_name', '')}",
            old_results=ind_old_results,
            rerun=rerun,
            kwargs=dict(
                trainer=ind_trainer,
                model=ind_model,
            ),
        )
        ind_test_results = {
            key: val
            for (key, val) in zip(keys, values)
        }
        print(
            f'Independent c_acc: {ind_test_results["test_acc_c"] * 100:.2f}%, '
            f'Independent y_acc: {ind_test_results["test_acc_y"] * 100:.2f}%, '
            f'Independent c_auc: {ind_test_results["test_auc_c"] * 100:.2f}%, '
            f'Independent y_auc: {ind_test_results["test_auc_y"] * 100:.2f}%'
        )


        seq_model.freeze()
        seq_trainer = pl.Trainer(
            gpus=gpu,
            logger=(
                logger or
                (WandbLogger(
                    name=seq_full_run_name,
                    project=project_name,
                    save_dir=os.path.join(result_dir, "logs"),
                ) if project_name and (rerun or (not chpt_exists)) else False)
            ),
        )
        values, _ = load_call(
            function=_inner_call,
            keys=keys,
            full_run_name=f"SequentialConceptBottleneckModel{config.get('extra_name', '')}",
            old_results=seq_old_results,
            rerun=rerun,
            kwargs=dict(
                trainer=seq_trainer,
                model=seq_model,
            ),
        )
        seq_test_results = {
            key: val
            for (key, val) in zip(keys, values)
        }
        print(
            f'Sequential c_acc: {seq_test_results["test_acc_c"] * 100:.2f}%, '
            f'Sequential y_acc: {seq_test_results["test_acc_y"] * 100:.2f}%, '
            f'Sequential c_auc: {seq_test_results["test_auc_c"] * 100:.2f}%, '
            f'Sequential y_auc: {seq_test_results["test_auc_y"] * 100:.2f}%'
        )
    else:
        ind_test_results = None
        seq_test_results = None
    return ind_model, ind_test_results, seq_model, seq_test_results


def update_statistics(results, config, model, test_results, save_model=True):
    full_run_name = f"{config['architecture']}{config.get('extra_name', '')}"
    results.update({
        f'test_acc_y_{full_run_name}': test_results['test_acc_y'],
        f'test_auc_y_{full_run_name}': test_results['test_auc_y'],
        f'test_f1_y_{full_run_name}': test_results['test_f1_y'],
        f'test_acc_c_{full_run_name}': test_results['test_acc_c'],
        f'test_auc_c_{full_run_name}': test_results['test_auc_c'],
        f'test_f1_c_{full_run_name}': test_results['test_f1_c'],
    })
    results[f'num_trainable_params_{full_run_name}'] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    results[f'num_non_trainable_params_{full_run_name}'] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    for key, val in test_results.items():
        if "test_y_top" in key:
            top_k = int(key[len("test_y_top_"):-len("_accuracy")])
            results[f'test_top_{top_k}_acc_y_{full_run_name}'] = val



def evaluate_representation_metrics(
    config,
    n_concepts,
    n_tasks,
    test_dl,
    full_run_name,
    split=0,
    imbalance=None,
    result_dir=None,
    sequential=False,
    independent=False,
    task_class_weights=None,
    gpu=int(torch.cuda.is_available()),
    rerun=False,
    seed=None,
    old_results=None,
    test_subsampling=1,
):
    if config.get("rerun_repr_evaluation", False):
        rerun = True
    if config.get("skip_repr_evaluation", False):
        return {}
    test_subsampling = config.get(
        'test_repr_subsampling',
        config.get('test_subsampling', test_subsampling),
    )
    if seed is not None:
        seed_everything(seed)

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
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    c_test = np.concatenate(c_test, axis=0)

    # Now include the competence that we will assume
    # for all concepts
    if test_subsampling not in [None, 0, 1]:
        np.random.seed(42)
        indices = np.random.permutation(x_test.shape[0])[
            :int(np.ceil(x_test.shape[0]*test_subsampling))
        ]
        x_test = x_test[indices]
        c_test = c_test[indices]
        y_test = y_test[indices]
        test_dl = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.FloatTensor(x_test).type(x_type),
                torch.FloatTensor(y_test).type(y_type),
                torch.FloatTensor(c_test).type(c_type),
            ),
            batch_size=test_dl.batch_size,
            num_workers=test_dl.num_workers,
        )

    cbm = load_trained_model(
        config=config,
        n_tasks=n_tasks,
        n_concepts=n_concepts,
        result_dir=result_dir,
        split=split,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
        intervene=True,
        sequential=sequential,
        independent=independent,
    )
    trainer = pl.Trainer(
        gpus=gpu,
        logger=False,
    )
    batch_results = trainer.predict(cbm, test_dl)
    c_sem = np.concatenate(
        list(map(lambda x: x[0].detach().cpu().numpy(), batch_results)),
        axis=0,
    )
    c_pred = np.concatenate(
        list(map(lambda x: x[1].detach().cpu().numpy(), batch_results)),
        axis=0,
    )

    c_pred = np.reshape(c_pred, (c_test.shape[0], n_concepts, -1))
    # We now need to reshuffle the c_pred matrix to recover is concept
    # structure
    ois_key = f'test_ois_{full_run_name}'
    print(f"Computing OIS score...")
    oracle_matrix = None
    if os.path.exists(
        os.path.join(result_dir, f'oracle_matrix.npy')
    ):
        oracle_matrix = np.load(os.path.join(result_dir, f'oracle_matrix.npy'))
    ois, loaded = _execute_and_save(
        fun=load_call,
        kwargs=dict(
            keys=[ois_key],
            old_results=old_results,
            rerun=rerun,
            function=oracle.oracle_impurity_score,
            full_run_name=full_run_name,
            kwargs=dict(
                c_soft=np.transpose(c_pred, (0, 2, 1)),
                c_true=c_test,
                predictor_train_kwags={
                    'epochs': config.get("ois_epochs", 50),
                    'batch_size': min(2048, c_test.shape[0]),
                    'verbose': 0,
                },
                test_size=0.2,
                oracle_matrix=oracle_matrix,
                jointly_learnt=True,
            )
        ),
        result_dir=result_dir,
        filename=f'{ois_key}_split_{split}.joblib',
        rerun=rerun,
    )
    if isinstance(ois, (tuple, list)):
        if len(ois) == 3:
            (ois, _, oracle_matrix) = ois
        else:
            ois = ois[0]
    print(f"\tDone....OIS score is {ois*100:.2f}%")
    if (oracle_matrix is not None) and (not os.path.exists(
        os.path.join(result_dir, f'oracle_matrix.npy')
    )):
        np.save(
            os.path.join(result_dir, f'oracle_matrix.npy'),
            oracle_matrix,
        )

    nis_key = f'test_nis_{full_run_name}'
    print(f"Computing NIS score...")
    nis, loaded = _execute_and_save(
        fun=load_call,
        kwargs=dict(
            keys=[ois_key],
            old_results=old_results,
            rerun=rerun,
            function=niching.niche_impurity_score,
            full_run_name=full_run_name,
            kwargs=dict(
                c_soft=np.transpose(c_pred, (0, 2, 1)),
                c_true=c_test,
                test_size=0.2,
            ),
        ),
        result_dir=result_dir,
        filename=f'{nis_key}_split_{split}.joblib',
        rerun=rerun,
    )
    if isinstance(nis, (tuple, list)):
        assert len(nis) == 1
        nis = nis[0]
    print("nis", nis)
    print(f"\tDone....NIS score is {nis*100:.2f}%")


    cas_key = f'test_cas_{full_run_name}'
    print(f"Computing entire representation CAS score with c_pred.shape =", c_pred.shape, "...")
    cas, loaded = _execute_and_save(
        fun=load_call,
        kwargs=dict(
            keys=[cas_key],
            old_results=old_results,
            rerun=rerun,
            function=homogeneity.embedding_homogeneity,
            full_run_name=full_run_name,
            kwargs=dict(
                c_vec=c_pred,
                c_test=c_test,
                y_test=y_test,
                step=config.get('cas_step', 2),
            ),
        ),
        result_dir=result_dir,
        filename=f'{cas_key}_split_{split}.joblib',
        rerun=rerun,
    )
    if isinstance(cas, (tuple, list)):
        cas = cas[0]
    print(f"\tDone....CAS score is {cas*100:.2f}%")

    prob_cas_key = f'test_cas_probs_only_{full_run_name}'
    print(f"Computing probability only CAS score with c_sem.shape =", c_sem.shape, "...")
    prob_cas, loaded = _execute_and_save(
        fun=load_call,
        kwargs=dict(
            keys=[prob_cas_key],
            old_results=old_results,
            rerun=rerun,
            function=homogeneity.embedding_homogeneity,
            full_run_name=full_run_name,
            kwargs=dict(
                c_vec=c_sem,
                c_test=c_test,
                y_test=y_test,
                step=config.get('cas_step', 2),
            ),
        ),
        result_dir=result_dir,
        filename=f'{prob_cas_key}_split_{split}.joblib',
        rerun=rerun,
    )
    if isinstance(prob_cas, (tuple, list)):
        prob_cas = prob_cas[0]
    print(f"\tDone....Probability CAS score is {prob_cas*100:.2f}%")

    comb_cas_key = f'test_cas_comb_{full_run_name}'
    print(f"Computing combined CAS score with c_sem.shape =", c_sem.shape, "...")
    comb_cas, loaded = _execute_and_save(
        fun=load_call,
        kwargs=dict(
            keys=[comb_cas_key],
            old_results=old_results,
            rerun=rerun,
            function=homogeneity.embedding_homogeneity,
            full_run_name=full_run_name,
            kwargs=dict(
                c_vec=np.concatenate([c_pred, np.expand_dims(c_sem, axis=-1)], axis=-1),
                c_test=c_test,
                y_test=y_test,
                step=config.get('cas_step', 2),
            ),
        ),
        result_dir=result_dir,
        filename=f'{comb_cas_key}_split_{split}.joblib',
        rerun=rerun,
    )
    if isinstance(comb_cas, (tuple, list)):
        comb_cas = comb_cas[0]
    print(f"\tDone....combined CAS score is {comb_cas*100:.2f}%")

    return {
        cas_key: cas,
        prob_cas_key: prob_cas,
        comb_cas_key: comb_cas,
        nis_key: nis,
        ois_key: ois,
    }