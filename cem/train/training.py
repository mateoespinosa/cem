import copy
import joblib
import numpy as np
import os
import pytorch_lightning as pl
import torch

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchvision.models import resnet18, resnet34, resnet50, densenet121

import cem.models.cem as models_cem
import cem.models.cbm as models_cbm
import cem.train.utils as utils




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
):
    if config["architecture"] in ["ConceptEmbeddingModel", "MixtureEmbModel"]:
        model_cls = models_cem.ConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "shared_prob_gen": config["shared_prob_gen"],
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.0,
            ),
            "embeding_activation": config.get("embeding_activation", None),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
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
        task_loss_weight=config.get('task_loss_weight', 1.0),
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        c_extractor_arch=utils.wrap_pretrained_model(c_extractor_arch),
        optimizer=config['optimizer'],
        top_k_accuracy=config.get('top_k_accuracy'),
        output_latent=output_latent,
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
        task_class_weights=(
            torch.FloatTensor(task_class_weights)
            if (task_class_weights is not None)
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
    )
    return x2c_model, c2y_model


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
    logger=None,
    project_name='',
    seed=None,
    save_model=True,
    activation_freq=0,
    single_frequency_epochs=0,
    gpu=int(torch.cuda.is_available()),
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
            independent=("Independent" in config['architecture']),
        )
    if seed is not None:
        seed_everything(split)

    extr_name = config['c_extractor_arch']
    if not isinstance(extr_name, str):
        extr_name = "lambda"
    if split is not None:
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}_"
            f"{extr_name}_fold_{split + 1}"
        )
    else:
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}_"
            f"{extr_name}"
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
                # Only use the wandb logger when it is a fresh run
                logger=(
                    logger or
                    (WandbLogger(
                        name=full_run_name,
                        project=project_name,
                        save_dir=os.path.join(result_dir, "logs"),
                    ) if rerun or (not os.path.exists(model_saved_path)) else True)
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
                [test_results] = trainer.test(model, test_dl)
                c_accuracy, y_accuracy = test_results["test_c_accuracy"], \
                    test_results["test_y_accuracy"]
                c_auc, y_auc = \
                    test_results["test_c_auc"], test_results["test_y_auc"]
                c_f1, y_f1 = \
                    test_results["test_c_f1"], test_results["test_y_f1"]
                print(
                    f'{full_run_name} c_acc: {c_accuracy:.4f}, '
                    f'{full_run_name} c_auc: {c_auc:.4f}, '
                    f'{full_run_name} c_f1: {c_f1:.4f}, '
                    f'{full_run_name} y_acc: {y_accuracy:.4f}, '
                    f'{full_run_name} y_auc: {y_auc:.4f}, '
                    f'{full_run_name} y_f1: {y_f1:.4f}'
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
            logger=logger or True,
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
            [test_results] = trainer.test(model, test_dl)
            c_accuracy, y_accuracy = \
                test_results["test_c_accuracy"], test_results["test_y_accuracy"]
            c_auc, y_auc = \
                test_results["test_c_auc"], test_results["test_y_auc"]
            c_f1, y_f1 = test_results["test_c_f1"], test_results["test_y_f1"]
            print(
                f'{full_run_name} c_acc: {c_accuracy:.4f}, '
                f'{full_run_name} c_auc: {c_auc:.4f}, '
                f'{full_run_name} c_f1: {c_f1:.4f}, '
                f'{full_run_name} y_acc: {y_accuracy:.4f}, '
                f'{full_run_name} y_auc: {y_auc:.4f}, '
                f'{full_run_name} y_f1: {y_f1:.4f}'
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
    logger=None,
    project_name='cub_concept_training',
    seed=None,
    save_model=True,
    activation_freq=0,
    single_frequency_epochs=0,
    gpu=int(torch.cuda.is_available()),
):
    if seed is not None:
        seed_everything(split)

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
        print("x_train.shape =", x_train.shape)
        y_train = np.concatenate(y_train, axis=0)
        print("y_train.shape =", y_train.shape)
        c_train = np.concatenate(c_train, axis=0)
        print("c_train.shape =", c_train.shape)

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
            print("x_val.shape =", x_val.shape)
            y_val = np.concatenate(y_val, axis=0)
            print("y_val.shape =", y_val.shape)
            c_val = np.concatenate(c_val, axis=0)
            print("c_val.shape =", c_val.shape)
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
                ) if project_name and (rerun or (not chpt_exists)) else True)
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
                    ) if project_name and (rerun or (not chpt_exists)) else True)
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
                    ) if project_name and (rerun or (not chpt_exists)) else True)
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
                ) if project_name and (rerun or (not chpt_exists)) else True)
            ),
        )
        [ind_test_results] = ind_trainer.test(ind_model, test_dl)
        c_accuracy, y_accuracy = ind_test_results["test_c_accuracy"], \
            ind_test_results["test_y_accuracy"]
        c_auc, y_auc = \
            ind_test_results["test_c_auc"], ind_test_results["test_y_auc"]
        c_f1, y_f1 = \
            ind_test_results["test_c_f1"], ind_test_results["test_y_f1"]
        print(
            f'{ind_full_run_name} c_acc: {c_accuracy:.4f}, '
            f'{ind_full_run_name} c_auc: {c_auc:.4f}, '
            f'{ind_full_run_name} c_f1: {c_f1:.4f}, '
            f'{ind_full_run_name} y_acc: {y_accuracy:.4f}, '
            f'{ind_full_run_name} y_auc: {y_auc:.4f}, '
            f'{ind_full_run_name} y_f1: {y_f1:.4f}'
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
                ) if project_name and (rerun or (not chpt_exists)) else True)
            ),
        )
        [seq_test_results] = seq_trainer.test(seq_model, test_dl)
        c_accuracy, y_accuracy = seq_test_results["test_c_accuracy"], \
            seq_test_results["test_y_accuracy"]
        c_auc, y_auc = \
            seq_test_results["test_c_auc"], seq_test_results["test_y_auc"]
        c_f1, y_f1 = \
            seq_test_results["test_c_f1"], seq_test_results["test_y_f1"]
        print(
            f'{seq_full_run_name} c_acc: {c_accuracy:.4f}, '
            f'{seq_full_run_name} c_auc: {c_auc:.4f}, '
            f'{seq_full_run_name} c_f1: {c_f1:.4f}, '
            f'{seq_full_run_name} y_acc: {y_accuracy:.4f}, '
            f'{seq_full_run_name} y_auc: {y_auc:.4f}, '
            f'{seq_full_run_name} y_f1: {y_f1:.4f}'
        )
    else:
        test_results = None
    return ind_model, ind_test_results, seq_model, seq_test_results


def update_statistics(results, config, model, test_results, save_model=True):
    full_run_name = f"{config['architecture']}{config.get('extra_name', '')}"
    results.update({
        f'test_acc_y_{full_run_name}': test_results['test_y_accuracy'],
        f'test_auc_y_{full_run_name}': test_results['test_y_auc'],
        f'test_f1_y_{full_run_name}': test_results['test_y_f1'],
        f'test_acc_c_{full_run_name}': test_results['test_c_accuracy'],
        f'test_auc_c_{full_run_name}': test_results['test_c_auc'],
        f'test_f1_c_{full_run_name}': test_results['test_c_f1'],
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
