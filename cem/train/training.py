import copy
import joblib
import logging
import numpy as np
import os
import pytorch_lightning as pl
import time
import torch

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from scipy.special import expit
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tensorflow as tf

import cem.metrics.niching as niching
import cem.metrics.oracle as oracle
import cem.train.utils as utils

from cem.metrics.cas import concept_alignment_score
from cem.models.construction import (
    construct_model,
    construct_sequential_models,
    load_trained_model,
)



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
    gradient_clip_val=0,
    old_results=None,
    enable_checkpointing=False,
    accelerator="auto",
    devices="auto",
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
            model.load_state_dict(
                torch.load(config.get("model_pretrain_path")),
                strict=False,
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
                accelerator=accelerator,
                devices=devices,
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
                    (
                        WandbLogger(
                            name=full_run_name,
                            project=project_name,
                            save_dir=os.path.join(result_dir, "logs"),
                        ) if rerun or (not os.path.exists(model_saved_path))
                        else False
                    )
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
                if os.path.exists(
                    model_saved_path.replace(".pt", "_training_times.npy")
                ):
                    [training_time, num_epochs] = np.load(
                        model_saved_path.replace(".pt", "_training_times.npy"),
                    )
                else:
                    training_time, num_epochs = 0, 0
            else:
                # Else it is time to train it
                start_time = time.time()
                fit_trainer.fit(model, train_dl, val_dl)
                num_epochs = fit_trainer.current_epoch
                training_time = time.time() - start_time
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
                    np.save(
                        model_saved_path.replace(".pt", "_training_times.npy"),
                        np.array([training_time, num_epochs]),
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
                            top_k = int(
                                key[len("test_y_top_"):-len("_accuracy")]
                            )
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
                values, _ = utils.load_call(
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
                test_results['training_time'] = training_time
                test_results['num_epochs'] = num_epochs
                print(
                    f'c_acc: {test_results["test_acc_c"]*100:.2f}%, '
                    f'y_acc: {test_results["test_acc_y"]*100:.2f}%, '
                    f'c_auc: {test_results["test_auc_c"]*100:.2f}%, '
                    f'y_auc: {test_results["test_auc_y"]*100:.2f}% with '
                    f'{num_epochs} epochs in {training_time:.2f} seconds'
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
            accelerator=accelerator,
            devices=devices,
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
            if os.path.exists(
                model_saved_path.replace(".pt", "_training_times.npy")
            ):
                [training_time, num_epochs] = np.load(
                    model_saved_path.replace(".pt", "_training_times.npy"),
                )
            else:
                training_time, num_epochs = 0, 0
        else:
            # Else it is time to train it
            start_time = time.time()
            fit_trainer.fit(model, train_dl, val_dl)
            training_time = time.time() - start_time
            num_epochs = fit_trainer.current_epoch
            if save_model and (result_dir is not None):
                torch.save(
                    model.state_dict(),
                    model_saved_path,
                )
                np.save(
                    model_saved_path.replace(".pt", "_training_times.npy"),
                    np.array([training_time, num_epochs]),
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
            values, _ = utils.load_call(
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
            test_results['training_time'] = training_time
            test_results['num_epochs'] = num_epochs
            print(
                f'c_acc: {test_results["test_acc_c"]*100:.2f}%, '
                f'y_acc: {test_results["test_acc_y"]*100:.2f}%, '
                f'c_auc: {test_results["test_auc_c"]*100:.2f}%, '
                f'y_auc: {test_results["test_auc_y"]*100:.2f}% with '
                f'{num_epochs} epochs in {training_time:.2f} seconds'
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
    project_name='',
    seed=None,
    save_model=True,
    activation_freq=0,
    single_frequency_epochs=0,
    accelerator="auto",
    devices="auto",
    ind_old_results=None,
    seq_old_results=None,
    enable_checkpointing=False,
):
    if seed is not None:
        seed_everything(seed)
    num_epochs = 0
    training_time = 0

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
            accelerator=accelerator,
            devices=devices,
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
            if os.path.exists(
                ind_model_saved_path.replace(".pt", "_training_times.npy")
            ):
                [ind_training_time, ind_num_epochs] = np.load(
                    ind_model_saved_path.replace(".pt", "_training_times.npy")
                )
            else:
                ind_training_time, ind_num_epochs = 0, 0

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
            if os.path.exists(
                seq_model_saved_path.replace(".pt", "_training_times.npy")
            ):
                [seq_training_time, seq_num_epochs] = np.load(
                    seq_model_saved_path.replace(".pt", "_training_times.npy")
                )
            else:
                seq_training_time, seq_num_epochs = 0, 0
        else:
            # First train the input to concept model
            print("[Training input to concept model]")
            start_time = time.time()
            x2c_trainer.fit(model, train_dl, val_dl)
            training_time += time.time() - start_time
            num_epochs += x2c_trainer.current_epoch
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
                accelerator=accelerator,
                devices=devices,
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
                    (
                        WandbLogger(
                            name=ind_full_run_name,
                            project=project_name,
                            save_dir=os.path.join(result_dir, "logs"),
                        ) if project_name and (rerun or (not chpt_exists))
                        else False
                    )
                ),
            )
            start_time = time.time()
            ind_c2y_trainer.fit(
                ind_c2y_model,
                ind_c2y_train_dl,
                ind_c2y_val_dl,
            )
            ind_training_time = training_time + time.time() - start_time
            ind_num_epochs = num_epochs + ind_c2y_trainer.current_epoch
            if ind_c2y_val_dl is not None:
                print(
                    "Independent validation results for c2y model:",
                    ind_c2y_trainer.test(ind_c2y_model, ind_c2y_val_dl),
                )

            # Train the sequential concept to label model
            print("[Training sequential concept to label model]")
            seq_c2y_trainer = pl.Trainer(
                accelerator=accelerator,
                devices=devices,
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
                    (
                        WandbLogger(
                            name=seq_full_run_name,
                            project=project_name,
                            save_dir=os.path.join(result_dir, "logs"),
                        ) if project_name and (rerun or (not chpt_exists))
                        else False
                    )
                ),
            )
            start_time = time.time()
            seq_c2y_trainer.fit(
                seq_c2y_model,
                seq_c2y_train_dl,
                seq_c2y_val_dl,
            )
            seq_training_time = training_time + time.time() - start_time
            seq_num_epochs = num_epochs + seq_c2y_trainer.current_epoch
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
                np.save(
                    ind_model_saved_path.replace(".pt", "_training_times.npy"),
                    np.array([ind_training_time, ind_num_epochs]),
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
                np.save(
                    seq_model_saved_path.replace(".pt", "_training_times.npy"),
                    np.array([seq_training_time, seq_num_epochs]),
                )

    if test_dl is not None:
        ind_model.freeze()
        ind_trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
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
        values, _ = utils.load_call(
            function=_inner_call,
            keys=keys,
            full_run_name=(
                f"IndependentConceptBottleneckModel{config.get('extra_name', '')}"
            ),
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
        ind_test_results['training_time'] = ind_training_time
        ind_test_results['num_epochs'] = ind_num_epochs
        print(
            f'Independent c_acc: {ind_test_results["test_acc_c"] * 100:.2f}%, '
            f'Independent y_acc: {ind_test_results["test_acc_y"] * 100:.2f}%, '
            f'Independent c_auc: {ind_test_results["test_auc_c"] * 100:.2f}%, '
            f'Independent y_auc: {ind_test_results["test_auc_y"] * 100:.2f}% with '
            f'{ind_num_epochs} epochs in {ind_training_time:.2f} seconds'
        )


        seq_model.freeze()
        seq_trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=(
                logger or
                (WandbLogger(
                    name=seq_full_run_name,
                    project=project_name,
                    save_dir=os.path.join(result_dir, "logs"),
                ) if project_name and (rerun or (not chpt_exists)) else False)
            ),
        )
        values, _ = utils.load_call(
            function=_inner_call,
            keys=keys,
            full_run_name=(
                f"SequentialConceptBottleneckModel{config.get('extra_name', '')}"
            ),
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
        seq_test_results['training_time'] = seq_training_time
        seq_test_results['num_epochs'] = seq_num_epochs
        print(
            f'Sequential c_acc: {seq_test_results["test_acc_c"] * 100:.2f}%, '
            f'Sequential y_acc: {seq_test_results["test_acc_y"] * 100:.2f}%, '
            f'Sequential c_auc: {seq_test_results["test_auc_c"] * 100:.2f}%, '
            f'Sequential y_auc: {seq_test_results["test_auc_y"] * 100:.2f}% with '
            f'{seq_num_epochs} epochs in {seq_training_time:.2f} seconds'
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
        f'training_epochs_{full_run_name}': test_results['num_epochs'],
        f'training_time_{full_run_name}': test_results['training_time'],
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

def representation_avg_task_pred(
    c_embs_train,
    c_embs_test,
    y_train,
    y_test,
    predictor_train_kwags=None,
):
    n_samples, n_concepts, concept_emb_dims = c_embs_train.shape
    n_classes = len(np.unique(y_train))
    predictor_train_kwags = predictor_train_kwags or {
        'epochs': 100,
        'batch_size': min(512, n_samples),
        'verbose': 0,
    }
    accs = []
    for concept_idx in tqdm(range(n_concepts)):
        classifier = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                32,
                activation='relu',
                name="predictor_fc_1",
            ),
            tf.keras.layers.Dense(
                n_classes if n_classes > 2 else 1,
                # We will merge the activation into the loss for numerical
                # stability
                activation=None,
                name="predictor_fc_out",
            ),
        ])

        loss = (
            tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ) if n_classes > 2 else
            tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
            )
        )
        classifier.compile(
            # Use ADAM optimizer by default
            optimizer='adam',
            # Note: we assume labels come without a one-hot-encoding in the
            #       case when the concepts are categorical.
            loss=loss,
        )
        # classifier = LogisticRegression(
        #     penalty='none',
        #     random_state=42,
        #     max_iter=predictor_train_kwags.get('max_iter', 100),
        # )
        classifier.fit(
            c_embs_train[:,concept_idx,:],
            y_train,
            **predictor_train_kwags,
        )
        # y_test_pred = classifier.predict_proba(c_embs_test[:, concept_idx, :])
        y_test_pred = classifier.predict(c_embs_test[:, concept_idx, :])
        if n_classes > 2:
            accs.append(accuracy_score(y_test, np.argmax(y_test_pred, axis=-1)))
        else:
            accs.append(accuracy_score(y_test, expit(y_test_pred) >=0.5))
    return np.mean(accs)


def evaluate_representation_metrics(
    config,
    n_concepts,
    n_tasks,
    test_dl,
    full_run_name,
    split=0,
    train_dl=None,
    imbalance=None,
    result_dir=None,
    sequential=False,
    independent=False,
    task_class_weights=None,
    accelerator="auto",
    devices="auto",
    rerun=False,
    seed=None,
    old_results=None,
    test_subsampling=1,
):
    result_dict = {}
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
        accelerator=accelerator,
        devices=devices,
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
    if config.get('extra_dims', 0) != 0:
        # Then we will only use the extra dims as the embedding as those
        # correspond to the learnt embeddings only
        c_pred = c_pred[:, -config.get('extra_dims', 0):]

    c_pred = np.reshape(c_pred, (c_test.shape[0], n_concepts, -1))

    oracle_matrix = None
    if config.get("run_ois", True):
        ois_key = f'test_ois_{full_run_name}'
        logging.info(f"Computing OIS score...")
        if os.path.exists(
            os.path.join(result_dir, f'oracle_matrix.npy')
        ):
            oracle_matrix = np.load(
                os.path.join(result_dir, f'oracle_matrix.npy')
            )
        ois, loaded = utils.execute_and_save(
            fun=utils.load_call,
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
                    output_matrices=True,
                ),
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
        logging.info(f"\tDone....OIS score is {ois*100:.2f}%")
        if (oracle_matrix is not None) and (not os.path.exists(
            os.path.join(result_dir, f'oracle_matrix.npy')
        )):
            np.save(
                os.path.join(result_dir, f'oracle_matrix.npy'),
                oracle_matrix,
            )
        result_dict[ois_key] = ois


    # Then let's try and see how predictive each representation is of the
    # downstream task
    if train_dl is not None and (
        config.get("run_repr_avg_pred", False)
    ):
        x_train, y_train, c_train = [], [], []
        for ds_data in train_dl:
            if len(ds_data) == 2:
                x, (y, c) = ds_data
            else:
                (x, y, c) = ds_data
            x_type = x.type()
            y_type = y.type()
            c_type = c.type()
            x_train.append(x)
            y_train.append(y)
            c_train.append(c)
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        c_train = np.concatenate(c_train, axis=0)

        used_train_dl = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.FloatTensor(x_train).type(x_type),
                torch.FloatTensor(y_train).type(y_type),
                torch.FloatTensor(c_train).type(c_type),
            ),
            batch_size=32,
            num_workers=train_dl.num_workers,
        )

        train_batch_results = trainer.predict(cbm, used_train_dl)
        c_pred_train = np.concatenate(
            list(map(
                lambda x: x[1].detach().cpu().numpy(),
                train_batch_results
            )),
            axis=0,
        )

        c_pred_train = np.reshape(
            c_pred_train,
            (c_pred_train.shape[0], n_concepts, -1),
        )

        repr_task_pred_key = f'test_repr_task_pred_{full_run_name}'
        logging.info(
            f"Computing avg task predictibility from learnt concept reprs..."
        )
        repr_task_pred, loaded = utils.execute_and_save(
            fun=utils.load_call,
            kwargs=dict(
                keys=[repr_task_pred_key],
                old_results=old_results,
                rerun=rerun,
                function=representation_avg_task_pred,
                full_run_name=full_run_name,
                kwargs=dict(
                    c_embs_train=c_pred_train,
                    c_embs_test=c_pred,
                    y_train=y_train,
                    y_test=y_test,
                ),
            ),
            result_dir=result_dir,
            filename=f'{repr_task_pred_key}_split_{split}.joblib',
            rerun=rerun,
        )
        logging.info(
            f"\tDone....average repr_task_pred is {repr_task_pred*100:.2f}%"
        )

        result_dict.update({
            repr_task_pred_key: repr_task_pred,
        })

    if config.get("run_nis", True):
        # Niche impurity score now
        nis_key = f'test_nis_{full_run_name}'
        logging.info(f"Computing NIS score...")
        nis, loaded = utils.execute_and_save(
            fun=utils.load_call,
            kwargs=dict(
                keys=[nis_key],
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
        logging.info(f"\tDone....NIS score is {nis*100:.2f}%")
        result_dict[nis_key] = nis

    if config.get("run_cas", True):
        cas_key = f'test_cas_{full_run_name}'
        logging.info(
            f"Computing entire representation CAS score..."
        )
        cas, _ = utils.execute_and_save(
            fun=utils.load_call,
            kwargs=dict(
                keys=[cas_key],
                old_results=old_results,
                rerun=rerun,
                function=concept_alignment_score,
                full_run_name=full_run_name,
                kwargs=dict(
                    c_vec=c_pred,
                    c_test=c_test,
                    y_test=y_test,
                    step=config.get('cas_step', 50),
                ),
            ),
            result_dir=result_dir,
            filename=f'{cas_key}_split_{split}.joblib',
            rerun=rerun,
        )
        if isinstance(cas, (tuple, list)):
            cas = cas[0]
        logging.info(f"\tDone....CAS score is {cas*100:.2f}%")
        result_dict[cas_key] = cas

    return result_dict