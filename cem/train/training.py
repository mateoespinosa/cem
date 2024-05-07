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

import cem.train.utils as utils
import cem.utils.data as data_utils

from cem.models.construction import (
    construct_model,
    construct_sequential_models,
)

def _make_callbacks(config):
    callbacks = []
    if 'early_stopping_monitor' in config:
        callbacks.append(
            EarlyStopping(
                monitor=config["early_stopping_monitor"],
                min_delta=config.get("early_stopping_delta", 0.00),
                patience=config.get('patience', 5),
                verbose=config.get("verbose", False),
                mode=config.get("early_stopping_mode", "min"),
            )
        )

    return callbacks
def _evaluate_cbm(
    model,
    trainer,
    config,
    run_name,
    old_results=None,
    rerun=False,
    test_dl=None,
    val_dl=None,
):
    eval_results = {}
    for (current_dl, dl_name) in [(val_dl, "val"), (test_dl, "test")]:
        if current_dl is None:
            pass
        model.freeze()
        def _inner_call():
            [eval_results] = trainer.test(model, current_dl)
            output = [
                eval_results[f"test_c_accuracy"],
                eval_results[f"test_y_accuracy"],
                eval_results[f"test_c_auc"],
                eval_results[f"test_y_auc"],
                eval_results[f"test_c_f1"],
                eval_results[f"test_y_f1"],
            ]
            top_k_vals = []
            for key, val in eval_results.items():
                if f"test_y_top" in key:
                    top_k = int(
                        key[len(f"test_y_top_"):-len("_accuracy")]
                    )
                    top_k_vals.append((top_k, val))
            output += list(map(
                lambda x: x[1],
                sorted(top_k_vals, key=lambda x: x[0]),
            ))
            return output

        keys = [
            f"{dl_name}_acc_c",
            f"{dl_name}_acc_y",
            f"{dl_name}_auc_c",
            f"{dl_name}_auc_y",
            f"{dl_name}_f1_c",
            f"{dl_name}_f1_y",
        ]
        if 'top_k_accuracy' in config:
            top_k_args = config['top_k_accuracy']
            if top_k_args is None:
                top_k_args = []
            if not isinstance(top_k_args, list):
                top_k_args = [top_k_args]
            for top_k in sorted(top_k_args):
                keys.append(f'{dl_name}_top_{top_k}_acc_y')
        values, _ = utils.load_call(
            function=_inner_call,
            keys=keys,
            run_name=run_name,
            old_results=old_results,
            rerun=rerun,
            kwargs={},
        )
        eval_results.update({
            key: val
            for (key, val) in zip(keys, values)
        })
    return eval_results


################################################################################
## MODEL TRAINING
################################################################################

def train_end_to_end_model(
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl,
    run_name,
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
    if seed is not None:
        seed_everything(seed)

    if split is not None:
        full_run_name = (
            f"{run_name}_fold_{split + 1}"
        )
    else:
        full_run_name = (
            f"{run_name}"
        )
    print(f"[Training {run_name}]")
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
        enter_obj = wandb.init(
            project=project_name,
            name=full_run_name,
            config=config,
            reinit=True
        )
    else:
        enter_obj = utils.EmptyEnter()

    with enter_obj as run:
        callbacks = _make_callbacks(config)
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=config['max_epochs'],
            check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
            callbacks=callbacks,
            logger=logger or False,
            enable_checkpointing=enable_checkpointing,
            gradient_clip_val=gradient_clip_val,
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
            if fit_trainer.interrupted:
                reply = None
                while reply not in ['y', 'n']:
                    if reply is not None:
                        print("Please provide only either 'y' or 'n'.")
                    reply = input(
                        "Would you like to manually interrupt this model's "
                        "training and continue the experiment? [y/n]\n"
                    ).strip().lower()
                if reply == "n":
                    raise ValueError(
                        'Experiment execution was manually interrupted!'
                    )
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
            f'{run_name}_experiment_config.joblib'
        )):
            # Then let's serialize the experiment config for this run
            config_copy = copy.deepcopy(config)
            if "c_extractor_arch" in config_copy and (
                not isinstance(config_copy["c_extractor_arch"], str)
            ):
                del config_copy["c_extractor_arch"]
            joblib.dump(config_copy, os.path.join(
                result_dir,
                f'{run_name}_experiment_config.joblib'
            ))
        eval_results = _evaluate_cbm(
            model=model,
            trainer=trainer,
            config=config,
            run_name=run_name,
            old_results=old_results,
            rerun=rerun,
            test_dl=test_dl,
            val_dl=val_dl,
        )
        eval_results['training_time'] = training_time
        eval_results['num_epochs'] = num_epochs
        if test_dl is not None:
            print(
                f'c_acc: {eval_results["test_acc_c"]*100:.2f}%, '
                f'y_acc: {eval_results["test_acc_y"]*100:.2f}%, '
                f'c_auc: {eval_results["test_auc_c"]*100:.2f}%, '
                f'y_auc: {eval_results["test_auc_y"]*100:.2f}% with '
                f'{num_epochs} epochs in {training_time:.2f} seconds'
            )
    return model, eval_results


def train_sequential_model(
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl,
    run_name,
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
    old_results=None,
    enable_checkpointing=False,
    gradient_clip_val=0,
):
    if seed is not None:
        seed_everything(seed)
    num_epochs = 0
    training_time = 0

    if split is not None:
        full_run_name = (
            f"{run_name}_fold_{split + 1}"
        )
    else:
        full_run_name = run_name
    print(f"[Training {full_run_name}]")
    print("config:")
    for key, val in config.items():
        print(f"\t{key} -> {val}")

    # Create the model we will manipulate
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
    model_saved_path = os.path.join(
        result_dir,
        f'{full_run_name}.pt'
    )
    chpt_exists = os.path.exists(model_saved_path)
    # Construct the datasets we will need for training if the model
    # has not been found
    if rerun or (not chpt_exists):
        x_train, y_train, c_train = data_utils.daloader_to_memory(train_dl)

        if test_dl:
            x_test, y_test, c_test = data_utils.daloader_to_memory(test_dl)
        if val_dl is not None:
            x_val, y_val, c_val = data_utils.daloader_to_memory(val_dl)


    if (project_name) and result_dir and (not chpt_exists):
        # Lazy import to avoid importing unless necessary
        import wandb
        enter_obj = wandb.init(
            project=project_name,
            name=full_run_name,
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
            callbacks=_make_callbacks(config),
            # Only use the wandb logger when it is a fresh run
            logger=(
                logger or
                (WandbLogger(
                    name=full_run_name,
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
            seq_model = construct_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                x2c_model=model.x2c_model,
                c2y_model=seq_c2y_model,
            )
            seq_model.load_state_dict(torch.load(model_saved_path))
            if os.path.exists(
                model_saved_path.replace(".pt", "_training_times.npy")
            ):
                [seq_training_time, seq_num_epochs] = np.load(
                    model_saved_path.replace(".pt", "_training_times.npy")
                )
            else:
                seq_training_time, seq_num_epochs = 0, 0
        else:
            # First train the input to concept model
            print("[Training input to concept model]")
            start_time = time.time()
            x2c_trainer.fit(model, train_dl, val_dl)
            if x2c_trainer.interrupted:
                reply = None
                while reply not in ['y', 'n']:
                    if reply is not None:
                        print("Please provide only either 'y' or 'n'.")
                    reply = input(
                        "Would you like to manually interrupt this model's "
                        "training and continue the experiment? [y/n]\n"
                    ).strip().lower()
                if reply == "n":
                    raise ValueError(
                        'Experiment execution was manually interrupted!'
                    )
            training_time += time.time() - start_time
            num_epochs += x2c_trainer.current_epoch
            if val_dl is not None:
                print(
                    "Validation results for x2c model:",
                    x2c_trainer.test(model, val_dl),
                )

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
                batch_size=config['dataset_config']['batch_size'],
                num_workers=config['dataset_config'].get('num_workers', 5),
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
                        num_workers=config['dataset_config'].get('num_workers', 5),
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
                    batch_size=config['dataset_config']['batch_size'],
                    num_workers=config['dataset_config'].get('num_workers', 5),
                )
            else:
                seq_c2y_val_dl = None


            # Train the sequential concept to label model
            print("[Training sequential concept to label model]")
            seq_c2y_trainer = pl.Trainer(
                accelerator=accelerator,
                devices=devices,
                # We will distribute half epochs in one model and half on the
                # other
                max_epochs=config.get('c2y_max_epochs', 50),
                enable_checkpointing=enable_checkpointing,
                gradient_clip_val=gradient_clip_val,
                check_val_every_n_epoch=config.get(
                    "check_val_every_n_epoch",
                    5,
                ),
                callbacks=_make_callbacks(config),
                # Only use the wandb logger when it is a fresh run
                logger=(
                    logger or
                    (
                        WandbLogger(
                            name=full_run_name,
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
            if seq_c2y_trainer.interrupted:
                reply = None
                while reply not in ['y', 'n']:
                    if reply is not None:
                        print("Please provide only either 'y' or 'n'.")
                    reply = input(
                        "Would you like to manually interrupt this model's "
                        "training and continue the experiment? [y/n]\n"
                    ).strip().lower()
                if reply == "n":
                    raise ValueError(
                        'Experiment execution was manually interrupted!'
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
                    f'{run_name}_experiment_config.joblib',
                ),
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
                    model_saved_path,
                )
                np.save(
                    model_saved_path.replace(".pt", "_training_times.npy"),
                    np.array([seq_training_time, seq_num_epochs]),
                )

    eval_results = _evaluate_cbm(
        model=model,
        trainer=trainer,
        config=config,
        run_name=run_name,
        old_results=old_results,
        rerun=rerun,
        test_dl=test_dl,
        val_dl=val_dl,
    )
    eval_results['training_time'] = training_time
    eval_results['num_epochs'] = num_epochs
    if test_dl is not None:
        print(
            f'c_acc: {eval_results["test_acc_c"]*100:.2f}%, '
            f'y_acc: {eval_results["test_acc_y"]*100:.2f}%, '
            f'c_auc: {eval_results["test_auc_c"]*100:.2f}%, '
            f'y_auc: {eval_results["test_auc_y"]*100:.2f}% with '
            f'{num_epochs} epochs in {training_time:.2f} seconds'
        )
    return seq_model, eval_results



def train_independent_model(
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl,
    run_name,
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
    old_results=None,
    enable_checkpointing=False,
    gradient_clip_val=0,
):
    if seed is not None:
        seed_everything(seed)
    num_epochs = 0
    training_time = 0

    if split is not None:
        full_run_name = (
            f"{run_name}_fold_{split + 1}"
        )
    else:
        full_run_name = run_name
    print(f"[Training {full_run_name}]")
    print("config:")
    for key, val in config.items():
        print(f"\t{key} -> {val}")

    # Let's construct the model we will need for this
    _, ind_c2y_model = construct_sequential_models(
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
    model_config['architecture'] = 'ConceptBottleneckModel'
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
    model_saved_path = os.path.join(
        result_dir,
        f'{full_run_name}.pt'
    )
    chpt_exists = os.path.exists(model_saved_path)
    # Construct the datasets we will need for training if the model
    # has not been found
    if rerun or (not chpt_exists):
        x_train, y_train, c_train = data_utils.daloader_to_memory(train_dl)

        if test_dl:
            x_test, y_test, c_test = data_utils.daloader_to_memory(test_dl)
        if val_dl is not None:
            x_val, y_val, c_val = data_utils.daloader_to_memory(val_dl)
        else:
            c2y_val_dl = None


    if (project_name) and result_dir and (not chpt_exists):
        # Lazy import to avoid importing unless necessary
        import wandb
        enter_obj = wandb.init(
            project=project_name,
            name=full_run_name,
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
            callbacks=_make_callbacks(config),
            # Only use the wandb logger when it is a fresh run
            logger=(
                logger or
                (WandbLogger(
                    name=full_run_name,
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
            ind_model.load_state_dict(torch.load(model_saved_path))
            if os.path.exists(
                model_saved_path.replace(".pt", "_training_times.npy")
            ):
                [ind_training_time, ind_num_epochs] = np.load(
                    model_saved_path.replace(".pt", "_training_times.npy")
                )
            else:
                ind_training_time, ind_num_epochs = 0, 0

        else:
            # First train the input to concept model
            print("[Training input to concept model]")
            start_time = time.time()
            x2c_trainer.fit(model, train_dl, val_dl)
            if x2c_trainer.interrupted:
                reply = None
                while reply not in ['y', 'n']:
                    if reply is not None:
                        print("Please provide only either 'y' or 'n'.")
                    reply = input(
                        "Would you like to manually interrupt this model's "
                        "training and continue the experiment? [y/n]\n"
                    ).strip().lower()
                if reply == "n":
                    raise ValueError(
                        'Experiment execution was manually interrupted!'
                    )
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
                batch_size=config['dataset_config']['batch_size'],
                num_workers=config['dataset_config'].get('num_workers', 5),
            )
            if val_dl is not None:
                ind_c2y_val_dl = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(
                        torch.from_numpy(
                            c_val
                        ),
                        torch.from_numpy(y_val),
                    ),
                    batch_size=config['dataset_config']['batch_size'],
                    num_workers=config['dataset_config'].get('num_workers', 5),
                )
            else:
                ind_c2y_val_dl = None

            # Train the independent concept to label model
            print("[Training independent concept to label model]")
            ind_c2y_trainer = pl.Trainer(
                accelerator=accelerator,
                devices=devices,
                # We will distribute half epochs in one model and half on the
                # other
                max_epochs=config.get('c2y_max_epochs', 50),
                enable_checkpointing=enable_checkpointing,
                gradient_clip_val=gradient_clip_val,
                check_val_every_n_epoch=config.get(
                    "check_val_every_n_epoch",
                    5,
                ),
                callbacks=_make_callbacks(config),
                # Only use the wandb logger when it is a fresh run
                logger=(
                    logger or
                    (
                        WandbLogger(
                            name=full_run_name,
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
            if ind_c2y_trainer.interrupted:
                reply = None
                while reply not in ['y', 'n']:
                    if reply is not None:
                        print("Please provide only either 'y' or 'n'.")
                    reply = input(
                        "Would you like to manually interrupt this model's "
                        "training and continue the experiment? [y/n]\n"
                    ).strip().lower()
                if reply == "n":
                    raise ValueError(
                        'Experiment execution was manually interrupted!'
                    )
            ind_training_time = training_time + time.time() - start_time
            ind_num_epochs = num_epochs + ind_c2y_trainer.current_epoch
            if ind_c2y_val_dl is not None:
                print(
                    "Independent validation results for c2y model:",
                    ind_c2y_trainer.test(ind_c2y_model, ind_c2y_val_dl),
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
                    f'{run_name}_experiment_config.joblib',
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
                    model_saved_path,
                )
                np.save(
                    model_saved_path.replace(".pt", "_training_times.npy"),
                    np.array([ind_training_time, ind_num_epochs]),
                )
    eval_results = _evaluate_cbm(
        model=model,
        trainer=trainer,
        config=config,
        run_name=run_name,
        old_results=old_results,
        rerun=rerun,
        test_dl=test_dl,
        val_dl=val_dl,
    )
    eval_results['training_time'] = training_time
    eval_results['num_epochs'] = num_epochs
    if test_dl is not None:
        print(
            f'c_acc: {eval_results["test_acc_c"]*100:.2f}%, '
            f'y_acc: {eval_results["test_acc_y"]*100:.2f}%, '
            f'c_auc: {eval_results["test_auc_c"]*100:.2f}%, '
            f'y_auc: {eval_results["test_auc_y"]*100:.2f}% with '
            f'{num_epochs} epochs in {training_time:.2f} seconds'
        )
    return ind_model, eval_results


def train_prob_cbm(
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl,
    run_name,
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
    assert activation_freq == 0, (
        'ProbCBM training currently does not support activation dumping during '
        'training.'
    )
    if seed is not None:
        seed_everything(seed)

    if split is not None:
        full_run_name = (
            f"{run_name}_fold_{split + 1}"
        )
    else:
        full_run_name = (
            f"{run_name}"
        )
    print(f"[Training {run_name}]")
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
        enter_obj = wandb.init(
            project=project_name,
            name=full_run_name,
            config=config,
            reinit=True
        )
        used_logger = WandbLogger(
            name=full_run_name,
            project=project_name,
            save_dir=os.path.join(result_dir, "logs"),
        ) if rerun or (not os.path.exists(model_saved_path)) else False
    else:
        enter_obj = utils.EmptyEnter()
        used_logger = logger or False

    model_saved_path = os.path.join(
        result_dir,
        f'{full_run_name}.pt'
    )
    trainer_args = dict(
        accelerator=accelerator,
        devices=devices,
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
        callbacks=_make_callbacks(config),
        enable_checkpointing=enable_checkpointing,
        gradient_clip_val=gradient_clip_val,
        # Only use the wandb logger when it is a fresh run
        logger=used_logger,
    )

    with enter_obj as run:
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
            # First train the concept model by setting the mode accordingly
            if model.train_class_mode == 'sequential':
                trainable_params = [
                    name for name, param in model.named_parameters()
                    if param.requires_grad
                ]
                old_lrs = [
                    config['learning_rate'],
                    config['learning_rate'] * config.get('lr_ratio', 10),
                ]
                if config.get('warmup_epochs', 5) != 0:
                    warmup_epochs = config.get('warmup_epochs', 5)
                    print(
                        f"\tWarming up ProbCBM for {warmup_epochs} epochs"
                    )
                    for p in model.cnn_module.parameters():
                        p.requires_grad = False
                    warmup_trainer = pl.Trainer(
                        max_epochs=config.get(
                            'warmup_epochs',
                            5,
                        ),
                        **trainer_args,
                    )
                    warmup_trainer.fit(model, train_dl, val_dl)
                    if warmup_trainer.interrupted:
                        reply = None
                        while reply not in ['y', 'n']:
                            if reply is not None:
                                print(
                                    "Please provide only either 'y' or 'n'."
                                )
                            reply = input(
                                "Would you like to manually interrupt this "
                                "model's training and continue the "
                                "experiment? [y/n]\n"
                            ).strip().lower()
                        if reply == "n":
                            raise ValueError(
                                'Experiment execution was manually '
                                'interrupted!'
                            )
                    for p in model.cnn_module.parameters():
                        p.requires_grad = True
                    print("\t\tDone with warmup!")
                print("\tTraining ProbCBM's concept model")
                model.stage = 'concept'
                params_to_train = [
                    name for name, _ in model.named_parameters()
                    if name not in model.params_to_classify()
                ]
                # Make sure we unfreeze only the correct weights
                for name, parameter in model.named_parameters():
                    if name not in params_to_train:
                        parameter.requires_grad = False
                    elif name in trainable_params:
                        parameter.requires_grad = True
                concept_trainer = pl.Trainer(
                    max_epochs=config.get(
                        'max_concept_epochs',
                        config.get('max_epochs', None),
                    ) - config.get('warmup_epochs', 5),
                    **trainer_args,
                )
                concept_trainer.fit(model, train_dl, val_dl)
                if concept_trainer.interrupted:
                    reply = None
                    while reply not in ['y', 'n']:
                        if reply is not None:
                            print("Please provide only either 'y' or 'n'.")
                        reply = input(
                            "Would you like to manually interrupt this model's "
                            "training and continue the experiment? [y/n]\n"
                        ).strip().lower()
                    if reply == "n":
                        raise ValueError(
                            'Experiment execution was manually interrupted!'
                        )
                num_epochs = concept_trainer.current_epoch
                print("\tTraining ProbCBM's task model")
                model.stage = 'class'
                params_to_train = [
                    name for name, _ in model.named_parameters()
                    if name in model.params_to_classify()
                ]
                # Make sure we unfreeze only the correct weights
                for name, parameter in model.named_parameters():
                    if name not in params_to_train:
                        parameter.requires_grad = False
                    elif name in trainable_params:
                        parameter.requires_grad = True
                # Reset learning rates too
                for g, old_lr in zip(
                    model.optimizers().param_groups,
                    old_lrs,
                ):
                    g['lr'] = old_lr
                task_trainer = pl.Trainer(
                    max_epochs=config.get(
                        'max_task_epochs',
                        config.get('max_epochs', None),
                    ),
                    **trainer_args,
                )
                task_trainer.fit(model, train_dl, val_dl)
                if task_trainer.interrupted:
                    reply = None
                    while reply not in ['y', 'n']:
                        if reply is not None:
                            print("Please provide only either 'y' or 'n'.")
                        reply = input(
                            "Would you like to manually interrupt this model's "
                            "training and continue the experiment? [y/n]\n"
                        ).strip().lower()
                    if reply == "n":
                        raise ValueError(
                            'Experiment execution was manually interrupted!'
                        )
                num_epochs += task_trainer.current_epoch
                training_time = time.time() - start_time
            elif model.train_class_mode == 'joint':
                print("\tTraining ProbCBM jointly")
                task_trainer = pl.Trainer(
                    max_epochs=config['max_epochs'],
                    **trainer_args,
                )
                model.stage = 'joint'
                task_trainer.fit(model, train_dl, val_dl)
                if task_trainer.interrupted:
                    reply = None
                    while reply not in ['y', 'n']:
                        if reply is not None:
                            print("Please provide only either 'y' or 'n'.")
                        reply = input(
                            "Would you like to manually interrupt this model's "
                            "training and continue the experiment? [y/n]\n"
                        ).strip().lower()
                    if reply == "n":
                        raise ValueError(
                            'Experiment execution was manually interrupted!'
                        )
                num_epochs = task_trainer.current_epoch
                training_time = time.time() - start_time
            else:
                raise ValueError(
                    f'Currently we only support sequential or jointly '
                    f'trained ProbCBMs. We do not support '
                    f'train_class_mode = {model.train_class_mode}.'
                )
            config_copy = copy.deepcopy(config)
            if "c_extractor_arch" in config_copy and (
                not isinstance(config_copy["c_extractor_arch"], str)
            ):
                del config_copy["c_extractor_arch"]
            joblib.dump(
                config_copy,
                os.path.join(
                    result_dir,
                    f'{run_name}_experiment_config.joblib',
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
    eval_results = _evaluate_cbm(
        model=model,
        trainer=task_trainer,
        config=config,
        run_name=run_name,
        old_results=old_results,
        rerun=rerun,
        test_dl=test_dl,
        val_dl=val_dl,
    )
    eval_results['training_time'] = training_time
    eval_results['num_epochs'] = num_epochs
    if test_dl is not None:
        print(
            f'c_acc: {eval_results["test_acc_c"]*100:.2f}%, '
            f'y_acc: {eval_results["test_acc_y"]*100:.2f}%, '
            f'c_auc: {eval_results["test_auc_c"]*100:.2f}%, '
            f'y_auc: {eval_results["test_auc_y"]*100:.2f}% with '
            f'{num_epochs} epochs in {training_time:.2f} seconds'
        )
    return model, eval_results


def update_statistics(
    aggregate_results,
    run_config,
    test_results,
    run_name,
    model=None,
    prefix='',
):
    for key, val in test_results.items():
        aggregate_results[prefix + key] = val
