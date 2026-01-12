import copy
import joblib
import numpy as np
import os
import pytorch_lightning as pl
import time
import torch

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import cem.train.utils as utils
import cem.utils.data as data_utils

from cem.models.construction import (
    construct_model,
    construct_sequential_models,
)
import cem.train.evaluate as evaluate

def _make_callbacks(config, result_dir, full_run_name):
    callbacks = []
    ckpt_callback = None
    if config.get('early_stopping_monitor', None) is not None:
        callbacks.append(
            EarlyStopping(
                monitor=config["early_stopping_monitor"],
                min_delta=config.get("early_stopping_delta", 0.00),
                patience=config.get('patience', 5),
                verbose=config.get("verbose", False),
                mode=config.get("early_stopping_mode", "min"),
            )
        )
        if config.get('early_stopping_best_model', False):
            best_model_path = os.path.join(
                result_dir,
                f'checkpoints_{full_run_name}/',
            )
            callbacks.append(
                ModelCheckpoint(
                    save_top_k=1,
                    monitor=config["early_stopping_monitor"],
                    mode=config.get("early_stopping_mode", "min"),
                    dirpath=best_model_path,
                    every_n_epochs=config.get("check_val_every_n_epoch", 5),
                    save_on_train_epoch_end=False,
                )
            )
            ckpt_callback = callbacks[-1]

    return callbacks, ckpt_callback


def _check_interruption(trainer):
    if trainer.interrupted:
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


def _restore_checkpoint(
    model,
    max_epochs,
    ckpt_call,
    trainer,
):
    if (ckpt_call is not None) and (
        trainer.current_epoch != max_epochs
    ):
        # Then restore the best validation model
        chkpoint = torch.load(ckpt_call.best_model_path)
        model.load_state_dict(chkpoint["state_dict"])

################################################################################
## MODEL TRAINING
################################################################################

def train_end_to_end_model(
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl=None,
    input_shape=None,
    run_name=None,
    result_dir=None,
    split=None,
    imbalance=None,
    task_class_weights=None,
    rerun=False,
    logger=False,
    project_name='',
    seed=None,
    save_model=True,
    gradient_clip_val=0,
    old_results=None,
    enable_checkpointing=False,
    accelerator="auto",
    devices="auto",
):
    enable_checkpointing = (
        True if config.get('early_stopping_best_model', False)
        else enable_checkpointing
    )
    if seed is not None:
        seed_everything(seed)

    if run_name is None:
        run_name = "CBM"

    if split is not None:
        full_run_name = (
            f"{run_name}_fold_{split + 1}"
        )
    else:
        full_run_name = (
            f"{run_name}"
        )
    print(f"[Training {run_name} (trial {split + 1})]")
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
        callbacks, ckpt_call = _make_callbacks(config, result_dir, full_run_name)
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
        fit_trainer = trainer

        # Else it is time to train it
        model_saved_path = os.path.join(
            result_dir or ".",
            f'{full_run_name}.pt'
        )
        if (not rerun) and os.path.exists(model_saved_path):
            # Then we simply load the model and proceed
            print(f"\tFound cached model at {model_saved_path}... loading it")
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
            training_time = 0
            num_epochs = 0
            warmup_epochs = config.get('concept_warmup_epochs', 0)
            if warmup_epochs:
                print(
                    f"\tWarming up concept generator for {warmup_epochs} epochs"
                )
                warmup_callbacks, warmup_ckpt_call = _make_callbacks(
                    config,
                    result_dir,
                    full_run_name,
                )
                prev_task_loss_weight = model.task_loss_weight
                prev_concept_loss_weight = model.concept_loss_weight
                model.task_loss_weight = 0
                model.concept_loss_weight = 1
                warmup_trainer = pl.Trainer(
                    accelerator=accelerator,
                    devices=devices,
                    max_epochs=warmup_epochs,
                    check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                    callbacks=warmup_callbacks,
                    logger=logger or False,
                    enable_checkpointing=enable_checkpointing,
                    gradient_clip_val=gradient_clip_val,
                )
                if val_dl is not None:
                    warmup_trainer.fit(model, train_dl, val_dl)
                else:
                    warmup_trainer.fit(model, train_dl, val_dl)
                _check_interruption(warmup_trainer)
                _restore_checkpoint(
                    model=model,
                    max_epochs=warmup_epochs,
                    ckpt_call=warmup_ckpt_call,
                    trainer=warmup_trainer,
                )
                model.task_loss_weight = prev_task_loss_weight
                model.concept_loss_weight = prev_concept_loss_weight
                training_time += time.time() - start_time
                num_epochs += warmup_trainer.current_epoch
                start_time = time.time()
                print(
                    f"\t\tDone after {num_epochs} epochs and {training_time} secs"
                )

            print(
                f"\tTraining end-to-end model for {config['max_epochs']} epochs"
            )
            if val_dl is not None:
                fit_trainer.fit(model, train_dl, val_dl)
            else:
                fit_trainer.fit(model, train_dl, val_dl)
            _check_interruption(fit_trainer)
            training_time += time.time() - start_time
            num_epochs += fit_trainer.current_epoch
            _restore_checkpoint(
                model=model,
                max_epochs=config['max_epochs'],
                ckpt_call=ckpt_call,
                trainer=trainer,
            )

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
        eval_results = evaluate.evaluate_cbm(
            model=model,
            trainer=trainer,
            config=config,
            run_name=run_name,
            old_results=old_results,
            rerun=rerun,
            test_dl=train_dl, # Evaluate training metrics
            dl_name="train",
        )
        eval_results['training_time'] = training_time
        eval_results['num_epochs'] = num_epochs
        eval_results[f'num_trainable_params'] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        eval_results[f'num_non_trainable_params'] = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )
        print(
            f'c_acc: {eval_results["train_acc_c"]*100:.2f}%, '
            f'y_acc: {eval_results["train_acc_y"]*100:.2f}%, '
            f'c_auc: {eval_results["train_auc_c"]*100:.2f}%, '
            f'y_auc: {eval_results["train_auc_y"]*100:.2f}% with '
            f'{num_epochs} epochs in {training_time:.2f} seconds'
        )
    return model, eval_results


def train_sequential_model(
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl=None,
    input_shape=None,
    run_name=None,
    result_dir=None,
    split=None,
    imbalance=None,
    task_class_weights=None,
    rerun=False,
    logger=False,
    project_name='',
    seed=None,
    save_model=True,
    accelerator="auto",
    devices="auto",
    old_results=None,
    enable_checkpointing=False,
    gradient_clip_val=0,
):
    if run_name is None:
        run_name = "SequentialCBM"
    enable_checkpointing = (
        True if config.get('early_stopping_best_model', False)
        else enable_checkpointing
    )
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
    print(f"[Training {full_run_name} (trial {split + 1})]")
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
        callbacks, ckpt_call = _make_callbacks(config, result_dir, full_run_name)
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            # We will distribute half epochs in one model and half on the other
            max_epochs=config['max_epochs'],
            check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
            callbacks=callbacks,
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
        x2c_trainer = trainer
        if (not rerun) and chpt_exists:
            # Then we simply load the model and proceed
            print(f"\tFound cached model at {model_saved_path}... loading it")
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
            if val_dl is not None:
                x2c_trainer.fit(model, train_dl, val_dl)
            else:
                x2c_trainer.fit(model, train_dl)
            _check_interruption(x2c_trainer)
            training_time += time.time() - start_time
            num_epochs += x2c_trainer.current_epoch
            _restore_checkpoint(
                model=model,
                max_epochs=config['max_epochs'],
                ckpt_call=ckpt_call,
                trainer=x2c_trainer,
            )

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
            callbacks, ckpt_call = _make_callbacks(config, result_dir, full_run_name)
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
                callbacks=callbacks,
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
            _check_interruption(seq_c2y_trainer)
            seq_training_time = training_time + time.time() - start_time
            seq_num_epochs = num_epochs + seq_c2y_trainer.current_epoch
            if config.get('early_stopping_best_model', False) and (
                seq_c2y_trainer.current_epoch != config['max_epochs']
            ):
                # Then restore the best validation model
                chkpoint = torch.load(ckpt_call.best_model_path)
                seq_c2y_model.load_state_dict(chkpoint["state_dict"])
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

    eval_results = evaluate.evaluate_cbm(
        model=model,
        trainer=trainer,
        config=config,
        run_name=run_name,
        old_results=old_results,
        rerun=rerun,
        test_dl=train_dl, # Evaluate training metrics
        dl_name="train",
    )
    eval_results['training_time'] = training_time
    eval_results['num_epochs'] = num_epochs
    eval_results[f'num_trainable_params'] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    eval_results[f'num_non_trainable_params'] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    print(
        f'c_acc: {eval_results["train_acc_c"]*100:.2f}%, '
        f'y_acc: {eval_results["train_acc_y"]*100:.2f}%, '
        f'c_auc: {eval_results["train_auc_c"]*100:.2f}%, '
        f'y_auc: {eval_results["train_auc_y"]*100:.2f}% with '
        f'{num_epochs} epochs in {training_time:.2f} seconds'
    )
    return seq_model, eval_results



def train_independent_model(
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl=None,
    input_shape=None,
    run_name=None,
    result_dir=None,
    split=None,
    imbalance=None,
    task_class_weights=None,
    rerun=False,
    logger=False,
    project_name='',
    seed=None,
    save_model=True,
    accelerator="auto",
    devices="auto",
    old_results=None,
    enable_checkpointing=False,
    gradient_clip_val=0,
):
    if run_name is None:
        run_name = "IndependentCBM"
    enable_checkpointing = (
        True if config.get('early_stopping_best_model', False)
        else enable_checkpointing
    )
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
    print(f"[Training {full_run_name} (trial {split + 1})]")
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
        callbacks, ckpt_call = _make_callbacks(config, result_dir, full_run_name)
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            # We will distribute half epochs in one model and half on the other
            max_epochs=config['max_epochs'],
            check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
            callbacks=callbacks,
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
        x2c_trainer = trainer

        if (not rerun) and chpt_exists:
            # Then we simply load the model and proceed
            print(f"\tFound cached model at {model_saved_path}... loading it")
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
            if val_dl is not None:
                x2c_trainer.fit(model, train_dl, val_dl)
            else:
                x2c_trainer.fit(model, train_dl)

            _check_interruption(x2c_trainer)

            training_time += time.time() - start_time
            num_epochs += x2c_trainer.current_epoch
            if config.get('early_stopping_best_model', False) and (
                x2c_trainer.current_epoch != config['max_epochs']
            ):
                # Then restore the best validation model
                chkpoint = torch.load(ckpt_call.best_model_path)
                model.load_state_dict(chkpoint["state_dict"])
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
            callbacks, ckpt_call = _make_callbacks(config, result_dir, full_run_name)
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
                callbacks=callbacks,
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
            _check_interruption(ind_c2y_trainer)
            ind_training_time = training_time + time.time() - start_time
            ind_num_epochs = num_epochs + ind_c2y_trainer.current_epoch
            if config.get('early_stopping_best_model', False) and (
                ind_c2y_trainer.current_epoch != config['max_epochs']
            ):
                # Then restore the best validation model
                chkpoint = torch.load(ckpt_call.best_model_path)
                ind_c2y_model.load_state_dict(chkpoint["state_dict"])
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
    eval_results = evaluate.evaluate_cbm(
        model=model,
        trainer=trainer,
        config=config,
        run_name=run_name,
        old_results=old_results,
        rerun=rerun,
        test_dl=train_dl, # Evaluate training metrics
        dl_name="train",
    )
    eval_results['training_time'] = training_time
    eval_results['num_epochs'] = num_epochs
    eval_results[f'num_trainable_params'] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    eval_results[f'num_non_trainable_params'] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    print(
        f'c_acc: {eval_results["train_acc_c"]*100:.2f}%, '
        f'y_acc: {eval_results["train_acc_y"]*100:.2f}%, '
        f'c_auc: {eval_results["train_auc_c"]*100:.2f}%, '
        f'y_auc: {eval_results["train_auc_y"]*100:.2f}% with '
        f'{num_epochs} epochs in {training_time:.2f} seconds'
    )
    return ind_model, eval_results
