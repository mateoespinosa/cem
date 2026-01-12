import copy
import joblib
import logging
import numpy as np
import os
import pytorch_lightning as pl
import time
import torch

from pytorch_lightning import seed_everything
from scipy.special import expit
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tensorflow as tf

import cem.metrics.niching as niching
import cem.metrics.oracle as oracle
import cem.train.utils as utils
import cem.utils.data as data_utils
import cem.metrics.accs as accs

from cem.metrics.cas import concept_alignment_score
from cem.models.cbm import ConceptBottleneckModel
from cem.models.construction import load_trained_model


def evaluate_cbm(
    model,
    trainer,
    config,
    run_name,
    old_results=None,
    rerun=False,
    test_dl=None,
    dl_name="test",
):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        eval_results = {}
        if test_dl is None:
            return eval_results
        include_c_aucs = config.get('include_c_aucs', False)
        def _inner_call():
            batch_results = trainer.predict(model, test_dl)
            y_true, c_true = data_utils.daloader_to_memory(
                test_dl,
                as_torch=True,
                only_labels=True,
                num_workers=config.get('num_load_workers', config.get('num_workers', 1)),
            )
            c_pred = torch.cat(
                list(map(lambda x: x[0].detach().cpu(), batch_results)),
                dim=0,
            )
            y_pred = torch.cat(
                list(map(lambda x: x[2].detach().cpu(), batch_results)),
                axis=0,
            )
            c_metrics, (y_accuracy, y_auc, y_f1) = \
                accs.compute_accuracy(
                    c_pred=c_pred,
                    y_pred=y_pred,
                    c_true=c_true,
                    y_true=y_true,
                    include_c_aucs=include_c_aucs,
                )

            if include_c_aucs:
                (c_accuracy, c_auc, c_f1, c_aucs_vector) = c_metrics
                output = [
                    c_accuracy,
                    y_accuracy,
                    c_auc,
                    y_auc,
                    c_f1,
                    y_f1,
                    c_aucs_vector,
                ]
            else:
                (c_accuracy, c_auc, c_f1) = c_metrics
                output = [
                    c_accuracy,
                    y_accuracy,
                    c_auc,
                    y_auc,
                    c_f1,
                    y_f1,
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

        if include_c_aucs:
            keys = [
                f"{dl_name}_acc_c",
                f"{dl_name}_acc_y",
                f"{dl_name}_auc_c",
                f"{dl_name}_auc_y",
                f"{dl_name}_f1_c",
                f"{dl_name}_aucs_c",
            ]
        else:
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
    if was_training:
        model.train()
    return eval_results


def evaluate_metric(
    metric_name,
    model,
    trainer,
    config,
    run_name,
    metric_config={},
    old_results=None,
    rerun=False,
    test_dl=None,
    dl_name="test",
    current_results=None,
):
    eval_results = {}
    if test_dl is None:
        return eval_results
    if metric_name == 'mixcem_sel':
        keys = [f"{dl_name}_mixcem_sel_acc", f"{dl_name}_mixcem_sel_auc"]
        current_results = current_results or {}
        def _inner_call():
            accs_to_include = []
            aucs_to_include = []
            y_true = None
            if (f"{dl_name}_acc_y" in current_results) and (
                f"{dl_name}_auc_y" in current_results
            ):
                vanilla_y_accuracy = current_results[f"{dl_name}_acc_y"]
                vanilla_y_auc = current_results[f"{dl_name}_auc_y"]
            else:
                y_true, _ = data_utils.daloader_to_memory(
                    test_dl,
                    as_torch=True,
                    only_labels=True,
                    num_workers=config.get('num_load_workers', config.get('num_workers', 1)),
                )
                # Make predictions without any interventions or global embedding use
                vanilla_batch_results = trainer.predict(model, test_dl)
                vanilla_y_pred = torch.cat(
                    list(map(lambda x: x[2].detach().cpu(), vanilla_batch_results)),
                    axis=0,
                )
                _, (vanilla_y_accuracy, vanilla_y_auc, _) = \
                    accs.compute_accuracy(
                        y_pred=vanilla_y_pred,
                        y_true=y_true,
                    )
            accs_to_include.append(vanilla_y_accuracy)
            aucs_to_include.append(vanilla_y_auc)



            # Force the model to intervene on all concepts
            if y_true is None:
                y_true, _ = data_utils.daloader_to_memory(
                    test_dl,
                    as_torch=True,
                    only_labels=True,
                    num_workers=config.get(
                        'num_load_workers',
                        config.get('num_workers', 1),
                    ),
                )
            prev_force_all_interventions = getattr(model, 'force_all_interventions', False)
            model.force_all_interventions = True
            all_int_batch_results = trainer.predict(model, test_dl)
            all_int_y_pred = torch.cat(
                list(map(lambda x: x[2].detach().cpu(), all_int_batch_results)),
                axis=0,
            )
            _, (all_int_y_accuracy, all_int_y_auc, _) = \
                accs.compute_accuracy(
                    y_pred=all_int_y_pred,
                    y_true=y_true,
                )
            accs_to_include.append(all_int_y_accuracy)
            aucs_to_include.append(all_int_y_auc)

            # Force the selection to always use the global embeddings when we
            # intervene on, and compute that accuracy
            if hasattr(model, 'hard_selection_value'):
                prev_hard_selection_value = model.hard_selection_value
                model.hard_selection_value = 1
                global_int_batch_results = trainer.predict(model, test_dl)
                global_int_y_pred = torch.cat(
                    list(map(lambda x: x[2].detach().cpu(), global_int_batch_results)),
                    axis=0,
                )
                _, (global_int_y_accuracy, global_int_y_auc, _) = \
                    accs.compute_accuracy(
                        y_pred=global_int_y_pred,
                        y_true=y_true,
                    )
                accs_to_include.append(global_int_y_accuracy)
                aucs_to_include.append(global_int_y_auc)


            # Reset the model accordingly
            if hasattr(model, 'hard_selection_value'):
                model.hard_selection_value = prev_hard_selection_value
            model.force_all_interventions = prev_force_all_interventions
            return [np.mean(accs_to_include), np.mean(aucs_to_include)]
    else:
        raise ValueError(
            f'Unsupported metric "{metric_name}".'
        )
    values, _ = utils.load_call(
        function=_inner_call,
        keys=keys,
        run_name=run_name,
        old_results=old_results,
        rerun=rerun,
        kwargs={},
    )
    for (key, val) in zip(keys, values):
        eval_results[key] = val
        print(
            f'{dl_name} {key}: {val}'
        )
    return eval_results


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
        classifier.fit(
            c_embs_train[:,concept_idx,:],
            y_train,
            **predictor_train_kwags,
        )
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
    run_name,
    split=0,
    train_dl=None,
    imbalance=None,
    result_dir=None,
    task_class_weights=None,
    accelerator="auto",
    devices="auto",
    rerun=False,
    seed=None,
    old_results=None,
    test_subsampling=1,
    dl_name="test",
    skip_datasets=None,
):
    result_dict = {}
    if config.get("rerun_repr_evaluation", False):
        rerun = True
    if config.get("skip_repr_evaluation", False) or (
        dl_name in (skip_datasets or [])
    ):
        return {}
    test_subsampling = config.get(
        'test_repr_subsampling',
        config.get('test_subsampling', test_subsampling),
    )
    if seed is not None:
        seed_everything(seed)

    x_test, y_test, c_test = data_utils.daloader_to_memory(
        test_dl,
        as_torch=True,
        num_workers=config.get('num_load_workers', config.get('num_workers', 1)),
    )
    c_test_np = None
    y_test_np = None

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
                x_test,
                y_test,
                c_test,
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
    # if config.get('extra_dims', 0) != 0:
    #     # Then we will only use the extra dims as the embedding as those
    #     # correspond to the learnt embeddings only
    #     c_pred = c_pred[:, -config.get('extra_dims', 0):]

    if getattr(cbm, 'extra_dims', 0) > 0:
        # Then we will copy the shared extra capacity dimensions into the
        # representations of all concepts to generate a (B, n_concepts, extra_dims + 1)
        # tensor
        c_pred_expanded = np.zeros(
            (c_pred.shape[0], n_concepts, cbm.extra_dims + 1),
        )
        for concept_idx in range(n_concepts):
            c_pred_expanded[:, concept_idx, 0] = c_pred[:, concept_idx]
            c_pred_expanded[:, concept_idx, 1:] = c_pred[:, -cbm.extra_dims:]
        c_pred = c_pred_expanded
    else:
        c_pred = np.reshape(c_pred, (c_test.shape[0], n_concepts, -1))
    c_soft_train = None
    y_train_np = None
    c_train = None
    if config.get("run_repr_task_matrix", False):
        repr_task_key = f'{dl_name}_repr_task_matrix'
        logging.info(f"Computing Representation-to-task Matrix...")
        if y_test_np is None:
            y_test_np = y_test.detach().cpu().numpy()
        if train_dl is not None:
            y_train, c_train = data_utils.daloader_to_memory(
                train_dl,
                as_torch=True,
                num_workers=config.get(
                    'num_load_workers',
                    config.get('num_workers', 1),
                ),
                only_labels=True,
            )
            train_batch_results = trainer.predict(cbm, train_dl)
            c_train_pred = np.concatenate(
                list(map(
                    lambda x: x[1].detach().cpu().numpy(),
                    train_batch_results,
                )),
                axis=0,
            )
            if getattr(cbm, 'extra_dims', 0) > 0:
                # Then we will copy the shared extra capacity dimensions into the
                # representations of all concepts to generate a
                # (B, n_concepts, extra_dims + 1) tensor
                c_train_pred_expanded = np.zeros(
                    (c_train_pred.shape[0], n_concepts, cbm.extra_dims + 1),
                )
                for concept_idx in range(n_concepts):
                    c_train_pred_expanded[:, concept_idx, 0] = \
                        c_train_pred[:, concept_idx]
                    c_train_pred_expanded[:, concept_idx, 1:] = \
                        c_train_pred[:, -cbm.extra_dims:]
                c_train_pred = c_train_pred_expanded
            else:
                c_train_pred = np.reshape(
                    c_train_pred,
                    (y_train.shape[0], n_concepts, -1),
                )
            c_soft_train = np.transpose(c_train_pred, (0, 2, 1))
            y_train_np = y_train.detach().cpu().numpy()
        else:
            c_soft_train = None
            y_train_np = None

        repr_task_matrix, loaded = utils.execute_and_save(
            fun=utils.load_call,
            kwargs=dict(
                keys=[repr_task_key],
                old_results=old_results,
                rerun=rerun,
                function=oracle.task_purity_matrix,
                run_name=run_name,
                kwargs=dict(
                    c_soft=np.transpose(c_pred, (0, 2, 1)),
                    y_true=y_test_np,
                    c_soft_train=c_soft_train,
                    y_true_train=y_train_np,
                    predictor_train_kwags={
                        'epochs': config.get("ois_epochs", 50),
                        'batch_size': min(
                            512,
                            (
                                y_test_np.shape[0] if y_train_np is None
                                else y_train_np.shape[0]
                            ),
                        ),
                        'verbose': 0,
                    },
                    test_size=0.2,
                ),
            ),
            result_dir=result_dir,
            filename=f'{repr_task_key}_{run_name}_split_{split}.joblib',
            rerun=rerun,
        )
        logging.info(f"\tDone.... mean repr task matrix score is {np.mean(repr_task_matrix)*100:.2f}%")
        logging.info(f"\t........ repr task matrix is {repr_task_matrix}")
        result_dict[repr_task_key] = repr_task_matrix
        result_dict["mean_" + repr_task_key] = np.mean(repr_task_matrix)

    oracle_matrix = None
    purity_matrix = None
    if config.get("run_purity_matrix", False):
        purity_key = f'{dl_name}_purity_matrix'
        logging.info(f"Computing Purity Matrix...")
        if c_test_np is None:
            c_test_np = c_test.detach().cpu().numpy()

        if ((c_soft_train is None) or (c_train is None)) and (
            train_dl is not None
        ):
            _, c_train = data_utils.daloader_to_memory(
                train_dl,
                as_torch=True,
                num_workers=config.get('num_load_workers', config.get('num_workers', 1)),
                only_labels=True,
            )
            train_batch_results = trainer.predict(cbm, train_dl)
            c_train_pred = np.concatenate(
                list(map(lambda x: x[1].detach().cpu().numpy(), train_batch_results)),
                axis=0,
            )
            if getattr(cbm, 'extra_dims', 0) > 0:
                # Then we will copy the shared extra capacity dimensions into the
                # representations of all concepts to generate a
                # (B, n_concepts, extra_dims + 1) tensor
                c_train_pred_expanded = np.zeros(
                    (c_train_pred.shape[0], n_concepts, cbm.extra_dims + 1),
                )
                for concept_idx in range(n_concepts):
                    c_train_pred_expanded[:, concept_idx, 0] = \
                        c_train_pred[:, concept_idx]
                    c_train_pred_expanded[:, concept_idx, 1:] = \
                        c_train_pred[:, -cbm.extra_dims:]
                c_train_pred = c_train_pred_expanded
            else:
                c_train_pred = np.reshape(
                    c_train_pred,
                    (y_train.shape[0], n_concepts, -1),
                )
            c_soft_train = np.transpose(c_train_pred, (0, 2, 1))
        if c_train is not None:
            c_train_np = c_train.detach().cpu().numpy()
        purity_matrix, loaded = utils.execute_and_save(
            fun=utils.load_call,
            kwargs=dict(
                keys=[purity_key],
                old_results=old_results,
                rerun=rerun,
                function=oracle.concept_purity_matrix,
                run_name=run_name,
                kwargs=dict(
                    c_soft=np.transpose(c_pred, (0, 2, 1)),
                    c_true=c_test_np,
                    c_soft_train=c_soft_train,
                    c_true_train=c_train_np,
                    predictor_train_kwags={
                        'epochs': config.get("ois_epochs", 50),
                        'batch_size': min(2048, c_test_np.shape[0]),
                        'verbose': 0,
                    },
                    test_size=0.2,
                    jointly_learnt=True,
                ),
            ),
            result_dir=result_dir,
            filename=f'{purity_key}_{run_name}_split_{split}.joblib',
            rerun=rerun,
        )
        logging.info(f"\tDone.... mean purity matrix score is {np.mean(purity_matrix)*100:.2f}%")
        logging.info(f"\tDone.... mean diag purity matrix score is {np.mean(np.diag(purity_matrix))*100:.2f}%")
        logging.info(f"\tDone.... mean off-diag purity matrix score is {np.sum(purity_matrix - np.diag(np.diag(purity_matrix)))*100/(n_concepts*(n_concepts-1)):.2f}%")
        logging.info(f"\t........ diagonal of purity matrix is {np.diag(purity_matrix)}")
        result_dict[purity_key] = purity_matrix
        result_dict["mean_" + purity_key] = np.mean(purity_matrix)
        result_dict["mean_diag_" + purity_key] = np.mean(np.diag(purity_matrix))
        result_dict["mean_offdiag_" + purity_key] = np.sum(purity_matrix - np.diag(np.diag(purity_matrix)))/(n_concepts*(n_concepts-1))

    if config.get("run_ois", True):
        ois_key = f'{dl_name}_ois'
        logging.info(f"Computing OIS score...")
        if os.path.exists(
            os.path.join(result_dir, f'oracle_matrix.npy')
        ):
            oracle_matrix = np.load(
                os.path.join(result_dir, f'oracle_matrix.npy')
            )
        if c_test_np is None:
            c_test_np = c_test.detach().cpu().numpy()
        ois, loaded = utils.execute_and_save(
            fun=utils.load_call,
            kwargs=dict(
                keys=[ois_key],
                old_results=old_results,
                rerun=rerun,
                function=oracle.oracle_impurity_score,
                run_name=run_name,
                kwargs=dict(
                    c_soft=np.transpose(c_pred, (0, 2, 1)),
                    c_true=c_test_np,
                    predictor_train_kwags={
                        'epochs': config.get("ois_epochs", 50),
                        'batch_size': min(2048, c_test_np.shape[0]),
                        'verbose': 0,
                    },
                    test_size=0.2,
                    oracle_matrix=oracle_matrix,
                    jointly_learnt=True,
                    output_matrices=True,
                    purity_matrix=purity_matrix,
                ),
            ),
            result_dir=result_dir,
            filename=f'{ois_key}_{run_name}_split_{split}.joblib',
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
        x_train, y_train, c_train = data_utils.daloader_to_memory(
            train_dl,
            as_torch=True,
            num_workers=config.get('num_load_workers', config.get('num_workers', 1)),
        )

        used_train_dl = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                x_train,
                y_train,
                c_train,
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

        repr_task_pred_key = f'{dl_name}_repr_task_pred'
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
                run_name=run_name,
                kwargs=dict(
                    c_embs_train=c_pred_train,
                    c_embs_test=c_pred,
                    y_train=y_train,
                    y_test=y_test,
                ),
            ),
            result_dir=result_dir,
            filename=f'{repr_task_pred_key}_{run_name}_split_{split}.joblib',
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
        nis_key = f'{dl_name}_nis'
        logging.info(f"Computing NIS score...")
        if c_test_np is None:
            c_test_np = c_test.detach().cpu().numpy()
        nis, loaded = utils.execute_and_save(
            fun=utils.load_call,
            kwargs=dict(
                keys=[nis_key],
                old_results=old_results,
                rerun=rerun,
                function=niching.niche_impurity_score,
                run_name=run_name,
                kwargs=dict(
                    c_soft=np.transpose(c_pred, (0, 2, 1)),
                    c_true=c_test_np,
                    test_size=0.2,
                ),
            ),
            result_dir=result_dir,
            filename=f'{nis_key}_{run_name}_split_{split}.joblib',
            rerun=rerun,
        )
        if isinstance(nis, (tuple, list)):
            assert len(nis) == 1
            nis = nis[0]
        logging.info(f"\tDone....NIS score is {nis*100:.2f}%")
        result_dict[nis_key] = nis

    if config.get("run_cas", True):
        cas_key = f'{dl_name}_cas'
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
                run_name=run_name,
                kwargs=dict(
                    c_vec=c_pred,
                    c_test=c_test,
                    y_test=y_test,
                    step=config.get('cas_step', 50),
                ),
            ),
            result_dir=result_dir,
            filename=f'{cas_key}_{run_name}_split_{split}.joblib',
            rerun=rerun,
        )
        if isinstance(cas, (tuple, list)):
            cas = cas[0]
        logging.info(f"\tDone....CAS score is {cas*100:.2f}%")
        result_dict[cas_key] = cas

    return result_dict