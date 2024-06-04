import numpy as np
import pytorch_lightning as pl

import cem.interventions.utils as intervention_utils
import cem.train.evaluate as evaluate


def evaluate_model(
    model,
    config,
    test_datasets,
    train_dl,
    val_dl=None,
    run_name=None,
    task_class_weights=None,
    imbalance=None,
    acquisition_costs=None,
    result_dir=None,
    concept_map=None,
    intervened_groups=None,
    accelerator="auto",
    devices="auto",
    split=0,
    rerun=False,
    old_results=None,
):
    eval_results = {}
    eval_trainer = pl.Trainer(
        accelerator=accelerator,
            devices=devices,
            max_epochs=-1,
            logger=False,
            enable_checkpointing=False,
    )
    for test_dataloader, dl_name in test_datasets:
        eval_results.update(evaluate.evaluate_cbm(
            model=model,
            trainer=eval_trainer,
            config=config,
            run_name=run_name,
            old_results=old_results,
            rerun=rerun,
            test_dl=test_dataloader,
            dl_name=dl_name,
        ))

        print(
            f'{dl_name} c_acc: {eval_results[f"{dl_name}_acc_c"]*100:.2f}%, '
            f'{dl_name} y_acc: {eval_results[f"{dl_name}_acc_y"]*100:.2f}%, '
            f'{dl_name} c_auc: {eval_results[f"{dl_name}_auc_c"]*100:.2f}%, '
            f'{dl_name} y_auc: {eval_results[f"{dl_name}_auc_y"]*100:.2f}%'
        )
        if (f'{dl_name}_intervention_config' in config) or (
            'intervention_config' in config
        ):
            if f'{dl_name}_intervention_config' in config:
                intervention_config = config[f'{dl_name}_intervention_config']
            else:
                intervention_config = config['intervention_config']
            test_int_args = dict(
                task_class_weights=task_class_weights,
                run_name=run_name,
                train_dl=train_dl,
                val_dl=val_dl,
                imbalance=imbalance,
                config=config,
                n_tasks=config['n_tasks'],
                n_concepts=config['n_concepts'],
                acquisition_costs=acquisition_costs,
                result_dir=result_dir,
                concept_map=concept_map,
                intervened_groups=intervened_groups,
                accelerator=accelerator,
                devices=devices,
                split=split,
                rerun=rerun,
                old_results=old_results,
                group_level_competencies=intervention_config.get(
                    "group_level_competencies",
                    False,
                ),
                competence_levels=intervention_config.get(
                    'competence_levels',
                    [1],
                ),
            )
            if "real_competencies" in intervention_config:
                for real_comp in intervention_config['real_competencies']:
                    def _real_competence_generator(x):
                        if real_comp == "same":
                            return x
                        if real_comp == "complement":
                            return 1 - x
                        if test_int_args['group_level_competencies']:
                            if real_comp == "unif":
                                batch_group_level_competencies = np.zeros(
                                    (x.shape[0], len(concept_map))
                                )
                                for batch_idx in range(x.shape[0]):
                                    for group_idx, (_, concept_members) in enumerate(
                                        concept_map.items()
                                    ):
                                        batch_group_level_competencies[
                                            batch_idx,
                                            group_idx,
                                        ] = np.random.uniform(
                                            1/len(concept_members),
                                            1,
                                        )
                            else:
                                batch_group_level_competencies = np.ones(
                                    (x.shape[0], len(concept_map))
                                ) * real_comp
                            return batch_group_level_competencies

                        if real_comp == "unif":
                            return np.random.uniform(
                                0.5,
                                1,
                                size=x.shape,
                            )
                        return np.ones(x.shape) * real_comp
                    if real_comp == "same":
                        # Then we will just run what we normally run
                        # as the provided competency matches the level
                        # of competency of the user
                        test_int_args.pop(
                            "real_competence_generator",
                            None,
                        )
                        test_int_args.pop(
                            "extra_suffix",
                            None,
                        )
                        test_int_args.pop(
                            "real_competence_level",
                            None,
                        )
                    else:
                        test_int_args['real_competence_generator'] = \
                                _real_competence_generator
                        test_int_args['extra_suffix'] = \
                            f"_real_comp_{real_comp}_"
                        test_int_args["real_competence_level"] = \
                            real_comp

            eval_results.update(intervention_utils.test_interventions(
                    dl_name=dl_name,
                    test_dl=test_dataloader,
                    **test_int_args
                ),
            )

        eval_results.update(evaluate.evaluate_representation_metrics(
                config=config,
                n_concepts=config['n_concepts'],
                n_tasks=config['n_tasks'],
                test_dl=test_dataloader,
                run_name=run_name,
                split=split,
                imbalance=imbalance,
                result_dir=result_dir,
                task_class_weights=task_class_weights,
                accelerator=accelerator,
                devices=devices,
                rerun=rerun,
                seed=42,
                old_results=old_results,
            )
        )

    return eval_results