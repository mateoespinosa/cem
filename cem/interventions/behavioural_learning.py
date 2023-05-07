import numpy as np
import torch
import logging
import pytorch_lightning as pl
import os
from tqdm import tqdm

from cem.interventions.intervention_policy import InterventionPolicy
from cem.interventions.optimal import GreedyOptimal
from cem.train.utils import WrapperModule


##########################
## Behavioral Cloning Policy Definition
##########################

class BehavioralLearningPolicy(InterventionPolicy):
    # Behavioral Learning Policy
    def __init__(
        self,
        concept_group_map,
        cbm,
        n_concepts,
        n_tasks,
        x_train,
        y_train,
        c_train,
        emb_size,
        full_run_name,
        max_horizon=None,
        batch_size=256,
        teacher_policy=GreedyOptimal,
        teacher_policy_kwargs=None,
        result_dir=".",
        train_epochs=100,
        use_concept_groups=False,
        num_groups_intervened=0,
        group_based=True,
        include_prior=False,
        dataset_size=256, #5000,
        horizon_rate=1.001,
        gpu=int(torch.cuda.is_available()),
        rerun=False,
        **kwargs
    ):
        self.n_tasks = n_tasks
        self.n_concepts = n_concepts
        max_horizon = max_horizon or int(np.ceil(n_concepts/2))
        self.num_groups_intervened = num_groups_intervened
        self.concept_group_map = concept_group_map
        self.group_based = group_based
        self.cbm = cbm
        self.include_prior = include_prior
        teacher_policy_kwargs = teacher_policy_kwargs or {"include_prior": False}
        self.emb_size = emb_size
        units = [
            n_concepts * self.emb_size + # Bottleneck
            n_concepts + # Prev interventions
            1 # Horizon
        ] + [256, 128, len(self.concept_group_map) if use_concept_groups else n_concepts]
        layers = []
        for i in range(1, len(units)):
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != len(units) - 1:
                layers.append(torch.nn.LeakyReLU())
        self.behavior_cloner = WrapperModule(
            model=torch.nn.Sequential(*layers),
            n_tasks=self.n_concepts, # One output per task
            momentum=0.9,
            learning_rate=0.01,
            weight_decay=4e-05,
            optimizer="sgd",
            top_k_accuracy=2,
            binary_output=False,
            weight_loss=None,
            sigmoidal_output=False,
        )
        model_saved_path = os.path.join(
            result_dir,
            f"behaviour_clone_model_{full_run_name}.pkl",
        )
        if rerun or (not os.path.exists(model_saved_path)):
            bc_train_ds = self._generate_behavioral_cloning_dataset(
                x_train=x_train,
                c_train=c_train,
                y_train=y_train,
                teacher_policy=teacher_policy,
                teacher_policy_kwargs=teacher_policy_kwargs,
                max_horizon=max_horizon,
                dataset_size=dataset_size,
                horizon_rate=horizon_rate,
                compute_batch_size=batch_size,
                batch_size=batch_size,
            )
            trainer = pl.Trainer(
                gpus=gpu,
                max_epochs=train_epochs,
                logger=False,
            )
            trainer.fit(self.behavior_cloner, bc_train_ds)
            torch.save(
                self.behavior_cloner.state_dict(),
                model_saved_path,
            )
        else:
            self.behavior_cloner.load_state_dict(torch.load(model_saved_path))

    def _compute_model_input(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        c,
        horizon,
        competencies=None,
        prev_interventions=None,
        use_concept_groups=False,
    ):
        if prev_interventions is None:
            prev_interventions = np.zeros(prob.shape)
        if competencies is None:
            competencies = np.ones(prob.shape)
        # Shape is [B, n_concepts, emb_size]
        prob = prev_interventions * c + (1 - prev_interventions) * prob
        embeddings = (
            np.expand_dims(prob, axis=-1) * pos_embeddings +
            (1 - np.expand_dims(prob, axis=-1)) * neg_embeddings
        )
        # Zero out embeddings of previously intervened concepts
        if use_concept_groups:
            available_groups = np.zeros((embeddings.shape[0], len(self.concept_group_map)))
            for group_idx, (_, group_concepts) in enumerate(self.concept_group_map.items()):
                available_groups[:, group_idx] = np.logical_not(np.any(
                    prev_interventions[:, group_concepts] > (1/len(self.concept_group_map)),
                ))
        else:
            available_groups = (1 - prev_interventions)

        emb_size = pos_embeddings.shape[-1]
        return np.concatenate(
            [
                np.reshape(embeddings, [-1, emb_size * self.n_concepts]),
                prev_interventions,
                np.ones((prev_interventions.shape[0], 1)) * horizon,
#                     competencies,
            ],
            axis=-1,
        )


    def _generate_behavioral_cloning_dataset(
        self,
        x_train,
        c_train,
        y_train,
        teacher_policy,
        teacher_policy_kwargs,
        max_horizon,
        dataset_size=5000,
        horizon_rate=1.001,
        compute_batch_size=256,
        batch_size=256,
    ):
        inputs = []
        targets = []
        horizon_limit = 1

        prev_policy = self.cbm.intervention_policy
        self.cbm.intervention_policy = teacher_policy(
            cbm=self.cbm,
            concept_group_map=self.concept_group_map,
            n_concepts=self.n_concepts,
            n_tasks=self.n_tasks,
            num_groups_intervened=1,
            group_based=self.group_based,
            **teacher_policy_kwargs,
        )
        print("Generating BC dataset....")
        latent = None
        x_train = torch.FloatTensor(x_train)
        c_train = torch.FloatTensor(c_train)
        y_train = torch.LongTensor(y_train)
        for sample_idx in tqdm(range(dataset_size//compute_batch_size)):
            # Sample an initial mask to start with
            competencies = None
            current_horizon = np.random.randint(
                0,
                min(int(horizon_limit), max_horizon),
            )
            initially_selected = np.random.randint(
                0,
                min(int(horizon_limit), self.n_concepts),
            )
            # Generate a sample of inputs we will use to learn embeddings
            selected_samples = np.random.choice(x_train.shape[0], replace=False, size=compute_batch_size)
            current_horizon = min(current_horizon, self.n_concepts - initially_selected)
            prev_interventions = np.zeros((len(selected_samples), self.n_concepts))
            for sample_idx in range(prev_interventions.shape[0]):
                prev_interventions[sample_idx, np.random.choice(self.n_concepts, size=initially_selected, replace=False)] = 1


            outputs = self.cbm._forward(
                x_train[selected_samples],
                intervention_idxs=None,
                c=c_train[selected_samples],
                y=y_train[selected_samples],
                train=False,
                competencies=competencies,
                prev_interventions=torch.FloatTensor(prev_interventions),
                output_embeddings=True,
                latent=latent,
                output_latent=True,
            )
            next_mask = outputs[3].detach().cpu().numpy()
            latent = outputs[4]
            pos_embeddings = outputs[-2].detach().cpu().numpy()
            neg_embeddings = outputs[-1].detach().cpu().numpy()
            next_inputs = self._compute_model_input(
                prob=outputs[0].detach().cpu().numpy(),
                pos_embeddings=pos_embeddings,
                neg_embeddings=neg_embeddings,
                c=c_train[selected_samples].detach().cpu().numpy(),
                competencies=competencies,
                prev_interventions=prev_interventions,
                use_concept_groups=False,
                horizon=current_horizon,
            )
            horizon_limit = min(horizon_rate * horizon_limit, max_horizon)
            inputs.append(next_inputs)
            next_intervention = np.argmax(next_mask - prev_interventions, axis=-1)
            targets.append(next_intervention)
        inputs = np.concatenate(inputs, axis=0)
        targets = np.concatenate(targets, axis=0)
        data = torch.utils.data.TensorDataset(
            torch.FloatTensor(inputs),
            torch.LongTensor(targets),
        )
        return torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
        )

    def _next_intervention(
        self,
        x,
        pred_c,
        c,
        competencies=None,
        prev_interventions=None,
        prior_distribution=None,
        latent=None,
    ):
        outputs = self.cbm._forward(
            x,
            intervention_idxs=torch.zeros(c.shape).to(c.device),
            c=c,
            train=False,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=True,
            latent=latent,
        )
        pos_embeddings = outputs[-2]
        neg_embeddings = outputs[-1]
        if prev_interventions is None:
            prev_interventions = np.zeros((x.shape[0], c.shape[-1]))
            mask = np.zeros((x.shape[0], c.shape[-1]))
        else:
            mask = prev_interventions.copy()
#         if not self.include_prior:
#             prior_distribution = None
#         elif prior_distribution is not None:
#             prior_distribution = prior_distribution.detach().cpu().numpy()


        scores = torch.softmax(
             self.behavior_cloner(torch.FloatTensor(
                 self._compute_model_input(
                    prob=pred_c.detach().cpu().numpy(),
                    pos_embeddings=pos_embeddings.detach().cpu().numpy(),
                    neg_embeddings=neg_embeddings.detach().cpu().numpy(),
                    c=c.detach().cpu().numpy(),
                    competencies=competencies,
                    prev_interventions=prev_interventions,
                    use_concept_groups=False,
                    horizon=1,
                )
             )),
            dim=-1
        ).detach().cpu().numpy()

#         if prior_distribution is not None:
#             # Then rescale the scores based on the prior
#             if not isinstance(prior_distribution, np.ndarray):
#                 prior_distribution = prior_distribution.detach().cpu().numpy()
#             scores *= prior_distribution
        if prev_interventions is not None:
            # Then zero out the scores of the concepts that have been previously intervened
            scores = np.where(
                prev_interventions == 1,
                -float("inf"),
                scores,
            )

        best_concepts = np.argsort(-scores, axis=-1)
        for sample_idx in range(c.shape[0]):
            if self.group_based:
                # We will assign each group a score based on the max score of its
                # corresponding concepts
                group_scores = np.zeros(len(self.concept_group_map))
                group_names = []
                for i, key in enumerate(self.concept_group_map):
                    group_scores[i] = np.max(scores[sample_idx, self.concept_group_map[key]], axis=-1)
                    group_names.append(key)
                # Sort them out
                best_group_scores = np.argsort(-group_scores, axis=-1)
                for selected_group in best_group_scores[: self.num_groups_intervened]:
                    mask[sample_idx, self.concept_group_map[group_names[selected_group]]] = 1

            else:
                # Else, previous interventions do not affect future ones
                mask[sample_idx, best_concepts[sample_idx, : self.num_groups_intervened]] = 1
        return mask

    def __call__(
        self,
        x,
        pred_c,
        c,
        y=None,
        competencies=None,
        prev_interventions=None,
        prior_distribution=None,
    ):
        outputs = self.cbm._forward(
            x,
            intervention_idxs=torch.zeros(c.shape).to(c.device),
            c=c,
            y=y,
            train=False,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=True,
            output_latent=True,
        )
        latent = outputs[4]
        if prev_interventions is None:
            mask = np.zeros((x.shape[0], c.shape[-1]))
        else:
            mask = prev_interventions.detach().cpu().numpy()
        for _ in range(self.num_groups_intervened):
            mask = self._next_intervention(
                x=x,
                pred_c=pred_c,
                c=c,
                competencies=competencies,
                prev_interventions=mask,
                prior_distribution=prior_distribution,
                latent=latent,
            )
        return mask, c